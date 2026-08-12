"""What would have had to be different.

The search runs on the raw applicant record, not the model vector, so the number
that comes out is something an applicant can act on — a dollar figure or a
percentage, not `log_income = 11.2`.

Two properties are non-negotiable:

* The value returned is **re-scored and verified** before it is emitted. A
  counterfactual that does not actually flip the decision is worse than none.
* The value never leaves the range observed in training. Telling an applicant that
  an income of four million dollars would have worked is technically true and
  practically an insult.

Binary search is only valid because the model is monotone in the searched feature —
which is exactly what `tests/test_model_monotonicity.py` establishes. For an
unconstrained feature the function may be non-monotone, so a scan is used instead
and the difference is disclosed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from tally.data import schema
from tally.data.features import FEATURE_SOURCE, RawApplicant, build_vector
from tally.explain.templates import template_for_feature
from tally.model.registry import LoadedModel

MAX_ITERATIONS = 40
SCAN_POINTS = 200


@dataclass(frozen=True)
class Counterfactual:
    feature: str
    display_name: str
    current_value: float | str
    current_display: str
    flip_value: float | str | None
    flip_display: str | None
    direction: int
    achievable: bool
    verified_probability: float | None
    verified_decision: str | None
    reason: str | None
    sentence: str
    method: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "display_name": self.display_name,
            "current_value": self.current_value,
            "current_display": self.current_display,
            "flip_value": self.flip_value,
            "flip_display": self.flip_display,
            "direction": self.direction,
            "achievable": self.achievable,
            "verified_probability": self.verified_probability,
            "verified_decision": self.verified_decision,
            "reason": self.reason,
            "sentence": self.sentence,
            "method": self.method,
        }


# --------------------------------------------------------------------------- display


def _format_usd(v: float) -> str:
    return f"${v:,.0f}"


def _format_pct(v: float) -> str:
    return f"{v:.0f}%" if abs(v - round(v)) < 0.05 else f"{v:.1f}%"


def _format_months(v: float) -> str:
    years = v / 12
    return f"{v:.0f} months ({years:.0f} years)" if abs(years - round(years)) < 0.05 else f"{v:.0f} months"


_FORMATTERS = {
    "income_usd": _format_usd,
    "loan_amount_usd": _format_usd,
    "cltv": _format_pct,
    "tract_to_msa_income_pct": _format_pct,
    "loan_term": _format_months,
}

#: Smallest step worth quoting, in raw units. Below this the number is noise.
_PRESENTATION_UNIT = {
    "income_usd": 1_000.0,
    "loan_amount_usd": 5_000.0,
    "cltv": 0.5,
    "tract_to_msa_income_pct": 1.0,
    "loan_term": 12.0,
}


def _display(source: str, value: float | str) -> str:
    if source == "dti_band":
        band = str(value)
        return schema.DTI_BAND_LABEL.get(band, band)
    if source == "units_band":
        return str(value)
    if isinstance(value, str):
        return value
    fmt = _FORMATTERS.get(source)
    return fmt(float(value)) if fmt else f"{float(value):,.2f}"


# --------------------------------------------------------------------------- search


def _score(applicant: RawApplicant, model: LoadedModel) -> float:
    x = build_vector(applicant, model.fit_params, model.feature_names)
    return float(model.probability(x)[0])


def _with(applicant: RawApplicant, source: str, value: Any) -> RawApplicant:
    return applicant.model_copy(update={source: value})


def _approves(applicant: RawApplicant, model: LoadedModel, source: str, value: Any) -> bool:
    return _score(_with(applicant, source, value), model) > model.threshold


def find_counterfactual(
    applicant: RawApplicant, feature: str, model: LoadedModel
) -> Counterfactual:
    """Find the value of `feature` at which this application would be approved."""
    source = FEATURE_SOURCE[feature]
    tpl = template_for_feature(feature)
    direction = model.monotone.get(feature, 0)
    current = getattr(applicant, source)

    if source in ("dti_band", "units_band"):
        return _ordinal_search(applicant, feature, source, tpl.factor, direction, model)
    if source in ("interest_only", "balloon", "neg_amortization", "preapproval"):
        return _boolean_search(applicant, feature, source, tpl.factor, direction, model)
    return _numeric_search(applicant, feature, source, tpl.factor, direction, model, current)


def _numeric_search(
    applicant: RawApplicant,
    feature: str,
    source: str,
    factor: str,
    direction: int,
    model: LoadedModel,
    current: float | None,
) -> Counterfactual:
    lo, hi = model.fit_params.raw_ranges.get(feature, (0.0, 1.0))
    cur = float(current) if current is not None else float((lo + hi) / 2)
    unit = _PRESENTATION_UNIT.get(source, 1.0)

    # The most favourable value the training data actually supports.
    if direction == 0:
        return _scan(applicant, feature, source, factor, model, lo, hi, cur, unit)
    endpoint = hi if direction == 1 else lo

    if not _approves(applicant, model, source, endpoint):
        return _unachievable(
            feature,
            source,
            factor,
            direction,
            cur,
            reason=(
                f"Even at {_display(source, endpoint)}, the most favourable value "
                "observed in the training data, this application would still have "
                "been declined. No single change to this factor is sufficient."
            ),
            method="endpoint probe",
        )

    a, b = (cur, endpoint) if direction == 1 else (endpoint, cur)
    # Invariant: the endpoint side approves, the current side does not.
    for _ in range(MAX_ITERATIONS):
        mid = (a + b) / 2
        if _approves(applicant, model, source, mid):
            if direction == 1:
                b = mid
            else:
                a = mid
        else:
            if direction == 1:
                a = mid
            else:
                b = mid
        if abs(b - a) < unit / 100:
            break

    flip = b if direction == 1 else a
    flip = _round_to_safe_side(applicant, model, source, flip, unit, direction, lo, hi)

    if flip is None:
        return _unachievable(
            feature,
            source,
            factor,
            direction,
            cur,
            reason=(
                "The change required is smaller than can be meaningfully quoted for "
                "this factor."
            ),
            method="binary search",
        )

    prob = _score(_with(applicant, source, flip), model)
    return _achievable(
        feature, source, factor, direction, cur, flip, prob, model, method="binary search"
    )


def _round_to_safe_side(
    applicant: RawApplicant,
    model: LoadedModel,
    source: str,
    flip: float,
    unit: float,
    direction: int,
    lo: float,
    hi: float,
) -> float | None:
    """Round to a quotable figure, then re-verify and step if rounding undid it.

    Rounding a boundary value toward the applicant's current position can push it
    back across the line. Quoting a number that does not work is the one failure
    this function exists to prevent.
    """
    step = unit if direction == 1 else -unit
    candidate = (math.ceil(flip / unit) if direction == 1 else math.floor(flip / unit)) * unit

    for _ in range(8):
        if lo <= candidate <= hi and _approves(applicant, model, source, candidate):
            return float(candidate)
        candidate += step
        if candidate < lo or candidate > hi:
            return None
    return None


def _scan(
    applicant: RawApplicant,
    feature: str,
    source: str,
    factor: str,
    model: LoadedModel,
    lo: float,
    hi: float,
    cur: float,
    unit: float,
) -> Counterfactual:
    """Unconstrained feature: monotonicity is not guaranteed, so binary search would
    be unsound. Scan the range and take the flip point nearest the current value."""
    grid = np.linspace(lo, hi, SCAN_POINTS)
    approving = [float(v) for v in grid if _approves(applicant, model, source, float(v))]
    if not approving:
        return _unachievable(
            feature,
            source,
            factor,
            0,
            cur,
            reason=(
                "No value of this factor within its observed range would change the "
                "outcome on its own."
            ),
            method="range scan",
        )
    flip = min(approving, key=lambda v: abs(v - cur))
    prob = _score(_with(applicant, source, flip), model)
    return _achievable(feature, source, factor, 0, cur, flip, prob, model, method="range scan")


def _ordinal_search(
    applicant: RawApplicant,
    feature: str,
    source: str,
    factor: str,
    direction: int,
    model: LoadedModel,
) -> Counterfactual:
    """Search the ordered band codes and report the band label, not the code."""
    order = schema.DTI_BAND_ORDER if source == "dti_band" else schema.UNITS_BAND_ORDER
    current = str(getattr(applicant, source))
    cur_idx = order.index(current) if current in order else len(order) // 2

    # Constraint is -1 for both, so lower codes are more favourable.
    candidates = list(range(cur_idx - 1, -1, -1)) if direction <= 0 else list(
        range(cur_idx + 1, len(order))
    )
    for idx in candidates:
        band = order[idx]
        if _approves(applicant, model, source, band):
            prob = _score(_with(applicant, source, band), model)
            return _achievable(
                feature, source, factor, direction, current, band, prob, model,
                method="ordinal band search",
            )

    return _unachievable(
        feature,
        source,
        factor,
        direction,
        current,
        reason=(
            "No band of this factor would change the outcome on its own, including "
            "the most favourable band observed."
        ),
        method="ordinal band search",
    )


def _boolean_search(
    applicant: RawApplicant,
    feature: str,
    source: str,
    factor: str,
    direction: int,
    model: LoadedModel,
) -> Counterfactual:
    current = bool(getattr(applicant, source))
    alternative = not current
    if _approves(applicant, model, source, alternative):
        prob = _score(_with(applicant, source, alternative), model)
        return _achievable(
            feature, source, factor, direction, current, alternative, prob, model,
            method="binary flip",
        )
    return _unachievable(
        feature,
        source,
        factor,
        direction,
        current,
        reason="Changing this feature alone would not have changed the outcome.",
        method="binary flip",
    )


# --------------------------------------------------------------------------- results


def _achievable(
    feature: str,
    source: str,
    factor: str,
    direction: int,
    current: Any,
    flip: Any,
    probability: float,
    model: LoadedModel,
    *,
    method: str,
) -> Counterfactual:
    decision = model.decide(probability)
    if decision != "APPROVED":  # pragma: no cover - guarded by construction
        raise AssertionError("counterfactual failed verification after re-scoring")

    flip_display = _display(source, flip)
    if isinstance(flip, bool):
        sentence = (
            f"Removing the {factor} would have produced an approval, holding all "
            "other factors constant."
            if not flip
            else f"Adding a {factor} would have produced an approval, holding all "
            "other factors constant."
        )
    elif direction == 1:
        sentence = (
            f"An {factor} at or above {flip_display} would have produced an approval, "
            "holding all other factors constant."
        )
    elif direction == -1:
        sentence = (
            f"A {factor} at or below {flip_display} would have produced an approval, "
            "holding all other factors constant."
        )
    else:
        sentence = (
            f"A {factor} of {flip_display} would have produced an approval, holding "
            "all other factors constant."
        )

    return Counterfactual(
        feature=feature,
        display_name=factor,
        current_value=current if isinstance(current, str) else float(current),
        current_display=_display(source, current),
        flip_value=flip if isinstance(flip, (str, bool)) else float(flip),
        flip_display=flip_display,
        direction=direction,
        achievable=True,
        verified_probability=float(probability),
        verified_decision=decision,
        reason=None,
        sentence=sentence,
        method=method,
    )


def _unachievable(
    feature: str,
    source: str,
    factor: str,
    direction: int,
    current: Any,
    *,
    reason: str,
    method: str,
) -> Counterfactual:
    return Counterfactual(
        feature=feature,
        display_name=factor,
        current_value=current if isinstance(current, str) else float(current),
        current_display=_display(source, current),
        flip_value=None,
        flip_display=None,
        direction=direction,
        achievable=False,
        verified_probability=None,
        verified_decision=None,
        reason=reason,
        sentence=reason,
        method=method,
    )
