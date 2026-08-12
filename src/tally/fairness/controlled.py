"""5b — controlled disparate impact.

Whether a disparity survives controlling for legitimate credit factors is the
question that gets argued in a fair lending matter. A raw rate gap is easy to explain
away; a coefficient that stays significant after the lender's own stated credit
factors are in the model is not.

Design choices worth defending
------------------------------
**Outcome is the model's decision**, not the historical one. The question is what
this model does, not what the training data recorded. A second specification on the
historical outcome is fitted alongside so the pack can separate "disparity the model
introduces" from "disparity already present in the labels" — a contrast that is
usually more informative than either number alone.

**Controls are exactly the model's own feature set.** Not a hand-picked subset, not a
richer set the model never saw. Any disparity surviving these controls is disparity
the model produces *beyond what its own stated credit factors explain*, which is the
sharpest form the question can take.

**Two specifications, both reported.** Adding county fixed effects is contested: it
controls for genuine geographic variation, and it also absorbs redlining. Reporting
one silently would be picking a side. Both are computed and labelled.

**Standard errors are clustered by county.** Applications within a county share
unobserved local conditions; ignoring that overstates precision.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from tally.config import get_config
from tally.fairness.groups import (
    ATTRIBUTE_LABEL,
    ATTRIBUTES,
    GROUP_ORDER,
    REFERENCE,
    assign_groups,
    min_group_n,
    resolve_outcome,
)

#: Controls with fewer distinct values than this carry no information and would make
#: the design matrix singular.
_MIN_DISTINCT = 2


def _usable_controls(df: pd.DataFrame, controls: list[str]) -> tuple[list[str], list[str]]:
    used, dropped = [], []
    for c in controls:
        if c not in df.columns:
            dropped.append(c)
            continue
        col = pd.to_numeric(df[c], errors="coerce")
        if col.notna().sum() == 0 or col.nunique(dropna=True) < _MIN_DISTINCT:
            dropped.append(c)
        else:
            used.append(c)
    return used, dropped


#: Newton is fastest and most accurate when the design is well conditioned, but it
#: fails outright on ill-conditioned problems where quasi-Newton methods converge
#: fine. Trying them in order avoids reporting a false non-convergence, and the
#: method that succeeded is recorded so the result stays auditable.
_OPTIMIZERS = ("newton", "bfgs", "lbfgs")


def _fit_logit(
    frame: pd.DataFrame,
    formula: str,
    cluster: pd.Series | None,
) -> tuple[Any, str | None, str | None]:
    """Fit, returning (result, failure_reason, optimizer). Never raises on a
    modelling failure — a fair lending report must say it could not estimate
    something rather than emit a number it cannot stand behind."""
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    kwargs: dict[str, Any] = {"disp": 0, "maxiter": 200}
    if cluster is not None and cluster.nunique() > 1:
        kwargs["cov_type"] = "cluster"
        kwargs["cov_kwds"] = {"groups": cluster.to_numpy()}

    last_failure = "did_not_converge"
    for optimizer in _OPTIMIZERS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = smf.logit(formula, data=frame).fit(method=optimizer, **kwargs)
        except PerfectSeparationError:
            return None, "separation", None
        except (np.linalg.LinAlgError, ValueError) as exc:
            last_failure = f"did_not_converge ({type(exc).__name__})"
            continue

        if result.mle_retvals.get("converged", False):
            return result, None, optimizer
        last_failure = "did_not_converge"

    return None, last_failure, None


def controlled_disparity(
    df: pd.DataFrame,
    *,
    outcome: str = "model_approved",
    controls: list[str] | None = None,
    attribute: str = "race",
    county_fixed_effects: bool = False,
    groups: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Logistic regression of the decision on protected class plus credit factors."""
    cfg = get_config()
    floor = min_group_n()
    controls = controls or cfg.feature_names

    groups = groups if groups is not None else assign_groups(df)
    frame = df.copy()
    frame["_y"] = resolve_outcome(df, outcome)
    frame["_group"] = groups[attribute]
    frame["_county"] = df.get("county_code", pd.Series(["_", ] * len(df), index=df.index)).astype(str)

    reference = REFERENCE[attribute]
    frame = frame[frame["_group"].notna()]

    # Groups too small for a stable estimate are reported as such, not estimated.
    sizes = frame["_group"].value_counts()
    small = [g for g, n in sizes.items() if n < floor and g != reference]
    estimable = frame[~frame["_group"].isin(small)].copy()

    used, dropped = _usable_controls(estimable, list(controls))
    for c in used:
        estimable[c] = pd.to_numeric(estimable[c], errors="coerce")
    estimable = estimable.dropna(subset=["_y", "_group", *used])

    result_out: dict[str, Any] = {
        "attribute": attribute,
        "label": ATTRIBUTE_LABEL[attribute],
        "outcome": outcome,
        "reference_group": reference,
        "controls_used": used,
        "controls_dropped": dropped,
        "county_fixed_effects": county_fixed_effects,
        "n_obs": int(len(estimable)),
        "groups": {},
        "specification": (
            "Logistic regression of the decision on protected class indicators plus "
            "the model's own credit factors. Standard errors clustered by county."
        ),
    }

    for g in small:
        result_out["groups"][g] = {
            "status": "insufficient_sample",
            "n_focal": int(sizes[g]),
            "note": f"fewer than {floor} applicants; no coefficient is estimated",
            "coef": None,
            "p_value": None,
            "ame_pp": None,
        }

    present = [g for g in GROUP_ORDER[attribute] if g in set(estimable["_group"])]
    if reference not in present or len(present) < 2 or len(estimable) < floor:
        result_out["status"] = "not_estimable"
        result_out["reason"] = (
            "The reference group or all focal groups are absent from the frame, so no "
            "comparison can be fitted."
        )
        result_out["df_model"] = 0
        return result_out

    terms = [f"C(_group, Treatment(reference='{reference}'))"]
    terms += [f"Q('{c}')" for c in used]
    if county_fixed_effects and estimable["_county"].nunique() > 1:
        terms.append("C(_county)")
    formula = "_y ~ " + " + ".join(terms)

    fitted, failure, optimizer = _fit_logit(estimable, formula, estimable["_county"])
    if fitted is None:
        result_out["status"] = failure
        result_out["reason"] = (
            "The regression did not converge, so no coefficients are reported. This "
            "usually indicates perfect or near-perfect separation between a group and "
            "the outcome."
        )
        result_out["df_model"] = 0
        for g in present:
            if g == reference:
                continue
            result_out["groups"][g] = {
                "status": failure,
                "n_focal": int((estimable["_group"] == g).sum()),
                "coef": None,
                "p_value": None,
                "ame_pp": None,
            }
        return result_out

    result_out["status"] = "ok"
    result_out["optimizer"] = optimizer
    result_out["df_model"] = int(fitted.df_model)
    result_out["pseudo_r2"] = float(fitted.prsquared)
    result_out["llr_p_value"] = float(fitted.llr_pvalue)

    margins = _marginal_effects(fitted, estimable, reference, attribute)
    conf = fitted.conf_int()
    n_ref = int((estimable["_group"] == reference).sum())

    for g in present:
        if g == reference:
            continue
        term = f"C(_group, Treatment(reference='{reference}'))[T.{g}]"
        if term not in fitted.params.index:
            continue
        result_out["groups"][g] = {
            "status": "ok",
            "coef": float(fitted.params[term]),
            "std_err": float(fitted.bse[term]),
            "z": float(fitted.tvalues[term]),
            "p_value": float(fitted.pvalues[term]),
            "ci_low": float(conf.loc[term, 0]),
            "ci_high": float(conf.loc[term, 1]),
            "odds_ratio": float(np.exp(fitted.params[term])),
            "ame_pp": margins.get(g),
            "n_focal": int((estimable["_group"] == g).sum()),
            "n_reference": n_ref,
            "significant": bool(fitted.pvalues[term] < 0.05),
        }

    return result_out


def _marginal_effects(
    fitted: Any, frame: pd.DataFrame, reference: str, attribute: str
) -> dict[str, float]:
    """Average marginal effect in percentage points.

    The coefficient is in log-odds, which no risk committee reasons in. The AME
    answers "how many approvals per hundred does membership of this group cost, on
    average, holding the credit factors fixed" — which is the number people argue
    about. Computed by counterfactual prediction rather than `get_margeff`, because
    that helper does not handle categorical contrasts cleanly.
    """
    out: dict[str, float] = {}
    baseline = frame.copy()
    baseline["_group"] = reference
    p_ref = fitted.predict(baseline)

    for g in frame["_group"].dropna().unique():
        if g == reference:
            continue
        alt = frame.copy()
        alt["_group"] = g
        out[str(g)] = float((fitted.predict(alt) - p_ref).mean() * 100.0)
    return out


def primary_disparity_ame(
    df: pd.DataFrame,
    *,
    outcome: str = "model_approved",
    controls: list[str] | None = None,
    groups: pd.DataFrame | None = None,
) -> float:
    """The single scalar every LDA challenger is measured by.

    Frozen deliberately: the average marginal effect, in percentage points, for the
    configured primary focal group, in the no-geography-fixed-effects specification
    on the model's own decision. Every challenger in the search is evaluated by this
    exact function on this exact frame, or the frontier compares nothing.

    Returns NaN when the group is not estimable, so the caller can exclude that
    challenger rather than silently treat it as fair.
    """
    cfg = get_config()
    focal = cfg.fairness["primary_focal_group"]
    result = controlled_disparity(
        df,
        outcome=outcome,
        controls=controls,
        attribute="race",
        county_fixed_effects=False,
        groups=groups,
    )
    entry = result["groups"].get(focal)
    if not entry or entry.get("status") != "ok" or entry.get("ame_pp") is None:
        return float("nan")
    return float(entry["ame_pp"])


def controlled_suite(
    df: pd.DataFrame,
    *,
    model_outcome: str = "model_approved",
    historical_outcome: str = "approved",
    controls: list[str] | None = None,
    groups: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """All specifications the validation pack reports for section 7."""
    suite: dict[str, Any] = {"specifications": {}}

    for attr in ATTRIBUTES:
        suite["specifications"][attr] = {
            "primary": controlled_disparity(
                df, outcome=model_outcome, controls=controls, attribute=attr, groups=groups
            ),
            "with_county_fixed_effects": controlled_disparity(
                df,
                outcome=model_outcome,
                controls=controls,
                attribute=attr,
                county_fixed_effects=True,
                groups=groups,
            ),
        }

    if historical_outcome in df.columns:
        suite["historical_labels"] = controlled_disparity(
            df, outcome=historical_outcome, controls=controls, attribute="race", groups=groups
        )
        suite["historical_note"] = (
            "Fitted on the historical lender decision rather than the model's. "
            "Comparing the two separates disparity the model introduces from "
            "disparity already present in the behaviour it was trained to replicate."
        )

    suite["geography_note"] = (
        "Two specifications are reported because the choice is contested. County "
        "fixed effects control for genuine local economic variation, and they also "
        "absorb geographic discrimination, which is itself a fair lending concern. "
        "Neither is presented as the single correct answer."
    )
    return suite
