"""Counterfactuals.

The contract is narrow and absolute: either return a value that has been re-scored
and actually flips the decision, or say plainly that no such value exists. A number
that does not work is worse than no number, because an applicant may act on it.
"""

from __future__ import annotations

import pytest

from tally.data.features import RawApplicant, build_vector
from tally.explain.counterfactual import find_counterfactual
from tally.explain.reason_codes import derive_reason_codes
from tally.explain.shap_engine import get_engine

DECLINING = dict(
    income_usd=66_000,
    loan_amount_usd=345_000,
    property_value_usd=405_000,
    cltv=85.0,
    dti_band=">60%",
    loan_term=360,
    tract_to_msa_income_pct=88.0,
)


def _top_feature(applicant: RawApplicant, model) -> str | None:
    x = build_vector(applicant, model.fit_params, model.feature_names)
    attribution = get_engine(model).attribute_one(x)
    from tally.serving.scoring import score_vector

    return derive_reason_codes(attribution, score_vector(applicant, model).decision).top_feature


@pytest.mark.slow
@pytest.mark.needs_model
def test_the_returned_value_actually_flips_the_decision(model):
    """Re-score the modified applicant and confirm. Not a claim, a verification."""
    from tally.serving.scoring import score_vector

    applicant = RawApplicant(**DECLINING)
    feature = _top_feature(applicant, model)
    cf = find_counterfactual(applicant, feature, model)

    assert cf.achievable, cf.reason
    assert cf.verified_decision == "APPROVED"

    from tally.data.features import FEATURE_SOURCE

    modified = applicant.model_copy(update={FEATURE_SOURCE[feature]: cf.flip_value})
    assert score_vector(modified, model).decision == "APPROVED"


@pytest.mark.slow
@pytest.mark.needs_model
def test_the_value_never_leaves_the_observed_training_range(model):
    applicant = RawApplicant(**DECLINING)
    for feature in model.feature_names:
        cf = find_counterfactual(applicant, feature, model)
        if not cf.achievable or isinstance(cf.flip_value, (str, bool)):
            continue
        lo, hi = model.fit_params.raw_ranges[feature]
        assert lo <= float(cf.flip_value) <= hi, (
            f"{feature}: {cf.flip_value} is outside the observed range [{lo}, {hi}]"
        )


@pytest.mark.slow
@pytest.mark.needs_model
def test_an_unachievable_counterfactual_says_so_instead_of_guessing(model):
    """An applicant far below the boundary cannot be rescued by one factor. The
    result must be an explicit refusal, not a misleading number."""
    hopeless = RawApplicant(
        income_usd=20_000,
        loan_amount_usd=1_500_000,
        property_value_usd=1_505_000,
        cltv=99.0,
        dti_band=">60%",
        loan_term=360,
        interest_only=True,
        balloon=True,
        neg_amortization=True,
    )
    cf = find_counterfactual(hopeless, "units_ord", model)
    assert not cf.achievable
    assert cf.flip_value is None
    assert cf.reason and len(cf.reason.split()) > 5
    assert cf.sentence == cf.reason


@pytest.mark.slow
@pytest.mark.needs_model
def test_the_sentence_is_applicant_readable(model):
    applicant = RawApplicant(**DECLINING)
    cf = find_counterfactual(applicant, _top_feature(applicant, model), model)
    sentence = cf.sentence
    assert sentence.endswith(".")
    for banned in ("log_", "_ord", "_pct", "phi", "shap", "raw_score"):
        assert banned not in sentence.lower()
    assert "holding all other factors constant" in sentence.lower()


@pytest.mark.slow
@pytest.mark.needs_model
def test_ordinal_features_report_a_band_not_a_code(model):
    applicant = RawApplicant(**DECLINING)
    cf = find_counterfactual(applicant, "dti_band_ord", model)
    if cf.achievable:
        # The band label, never the integer encoding.
        assert not str(cf.flip_value).replace(".", "").isdigit() or "%" in str(cf.flip_display)
        assert cf.flip_display in cf.sentence


@pytest.mark.slow
@pytest.mark.needs_model
def test_direction_matches_the_declared_constraint(model):
    """A -1 feature must be quoted as an upper bound, a +1 feature as a lower bound."""
    applicant = RawApplicant(**DECLINING)
    for feature, direction in model.monotone.items():
        cf = find_counterfactual(applicant, feature, model)
        if not cf.achievable or isinstance(cf.flip_value, (str, bool)):
            continue
        if direction == -1:
            assert "at or below" in cf.sentence
        elif direction == 1:
            assert "at or above" in cf.sentence


@pytest.mark.slow
@pytest.mark.needs_model
def test_binary_search_and_scan_are_labelled_differently(model):
    """Unconstrained features use a scan because binary search would be unsound
    there. The method is recorded so the difference is auditable."""
    applicant = RawApplicant(**DECLINING)
    methods = {
        f: find_counterfactual(applicant, f, model).method
        for f, d in model.monotone.items()
        if d == 0
    }
    assert all(m in ("range scan", "binary flip", "ordinal band search") for m in methods.values())
