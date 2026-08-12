"""Reason codes and the threshold-anchored baseline they rest on."""

from __future__ import annotations

import numpy as np
import pytest
import shap

from tally.config import get_config
from tally.explain.reason_codes import derive_reason_codes
from tally.explain.shap_engine import Attribution, raw_threshold
from tally.explain.templates import (
    FEATURE_TO_CODE,
    TEMPLATES,
    UnmappedReasonCode,
    code_for_feature,
    template_for_code,
)

# --------------------------------------------------------------------------- vocabulary


def test_every_mapped_code_has_plain_language_text():
    """A code with no text cannot be disclosed, so it must not be emittable."""
    missing = set(FEATURE_TO_CODE.values()) - set(TEMPLATES)
    assert not missing, f"codes with no plain-language text: {sorted(missing)}"


def test_every_model_feature_has_a_reason_code():
    cfg = get_config()
    missing = [f for f in cfg.feature_names if f not in FEATURE_TO_CODE]
    assert not missing, f"features that could be a top contributor with no code: {missing}"


def test_an_unmapped_feature_raises_rather_than_leaking_its_name():
    with pytest.raises(UnmappedReasonCode):
        code_for_feature("some_internal_feature_name")


def test_an_unmapped_code_raises():
    with pytest.raises(UnmappedReasonCode):
        template_for_code("NO_SUCH_CODE")


def test_applicant_text_never_exposes_model_internals():
    banned = ("shap", "score", "threshold", "model", "percentile", "coefficient", "_")
    for tpl in TEMPLATES.values():
        low = tpl.applicant_text.lower()
        for word in banned:
            assert word not in low, f"{tpl.code} applicant text mentions {word!r}"


def test_applicant_text_is_a_sentence_not_a_label():
    for tpl in TEMPLATES.values():
        assert len(tpl.applicant_text.split()) >= 5, tpl.code
        assert tpl.applicant_text[0].isupper(), tpl.code


# --------------------------------------------------------------------------- selection


def _attribution(values: dict[str, float]) -> Attribution:
    names = list(values)
    return Attribution(
        feature_names=names,
        values=np.array([values[n] for n in names], dtype=float),
        base_value=0.0,
        raw_score=float(sum(values.values())),
        raw_threshold=0.0,
    )


def test_approved_applications_produce_no_adverse_action_reasons():
    attribution = _attribution({"cltv": -1.0, "dti_band_ord": -2.0})
    assert derive_reason_codes(attribution, "APPROVED").codes == []


def test_reasons_are_ordered_most_damaging_first():
    attribution = _attribution(
        {"cltv": -0.5, "dti_band_ord": -2.0, "log_income": -1.0, "loan_term": 3.0}
    )
    codes = derive_reason_codes(attribution, "DECLINED").codes
    assert [c.feature for c in codes] == ["dti_band_ord", "log_income", "cltv"]
    assert [c.rank for c in codes] == [1, 2, 3]


def test_positive_contributors_are_never_reported_as_reasons_for_decline():
    attribution = _attribution({"cltv": -1.0, "log_income": 5.0})
    codes = derive_reason_codes(attribution, "DECLINED").codes
    assert [c.feature for c in codes] == ["cltv"]


def test_never_more_than_four_reasons():
    attribution = _attribution({name: -(i + 1.0) for i, name in enumerate(FEATURE_TO_CODE)})
    codes = derive_reason_codes(attribution, "DECLINED").codes
    assert len(codes) == int(get_config().explain["max_reason_codes"]) == 4


# --------------------------------------------------------------------------- baseline


@pytest.mark.slow
@pytest.mark.needs_model
def test_background_is_anchored_at_the_decision_threshold(model):
    from tally.explain.shap_engine import ShapEngine

    engine = ShapEngine(model)
    tol = float(get_config().explain["baseline_residual_tolerance"])
    assert abs(engine.residual) <= tol
    assert engine.f_star == pytest.approx(raw_threshold(model.threshold))


@pytest.mark.slow
@pytest.mark.needs_model
def test_the_background_is_actually_used(model):
    """The trap this guards: feature_perturbation='tree_path_dependent' is SHAP's
    default for a bare Booster and silently ignores `data=`. If that happened, the
    explainer would fall back to a mean baseline while the pack claims otherwise."""
    from tally.explain.shap_engine import ShapEngine

    anchored = ShapEngine(model).base_value
    naive = float(np.ravel(shap.TreeExplainer(model.booster).expected_value)[0])
    assert abs(anchored - naive) > 0.5, (
        "the anchored baseline is indistinguishable from the mean baseline, which "
        "means the background dataset was ignored"
    )


@pytest.mark.slow
@pytest.mark.needs_model
def test_contributions_sum_to_the_decision_margin(model, eval_frame):
    """The property that makes these contributions answer the Regulation B question
    rather than 'why is this applicant unusual'."""
    from tally.data.features import feature_matrix
    from tally.explain.shap_engine import ShapEngine

    engine = ShapEngine(model)
    X = feature_matrix(eval_frame.head(200), model.fit_params)[model.feature_names]
    for attribution in engine.attribute(X):
        assert attribution.values.sum() == pytest.approx(
            attribution.margin - attribution.residual, abs=1e-5
        )


@pytest.mark.slow
@pytest.mark.needs_model
def test_reason_codes_are_deterministic_for_a_fixed_input(model, tmp_path):
    from tally.data.features import RawApplicant
    from tally.ledger.store import LedgerStore
    from tally.serving.scoring import score_applicant

    applicant = RawApplicant(
        income_usd=66_000, loan_amount_usd=345_000, cltv=85.0, dti_band=">60%", loan_term=360
    )
    ledger = LedgerStore(tmp_path / "d.db")
    first = score_applicant(applicant, model=model, ledger=ledger)
    second = score_applicant(applicant, model=model, ledger=ledger)

    assert first.decision == second.decision
    assert [c.code for c in first.reasons.codes] == [c.code for c in second.reasons.codes]
    assert first.probability == second.probability
    # Different ledger entries, identical explanation.
    assert first.decision_id != second.decision_id
