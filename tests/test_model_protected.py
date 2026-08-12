"""The protected-attribute boundary.

This is a showcase artifact as much as a test: it is the mechanical proof that race,
ethnicity, sex, and age cannot reach a feature vector. Do not weaken it.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from tally.config import get_config
from tally.data import schema
from tally.data.features import (
    ApplicantContext,
    ProtectedLeakage,
    RawApplicant,
)


def test_feature_list_and_protected_are_disjoint():
    cfg = get_config()
    assert set(cfg.feature_names) & set(cfg.protected) == set()
    assert set(cfg.feature_names) & set(schema.PROTECTED_COLUMNS) == set()


def test_config_and_schema_agree_on_what_is_protected():
    cfg = get_config()
    assert set(cfg.protected) == set(schema.PROTECTED_COLUMNS)


def test_raw_applicant_has_no_protected_field():
    """Structural, not behavioural: there is no field to put a protected value in."""
    fields = set(RawApplicant.model_fields)
    assert fields & set(schema.PROTECTED_COLUMNS) == set()
    for banned in ("race", "ethnicity", "sex", "gender", "age"):
        assert not any(banned in f for f in fields), f"{banned!r} appears in RawApplicant"


def test_protected_attributes_live_only_on_the_context_model():
    ctx = set(ApplicantContext.model_fields)
    assert set(schema.PROTECTED_COLUMNS) <= ctx


def test_raw_applicant_rejects_unknown_fields():
    """Submitting a protected attribute must fail loudly. Silently dropping it would
    leave the caller believing it had been considered."""
    with pytest.raises(ValidationError) as exc:
        RawApplicant(
            income_usd=100_000,
            loan_amount_usd=300_000,
            cltv=80,
            dti_band="36",
            derived_race="White",  # not a field, and must not be silently accepted
        )
    assert any(e["type"] == "extra_forbidden" for e in exc.value.errors())


def test_feature_matrix_raises_if_a_protected_column_would_leak():
    """Simulates someone widening feature_matrix to pass a protected column through."""
    from tally.data import features as feat

    df = pd.DataFrame(
        {
            "income_usd": [100_000.0],
            "loan_amount_usd": [300_000.0],
            "cltv": [80.0],
            "dti_band_ord": [5.0],
            "loan_term": [360.0],
            "interest_only": [0.0],
            "balloon": [0.0],
            "neg_amortization": [0.0],
            "tract_to_msa_income_pct": [100.0],
            "units_ord": [0.0],
            "preapproval": [0.0],
        }
    )
    with pytest.raises(ProtectedLeakage):
        feat._assert_no_protected([*df.columns, "derived_race"])


def test_trained_model_feature_list_excludes_protected(model):
    cfg = get_config()
    assert set(model.feature_names) & set(cfg.protected) == set()
    assert set(model.booster.feature_name()) & set(cfg.protected) == set()


def test_excluded_proxy_is_not_in_the_baseline_model(model):
    """tract_minority_population_percent is a known geographic proxy for race. It is
    excluded from the incumbent deliberately, and priced by the LDA search instead."""
    assert "tract_minority_pct" not in model.feature_names
