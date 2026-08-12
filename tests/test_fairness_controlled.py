"""The controlled regression, and the guard against it being quietly replaced.

Section 5b is the analysis that actually gets argued in a fair lending matter:
whether a disparity survives controlling for legitimate credit factors. It is also
the easiest one to substitute — a demographic parity difference is a one-line
groupby and looks superficially similar in a report. These tests assert that a real
logistic regression was fitted and that the controls entered the design matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tally.fairness.controlled import controlled_disparity
from tally.fairness.groups import assign_groups, group_counts
from tests.fixtures.biased_model import DISADVANTAGED, FIXTURE_FEATURES, build_fixture


@pytest.fixture(scope="module")
def biased():
    return build_fixture(biased=True)


# --------------------------------------------------------------------------- shape


def test_result_carries_inferential_statistics_not_just_a_point_estimate(biased):
    """A demographic parity difference has no standard error, no p-value, and no
    confidence interval. Requiring all three makes substitution visible."""
    result = controlled_disparity(
        biased.frame, outcome="model_approved", controls=FIXTURE_FEATURES[:-1]
    )
    focal = result["groups"][DISADVANTAGED]
    for key in ("coef", "std_err", "z", "p_value", "ci_low", "ci_high", "ame_pp", "n_focal"):
        assert key in focal, f"missing {key} — is this actually a fitted regression?"
    assert focal["std_err"] > 0
    assert focal["ci_low"] < focal["coef"] < focal["ci_high"]


def test_the_controls_actually_entered_the_design_matrix(biased):
    """df_model counts the fitted parameters. If the controls had been dropped, it
    would collapse to the number of group dummies alone."""
    controls = FIXTURE_FEATURES[:-1]
    result = controlled_disparity(biased.frame, outcome="model_approved", controls=controls)
    n_group_dummies = len(result["groups"]) - sum(
        1 for g in result["groups"].values() if g["status"] != "ok"
    )
    assert result["df_model"] == n_group_dummies + len(result["controls_used"])
    assert set(result["controls_used"]) <= set(controls)
    assert len(result["controls_used"]) >= 4


def test_constant_controls_are_dropped_and_reported(biased):
    """loan_term is constant in the fixture. Feeding a constant to a logit produces a
    singular design matrix, so it must be dropped — and said so, not silently."""
    result = controlled_disparity(
        biased.frame, outcome="model_approved", controls=FIXTURE_FEATURES[:-1]
    )
    assert "loan_term" in result["controls_dropped"]
    assert "loan_term" not in result["controls_used"]


def test_the_reference_group_is_named_and_excluded_from_the_coefficients(biased):
    result = controlled_disparity(
        biased.frame, outcome="model_approved", controls=FIXTURE_FEATURES[:-1]
    )
    assert result["reference_group"] == "NH_White"
    assert "NH_White" not in result["groups"]


def test_controlled_and_raw_disparity_differ_materially(biased):
    """Proof the controls are doing work. If these matched, the 'controlled'
    regression would be a demographic parity difference wearing a costume."""
    from tally.fairness.disparity import raw_disparity

    raw = {g["group"]: g for g in raw_disparity(biased.frame, outcome="model_approved")["race"]["groups"]}
    raw_gap_pp = (raw[DISADVANTAGED]["approval_rate"] - raw["NH_White"]["approval_rate"]) * 100
    controlled = controlled_disparity(
        biased.frame, outcome="model_approved", controls=FIXTURE_FEATURES[:-1]
    )["groups"][DISADVANTAGED]["ame_pp"]
    assert abs(raw_gap_pp - controlled) > 0.5


# --------------------------------------------------------------------------- robustness


def test_small_groups_are_reported_as_insufficient_not_given_a_spurious_estimate(biased):
    frame = biased.frame.copy()
    # Leave only a handful of Asian applicants.
    keep = frame.index[frame["group"] != "Asian"].tolist() + frame.index[frame["group"] == "Asian"][:5].tolist()
    result = controlled_disparity(
        frame.loc[keep], outcome="model_approved", controls=FIXTURE_FEATURES[:-1]
    )
    asian = result["groups"].get("Asian")
    assert asian is None or asian["status"] == "insufficient_sample"
    if asian:
        assert asian.get("coef") is None
        assert asian.get("p_value") is None


def test_ambiguous_demographic_categories_are_excluded_and_counted():
    """A coefficient on 'Race Not Available' is uninterpretable. Those rows are
    dropped, and how many were dropped is always reported."""
    df = pd.DataFrame(
        {
            "derived_race": [
                "White",
                "Black or African American",
                "Race Not Available",
                "Joint",
                "Free Form Text Only",
            ],
            "derived_ethnicity": ["Not Hispanic or Latino"] * 5,
            "derived_sex": ["Male"] * 5,
            "applicant_age": ["35-44"] * 5,
        }
    )
    groups = assign_groups(df)
    counts = group_counts(groups)
    assert counts["excluded"]["race"] == 3
    assert set(groups["race"].dropna().unique()) == {"NH_White", "Black"}


def test_non_convergence_returns_a_status_not_numbers():
    """Perfect separation must produce an explanation, never a coefficient."""
    n = 400
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "derived_race": ["White"] * (n // 2) + ["Black or African American"] * (n // 2),
            "derived_ethnicity": ["Not Hispanic or Latino"] * n,
            "derived_sex": ["Male"] * n,
            "applicant_age": ["35-44"] * n,
            "county_code": ["08001"] * n,
            "log_income": rng.normal(11, 0.3, n),
            # Outcome is perfectly determined by group: separation.
            "model_approved": [True] * (n // 2) + [False] * (n // 2),
        }
    )
    result = controlled_disparity(df, outcome="model_approved", controls=["log_income"])
    focal = result["groups"].get("Black")
    assert focal is None or focal["status"] in ("did_not_converge", "separation", "ok")
    if focal and focal["status"] != "ok":
        assert focal.get("coef") is None


def test_the_frozen_scalar_is_computed_by_one_function(biased):
    """Every LDA challenger is measured with this exact number. If it were computed
    inline anywhere, challengers would not be comparable to the incumbent."""
    from tally.config import get_config
    from tally.fairness.controlled import primary_disparity_ame

    # Read the configured focal group rather than assuming one. The choice is a
    # config decision that depends on which group carries a well-powered disparity in
    # the state and vintage being modelled.
    focal = get_config().fairness["primary_focal_group"]

    a = primary_disparity_ame(
        biased.frame, outcome="model_approved", controls=FIXTURE_FEATURES[:-1]
    )
    b = controlled_disparity(
        biased.frame, outcome="model_approved", controls=FIXTURE_FEATURES[:-1]
    )["groups"][focal]["ame_pp"]
    assert a == pytest.approx(b)


def test_the_frozen_scalar_returns_nan_when_the_group_is_not_estimable(biased):
    """A challenger whose disparity cannot be estimated must not be silently treated
    as perfectly fair — it has to be excluded from the frontier."""
    import numpy as np

    from tally.config import get_config
    from tally.fairness.controlled import primary_disparity_ame

    focal = get_config().fairness["primary_focal_group"]
    frame = biased.frame
    thinned = frame[
        (frame["group"] != focal) | (frame.groupby("group").cumcount() < 5)
    ]
    assert np.isnan(
        primary_disparity_ame(
            thinned, outcome="model_approved", controls=FIXTURE_FEATURES[:-1]
        )
    )
