"""The proof that the fairness engine catches something.

A model with a deliberate, obvious disparity must be flagged by all three analyses.
A comparable model without it must not be. Both halves matter: a detector that always
alarms tells you nothing.
"""

from __future__ import annotations

import pytest

from tally.fairness.controlled import controlled_disparity
from tally.fairness.disparity import raw_disparity
from tally.fairness.proxy import proxy_scan
from tests.fixtures.biased_model import DISADVANTAGED, FIXTURE_FEATURES, build_fixture


@pytest.fixture(scope="module")
def biased():
    return build_fixture(biased=True)


@pytest.fixture(scope="module")
def clean():
    return build_fixture(biased=False)


# --------------------------------------------------------------------------- 5a


def test_raw_disparity_flags_the_injected_bias(biased):
    report = raw_disparity(biased.frame, outcome="model_approved")
    race = {g["group"]: g for g in report["race"]["groups"]}
    focal = race[DISADVANTAGED]
    assert focal["status"] == "ok"
    assert focal["ratio"] < 0.80
    assert focal["four_fifths_flag"] is True
    assert report["race"]["any_flag"] is True


def test_raw_disparity_does_not_fire_on_the_clean_model(clean):
    report = raw_disparity(clean.frame, outcome="model_approved")
    race = {g["group"]: g for g in report["race"]["groups"]}
    assert race[DISADVANTAGED]["four_fifths_flag"] is False


# --------------------------------------------------------------------------- 5b


def test_controlled_regression_flags_the_injected_bias(biased):
    result = controlled_disparity(
        biased.frame, outcome="model_approved", controls=FIXTURE_FEATURES[:-1]
    )
    focal = result["groups"][DISADVANTAGED]
    assert focal["status"] == "ok"
    assert focal["p_value"] < 0.01
    assert focal["coef"] < 0  # membership reduces the odds of approval
    assert focal["ame_pp"] < -2.0  # and by a materially large margin
    assert focal["ci_high"] < 0  # the interval excludes zero


def test_controlled_regression_does_not_fire_on_the_clean_model(clean):
    result = controlled_disparity(
        clean.frame, outcome="model_approved", controls=FIXTURE_FEATURES[:-1]
    )
    focal = result["groups"][DISADVANTAGED]
    assert focal["status"] == "ok"
    assert abs(focal["ame_pp"]) < 2.0
    assert focal["ci_low"] <= 0 <= focal["ci_high"] or focal["p_value"] > 0.01


def test_including_the_proxy_as_a_control_does_not_fabricate_a_clean_result(biased):
    """Adding the proxy to the controls is how a disparity gets explained away: the
    proxy stands in for group membership, so the group coefficient has nothing left
    to attribute. Here it goes further and induces quasi-separation.

    Either outcome is acceptable; what is not acceptable is a coefficient with no
    standard error behind it. Assert that the engine reports a status rather than a
    reassuring number it cannot support."""
    without = controlled_disparity(
        biased.frame, outcome="model_approved", controls=FIXTURE_FEATURES[:-1]
    )["groups"][DISADVANTAGED]
    with_proxy = controlled_disparity(
        biased.frame, outcome="model_approved", controls=FIXTURE_FEATURES
    )["groups"][DISADVANTAGED]

    assert without["status"] == "ok"
    if with_proxy["status"] == "ok":
        assert abs(with_proxy["ame_pp"]) < abs(without["ame_pp"])
    else:
        # Refused to estimate. Every numeric field must be empty, not defaulted.
        assert with_proxy["coef"] is None
        assert with_proxy["p_value"] is None
        assert with_proxy["ame_pp"] is None


# --------------------------------------------------------------------------- 5c


def test_proxy_scan_finds_the_planted_proxy(biased):
    result = proxy_scan(biased.frame, features=FIXTURE_FEATURES)
    ranked = [r["feature"] for r in result["ranked"]]
    assert ranked[0] == "neighborhood_score"

    top = result["ranked"][0]
    assert top["flagged"] is True
    assert top["max_nmi"] > result["negative_control"]["max_nmi"] * 3
    assert top["max_attribute"] == "race"


def test_constant_features_are_not_ranked_as_perfect_proxies(biased):
    """sklearn's NMI returns 1.0 when both labellings are constant, which would rank
    every constant feature as a perfect proxy. A constant feature carries no
    information, so it must score zero and be labelled as constant."""
    result = proxy_scan(biased.frame, features=FIXTURE_FEATURES)
    constant = [r for r in result["ranked"] if r["status"] == "constant"]
    assert constant, "the fixture has constant features; the guard should see them"
    for entry in constant:
        assert entry["max_nmi"] == 0.0
        assert entry["flagged"] is False


def test_proxy_scan_does_not_flag_the_negative_control(biased):
    """A seeded random column is included in every scan. If it were ever flagged,
    the threshold would be meaningless."""
    result = proxy_scan(biased.frame, features=FIXTURE_FEATURES)
    assert result["negative_control"]["flagged"] is False


def test_proxy_scan_is_quiet_on_the_clean_model(clean):
    result = proxy_scan(clean.frame, features=FIXTURE_FEATURES)
    by_name = {r["feature"]: r for r in result["ranked"]}
    assert by_name["neighborhood_score"]["flagged"] is False


def test_set_level_predictability_rises_with_the_proxy(biased, clean):
    """Pairwise mutual information misses joint proxying. Asking whether the feature
    set as a whole predicts group membership catches what pairwise scanning cannot,
    and 'the feature set predicts race with AUC 0.9' is a more legible finding than
    any single MI value."""
    hot = proxy_scan(biased.frame, features=FIXTURE_FEATURES)["set_level"]["macro_auc"]
    cold = proxy_scan(clean.frame, features=FIXTURE_FEATURES)["set_level"]["macro_auc"]
    assert hot > 0.70
    assert hot > cold + 0.10
    # The clean model still sits above chance, because income and loan-to-value
    # genuinely differ by group. That is the honest baseline, not a failure — and it
    # is exactly the joint signal a pairwise scan would miss.
    assert cold > 0.55


# --------------------------------------------------------------------------- all three


def test_all_three_analyses_agree_on_the_biased_model(biased):
    """The headline claim in the README: the tool has been shown to catch something."""
    a = raw_disparity(biased.frame, outcome="model_approved")["race"]["any_flag"]
    b = (
        controlled_disparity(
            biased.frame, outcome="model_approved", controls=FIXTURE_FEATURES[:-1]
        )["groups"][DISADVANTAGED]["p_value"]
        < 0.01
    )
    c = proxy_scan(biased.frame, features=FIXTURE_FEATURES)["ranked"][0]["flagged"]
    assert (a, b, c) == (True, True, True)
