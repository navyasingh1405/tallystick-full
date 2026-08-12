"""The challenger grid and the search's resumability."""

from __future__ import annotations

import json

import pytest

from tally.lda.grid import PROXY_FEATURE, build_grid, fairness_weights, feature_subsets

BASELINE = [
    "log_income",
    "log_loan_amount",
    "cltv",
    "dti_band_ord",
    "loan_term",
    "interest_only",
    "balloon",
    "neg_amortization",
    "tract_to_msa_income_pct",
    "units_ord",
    "preapproval",
]


# --------------------------------------------------------------------------- grid


def test_the_grid_is_deterministic():
    """The search plan is an artifact. Two runs must enumerate the same challengers,
    or 'we looked and found nothing better' means nothing."""
    a = build_grid(BASELINE, ["cltv", "loan_term"])
    b = build_grid(BASELINE, ["cltv", "loan_term"])
    assert [c.as_dict() for c in a.challengers] == [c.as_dict() for c in b.challengers]


def test_the_grid_covers_feature_subsets_crossed_with_fairness_weight():
    grid = build_grid(BASELINE, ["cltv", "loan_term"])
    factorial = [c for c in grid.challengers if c.origin == "factorial"]
    weights = {c.fairness_weight for c in factorial}
    subsets = {c.subset for c in factorial}
    assert len(factorial) == len(weights) * len(subsets)


def test_the_known_proxy_is_offered_as_a_search_axis():
    """Including it is deliberate: the search prices what using a known geographic
    proxy would cost in measured disparity, rather than asserting it is unacceptable."""
    subsets = feature_subsets(BASELINE, ["cltv"])
    assert PROXY_FEATURE in subsets["add_known_proxy"]
    assert PROXY_FEATURE not in subsets["baseline"]


def test_the_search_drops_the_features_the_proxy_scan_ranked_highest():
    subsets = feature_subsets(BASELINE, ["cltv", "loan_term"])
    assert "cltv" not in subsets["drop_cltv"]
    assert "loan_term" not in subsets["drop_loan_term"]
    assert not {"cltv", "loan_term"} & set(subsets["drop_cltv_and_loan_term"])


def test_every_subset_keeps_at_least_two_features():
    for name, features in feature_subsets(BASELINE, ["cltv", "loan_term"]).items():
        assert len(features) >= 2, name


def test_the_quick_grid_is_much_smaller():
    full = build_grid(BASELINE, ["cltv"])
    quick = build_grid(BASELINE, ["cltv"], quick=True)
    assert len(quick.challengers) < len(full.challengers) / 3


def test_cache_keys_separate_configuration_from_data():
    grid = build_grid(BASELINE, ["cltv"])
    a, b = grid.challengers[0], grid.challengers[1]
    assert a.cache_key("data-1") != b.cache_key("data-1")
    # Same challenger, different training data, must not reuse a cached result.
    assert a.cache_key("data-1") != a.cache_key("data-2")
    assert a.cache_key("data-1") == a.cache_key("data-1")


# --------------------------------------------------------------------------- weights


def test_fairness_weights_reduce_to_uniform_at_zero():
    import pandas as pd

    groups = pd.Series(["A", "B", "A", "B"])
    weights = fairness_weights(groups, {"A": 0.9, "B": 0.5}, 0.0)
    assert list(weights) == [1.0, 1.0, 1.0, 1.0]


def test_fairness_weights_upweight_the_lower_rate_group():
    import pandas as pd

    groups = pd.Series(["A", "B"])
    weights = fairness_weights(groups, {"A": 0.9, "B": 0.45}, 1.0)
    assert weights[0] == pytest.approx(1.0)  # highest-rate group is the reference
    assert weights[1] == pytest.approx(2.0)  # half the rate, twice the weight


def test_fairness_weighting_is_monotone_in_lambda():
    import pandas as pd

    groups = pd.Series(["B"])
    previous = 0.0
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0):
        w = fairness_weights(groups, {"A": 0.9, "B": 0.6}, lam)[0]
        assert w >= previous
        previous = w


def test_weighting_needs_the_protected_attribute_only_at_fit_time():
    """The deployed model never sees a protected attribute. That is what makes this a
    defensible in-processing method rather than disparate treatment."""
    import inspect

    from tally.lda import search

    source = inspect.getsource(search._train_challenger)
    for banned in ("derived_race", "derived_sex", "applicant_age", "group"):
        assert banned not in source


# --------------------------------------------------------------------------- resume


def test_a_truncated_cache_line_is_skipped_not_fatal(tmp_path):
    """A killed process leaves a partial final line. It must be recomputed silently
    rather than crashing the next run."""
    from tally.lda.search import _load_cache

    path = tmp_path / "results.jsonl"
    good = {"cache_key": "abc", "id": "c000", "auc": 0.9}
    path.write_text(json.dumps(good) + "\n" + '{"cache_key": "def", "auc": 0.', encoding="utf-8")

    cache = _load_cache(path)
    assert set(cache) == {"abc"}


def test_a_missing_cache_file_is_an_empty_cache(tmp_path):
    from tally.lda.search import _load_cache

    assert _load_cache(tmp_path / "nothing.jsonl") == {}


def test_blank_lines_are_tolerated(tmp_path):
    from tally.lda.search import _load_cache

    path = tmp_path / "r.jsonl"
    path.write_text('\n{"cache_key":"a"}\n\n{"cache_key":"b"}\n\n', encoding="utf-8")
    assert set(_load_cache(path)) == {"a", "b"}


@pytest.mark.slow
@pytest.mark.needs_model
def test_the_completed_search_recorded_what_it_searched():
    from tally.lda.search import load_lda_summary

    try:
        summary = load_lda_summary()
    except FileNotFoundError:
        pytest.skip("no lda artifact — run: uv run tally lda")

    assert summary["n_challengers"] > 0
    assert summary["frontier"]
    assert summary["headline"]
    assert summary["grid_axes"]["seed"]
    assert "bounded hypothesis space" in summary["doctrine_note"]


@pytest.mark.slow
@pytest.mark.needs_model
def test_the_search_records_whether_its_target_was_real():
    """A frontier is only meaningful if the quantity on its axis is distinguishable
    from zero for the incumbent."""
    from tally.lda.search import load_lda_summary

    try:
        objective = load_lda_summary()["objective"]
    except FileNotFoundError:
        pytest.skip("no lda artifact")

    assert "focal_group" in objective
    assert "target_is_significant" in objective
    if not objective["target_is_significant"]:
        assert objective.get("warning"), "an insignificant target must carry a warning"


@pytest.mark.slow
@pytest.mark.needs_model
def test_frontier_members_are_not_dominated():
    """The property the whole feature rests on, checked against the real results."""
    from tally.lda.pareto import Point, dominates
    from tally.lda.search import load_lda_summary

    try:
        summary = load_lda_summary()
    except FileNotFoundError:
        pytest.skip("no lda artifact")

    points = [
        Point(r["id"], round(r["auc"], 6), round(r["disparity_ame"], 6))
        for r in summary["results"]
        if r["disparity_status"] == "ok"
    ]
    frontier_ids = {r["id"] for r in summary["frontier"]}
    for member in (p for p in points if p.id in frontier_ids):
        assert not any(dominates(other, member) for other in points), member.id
