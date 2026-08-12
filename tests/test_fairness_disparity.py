"""5a and 5c unit behaviour, plus the guarantees the report must keep on real data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tally.fairness.disparity import FOUR_FIFTHS_CAVEAT, raw_disparity
from tally.fairness.groups import assign_groups
from tally.fairness.proxy import NEGATIVE_CONTROL, proxy_scan


def _frame(n_per_group: dict[str, int], rates: dict[str, float], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for group, n in n_per_group.items():
        race, eth = {
            "NH_White": ("White", "Not Hispanic or Latino"),
            "Black": ("Black or African American", "Not Hispanic or Latino"),
            "Hispanic": ("White", "Hispanic or Latino"),
        }[group]
        for _ in range(n):
            rows.append(
                {
                    "derived_race": race,
                    "derived_ethnicity": eth,
                    "derived_sex": "Male",
                    "applicant_age": "35-44",
                    "county_code": "08001",
                    "model_approved": rng.random() < rates[group],
                    "x": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 5a


def test_ratio_is_taken_against_the_highest_rate_group_not_the_reference():
    """The four-fifths guideline compares against the most-favoured group, which is
    not necessarily the designated reference group."""
    df = _frame(
        {"NH_White": 1000, "Black": 1000, "Hispanic": 1000},
        {"NH_White": 0.80, "Black": 0.95, "Hispanic": 0.60},
    )
    report = raw_disparity(df, outcome="model_approved")
    assert report["race"]["highest_rate_group"] == "Black"
    rows = {g["group"]: g for g in report["race"]["groups"]}
    assert rows["NH_White"]["ratio"] == pytest.approx(
        rows["NH_White"]["approval_rate"] / rows["Black"]["approval_rate"]
    )


def test_the_screening_caveat_travels_with_the_numbers():
    """The caveat has to be attached to the output, not left in prose someone can
    quote the table without."""
    df = _frame({"NH_White": 500, "Black": 500}, {"NH_White": 0.9, "Black": 0.5})
    report = raw_disparity(df, outcome="model_approved")
    assert report["caveat"] == FOUR_FIFTHS_CAVEAT
    assert "not the legal standard" in report["caveat"]


def test_small_groups_get_no_rate_and_no_ratio():
    df = _frame({"NH_White": 1000, "Black": 20}, {"NH_White": 0.9, "Black": 0.1})
    rows = {g["group"]: g for g in raw_disparity(df, outcome="model_approved")["race"]["groups"]}
    assert rows["Black"]["status"] == "insufficient_sample"
    assert rows["Black"]["approval_rate"] is None
    assert rows["Black"]["ratio"] is None
    assert rows["Black"]["four_fifths_flag"] is None


def test_confidence_intervals_stay_inside_zero_to_one():
    """Wilson rather than normal approximation, because rates here sit near 1."""
    df = _frame({"NH_White": 200, "Black": 150}, {"NH_White": 0.995, "Black": 0.99})
    for row in raw_disparity(df, outcome="model_approved")["race"]["groups"]:
        if row["status"] == "ok":
            assert 0.0 <= row["ci_low"] <= row["ci_high"] <= 1.0


def test_flag_fires_below_four_fifths_and_not_above():
    below = _frame({"NH_White": 2000, "Black": 2000}, {"NH_White": 0.90, "Black": 0.65})
    above = _frame({"NH_White": 2000, "Black": 2000}, {"NH_White": 0.90, "Black": 0.85})
    b = {g["group"]: g for g in raw_disparity(below, outcome="model_approved")["race"]["groups"]}
    a = {g["group"]: g for g in raw_disparity(above, outcome="model_approved")["race"]["groups"]}
    assert b["Black"]["four_fifths_flag"] is True
    assert a["Black"]["four_fifths_flag"] is False


# --------------------------------------------------------------------------- 5c


def test_the_negative_control_is_present_in_every_scan():
    df = _frame({"NH_White": 1000, "Black": 1000}, {"NH_White": 0.9, "Black": 0.6})
    result = proxy_scan(df, features=["x"])
    assert result["negative_control"]["feature"] == NEGATIVE_CONTROL
    assert result["negative_control"]["flagged"] is False
    assert NEGATIVE_CONTROL not in [r["feature"] for r in result["ranked"]]


def test_a_perfect_proxy_ranks_far_above_the_noise_floor():
    df = _frame({"NH_White": 1500, "Black": 1500}, {"NH_White": 0.9, "Black": 0.6})
    groups = assign_groups(df)
    df["perfect_proxy"] = (groups["race"] == "Black").astype(float)
    result = proxy_scan(df, features=["x", "perfect_proxy"])
    assert result["ranked"][0]["feature"] == "perfect_proxy"
    assert result["ranked"][0]["flagged"] is True
    assert result["ranked"][0]["control_multiple"] > 10


def test_scan_reports_rank_and_multiple_of_the_noise_floor():
    """At large n every real feature has some dependence on group, so the absolute
    NMI is hard to read on its own. The multiple of the control is what makes it
    interpretable."""
    df = _frame({"NH_White": 1000, "Black": 1000}, {"NH_White": 0.9, "Black": 0.6})
    result = proxy_scan(df, features=["x"])
    assert result["ranked"][0]["rank"] == 1
    assert "control_multiple" in result["ranked"][0]
    assert "calibrated heuristic" in result["threshold_note"]


# --------------------------------------------------------------------------- real data


@pytest.mark.slow
@pytest.mark.needs_model
def test_the_known_geographic_proxy_is_flagged_on_real_data():
    """tract_minority_population_percent is a textbook proxy for race and is excluded
    from the model. The scan must still find it, and rank it above every feature the
    model does use — otherwise the detector is not calibrated."""
    from tally.fairness.report import load_fairness_report

    try:
        report = load_fairness_report()
    except FileNotFoundError:
        pytest.skip("no fairness report — run: uv run tally fairness")

    ranked = report["proxy"]["ranked"]
    top = ranked[0]
    assert top["feature"] == "tract_minority_pct"
    assert top["flagged"] is True
    assert top["in_model"] is False
    assert not any(r["flagged"] for r in ranked if r["in_model"])


@pytest.mark.slow
@pytest.mark.needs_model
def test_groups_below_the_minimum_are_never_given_a_ratio_on_real_data():
    from tally.fairness.report import load_fairness_report

    try:
        report = load_fairness_report()
    except FileNotFoundError:
        pytest.skip("no fairness report — run: uv run tally fairness")

    insufficient = [
        g
        for attr in ("race", "sex", "age")
        for g in report["raw_disparity"][attr]["groups"]
        if g["status"] == "insufficient_sample"
    ]
    assert insufficient, "expected at least one small group in the real evaluation frame"
    for g in insufficient:
        assert g["approval_rate"] is None and g["ratio"] is None
