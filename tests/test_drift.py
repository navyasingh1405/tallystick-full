"""Drift measurement."""

from __future__ import annotations

import numpy as np
import pytest

from tally.drift.report import band, psi


def test_identical_distributions_have_no_drift():
    x = np.random.default_rng(0).normal(size=5000)
    value, _ = psi(x, x.copy())
    assert value == pytest.approx(0.0, abs=1e-9)


def test_a_shifted_distribution_registers_drift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 8000)
    shifted = rng.normal(1.2, 1, 8000)
    value, _ = psi(reference, shifted)
    assert value > 0.25
    assert band(value, [0.10, 0.25]) == "significant"


def test_drift_is_graded_not_binary():
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 20000)
    small, _ = psi(reference, rng.normal(0.05, 1, 20000))
    medium, _ = psi(reference, rng.normal(0.35, 1, 20000))
    large, _ = psi(reference, rng.normal(1.5, 1, 20000))
    assert small < medium < large
    assert band(small, [0.10, 0.25]) == "stable"
    assert band(large, [0.10, 0.25]) == "significant"


def test_bin_edges_come_from_the_reference_only():
    """Re-deriving quantiles on each period would compare the comparison against
    itself and report stability no matter what moved. This asserts the edges are
    frozen: a shifted distribution must not be able to hide by being self-consistent."""
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, 10000)
    # Same shape, different location. Self-quantiled bins would show zero drift.
    shifted = rng.normal(3.0, 1, 10000)
    value, detail = psi(reference, shifted)
    assert value > 1.0
    assert len(detail) >= 5


def test_empty_bins_do_not_produce_infinity():
    """A comparison bin with no observations would give log(0) without the epsilon
    floor, and one empty bin would make the whole statistic unusable."""
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, 5000)
    disjoint = rng.normal(50, 0.1, 5000)
    value, _ = psi(reference, disjoint)
    assert np.isfinite(value)
    assert value > 1.0


def test_a_constant_reference_is_reported_rather_than_dividing_by_zero():
    value, detail = psi(np.zeros(1000), np.zeros(1000))
    assert value == 0.0
    assert detail == []


def test_empty_input_is_not_computed():
    value, _ = psi(np.array([]), np.array([1.0, 2.0]))
    assert np.isnan(value)
    assert band(value, [0.1, 0.25]) == "not computed"


@pytest.mark.slow
@pytest.mark.needs_model
def test_the_report_compares_two_real_vintages():
    from tally.drift.report import load_drift_report

    try:
        report = load_drift_report()
    except FileNotFoundError:
        pytest.skip("no drift report — run: uv run tally drift")

    assert report["reference"]["year"] != report["comparison"]["year"]
    assert report["features"]
    assert "psi" in report["score"] and "ks" in report["score"]


@pytest.mark.slow
@pytest.mark.needs_model
def test_alerting_is_on_effect_size_not_p_values():
    """At fifty thousand observations every two-sample test is significant, so a
    p-value-driven alert fires on everything and is then ignored."""
    from tally.drift.report import load_drift_report

    try:
        report = load_drift_report()
    except FileNotFoundError:
        pytest.skip("no drift report")

    serialised = str(report)
    assert "p_value" not in serialised
    assert set(report["thresholds"]) == {"psi_bands", "ks_alert_d"}
    assert "p-value" in report["alerting_note"]
