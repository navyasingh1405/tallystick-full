"""Monotonicity is asserted empirically, not taken on trust from the training params.

Passing `monotone_constraints` to LightGBM does not, by itself, prove the shipped
booster honours them — a wrong feature order in the constraint list would silently
constrain the wrong columns. So sweep each constrained feature across its range on
sampled real rows and check the score never moves the wrong way.

The counterfactual binary search is only valid because this holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from tally.data.features import feature_matrix

N_ROWS = 500
N_STEPS = 25
TOL = 1e-9


@pytest.mark.slow
@pytest.mark.needs_model
def test_constraint_list_aligns_with_feature_order(model):
    assert len(model.monotone_constraints) == len(model.feature_names)
    from tally.config import get_config

    cfg = get_config()
    for name, direction in model.monotone.items():
        assert cfg.feature(name).monotone == direction


@pytest.mark.slow
@pytest.mark.needs_model
def test_score_is_monotone_in_every_constrained_feature(model, eval_frame):
    """For each constrained feature, sweep it across its observed range on 500 rows
    holding everything else fixed. The score must never move against the constraint."""
    sample = eval_frame.sample(n=min(N_ROWS, len(eval_frame)), random_state=7)
    X = feature_matrix(sample, model.fit_params)[model.feature_names].to_numpy(dtype=float)

    violations: list[str] = []
    for j, (name, direction) in enumerate(model.monotone.items()):
        if direction == 0:
            continue
        lo, hi = model.fit_params.feature_ranges[name]
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            continue
        grid = np.linspace(lo, hi, N_STEPS)

        # One prediction call per grid point over all rows, rather than per row.
        curves = np.empty((len(X), N_STEPS), dtype=float)
        for k, value in enumerate(grid):
            Xk = X.copy()
            Xk[:, j] = value
            curves[:, k] = model.raw_score(Xk)

        diffs = np.diff(curves, axis=1)
        bad = (diffs < -TOL) if direction == 1 else (diffs > TOL)
        n_bad = int(bad.sum())
        if n_bad:
            worst = float(np.abs(diffs[bad]).max())
            violations.append(
                f"{name} (constraint {direction:+d}): {n_bad} of {diffs.size} steps "
                f"move the wrong way, worst {worst:.2e}"
            )

    assert not violations, "monotonicity violated:\n  " + "\n  ".join(violations)


@pytest.mark.slow
@pytest.mark.needs_model
def test_unconstrained_features_are_declared_not_accidental(model):
    """A zero constraint has to be a documented decision, not an omission."""
    from tally.config import get_config

    cfg = get_config()
    for name, direction in model.monotone.items():
        if direction == 0:
            rationale = cfg.feature(name).rationale
            assert len(rationale.split()) >= 15, (
                f"{name} is unconstrained but its rationale is too thin to defend "
                "in the validation pack"
            )
