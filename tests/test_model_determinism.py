"""Two training runs on the same data and seed must produce the same model.

Without this, a decision recorded in the ledger cannot be replayed a year later:
"model v003" would not identify a specific set of bytes.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tally.config import get_config
from tally.data.features import fit_params
from tally.model.registry import data_fingerprint
from tally.model.train import train_booster


def _booster_hash(booster) -> str:
    """Hash the serialised model, ignoring the header line which carries a version
    string and can differ across environments without changing the model."""
    text = booster.model_to_string()
    body = "\n".join(line for line in text.splitlines() if not line.startswith("version="))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@pytest.mark.slow
@pytest.mark.needs_data
def test_two_runs_with_the_same_seed_produce_the_same_model(train_frame, eval_frame):
    cfg = get_config()
    fitted = fit_params(train_frame)
    kwargs = dict(
        feature_names=cfg.feature_names,
        monotone=cfg.monotone_constraints,
    )
    a = train_booster(train_frame, eval_frame, fitted, **kwargs)
    b = train_booster(train_frame, eval_frame, fitted, **kwargs)
    assert _booster_hash(a) == _booster_hash(b)


@pytest.mark.slow
@pytest.mark.needs_data
def test_predictions_are_bit_identical_across_runs(train_frame, eval_frame):
    cfg = get_config()
    fitted = fit_params(train_frame)
    from tally.data.features import feature_matrix

    X = feature_matrix(eval_frame.head(500), fitted)[cfg.feature_names]
    kwargs = dict(feature_names=cfg.feature_names, monotone=cfg.monotone_constraints)
    a = train_booster(train_frame, eval_frame, fitted, **kwargs).predict(X, raw_score=True)
    b = train_booster(train_frame, eval_frame, fitted, **kwargs).predict(X, raw_score=True)
    np.testing.assert_array_equal(a, b)


def test_data_fingerprint_is_order_independent():
    import pandas as pd

    a = data_fingerprint(pd.Series(["2024-0000003", "2024-0000001", "2024-0000002"]))
    b = data_fingerprint(pd.Series(["2024-0000001", "2024-0000002", "2024-0000003"]))
    assert a == b
    assert a["row_count"] == 3


def test_data_fingerprint_detects_a_changed_row():
    import pandas as pd

    a = data_fingerprint(pd.Series(["2024-0000001", "2024-0000002"]))
    b = data_fingerprint(pd.Series(["2024-0000001", "2024-0000003"]))
    assert a["row_id_hash"] != b["row_id_hash"]


@pytest.mark.slow
@pytest.mark.needs_model
def test_registry_load_verifies_the_model_hash(model, tmp_path):
    """A tampered booster file must fail to load, not load silently."""
    import shutil

    from tally.model.registry import ModelRegistry

    staging = tmp_path / "models"
    shutil.copytree(model.directory.parent, staging)
    booster_file = staging / model.version / "booster.txt"
    booster_file.write_text(booster_file.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        ModelRegistry(root=staging).load(model.version)
