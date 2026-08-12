"""Train / test / out-of-time splits.

The out-of-time frame is a different activity year, not a random slice. HMDA's
public LAR carries no date field beyond `activity_year`, so an intra-year temporal
split is impossible; using the following year is the only honest way to get an
out-of-time frame, and it doubles as the drift comparison in F13.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tally.config import get_config


def make_splits(train_year_frame: Path, oot_year_frame: Path) -> dict[str, Any]:
    cfg = get_config()
    seed = int(cfg.split["seed"])
    test_size = float(cfg.split["test_size"])
    out = cfg.paths.data_processed

    dev = pd.read_parquet(train_year_frame)
    oot = pd.read_parquet(oot_year_frame)

    # Deterministic split on a stable key, not on row position: reordering the
    # source file must not reshuffle the split.
    dev = dev.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    rng = _stable_uniform(dev["row_id"], seed)
    is_test = rng < test_size

    train = dev.loc[~is_test].reset_index(drop=True)
    test = dev.loc[is_test].reset_index(drop=True)
    oot = oot.sort_values("row_id", kind="mergesort").reset_index(drop=True)

    train.to_parquet(out / "train.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    oot.to_parquet(out / "oot.parquet", index=False)

    train_year = int(dev["activity_year"].iloc[0])
    oot_year = int(oot["activity_year"].iloc[0])

    return {
        "method": "hash-partitioned holdout within the training year; the following "
        "year is held out entirely as an out-of-time frame",
        "seed": seed,
        "test_size": test_size,
        "train": {
            "rows": int(len(train)),
            "year": train_year,
            "approval_rate": round(float(train["approved"].mean()), 4),
            "note": f"{train_year} development sample",
        },
        "test": {
            "rows": int(len(test)),
            "year": train_year,
            "approval_rate": round(float(test["approved"].mean()), 4),
            "note": f"{train_year} in-time holdout",
        },
        "oot": {
            "rows": int(len(oot)),
            "year": oot_year,
            "approval_rate": round(float(oot["approved"].mean()), 4),
            "note": f"{oot_year} out-of-time holdout — never seen in training",
        },
    }


def _stable_uniform(keys: pd.Series, seed: int) -> pd.Series:
    """Map each key deterministically into [0, 1) using a seeded content hash."""
    import hashlib

    salt = str(seed).encode()

    def one(k: str) -> float:
        h = hashlib.sha256(salt + k.encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    return keys.map(one)


def load_split(name: str) -> pd.DataFrame:
    cfg = get_config()
    path = cfg.paths.data_processed / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{name}.parquet missing — run: uv run tally data")
    return pd.read_parquet(path)
