"""Performance, segment, and calibration measurement.

Pure functions plus one artifact writer. Every number that reaches the validation
pack comes from here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tally.config import get_config
from tally.data.features import feature_matrix
from tally.data.splits import load_split
from tally.model.registry import load_current


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Maximum separation between the two class score distributions."""
    pos = np.sort(p[y == 1])
    neg = np.sort(p[y == 0])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    grid = np.sort(np.unique(np.concatenate([pos, neg])))
    cdf_pos = np.searchsorted(pos, grid, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, grid, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def basic_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p > threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ks": ks_statistic(y, p),
        "approval_rate": float(pred.mean()),
        "actual_approval_rate": float(y.mean()),
        "accuracy": float((pred == y).mean()),
        "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "brier": float(np.mean((p - y) ** 2)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def calibration_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Deciles of predicted probability against realised frequency."""
    order = np.argsort(p)
    y, p = y[order], p[order]
    chunks = np.array_split(np.arange(len(p)), bins)
    out = []
    for i, idx in enumerate(chunks):
        if len(idx) == 0:
            continue
        out.append(
            {
                "bin": i + 1,
                "n": int(len(idx)),
                "mean_predicted": float(p[idx].mean()),
                "observed": float(y[idx].mean()),
                "low": float(p[idx].min()),
                "high": float(p[idx].max()),
            }
        )
    return out


def _band(series: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=[-np.inf, *edges, np.inf], labels=labels, right=True)


def segment_breakdown(frame: pd.DataFrame, y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    """Performance by geography, loan size, and income band.

    SR 11-7 outcomes analysis expects performance to be shown by segment, not only
    in aggregate: a model can be accurate overall and poor for a subpopulation.
    """
    segments: dict[str, Any] = {}

    loan_labels = ["< $250k", "$250k-$500k", "$500k-$750k", "$750k+"]
    income_labels = ["< $75k", "$75k-$150k", "$150k-$250k", "$250k+"]

    definitions = {
        "loan_size": _band(frame["loan_amount_usd"], [250_000, 500_000, 750_000], loan_labels),
        "income": _band(frame["income_usd"], [75_000, 150_000, 250_000], income_labels),
        "county": frame["county_code"].astype(str),
    }

    for seg_name, keys in definitions.items():
        rows = []
        for key, idx in pd.Series(range(len(frame)), index=keys.to_numpy()).groupby(level=0):
            i = idx.to_numpy()
            if len(i) < 100:
                rows.append({"segment": str(key), "n": int(len(i)), "status": "insufficient sample"})
                continue
            ys, ps = y[i], p[i]
            rows.append(
                {
                    "segment": str(key),
                    "n": int(len(i)),
                    "status": "ok",
                    "auc": float(roc_auc_score(ys, ps)) if len(np.unique(ys)) > 1 else None,
                    "ks": ks_statistic(ys, ps),
                    "approval_rate": float((ps > threshold).mean()),
                    "actual_approval_rate": float(ys.mean()),
                }
            )
        rows.sort(key=lambda r: -r["n"])
        if seg_name == "county":
            rows = rows[:12]
        segments[seg_name] = rows

    return segments


def write_evaluation_artifacts() -> list[Path]:
    cfg = get_config()
    model = load_current()
    out: list[Path] = []

    performance: dict[str, Any] = {"model_version": model.version, "model_hash": model.model_hash}
    calibration: dict[str, Any] = {}
    segments: dict[str, Any] = {}

    for name in ("train", "test", "oot"):
        frame = load_split(name)
        X = feature_matrix(frame, model.fit_params)[model.feature_names]
        p = model.probability(X)
        y = frame["approved"].astype(int).to_numpy()
        performance[name] = basic_metrics(y, p, model.threshold)
        calibration[name] = calibration_curve(y, p)
        if name in ("test", "oot"):
            segments[name] = segment_breakdown(frame, y, p, model.threshold)

    for filename, payload in (
        ("performance.json", performance),
        ("calibration.json", calibration),
        ("segments.json", segments),
    ):
        path = cfg.paths.artifact("metrics", filename)
        path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        out.append(path)
    return out


def load_metric_artifact(name: str) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.paths.artifact("metrics", f"{name}.json")
    if not path.exists():
        raise FileNotFoundError(f"{name}.json missing — run: uv run tally evaluate")
    return json.loads(path.read_text(encoding="utf-8"))
