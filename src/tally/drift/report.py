"""Drift between the training vintage and the out-of-time vintage.

This is real distributional change measured on real data, not perturbation. HMDA
publishes annually and carries no finer date field, so the honest comparison is one
year against the next — which is also exactly the comparison a lender would run when
a new vintage lands.

Alerting is on effect size, never on p-values. At fifty thousand observations every
two-sample test is significant, so a p-value-driven alert fires on everything and is
then ignored. That is how drift monitoring becomes decoration.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from tally.config import get_config
from tally.data.features import feature_matrix
from tally.data.splits import load_split
from tally.model.registry import LoadedModel, load_current

EPSILON = 1e-6


def psi(reference: np.ndarray, comparison: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """Population Stability Index with bin edges frozen from the reference.

    Freezing the edges is the whole point: re-deriving quantiles on each period would
    measure the comparison against itself and report stability no matter what moved.
    """
    ref = reference[np.isfinite(reference)]
    cmp_ = comparison[np.isfinite(comparison)]
    if len(ref) == 0 or len(cmp_) == 0:
        return float("nan"), []

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(ref, quantiles))
    if len(edges) < 3:
        return 0.0, []
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cmp_counts, _ = np.histogram(cmp_, bins=edges)
    ref_share = np.maximum(ref_counts / len(ref), EPSILON)
    cmp_share = np.maximum(cmp_counts / len(cmp_), EPSILON)

    contributions = (cmp_share - ref_share) * np.log(cmp_share / ref_share)
    detail = [
        {
            "bin": i + 1,
            "low": float(edges[i]) if np.isfinite(edges[i]) else None,
            "high": float(edges[i + 1]) if np.isfinite(edges[i + 1]) else None,
            "reference_share": float(ref_share[i]),
            "comparison_share": float(cmp_share[i]),
            "contribution": float(contributions[i]),
        }
        for i in range(len(contributions))
    ]
    return float(contributions.sum()), detail


def band(value: float, bands: list[float]) -> str:
    if not np.isfinite(value):
        return "not computed"
    if value < bands[0]:
        return "stable"
    if value < bands[1]:
        return "moderate"
    return "significant"


def build_drift_report(model: LoadedModel | None = None) -> tuple[dict[str, Any], Path]:
    cfg = get_config()
    model = model or load_current()
    bands = list(cfg.drift["psi_bands"])
    bins = int(cfg.drift["psi_bins"])
    alert_d = float(cfg.drift["ks_alert_d"])

    reference = load_split("train")
    comparison = load_split("oot")

    Xref = feature_matrix(reference, model.fit_params)[model.feature_names]
    Xcmp = feature_matrix(comparison, model.fit_params)[model.feature_names]

    features: list[dict[str, Any]] = []
    for name in model.feature_names:
        ref = Xref[name].to_numpy(dtype=float)
        cmp_ = Xcmp[name].to_numpy(dtype=float)
        value, detail = psi(ref, cmp_, bins)
        ks = None
        if len(np.unique(ref)) > 2:
            ks = float(stats.ks_2samp(ref, cmp_).statistic)
        features.append(
            {
                "feature": name,
                "psi": value,
                "band": band(value, bands),
                "ks": ks,
                "reference_mean": float(np.nanmean(ref)),
                "comparison_mean": float(np.nanmean(cmp_)),
                "bins": detail,
                "alert": band(value, bands) == "significant"
                or (ks is not None and ks >= alert_d),
            }
        )

    features.sort(key=lambda f: -(f["psi"] if np.isfinite(f["psi"]) else -1))

    p_ref = model.probability(Xref)
    p_cmp = model.probability(Xcmp)
    score_psi, score_detail = psi(p_ref, p_cmp, bins)
    score_ks = float(stats.ks_2samp(p_ref, p_cmp).statistic)

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": {"version": model.version, "hash": model.model_hash},
        "reference": {
            "split": "train",
            "year": int(reference["activity_year"].iloc[0]),
            "n": int(len(reference)),
        },
        "comparison": {
            "split": "oot",
            "year": int(comparison["activity_year"].iloc[0]),
            "n": int(len(comparison)),
        },
        "features": features,
        "score": {
            "psi": score_psi,
            "band": band(score_psi, bands),
            "ks": score_ks,
            "reference_mean": float(p_ref.mean()),
            "comparison_mean": float(p_cmp.mean()),
            "bins": score_detail,
            "alert": band(score_psi, bands) == "significant" or score_ks >= alert_d,
        },
        "thresholds": {"psi_bands": bands, "ks_alert_d": alert_d},
        "bands_note": (
            f"Below {bands[0]} is read as stable, {bands[0]} to {bands[1]} as moderate, "
            f"and {bands[1]} and above as significant."
        ),
        "alerting_note": (
            "Alerts fire on effect size, never on p-values. With tens of thousands of "
            "observations on each side, every two-sample test is statistically "
            "significant regardless of whether the change matters, so a p-value-driven "
            "alert fires constantly and is then ignored."
        ),
        "summary": {
            "n_significant": sum(1 for f in features if f["band"] == "significant"),
            "n_moderate": sum(1 for f in features if f["band"] == "moderate"),
            "n_alerting": sum(1 for f in features if f["alert"]),
            "largest": features[0]["feature"] if features else None,
        },
    }

    path = cfg.paths.artifact("drift", "drift_report.json")
    path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return report, path


def load_drift_report() -> dict[str, Any]:
    cfg = get_config()
    path = cfg.paths.artifact("drift", "drift_report.json")
    if not path.exists():
        raise FileNotFoundError("drift_report.json missing — run: uv run tally drift")
    return json.loads(path.read_text(encoding="utf-8"))
