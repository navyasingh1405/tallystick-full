"""Assembles the three analyses into one artifact.

The evaluation frame is the out-of-time year, not a slice of the training year.
Measuring fairness on data the model has never seen is both the stronger claim and
the larger sample, and it is what an examiner would ask for.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tally.config import get_config
from tally.data.features import feature_matrix
from tally.data.splits import load_split
from tally.fairness.controlled import controlled_suite, primary_disparity_ame
from tally.fairness.disparity import raw_disparity
from tally.fairness.groups import ATTRIBUTES, assign_groups, exclusion_summary
from tally.fairness.proxy import proxy_scan
from tally.model.registry import LoadedModel, load_current


def scored_evaluation_frame(
    model: LoadedModel | None = None, split: str = "oot"
) -> pd.DataFrame:
    """The evaluation frame with the model's decisions attached.

    Protected attributes ride along on this frame and are never passed to
    `feature_matrix`, which raises if they are.
    """
    model = model or load_current()
    frame = load_split(split).copy()
    X = feature_matrix(frame, model.fit_params)
    probability = model.probability(X[model.feature_names])

    frame["model_probability"] = probability
    frame["model_approved"] = probability > model.threshold
    frame["model_decision"] = np.where(frame["model_approved"], "APPROVED", "DECLINED")

    # The excluded proxy is carried onto the frame for the scan only.
    for col in X.columns:
        if col not in frame.columns:
            frame[col] = X[col]
    return frame


def build_fairness_report(
    model: LoadedModel | None = None, split: str = "oot"
) -> tuple[dict[str, Any], Path]:
    cfg = get_config()
    model = model or load_current()
    frame = scored_evaluation_frame(model, split)
    groups = assign_groups(frame)

    controls = list(model.feature_names)
    # The known proxy is scanned even though the model excludes it — demonstrating
    # that the detector fires on a textbook geographic proxy is the point.
    scan_features = [*controls, "tract_minority_pct"]

    raw = raw_disparity(frame, outcome="model_approved", groups=groups)
    controlled = controlled_suite(
        frame,
        model_outcome="model_approved",
        historical_outcome="approved",
        controls=controls,
        groups=groups,
    )
    proxies = proxy_scan(frame, features=scan_features, groups=groups)

    significant = [
        {"attribute": attr, "group": g, "ame_pp": v["ame_pp"], "p_value": v["p_value"]}
        for attr in ATTRIBUTES
        for g, v in controlled["specifications"][attr]["primary"]["groups"].items()
        if v.get("status") == "ok" and v.get("p_value") is not None and v["p_value"] < 0.05
    ]

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": {"version": model.version, "hash": model.model_hash},
        "evaluation_frame": {
            "split": split,
            "n": int(len(frame)),
            "activity_year": int(frame["activity_year"].iloc[0]),
            "note": (
                "Fairness is measured on the out-of-time frame, which the model has "
                "never seen. It is both the larger sample and the stronger claim."
            ),
        },
        "group_exclusions": exclusion_summary(groups),
        "raw_disparity": raw,
        "controlled": controlled,
        "proxy": proxies,
        "primary_disparity_ame": primary_disparity_ame(
            frame, outcome="model_approved", controls=controls, groups=groups
        ),
        "summary": {
            "four_fifths_flags": sum(
                1
                for attr in ATTRIBUTES
                for g in raw[attr]["groups"]
                if g.get("four_fifths_flag")
            ),
            "significant_controlled": len(significant),
            "significant_detail": significant,
            "proxy_flags": proxies["n_flagged"],
            "set_level_auc": proxies["set_level"].get("macro_auc"),
        },
    }

    path = cfg.paths.artifact("fairness", "fairness_report.json")
    path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    return report, path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def load_fairness_report() -> dict[str, Any]:
    cfg = get_config()
    path = cfg.paths.artifact("fairness", "fairness_report.json")
    if not path.exists():
        raise FileNotFoundError("fairness_report.json missing — run: uv run tally fairness")
    return json.loads(path.read_text(encoding="utf-8"))
