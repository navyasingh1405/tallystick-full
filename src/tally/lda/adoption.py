"""What adopting a challenger would actually do.

A frontier plot shows that a trade exists. A committee cannot act on that: it needs to
know how many applications change outcome, in which direction, and for whom.

The framing decision that matters
---------------------------------
Comparing incumbent and challenger at the same numeric threshold conflates two
different things: a change in who gets approved, and a change in how the model
calibrates. A challenger might approve five percent more people simply because its
scores run higher, and the disparity would move for reasons that have nothing to do
with fairness.

So the challenger's threshold is recalibrated to hold the approval rate equal to the
incumbent's. The comparison then reads "same volume, different applicants", which is
the question a committee is actually asking. The fixed-threshold comparison is
reported alongside it, labelled, because it is what would happen if the threshold were
left alone.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from tally.config import get_config
from tally.data.features import feature_matrix
from tally.data.splits import load_split
from tally.fairness.controlled import controlled_disparity
from tally.fairness.groups import assign_groups
from tally.lda.grid import Challenger
from tally.lda.search import INCUMBENT_ID, _train_challenger
from tally.model.metrics import ks_statistic
from tally.model.registry import load_current


def _challenger_from_result(result: dict[str, Any]) -> Challenger:
    params = result["params"]
    return Challenger(
        id=result["id"],
        subset=result["subset"],
        features=tuple(result["features"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        min_child_samples=int(params["min_child_samples"]),
        reg_lambda=float(params["reg_lambda"]),
        fairness_weight=float(params["fairness_weight"]),
        origin=result.get("origin", "random"),
    )


def _threshold_for_rate(scores: np.ndarray, target_rate: float) -> float:
    """The cutoff that approves exactly `target_rate` of this population."""
    return float(np.quantile(scores, 1.0 - target_rate))


def adoption_report(challenger_id: str, *, hold_approval_rate: bool = True) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    from tally.fairness.report import scored_evaluation_frame
    from tally.lda.grid import fairness_weights
    from tally.lda.search import load_lda_summary

    cfg = get_config()
    model = load_current()
    summary = load_lda_summary()

    entry = next((r for r in summary["results"] if r["id"] == challenger_id), None)
    if entry is None:
        raise KeyError(f"no challenger {challenger_id!r} in the search results")
    if challenger_id == INCUMBENT_ID:
        raise ValueError("the incumbent cannot be adopted against itself")

    train = load_split("train")
    valid = load_split("test")
    eval_frame = scored_evaluation_frame(model, "oot")
    groups = assign_groups(eval_frame)
    controls = list(model.feature_names)

    challenger = _challenger_from_result(entry)
    weights = None
    if challenger.fairness_weight > 0:
        train_groups = assign_groups(train)["race"]
        base_rates = (
            train.assign(_g=train_groups)
            .dropna(subset=["_g"])
            .groupby("_g")["approved"]
            .mean()
            .to_dict()
        )
        weights = fairness_weights(
            train_groups.fillna("_unassigned"), base_rates, challenger.fairness_weight
        )

    booster = _train_challenger(challenger, train, valid, model, weights)

    X_inc = feature_matrix(eval_frame, model.fit_params)[model.feature_names]
    X_chl = feature_matrix(eval_frame, model.fit_params)[list(challenger.features)]

    p_inc = model.probability(X_inc)
    p_chl = 1.0 / (1.0 + np.exp(-booster.predict(X_chl, raw_score=True)))
    y = eval_frame["approved"].astype(int).to_numpy()

    incumbent_rate = float((p_inc > model.threshold).mean())
    fixed_threshold = model.threshold
    matched_threshold = _threshold_for_rate(p_chl, incumbent_rate)
    threshold = matched_threshold if hold_approval_rate else fixed_threshold

    approve_inc = p_inc > model.threshold
    approve_chl = p_chl > threshold

    gained = approve_chl & ~approve_inc  # declined by incumbent, approved by challenger
    lost = approve_inc & ~approve_chl

    scored = eval_frame.copy()
    scored["model_approved"] = approve_chl
    challenger_disparity = controlled_disparity(
        scored, outcome="model_approved", controls=controls, groups=groups
    )
    incumbent_disparity = controlled_disparity(
        eval_frame.assign(model_approved=approve_inc),
        outcome="model_approved",
        controls=controls,
        groups=groups,
    )

    report: dict[str, Any] = {
        "challenger_id": challenger_id,
        "model_version": model.version,
        "subset": challenger.subset,
        "features": list(challenger.features),
        "feature_delta": {
            "removed": [f for f in model.feature_names if f not in challenger.features],
            "added": [f for f in challenger.features if f not in model.feature_names],
        },
        "params": challenger.config(),
        "threshold": {
            "incumbent": fixed_threshold,
            "challenger_applied": threshold,
            "challenger_recalibrated": matched_threshold,
            "hold_approval_rate": hold_approval_rate,
            "note": (
                "The challenger's threshold is recalibrated so it approves the same "
                "share of applications as the incumbent. Without that, a difference "
                "in calibration would masquerade as a difference in fairness."
                if hold_approval_rate
                else "Both models are evaluated at the incumbent's threshold, so the "
                "approval volume changes as well as its composition."
            ),
        },
        "performance": {
            "incumbent": {
                "auc": float(roc_auc_score(y, p_inc)),
                "ks": float(ks_statistic(y, p_inc)),
                "approval_rate": incumbent_rate,
            },
            "challenger": {
                "auc": float(roc_auc_score(y, p_chl)),
                "ks": float(ks_statistic(y, p_chl)),
                "approval_rate": float(approve_chl.mean()),
            },
        },
        "flips": {
            "n_evaluated": int(len(eval_frame)),
            "approvals_gained": int(gained.sum()),
            "approvals_lost": int(lost.sum()),
            "net": int(gained.sum() - lost.sum()),
            "unchanged": int((approve_inc == approve_chl).sum()),
            "composition": _flip_composition(groups["race"], gained, lost),
        },
        "disparity": _disparity_delta(incumbent_disparity, challenger_disparity),
        "interpretation_note": (
            "The applications that change outcome are the substance of this decision. "
            "A challenger that improves an aggregate fairness metric while declining "
            "applicants who would have been approved is a different proposition from "
            "one that approves applicants who would have been declined, even when the "
            "metric moves identically."
        ),
    }

    path = cfg.paths.artifact("adoption", f"{challenger_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return report


def _flip_composition(
    race: pd.Series, gained: np.ndarray, lost: np.ndarray
) -> list[dict[str, Any]]:
    """Who changes outcome. The number a committee remembers."""
    rows: list[dict[str, Any]] = []
    assigned = race.notna().to_numpy()
    for group in sorted(race.dropna().unique()):
        mask = (race == group).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        g = int((gained & mask).sum())
        lo = int((lost & mask).sum())
        rows.append(
            {
                "group": str(group),
                "n": n,
                "approvals_gained": g,
                "approvals_lost": lo,
                "net": g - lo,
                "net_rate_pp": round(100.0 * (g - lo) / n, 3),
                "share_of_gains": round(g / max(int((gained & assigned).sum()), 1), 4),
            }
        )
    rows.sort(key=lambda r: -r["n"])
    return rows


def _disparity_delta(incumbent: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"groups": [], "reference_group": incumbent.get("reference_group")}
    for group, inc in incumbent.get("groups", {}).items():
        chl = challenger.get("groups", {}).get(group, {})
        if inc.get("status") != "ok" or chl.get("status") != "ok":
            out["groups"].append(
                {
                    "group": group,
                    "status": inc.get("status") if inc.get("status") != "ok" else chl.get("status"),
                }
            )
            continue
        out["groups"].append(
            {
                "group": group,
                "status": "ok",
                "n": inc["n_focal"],
                "incumbent_ame_pp": inc["ame_pp"],
                "incumbent_ci": [inc["ci_low"], inc["ci_high"]],
                "incumbent_p": inc["p_value"],
                "challenger_ame_pp": chl["ame_pp"],
                "challenger_ci": [chl["ci_low"], chl["ci_high"]],
                "challenger_p": chl["p_value"],
                "delta_pp": chl["ame_pp"] - inc["ame_pp"],
                "improved": abs(chl["ame_pp"]) < abs(inc["ame_pp"]),
            }
        )
    return out
