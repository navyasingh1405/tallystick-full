"""The less discriminatory alternative search.

Under disparate impact doctrine a business-justified model is still challengeable if
a comparably effective, less discriminatory alternative exists. Nobody can answer
that by inspection, so this searches for one.

Every challenger is measured by the same frozen scalar as the incumbent
(`primary_disparity_ame`), computed by one function on one frame. If that number
were computed inline anywhere else, the frontier would be comparing quantities that
merely look alike.

The search is the slowest thing in the repository, so it is resumable: results are
appended one JSON object per line, and a restart skips anything already computed. A
truncated final line from a killed process fails to parse and is simply recomputed.

Concurrency is deliberately sequential. LightGBM already parallelises internally, and
a process pool on Windows means `spawn`, which re-imports LightGBM, SHAP, and
statsmodels in every worker — a reliable way to turn a twenty-minute search into a
hang.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tally.config import get_config
from tally.data.features import feature_matrix
from tally.data.splits import load_split
from tally.fairness.controlled import primary_disparity_ame
from tally.fairness.groups import assign_groups
from tally.lda.grid import Challenger, build_grid, fairness_weights
from tally.lda.pareto import Point, frontier, headline
from tally.model.metrics import ks_statistic
from tally.model.registry import LoadedModel, load_current

INCUMBENT_ID = "incumbent"


@dataclass
class ChallengerResult:
    id: str
    subset: str
    origin: str
    features: list[str]
    params: dict[str, Any]
    auc: float
    ks: float
    approval_rate: float
    disparity_ame: float
    disparity_status: str
    cache_key: str
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subset": self.subset,
            "origin": self.origin,
            "features": self.features,
            "n_features": len(self.features),
            "params": self.params,
            "auc": self.auc,
            "ks": self.ks,
            "approval_rate": self.approval_rate,
            "disparity_ame": self.disparity_ame,
            "disparity_status": self.disparity_status,
            "cache_key": self.cache_key,
            "seconds": round(self.seconds, 2),
        }


def _train_challenger(
    challenger: Challenger,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    model: LoadedModel,
    weights: np.ndarray | None,
):
    import lightgbm as lgb

    cfg = get_config()
    m = cfg.model
    features = list(challenger.features)
    monotone = [_constraint(f) for f in features]

    Xtr = feature_matrix(train, model.fit_params)[features]
    Xva = feature_matrix(valid, model.fit_params)[features]

    params = dict(m["params"])
    params.update(
        {
            "num_leaves": challenger.num_leaves,
            "max_depth": challenger.max_depth,
            "min_child_samples": challenger.min_child_samples,
            "reg_lambda": challenger.reg_lambda,
            "monotone_constraints": monotone,
            "seed": m["seed"],
            "bagging_seed": m["seed"],
            "feature_fraction_seed": m["seed"],
            "data_random_seed": m["seed"],
            "num_threads": m["num_threads"],
        }
    )

    dtrain = lgb.Dataset(Xtr, label=train["approved"].astype(int), weight=weights)
    dvalid = lgb.Dataset(Xva, label=valid["approved"].astype(int), reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=int(m["num_boost_round"]),
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(int(m["early_stopping_rounds"]), verbose=False)],
    )


def _constraint(feature: str) -> int:
    cfg = get_config()
    try:
        return cfg.feature(feature).monotone
    except KeyError:
        # The known proxy is not a configured model feature; leave it unconstrained
        # so the search measures its effect rather than shaping it.
        return 0


def _evaluate(
    challenger: Challenger,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    eval_frame: pd.DataFrame,
    groups: pd.DataFrame,
    model: LoadedModel,
    base_rates: dict[str, float],
    cache_key: str,
    controls: list[str],
) -> ChallengerResult:
    from sklearn.metrics import roc_auc_score

    started = time.perf_counter()
    weights = (
        fairness_weights(
            assign_groups(train)["race"].fillna("_unassigned"),
            base_rates,
            challenger.fairness_weight,
        )
        if challenger.fairness_weight > 0
        else None
    )

    booster = _train_challenger(challenger, train, valid, model, weights)

    features = list(challenger.features)
    Xev = feature_matrix(eval_frame, model.fit_params)[features]
    probability = 1.0 / (1.0 + np.exp(-booster.predict(Xev, raw_score=True)))
    y = eval_frame["approved"].astype(int).to_numpy()

    scored = eval_frame.copy()
    scored["model_approved"] = probability > model.threshold

    # The control set is FIXED across every challenger and the incumbent. If each
    # challenger were controlled for its own feature set, the frontier's horizontal
    # axis would not be one quantity — and worse, a challenger that used a proxy
    # would look fairer, because the proxy would absorb group membership among the
    # controls. Holding the controls fixed measures every model against the same
    # definition of "explained by legitimate credit factors".
    ame = primary_disparity_ame(
        scored, outcome="model_approved", controls=controls, groups=groups
    )

    return ChallengerResult(
        id=challenger.id,
        subset=challenger.subset,
        origin=challenger.origin,
        features=features,
        params=challenger.config(),
        auc=float(roc_auc_score(y, probability)),
        ks=float(ks_statistic(y, probability)),
        approval_rate=float(scored["model_approved"].mean()),
        disparity_ame=float(ame),
        disparity_status="ok" if np.isfinite(ame) else "not_estimable",
        cache_key=cache_key,
        seconds=time.perf_counter() - started,
    )


def _incumbent_result(
    model: LoadedModel, eval_frame: pd.DataFrame, groups: pd.DataFrame, controls: list[str]
) -> ChallengerResult:
    from sklearn.metrics import roc_auc_score

    X = feature_matrix(eval_frame, model.fit_params)[model.feature_names]
    probability = model.probability(X)
    y = eval_frame["approved"].astype(int).to_numpy()

    scored = eval_frame.copy()
    scored["model_approved"] = probability > model.threshold
    ame = primary_disparity_ame(
        scored, outcome="model_approved", controls=controls, groups=groups
    )

    return ChallengerResult(
        id=INCUMBENT_ID,
        subset="baseline",
        origin="incumbent",
        features=list(model.feature_names),
        params={"model_version": model.version},
        auc=float(roc_auc_score(y, probability)),
        ks=float(ks_statistic(y, probability)),
        approval_rate=float(scored["model_approved"].mean()),
        disparity_ame=float(ame),
        disparity_status="ok" if np.isfinite(ame) else "not_estimable",
        cache_key="incumbent",
        seconds=0.0,
    )


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A partial final line from a killed process. Skip and recompute.
                continue
            out[row["cache_key"]] = row
    return out


def run_search(*, quick: bool = False, resume: bool = True) -> dict[str, Any]:
    from tally.fairness.report import load_fairness_report, scored_evaluation_frame

    cfg = get_config()
    model = load_current()
    train = load_split("train")
    valid = load_split("test")

    # The same frame construction the fairness report uses, so the transformed
    # feature columns exist and the disparity measured here is the same quantity
    # reported in section 7 rather than one that merely resembles it.
    eval_frame = scored_evaluation_frame(model, "oot")
    groups = assign_groups(eval_frame)
    controls = list(model.feature_names)

    # The proxy scan tells the search which features are worth trying to remove, so
    # the space is informed by measurement rather than intuition.
    try:
        report = load_fairness_report()
        top_proxies = [
            r["feature"]
            for r in report["proxy"]["ranked"]
            if r["in_model"] and r["status"] == "ok"
        ][:2]
    except FileNotFoundError:
        top_proxies = []

    grid = build_grid(model.feature_names, top_proxies, quick=quick)

    lda_dir = cfg.paths.artifact("lda")
    lda_dir.mkdir(parents=True, exist_ok=True)
    (lda_dir / "grid.json").write_text(json.dumps(grid.as_dict(), indent=2), encoding="utf-8")

    results_path = lda_dir / "results.jsonl"
    if not resume and results_path.exists():
        results_path.unlink()

    fingerprint = model.training_fingerprint.get("row_id_hash", "")
    cache = _load_cache(results_path)

    train_groups = assign_groups(train)["race"]
    base_rates = (
        train.assign(_g=train_groups)
        .dropna(subset=["_g"])
        .groupby("_g")["approved"]
        .mean()
        .to_dict()
    )

    results: list[ChallengerResult] = []
    n_cached = n_new = 0
    total = len(grid.challengers)

    for i, challenger in enumerate(grid.challengers, start=1):
        key = challenger.cache_key(fingerprint)
        if key in cache:
            results.append(ChallengerResult(**_from_cache(cache[key])))
            n_cached += 1
            continue

        result = _evaluate(
            challenger, train, valid, eval_frame, groups, model, base_rates, key, controls
        )
        with results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result.as_dict()) + "\n")
        results.append(result)
        n_new += 1
        print(
            f"    [{i:>3}/{total}] {challenger.id} {challenger.subset:<28} "
            f"auc={result.auc:.4f} ame={result.disparity_ame:+.2f}pp "
            f"({result.seconds:.1f}s)",
            flush=True,
        )

    incumbent = _incumbent_result(model, eval_frame, groups, controls)
    all_results = [incumbent, *results]

    points = [
        Point(id=r.id, auc=r.auc, disparity=r.disparity_ame)
        for r in all_results
        if r.disparity_status == "ok"
    ]
    front = frontier(points)
    front_ids = {p.id for p in front}
    story = headline(points, INCUMBENT_ID, float(cfg.lda["auc_tolerance"]))

    frame = pd.DataFrame([r.as_dict() for r in all_results])
    frame["on_frontier"] = frame["id"].isin(front_ids)
    frame["is_incumbent"] = frame["id"] == INCUMBENT_ID
    frame = frame.sort_values(["on_frontier", "auc"], ascending=[False, False])

    csv_path = lda_dir / "lda_results.csv"
    frame.drop(columns=["features"]).to_csv(csv_path, index=False)

    from tally.lda.plot import plot_frontier

    plot_path = plot_frontier(all_results, front_ids, INCUMBENT_ID, story.text)

    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "model_version": model.version,
        "model_hash": model.model_hash,
        "objective": _objective_provenance(),
        "quick": quick,
        "n_challengers": total,
        "n_evaluated": n_new,
        "n_cached": n_cached,
        "incumbent": incumbent.as_dict(),
        "frontier": [
            next(r.as_dict() for r in all_results if r.id == p.id) for p in front
        ],
        "headline": story.text,
        "headline_challenger": story.challenger_id,
        "auc_delta": story.auc_delta,
        "disparity_reduction": story.disparity_reduction,
        "results": [r.as_dict() for r in all_results],
        "grid_axes": grid.axes,
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
        "doctrine_note": (
            "Under disparate impact doctrine a business-justified model remains "
            "challengeable if a comparably effective, less discriminatory alternative "
            "exists. This search covers a bounded hypothesis space; the absence of a "
            "better alternative within it is not proof that none exists."
        ),
    }

    (lda_dir / "lda_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    return summary


def _objective_provenance() -> dict[str, Any]:
    """Record what the search steered by, and whether that target was real.

    A frontier is only meaningful if the quantity on its axis is distinguishable
    from zero for the incumbent. Stating that explicitly stops the search from
    looking authoritative when it was chasing noise.
    """
    cfg = get_config()
    focal = cfg.fairness["primary_focal_group"]
    out: dict[str, Any] = {
        "focal_group": focal,
        "metric": (
            "Average marginal effect on P(approve) in percentage points, from a "
            "logistic regression of the model's decision on protected class plus the "
            "incumbent's credit factors, without geographic fixed effects. The control "
            "set is held fixed across every challenger so that the axis is a single "
            "quantity; controlling each challenger for its own features would reward "
            "a model whose proxy absorbs group membership among the controls."
        ),
    }
    try:
        from tally.fairness.report import load_fairness_report

        entry = (
            load_fairness_report()["controlled"]["specifications"]["race"]["primary"][
                "groups"
            ].get(focal, {})
        )
        out["incumbent_ame_pp"] = entry.get("ame_pp")
        out["incumbent_p_value"] = entry.get("p_value")
        out["incumbent_n"] = entry.get("n_focal")
        p = entry.get("p_value")
        out["target_is_significant"] = bool(p is not None and p < 0.05)
        if not out["target_is_significant"]:
            out["warning"] = (
                "The incumbent's disparity for this group is not statistically "
                "distinguishable from zero, so differences along the frontier may "
                "reflect sampling variation rather than real improvement."
            )
    except (FileNotFoundError, KeyError):
        out["incumbent_ame_pp"] = None
    return out


def _from_cache(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "subset": row["subset"],
        "origin": row["origin"],
        "features": row["features"],
        "params": row["params"],
        "auc": row["auc"],
        "ks": row["ks"],
        "approval_rate": row["approval_rate"],
        "disparity_ame": row["disparity_ame"],
        "disparity_status": row["disparity_status"],
        "cache_key": row["cache_key"],
        "seconds": row["seconds"],
    }


def load_lda_summary() -> dict[str, Any]:
    cfg = get_config()
    path = cfg.paths.artifact("lda", "lda_summary.json")
    if not path.exists():
        raise FileNotFoundError("lda_summary.json missing — run: uv run tally lda")
    return json.loads(path.read_text(encoding="utf-8"))
