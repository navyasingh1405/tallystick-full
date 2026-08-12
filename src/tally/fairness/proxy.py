"""5c — proxy detection.

Excluding race from a model does not remove it if another feature carries the same
information. This scan measures how much each feature reveals about each protected
attribute.

Three choices that make the output trustworthy
----------------------------------------------
**Normalised mutual information, not raw MI.** Raw MI is unbounded and its scale
depends on the entropy of the variables involved, so a single threshold cannot mean
the same thing across pairs. NMI is bounded in [0, 1] and comparable.

**A negative control in every scan.** A seeded random column is scored alongside the
real features. It establishes the detector's noise floor, and a threshold that would
flag it is a threshold that means nothing.

**Set-level predictability alongside the pairwise scan.** Pairwise MI cannot see
joint proxying: several features, each individually uninformative, can identify a
group in combination. Fitting a small model to predict group membership from the
feature set catches that. "The feature set predicts applicant race with AUC 0.71" is
also a far more legible finding than any mutual information value.

`tract_minority_population_percent` is scanned even though it is excluded from the
model. Demonstrating that the detector fires on a known proxy is the point.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from tally.config import get_config
from tally.fairness.groups import ATTRIBUTES, assign_groups

N_BINS = 20
NEGATIVE_CONTROL = "_random_negative_control"
SEED = 20260810


def _binned(values: pd.Series) -> np.ndarray:
    """Quantile-bin a numeric column so MI is computed on comparable supports."""
    v = pd.to_numeric(values, errors="coerce")
    if v.notna().sum() == 0:
        return np.zeros(len(v), dtype=int)
    if v.nunique(dropna=True) <= N_BINS:
        codes = pd.Categorical(v.fillna(v.median())).codes
        return np.asarray(codes, dtype=int)
    try:
        binned = pd.qcut(v.rank(method="first"), N_BINS, labels=False, duplicates="drop")
    except ValueError:  # pragma: no cover - degenerate distribution
        binned = pd.cut(v, N_BINS, labels=False, duplicates="drop")
    return np.asarray(pd.Series(binned).fillna(-1), dtype=int)


def proxy_scan(
    df: pd.DataFrame,
    *,
    features: list[str] | None = None,
    groups: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Normalised mutual information between every feature and every protected attribute."""
    cfg = get_config()
    threshold = float(cfg.fairness["nmi_flag_threshold"])
    features = list(features or cfg.feature_names)

    groups = groups if groups is not None else assign_groups(df)

    scan = df.copy()
    rng = np.random.default_rng(SEED)
    scan[NEGATIVE_CONTROL] = rng.random(len(scan))
    columns = [*features, NEGATIVE_CONTROL]

    rows: list[dict[str, Any]] = []
    for col in columns:
        if col not in scan.columns:
            continue
        binned = _binned(scan[col])
        is_constant = len(np.unique(binned)) < 2
        entry: dict[str, Any] = {
            "feature": col,
            "is_negative_control": col == NEGATIVE_CONTROL,
            "in_model": col in cfg.feature_names,
            "status": "constant" if is_constant else "ok",
            "nmi": {},
        }
        for attr in ATTRIBUTES:
            g = groups[attr]
            mask = g.notna().to_numpy()
            labels = g[mask].astype(str).to_numpy()

            # Guard rail, and not a theoretical one: sklearn's
            # normalized_mutual_info_score returns 1.0 when BOTH labellings have a
            # single distinct value, because it treats two trivial partitions as
            # perfectly agreeing. Left unguarded, every constant feature ranks as a
            # perfect proxy. A constant feature carries no information about
            # anything, so the honest answer is zero; a constant protected attribute
            # makes the comparison undefined, so the honest answer is nothing.
            if mask.sum() < 100 or len(np.unique(labels)) < 2:
                entry["nmi"][attr] = None
                continue
            if is_constant:
                entry["nmi"][attr] = 0.0
                continue

            entry["nmi"][attr] = float(
                normalized_mutual_info_score(
                    labels, binned[mask], average_method="arithmetic"
                )
            )
        scores = [v for v in entry["nmi"].values() if v is not None]
        entry["max_nmi"] = max(scores) if scores else 0.0
        entry["max_attribute"] = (
            max(
                (a for a in ATTRIBUTES if entry["nmi"].get(a) is not None),
                key=lambda a: entry["nmi"][a],
            )
            if scores
            else None
        )
        entry["flagged"] = bool(entry["max_nmi"] >= threshold and col != NEGATIVE_CONTROL)
        rows.append(entry)

    ranked = sorted(
        (r for r in rows if not r["is_negative_control"]),
        key=lambda r: (-r["max_nmi"], r["feature"]),
    )
    control = next(r for r in rows if r["is_negative_control"])

    # How far above the noise floor each feature sits. At large n every real feature
    # has *some* measurable dependence on group, so the absolute value alone is hard
    # to read; the multiple of the negative control is what makes it interpretable.
    floor = max(control["max_nmi"], 1e-9)
    for entry in [*ranked, control]:
        entry["control_multiple"] = round(entry["max_nmi"] / floor, 1)
    for rank, entry in enumerate(ranked, start=1):
        entry["rank"] = rank

    return {
        "threshold": threshold,
        "n_bins": N_BINS,
        "ranked": ranked,
        "negative_control": control,
        "n_flagged": sum(1 for r in ranked if r["flagged"]),
        "n_constant": sum(1 for r in ranked if r["status"] == "constant"),
        "set_level": set_level_predictability(scan, features, groups),
        "method": (
            "Normalised mutual information between each feature, quantile-binned into "
            f"{N_BINS} bins, and each protected attribute. A seeded random column is "
            "scanned alongside as a negative control to establish the noise floor."
        ),
        "threshold_note": (
            f"The flag threshold ({threshold}) is a calibrated heuristic, not a "
            "natural constant: NMI magnitudes depend on sample size and binning. It "
            "is set so that a known geographic proxy is flagged while ordinary credit "
            "features are not. The negative control scores "
            f"{control['max_nmi']:.4f}, which is the floor this scan can resolve; "
            "each feature's multiple of that floor is reported so the calibration can "
            "be judged directly rather than taken on trust."
        ),
        "caveat": (
            "A high value indicates the feature carries information about group "
            "membership. It does not by itself establish that the model uses the "
            "feature as a proxy — that is what the controlled analysis and the "
            "less discriminatory alternative search address."
        ),
    }


def set_level_predictability(
    df: pd.DataFrame, features: list[str], groups: pd.DataFrame
) -> dict[str, Any]:
    """How well the feature set as a whole predicts group membership.

    Pairwise mutual information misses joint proxying entirely. This does not.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    usable = [f for f in features if f in df.columns]
    g = groups["race"]
    mask = g.notna().to_numpy()
    if mask.sum() < 500 or not usable:
        return {"status": "insufficient_sample", "macro_auc": None}

    X = df.loc[mask, usable].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y_raw = g[mask].astype(str)

    # Keep classes with enough support for a stratified split.
    counts = y_raw.value_counts()
    keep = y_raw.isin(counts[counts >= 50].index)
    X, y_raw = X[keep.to_numpy()], y_raw[keep.to_numpy()]
    classes = sorted(y_raw.unique())
    if len(classes) < 2:
        return {"status": "insufficient_sample", "macro_auc": None}

    codes = pd.Categorical(y_raw, categories=classes).codes
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, codes, test_size=0.3, random_state=SEED, stratify=codes
    )

    params = dict(
        objective="multiclass" if len(classes) > 2 else "binary",
        num_leaves=15,
        min_child_samples=50,
        learning_rate=0.1,
        verbose=-1,
        deterministic=True,
        force_row_wise=True,
        num_threads=4,
        seed=SEED,
    )
    if len(classes) > 2:
        params["num_class"] = len(classes)

    booster = lgb.train(params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=120)
    pred = booster.predict(X_te)

    try:
        if len(classes) > 2:
            auc = float(roc_auc_score(y_te, pred, multi_class="ovr", average="macro"))
        else:
            auc = float(roc_auc_score(y_te, pred))
    except ValueError:  # pragma: no cover - degenerate split
        return {"status": "not_estimable", "macro_auc": None}

    return {
        "status": "ok",
        "macro_auc": auc,
        "classes": classes,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "interpretation": (
            f"The model's feature set predicts an applicant's race and ethnicity "
            f"group with macro-averaged AUC {auc:.3f}. An AUC of 0.5 would mean the "
            "features carry no information about group membership."
        ),
        "note": (
            "This captures joint proxying — combinations of features that together "
            "identify a group even when no single feature does — which the pairwise "
            "scan cannot see."
        ),
    }
