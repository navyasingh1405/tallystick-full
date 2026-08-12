"""Train the incumbent model.

The interesting parts are the constraints and the determinism, not the accuracy.
`deterministic`, `force_row_wise`, and a fixed `num_threads` are all required for
two runs on the same data to produce byte-identical model files: LightGBM's
histogram construction order otherwise depends on how work is split across threads.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from tally.config import get_config
from tally.data.features import FitParams, feature_matrix, fit_params, missingness_report
from tally.data.splits import load_split
from tally.model.registry import LoadedModel, ModelRegistry, data_fingerprint


def build_training_params(cfg_model: dict[str, Any], monotone: list[int]) -> dict[str, Any]:
    params = dict(cfg_model["params"])
    params.update(
        {
            "monotone_constraints": monotone,
            "seed": cfg_model["seed"],
            "bagging_seed": cfg_model["seed"],
            "feature_fraction_seed": cfg_model["seed"],
            "data_random_seed": cfg_model["seed"],
            "num_threads": cfg_model["num_threads"],
        }
    )
    return params


def train_booster(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    params_fitted: FitParams,
    *,
    feature_names: list[str],
    monotone: list[int],
    sample_weight: np.ndarray | None = None,
) -> lgb.Booster:
    """Fit one booster. Shared by the incumbent and every LDA challenger."""
    cfg = get_config()
    m = cfg.model

    Xtr = feature_matrix(train, params_fitted)[feature_names]
    Xva = feature_matrix(valid, params_fitted)[feature_names]
    _assert_no_missing_at_training(Xtr)
    ytr = train["approved"].astype(int).to_numpy()
    yva = valid["approved"].astype(int).to_numpy()

    dtrain = lgb.Dataset(Xtr, label=ytr, weight=sample_weight, free_raw_data=False)
    dvalid = lgb.Dataset(Xva, label=yva, reference=dtrain, free_raw_data=False)

    params = build_training_params(m, monotone)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=int(m["num_boost_round"]),
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(int(m["early_stopping_rounds"]), verbose=False)],
    )


def _assert_no_missing_at_training(X: pd.DataFrame) -> None:
    """No feature may reach LightGBM with missing values.

    A missing value in *any* feature creates missing-bin splits, and rows routed
    through them escape monotone-constraint enforcement on the *constrained*
    features as well. The model would then claim constraints it does not honour,
    which also invalidates the counterfactual binary search. Fail loudly instead of
    shipping that. See features._impute_missing for the measurements.
    """
    offenders = sorted(c for c in X.columns if X[c].isna().any())
    if offenders:
        raise ValueError(
            f"feature(s) {offenders} reach training with missing values. LightGBM "
            "does not enforce monotone constraints for rows routed through a "
            "missing-bin split, so the model would violate its own stated "
            "constraints. Ensure FitParams.feature_medians covers these "
            "(see features._impute_missing)."
        )


def train_and_register() -> LoadedModel:
    from tally.explain.shap_engine import threshold_background
    from tally.model.metrics import basic_metrics

    cfg = get_config()
    train = load_split("train")
    test = load_split("test")
    oot = load_split("oot")

    feature_names = cfg.feature_names
    monotone = cfg.monotone_constraints
    fitted = fit_params(train)

    booster = train_booster(
        train, test, fitted, feature_names=feature_names, monotone=monotone
    )

    Xtr = feature_matrix(train, fitted)[feature_names]
    metrics: dict[str, Any] = {}
    for name, frame in (("train", train), ("test", test), ("oot", oot)):
        X = feature_matrix(frame, fitted)[feature_names]
        p = 1.0 / (1.0 + np.exp(-booster.predict(X, raw_score=True)))
        y = frame["approved"].astype(int).to_numpy()
        for k, v in basic_metrics(y, p, cfg.threshold).items():
            metrics[f"{name}_{k}"] = v

    background = threshold_background(
        booster,
        Xtr,
        threshold=cfg.threshold,
        n=int(cfg.explain["shap_background_n"]),
        seed=int(cfg.explain["shap_background_seed"]),
    )

    reg = ModelRegistry()
    return reg.save(
        booster,
        feature_names=feature_names,
        monotone_constraints=monotone,
        threshold=cfg.threshold,
        fit_params=fitted,
        training_data={
            **data_fingerprint(train["row_id"]),
            "state": cfg.data["state"],
            "activity_year": int(train["activity_year"].iloc[0]),
            "approval_rate": round(float(train["approved"].mean()), 6),
        },
        params=build_training_params(cfg.model, monotone),
        metrics={
            **metrics,
            "best_iteration": int(booster.best_iteration or booster.current_iteration()),
        },
        background=background,
        extra={
            "feature_specs": [
                {
                    "name": f.name,
                    "display_name": f.display_name,
                    "source": f.source,
                    "transform": f.transform,
                    "unit": f.unit,
                    "monotone": f.monotone,
                    "rationale": " ".join(f.rationale.split()),
                }
                for f in cfg.features
            ],
            "excluded_features": cfg.excluded,
            "protected_attributes": list(cfg.protected),
            "imputation": {
                "method": "training median",
                "applied_to": "every model feature",
                "rationale": (
                    "LightGBM does not enforce monotone constraints for rows routed "
                    "through a missing-bin split, and the failure is not confined to "
                    "the feature that is missing. Eliminating missing values at "
                    "training is what makes the stated constraints real, and is a "
                    "precondition for the counterfactual binary search."
                ),
                "share_imputed": missingness_report(train, fitted),
                "values": fitted.feature_medians,
            },
        },
    )
