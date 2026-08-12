"""Model registry: versioned, hashed, immutable once written.

A decision recorded in the ledger names a model version and a model hash. Replaying
that decision means loading exactly those bytes back. Everything needed to do that
lives in `models/{version}/`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from tally.config import get_config
from tally.data.features import FitParams

CURRENT_POINTER = "CURRENT"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def data_fingerprint(row_ids: pd.Series) -> dict[str, Any]:
    """Row count plus a hash of the sorted row ids.

    Identifies the exact training sample without storing it, so a reviewer can tell
    whether two models saw the same data.
    """
    ids = sorted(str(v) for v in row_ids)
    h = hashlib.sha256()
    for i in ids:
        h.update(i.encode("utf-8"))
        h.update(b"\n")
    return {"row_count": len(ids), "row_id_hash": h.hexdigest()}


@dataclass(frozen=True)
class LoadedModel:
    """Everything pinned to a model version. Treat as immutable."""

    version: str
    directory: Path
    booster: lgb.Booster
    feature_names: list[str]
    monotone_constraints: list[int]
    threshold: float
    model_hash: str
    fit_params: FitParams
    manifest: dict[str, Any]

    @property
    def metrics(self) -> dict[str, Any]:
        return self.manifest.get("metrics", {})

    @property
    def training_fingerprint(self) -> dict[str, Any]:
        return self.manifest.get("training_data", {})

    @property
    def monotone(self) -> dict[str, int]:
        return dict(zip(self.feature_names, self.monotone_constraints, strict=True))

    @property
    def background_path(self) -> Path:
        return self.directory / "shap_background.parquet"

    def raw_score(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Margin (log-odds) score. All attribution happens in this space."""
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return np.asarray(self.booster.predict(arr, raw_score=True), dtype=float)

    def probability(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return _sigmoid(self.raw_score(X))

    def decide(self, prob: float) -> str:
        return "APPROVED" if prob > self.threshold else "DECLINED"

    @property
    def raw_threshold(self) -> float:
        """logit(threshold). Since sigma is strictly monotone, deciding on
        probability > t is exactly deciding on raw margin > logit(t)."""
        t = self.threshold
        return float(np.log(t / (1.0 - t)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class ModelRegistry:
    def __init__(self, root: Path | None = None) -> None:
        cfg = get_config()
        self.root = root or cfg.paths.models
        self.root.mkdir(parents=True, exist_ok=True)

    # -- writing ----------------------------------------------------------
    def next_version(self) -> str:
        existing = [p.name for p in self.root.iterdir() if p.is_dir() and p.name.startswith("v")]
        n = 0
        for name in existing:
            try:
                n = max(n, int(name[1:].split("-")[0]))
            except ValueError:
                continue
        return f"v{n + 1:03d}"

    def save(
        self,
        booster: lgb.Booster,
        *,
        feature_names: list[str],
        monotone_constraints: list[int],
        threshold: float,
        fit_params: FitParams,
        training_data: dict[str, Any],
        params: dict[str, Any],
        metrics: dict[str, Any],
        background: pd.DataFrame,
        extra: dict[str, Any] | None = None,
    ) -> LoadedModel:
        version = self.next_version()
        d = self.root / version
        d.mkdir(parents=True, exist_ok=False)

        model_file = d / "booster.txt"
        booster.save_model(str(model_file))
        model_hash = sha256_file(model_file)

        bg_path = d / "shap_background.parquet"
        background.to_parquet(bg_path, index=False)

        manifest: dict[str, Any] = {
            "version": version,
            "created_at": datetime.now(UTC).isoformat(),
            "model_file": model_file.name,
            "model_hash": model_hash,
            "background_file": bg_path.name,
            "background_hash": sha256_file(bg_path),
            "feature_names": feature_names,
            "monotone_constraints": monotone_constraints,
            "threshold": threshold,
            "fit_params": fit_params.to_json(),
            "training_data": training_data,
            "params": params,
            "metrics": metrics,
        }
        if extra:
            manifest.update(extra)

        (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (self.root / CURRENT_POINTER).write_text(version, encoding="utf-8")

        return self.load(version)

    # -- reading ----------------------------------------------------------
    def load(self, version: str) -> LoadedModel:
        d = self.root / version
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no model version {version!r} in {self.root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        model_file = d / manifest["model_file"]
        actual = sha256_file(model_file)
        if actual != manifest["model_hash"]:
            raise ValueError(
                f"model {version} hash mismatch: manifest says {manifest['model_hash']}, "
                f"file is {actual}. The pinned artifact has been altered."
            )

        booster = lgb.Booster(model_file=str(model_file))
        return LoadedModel(
            version=version,
            directory=d,
            booster=booster,
            feature_names=list(manifest["feature_names"]),
            monotone_constraints=list(manifest["monotone_constraints"]),
            threshold=float(manifest["threshold"]),
            model_hash=manifest["model_hash"],
            fit_params=FitParams.from_json(manifest["fit_params"]),
            manifest=manifest,
        )

    def current_version(self) -> str:
        pointer = self.root / CURRENT_POINTER
        if not pointer.exists():
            raise FileNotFoundError("no CURRENT model — run: uv run tally train")
        return pointer.read_text(encoding="utf-8").strip()

    def current(self) -> LoadedModel:
        return self.load(self.current_version())

    def versions(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and (p / "manifest.json").exists())


_cache: dict[str, LoadedModel] = {}


def load_current() -> LoadedModel:
    """Process-cached current model. The registry is immutable, so caching is safe."""
    reg = ModelRegistry()
    version = reg.current_version()
    if version not in _cache:
        _cache[version] = reg.load(version)
    return _cache[version]


def load_version(version: str) -> LoadedModel:
    if version not in _cache:
        _cache[version] = ModelRegistry().load(version)
    return _cache[version]


def clear_cache() -> None:
    _cache.clear()
