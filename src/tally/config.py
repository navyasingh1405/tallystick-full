"""Configuration loading and path resolution.

Everything that affects a model or a metric lives in `config/*.yaml` so that the
validation pack can report it without anything being hand-typed.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Resolve the repository root.

    Honours TALLY_ROOT so tests and the packaged CLI can point elsewhere; otherwise
    walks up from this file until it finds the config directory.
    """
    env = os.environ.get("TALLY_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "config.yaml").exists():
            return parent
    # Installed without the repo alongside: fall back to cwd.
    return Path.cwd()


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def ledger_db(self) -> Path:
        return self.root / "ledger" / "decisions.db"

    @property
    def web(self) -> Path:
        return self.root / "web"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    def artifact(self, *parts: str) -> Path:
        return self.artifacts.joinpath(*parts)

    def ensure(self) -> None:
        for p in (
            self.data_raw,
            self.data_interim,
            self.data_processed,
            self.models,
            self.artifacts,
            self.artifact("metrics"),
            self.artifact("fairness"),
            self.artifact("lda"),
            self.artifact("lda", "cache"),
            self.artifact("drift"),
            self.artifact("adoption"),
            self.artifact("batches"),
            self.artifact("packs"),
            self.ledger_db.parent,
        ):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class FeatureSpec:
    """One model feature, with the constraint rationale that flows into the pack."""

    name: str
    display_name: str
    source: str
    transform: str
    unit: str
    monotone: int
    rationale: str


class Config:
    """Loaded configuration. Read-only by convention; never mutate at runtime."""

    def __init__(self, root: Path | None = None) -> None:
        self.paths = Paths(root or project_root())
        self._raw = _read_yaml(self.paths.config / "config.yaml")
        self._features_raw = _read_yaml(self.paths.config / "features.yaml")
        self.regulatory_map = _read_yaml(self.paths.config / "regulatory_map.yaml")

        self.features: list[FeatureSpec] = [
            FeatureSpec(**_require_feature_fields(f)) for f in self._features_raw["features"]
        ]
        _apply_env_overrides(self._raw)
        self.excluded: list[dict[str, str]] = self._features_raw.get("excluded", [])
        self.protected: tuple[str, ...] = tuple(self._features_raw["protected"])

    # -- section accessors ------------------------------------------------
    @property
    def data(self) -> dict[str, Any]:
        return self._raw["data"]

    @property
    def split(self) -> dict[str, Any]:
        return self._raw["split"]

    @property
    def model(self) -> dict[str, Any]:
        return self._raw["model"]

    @property
    def explain(self) -> dict[str, Any]:
        return self._raw["explain"]

    @property
    def fairness(self) -> dict[str, Any]:
        return self._raw["fairness"]

    @property
    def lda(self) -> dict[str, Any]:
        return self._raw["lda"]

    @property
    def drift(self) -> dict[str, Any]:
        return self._raw["drift"]

    @property
    def service(self) -> dict[str, Any]:
        return self._raw["service"]

    @property
    def report(self) -> dict[str, Any]:
        return self._raw["report"]

    @property
    def auth(self) -> dict[str, Any]:
        return self._raw["auth"]

    # -- derived ----------------------------------------------------------
    @property
    def feature_names(self) -> list[str]:
        """Ordered feature list. Order is part of the model contract."""
        return [f.name for f in self.features]

    @property
    def monotone_constraints(self) -> list[int]:
        return [f.monotone for f in self.features]

    def feature(self, name: str) -> FeatureSpec:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(f"unknown feature: {name}")

    @property
    def threshold(self) -> float:
        return float(self.model["threshold"])

    def as_dict(self) -> dict[str, Any]:
        """Full config for the validation pack. Nothing is hidden from the document."""
        return dict(self._raw)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing config file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


_FEATURE_FIELDS = {"name", "display_name", "source", "transform", "unit", "monotone", "rationale"}


def _require_feature_fields(entry: dict[str, Any]) -> dict[str, Any]:
    missing = _FEATURE_FIELDS - set(entry)
    if missing:
        raise ValueError(
            f"feature {entry.get('name', '<unnamed>')} in features.yaml is missing: "
            f"{sorted(missing)}. Every feature needs a written constraint rationale — "
            "it is rendered verbatim in validation pack section 3."
        )
    extra = set(entry) - _FEATURE_FIELDS
    if extra:
        raise ValueError(f"feature {entry['name']} has unknown fields: {sorted(extra)}")
    return entry


@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide config singleton."""
    cfg = Config()
    return cfg


def reset_config_cache() -> None:
    """Test hook — drops the singleton so TALLY_ROOT changes take effect."""
    get_config.cache_clear()


#: Settings that must be changeable at deploy time without editing a file baked into
#: a container image. Everything else stays in YAML, where it is reviewable and
#: version controlled — these are the handful that legitimately differ per
#: environment.
_ENV_OVERRIDES: dict[str, tuple[str, str, type]] = {
    "TALLY_HOST": ("service", "host", str),
    "TALLY_PORT": ("service", "port", int),
    "TALLY_ALLOWED_ORIGINS": ("service", "allowed_origins", list),
    "TALLY_COOKIE_SECURE": ("auth", "cookie_secure", bool),
    "TALLY_ALLOW_INSTANT_DEMO": ("auth", "allow_instant_demo", bool),
    "TALLY_SESSION_HOURS": ("auth", "session_hours", int),
}


def _coerce(value: str, kind: type) -> Any:
    if kind is bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if kind is int:
        return int(value)
    if kind is list:
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


def _apply_env_overrides(raw: dict[str, Any]) -> None:
    for env_name, (section, key, kind) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if value is None:
            continue
        raw.setdefault(section, {})[key] = _coerce(value, kind)
