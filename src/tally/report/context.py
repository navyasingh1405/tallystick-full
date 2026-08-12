"""Assemble the render context from artifacts.

Every value the template can reach comes from a file on disk that some other command
produced. Nothing is computed here and nothing is typed here. If a number is wrong in
the PDF, it is wrong in an artifact, and that artifact has a command that regenerates
it — which is the property that makes the document trustworthy rather than decorative.

Prose in the pack is limited to explaining method and interpretation. It never states
a result; results are interpolated.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tally.config import get_config
from tally.data.build import load_lineage
from tally.fairness.report import load_fairness_report
from tally.model.metrics import load_metric_artifact
from tally.model.registry import load_current
from tally.report.sections import SECTIONS, frameworks, regulatory_appendix


class MissingArtifact(RuntimeError):
    """A required artifact has not been generated."""


def _b64_png(path: Path) -> str | None:
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _read_markdown(path: Path) -> list[dict[str, Any]]:
    """Parse `docs/limitations.md` into headed blocks.

    Section 10 is pulled verbatim from that file rather than restated, so the
    limitations a reader sees in the PDF are the same text the repository maintains.
    """
    if not path.exists():
        raise MissingArtifact(f"{path.name} is missing; section 10 is sourced from it")

    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current = {"heading": stripped[3:].strip(), "paragraphs": []}
            blocks.append(current)
        elif stripped.startswith("# "):
            continue
        elif stripped and current is not None:
            if current["paragraphs"] and not stripped.startswith("-"):
                current["paragraphs"][-1] += " " + stripped
            else:
                current["paragraphs"].append(stripped)
        elif not stripped and current is not None and current["paragraphs"]:
            current["paragraphs"].append("")
    for block in blocks:
        block["paragraphs"] = [p.strip() for p in block["paragraphs"] if p.strip()]
    return blocks


def build_context() -> dict[str, Any]:
    cfg = get_config()
    model = load_current()

    try:
        lineage = load_lineage()
        performance = load_metric_artifact("performance")
        calibration = load_metric_artifact("calibration")
        segments = load_metric_artifact("segments")
        fairness = load_fairness_report()
    except FileNotFoundError as exc:
        raise MissingArtifact(str(exc)) from exc

    from tally.lda.search import load_lda_summary

    try:
        lda = load_lda_summary()
    except FileNotFoundError as exc:
        raise MissingArtifact(str(exc)) from exc

    try:
        from tally.drift.report import load_drift_report

        drift = load_drift_report()
    except FileNotFoundError:
        drift = None

    generated_at = datetime.now(UTC)

    return {
        "generated_at": generated_at.strftime("%d %B %Y at %H:%M UTC"),
        "generated_at_iso": generated_at.isoformat(),
        "model": {
            "version": model.version,
            "hash": model.model_hash,
            "hash_short": model.model_hash[:16],
            "created_at": model.manifest["created_at"],
            "threshold": model.threshold,
            "feature_count": len(model.feature_names),
            "training": model.training_fingerprint,
            "params": model.manifest["params"],
            "metrics": model.metrics,
            "imputation": model.manifest.get("imputation", {}),
            "feature_specs": model.manifest.get("feature_specs", []),
            "excluded_features": model.manifest.get("excluded_features", []),
            "protected_attributes": model.manifest.get("protected_attributes", []),
            "background_hash": model.manifest.get("background_hash", "")[:16],
        },
        "config": {
            "state": cfg.data["state"],
            "train_year": cfg.data["train_year"],
            "oot_year": cfg.data["oot_year"],
            "filters": cfg.data["filters"],
            "min_group_n": cfg.fairness["min_group_n"],
            "nmi_threshold": cfg.fairness["nmi_flag_threshold"],
            "primary_focal_group": cfg.fairness["primary_focal_group"],
        },
        "issuer": cfg.report,
        "sections": SECTIONS,
        "lineage": lineage,
        "performance": performance,
        "calibration": calibration,
        "segments": segments,
        "fairness": fairness,
        "lda": lda,
        "drift": drift,
        "limitations": _read_markdown(cfg.paths.docs / "limitations.md"),
        "monitoring": _read_markdown(cfg.paths.docs / "monitoring.md"),
        "regulatory": regulatory_appendix(),
        "frameworks": frameworks(),
        "frontier_png": _b64_png(cfg.paths.artifact("lda", "lda_frontier.png")),
        "sample_notice": None,  # filled by render when a declined example is available
    }
