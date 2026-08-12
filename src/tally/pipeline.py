"""End-to-end build orchestration.

`tally demo` calls `ensure_artifacts` so a clean clone can go straight to the demo
without the operator having to know which command produces which file.
"""

from __future__ import annotations

import time

from tally.config import get_config


def _rule(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 66 - len(title)))


def missing_artifacts() -> list[str]:
    cfg = get_config()
    p = cfg.paths
    checks = [
        ("data", (p.data_processed / "oot.parquet").exists()),
        ("model", (p.models / "CURRENT").exists()),
        ("evaluate", p.artifact("metrics", "performance.json").exists()),
        ("fairness", p.artifact("fairness", "fairness_report.json").exists()),
        ("lda", p.artifact("lda", "lda_summary.json").exists()),
        ("drift", p.artifact("drift", "drift_report.json").exists()),
    ]
    return [name for name, present in checks if not present]


def build_everything(*, quick: bool = False, only: list[str] | None = None) -> None:
    from tally.data.build import build_all
    from tally.drift.report import build_drift_report
    from tally.fairness.report import build_fairness_report
    from tally.lda.search import run_search
    from tally.model.metrics import write_evaluation_artifacts
    from tally.model.train import train_and_register

    steps = {
        "data": ("Downloading and building data", lambda: build_all()),
        "model": ("Training the model", train_and_register),
        "evaluate": ("Measuring performance", write_evaluation_artifacts),
        "fairness": ("Running fair lending analysis", build_fairness_report),
        "drift": ("Measuring drift", build_drift_report),
        "lda": (
            "Searching for a less discriminatory alternative"
            + (" (quick grid)" if quick else " (full grid, this is the slow one)"),
            lambda: run_search(quick=quick),
        ),
    }
    order = ["data", "model", "evaluate", "fairness", "drift", "lda"]
    todo = [s for s in order if only is None or s in only]

    for name in todo:
        label, fn = steps[name]
        _rule(label)
        started = time.perf_counter()
        fn()
        print(f"  {name} finished in {time.perf_counter() - started:.1f}s")


def ensure_artifacts(*, quick: bool = True) -> None:
    """Build only what is missing, and say how long it will take before starting."""
    missing = missing_artifacts()
    if not missing:
        return

    slow = "lda" in missing or "data" in missing
    _rule("first run")
    print(f"  Missing artifacts: {', '.join(missing)}")
    if slow:
        print(
            "  Building them now. The alternative search is the slow step; the "
            f"{'reduced' if quick else 'full'} grid takes "
            f"{'about 4 minutes' if quick else 'about 20 minutes'}."
        )
    print("  Subsequent runs start instantly.\n")
    build_everything(quick=quick, only=missing)
