"""`cal` — the single task runner.

The Makefile and make.ps1 are three-line delegators to this. Command bodies import
lazily so that `tally doctor` still runs when a heavy dependency is broken.
"""

from __future__ import annotations

import sys
import time
from typing import Annotated

import typer

# The Windows console defaults to cp1252, which mangles the em dashes and box
# characters used throughout the reports. Reconfigure rather than avoid them.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - non-tty streams
            pass

app = typer.Typer(
    name="cal",
    help="Tallystick — governance around a credit model.",
    no_args_is_help=True,
    add_completion=False,
)


def _register_user_commands() -> None:
    from tally.auth.cli import app as user_app

    app.add_typer(user_app, name="user", help="Manage accounts.")


_register_user_commands()


def _rule(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 66 - len(title)))


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"


# --------------------------------------------------------------------------- setup


@app.command()
def doctor() -> None:
    """Check the environment before anything expensive runs."""
    from tally.doctor import report, run_checks

    _rule("environment")
    ok = report(run_checks())
    raise typer.Exit(0 if ok else 1)


@app.command()
def setup() -> None:
    """Install dependencies and the Chromium build used for PDF rendering."""
    import subprocess

    _rule("uv sync")
    subprocess.run(["uv", "sync", "--extra", "dev"], check=True)
    _rule("playwright install chromium")
    subprocess.run(["uv", "run", "playwright", "install", "chromium"], check=True)
    _rule("environment")
    from tally.doctor import report, run_checks

    ok = report(run_checks())
    raise typer.Exit(0 if ok else 1)


# --------------------------------------------------------------------------- data


@app.command()
def data(
    force: Annotated[bool, typer.Option(help="Re-download even if raw CSVs exist.")] = False,
) -> None:
    """Download HMDA data and build the modelling frames."""
    from tally.data.build import build_all

    start = time.perf_counter()
    lineage = build_all(force=force)
    _rule("data lineage")
    for stage in lineage["stages"]:
        print(f"  {stage['name']:<34} {stage['rows']:>9,}   {stage['note']}")
    print(f"\n  done in {_elapsed(start)}")


# --------------------------------------------------------------------------- model


@app.command()
def train() -> None:
    """Train the incumbent model and register it."""
    from tally.model.train import train_and_register

    start = time.perf_counter()
    loaded = train_and_register()
    _rule("model registered")
    print(f"  version      {loaded.version}")
    print(f"  hash         {loaded.model_hash}")
    print(f"  features     {len(loaded.feature_names)}")
    print(f"  threshold    {loaded.threshold}")
    for k, v in loaded.metrics.items():
        if isinstance(v, float):
            print(f"  {k:<12} {v:.4f}")
    print(f"\n  done in {_elapsed(start)}")


@app.command()
def evaluate() -> None:
    """Compute performance, segment, and calibration artifacts."""
    from tally.model.metrics import write_evaluation_artifacts

    start = time.perf_counter()
    out = write_evaluation_artifacts()
    _rule("evaluation")
    for path in out:
        print(f"  wrote {path}")
    print(f"\n  done in {_elapsed(start)}")


# --------------------------------------------------------------------------- fairness


@app.command()
def fairness() -> None:
    """Run the three fairness analyses and write fairness_report.json."""
    from tally.fairness.report import build_fairness_report

    start = time.perf_counter()
    rep, path = build_fairness_report()
    _rule("fairness")
    print(f"  wrote {path}")
    print(f"  four-fifths flags     {rep['summary']['four_fifths_flags']}")
    print(f"  significant controlled disparities  {rep['summary']['significant_controlled']}")
    print(f"  proxy flags           {rep['summary']['proxy_flags']}")
    print(f"\n  done in {_elapsed(start)}")


@app.command()
def lda(
    quick: Annotated[bool, typer.Option(help="Run the reduced 40-challenger grid.")] = False,
    resume: Annotated[bool, typer.Option(help="Reuse cached challenger results.")] = True,
) -> None:
    """Search for a less discriminatory alternative and compute the Pareto frontier."""
    from tally.lda.search import run_search

    start = time.perf_counter()
    result = run_search(quick=quick, resume=resume)
    _rule("less discriminatory alternative search")
    print(f"  challengers evaluated {result['n_evaluated']} (cached: {result['n_cached']})")
    print(f"  frontier size         {len(result['frontier'])}")
    print(f"\n  {result['headline']}\n")
    print(f"  wrote {result['csv_path']}")
    print(f"  wrote {result['plot_path']}")
    print(f"\n  done in {_elapsed(start)}")


@app.command()
def drift() -> None:
    """Compare the out-of-time frame against the training reference."""
    from tally.drift.report import build_drift_report

    start = time.perf_counter()
    rep, path = build_drift_report()
    _rule("drift")
    print(f"  wrote {path}")
    print(f"  features at or above 'significant'  {rep['summary']['n_significant']}")
    print(f"  score PSI                           {rep['score']['psi']:.4f}")
    print(f"\n  done in {_elapsed(start)}")


# --------------------------------------------------------------------------- ledger


@app.command("verify-ledger")
def verify_ledger() -> None:
    """Walk the decision ledger hash chain from genesis."""
    from tally.ledger.store import LedgerStore
    from tally.ledger.verify import verify_chain

    _rule("ledger chain")
    store = LedgerStore.default()
    result = verify_chain(store)
    print(f"  rows checked  {result.rows_checked:,}")
    print(f"  head hash     {result.head_hash or '-'}")
    if result.ok:
        print("\n  chain intact.")
    else:
        print(f"\n  BROKEN at seq {result.first_broken_seq}: {result.first_broken_reason}")
        raise typer.Exit(1)


@app.command("reset-ledger")
def reset_ledger(
    yes: Annotated[bool, typer.Option("--yes", help="Confirm deletion.")] = False,
) -> None:
    """Delete the ledger database. The ledger is append-only; this destroys it wholesale."""
    from tally.config import get_config

    cfg = get_config()
    db = cfg.paths.ledger_db
    if not yes:
        print(f"  This deletes {db} and every decision recorded in it.")
        print("  Re-run with --yes to confirm.")
        raise typer.Exit(1)
    for suffix in ("", "-wal", "-shm"):
        p = db.with_name(db.name + suffix)
        if p.exists():
            p.unlink()
            print(f"  removed {p.name}")
    print("  ledger reset.")


# --------------------------------------------------------------------------- reports


@app.command()
def pack() -> None:
    """Generate the model validation pack PDF."""
    from tally.report.render import generate_validation_pack

    start = time.perf_counter()
    result = generate_validation_pack()
    _rule("validation pack")
    print(f"  wrote {result.pdf_path}")
    print(f"  pages {result.pages}   size {result.size_bytes / 1024:.0f} KB")
    print(f"\n  done in {_elapsed(start)}")


@app.command("adverse-action")
def adverse_action(decision_id: str) -> None:
    """Generate an adverse action notice for a recorded decline."""
    from tally.report.render import generate_adverse_action

    result = generate_adverse_action(decision_id)
    _rule("adverse action notice")
    print(f"  wrote {result.pdf_path}")


@app.command("batch-demo")
def batch_demo(
    n: Annotated[int, typer.Option(help="Number of applicants to sample.")] = 2000,
) -> None:
    """Score a sampled portfolio through the batch path and run fairness on it."""
    from tally.batch.runner import run_demo_batch

    start = time.perf_counter()
    result = run_demo_batch(n=n)
    _rule("batch portfolio review")
    print(f"  batch_id      {result['batch_id']}")
    print(f"  rows scored   {result['row_count']:,}")
    print(f"  approval rate {result['approval_rate']:.1%}")
    print(f"  batch root    {result['batch_root_hash']}")
    print(f"\n  done in {_elapsed(start)}")


# --------------------------------------------------------------------------- run


@app.command()
def demo(
    port: Annotated[int, typer.Option(help="Port to serve on.")] = 0,
    build: Annotated[bool, typer.Option(help="Build missing artifacts first.")] = True,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
) -> None:
    """Serve the demo application."""
    import threading
    import webbrowser

    import uvicorn

    from tally.config import get_config
    from tally.pipeline import ensure_artifacts

    cfg = get_config()
    if build:
        ensure_artifacts()

    host = cfg.service["host"]
    port = port or int(cfg.service["port"])
    url = f"http://{host}:{port}"

    _rule("serving")
    print(f"  {url}")
    print("  ctrl-c to stop\n")

    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    # workers=1 is mandatory: the ledger is a single-writer SQLite database.
    uvicorn.run("tally.serving.app:app", host=host, port=port, workers=1, log_level="warning")


@app.command("all")
def run_all(
    quick: Annotated[bool, typer.Option(help="Reduced LDA grid for a fast first run.")] = False,
) -> None:
    """Build every artifact end to end."""
    from tally.pipeline import build_everything

    start = time.perf_counter()
    build_everything(quick=quick)
    print(f"\n  full build finished in {_elapsed(start)}")


# --------------------------------------------------------------------------- dev


@app.command()
def test(
    fast: Annotated[bool, typer.Option(help="Skip tests marked slow.")] = False,
    k: Annotated[str, typer.Option(help="pytest -k expression.")] = "",
) -> None:
    """Run the test suite."""
    import subprocess

    cmd = ["uv", "run", "pytest"]
    if fast:
        cmd += ["-m", "not slow"]
    if k:
        cmd += ["-k", k]
    raise typer.Exit(subprocess.run(cmd).returncode)


@app.command("test-fast")
def test_fast() -> None:
    """Run only the tests that need no data or model."""
    import subprocess

    raise typer.Exit(subprocess.run(["uv", "run", "pytest", "-m", "not slow"]).returncode)


@app.command()
def fmt() -> None:
    """Format and autofix."""
    import subprocess

    subprocess.run(["uv", "run", "ruff", "format", "."], check=False)
    raise typer.Exit(subprocess.run(["uv", "run", "ruff", "check", "--fix", "."]).returncode)


@app.command()
def lint() -> None:
    """Lint without fixing."""
    import subprocess

    raise typer.Exit(subprocess.run(["uv", "run", "ruff", "check", "."]).returncode)


@app.command()
def clean() -> None:
    """Remove generated artifacts. Leaves raw data and the ledger alone."""
    import shutil

    from tally.config import get_config

    cfg = get_config()
    for target in (cfg.paths.artifacts, cfg.paths.models):
        if target.exists():
            shutil.rmtree(target)
            print(f"  removed {target}")
    cfg.paths.ensure()
    print("  clean.")


def main() -> None:  # pragma: no cover - console entry point
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    main()
