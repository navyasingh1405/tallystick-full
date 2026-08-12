"""Environment preflight.

Runs before anything expensive so failures are one readable line rather than a
stack trace forty minutes into a build.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from tally.config import get_config

REQUIRED_MODULES = [
    "duckdb",
    "pandas",
    "pyarrow",
    "numpy",
    "scipy",
    "sklearn",
    "lightgbm",
    "shap",
    "statsmodels",
    "matplotlib",
    "fastapi",
    "uvicorn",
    "pydantic",
    "jinja2",
    "playwright",
    "typer",
    "yaml",
    "pypdf",
]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True

    @property
    def symbol(self) -> str:
        if self.ok:
            return "[ok]"
        return "[!!]" if self.fatal else "[--]"


def _check_python() -> Check:
    v = sys.version_info
    ok = (v.major, v.minor) == (3, 11)
    return Check(
        "python 3.11",
        ok,
        f"{v.major}.{v.minor}.{v.micro}"
        + ("" if ok else "  (LightGBM/SHAP pins in pyproject.toml target 3.11)"),
    )


def _check_modules() -> list[Check]:
    out = []
    for mod in REQUIRED_MODULES:
        found = importlib.util.find_spec(mod) is not None
        out.append(Check(f"import {mod}", found, "" if found else "run: uv sync"))
    return out


def _browsers_root() -> Path:
    """Where `playwright install` puts browsers, per platform."""
    override = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if override:
        return Path(override)
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", "~")).expanduser() / "ms-playwright"
    if sys.platform == "darwin":
        return Path("~/Library/Caches/ms-playwright").expanduser()
    return Path("~/.cache/ms-playwright").expanduser()


def chromium_executable() -> Path | None:
    """Locate the installed Chromium without starting the Playwright driver.

    Starting the driver just to read a path spawns a node process and leaves
    asyncio teardown noise on stderr, which makes `tally doctor` look broken.
    """
    root = _browsers_root()
    if not root.exists():
        return None
    # Directory names vary by build: chrome-win, chrome-win64, chrome-linux, chrome-mac.
    patterns = (
        "chromium-*/chrome-win*/chrome.exe",
        "chromium-*/chrome-linux*/chrome",
        "chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium",
        "chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell*",
    )
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[-1]
    return None


def _check_chromium() -> Check:
    if importlib.util.find_spec("playwright") is None:
        return Check("playwright chromium", False, "playwright not installed; run: uv sync")
    exe = chromium_executable()
    if exe is None:
        return Check(
            "playwright chromium", False, "run: uv run playwright install chromium"
        )
    return Check("playwright chromium", True, exe.parent.parent.name)


def _check_onedrive(root: Path) -> Check:
    """OneDrive can lock or copy SQLite -wal/-shm mid-write. That corrupts the ledger."""
    parts = {p.lower() for p in root.parts}
    inside = any("onedrive" in p for p in parts)
    return Check(
        "repo outside OneDrive",
        not inside,
        str(root)
        + (
            ""
            if not inside
            else "  <- OneDrive sync can lock or copy decisions.db-wal mid-write. "
            "Move the repository outside OneDrive."
        ),
        fatal=False,
    )


def _check_writable(root: Path) -> Check:
    cfg = get_config()
    try:
        cfg.paths.ensure()
        probe = cfg.paths.artifacts / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return Check("directories writable", True, str(root))
    except Exception as exc:  # noqa: BLE001
        return Check("directories writable", False, f"{type(exc).__name__}: {exc}")


def _check_git() -> Check:
    found = shutil.which("git") is not None
    return Check("git present", found, "" if found else "optional, used for provenance", fatal=False)


def run_checks() -> list[Check]:
    cfg = get_config()
    root = cfg.paths.root
    checks: list[Check] = [_check_python()]
    checks.extend(_check_modules())
    checks.append(_check_chromium())
    checks.append(_check_writable(root))
    checks.append(_check_onedrive(root))
    checks.append(_check_git())
    return checks


def report(checks: list[Check]) -> bool:
    """Print the check table. Returns True when no fatal check failed."""
    width = max(len(c.name) for c in checks) + 2
    for c in checks:
        line = f"  {c.symbol} {c.name.ljust(width)} {c.detail}".rstrip()
        print(line)
    failures = [c for c in checks if not c.ok and c.fatal]
    warnings = [c for c in checks if not c.ok and not c.fatal]
    print()
    if failures:
        print(f"  {len(failures)} fatal problem(s). Fix these before continuing.")
        return False
    if warnings:
        print(f"  ok, with {len(warnings)} warning(s).")
    else:
        print("  all checks passed.")
    return True
