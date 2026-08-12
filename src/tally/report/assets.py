"""Embedded assets for the generated documents.

Fonts are inlined as data URIs. Chromium substitutes a system face for any font it
cannot load, so a PDF that references fonts by name renders differently on different
machines — which would make "regenerate and compare" meaningless.
"""

from __future__ import annotations

import base64
import functools
from pathlib import Path

from tally.config import get_config

#: family name -> (file stem, generic fallback stack)
FONTS: dict[str, tuple[str, str]] = {
    "Inter": ("Inter", "system-ui, -apple-system, 'Segoe UI', sans-serif"),
    "Source Serif 4": ("SourceSerif4", "Georgia, 'Times New Roman', serif"),
    "JetBrains Mono": ("JetBrainsMono", "ui-monospace, Consolas, monospace"),
}


def fonts_dir() -> Path:
    return get_config().paths.web / "assets" / "fonts"


@functools.lru_cache(maxsize=1)
def fonts_css() -> str:
    """@font-face rules with the woff2 payload inlined.

    All three families are variable fonts, so one file covers the full weight range
    and `font-weight` is declared as a range rather than a single value.
    """
    directory = fonts_dir()
    rules: list[str] = []

    for family, (stem, _fallback) in FONTS.items():
        path = directory / f"{stem}.woff2"
        if not path.exists():
            continue
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        rules.append(
            "@font-face{"
            f"font-family:'{family}';"
            "font-style:normal;"
            "font-weight:100 900;"
            "font-display:block;"
            f"src:url(data:font/woff2;base64,{payload}) format('woff2');"
            "}"
        )

    if not rules:
        rules.append(
            "/* No embedded fonts found. Run: uv run python fetch_fonts.py\n"
            "   Rendering will fall back to system faces and will not be\n"
            "   reproducible across machines. */"
        )
    return "\n".join(rules)


@functools.lru_cache(maxsize=8)
def stylesheet(name: str) -> str:
    return (Path(__file__).parent / "templates" / name).read_text(encoding="utf-8")


def fonts_available() -> bool:
    return all((fonts_dir() / f"{stem}.woff2").exists() for stem, _ in FONTS.values())
