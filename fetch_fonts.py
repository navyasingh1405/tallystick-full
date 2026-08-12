"""Fetch the woff2 files the app and the generated PDFs embed.

Run once; the files are committed so `tally setup` needs no network for fonts and the
PDFs render identically on every machine. Chromium substitutes system fonts otherwise,
which makes byte-comparison of two runs meaningless.

    uv run python fetch_fonts.py
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx

OUT = Path("web/assets/fonts")
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

FAMILIES = {
    "Inter": "Inter:wght@400;500;600;700",
    "SourceSerif4": "Source+Serif+4:opsz,wght@8..60,400;8..60,600",
    "JetBrainsMono": "JetBrains+Mono:wght@400;500",
}

BLOCK = re.compile(r"/\*\s*([a-z0-9-]+)\s*\*/\s*@font-face\s*\{(.*?)\}", re.S)
URL = re.compile(r"url\((https://[^)]+?\.woff2)\)")
WEIGHT = re.compile(r"font-weight:\s*(\d+)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, spec in FAMILIES.items():
        css = httpx.get(
            f"https://fonts.googleapis.com/css2?family={spec}&display=swap",
            headers=UA,
            timeout=30,
        ).text

        # Latin only. The cyrillic and greek subsets triple the payload for glyphs
        # this project never renders.
        #
        # All three families are variable fonts, so every requested weight resolves
        # to the same file. One file per family covers the whole weight range.
        for subset, body in BLOCK.findall(css):
            if subset != "latin":
                continue
            url = URL.search(body)
            if not url:
                continue
            data = httpx.get(url.group(1), headers=UA, timeout=60).content
            (OUT / f"{name}.woff2").write_bytes(data)
            print(f"  {name + '.woff2':<24} {len(data) / 1024:6.1f} KB")
            break


if __name__ == "__main__":
    main()
