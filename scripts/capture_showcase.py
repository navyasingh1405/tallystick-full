"""Screenshot the deployed site for the showcase repository.

Shot against the live URL rather than a local build, so what the showcase claims is
on the internet is what was photographed. Re-run after a deploy that changes the
look of anything below.

    uv run python scripts/capture_showcase.py --out ../tallystick-showcase/images
"""

from __future__ import annotations

import argparse
from pathlib import Path

SITE = "https://tallystick-one.vercel.app"

#: name, path, and what to do before the shutter. Full-page shots are avoided for
#: the long marketing page — a 6000px tall image is unreadable in a README.
SHOTS: list[dict] = [
    {
        "name": "01-landing-hero",
        "path": "/",
        "wait": 9000,
        "clip_selector": None,
        "height": 900,
    },
    {
        "name": "02-scorer",
        "path": "/#try",
        "wait": 9000,
        "clip_selector": "#try",
    },
    {
        "name": "03-ledger-chain",
        "path": "/#ledger",
        "wait": 7000,
        "clip_selector": "#ledger",
    },
    {
        "name": "04-fairness",
        "path": "/#fairness",
        "wait": 7000,
        "clip_selector": "#fairness",
    },
    {
        "name": "05-walkthrough-score",
        "path": "/walkthrough.html#/score",
        "wait": 5000,
        "clip_selector": None,
        "height": 1000,
    },
    {
        "name": "06-walkthrough-ledger",
        "path": "/walkthrough.html#/ledger",
        "wait": 5000,
        "clip_selector": None,
        "height": 1000,
        # Past the retrieval form to the three verification checks. They are the
        # claim the pane exists to make, and a shot that stops above them shows
        # only the question.
        "scroll": 620,
    },
    {
        "name": "07-walkthrough-fairness",
        "path": "/walkthrough.html#/fairness",
        "wait": 5000,
        "clip_selector": None,
        "height": 1000,
    },
    {
        "name": "08-walkthrough-portfolio",
        "path": "/walkthrough.html#/batch",
        "wait": 5000,
        "clip_selector": None,
        "height": 1000,
    },
    {
        "name": "09-request-access",
        "path": "/request-access.html",
        "wait": 3000,
        "clip_selector": None,
        "height": 900,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Directory to write PNGs into.")
    parser.add_argument("--site", default=SITE)
    parser.add_argument("--width", type=int, default=1440)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for shot in SHOTS:
            height = shot.get("height", 900)
            page = browser.new_page(
                viewport={"width": args.width, "height": height},
                # Retina, so the images stay sharp when GitHub scales them down.
                device_scale_factor=2,
            )
            page.goto(f"{args.site}{shot['path']}", wait_until="networkidle")
            page.wait_for_timeout(shot["wait"])

            # The landing page reveals sections on scroll. Anything still waiting on
            # an observer would photograph as blank space.
            page.evaluate(
                "() => { document.documentElement.classList.remove('js-reveal');"
                " document.querySelectorAll('.band, .hero-visual, .problem, .limit')"
                " .forEach(el => el.classList.add('in')); }"
            )
            if shot.get("scroll"):
                page.evaluate(
                    "(y) => { const m = document.querySelector('#outlet');"
                    " if (m && m.scrollHeight > m.clientHeight) m.scrollTop = y;"
                    " else window.scrollTo(0, y); }",
                    shot["scroll"],
                )
                page.wait_for_timeout(600)

            page.wait_for_timeout(700)

            target = out / f"{shot['name']}.png"
            selector = shot["clip_selector"]
            if selector:
                page.locator(selector).screenshot(path=str(target))
            else:
                page.screenshot(path=str(target))

            size = target.stat().st_size / 1024
            print(f"  {shot['name']:<28} {size:>7.0f} KB")
            page.close()

        browser.close()

    total = sum(f.stat().st_size for f in out.glob("*.png")) / 1024 / 1024
    print(f"\n  wrote {len(SHOTS)} images to {out}  ({total:.1f} MB)")


if __name__ == "__main__":
    main()
