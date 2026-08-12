"""Record a walkthrough of the running application for the static site.

The marketing site has no service behind it, so the dashboard cannot be shipped —
it would 401 on every call. What ships instead is this: the real panes, driven by
Playwright against a live instance, captured as HTML after the data has loaded.

Recording it rather than rebuilding it by hand is the point. A hand-written mockup
drifts from the product the moment either changes, and every figure in it would be a
number someone typed — which is the exact habit this project argues against. Here
the walkthrough is generated from the same artifacts, by the same code paths, as the
thing it depicts. If a pane changes, re-run this.

    uv run tally demo --port 8787 --no-open      # in one shell
    uv run python scripts/capture_walkthrough.py --url http://127.0.0.1:8787

Writes web/assets/walkthrough.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "assets" / "walkthrough.json"

#: What each pane is for, in the voice of the landing page. Commentary is authored;
#: every figure inside the captured markup is not.
PANES: list[dict[str, str]] = [
    {
        "path": "/score",
        "label": "Score",
        "key": "s",
        "title": "Scoring writes the record",
        "note": (
            "There is no code path through this service that returns a decision "
            "without also writing it to the ledger. The two are one operation, so "
            "the record cannot fall behind the decision — the usual failure, where "
            "an audit trail is a separate system someone remembers to update."
        ),
    },
    {
        "path": "/ledger",
        "label": "Ledger",
        "key": "l",
        "title": "Retrieve a decision and prove it",
        "note": (
            "This is the examiner's question: explain the decision you made on this "
            "applicant months ago, and prove it has not been altered since. "
            "Retrieval walks the hash chain from genesis, loads the exact model "
            "version the decision was pinned to, and re-scores the stored "
            "application. Agreement to the last bit is the claim."
        ),
    },
    {
        "path": "/fairness",
        "label": "Fairness",
        "key": "f",
        "title": "Three questions, asked separately",
        "note": (
            "Raw approval-rate disparity, disparity that survives controls, and how "
            "much a protected characteristic can be reconstructed from the features "
            "themselves. They are different questions with different answers, and "
            "collapsing them into one fairness score loses the distinction that "
            "matters to a regulator."
        ),
    },
    {
        "path": "/batch",
        "label": "Portfolio",
        "key": "b",
        "title": "The same guarantees in bulk",
        "note": (
            "A portfolio run scores through the identical path — every row "
            "explained, every row recorded — and then re-runs the fairness "
            "analyses on what was actually decided rather than on a sample."
        ),
    },
    {
        "path": "/pack",
        "label": "Validation pack",
        "key": "p",
        "title": "The document writes itself",
        "note": (
            "Nothing in the pack is typed by hand. It is assembled from the "
            "artifacts the commands produced, which is why it can be regenerated "
            "whenever the model changes instead of being rewritten."
        ),
    },
]


#: innerHTML serialises attributes, not properties. A field filled in by script has
#: its value in the property only, so a naive capture produces a form of empty boxes
#: — the exact opposite of the point, which is to show the application holding real
#: input. This writes the live state back into the attributes first.
SERIALISE_FORM_STATE = """
() => {
    const root = document.querySelector('#outlet');
    root.querySelectorAll('input').forEach((el) => {
        if (el.type === 'checkbox' || el.type === 'radio') {
            el.toggleAttribute('checked', el.checked);
        } else {
            el.setAttribute('value', el.value);
        }
    });
    root.querySelectorAll('textarea').forEach((el) => { el.textContent = el.value; });
    root.querySelectorAll('select').forEach((el) => {
        [...el.options].forEach((o) => o.toggleAttribute('selected', o.selected));
    });
}
"""


def capture(url: str, timeout_ms: int = 45_000) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    base = url.rstrip("/")
    captured: list[dict[str, Any]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        # Wide enough that the panes lay out in their desktop arrangement. A capture
        # taken at a narrow width would bake the stacked mobile layout into markup
        # that then cannot respond, which is worse than either layout alone.
        page = browser.new_page(viewport={"width": 1480, "height": 1200})

        page.goto(f"{base}/login", wait_until="networkidle")
        page.click("text=Open the demo without an account")
        page.wait_for_url(re.compile(r".*/app.*"), timeout=timeout_ms)
        page.wait_for_selector("#outlet", timeout=timeout_ms)

        for spec in PANES:
            page.goto(f"{base}/app#{spec['path']}", wait_until="networkidle")
            page.wait_for_timeout(1200)

            if spec["path"] == "/score":
                # An empty form says nothing. Submitting one produces the decision,
                # the reasons, and the ledger receipt that the pane exists to show.
                page.click("text=Score application")
                page.wait_for_selector(".decision-word", timeout=timeout_ms)
                page.wait_for_timeout(2000)

            if spec["path"] == "/ledger":
                # Retrieve the decision just written, then verify and replay it. The
                # three checks that follow are the whole argument of the pane; a
                # capture that stopped at the retrieval form would omit the point.
                page.wait_for_timeout(1500)
                page.get_by_role("button", name="Open").first.click()
                page.wait_for_timeout(2500)
                page.get_by_role("button", name="Verify & replay").first.click()
                page.wait_for_selector("text=Walk the hash chain", timeout=timeout_ms)
                page.wait_for_timeout(2500)

            if spec["path"] == "/batch":
                # A portfolio pane with no portfolio in it shows a size picker and
                # nothing else. Running the smallest book keeps the capture quick
                # while still producing a real receipt hash and real fairness output.
                page.get_by_role("button", name=re.compile(r"^2,000")).click()
                # Wait on the results container having content, not on a word. The
                # word "batch" is in the pane's own prose, so a text match returned
                # instantly and captured the picker it was supposed to wait past.
                page.wait_for_function(
                    "() => { const el = document.querySelector('#batch-result');"
                    " return el && el.textContent.trim().length > 200; }",
                    timeout=300_000,
                )
                page.wait_for_timeout(4000)

            if spec["path"] == "/pack":
                page.get_by_role("button", name="Generate validation pack").click()
                # Chromium print-to-PDF over a twelve-section document; the service
                # took 32s on this machine, so the wait is generous rather than tight.
                page.wait_for_selector("text=/pages/i", timeout=240_000)
                page.wait_for_timeout(2500)

            page.wait_for_timeout(800)
            page.evaluate(SERIALISE_FORM_STATE)
            html = page.eval_on_selector("#outlet", "el => el.innerHTML")
            captured.append({**spec, "html": html})
            print(f"  captured {spec['path']:<12} {len(html) / 1024:>7.1f} KB")

        rail = {
            "account": page.eval_on_selector("#account", "el => el.innerHTML"),
            "status": page.eval_on_selector("#status", "el => el.innerHTML"),
        }

        # Sign the guest session out. Leaving it live would keep a session row valid
        # against a real ledger for no reason once the capture is on disk.
        try:
            page.goto(f"{base}/app", wait_until="networkidle")
            page.click("#signout", timeout=5000)
            page.wait_for_timeout(800)
        except Exception:
            pass

        browser.close()

    return {"panes": captured, "rail": rail}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8787", help="A running service.")
    args = parser.parse_args()

    payload = capture(args.url)

    import datetime as _dt

    payload["captured_at"] = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    size = OUT.stat().st_size / 1024
    print(f"\n  wrote {OUT.relative_to(ROOT)}  ({size:.0f} KB, {len(payload['panes'])} panes)")


if __name__ == "__main__":
    main()
