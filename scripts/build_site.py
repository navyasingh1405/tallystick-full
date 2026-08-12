"""Assemble the public marketing site for a static host.

Only the landing page belongs on a CDN. `app.html` and `login.html` are shells for a
service that will not be there — serving them from the static host would give a
visitor a dashboard that 401s on every call and a sign-in form that posts into the
void. This copies the landing page and exactly the assets it references, and nothing
else.

    uv run python scripts/build_site.py --api https://tallystick.fly.dev

The API base is injected as a global rather than compiled into the JavaScript, so the
same `config.js` works unmodified when the service serves the page itself.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DIST = ROOT / "dist"

#: Everything the landing page actually loads. Listed explicitly rather than copying
#: `web/` wholesale, so a new dashboard-only asset cannot quietly ship to the CDN.
CSS = [
    "tokens.css", "base.css", "components.css", "landing.css",
    # The walkthrough renders the application's own captured markup, so it needs the
    # application's own rules. auth.css is for the access request page.
    "panes.css", "account.css", "walkthrough.css", "auth.css",
]
JS = [
    "config.js", "motion.js", "scorer.js", "landing.js",
    "walkthrough.js", "request_access.js",
]

#: Pages beyond the landing page. app.html and login.html are still excluded: both
#: need a service behind them, and these two deliberately do not.
PAGES = ["walkthrough.html", "request-access.html"]
FONTS = ["Inter.woff2", "JetBrainsMono.woff2", "SourceSerif4.woff2", "InstrumentSerif.woff2"]


def build(api_base: str, app_base: str = "", contact: str = "hello@example.com") -> Path:
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "css").mkdir(parents=True)
    (DIST / "js").mkdir(parents=True)
    (DIST / "assets" / "fonts").mkdir(parents=True)

    for name in CSS:
        shutil.copy2(WEB / "css" / name, DIST / "css" / name)
    for name in JS:
        shutil.copy2(WEB / "js" / name, DIST / "js" / name)
    for name in FONTS:
        source = WEB / "assets" / "fonts" / name
        if source.exists():
            shutil.copy2(source, DIST / "assets" / "fonts" / name)

    # The exported model, so the page can score with no service behind it. Verified
    # against the Python by scripts/verify_browser_model.mjs before it ships — an
    # unverified reimplementation would look live while quietly disagreeing with the
    # system it claims to be, which is worse than a screenshot.
    model = WEB / "assets" / "model.json"
    if model.exists():
        shutil.copy2(model, DIST / "assets" / "model.json")
    elif not api_base:
        raise SystemExit(
            "build_site: no service configured and web/assets/model.json is missing, "
            "so the page would have no way to score.\n"
            "            Run: uv run python scripts/export_model.py"
        )

    walkthrough = WEB / "assets" / "walkthrough.json"
    if walkthrough.exists():
        capture = walkthrough.read_text(encoding="utf-8")
        if not api_base:
            # The captured panes reference the service for two files this build also
            # ships. Pointing them at the shipped copies costs nothing and turns two
            # dead links into the two artifacts a visitor most wants to open. Every
            # other /api/ reference stays as it is and walkthrough.js renders it
            # inert, which is the truthful treatment of a link to a service that is
            # not there.
            capture = capture.replace("/api/lda/frontier.png", "/assets/lda_frontier.png")
            capture = re.sub(
                r"/api/validation-pack/file/[^\"'\\\\]+\.pdf",
                "/assets/validation_pack_sample.pdf",
                capture,
            )
        (DIST / "assets" / "walkthrough.json").write_text(capture, encoding="utf-8")
    elif not api_base:
        raise SystemExit(
            "build_site: web/assets/walkthrough.json is missing, so the demo CTA "
            "would lead to an empty page.\n"
            "            Run, against a live instance:\n"
            "              uv run python scripts/capture_walkthrough.py"
        )

    html = (WEB / "landing.html").read_text(encoding="utf-8")

    base = api_base.rstrip("/")

    # With no service to ask, the figures are baked from the same artifacts the
    # validation pack reads, and stamped with the date they were measured. A
    # marketing page quoting undated numbers is exactly the habit this project
    # argues against; quoting them as "as at <date>, model <version>" is not.
    if not base:
        payload = json.dumps(_snapshot(), separators=(",", ":"))
        marker = '<script type="module" src="/js/landing.js"></script>'
        html = html.replace(
            marker,
            f"<script>window.__TALLY_SNAPSHOT__ = {payload};</script>\n{marker}",
        )
        shutil.copy2(
            ROOT / "artifacts" / "lda" / "lda_frontier.png",
            DIST / "assets" / "lda_frontier.png",
        )
        html = html.replace('src="/api/public/frontier.png"', 'src="/assets/lda_frontier.png"')

    # Where the calls to action lead.
    #
    # With a service configured they lead to it. With none, "open the demo" would be
    # a button that goes nowhere, so it leads instead to the recorded walkthrough:
    # the product's own panes, captured from a running instance with real output in
    # them. It is not live, and the page it opens says so without being asked.
    if not app_base:
        pack = ROOT / "docs" / "sample" / "validation_pack_sample.pdf"
        if pack.exists():
            shutil.copy2(pack, DIST / "assets" / "validation_pack_sample.pdf")
        notice = ROOT / "docs" / "sample" / "adverse_action_sample.pdf"
        if notice.exists():
            shutil.copy2(notice, DIST / "assets" / "adverse_action_sample.pdf")

        html = html.replace(
            '<button class="btn primary large" id="cta-demo">\n'
            "                Open the demo — no account needed\n"
            "            </button>",
            '<a class="btn primary large" href="/walkthrough.html">'
            "Walk through the platform — no account needed</a>",
        )
        # A form on a page, not a mail client thrown open by a click. Someone still
        # deciding should be able to read what is being asked of them before
        # anything of theirs is involved.
        html = html.replace(
            '<a class="btn ghost large" href="/login" data-app-link>Sign in or request access</a>',
            '<a class="btn ghost large" href="/request-access.html">Request access</a>',
        )
        html = html.replace(
            '<a class="btn ghost tiny" href="/login" data-app-link>Sign in</a>',
            '<a class="btn ghost tiny" href="/request-access.html">Request access</a>',
        )
        html = html.replace(
            '<a class="btn primary tiny" href="#access">Open the demo</a>',
            '<a class="btn primary tiny" href="/walkthrough.html">Walkthrough</a>',
        )

    if base:
        # Injected before the module script so `config.js` sees it on first read.
        injection = (
            f'<script>window.__TALLY_API__ = "{base}";</script>\n'
            '<script type="module" src="/js/landing.js"></script>'
        )
        html = html.replace('<script type="module" src="/js/landing.js"></script>', injection)

    # The frontier image is served by the API. Left as a relative path it would 404
    # on the CDN; landing.js also rewrites it at runtime, and this keeps the markup
    # honest for anyone reading the source.
    if base:
        html = html.replace('src="/api/public/frontier.png"', f'src="{base}/api/public/frontier.png"')

        # Links into the application are rewritten here as well as at runtime. Doing
        # it only in JavaScript would leave a page whose sign-in button 404s for
        # anyone whose script failed to load — the one link that absolutely has to
        # work even when everything else on the page has not.
        html = re.sub(
            r'href="(/(?:login|app)[^"]*)"(\s+data-app-link)',
            lambda m: f'href="{base}{m.group(1)}"{m.group(2)}',
            html,
        )

    (DIST / "index.html").write_text(html, encoding="utf-8")

    for name in PAGES:
        page = (WEB / name).read_text(encoding="utf-8")
        if name == "request-access.html":
            # Injected rather than compiled in, so the same page works unmodified
            # whoever builds it and whatever address they build it for.
            page = page.replace(
                '<script type="module" src="/js/request_access.js"></script>',
                f'<script>window.__TALLY_CONTACT__ = "{contact}";</script>\n'
                '<script type="module" src="/js/request_access.js"></script>',
            )
        (DIST / name).write_text(page, encoding="utf-8")
        _verify(page, base, page_name=name)

    _verify(html, base)
    return DIST


def _snapshot() -> dict:
    """The same figures the public summary endpoint would return, read from disk.

    Deliberately assembled by importing the service's own loaders rather than by
    re-reading the JSON here: one definition of what each figure means, shared with
    the endpoint and the validation pack.
    """
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from tally.fairness.report import load_fairness_report
    from tally.lda.search import load_lda_summary
    from tally.model.registry import load_current
    from tally.report.render import latest_pack_pages

    model = load_current()
    fairness = load_fairness_report()
    lda = load_lda_summary()

    race = {g["group"]: g for g in fairness["raw_disparity"]["race"]["groups"]}
    controlled = fairness["controlled"]["specifications"]["race"]["primary"]["groups"]
    significant = sorted(
        (
            {
                "group": name,
                "ame_pp": e["ame_pp"],
                "p_value": e["p_value"],
                "n": e["n_focal"],
                "raw_rate": race.get(name, {}).get("approval_rate"),
                "raw_ratio": race.get(name, {}).get("ratio"),
            }
            for name, e in controlled.items()
            if e.get("status") == "ok" and e.get("significant")
        ),
        key=lambda g: g["ame_pp"],
    )

    return {
        "measured_at": fairness["generated_at"][:10],
        "model_version": model.version,
        "model_hash": model.model_hash,
        "ledger_height": None,
        "chain_ok": None,
        "pack_pages": latest_pack_pages(),
        "fairness": {
            "evaluation_n": fairness["evaluation_frame"]["n"],
            "evaluation_year": fairness["evaluation_frame"]["activity_year"],
            "four_fifths_flags": fairness["summary"]["four_fifths_flags"],
            "significant": significant[:3],
            "proxy_top": (fairness["proxy"]["ranked"] or [{}])[0],
            "proxy_floor": fairness["proxy"]["negative_control"]["max_nmi"],
            "set_level_auc": fairness["proxy"]["set_level"].get("macro_auc"),
        },
        "lda": {
            "headline": lda["headline"],
            "n_challengers": lda["n_challengers"],
            "frontier_size": len(lda["frontier"]),
            "incumbent_auc": lda["incumbent"]["auc"],
            "incumbent_disparity": lda["incumbent"]["disparity_ame"],
            "focal_group": lda["objective"]["focal_group"],
        },
    }


def _verify(html: str, base: str, page_name: str = "landing") -> None:
    """Fail loudly rather than shipping a page that half works."""
    problems: list[str] = []
    servable = {"/", *(f"/{name}" for name in PAGES)}

    for raw in re.findall(r'(?:href|src)="(/[^"]+)"', html):
        # A fragment is resolved by the browser, not requested from the host, so
        # "/#access" is a request for "/" and must not be judged as a missing file.
        path = raw.split("#", 1)[0] or "/"
        if path.startswith(("/css/", "/js/", "/assets/")):
            if not (DIST / path.lstrip("/")).exists():
                problems.append(f"references {path}, which is not in dist/")
        elif path.startswith("/api/"):
            problems.append(f"references {path} relatively; it will 404 on a static host")
        elif path not in servable:
            problems.append(f"references {path}, which the static host does not serve")

    if base and 'window.__TALLY_API__' not in html:
        problems.append("API base was requested but not injected")

    if problems:
        raise SystemExit(f"build_site [{page_name}]: " + "\n            ".join(problems))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api",
        default="",
        help="Base URL of a running service, e.g. https://tallystick.fly.dev. Omit to "
        "build a standalone site that scores in the browser and quotes figures "
        "measured at build time.",
    )
    parser.add_argument(
        "--contact",
        default="hello@example.com",
        help="Address behind the request-access link on a standalone build.",
    )
    args = parser.parse_args()

    out = build(args.api, app_base=args.api, contact=args.contact)
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    files = sum(1 for f in out.rglob("*") if f.is_file())
    print(f"  built {out.relative_to(ROOT)}  ({files} files, {total / 1024:.0f} KB)")
    print(f"  API base: {args.api or '(same origin)'}")


if __name__ == "__main__":
    main()
