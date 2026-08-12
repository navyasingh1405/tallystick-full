"""Render the generated documents to PDF via headless Chromium.

Chromium rather than a Python HTML-to-PDF library: it is the only renderer with
complete support for the CSS these documents use (grid, page-break control, running
headers), and on Windows it needs no native library beyond the browser Playwright
already installs.

Two Chromium print behaviours are worth knowing, because both fail silently:

* `wait_until="networkidle"` never fires for a page loaded via `set_content` with
  data-URI assets, so the call hangs until timeout. Wait for `load`, then explicitly
  for `document.fonts.ready`.
* Header and footer templates are clipped to nothing unless the page margins exceed
  the template height, and they need an explicit `font-size` because they inherit
  none from the page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tally.config import get_config
from tally.report.assets import fonts_css, stylesheet

TEMPLATE_DIR = Path(__file__).parent / "templates"

#: Tokens that cannot occur in legitimate prose. Checked against the extracted PDF
#: text. A validation pack with a placeholder in it is worse than no pack: it looks
#: complete and is not.
PLACEHOLDER_PATTERNS = (
    r"\bTODO\b",
    r"\bTBD\b",
    r"\bFIXME\b",
    r"\bXXX\b",
    r"lorem ipsum",
    r"\bplaceholder\b",
    r"\{\{",
    r"\}\}",
    r"\bUndefined\b",
)

#: Values that indicate a template variable resolved to nothing. Checked against the
#: HTML, where an unresolved value sits alone inside an element and can be matched
#: exactly — unlike in extracted PDF text, where "none" is ordinary English and
#: flagging it produces false alarms that train people to ignore the check.
EMPTY_CELL_PATTERNS = (
    r">\s*None\s*<",
    r">\s*nan\s*<",
    r">\s*NaN\s*<",
    r">\s*undefined\s*<",
    r">\s*null\s*<",
    r">\s*Undefined\s*<",
    r"\bnan\s*(?:pp|%)",
    r">\s*\$?nan\b",
)


class PlaceholderFound(RuntimeError):
    """Generated output contains unresolved or placeholder content."""


@dataclass(frozen=True)
class RenderResult:
    pdf_path: Path
    html_path: Path
    pages: int
    size_bytes: int
    seconds: float


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,  # a missing variable must fail, not render blank
        autoescape=False,  # content is ours; tables contain HTML entities
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["comma"] = lambda v: f"{v:,}"
    return env


def render_html(template: str, context: dict[str, Any], css: str) -> str:
    env = _env()
    return env.get_template(template).render(
        **context, pack_css=stylesheet(css), fonts_css=fonts_css()
    )


def html_to_pdf(
    html: str,
    out: Path,
    *,
    header: str = "",
    footer: str = "",
    margins: dict[str, str] | None = None,
) -> int:
    """Print HTML to PDF. Returns the page count."""
    from playwright.sync_api import sync_playwright

    from tally.doctor import chromium_executable

    if chromium_executable() is None:
        raise RuntimeError(
            "Chromium is not installed. Run:  uv run playwright install chromium"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    margins = margins or {"top": "22mm", "bottom": "20mm", "left": "18mm", "right": "18mm"}

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(html, wait_until="load")
            # networkidle never settles for data-URI assets; wait on fonts instead.
            page.evaluate("() => document.fonts.ready")
            page.emulate_media(media="print")
            page.pdf(
                path=str(out),
                format="A4",
                print_background=True,
                display_header_footer=bool(header or footer),
                header_template=header or "<span></span>",
                footer_template=footer or "<span></span>",
                margin=margins,
                prefer_css_page_size=False,
            )
        finally:
            browser.close()

    return _page_count(out)


def _page_count(pdf: Path) -> int:
    from pypdf import PdfReader

    return len(PdfReader(str(pdf)).pages)


def latest_pack_pages() -> int | None:
    """Page count of the most recently generated validation pack, or None.

    Exists so the landing page can state the pack's length without anyone typing a
    number into the markup. A pack that grows a section and a page count that does
    not is a small lie, but it is the same kind of lie the whole product argues
    against, so it is worth not telling.
    """
    from pypdf import PdfReader

    pack = get_config().paths.artifact("packs", "validation_pack_latest.pdf")
    if not pack.exists():
        return None
    return len(PdfReader(str(pack)).pages)


def _footer_template(model_version: str, model_hash: str, generated: str) -> str:
    """Running footer. Needs an explicit font-size or Chromium renders it invisibly."""
    return f"""
    <div style="width:100%;font-family:Inter,'Segoe UI',sans-serif;font-size:7pt;
                color:#8a939e;padding:0 18mm;display:flex;justify-content:space-between;">
      <span>Model {model_version} &middot; SHA-256 {model_hash[:16]}&hellip;</span>
      <span>{generated}</span>
      <span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>
    </div>
    """


def _matches(text: str, patterns: tuple[str, ...], flags: int = 0) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=flags):
            start = max(0, match.start() - 70)
            excerpt = " ".join(text[start : match.end() + 70].split())
            found.append(f"{match.group(0)!r} in: …{excerpt}…")
    return found


def assert_no_placeholders(text: str, *, where: str) -> None:
    """Fail if generated output contains unresolved template content."""
    found = _matches(text, PLACEHOLDER_PATTERNS, re.IGNORECASE)
    if found:
        raise PlaceholderFound(
            f"{where} contains {len(found)} placeholder value(s):\n  " + "\n  ".join(found[:8])
        )


def assert_no_empty_values(html: str, *, where: str) -> None:
    """Fail if any rendered element resolved to a missing value.

    This is the check that catches a template variable silently rendering as None and
    a number quietly disappearing from a table — the failure mode that would let a
    validation pack look complete while being wrong.
    """
    found = _matches(html, EMPTY_CELL_PATTERNS)
    if found:
        raise PlaceholderFound(
            f"{where} contains {len(found)} unresolved value(s):\n  " + "\n  ".join(found[:8])
        )


def generate_validation_pack(*, check_placeholders: bool = True) -> RenderResult:
    import time

    from tally.report.context import build_context

    started = time.perf_counter()
    cfg = get_config()
    context = build_context()
    context["sample_notice"] = _sample_declined_decision()
    context["sign_agreement"] = _sign_agreement()

    html = render_html("validation_pack.html.j2", context, "pack.css")
    if check_placeholders:
        assert_no_empty_values(html, where="validation pack HTML")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = cfg.paths.artifact("packs", f"validation_pack_{context['model']['version']}_{stamp}")
    html_path = base.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    pdf_path = base.with_suffix(".pdf")
    pages = html_to_pdf(
        html,
        pdf_path,
        footer=_footer_template(
            context["model"]["version"], context["model"]["hash"], context["generated_at"]
        ),
    )

    if check_placeholders:
        assert_no_placeholders(pdf_text(pdf_path), where=pdf_path.name)

    # A stable path for the UI to link to, alongside the timestamped archive copy.
    latest = cfg.paths.artifact("packs", "validation_pack_latest.pdf")
    latest.write_bytes(pdf_path.read_bytes())

    return RenderResult(
        pdf_path=pdf_path,
        html_path=html_path,
        pages=pages,
        size_bytes=pdf_path.stat().st_size,
        seconds=time.perf_counter() - started,
    )


def generate_adverse_action(
    decision_id: str, ledger: Any = None, *, check_placeholders: bool = True
) -> RenderResult:
    """Render an adverse action notice from a recorded decline.

    Built from the ledger row, not from a live re-score, so the notice can never
    state a decision other than the one on record. The chain is verified at
    generation time and the status is printed on the notice.
    """
    import json
    import time

    from tally.ledger.store import LedgerStore
    from tally.ledger.verify import verify_chain

    started = time.perf_counter()
    cfg = get_config()
    ledger = ledger or LedgerStore.default()

    row = ledger.get(decision_id)
    if row is None:
        raise KeyError(f"no decision {decision_id!r} in the ledger")
    if row["decision"] != "DECLINED":
        raise ValueError(
            f"decision {decision_id} was {row['decision']}. An adverse action notice "
            "is required only for adverse action."
        )

    chain = verify_chain(ledger)
    reason_codes = json.loads(row["reason_codes_json"])
    if not reason_codes:
        raise ValueError(
            f"decision {decision_id} carries no reason codes, so no notice can state "
            "specific principal reasons as Regulation B requires."
        )

    counterfactual = _counterfactual_for(row)

    context = {
        "issuer": cfg.report,
        "decision_id": decision_id,
        "created_at": row["created_at"],
        "notice_date": datetime.now(UTC).strftime("%d %B %Y"),
        "model_version": row["model_version"],
        "model_hash": row["model_hash"],
        "ledger_seq": row["seq"],
        "row_hash": row["row_hash"],
        "chain_status": (
            f"verified intact across {chain.rows_checked:,} decisions"
            if chain.ok
            else f"BROKEN at position {chain.first_broken_seq}"
        ),
        "reason_codes": reason_codes,
        "counterfactual": counterfactual,
    }

    html = render_html("adverse_action.html.j2", context, "letter.css")
    if check_placeholders:
        assert_no_empty_values(html, where="adverse action notice HTML")
    pdf_path = cfg.paths.artifact("packs", f"adverse_action_{decision_id}.pdf")
    html_path = pdf_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    pages = html_to_pdf(html, pdf_path, margins={"top": "16mm", "bottom": "16mm", "left": "20mm", "right": "20mm"})

    if check_placeholders:
        assert_no_placeholders(pdf_text(pdf_path), where=pdf_path.name)

    return RenderResult(
        pdf_path=pdf_path,
        html_path=html_path,
        pages=pages,
        size_bytes=pdf_path.stat().st_size,
        seconds=time.perf_counter() - started,
    )


def _counterfactual_for(row: Any) -> dict[str, Any] | None:
    """Recompute the counterfactual from the stored input under the pinned model."""
    import json

    from tally.data.features import RawApplicant
    from tally.explain.counterfactual import find_counterfactual
    from tally.model.registry import load_version

    reasons = json.loads(row["reason_codes_json"])
    if not reasons:
        return None
    try:
        model = load_version(row["model_version"])
        applicant = RawApplicant.model_validate(json.loads(row["input_json"]))
        return find_counterfactual(applicant, reasons[0]["feature"], model).as_dict()
    except (FileNotFoundError, KeyError, ValueError):
        return None


def pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _sample_declined_decision() -> dict[str, Any] | None:
    """A real declined application from the evaluation frame, scored and explained.

    Section 9 shows a worked example rather than describing one. Written to a
    throwaway ledger so generating documentation never appends to the real one.
    """
    import tempfile

    from tally.data.features import applicant_from_frame_row
    from tally.data.splits import load_split
    from tally.ledger.store import LedgerStore
    from tally.model.registry import load_current
    from tally.serving.scoring import score_applicant

    try:
        model = load_current()
        frame = load_split("oot")
    except FileNotFoundError:
        return None

    ledger = LedgerStore(Path(tempfile.mkdtemp(prefix="cal-pack-")) / "sample.db")
    for _, row in frame[~frame["approved"]].head(60).iterrows():
        result = score_applicant(
            applicant_from_frame_row(row), model=model, ledger=ledger
        )
        if result.decision == "DECLINED" and result.reasons.codes:
            payload = result.as_dict()
            payload["counterfactual"] = (
                result.counterfactual.as_dict() if result.counterfactual else None
            )
            return payload
    return None


def _sign_agreement() -> dict[str, Any] | None:
    from tally.data.features import feature_matrix
    from tally.data.splits import load_split
    from tally.explain.shap_engine import get_engine
    from tally.model.registry import load_current

    try:
        model = load_current()
        frame = load_split("oot").head(1500)
    except FileNotFoundError:
        return None
    X = feature_matrix(frame, model.fit_params)[model.feature_names]
    return get_engine(model).sign_agreement(X)
