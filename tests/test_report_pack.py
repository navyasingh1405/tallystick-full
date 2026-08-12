"""The validation pack, and the guarantee that nothing in it was typed by hand."""

from __future__ import annotations

import re

import pytest

from tally.report.render import (
    PlaceholderFound,
    assert_no_empty_values,
    assert_no_placeholders,
)
from tally.report.sections import SECTIONS, regulatory_appendix

# --------------------------------------------------------------------------- structure


def test_twelve_sections_numbered_without_gaps():
    assert [s.number for s in SECTIONS] == list(range(1, 13))


def test_every_section_has_a_purpose_line():
    for s in SECTIONS:
        assert len(s.purpose.split()) >= 6, s.slug


def test_regulatory_appendix_covers_every_section():
    """The appendix and the table of contents are generated from one list, so they
    cannot drift apart. This asserts that property rather than trusting it."""
    rows = regulatory_appendix()
    assert [r["number"] for r in rows] == [s.number for s in SECTIONS]
    assert [r["title"] for r in rows] == [s.title for s in SECTIONS]


def test_every_section_maps_to_at_least_one_framework():
    for row in regulatory_appendix():
        assert row["sr117"] or row["eu_ai_act"] or row["ecoa"], row["number"]


# --------------------------------------------------------------------------- guards


def test_placeholder_detector_catches_unrendered_template_syntax():
    with pytest.raises(PlaceholderFound):
        assert_no_placeholders("AUC is {{ model.auc }} on the holdout", where="test")


def test_placeholder_detector_catches_todo_markers():
    for marker in ("TODO", "TBD", "FIXME", "lorem ipsum"):
        with pytest.raises(PlaceholderFound):
            assert_no_placeholders(f"Section 4 {marker} write this up", where="test")


def test_placeholder_detector_does_not_fire_on_ordinary_english():
    """'none' is a normal English word. A check that flags it produces false alarms,
    and a check that produces false alarms gets switched off."""
    assert_no_placeholders(
        "None of it detects a problem present from the beginning, and none was "
        "entered by hand.",
        where="test",
    )


def test_empty_value_detector_catches_a_missing_value_in_a_cell():
    with pytest.raises(PlaceholderFound):
        assert_no_empty_values("<td>None</td>", where="test")
    with pytest.raises(PlaceholderFound):
        assert_no_empty_values("<td>nan</td>", where="test")
    with pytest.raises(PlaceholderFound):
        assert_no_empty_values("<dd>undefined</dd>", where="test")


def test_empty_value_detector_allows_prose_containing_none():
    assert_no_empty_values("<p>None of these were imputed.</p>", where="test")


# --------------------------------------------------------------------------- output


@pytest.mark.slow
@pytest.mark.needs_model
@pytest.mark.needs_chromium
def test_pack_generates_within_the_time_budget(tmp_path):
    from tally.report.render import generate_validation_pack

    result = generate_validation_pack()
    assert result.pdf_path.exists()
    assert result.pages >= 12, "a twelve-section document cannot fit in fewer pages"
    assert result.seconds < 90, f"took {result.seconds:.0f}s, budget is 90s"


@pytest.mark.slow
@pytest.mark.needs_model
@pytest.mark.needs_chromium
def test_pack_contains_every_section_heading():
    from tally.report.render import generate_validation_pack, pdf_text

    text = pdf_text(generate_validation_pack().pdf_path)
    for section in SECTIONS:
        # The title appears in the contents and again as the heading.
        head = section.title.split(",")[0][:28]
        assert text.count(head) >= 2, f"section {section.number} heading missing"


@pytest.mark.slow
@pytest.mark.needs_model
@pytest.mark.needs_chromium
def test_pack_carries_the_current_model_version_and_hash():
    """Retraining and regenerating must produce a document describing the new model,
    with no manual editing anywhere."""
    from tally.model.registry import load_current
    from tally.report.render import generate_validation_pack, pdf_text

    model = load_current()
    text = pdf_text(generate_validation_pack().pdf_path)
    assert model.version in text
    assert model.model_hash in text


@pytest.mark.slow
@pytest.mark.needs_model
@pytest.mark.needs_chromium
def test_pack_states_what_the_model_does_not_do():
    """The single most important sentence in the document."""
    from tally.report.render import generate_validation_pack, pdf_text

    text = pdf_text(generate_validation_pack().pdf_path).lower()
    assert "does not estimate credit risk" in text
    assert "credit score" in text
    assert "replicate historical" in text or "reproduce historical" in text


@pytest.mark.slow
@pytest.mark.needs_model
@pytest.mark.needs_chromium
def test_pack_has_a_running_footer_with_provenance():
    from tally.model.registry import load_current
    from tally.report.render import generate_validation_pack, pdf_text

    model = load_current()
    result = generate_validation_pack()
    text = pdf_text(result.pdf_path)
    assert text.count(model.model_hash[:16]) >= result.pages - 1


@pytest.mark.slow
@pytest.mark.needs_model
@pytest.mark.needs_chromium
def test_fonts_are_embedded_so_rendering_is_reproducible():
    """Chromium substitutes system faces for fonts it cannot load, which would make
    two machines produce different documents from identical inputs."""
    from tally.report.assets import fonts_available, fonts_css

    assert fonts_available(), "run: uv run python fetch_fonts.py"
    css = fonts_css()
    assert "data:font/woff2;base64," in css
    assert css.count("@font-face") == 3


@pytest.mark.slow
@pytest.mark.needs_model
@pytest.mark.needs_chromium
def test_adverse_action_notice_carries_the_ecoa_language(tmp_path):
    from tally.data.features import RawApplicant
    from tally.ledger.store import LedgerStore
    from tally.report.render import generate_adverse_action, pdf_text
    from tally.serving.scoring import score_applicant

    ledger = LedgerStore(tmp_path / "d.db")
    applicant = RawApplicant(
        income_usd=66_000, loan_amount_usd=345_000, cltv=92, dti_band=">60%", loan_term=360
    )
    result = score_applicant(applicant, ledger=ledger)
    assert result.decision == "DECLINED"

    notice = generate_adverse_action(result.decision_id, ledger)
    text = pdf_text(notice.pdf_path)

    # The verbatim ECOA notice required by 12 CFR 1002.9(b)(1).
    assert "Equal Credit Opportunity Act prohibits creditors from discriminating" in text
    assert "public assistance program" in text
    assert "Consumer Credit Protection Act" in text
    # Identity, agency, and the specific principal reasons.
    assert "federal agency that administers compliance" in text
    assert result.decision_id in text
    for code in result.reasons.codes:
        assert code.applicant_text.rstrip(".") in text.replace("\n", " ")


@pytest.mark.slow
@pytest.mark.needs_model
@pytest.mark.needs_chromium
def test_notice_explains_why_fcra_disclosures_are_absent(tmp_path):
    """Public HMDA has no credit score, so the FCRA disclosures do not apply.
    Explaining the absence is more convincing than omitting it silently — or than
    including boilerplate that does not apply."""
    from tally.data.features import RawApplicant
    from tally.ledger.store import LedgerStore
    from tally.report.render import generate_adverse_action, pdf_text
    from tally.serving.scoring import score_applicant

    ledger = LedgerStore(tmp_path / "d.db")
    result = score_applicant(
        RawApplicant(
            income_usd=66_000, loan_amount_usd=345_000, cltv=92, dti_band=">60%", loan_term=360
        ),
        ledger=ledger,
    )
    text = pdf_text(generate_adverse_action(result.decision_id, ledger).pdf_path)
    assert "No consumer report or credit score was used" in text
    assert "Fair Credit Reporting Act" in text


@pytest.mark.slow
@pytest.mark.needs_model
@pytest.mark.needs_chromium
def test_no_notice_is_issued_for_an_approval(tmp_path):
    from tally.data.features import RawApplicant
    from tally.ledger.store import LedgerStore
    from tally.report.render import generate_adverse_action
    from tally.serving.scoring import score_applicant

    ledger = LedgerStore(tmp_path / "d.db")
    result = score_applicant(
        RawApplicant(
            income_usd=250_000, loan_amount_usd=300_000, cltv=55, dti_band="<20%", loan_term=360
        ),
        ledger=ledger,
    )
    assert result.decision == "APPROVED"
    with pytest.raises(ValueError, match="adverse action"):
        generate_adverse_action(result.decision_id, ledger)


@pytest.mark.slow
@pytest.mark.needs_model
@pytest.mark.needs_chromium
def test_notice_is_marked_as_a_demonstration_artifact(tmp_path):
    from tally.data.features import RawApplicant
    from tally.ledger.store import LedgerStore
    from tally.report.render import generate_adverse_action, pdf_text
    from tally.serving.scoring import score_applicant

    ledger = LedgerStore(tmp_path / "d.db")
    result = score_applicant(
        RawApplicant(
            income_usd=66_000, loan_amount_usd=345_000, cltv=92, dti_band=">60%", loan_term=360
        ),
        ledger=ledger,
    )
    text = pdf_text(generate_adverse_action(result.decision_id, ledger).pdf_path)
    assert re.search(r"DEMONSTRATION ARTIFACT", text, re.IGNORECASE)
    assert "not a legal notice" in text.lower()
