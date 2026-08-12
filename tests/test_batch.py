"""Batch portfolio review, and the protected-attribute boundary it has to hold.

A batch CSV must carry protected attributes, because fairness cannot be measured
without them, while those columns must never reach a feature vector. That tension is
where a leak would happen, so it is tested directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tally.batch.runner import CONTEXT_COLUMNS, parse_batch
from tally.data.features import ApplicantContext, ProtectedLeakage, RawApplicant


def _row(**over):
    base = {
        "income_usd": 120000,
        "loan_amount_usd": 380000,
        "property_value_usd": 450000,
        "cltv": 84,
        "dti_band": "38",
        "loan_term": 360,
        "units_band": "1",
        "preapproval": False,
        "interest_only": False,
        "balloon": False,
        "neg_amortization": False,
        "tract_to_msa_income_pct": 105,
        "derived_race": "Black or African American",
        "derived_ethnicity": "Not Hispanic or Latino",
        "derived_sex": "Female",
        "applicant_age": "35-44",
        "county_code": "08031",
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------- boundary


def test_the_two_models_share_no_fields():
    """The boundary is structural. Neither object can hold the other's data."""
    assert set(RawApplicant.model_fields) & set(ApplicantContext.model_fields) == set()


def test_protected_columns_belong_only_to_the_context_model():
    for column in ("derived_race", "derived_ethnicity", "derived_sex", "applicant_age"):
        assert column in CONTEXT_COLUMNS
        assert column not in RawApplicant.model_fields


def test_parsing_splits_each_row_and_never_merges_them():
    parsed = parse_batch(pd.DataFrame([_row(), _row(income_usd=90000)]))
    assert parsed.ok
    assert len(parsed.applicants) == 2
    assert len(parsed.contexts) == 2

    for applicant, context in zip(parsed.applicants, parsed.contexts, strict=True):
        dumped = applicant.model_dump()
        assert "derived_race" not in dumped
        assert context.derived_race == "Black or African American"


def test_feature_matrix_refuses_a_protected_column():
    frame = pd.DataFrame([_row()])
    frame["dti_band_ord"] = 5.0
    frame["units_ord"] = 0.0
    from tally.data import features as feat

    with pytest.raises(ProtectedLeakage):
        feat._assert_no_protected([*frame.columns, "derived_race"])


# --------------------------------------------------------------------------- validation


def test_row_errors_are_reported_with_row_and_column():
    frame = pd.DataFrame([_row(), _row(income_usd=-10), _row(cltv=900)])
    parsed = parse_batch(frame)
    assert not parsed.ok
    assert len(parsed.applicants) == 1

    by_row = {e.row: e for e in parsed.errors}
    assert 2 in by_row and by_row[2].column == "income_usd"
    assert 3 in by_row and by_row[3].column == "cltv"
    for error in parsed.errors:
        assert error.message


def test_a_missing_required_column_is_an_error_not_a_default():
    frame = pd.DataFrame([{k: v for k, v in _row().items() if k != "income_usd"}])
    parsed = parse_batch(frame)
    assert not parsed.ok
    assert any(e.column == "income_usd" for e in parsed.errors)


def test_unknown_columns_are_ignored_rather_than_rejected():
    """Real portfolio exports carry loan numbers and branch codes. Those should not
    stop a review, but they must not reach the model either."""
    parsed = parse_batch(pd.DataFrame([_row(internal_loan_no="ABC-123", branch="Denver-4")]))
    assert parsed.ok
    assert "internal_loan_no" not in parsed.applicants[0].model_dump()


def test_missing_optional_values_are_tolerated():
    row = _row()
    row["property_value_usd"] = np.nan
    row["tract_to_msa_income_pct"] = np.nan
    parsed = parse_batch(pd.DataFrame([row]))
    assert parsed.ok


# --------------------------------------------------------------------------- scoring


@pytest.mark.slow
@pytest.mark.needs_model
def test_batch_scoring_chains_every_row_and_issues_one_receipt(tmp_path):
    from tally.batch.runner import score_batch
    from tally.ledger.store import LedgerStore
    from tally.ledger.verify import verify_chain

    ledger = LedgerStore(tmp_path / "d.db")
    parsed = parse_batch(pd.DataFrame([_row(income_usd=90000 + i * 500) for i in range(120)]))
    result = score_batch(parsed, ledger=ledger, filename="portfolio.csv")

    assert result["row_count"] == 120
    chain = verify_chain(ledger)
    assert chain.ok
    assert chain.rows_checked == 120
    # The batch receipt is the row hash of its final entry.
    assert result["batch_root_hash"] == chain.head_hash


@pytest.mark.slow
@pytest.mark.needs_model
def test_the_ledger_row_stores_no_protected_attribute(tmp_path):
    """The decision record holds what the model saw. The demographic context lives on
    the batch artifact, keyed separately."""
    import json

    from tally.batch.runner import score_batch
    from tally.ledger.store import LedgerStore

    ledger = LedgerStore(tmp_path / "d.db")
    parsed = parse_batch(pd.DataFrame([_row() for _ in range(5)]))
    score_batch(parsed, ledger=ledger, filename="p.csv")

    for row in ledger.all_rows():
        stored = json.loads(row["input_json"])
        assert "derived_race" not in stored
        assert "derived_sex" not in stored
        assert "applicant_age" not in stored
        assert set(stored) <= set(RawApplicant.model_fields)


@pytest.mark.slow
@pytest.mark.needs_model
def test_batch_fairness_is_scoped_to_the_batch(tmp_path):
    from tally.batch.runner import run_demo_batch
    from tally.ledger.store import LedgerStore

    result = run_demo_batch(n=2500, ledger=LedgerStore(tmp_path / "d.db"))
    fairness = result["fairness"]
    assert fairness["status"] == "ok"
    assert fairness["n"] == result["row_count"]
    assert "raw_disparity" in fairness and "controlled" in fairness


@pytest.mark.slow
@pytest.mark.needs_model
def test_a_batch_measures_the_same_quantity_as_the_published_report(tmp_path):
    """Two properties in one pass, because each costs a SHAP run over ten thousand
    rows and asserting them separately doubles the suite's runtime for nothing.

    First: the batch frame carries raw applicant fields while the model's controls
    include transformed ones. If those are missing the regression quietly drops them
    and measures something else.

    Second: the resulting disparity has to be comparable to the figure in section 7,
    or the quarterly review and the validation pack would quote different numbers for
    the same thing.
    """
    from tally.batch.runner import run_demo_batch
    from tally.fairness.report import load_fairness_report
    from tally.ledger.store import LedgerStore
    from tally.model.registry import load_current

    result = run_demo_batch(n=10_000, ledger=LedgerStore(tmp_path / "d.db"))
    fairness = result["fairness"]

    assert fairness["status"] == "ok"
    assert fairness["n"] == result["row_count"]
    assert fairness["controls_missing"] == []

    controlled = fairness["controlled"]
    assert controlled["status"] == "ok"
    assert set(controlled["controls_used"]) == set(load_current().feature_names)

    try:
        published = load_fairness_report()
    except FileNotFoundError:
        pytest.skip("no fairness report — run: uv run tally fairness")

    focal = "Hispanic"
    reference = published["controlled"]["specifications"]["race"]["primary"]["groups"]
    if reference.get(focal, {}).get("status") != "ok":
        pytest.skip(f"{focal} not estimable on the published frame")
    batch = controlled["groups"].get(focal, {})
    if batch.get("status") != "ok":
        pytest.skip("batch regression did not converge at this sample size")

    # A random ten-thousand-row sample of the same population should land close to
    # the full frame; a large gap would mean the two are measuring different things.
    assert abs(batch["ame_pp"] - reference[focal]["ame_pp"]) < 1.5


@pytest.mark.slow
@pytest.mark.needs_model
def test_a_batch_without_demographics_says_so_rather_than_reporting_nothing(tmp_path):
    from tally.batch.runner import score_batch
    from tally.ledger.store import LedgerStore

    anonymous = {k: v for k, v in _row().items() if k not in CONTEXT_COLUMNS}
    parsed = parse_batch(pd.DataFrame([anonymous for _ in range(40)]))
    result = score_batch(parsed, ledger=LedgerStore(tmp_path / "d.db"), filename="anon.csv")

    assert result["fairness"]["status"] == "insufficient_demographics"
    assert "derived_race" in result["fairness"]["note"]
