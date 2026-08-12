"""Append-only enforcement, chain verification, and tamper localisation.

Written before the implementation.
"""

from __future__ import annotations

import shutil
import sqlite3

import pytest

from tally.ledger.canonical import GENESIS_HASH, canonical_json
from tally.ledger.store import DecisionRecord, LedgerStore
from tally.ledger.verify import verify_chain


def make_record(i: int, **over) -> DecisionRecord:
    kwargs = dict(
        model_version="v001",
        model_hash="a" * 64,
        input_payload={"income_usd": 100000.0 + i, "cltv": 80.0},
        raw_score=0.4 + (i % 10) / 100,
        threshold=0.5,
        decision="DECLINED" if i % 3 else "APPROVED",
        reason_codes=[{"rank": 1, "code": "DTI_HIGH"}] if i % 3 else [],
    )
    kwargs.update(over)
    return DecisionRecord(**kwargs)


def fill(store: LedgerStore, n: int) -> list[str]:
    return [store.append(make_record(i)).decision_id for i in range(n)]


# --------------------------------------------------------------------------- triggers


def test_update_is_refused_by_the_database(tmp_ledger):
    fill(tmp_ledger, 3)
    with sqlite3.connect(tmp_ledger.path) as con:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute("UPDATE decisions SET decision = 'APPROVED'")


def test_delete_is_refused_by_the_database(tmp_ledger):
    fill(tmp_ledger, 3)
    with sqlite3.connect(tmp_ledger.path) as con:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute("DELETE FROM decisions")


def test_a_single_row_update_is_refused_too(tmp_ledger):
    ids = fill(tmp_ledger, 3)
    with sqlite3.connect(tmp_ledger.path) as con:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute("UPDATE decisions SET raw_score = 0.99 WHERE decision_id = ?", (ids[1],))


# --------------------------------------------------------------------------- chain


def test_empty_ledger_verifies_clean(tmp_ledger):
    result = verify_chain(tmp_ledger)
    assert result.ok
    assert result.rows_checked == 0


def test_genesis_row_links_to_the_zero_hash(tmp_ledger):
    rec = tmp_ledger.append(make_record(0))
    assert rec.prev_hash == GENESIS_HASH
    assert rec.seq == 1


def test_thousand_row_chain_verifies_clean(tmp_ledger):
    fill(tmp_ledger, 1000)
    result = verify_chain(tmp_ledger)
    assert result.ok, result.first_broken_reason
    assert result.rows_checked == 1000
    assert result.first_broken_seq is None


def test_each_row_links_to_its_predecessor(tmp_ledger):
    fill(tmp_ledger, 20)
    rows = tmp_ledger.all_rows()
    for prev, cur in zip(rows, rows[1:], strict=False):
        assert cur["prev_hash"] == prev["row_hash"]


# --------------------------------------------------------------------------- tampering


def _drop_triggers_and_corrupt(path, seq: int, column: str, value):
    """An attacker with file access can drop the triggers. The hash chain is what
    detects that; the triggers only stop accidents."""
    con = sqlite3.connect(path)
    con.execute("DROP TRIGGER decisions_no_update")
    con.execute(f"UPDATE decisions SET {column} = ? WHERE seq = ?", (value, seq))
    con.commit()
    con.close()


def test_corrupting_one_input_identifies_that_exact_row(tmp_ledger, tmp_path):
    fill(tmp_ledger, 1000)
    copy = tmp_path / "corrupt.db"
    shutil.copy(tmp_ledger.path, copy)

    _drop_triggers_and_corrupt(copy, 500, "input_json", canonical_json({"income_usd": 1.0}))

    result = verify_chain(LedgerStore(copy))
    assert not result.ok
    assert result.first_broken_seq == 500
    assert result.first_broken_reason == "row_hash_mismatch"


def test_corrupting_the_score_is_detected(tmp_ledger, tmp_path):
    fill(tmp_ledger, 50)
    copy = tmp_path / "c.db"
    shutil.copy(tmp_ledger.path, copy)
    _drop_triggers_and_corrupt(copy, 7, "raw_score_txt", "0.99")
    result = verify_chain(LedgerStore(copy))
    assert not result.ok
    assert result.first_broken_seq == 7


def test_relinking_a_row_is_detected_as_a_broken_link(tmp_ledger, tmp_path):
    """Rewriting prev_hash to something else breaks the linkage even if the row's own
    hash is recomputed consistently."""
    fill(tmp_ledger, 20)
    copy = tmp_path / "c.db"
    shutil.copy(tmp_ledger.path, copy)
    _drop_triggers_and_corrupt(copy, 10, "prev_hash", "f" * 64)
    result = verify_chain(LedgerStore(copy))
    assert not result.ok
    assert result.first_broken_seq == 10


def test_suffix_verification_matches_a_full_walk(tmp_ledger):
    """The chain property is inductive, so verifying a suffix from a known head is
    equivalent to walking from genesis. The service relies on this to keep the status
    poll cheap on a large ledger."""
    fill(tmp_ledger, 200)
    prefix = verify_chain(tmp_ledger)
    fill(tmp_ledger, 50)

    full = verify_chain(tmp_ledger)
    incremental = verify_chain(
        tmp_ledger,
        after_seq=prefix.rows_checked,
        expected_prev_hash=prefix.head_hash,
        rows_already_checked=prefix.rows_checked,
    )
    assert incremental.ok == full.ok
    assert incremental.rows_checked == full.rows_checked == 250
    assert incremental.head_hash == full.head_hash


def test_suffix_verification_detects_a_break_in_the_new_rows(tmp_ledger, tmp_path):
    fill(tmp_ledger, 100)
    prefix = verify_chain(tmp_ledger)
    fill(tmp_ledger, 20)

    copy = tmp_path / "c.db"
    shutil.copy(tmp_ledger.path, copy)
    _drop_triggers_and_corrupt(copy, 110, "input_json", "{}")

    result = verify_chain(
        LedgerStore(copy),
        after_seq=prefix.rows_checked,
        expected_prev_hash=prefix.head_hash,
        rows_already_checked=prefix.rows_checked,
    )
    assert not result.ok
    assert result.first_broken_seq == 110


def test_suffix_verification_detects_a_suffix_that_does_not_link_back(tmp_ledger):
    """If the suffix does not link to the head it was told to expect, that is a break
    even though every row in the suffix is internally consistent."""
    fill(tmp_ledger, 20)
    result = verify_chain(
        tmp_ledger, after_seq=10, expected_prev_hash="f" * 64, rows_already_checked=10
    )
    assert not result.ok
    assert result.first_broken_seq == 11
    assert result.first_broken_reason == "prev_hash_mismatch"


def test_verification_reports_the_first_break_not_a_later_one(tmp_ledger, tmp_path):
    fill(tmp_ledger, 100)
    copy = tmp_path / "c.db"
    shutil.copy(tmp_ledger.path, copy)
    con = sqlite3.connect(copy)
    con.execute("DROP TRIGGER decisions_no_update")
    con.execute("UPDATE decisions SET input_json = '{}' WHERE seq IN (30, 60)")
    con.commit()
    con.close()
    result = verify_chain(LedgerStore(copy))
    assert result.first_broken_seq == 30


# --------------------------------------------------------------------------- overrides


def test_an_override_is_a_new_row_not_an_edit(tmp_ledger):
    original = tmp_ledger.append(make_record(0, decision="DECLINED"))
    override = tmp_ledger.override(
        original.decision_id,
        decision="APPROVED",
        actor="credit.officer@example.com",
        reason="Documented compensating factors reviewed manually.",
    )
    assert override.decision_id != original.decision_id
    assert override.supersedes == original.decision_id
    assert override.decision == "APPROVED"
    assert tmp_ledger.height() == 2
    # The original is untouched.
    assert tmp_ledger.get(original.decision_id)["decision"] == "DECLINED"
    assert verify_chain(tmp_ledger).ok


def test_overriding_a_missing_decision_raises(tmp_ledger):
    with pytest.raises(KeyError):
        tmp_ledger.override("no-such-id", decision="APPROVED", actor="x", reason="y")


def test_superseded_chain_is_reported(tmp_ledger):
    a = tmp_ledger.append(make_record(0, decision="DECLINED"))
    b = tmp_ledger.override(a.decision_id, decision="APPROVED", actor="x", reason="documented")
    assert tmp_ledger.superseded_by(a.decision_id) == b.decision_id
    assert tmp_ledger.superseded_by(b.decision_id) is None


# --------------------------------------------------------------------------- batch


def test_batch_write_preserves_the_chain(tmp_ledger):
    tmp_ledger.append(make_record(0))
    records = [make_record(i) for i in range(1, 501)]
    summary = tmp_ledger.append_batch(records, filename="portfolio.csv")

    assert summary.row_count == 500
    assert tmp_ledger.height() == 501
    result = verify_chain(tmp_ledger)
    assert result.ok, result.first_broken_reason
    assert summary.batch_root_hash == result.head_hash


def test_batch_rows_carry_the_batch_id(tmp_ledger):
    records = [make_record(i) for i in range(10)]
    summary = tmp_ledger.append_batch(records, filename="p.csv")
    rows = tmp_ledger.batch_rows(summary.batch_id)
    assert len(rows) == 10
    assert all(r["batch_id"] == summary.batch_id for r in rows)


def test_batch_and_single_writes_interleave_without_breaking_the_chain(tmp_ledger):
    tmp_ledger.append(make_record(0))
    tmp_ledger.append_batch([make_record(i) for i in range(1, 21)], filename="a.csv")
    tmp_ledger.append(make_record(99))
    tmp_ledger.append_batch([make_record(i) for i in range(21, 31)], filename="b.csv")
    assert verify_chain(tmp_ledger).ok
    assert tmp_ledger.height() == 32
