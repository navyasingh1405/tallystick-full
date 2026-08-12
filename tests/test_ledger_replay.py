"""Replay: "explain applicant #88214 from March".

Load the model version pinned on the stored decision, feed back the stored input,
and confirm the score reproduces. This is the acceptance criterion that makes the
ledger more than a log.
"""

from __future__ import annotations

import pytest

from tally.ledger.verify import replay, verify_chain

REPLAY_TOLERANCE = 1e-9


@pytest.mark.slow
@pytest.mark.needs_model
def test_a_recorded_decision_rescores_to_the_stored_value(tmp_ledger, model, eval_frame):
    from tally.data.features import applicant_from_frame_row
    from tally.serving.scoring import score_applicant

    for _, row in eval_frame.head(25).iterrows():
        applicant = applicant_from_frame_row(row)
        result = score_applicant(applicant, model=model, ledger=tmp_ledger)

        rp = replay(tmp_ledger, result.decision_id)
        assert rp.match, rp.detail
        assert rp.abs_diff <= REPLAY_TOLERANCE
        assert rp.model_version == model.version
        assert rp.stored_decision == rp.recomputed_decision


@pytest.mark.slow
@pytest.mark.needs_model
def test_replay_uses_the_pinned_version_not_whatever_is_current(tmp_ledger, model):
    """The stored decision names a version; replay must load that one."""
    from tally.data.features import RawApplicant
    from tally.serving.scoring import score_applicant

    applicant = RawApplicant(
        income_usd=95_000, loan_amount_usd=420_000, cltv=95, dti_band="47", loan_term=360
    )
    result = score_applicant(applicant, model=model, ledger=tmp_ledger)
    stored = tmp_ledger.get(result.decision_id)
    assert stored["model_version"] == model.version
    assert stored["model_hash"] == model.model_hash

    rp = replay(tmp_ledger, result.decision_id)
    assert rp.model_version == stored["model_version"]
    assert rp.model_hash_matches


@pytest.mark.slow
@pytest.mark.needs_model
def test_replay_detects_a_tampered_input(tmp_ledger, model, tmp_path):
    import shutil
    import sqlite3

    from tally.data.features import RawApplicant
    from tally.ledger.canonical import canonical_json
    from tally.ledger.store import LedgerStore
    from tally.serving.scoring import score_applicant

    applicant = RawApplicant(
        income_usd=150_000, loan_amount_usd=400_000, cltv=80, dti_band="38", loan_term=360
    )
    result = score_applicant(applicant, model=model, ledger=tmp_ledger)

    copy = tmp_path / "c.db"
    shutil.copy(tmp_ledger.path, copy)
    con = sqlite3.connect(copy)
    con.execute("DROP TRIGGER decisions_no_update")
    con.execute(
        "UPDATE decisions SET input_json = ? WHERE decision_id = ?",
        (canonical_json({**applicant.model_dump(), "income_usd": 15_000.0}), result.decision_id),
    )
    con.commit()
    con.close()

    tampered = LedgerStore(copy)
    # Both signals fire: the chain no longer verifies, and the score no longer
    # reproduces. Either alone would be enough; both together is the point.
    assert not verify_chain(tampered).ok
    assert not replay(tampered, result.decision_id).match


@pytest.mark.slow
@pytest.mark.needs_model
def test_replay_of_a_missing_decision_raises(tmp_ledger):
    with pytest.raises(KeyError):
        replay(tmp_ledger, "00000000-0000-0000-0000-000000000000")
