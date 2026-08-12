"""Chain verification and decision replay.

Verification recomputes each row's hash from the bytes stored in SQLite — it never
re-serialises a parsed object. That is what makes "the chain verified" a statement
about the stored data rather than about the code's ability to reproduce its own
output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from tally.ledger.canonical import GENESIS_HASH, HASHED_FIELDS, row_hash
from tally.ledger.store import LedgerStore


@dataclass(frozen=True)
class ChainResult:
    ok: bool
    rows_checked: int
    head_hash: str | None
    first_broken_seq: int | None = None
    first_broken_reason: str | None = None
    first_broken_detail: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "rows_checked": self.rows_checked,
            "head_hash": self.head_hash,
            "first_broken_seq": self.first_broken_seq,
            "first_broken_reason": self.first_broken_reason,
            "first_broken_detail": self.first_broken_detail,
        }


def verify_chain(
    store: LedgerStore,
    *,
    after_seq: int = 0,
    expected_prev_hash: str | None = None,
    rows_already_checked: int = 0,
) -> ChainResult:
    """Walk the chain, recomputing every row hash and checking every link.

    By default this starts at genesis. The optional arguments allow verifying only a
    suffix, given the hash the suffix must link back to.

    That is sound rather than a shortcut: the chain property is inductive. If rows 1
    to N were verified and produced head H, and rows N+1 onward each verify and the
    first of them links to H, then rows 1 to M are verified. The caller is
    responsible for having established the prefix — see ServiceState.chain_status,
    which only ever extends a result it computed itself in this process.
    """
    expected_prev = expected_prev_hash or GENESIS_HASH
    expected_seq = after_seq + 1
    checked = rows_already_checked
    head: str | None = expected_prev_hash

    for row in store.iter_rows(after_seq=after_seq):
        seq = int(row["seq"])

        if seq != expected_seq:
            return ChainResult(
                ok=False,
                rows_checked=checked,
                head_hash=head,
                first_broken_seq=seq,
                first_broken_reason="seq_gap",
                first_broken_detail=f"expected seq {expected_seq}, found {seq}",
            )

        if row["prev_hash"] != expected_prev:
            return ChainResult(
                ok=False,
                rows_checked=checked,
                head_hash=head,
                first_broken_seq=seq,
                first_broken_reason="prev_hash_mismatch",
                first_broken_detail=(
                    f"row links to {row['prev_hash'][:16]}..., "
                    f"predecessor hashes to {expected_prev[:16]}..."
                ),
            )

        recomputed = row_hash({k: row[k] for k in HASHED_FIELDS})
        if recomputed != row["row_hash"]:
            return ChainResult(
                ok=False,
                rows_checked=checked,
                head_hash=head,
                first_broken_seq=seq,
                first_broken_reason="row_hash_mismatch",
                first_broken_detail=(
                    f"stored {row['row_hash'][:16]}..., recomputed {recomputed[:16]}... "
                    "— the row's contents were altered after it was written"
                ),
            )

        expected_prev = row["row_hash"]
        head = row["row_hash"]
        expected_seq = seq + 1
        checked += 1

    return ChainResult(ok=True, rows_checked=checked, head_hash=head)


@dataclass(frozen=True)
class ReplayResult:
    decision_id: str
    model_version: str
    model_hash_matches: bool
    stored_score: float
    recomputed_score: float
    abs_diff: float
    match: bool
    stored_decision: str
    recomputed_decision: str
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "model_version": self.model_version,
            "model_hash_matches": self.model_hash_matches,
            "stored_score": self.stored_score,
            "recomputed_score": self.recomputed_score,
            "abs_diff": self.abs_diff,
            "match": self.match,
            "stored_decision": self.stored_decision,
            "recomputed_decision": self.recomputed_decision,
            "detail": self.detail,
        }


REPLAY_TOLERANCE = 1e-9


def replay(store: LedgerStore, decision_id: str) -> ReplayResult:
    """Re-score a stored decision under the model version it was made with.

    This is the "explain applicant #88214 from March" path: nothing about the
    current model is used. The version named on the row is loaded, the stored input
    is fed back, and the score is compared.
    """
    from tally.data.features import RawApplicant
    from tally.model.registry import load_version
    from tally.serving.scoring import score_vector

    row = store.get(decision_id)
    if row is None:
        raise KeyError(f"no decision {decision_id!r}")

    model = load_version(row["model_version"])
    model_hash_matches = model.model_hash == row["model_hash"]

    payload = json.loads(row["input_json"])
    # canonical_json stores numbers as strings; Pydantic coerces them back.
    applicant = RawApplicant.model_validate(payload)
    recomputed = score_vector(applicant, model)

    stored = float(row["raw_score"])
    diff = abs(stored - recomputed.probability)
    match = bool(diff <= REPLAY_TOLERANCE and model_hash_matches)

    if not model_hash_matches:
        detail = (
            f"model {row['model_version']} on disk hashes to {model.model_hash[:16]}..., "
            f"the decision was made with {row['model_hash'][:16]}..."
        )
    elif match:
        detail = f"score reproduced exactly under model {row['model_version']}"
    else:
        detail = f"score differs by {diff:.3e}, above the {REPLAY_TOLERANCE:.0e} tolerance"

    return ReplayResult(
        decision_id=decision_id,
        model_version=row["model_version"],
        model_hash_matches=model_hash_matches,
        stored_score=stored,
        recomputed_score=float(recomputed.probability),
        abs_diff=float(diff),
        match=match,
        stored_decision=row["decision"],
        recomputed_decision=recomputed.decision,
        detail=detail,
    )


def _fmt(x: float) -> str:
    return "0.0e+00" if x == 0 else f"{x:.1e}"


def format_diff(x: float) -> str:
    """Display form for the replay difference. Exact zero should read as exact."""
    return _fmt(float(np.abs(x)))
