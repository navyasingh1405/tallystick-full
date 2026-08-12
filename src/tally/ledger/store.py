"""Append-only decision store.

There is no update path and no delete path in this module, by design. Corrections
are new rows carrying `supersedes`.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tally.config import get_config
from tally.ledger.canonical import (
    GENESIS_HASH,
    HASHED_FIELDS,
    canon_num,
    canonical_json,
    row_hash,
)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def utc_now() -> str:
    """ISO-8601 with microseconds and an explicit Z. Fixed width, sorts correctly."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


@dataclass
class DecisionRecord:
    """A decision waiting to be written."""

    model_version: str
    model_hash: str
    input_payload: dict[str, Any]
    raw_score: float
    threshold: float
    decision: str
    reason_codes: list[dict[str, Any]] = field(default_factory=list)
    supersedes: str | None = None
    override_decision: str | None = None
    override_actor: str | None = None
    override_reason: str | None = None
    override_at: str | None = None
    batch_id: str | None = None
    decision_id: str | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class WrittenDecision:
    seq: int
    decision_id: str
    created_at: str
    decision: str
    raw_score: float
    prev_hash: str
    row_hash: str
    supersedes: str | None
    batch_id: str | None


@dataclass(frozen=True)
class BatchSummary:
    batch_id: str
    row_count: int
    first_seq: int
    last_seq: int
    batch_root_hash: str
    created_at: str


class LedgerStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @classmethod
    def default(cls) -> LedgerStore:
        return cls(get_config().paths.ledger_db)

    # -- connection -------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA busy_timeout = 30000")
        return con

    def _init_schema(self) -> None:
        con = self._connect()
        try:
            con.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        finally:
            con.close()

    # -- writing ----------------------------------------------------------
    def append(self, record: DecisionRecord) -> WrittenDecision:
        written = self.append_many([record])
        return written[0]

    def append_many(self, records: Sequence[DecisionRecord]) -> list[WrittenDecision]:
        """Write N records as one transaction, chaining hashes forward in memory.

        Hashing is cheap; the transaction is not. One commit for the whole batch is
        what makes a ten-thousand-row portfolio review finish in seconds rather than
        minutes, without weakening the chain.
        """
        if not records:
            return []

        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            prev = self._head_hash(con)
            rows: list[dict[str, Any]] = []
            written: list[WrittenDecision] = []

            for rec in records:
                row = self._to_row(rec, prev)
                rows.append(row)
                prev = row["row_hash"]

            con.executemany(
                """
                INSERT INTO decisions (
                    decision_id, created_at, model_version, model_hash, input_json,
                    raw_score, raw_score_txt, threshold, threshold_txt, decision,
                    reason_codes_json, supersedes, override_decision, override_actor,
                    override_reason, override_at, batch_id, prev_hash, row_hash
                ) VALUES (
                    :decision_id, :created_at, :model_version, :model_hash, :input_json,
                    :raw_score, :raw_score_txt, :threshold, :threshold_txt, :decision,
                    :reason_codes_json, :supersedes, :override_decision, :override_actor,
                    :override_reason, :override_at, :batch_id, :prev_hash, :row_hash
                )
                """,
                rows,
            )

            last_seq = con.execute("SELECT last_insert_rowid()").fetchone()[0]
            first_seq = last_seq - len(rows) + 1
            con.execute("COMMIT")

            for offset, row in enumerate(rows):
                written.append(
                    WrittenDecision(
                        seq=first_seq + offset,
                        decision_id=row["decision_id"],
                        created_at=row["created_at"],
                        decision=row["decision"],
                        raw_score=row["raw_score"],
                        prev_hash=row["prev_hash"],
                        row_hash=row["row_hash"],
                        supersedes=row["supersedes"],
                        batch_id=row["batch_id"],
                    )
                )
            return written
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.close()

    def append_batch(
        self, records: Sequence[DecisionRecord], *, filename: str | None = None
    ) -> BatchSummary:
        """Write a portfolio in one transaction and record a batch receipt."""
        batch_id = str(uuid.uuid4())
        for rec in records:
            rec.batch_id = batch_id

        written = self.append_many(records)
        summary = BatchSummary(
            batch_id=batch_id,
            row_count=len(written),
            first_seq=written[0].seq,
            last_seq=written[-1].seq,
            batch_root_hash=written[-1].row_hash,
            created_at=written[0].created_at,
        )

        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO batches (batch_id, created_at, filename, row_count,
                                     first_seq, last_seq, batch_root_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.batch_id,
                    summary.created_at,
                    filename,
                    summary.row_count,
                    summary.first_seq,
                    summary.last_seq,
                    summary.batch_root_hash,
                ),
            )
        finally:
            con.close()
        return summary

    def override(
        self, decision_id: str, *, decision: str, actor: str, reason: str
    ) -> WrittenDecision:
        """Record a manual override as a NEW row referencing the original.

        The original is never modified. Both rows remain in the chain, which is the
        point: an examiner can see the machine decision, the human decision, who
        made it, and why.
        """
        original = self.get(decision_id)
        if original is None:
            raise KeyError(f"no decision {decision_id!r}")
        if decision not in ("APPROVED", "DECLINED"):
            raise ValueError("decision must be APPROVED or DECLINED")
        if not reason.strip():
            raise ValueError("an override requires a documented reason")

        now = utc_now()
        record = DecisionRecord(
            model_version=original["model_version"],
            model_hash=original["model_hash"],
            input_payload=_loads(original["input_json"]),
            raw_score=float(original["raw_score"]),
            threshold=float(original["threshold"]),
            decision=decision,
            reason_codes=_loads(original["reason_codes_json"]),
            supersedes=decision_id,
            override_decision=decision,
            override_actor=actor,
            override_reason=reason,
            override_at=now,
            created_at=now,
        )
        return self.append(record)

    # -- reading ----------------------------------------------------------
    def get(self, decision_id: str) -> sqlite3.Row | None:
        con = self._connect()
        try:
            return con.execute(
                "SELECT * FROM decisions WHERE decision_id = ?", (decision_id,)
            ).fetchone()
        finally:
            con.close()

    def recent(self, limit: int = 50) -> list[sqlite3.Row]:
        con = self._connect()
        try:
            return list(
                con.execute("SELECT * FROM decisions ORDER BY seq DESC LIMIT ?", (limit,))
            )
        finally:
            con.close()

    def neighbours(self, decision_id: str) -> dict[str, sqlite3.Row | None]:
        """The row plus its predecessor and successor, for the chain visualisation."""
        con = self._connect()
        try:
            row = con.execute(
                "SELECT * FROM decisions WHERE decision_id = ?", (decision_id,)
            ).fetchone()
            if row is None:
                return {"previous": None, "current": None, "next": None}
            seq = row["seq"]
            return {
                "previous": con.execute(
                    "SELECT * FROM decisions WHERE seq < ? ORDER BY seq DESC LIMIT 1", (seq,)
                ).fetchone(),
                "current": row,
                "next": con.execute(
                    "SELECT * FROM decisions WHERE seq > ? ORDER BY seq ASC LIMIT 1", (seq,)
                ).fetchone(),
            }
        finally:
            con.close()

    def all_rows(self) -> list[sqlite3.Row]:
        con = self._connect()
        try:
            return list(con.execute("SELECT * FROM decisions ORDER BY seq"))
        finally:
            con.close()

    def iter_rows(self, chunk: int = 5000, *, after_seq: int = 0) -> Iterator[sqlite3.Row]:
        """Stream in sequence order. verify_chain walks the ledger this way.

        `after_seq` lets a caller verify only the rows appended since a known point.
        """
        con = self._connect()
        try:
            cur = con.execute(
                "SELECT * FROM decisions WHERE seq > ? ORDER BY seq", (after_seq,)
            )
            while batch := cur.fetchmany(chunk):
                yield from batch
        finally:
            con.close()

    def batch_rows(self, batch_id: str) -> list[sqlite3.Row]:
        con = self._connect()
        try:
            return list(
                con.execute(
                    "SELECT * FROM decisions WHERE batch_id = ? ORDER BY seq", (batch_id,)
                )
            )
        finally:
            con.close()

    def batches(self, limit: int = 25) -> list[sqlite3.Row]:
        con = self._connect()
        try:
            return list(
                con.execute("SELECT * FROM batches ORDER BY created_at DESC LIMIT ?", (limit,))
            )
        finally:
            con.close()

    def superseded_by(self, decision_id: str) -> str | None:
        con = self._connect()
        try:
            row = con.execute(
                "SELECT decision_id FROM decisions WHERE supersedes = ? ORDER BY seq LIMIT 1",
                (decision_id,),
            ).fetchone()
            return row["decision_id"] if row else None
        finally:
            con.close()

    def height(self) -> int:
        con = self._connect()
        try:
            return int(con.execute("SELECT count(*) FROM decisions").fetchone()[0])
        finally:
            con.close()

    def head_hash(self) -> str:
        con = self._connect()
        try:
            return self._head_hash(con)
        finally:
            con.close()

    # -- internals --------------------------------------------------------
    @staticmethod
    def _head_hash(con: sqlite3.Connection) -> str:
        row = con.execute("SELECT row_hash FROM decisions ORDER BY seq DESC LIMIT 1").fetchone()
        return row["row_hash"] if row else GENESIS_HASH

    @staticmethod
    def _to_row(rec: DecisionRecord, prev_hash: str) -> dict[str, Any]:
        raw_score = float(rec.raw_score)
        threshold = float(rec.threshold)
        row: dict[str, Any] = {
            "decision_id": rec.decision_id or str(uuid.uuid4()),
            "created_at": rec.created_at or utc_now(),
            "model_version": rec.model_version,
            "model_hash": rec.model_hash,
            "input_json": canonical_json(rec.input_payload),
            "raw_score": raw_score,
            "raw_score_txt": canon_num(raw_score),
            "threshold": threshold,
            "threshold_txt": canon_num(threshold),
            "decision": rec.decision,
            "reason_codes_json": canonical_json(rec.reason_codes),
            "supersedes": rec.supersedes,
            "override_decision": rec.override_decision,
            "override_actor": rec.override_actor,
            "override_reason": rec.override_reason,
            "override_at": rec.override_at,
            "batch_id": rec.batch_id,
            "prev_hash": prev_hash,
        }
        row["row_hash"] = row_hash({k: row.get(k) for k in HASHED_FIELDS})
        return row


def _loads(text: str) -> Any:
    import json

    return json.loads(text)
