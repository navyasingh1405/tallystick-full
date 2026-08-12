-- Decision ledger. Append-only, hash-chained.
--
-- `seq` rather than `decision_id` is the primary key because the chain needs a
-- deterministic order and a uuid4 provides none. `decision_id` is the natural key
-- and is unique.
--
-- Numeric values that participate in the hash are stored twice: once as REAL so
-- the column is queryable, once as TEXT in canonical form so that what is hashed
-- is exactly what is stored.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS decisions (
    seq                INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id        TEXT    NOT NULL UNIQUE,
    created_at         TEXT    NOT NULL,
    model_version      TEXT    NOT NULL,
    model_hash         TEXT    NOT NULL,
    input_json         TEXT    NOT NULL,
    -- The decision score: P(approve). The threshold is expressed on probability,
    -- so this is the quantity the decision actually compares against.
    raw_score          REAL    NOT NULL,
    raw_score_txt      TEXT    NOT NULL,
    threshold          REAL    NOT NULL,
    threshold_txt      TEXT    NOT NULL,
    decision           TEXT    NOT NULL CHECK (decision IN ('APPROVED', 'DECLINED')),
    reason_codes_json  TEXT    NOT NULL,
    supersedes         TEXT             REFERENCES decisions (decision_id),
    override_decision  TEXT,
    override_actor     TEXT,
    override_reason    TEXT,
    override_at        TEXT,
    batch_id           TEXT,
    prev_hash          TEXT    NOT NULL,
    row_hash           TEXT    NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS ix_decisions_created ON decisions (created_at);
CREATE INDEX IF NOT EXISTS ix_decisions_batch   ON decisions (batch_id);
CREATE INDEX IF NOT EXISTS ix_decisions_super   ON decisions (supersedes);

CREATE TABLE IF NOT EXISTS batches (
    batch_id         TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    filename         TEXT,
    row_count        INTEGER NOT NULL,
    first_seq        INTEGER,
    last_seq         INTEGER,
    -- row_hash of the batch's last row: a one-line receipt for the whole batch.
    batch_root_hash  TEXT NOT NULL
);

-- Append-only, enforced by the database rather than by convention.
--
-- These stop accidents and ordinary application bugs. They do not stop
-- DROP TRIGGER, PRAGMA writable_schema, or editing the file directly — the hash
-- chain is what detects those. Neither stops an adversary who can also recompute
-- every subsequent row_hash; that would need the head hash anchored to an external
-- witness, which is out of scope and stated as such in docs/limitations.md.
CREATE TRIGGER IF NOT EXISTS decisions_no_update
BEFORE UPDATE ON decisions
BEGIN
    SELECT RAISE(ABORT, 'decisions is append-only');
END;

CREATE TRIGGER IF NOT EXISTS decisions_no_delete
BEFORE DELETE ON decisions
BEGIN
    SELECT RAISE(ABORT, 'decisions is append-only');
END;
