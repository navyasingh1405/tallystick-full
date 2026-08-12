-- Accounts and sessions.
--
-- Deliberately a different database from the decision ledger. The ledger is
-- append-only, enforced by triggers, and chained by hash; user records must be
-- updatable (password changes, last login, deactivation) and must be deletable on
-- request. Putting them in the same file would either weaken the ledger's guarantees
-- or make ordinary account management impossible.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id             TEXT PRIMARY KEY,
    email          TEXT NOT NULL UNIQUE,       -- stored lowercased and trimmed
    password_hash  TEXT NOT NULL,              -- argon2id; never leaves the process
    full_name      TEXT,
    organisation   TEXT,
    role           TEXT NOT NULL DEFAULT 'demo'
                        CHECK (role IN ('demo', 'analyst', 'admin')),
    is_active      INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL,
    last_login_at  TEXT
);

CREATE INDEX IF NOT EXISTS ix_users_email ON users (email);

CREATE TABLE IF NOT EXISTS sessions (
    -- The SHA-256 of the session token, never the token itself. A read of this
    -- table does not let an attacker mint a usable cookie.
    token_hash   TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    created_at   TEXT NOT NULL,
    expires_at   TEXT NOT NULL,
    user_agent   TEXT
);

CREATE INDEX IF NOT EXISTS ix_sessions_user    ON sessions (user_id);
CREATE INDEX IF NOT EXISTS ix_sessions_expires ON sessions (expires_at);

-- Failed sign-in attempts, for rate limiting. Recording the email as given (after
-- normalisation) is necessary to throttle credential stuffing against one account;
-- passwords are never recorded, in any form.
CREATE TABLE IF NOT EXISTS login_attempts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    email       TEXT NOT NULL,
    client_ip   TEXT,
    attempted_at TEXT NOT NULL,
    succeeded   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_attempts_email ON login_attempts (email, attempted_at);
CREATE INDEX IF NOT EXISTS ix_attempts_ip    ON login_attempts (client_ip, attempted_at);
