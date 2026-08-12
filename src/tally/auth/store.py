"""Accounts, sessions, and sign-in throttling.

Session design worth stating plainly, because the alternative is common and worse:
the cookie carries an opaque random token, and the database stores only its SHA-256.
A signed cookie containing the user's identity (the usual `SessionMiddleware`
approach) puts the session state on the client, cannot be revoked, and leaks whatever
it carries to anyone who reads the cookie jar. Server-side sessions can be revoked
individually, expire server-side, and survive a database read without granting
anybody a usable cookie.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tally.auth import passwords
from tally.config import get_config

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

ROLES = ("demo", "analyst", "admin")

#: Actions each role may take. Everything not listed is denied.
PERMISSIONS: dict[str, frozenset[str]] = {
    "demo": frozenset({"score", "read", "verify", "batch", "pack", "adopt", "override"}),
    "analyst": frozenset({"score", "read", "verify", "batch", "pack", "adopt", "override"}),
    "admin": frozenset(
        {"score", "read", "verify", "batch", "pack", "adopt", "override", "admin"}
    ),
}


class AuthError(RuntimeError):
    """Base class for authentication failures."""


class EmailAlreadyRegistered(AuthError):
    pass


class InvalidCredentials(AuthError):
    """Deliberately raised for both 'no such account' and 'wrong password'."""


class AccountDisabled(AuthError):
    pass


class RateLimited(AuthError):
    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__(
            f"Too many sign-in attempts. Try again in {retry_after_seconds} seconds."
        )
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class User:
    id: str
    email: str
    full_name: str | None
    organisation: str | None
    role: str
    is_active: bool
    created_at: str
    last_login_at: str | None

    def can(self, action: str) -> bool:
        return action in PERMISSIONS.get(self.role, frozenset())

    def as_dict(self) -> dict[str, Any]:
        """Safe to serialise. There is no field here that could carry a secret."""
        return {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name,
            "organisation": self.organisation,
            "role": self.role,
            "is_demo": self.role == "demo",
            "created_at": self.created_at,
            "last_login_at": self.last_login_at,
            "permissions": sorted(PERMISSIONS.get(self.role, frozenset())),
        }


@dataclass(frozen=True)
class Session:
    token: str  # returned once, at creation; never stored
    user: User
    expires_at: str


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _parse(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


def normalise_email(email: str) -> str:
    return email.strip().lower()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @classmethod
    def default(cls) -> AuthStore:
        cfg = get_config()
        return cls(cfg.paths.root / "ledger" / "accounts.db")

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

    # -- accounts ---------------------------------------------------------
    def create_user(
        self,
        email: str,
        password: str,
        *,
        full_name: str | None = None,
        organisation: str | None = None,
        role: str = "demo",
    ) -> User:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")

        address = normalise_email(email)
        if "@" not in address or "." not in address.split("@")[-1]:
            raise ValueError("that does not look like an email address")

        # Hash before touching the database: a policy failure should not leave a
        # half-created account behind.
        digest = passwords.hash_password(password, email=address)

        con = self._connect()
        try:
            user_id = str(uuid.uuid4())
            con.execute(
                """
                INSERT INTO users (id, email, password_hash, full_name, organisation,
                                   role, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (user_id, address, digest, full_name, organisation, role, _iso(_now())),
            )
        except sqlite3.IntegrityError as exc:
            raise EmailAlreadyRegistered("that email is already registered") from exc
        finally:
            con.close()

        user = self.get_user_by_email(address)
        assert user is not None
        return user

    def get_user_by_email(self, email: str) -> User | None:
        con = self._connect()
        try:
            row = con.execute(
                "SELECT * FROM users WHERE email = ?", (normalise_email(email),)
            ).fetchone()
        finally:
            con.close()
        return _to_user(row) if row else None

    def get_user(self, user_id: str) -> User | None:
        con = self._connect()
        try:
            row = con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        finally:
            con.close()
        return _to_user(row) if row else None

    def list_users(self) -> list[User]:
        con = self._connect()
        try:
            return [_to_user(r) for r in con.execute("SELECT * FROM users ORDER BY created_at")]
        finally:
            con.close()

    def count_users(self) -> int:
        con = self._connect()
        try:
            return int(con.execute("SELECT count(*) FROM users").fetchone()[0])
        finally:
            con.close()

    def set_role(self, email: str, role: str) -> None:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        con = self._connect()
        try:
            con.execute(
                "UPDATE users SET role = ? WHERE email = ?", (role, normalise_email(email))
            )
        finally:
            con.close()

    def set_active(self, email: str, active: bool) -> None:
        con = self._connect()
        try:
            con.execute(
                "UPDATE users SET is_active = ? WHERE email = ?",
                (1 if active else 0, normalise_email(email)),
            )
        finally:
            con.close()

    def change_password(self, email: str, new_password: str) -> None:
        address = normalise_email(email)
        digest = passwords.hash_password(new_password, email=address)
        con = self._connect()
        try:
            con.execute(
                "UPDATE users SET password_hash = ? WHERE email = ?", (digest, address)
            )
            # Every existing session for this account is invalidated. A password
            # change that leaves old sessions alive is not a password change.
            con.execute(
                "DELETE FROM sessions WHERE user_id = (SELECT id FROM users WHERE email = ?)",
                (address,),
            )
        finally:
            con.close()

    # -- sign in ----------------------------------------------------------
    def authenticate(
        self, email: str, password: str, *, client_ip: str | None = None
    ) -> User:
        """Verify credentials, or raise. Never reveals whether the account exists."""
        address = normalise_email(email)
        self._enforce_rate_limit(address, client_ip)

        con = self._connect()
        try:
            row = con.execute("SELECT * FROM users WHERE email = ?", (address,)).fetchone()
        finally:
            con.close()

        if row is None:
            # Spend comparable CPU so the response time does not disclose that the
            # account is unknown.
            passwords.dummy_verify()
            self._record_attempt(address, client_ip, succeeded=False)
            raise InvalidCredentials("email or password is incorrect")

        if not passwords.verify_password(row["password_hash"], password):
            self._record_attempt(address, client_ip, succeeded=False)
            raise InvalidCredentials("email or password is incorrect")

        if not row["is_active"]:
            self._record_attempt(address, client_ip, succeeded=False)
            raise AccountDisabled("this account has been deactivated")

        if passwords.needs_rehash(row["password_hash"]):
            self._rehash(row["id"], password, address)

        self._record_attempt(address, client_ip, succeeded=True)
        self._touch_login(row["id"])
        user = self.get_user(row["id"])
        assert user is not None
        return user

    def _rehash(self, user_id: str, password: str, email: str) -> None:
        con = self._connect()
        try:
            con.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (passwords.hash_password(password, email=email), user_id),
            )
        finally:
            con.close()

    def _touch_login(self, user_id: str) -> None:
        con = self._connect()
        try:
            con.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (_iso(_now()), user_id))
        finally:
            con.close()

    # -- throttling -------------------------------------------------------
    def _enforce_rate_limit(self, email: str, client_ip: str | None) -> None:
        cfg = get_config().auth
        window = int(cfg["lockout_window_seconds"])
        limit = int(cfg["max_failed_attempts"])
        since = _iso(_now() - timedelta(seconds=window))

        con = self._connect()
        try:
            by_email = con.execute(
                """
                SELECT count(*) FROM login_attempts
                WHERE email = ? AND attempted_at > ? AND succeeded = 0
                """,
                (email, since),
            ).fetchone()[0]
            by_ip = 0
            if client_ip:
                by_ip = con.execute(
                    """
                    SELECT count(*) FROM login_attempts
                    WHERE client_ip = ? AND attempted_at > ? AND succeeded = 0
                    """,
                    (client_ip, since),
                ).fetchone()[0]
        finally:
            con.close()

        # Throttle per account and per source. Per-account alone lets one host walk
        # a list of addresses; per-source alone lets a distributed attempt through.
        if by_email >= limit or by_ip >= limit * 3:
            raise RateLimited(window)

    def _record_attempt(self, email: str, client_ip: str | None, *, succeeded: bool) -> None:
        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO login_attempts (email, client_ip, attempted_at, succeeded)
                VALUES (?, ?, ?, ?)
                """,
                (email, client_ip, _iso(_now()), 1 if succeeded else 0),
            )
            if succeeded:
                # A successful sign-in clears the account's failure budget.
                con.execute(
                    "DELETE FROM login_attempts WHERE email = ? AND succeeded = 0", (email,)
                )
        finally:
            con.close()

    # -- sessions ---------------------------------------------------------
    def start_session(self, user: User, *, user_agent: str | None = None) -> Session:
        cfg = get_config().auth
        token = secrets.token_urlsafe(32)
        expires = _now() + timedelta(hours=int(cfg["session_hours"]))

        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO sessions (token_hash, user_id, created_at, expires_at, user_agent)
                VALUES (?, ?, ?, ?, ?)
                """,
                (hash_token(token), user.id, _iso(_now()), _iso(expires), (user_agent or "")[:300]),
            )
        finally:
            con.close()

        return Session(token=token, user=user, expires_at=_iso(expires))

    def resolve_session(self, token: str | None) -> User | None:
        """Return the signed-in user, or None. Expired sessions are removed on sight."""
        if not token:
            return None

        con = self._connect()
        try:
            row = con.execute(
                """
                SELECT s.expires_at, s.token_hash, u.*
                FROM sessions s JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = ?
                """,
                (hash_token(token),),
            ).fetchone()
            if row is None:
                return None
            if _parse(row["expires_at"]) <= _now():
                con.execute("DELETE FROM sessions WHERE token_hash = ?", (row["token_hash"],))
                return None
            if not row["is_active"]:
                return None
        finally:
            con.close()

        return _to_user(row)

    def end_session(self, token: str | None) -> None:
        if not token:
            return
        con = self._connect()
        try:
            con.execute("DELETE FROM sessions WHERE token_hash = ?", (hash_token(token),))
        finally:
            con.close()

    def end_all_sessions(self, user_id: str) -> None:
        con = self._connect()
        try:
            con.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        finally:
            con.close()

    def purge_expired(self) -> int:
        con = self._connect()
        try:
            cur = con.execute("DELETE FROM sessions WHERE expires_at <= ?", (_iso(_now()),))
            return cur.rowcount or 0
        finally:
            con.close()

    def active_session_count(self, user_id: str | None = None) -> int:
        con = self._connect()
        try:
            if user_id:
                return int(
                    con.execute(
                        "SELECT count(*) FROM sessions WHERE user_id = ? AND expires_at > ?",
                        (user_id, _iso(_now())),
                    ).fetchone()[0]
                )
            return int(
                con.execute(
                    "SELECT count(*) FROM sessions WHERE expires_at > ?", (_iso(_now()),)
                ).fetchone()[0]
            )
        finally:
            con.close()


def _to_user(row: sqlite3.Row) -> User:
    return User(
        id=row["id"],
        email=row["email"],
        full_name=row["full_name"],
        organisation=row["organisation"],
        role=row["role"],
        is_active=bool(row["is_active"]),
        created_at=row["created_at"],
        last_login_at=row["last_login_at"],
    )
