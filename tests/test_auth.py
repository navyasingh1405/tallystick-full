"""Authentication.

The properties tested here are the ones whose absence is invisible until it matters:
that a stored password cannot be reversed, that a database read does not yield usable
sessions, that a sign-in form cannot be used to enumerate accounts, and that the audit
trail records who actually acted rather than who claimed to.
"""

from __future__ import annotations

import sqlite3

import pytest

from tally.auth import passwords
from tally.auth.store import (
    AuthStore,
    EmailAlreadyRegistered,
    InvalidCredentials,
    RateLimited,
    hash_token,
)


@pytest.fixture
def store(tmp_path):
    return AuthStore(tmp_path / "accounts.db")


GOOD_PASSWORD = "tessellate-harbour-9182"


# --------------------------------------------------------------------------- hashing


def test_password_is_not_recoverable_from_the_stored_value():
    digest = passwords.hash_password(GOOD_PASSWORD)
    assert GOOD_PASSWORD not in digest
    assert digest.startswith("$argon2id$")
    assert passwords.verify_password(digest, GOOD_PASSWORD)


def test_the_same_password_hashes_differently_every_time():
    """Per-password salting. Identical hashes would reveal which accounts share a
    password, and would make a precomputed table worth building."""
    a = passwords.hash_password(GOOD_PASSWORD)
    b = passwords.hash_password(GOOD_PASSWORD)
    assert a != b
    assert passwords.verify_password(a, GOOD_PASSWORD)
    assert passwords.verify_password(b, GOOD_PASSWORD)


def test_a_wrong_password_does_not_verify():
    digest = passwords.hash_password(GOOD_PASSWORD)
    assert not passwords.verify_password(digest, GOOD_PASSWORD + "x")
    assert not passwords.verify_password(digest, "")


def test_a_malformed_hash_returns_false_rather_than_raising():
    """A corrupted row must fail closed, not take the process down."""
    assert not passwords.verify_password("not-a-hash", GOOD_PASSWORD)
    assert not passwords.verify_password("", GOOD_PASSWORD)


def test_unicode_normalisation_so_the_same_password_matches():
    """The same characters typed on two machines can arrive composed or decomposed.
    Without normalisation the second machine cannot sign in, and the cause is
    invisible."""
    composed = "café-harbour-tessellate"
    decomposed = "café-harbour-tessellate"
    assert composed != decomposed
    assert passwords.verify_password(passwords.hash_password(composed), decomposed)


@pytest.mark.parametrize(
    "bad",
    [
        "short",
        "password1234",  # a common base padded to length
        "Password123!",  # composition rules satisfied, still guessable
        "aaaaaaaaaaaaaaa",
        "123456789012",
        "qwertyuiopqw",
        "",
    ],
)
def test_weak_passwords_are_refused(bad):
    with pytest.raises(passwords.PasswordPolicyError):
        passwords.hash_password(bad)


@pytest.mark.parametrize(
    "good",
    [
        "tessellate-harbour-9182",
        "correct horse battery staple",
        "the quiet ledger walks north",
    ],
)
def test_reasonable_passphrases_are_accepted(good):
    assert passwords.check_policy(good).ok


def test_policy_is_length_led_not_composition_led():
    """A long passphrase with no digits or symbols is accepted; the composition-rule
    habit produces `Password1!` and teaches people to write passwords down."""
    assert passwords.check_policy("correct horse battery staple").ok


def test_a_password_equal_to_the_email_is_refused():
    result = passwords.check_policy("someone@example.com", email="someone@example.com")
    assert not result.ok


# --------------------------------------------------------------------------- accounts


def test_creating_and_finding_an_account(store):
    user = store.create_user(
        "Risk.Officer@Meridian.example", GOOD_PASSWORD, full_name="Alex", role="analyst"
    )
    assert user.email == "risk.officer@meridian.example"  # normalised
    assert user.role == "analyst"
    assert store.get_user_by_email("RISK.OFFICER@meridian.example") is not None


def test_duplicate_registration_is_refused(store):
    store.create_user("a@b.com", GOOD_PASSWORD)
    with pytest.raises(EmailAlreadyRegistered):
        store.create_user("A@B.com", GOOD_PASSWORD)


def test_the_user_object_cannot_carry_a_password_hash(store):
    """Whatever is serialised to a client must not be able to include the hash."""
    user = store.create_user("a@b.com", GOOD_PASSWORD)
    serialised = user.as_dict()
    assert "password" not in str(serialised).lower()
    assert not hasattr(user, "password_hash")


def test_a_failed_policy_leaves_no_account_behind(store):
    with pytest.raises(passwords.PasswordPolicyError):
        store.create_user("a@b.com", "short")
    assert store.get_user_by_email("a@b.com") is None
    assert store.count_users() == 0


# --------------------------------------------------------------------------- sign in


def test_correct_credentials_authenticate(store):
    store.create_user("a@b.com", GOOD_PASSWORD)
    assert store.authenticate("a@b.com", GOOD_PASSWORD).email == "a@b.com"


def test_wrong_password_and_unknown_account_fail_identically(store):
    """Different messages here turn the sign-in form into an account directory."""
    store.create_user("a@b.com", GOOD_PASSWORD)

    with pytest.raises(InvalidCredentials) as wrong:
        store.authenticate("a@b.com", "wrong-password-entirely")
    with pytest.raises(InvalidCredentials) as unknown:
        store.authenticate("nobody@b.com", GOOD_PASSWORD)

    assert str(wrong.value) == str(unknown.value)


def test_a_disabled_account_cannot_sign_in(store):
    from tally.auth.store import AccountDisabled

    store.create_user("a@b.com", GOOD_PASSWORD)
    store.set_active("a@b.com", False)
    with pytest.raises(AccountDisabled):
        store.authenticate("a@b.com", GOOD_PASSWORD)


def test_repeated_failures_are_throttled(store):
    store.create_user("a@b.com", GOOD_PASSWORD)
    for _ in range(8):
        with pytest.raises(InvalidCredentials):
            store.authenticate("a@b.com", "wrong", client_ip="10.0.0.1")
    with pytest.raises(RateLimited):
        store.authenticate("a@b.com", "wrong", client_ip="10.0.0.1")
    # The throttle protects the account, so the correct password is also held off.
    with pytest.raises(RateLimited):
        store.authenticate("a@b.com", GOOD_PASSWORD, client_ip="10.0.0.1")


def test_a_successful_sign_in_clears_the_failure_budget(store):
    store.create_user("a@b.com", GOOD_PASSWORD)
    for _ in range(4):
        with pytest.raises(InvalidCredentials):
            store.authenticate("a@b.com", "wrong", client_ip="10.0.0.2")
    store.authenticate("a@b.com", GOOD_PASSWORD, client_ip="10.0.0.2")
    for _ in range(4):
        with pytest.raises(InvalidCredentials):
            store.authenticate("a@b.com", "wrong", client_ip="10.0.0.2")
    # Still under the limit, because the budget reset on success.
    with pytest.raises(InvalidCredentials):
        store.authenticate("a@b.com", "wrong", client_ip="10.0.0.2")


def test_no_password_is_ever_written_to_the_attempts_table(store, tmp_path):
    store.create_user("a@b.com", GOOD_PASSWORD)
    with pytest.raises(InvalidCredentials):
        store.authenticate("a@b.com", "a-very-distinctive-wrong-password", client_ip="1.2.3.4")

    con = sqlite3.connect(store.path)
    rows = con.execute("SELECT * FROM login_attempts").fetchall()
    con.close()
    assert rows
    assert "a-very-distinctive-wrong-password" not in str(rows)


# --------------------------------------------------------------------------- sessions


def test_a_session_resolves_to_its_user(store):
    user = store.create_user("a@b.com", GOOD_PASSWORD)
    session = store.start_session(user)
    assert store.resolve_session(session.token).id == user.id


def test_the_database_stores_a_hash_not_the_token(store):
    """A read of the sessions table must not hand anybody a working cookie."""
    user = store.create_user("a@b.com", GOOD_PASSWORD)
    session = store.start_session(user)

    con = sqlite3.connect(store.path)
    stored = [r[0] for r in con.execute("SELECT token_hash FROM sessions")]
    con.close()

    assert session.token not in stored
    assert hash_token(session.token) in stored


def test_an_unknown_or_missing_token_resolves_to_nobody(store):
    assert store.resolve_session(None) is None
    assert store.resolve_session("") is None
    assert store.resolve_session("made-up-token") is None


def test_signing_out_revokes_the_session(store):
    user = store.create_user("a@b.com", GOOD_PASSWORD)
    session = store.start_session(user)
    store.end_session(session.token)
    assert store.resolve_session(session.token) is None


def test_an_expired_session_stops_working(store, monkeypatch):
    user = store.create_user("a@b.com", GOOD_PASSWORD)
    session = store.start_session(user)

    con = sqlite3.connect(store.path)
    con.execute(
        "UPDATE sessions SET expires_at = ? WHERE token_hash = ?",
        ("2000-01-01T00:00:00.000000Z", hash_token(session.token)),
    )
    con.commit()
    con.close()

    assert store.resolve_session(session.token) is None


def test_disabling_an_account_invalidates_its_live_sessions(store):
    user = store.create_user("a@b.com", GOOD_PASSWORD)
    session = store.start_session(user)
    store.set_active("a@b.com", False)
    assert store.resolve_session(session.token) is None


def test_changing_a_password_signs_out_everywhere(store):
    """A password change that leaves old sessions alive is not a password change —
    it is exactly the situation someone changes their password to end."""
    user = store.create_user("a@b.com", GOOD_PASSWORD)
    first = store.start_session(user)
    second = store.start_session(user)

    store.change_password("a@b.com", "a-different-long-passphrase")

    assert store.resolve_session(first.token) is None
    assert store.resolve_session(second.token) is None
    assert store.authenticate("a@b.com", "a-different-long-passphrase")


# --------------------------------------------------------------------------- roles


def test_roles_grant_the_expected_actions(store):
    demo = store.create_user("d@b.com", GOOD_PASSWORD, role="demo")
    admin = store.create_user("x@b.com", GOOD_PASSWORD, role="admin")

    assert demo.can("score") and demo.can("read") and demo.can("override")
    assert not demo.can("admin")
    assert admin.can("admin")


def test_an_unknown_role_is_refused(store):
    with pytest.raises(ValueError, match="unknown role"):
        store.create_user("a@b.com", GOOD_PASSWORD, role="superuser")


def test_an_unrecognised_permission_is_denied_by_default(store):
    user = store.create_user("a@b.com", GOOD_PASSWORD)
    assert not user.can("delete_the_ledger")


# --------------------------------------------------------------------------- storage


def test_accounts_live_apart_from_the_decision_ledger():
    """The ledger is append-only by trigger and chained by hash; accounts must be
    updatable and deletable. One file cannot honour both."""
    from tally.config import get_config

    cfg = get_config()
    assert AuthStore.default().path != cfg.paths.ledger_db


def test_the_accounts_schema_has_no_ledger_tables(store):
    con = sqlite3.connect(store.path)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert "decisions" not in tables
    assert {"users", "sessions", "login_attempts"} <= tables
