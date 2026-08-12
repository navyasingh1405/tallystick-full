"""Password hashing and policy.

Argon2id, at the reference implementation's defaults (64 MiB, 3 iterations, 4 lanes).
Nothing here is hand-rolled: a password KDF is the wrong place to be inventive, and
the failure mode of getting it wrong is silent.

The policy follows NIST SP 800-63B rather than the older composition-rule habit:
length is what matters, arbitrary character-class requirements push people toward
`Password1!` and a sticky note, and forced rotation makes that worse.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

#: Shared hasher. Parameters live in the hash string, so raising them later still
#: verifies existing passwords and `needs_rehash` reports which to upgrade.
_hasher = PasswordHasher()

MIN_LENGTH = 12
MAX_LENGTH = 1024  # bounded so a huge input cannot be used to burn CPU

#: Common bases. Rejected as substrings, not just as exact matches: padding
#: `password` out to twelve characters with digits produces something that satisfies
#: a length rule and none of its purpose. Not a substitute for a real
#: breached-password corpus, which would need an external service; this catches the
#: handful that actually show up when someone is clicking through a signup.
_OBVIOUS_BASES = (
    "password",
    "passw0rd",
    "qwerty",
    "letmein",
    "welcome",
    "changeme",
    "iloveyou",
    "monkey",
    "dragon",
    "football",
    "baseball",
    "sunshine",
    "princess",
    "trustno1",
    "creditaudit",
    "administrator",
    "admin",
)

#: Long runs of sequential or repeated characters, which pass a naive variety check.
_SEQUENCES = ("0123456789", "abcdefghijklmnopqrstuvwxyz", "qwertyuiop", "asdfghjkl")


def _is_obvious(candidate: str) -> bool:
    """True when the password is a well-known base with decoration around it."""
    lowered = candidate.lower()
    stripped = lowered.strip("0123456789!@#$%^&*()_+-=. ")

    if any(base in lowered for base in _OBVIOUS_BASES):
        return True
    if stripped and stripped in _OBVIOUS_BASES:
        return True
    # A password that is entirely a walk along one keyboard row or the digits.
    return any(
        len(lowered) >= 8 and lowered in sequence + sequence for sequence in _SEQUENCES
    )


class PasswordPolicyError(ValueError):
    """The supplied password does not meet the policy."""


@dataclass(frozen=True)
class PolicyResult:
    ok: bool
    problems: tuple[str, ...] = ()


def normalise(password: str) -> str:
    """NFKC-normalise so visually identical passwords hash identically.

    Without this, a password typed with a composed accent on one machine and a
    decomposed one on another would not match, which is maddening to diagnose.
    """
    return unicodedata.normalize("NFKC", password)


def check_policy(password: str, *, email: str = "") -> PolicyResult:
    problems: list[str] = []
    candidate = normalise(password)

    if len(candidate) < MIN_LENGTH:
        problems.append(f"Use at least {MIN_LENGTH} characters.")
    if len(candidate) > MAX_LENGTH:
        problems.append(f"Use at most {MAX_LENGTH} characters.")
    if _is_obvious(candidate):
        problems.append("That password is too easily guessed. Try an unrelated phrase.")
    if email and candidate.lower().strip() == email.lower().strip():
        problems.append("The password cannot be your email address.")
    if candidate and len(set(candidate)) < 5:
        problems.append("Use a greater variety of characters.")

    return PolicyResult(ok=not problems, problems=tuple(problems))


def hash_password(password: str, *, email: str = "") -> str:
    result = check_policy(password, email=email)
    if not result.ok:
        raise PasswordPolicyError(" ".join(result.problems))
    return _hasher.hash(normalise(password))


def verify_password(stored_hash: str, password: str) -> bool:
    """Constant-time verification. Returns False rather than raising on a mismatch."""
    try:
        return _hasher.verify(stored_hash, normalise(password))
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(stored_hash: str) -> bool:
    """True when the hash was made with weaker parameters than the current defaults."""
    try:
        return _hasher.check_needs_rehash(stored_hash)
    except InvalidHashError:
        return True


#: A real hash of a value nobody will ever submit, computed once at import. Used to
#: spend the same CPU on a nonexistent account as on a real one. It has to be a
#: genuine hash: a malformed string would fail to parse immediately, which is exactly
#: the fast path this exists to avoid.
_DECOY_HASH = _hasher.hash("decoy-for-constant-time-signin-comparison")


def dummy_verify() -> None:
    """Burn a comparable amount of CPU when an account does not exist.

    Without this, a failed sign-in for an unknown email returns measurably faster
    than one for a known email with the wrong password, which turns the login form
    into an account enumeration oracle.
    """
    try:
        _hasher.verify(_DECOY_HASH, "not-the-password")
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        pass
