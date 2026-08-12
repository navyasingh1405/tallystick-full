"""The one canonical serialiser.

Everything in this project that gets hashed goes through here: ledger rows, and the
LDA challenger cache keys. One canonicaliser means the write path and the verify
path cannot drift apart, which is the failure mode that makes a hash chain
meaningless while looking like it works.

Three design rules
------------------
1. **Store the canonical text; hash the stored text.** The ledger's `input_json`
   column holds exactly the string `canonical_json` produced, and verification
   hashes the bytes read back from SQLite rather than re-serialising a parsed
   object. That removes the entire "round trip changed the encoding" bug class.
2. **Numbers become strings.** JSON has one numeric type and implementations differ
   on how they render it. Converting to a string with an exact round-trip format
   here means the representation is decided in one place.
3. **The preimage is injective.** Fields are joined with newline and NULL is marked
   with a NUL byte, neither of which `canonical_json` can emit unescaped, so no two
   distinct rows can produce the same preimage.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Final
from uuid import UUID

#: Domain separator, mixed into every row hash.
#:
#: Bumping this invalidates every previously written chain, so it must be deliberate.
#: It has been bumped exactly once — from CAL-LEDGER-v1, at the point the project was
#: named — because a pre-release rename is the last moment such a change costs
#: nothing. After a chain exists that anyone relies on, this string is frozen: the
#: correct way to change the format then is a new version written alongside the old,
#: not a rewrite of what the old hashes meant.
CANON_VERSION: Final[str] = "TALLY-LEDGER-v1"

#: 17 significant digits round-trips any IEEE-754 double exactly and is stable
#: across platforms, unlike repr() which has changed between Python versions.
_FLOAT_FMT: Final[str] = "%.17g"

GENESIS_HASH: Final[str] = "0" * 64

#: Order matters and is part of the format. Appending is a version bump.
HASHED_FIELDS: Final[tuple[str, ...]] = (
    "decision_id",
    "created_at",
    "model_version",
    "model_hash",
    "input_json",
    "raw_score_txt",
    "threshold_txt",
    "decision",
    "reason_codes_json",
    "supersedes",
    "override_decision",
    "override_actor",
    "override_reason",
    "override_at",
    "batch_id",
    "prev_hash",
)

_FIELD_SEP: Final[str] = "\n"
_NULL_MARK: Final[str] = "\x00"


def canon_num(x: float | int) -> str:
    """Render a number as its canonical string."""
    if isinstance(x, bool):
        # bool subclasses int; hashing True as "1" would collide with the integer 1.
        raise TypeError("bool is not a number here — serialise it as a JSON boolean")
    if isinstance(x, int):
        return str(x)
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"{x!r} has no canonical representation")
    if x == 0.0:
        return "0"  # collapses -0.0, which is == 0.0 but reprs differently
    return _FLOAT_FMT % x


def _norm(o: Any) -> Any:
    """Recursively normalise a value into canonical JSON-safe form."""
    if o is None or isinstance(o, (str, bool)):
        return o
    if isinstance(o, (int, float, Decimal)):
        return canon_num(float(o) if isinstance(o, Decimal) else o)
    if isinstance(o, Mapping):
        if not all(isinstance(k, str) for k in o):
            raise TypeError("canonical JSON requires string keys")
        return {k: _norm(o[k]) for k in sorted(o)}
    if isinstance(o, (list, tuple)):
        return [_norm(v) for v in o]
    if isinstance(o, (datetime, date, UUID)):
        raise TypeError(
            f"{type(o).__name__} must be converted to a string explicitly — an "
            "implicit format here would be an undocumented part of the hash"
        )
    raise TypeError(f"non-canonicalisable type: {type(o).__name__}")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, numbers as strings."""
    return json.dumps(
        _norm(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def row_hash(row: Mapping[str, str | None]) -> str:
    """SHA-256 over the fixed field tuple.

    Every value is already a canonical string, so this function performs no
    serialisation of its own — it only concatenates. That is deliberate: there is
    exactly one place where a Python object becomes text.
    """
    parts = [CANON_VERSION]
    for field in HASHED_FIELDS:
        if field not in row:
            raise KeyError(f"row is missing hashed field {field!r}")
        value = row[field]
        parts.append(_NULL_MARK if value is None else str(value))
    return hashlib.sha256(_FIELD_SEP.join(parts).encode("utf-8")).hexdigest()


def content_hash(obj: Any) -> str:
    """SHA-256 of a canonicalised object. Used for LDA challenger cache keys."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
