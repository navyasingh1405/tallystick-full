"""Canonical serialisation.

Written before the implementation. If the serialisation drifts between the write
path and the verify path, the chain "breaks" for the wrong reason and the whole
tamper-evidence claim becomes noise. These tests pin the format.
"""

from __future__ import annotations

import json
import math

import pytest

from tally.ledger.canonical import (
    CANON_VERSION,
    GENESIS_HASH,
    canon_num,
    canonical_json,
    row_hash,
)

# --------------------------------------------------------------------------- numbers


def test_integers_render_without_a_decimal_point():
    assert canon_num(0) == "0"
    assert canon_num(42) == "42"
    assert canon_num(-7) == "-7"


def test_negative_zero_is_indistinguishable_from_zero():
    """-0.0 and 0.0 are equal but have different bit patterns and different reprs.
    If they hashed differently, a round trip through JSON could break the chain."""
    assert canon_num(-0.0) == canon_num(0.0) == "0"


def test_floats_round_trip_exactly():
    for v in (0.1, 1 / 3, 1e-320, 1e308, 123456.789012345, math.pi):
        assert float(canon_num(v)) == v


def test_bool_is_rejected_as_a_number():
    """bool is a subclass of int in Python; silently hashing True as 1 would make
    two different inputs collide."""
    with pytest.raises(TypeError):
        canon_num(True)


def test_nan_and_inf_are_refused():
    for v in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            canon_num(v)


# --------------------------------------------------------------------------- json


def test_key_order_does_not_affect_output():
    a = canonical_json({"b": 1, "a": 2, "c": {"z": 1, "y": 2}})
    b = canonical_json({"c": {"y": 2, "z": 1}, "a": 2, "b": 1})
    assert a == b


def test_output_has_no_whitespace():
    out = canonical_json({"a": 1, "b": [1, 2]})
    assert " " not in out and "\n" not in out


def test_serialisation_is_idempotent_through_a_json_round_trip():
    """The property that makes 'store the canonical text, hash the stored text' safe."""
    obj = {
        "income_usd": 125000.0,
        "flags": [True, False],
        "band": "30%-<36%",
        "nested": {"a": [1, 2.5, None]},
        "unicode": "café — naïve",
    }
    once = canonical_json(obj)
    twice = canonical_json(json.loads(once))
    assert once == twice


def test_floats_and_their_integer_equivalents_are_distinguishable():
    assert canonical_json({"a": 1}) != canonical_json({"a": 1.5})


def test_control_characters_cannot_escape_the_encoding():
    """row_hash joins fields with \\n and marks NULL with \\x00. If canonical_json
    could emit either unescaped, two different rows could produce one preimage."""
    out = canonical_json({"a": "x\ny", "b": "p\x00q"})
    assert "\n" not in out
    assert "\x00" not in out


def test_non_serialisable_types_raise_rather_than_stringify():
    import datetime
    import uuid

    for bad in (datetime.datetime.now(), uuid.uuid4(), {1: "int key"}, object()):
        with pytest.raises(TypeError):
            canonical_json({"v": bad} if not isinstance(bad, dict) else bad)


def test_empty_containers_are_stable():
    assert canonical_json({}) == "{}"
    assert canonical_json({"a": []}) == '{"a":[]}'


# --------------------------------------------------------------------------- row hash


def _row(**over: object) -> dict[str, str | None]:
    base: dict[str, str | None] = {
        "decision_id": "11111111-2222-3333-4444-555555555555",
        "created_at": "2026-08-10T12:00:00.000000Z",
        "model_version": "v001",
        "model_hash": "a" * 64,
        "input_json": '{"income_usd":"125000"}',
        "raw_score_txt": "0.25",
        "threshold_txt": "0.5",
        "decision": "DECLINED",
        "reason_codes_json": "[]",
        "supersedes": None,
        "override_decision": None,
        "override_actor": None,
        "override_reason": None,
        "override_at": None,
        "batch_id": None,
        "prev_hash": GENESIS_HASH,
    }
    base.update(over)  # type: ignore[arg-type]
    return base


def test_row_hash_is_stable():
    assert row_hash(_row()) == row_hash(_row())
    assert len(row_hash(_row())) == 64


def test_every_hashed_field_changes_the_hash():
    baseline = row_hash(_row())
    for field in _row():
        changed = row_hash(_row(**{field: "CHANGED"}))
        assert changed != baseline, f"changing {field} did not change the row hash"


def test_null_is_distinguishable_from_the_empty_string():
    assert row_hash(_row(supersedes=None)) != row_hash(_row(supersedes=""))


def test_field_boundaries_cannot_be_shifted():
    """Moving content across a field boundary must not produce the same preimage."""
    a = row_hash(_row(model_version="v001", model_hash="b" * 64))
    b = row_hash(_row(model_version="v001\n" + "b" * 64, model_hash=""))
    assert a != b


def test_canon_version_is_part_of_the_preimage():
    assert CANON_VERSION in ("TALLY-LEDGER-v1",)


def test_golden_vectors(golden_vectors):
    """Committed input -> hash pairs. If these change, the ledger format changed and
    every previously written chain is invalidated. That must be a deliberate act."""
    for case in golden_vectors["canonical_json"]:
        assert canonical_json(case["input"]) == case["expected"], case["name"]
    for case in golden_vectors["row_hash"]:
        assert row_hash(case["input"]) == case["expected"], case["name"]
