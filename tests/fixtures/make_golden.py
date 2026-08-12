"""Regenerate the canonical golden vectors.

Run this ONLY when the ledger serialisation is deliberately changed, alongside a
CANON_VERSION bump. Every chain written under the previous format becomes
unverifiable, so this is not a routine operation.

    uv run python tests/fixtures/make_golden.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from tally.ledger.canonical import CANON_VERSION, GENESIS_HASH, canonical_json, row_hash

JSON_CASES = [
    ("empty object", {}),
    ("empty list", {"a": []}),
    ("key ordering", {"b": 1, "a": 2, "c": 3}),
    ("nested ordering", {"z": {"b": 1, "a": 2}, "a": [3, 2, 1]}),
    ("negative zero", {"v": -0.0}),
    ("integers", {"a": 0, "b": -7, "c": 1000000}),
    ("float precision", {"a": 0.1, "b": 1 / 3, "c": math.pi}),
    ("float extremes", {"small": 1e-320, "large": 1e308}),
    ("booleans and null", {"t": True, "f": False, "n": None}),
    ("unicode", {"s": "café — naïve 日本語"}),
    ("control characters", {"nl": "x\ny", "nul": "p\x00q", "tab": "a\tb"}),
    (
        "applicant payload",
        {
            "income_usd": 125000.0,
            "loan_amount_usd": 415000.0,
            "property_value_usd": 495000.0,
            "cltv": 83.8383838383,
            "dti_band": "30%-<36%",
            "loan_term": 360.0,
            "interest_only": False,
            "balloon": False,
            "neg_amortization": False,
            "units_band": "1",
            "preapproval": False,
            "tract_to_msa_income_pct": 112.5,
        },
    ),
    (
        "reason codes",
        [
            {"rank": 1, "code": "DTI_HIGH", "contribution": -1.2345},
            {"rank": 2, "code": "LTV_HIGH", "contribution": -0.4321},
        ],
    ),
]


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


ROW_CASES = [
    ("genesis decline", _row()),
    ("approval", _row(decision="APPROVED", raw_score_txt="0.87")),
    ("linked row", _row(prev_hash="c" * 64)),
    ("override row", _row(
        supersedes="99999999-8888-7777-6666-555555555555",
        override_decision="APPROVED",
        override_actor="credit.officer@example.com",
        override_reason="Documented compensating factors.",
        override_at="2026-08-10T13:00:00.000000Z",
    )),
    ("batch row", _row(batch_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")),
]


def main() -> None:
    payload = {
        "canon_version": CANON_VERSION,
        "canonical_json": [
            {"name": name, "input": value, "expected": canonical_json(value)}
            for name, value in JSON_CASES
        ],
        "row_hash": [
            {"name": name, "input": value, "expected": row_hash(value)}
            for name, value in ROW_CASES
        ],
    }
    out = Path(__file__).with_name("canonical_golden.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({len(payload['canonical_json'])} json, {len(payload['row_hash'])} rows)")


if __name__ == "__main__":
    main()
