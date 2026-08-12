"""API request and response schemas.

Pydantic at the boundary is part of the audit story, not decoration: an application
that fails validation never reaches the model, so the ledger cannot contain a decision
made on input the system could not describe.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tally.data.features import RawApplicant


class ScoreRequest(RawApplicant):
    """An application to score. Inherits RawApplicant, so protected attributes are
    structurally impossible to submit — there is no field for them."""

    model_config = ConfigDict(extra="forbid")


class ReasonCodeOut(BaseModel):
    rank: int
    code: str
    feature: str
    text: str
    internal_text: str
    factor: str
    contribution: float


class CounterfactualOut(BaseModel):
    feature: str
    display_name: str
    current_display: str
    flip_display: str | None
    achievable: bool
    verified_probability: float | None
    verified_decision: str | None
    reason: str | None
    sentence: str
    method: str


class LedgerRef(BaseModel):
    seq: int
    row_hash: str
    prev_hash: str
    created_at: str


class ModelRef(BaseModel):
    version: str
    hash: str


class ScoreResponse(BaseModel):
    decision_id: str
    decision: Literal["APPROVED", "DECLINED"]
    probability: float
    raw_score: float
    threshold: float
    margin: float
    model: ModelRef
    reason_codes: list[ReasonCodeOut]
    counterfactual: CounterfactualOut | None
    contributions: dict[str, float]
    ledger: LedgerRef


class OverrideRequest(BaseModel):
    """A manual override.

    There is deliberately no `actor` field. The actor is the authenticated user, taken
    from the session; accepting it from the caller would let anyone write anyone
    else's name into the audit trail, which would make the record worthless for the
    one purpose it exists to serve.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Literal["APPROVED", "DECLINED"]
    reason: str = Field(min_length=10, max_length=1000)


class ChainOut(BaseModel):
    ok: bool
    rows_checked: int
    head_hash: str | None
    first_broken_seq: int | None = None
    first_broken_reason: str | None = None
    first_broken_detail: str | None = None


class ReplayOut(BaseModel):
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


class VerifyResponse(BaseModel):
    chain: ChainOut
    replay: ReplayOut


class DecisionOut(BaseModel):
    seq: int
    decision_id: str
    created_at: str
    model_version: str
    model_hash: str
    decision: str
    probability: float
    threshold: float
    input: dict[str, Any]
    reason_codes: list[dict[str, Any]]
    supersedes: str | None
    superseded_by: str | None
    override_actor: str | None
    override_reason: str | None
    batch_id: str | None
    prev_hash: str
    row_hash: str
    neighbours: dict[str, Any]


class HealthResponse(BaseModel):
    ok: bool
    model_version: str | None
    model_hash: str | None
    ledger_height: int
    chain_ok: bool
    artifacts: dict[str, bool]


class AdoptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    challenger_id: str
    hold_approval_rate: bool = True


class JobOut(BaseModel):
    job_id: str
    status: Literal["running", "done", "error"]
    progress: float = 0.0
    detail: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody
