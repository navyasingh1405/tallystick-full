"""Scoring: the one path that produces a decision.

Everything the governance layer promises is a byproduct of calling `score_applicant`
once — the decision, the reason codes, the counterfactual, and the ledger row all
come out of the same call. There is no way to score an applicant without recording
it, which is the property the whole project rests on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tally.data.features import RawApplicant, build_vector
from tally.explain.counterfactual import Counterfactual, find_counterfactual
from tally.explain.reason_codes import ReasonSet, derive_reason_codes
from tally.explain.shap_engine import get_engine
from tally.ledger.store import DecisionRecord, LedgerStore, WrittenDecision
from tally.model.registry import LoadedModel, load_current


@dataclass(frozen=True)
class ScoreOnly:
    """A score with no ledger side effect. Used by replay and by batch preflight."""

    probability: float
    raw_score: float
    decision: str
    margin: float


def score_vector(applicant: RawApplicant, model: LoadedModel) -> ScoreOnly:
    x = build_vector(applicant, model.fit_params, model.feature_names)
    raw = float(model.raw_score(x)[0])
    prob = float(1.0 / (1.0 + np.exp(-raw)))
    return ScoreOnly(
        probability=prob,
        raw_score=raw,
        decision=model.decide(prob),
        margin=raw - model.raw_threshold,
    )


@dataclass
class ScoreResult:
    decision_id: str
    decision: str
    probability: float
    raw_score: float
    threshold: float
    margin: float
    model_version: str
    model_hash: str
    reasons: ReasonSet
    counterfactual: Counterfactual | None
    contributions: dict[str, float] = field(default_factory=dict)
    ledger: WrittenDecision | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "decision": self.decision,
            "probability": round(self.probability, 6),
            "raw_score": round(self.raw_score, 6),
            "threshold": self.threshold,
            "margin": round(self.margin, 6),
            "model": {"version": self.model_version, "hash": self.model_hash},
            "reason_codes": self.reasons.as_list(),
            "counterfactual": self.counterfactual.as_dict() if self.counterfactual else None,
            "contributions": {k: round(v, 6) for k, v in self.contributions.items()},
            "ledger": (
                {
                    "seq": self.ledger.seq,
                    "row_hash": self.ledger.row_hash,
                    "prev_hash": self.ledger.prev_hash,
                    "created_at": self.ledger.created_at,
                }
                if self.ledger
                else None
            ),
        }


def score_applicant(
    applicant: RawApplicant,
    *,
    model: LoadedModel | None = None,
    ledger: LedgerStore | None = None,
    with_counterfactual: bool = True,
    batch_id: str | None = None,
) -> ScoreResult:
    """Score one applicant, explain the outcome, and record it."""
    model = model or load_current()
    ledger = ledger or LedgerStore.default()

    scored = score_vector(applicant, model)
    engine = get_engine(model)
    x = build_vector(applicant, model.fit_params, model.feature_names)
    attribution = engine.attribute_one(x)
    reasons = derive_reason_codes(attribution, scored.decision)

    counterfactual = None
    if with_counterfactual and reasons.top_feature is not None:
        counterfactual = find_counterfactual(applicant, reasons.top_feature, model)

    written = ledger.append(
        DecisionRecord(
            model_version=model.version,
            model_hash=model.model_hash,
            input_payload=applicant.model_dump(),
            raw_score=scored.probability,
            threshold=model.threshold,
            decision=scored.decision,
            reason_codes=reasons.as_list(),
            batch_id=batch_id,
        )
    )

    return ScoreResult(
        decision_id=written.decision_id,
        decision=scored.decision,
        probability=scored.probability,
        raw_score=scored.raw_score,
        threshold=model.threshold,
        margin=scored.margin,
        model_version=model.version,
        model_hash=model.model_hash,
        reasons=reasons,
        counterfactual=counterfactual,
        contributions=attribution.as_dict(),
        ledger=written,
    )
