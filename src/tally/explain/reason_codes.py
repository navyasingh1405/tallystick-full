"""Adverse action reason codes from threshold-anchored SHAP contributions.

Take the features working against the applicant, most damaging first, cap at four,
and map each through the vocabulary in `templates.py`. Approvals produce no reasons:
Regulation B's disclosure duty attaches to adverse action, and manufacturing reasons
for an approval would be noise in the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tally.config import get_config
from tally.explain.shap_engine import Attribution
from tally.explain.templates import template_for_feature


@dataclass(frozen=True)
class ReasonCode:
    rank: int
    code: str
    feature: str
    contribution: float
    applicant_text: str
    internal_text: str
    factor: str

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "code": self.code,
            "feature": self.feature,
            "contribution": round(float(self.contribution), 6),
            "text": self.applicant_text,
            "internal_text": self.internal_text,
            "factor": self.factor,
        }


@dataclass(frozen=True)
class ReasonSet:
    decision: str
    codes: list[ReasonCode] = field(default_factory=list)

    def as_list(self) -> list[dict[str, object]]:
        return [c.as_dict() for c in self.codes]

    @property
    def top_feature(self) -> str | None:
        return self.codes[0].feature if self.codes else None


def derive_reason_codes(attribution: Attribution, decision: str) -> ReasonSet:
    """Top-N negative contributors, mapped to plain language.

    Contributions are measured against the decision threshold, so a negative value
    means "this factor moved the applicant away from approval" rather than "this
    applicant is unusual on this factor". That is the question Regulation B asks.
    """
    if decision == "APPROVED":
        return ReasonSet(decision=decision, codes=[])

    cfg = get_config()
    limit = int(cfg.explain["max_reason_codes"])

    codes: list[ReasonCode] = []
    for rank, (feature, contribution) in enumerate(attribution.sorted_negative()[:limit], start=1):
        tpl = template_for_feature(feature)
        codes.append(
            ReasonCode(
                rank=rank,
                code=tpl.code,
                feature=feature,
                contribution=contribution,
                applicant_text=tpl.applicant_text,
                internal_text=tpl.internal_text,
                factor=tpl.factor,
            )
        )
    return ReasonSet(decision=decision, codes=codes)
