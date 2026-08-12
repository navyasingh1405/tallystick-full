"""Reason code vocabulary.

Regulation B requires the specific principal reasons for an adverse action. A raw
feature name is not a reason — "dti_band_ord" tells an applicant nothing. Every code
that can be emitted has plain-language text here, and a test asserts that a code
cannot exist without one.

The applicant-facing text is written in second person and describes the applicant's
situation, not the model's internals. It never mentions scores, thresholds, SHAP, or
percentiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


class UnmappedReasonCode(RuntimeError):
    """A code was produced with no plain-language text. Never fall back to a name."""


@dataclass(frozen=True)
class ReasonTemplate:
    code: str
    #: Shown to the applicant in the adverse action notice.
    applicant_text: str
    #: Shown to internal reviewers, may name the factor precisely.
    internal_text: str
    #: The factor as a human noun phrase, used in counterfactual sentences.
    factor: str


TEMPLATES: Final[dict[str, ReasonTemplate]] = {
    "INCOME_LOW": ReasonTemplate(
        code="INCOME_LOW",
        applicant_text="Your income is low relative to the amount you requested",
        internal_text="Income low relative to requested loan amount",
        factor="annual income",
    ),
    "LOAN_AMOUNT_HIGH": ReasonTemplate(
        code="LOAN_AMOUNT_HIGH",
        applicant_text="The loan amount requested is high relative to your income",
        internal_text="Requested loan amount high relative to capacity",
        factor="requested loan amount",
    ),
    "LTV_HIGH": ReasonTemplate(
        code="LTV_HIGH",
        applicant_text=(
            "The loan amount is high relative to the value of the property, "
            "leaving limited equity"
        ),
        internal_text="Combined loan-to-value ratio high",
        factor="combined loan-to-value ratio",
    ),
    "DTI_HIGH": ReasonTemplate(
        code="DTI_HIGH",
        applicant_text=(
            "Your total monthly debt payments are high relative to your monthly income"
        ),
        internal_text="Debt-to-income ratio high",
        factor="debt-to-income ratio",
    ),
    "TERM_UNFAVOURABLE": ReasonTemplate(
        code="TERM_UNFAVOURABLE",
        applicant_text="The repayment term requested is outside the range typically approved",
        internal_text="Loan term outside the typical approved range",
        factor="loan term",
    ),
    "INTEREST_ONLY": ReasonTemplate(
        code="INTEREST_ONLY",
        applicant_text=(
            "The loan requested has an interest-only period, which defers repayment "
            "of the principal balance"
        ),
        internal_text="Interest-only payment feature present",
        factor="interest-only payment feature",
    ),
    "BALLOON_PAYMENT": ReasonTemplate(
        code="BALLOON_PAYMENT",
        applicant_text=(
            "The loan requested includes a balloon payment, which concentrates "
            "repayment at the end of the term"
        ),
        internal_text="Balloon payment feature present",
        factor="balloon payment feature",
    ),
    "NEG_AMORTIZATION": ReasonTemplate(
        code="NEG_AMORTIZATION",
        applicant_text=(
            "The loan requested allows the balance owed to increase over time"
        ),
        internal_text="Negative amortization feature present",
        factor="negative amortization feature",
    ),
    "AREA_INCOME_LOW": ReasonTemplate(
        code="AREA_INCOME_LOW",
        applicant_text=(
            "Economic indicators for the area in which the property is located"
        ),
        internal_text="Tract income relative to the metropolitan area",
        factor="tract-to-area income percentage",
    ),
    "PROPERTY_UNITS": ReasonTemplate(
        code="PROPERTY_UNITS",
        applicant_text="The number of units in the property securing the loan",
        internal_text="Number of dwelling units in the collateral property",
        factor="number of units",
    ),
    "NO_PREAPPROVAL": ReasonTemplate(
        code="NO_PREAPPROVAL",
        applicant_text="No preapproval was requested for this application",
        internal_text="Preapproval not requested",
        factor="preapproval request",
    ),
}

#: Model feature -> reason code. Every constrained feature must appear here.
FEATURE_TO_CODE: Final[dict[str, str]] = {
    "log_income": "INCOME_LOW",
    "log_loan_amount": "LOAN_AMOUNT_HIGH",
    "cltv": "LTV_HIGH",
    "dti_band_ord": "DTI_HIGH",
    "loan_term": "TERM_UNFAVOURABLE",
    "interest_only": "INTEREST_ONLY",
    "balloon": "BALLOON_PAYMENT",
    "neg_amortization": "NEG_AMORTIZATION",
    "tract_to_msa_income_pct": "AREA_INCOME_LOW",
    "units_ord": "PROPERTY_UNITS",
    "preapproval": "NO_PREAPPROVAL",
}


def code_for_feature(feature: str) -> str:
    code = FEATURE_TO_CODE.get(feature)
    if code is None:
        raise UnmappedReasonCode(
            f"feature {feature!r} has no reason code. A raw feature name must never "
            "reach an applicant-facing field — add a mapping in explain/templates.py."
        )
    return code


def template_for_code(code: str) -> ReasonTemplate:
    tpl = TEMPLATES.get(code)
    if tpl is None:
        raise UnmappedReasonCode(
            f"reason code {code!r} has no plain-language text. Regulation B requires "
            "specific principal reasons; an unmapped code cannot be disclosed."
        )
    return tpl


def template_for_feature(feature: str) -> ReasonTemplate:
    return template_for_code(code_for_feature(feature))
