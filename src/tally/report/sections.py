"""Validation pack structure.

One list drives the table of contents, the section headings, and the regulatory
mapping appendix. They cannot drift apart because there is only one of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tally.config import get_config


@dataclass(frozen=True)
class Section:
    number: int
    slug: str
    title: str
    #: One line under the heading saying what the section is for.
    purpose: str


SECTIONS: tuple[Section, ...] = (
    Section(
        1,
        "purpose",
        "Model purpose, intended use, and limitations of use",
        "What the model is for, and — stated explicitly — what it is not for.",
    ),
    Section(
        2,
        "lineage",
        "Data lineage",
        "Source, vintage, every filter applied, and the row count at each stage.",
    ),
    Section(
        3,
        "features",
        "Features, transformations, and constraints",
        "Each input, how it is derived, and the written rationale for its monotonicity constraint.",
    ),
    Section(
        4,
        "methodology",
        "Development methodology",
        "Algorithm, hyperparameters, and the measures taken to make training reproducible.",
    ),
    Section(
        5,
        "performance",
        "Performance",
        "Discrimination and calibration overall and by segment.",
    ),
    Section(
        6,
        "stability",
        "Stability and out-of-time validation",
        "How performance holds up on a vintage the model never saw, and how the inputs have shifted.",
    ),
    Section(
        7,
        "fairness",
        "Fair lending analysis",
        "Raw disparity, disparity surviving controls, and a scan for proxy variables.",
    ),
    Section(
        8,
        "lda",
        "Less discriminatory alternative analysis",
        "A search for a comparably effective model with lower measured disparity.",
    ),
    Section(
        9,
        "explainability",
        "Explainability and adverse action notices",
        "How reasons for a decline are derived, and a worked example.",
    ),
    Section(
        10,
        "limitations",
        "Known limitations",
        "What this model and this document do not establish.",
    ),
    Section(
        11,
        "monitoring",
        "Monitoring plan and revalidation triggers",
        "What is watched after deployment, and what would force a rebuild.",
    ),
    Section(
        12,
        "regulatory",
        "Appendix: regulatory mapping",
        "Each section cross-referenced to the frameworks it is written against.",
    ),
)

SECTION_BY_NUMBER: dict[int, Section] = {s.number: s for s in SECTIONS}


def regulatory_appendix() -> list[dict[str, Any]]:
    """Join the section list to the regulatory map. Drives appendix 12."""
    cfg = get_config()
    mapping = {int(entry["id"]): entry for entry in cfg.regulatory_map["sections"]}
    frameworks = cfg.regulatory_map["frameworks"]

    del frameworks  # exposed separately; see `frameworks()`
    return [
        {
            "number": section.number,
            "title": section.title,
            "sr117": mapping.get(section.number, {}).get("sr117", []),
            "eu_ai_act": mapping.get(section.number, {}).get("eu_ai_act", []),
            "ecoa": mapping.get(section.number, {}).get("ecoa", []),
        }
        for section in SECTIONS
    ]


def frameworks() -> dict[str, Any]:
    return get_config().regulatory_map["frameworks"]
