"""Ground truth for the browser implementation.

Writes what the service's own code path produces for a spread of applicants, so
verify_browser_model.mjs can check the JavaScript against it rather than against a
reimplementation of the same assumptions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist-verify" / "expected.json"

#: Chosen to cover the decision boundary from both sides, the extremes of each
#: slider, and the band edges — not a random sample, which would mostly land in the
#: approved bulk and prove little.
CASES = [
    ("stretched", dict(income_usd=66_000, loan_amount_usd=345_000, cltv=92, dti_band=">60%")),
    ("marginal", dict(income_usd=96_000, loan_amount_usd=465_000, cltv=94, dti_band="47")),
    ("strong", dict(income_usd=185_000, loan_amount_usd=415_000, cltv=67, dti_band="30%-<36%")),
    ("just below", dict(income_usd=120_000, loan_amount_usd=430_000, cltv=95, dti_band="49")),
    ("just above", dict(income_usd=130_000, loan_amount_usd=420_000, cltv=88, dti_band="45")),
    ("low income", dict(income_usd=30_000, loan_amount_usd=200_000, cltv=80, dti_band="42")),
    ("high income", dict(income_usd=400_000, loan_amount_usd=900_000, cltv=70, dti_band="36")),
    ("max ltv", dict(income_usd=90_000, loan_amount_usd=380_000, cltv=100, dti_band="44")),
    ("min ltv", dict(income_usd=90_000, loan_amount_usd=380_000, cltv=30, dti_band="44")),
    ("lowest dti", dict(income_usd=110_000, loan_amount_usd=300_000, cltv=75, dti_band="<20%")),
    ("highest dti", dict(income_usd=110_000, loan_amount_usd=300_000, cltv=75, dti_band=">60%")),
    ("large loan", dict(income_usd=250_000, loan_amount_usd=1_200_000, cltv=85, dti_band="43")),
    ("small loan", dict(income_usd=250_000, loan_amount_usd=80_000, cltv=40, dti_band="38")),
    ("band edge 36", dict(income_usd=100_000, loan_amount_usd=350_000, cltv=85, dti_band="36")),
    ("band edge 49", dict(income_usd=100_000, loan_amount_usd=350_000, cltv=85, dti_band="49")),
    ("band 50-60", dict(income_usd=100_000, loan_amount_usd=350_000, cltv=85, dti_band="50%-60%")),
]

DEFAULTS = dict(
    loan_term=360,
    units_band="1",
    tract_to_msa_income_pct=100,
    preapproval=False,
    interest_only=False,
    balloon=False,
    neg_amortization=False,
)


def main() -> None:
    from tally.data.features import RawApplicant, build_vector
    from tally.explain.reason_codes import derive_reason_codes
    from tally.explain.shap_engine import get_engine
    from tally.model.registry import load_current
    from tally.serving.scoring import score_vector

    model = load_current()
    engine = get_engine(model)

    cases = []
    for name, overrides in CASES:
        applicant = RawApplicant(**{**DEFAULTS, **overrides})
        x = build_vector(applicant, model.fit_params, model.feature_names)
        scored = score_vector(applicant, model)
        attribution = engine.attribute_one(x)
        reasons = derive_reason_codes(attribution, scored.decision)

        cases.append(
            {
                "name": name,
                "applicant": {**DEFAULTS, **overrides},
                "vector": [float(v) for v in x],
                "raw_score": float(scored.raw_score),
                "probability": float(scored.probability),
                "decision": scored.decision,
                "shap": [float(v) for v in attribution.values],
                "reason_codes": [c.code for c in reasons.codes],
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "model_version": model.version,
                "features": list(model.feature_names),
                "base_value": float(engine.base_value),
                "cases": cases,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    declined = sum(1 for c in cases if c["decision"] == "DECLINED")
    print(f"  wrote {len(cases)} cases ({declined} declined) to {OUT.relative_to(ROOT)}")
    print(f"  base value {engine.base_value:+.6f}  ·  model {model.version}")


if __name__ == "__main__":
    main()
