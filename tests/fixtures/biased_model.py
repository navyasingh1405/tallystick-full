"""The adversarial fixture.

A governance tool that has never been shown to catch anything is not a governance
tool. This builds a model with a deliberate, obvious, measurable disparity and the
test suite asserts that all three fairness analyses catch it.

The construction is honest about what it is: a synthetic feature correlated 0.9 with
a protected attribute, given heavy weight in the label. Nothing subtle. If the
fairness engine cannot find this, it cannot find anything.

A matching *clean* fixture is built the same way with the correlation removed, so the
tests can also assert the engine does not fire on a fair model. A detector that
always alarms is as useless as one that never does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Correlation between the synthetic proxy and membership of the disadvantaged group.
PROXY_CORRELATION = 0.9

#: How strongly the proxy drives the outcome, in log-odds.
PROXY_WEIGHT = 2.5

GROUPS = ("NH_White", "Black", "Hispanic", "Asian")
GROUP_SHARES = (0.70, 0.12, 0.13, 0.05)

#: The group the injected bias disadvantages.
DISADVANTAGED = "Black"


@dataclass(frozen=True)
class BiasedFixture:
    """An evaluation frame plus the decisions of a model trained on it."""

    frame: pd.DataFrame
    biased: bool
    proxy_correlation: float

    @property
    def n(self) -> int:
        return len(self.frame)


#: Group differences in legitimate credit factors, in log-income and CLTV points.
#: Real populations differ on these, and a controlled analysis that had nothing to
#: control for would not be testing anything. The clean fixture keeps these
#: differences and drops only the proxy, so the controlled analysis has to
#: distinguish "explained by credit factors" from "not explained".
_INCOME_SHIFT = {"NH_White": 0.10, "Black": -0.22, "Hispanic": -0.18, "Asian": 0.08}
_CLTV_SHIFT = {"NH_White": -1.5, "Black": 4.0, "Hispanic": 3.5, "Asian": -1.0}


def _make_base(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = rng.choice(GROUPS, size=n, p=GROUP_SHARES)

    # Legitimate credit factors, correlated with group as they are in reality.
    log_income = rng.normal(11.6, 0.5, n) + np.array([_INCOME_SHIFT[g] for g in group])
    log_loan_amount = rng.normal(12.9, 0.4, n)
    cltv = np.clip(
        rng.normal(80, 12, n) + np.array([_CLTV_SHIFT[g] for g in group]), 20, 100
    )
    dti_band_ord = np.clip(rng.normal(8, 4, n), 0, 18).round()
    loan_term = np.full(n, 360.0)

    return pd.DataFrame(
        {
            "row_id": [f"synthetic-{i:06d}" for i in range(n)],
            "group": group,
            "derived_race": np.where(
                group == "NH_White",
                "White",
                np.where(
                    group == "Black",
                    "Black or African American",
                    np.where(group == "Asian", "Asian", "White"),
                ),
            ),
            "derived_ethnicity": np.where(
                group == "Hispanic", "Hispanic or Latino", "Not Hispanic or Latino"
            ),
            "derived_sex": rng.choice(["Male", "Female"], size=n),
            # Spans the reference band, the 62-plus bands, and under-25 so the age
            # axis of every analysis is actually exercised.
            "applicant_age": rng.choice(
                ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"], size=n
            ),
            "county_code": rng.choice([f"08{i:03d}" for i in range(1, 9)], size=n),
            "log_income": log_income,
            "log_loan_amount": log_loan_amount,
            "cltv": cltv,
            "dti_band_ord": dti_band_ord,
            "loan_term": loan_term,
            "interest_only": np.zeros(n),
            "balloon": np.zeros(n),
            "neg_amortization": np.zeros(n),
            "tract_to_msa_income_pct": rng.normal(100, 25, n),
            "units_ord": np.zeros(n),
            "preapproval": rng.integers(0, 2, n).astype(float),
        }
    )


def _proxy(is_disadvantaged: np.ndarray, correlation: float, rng: np.random.Generator) -> np.ndarray:
    """A feature correlated `correlation` with group membership.

    Built as a mixture rather than by rescaling, so the correlation is exact by
    construction and the feature stays interpretable as a continuous score.
    """
    n = len(is_disadvantaged)
    signal = np.where(is_disadvantaged, -1.0, 1.0)
    noise = rng.normal(0, 1, n)
    return correlation * signal + np.sqrt(max(0.0, 1 - correlation**2)) * noise


def build_fixture(*, biased: bool, n: int = 6000, seed: int = 2026) -> BiasedFixture:
    """Build an evaluation frame with, or without, an injected disparity."""
    rng = np.random.default_rng(seed + (1 if biased else 0))
    df = _make_base(n, seed)
    is_disadvantaged = (df["group"] == DISADVANTAGED).to_numpy()

    correlation = PROXY_CORRELATION if biased else 0.0
    df["neighborhood_score"] = _proxy(is_disadvantaged, correlation, rng)

    # The outcome depends on legitimate factors plus, when biased, the proxy.
    logit = (
        1.2 * (df["log_income"] - 11.6)
        - 0.9 * (df["log_loan_amount"] - 12.9)
        - 0.04 * (df["cltv"] - 80)
        - 0.18 * (df["dti_band_ord"] - 8)
        + 1.5
    )
    if biased:
        logit = logit + PROXY_WEIGHT * df["neighborhood_score"]
    else:
        # Keep the feature present but inert, so the proxy scan has something to
        # score in both worlds and the comparison is like for like.
        logit = logit + 0.05 * df["neighborhood_score"]

    p = 1.0 / (1.0 + np.exp(-logit))
    df["approved"] = rng.random(n) < p
    df["model_probability"] = p
    df["model_decision"] = np.where(p > 0.5, "APPROVED", "DECLINED")
    df["model_approved"] = p > 0.5

    return BiasedFixture(frame=df, biased=biased, proxy_correlation=correlation)


#: Feature columns the fairness engine should treat as the model's own.
FIXTURE_FEATURES = [
    "log_income",
    "log_loan_amount",
    "cltv",
    "dti_band_ord",
    "loan_term",
    "interest_only",
    "balloon",
    "neg_amortization",
    "tract_to_msa_income_pct",
    "units_ord",
    "preapproval",
    "neighborhood_score",
]
