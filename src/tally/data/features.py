"""Feature construction and the protected-attribute boundary.

Two Pydantic models enforce the boundary structurally rather than by discipline:
`RawApplicant` carries what the model may see, `ApplicantContext` carries the
protected attributes. `build_vector` accepts only the former, so a protected
attribute cannot reach a feature vector without someone changing a type.

All transforms are pure functions of a `FitParams` record, which is persisted with
the model. Scoring an applicant a year later reproduces the same vector.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from tally.config import get_config
from tally.data import schema


class ProtectedLeakage(RuntimeError):
    """Raised when a protected attribute is requested as a model feature."""


class RawApplicant(BaseModel):
    """What the model is allowed to see, in human units.

    Protected attributes are structurally absent — there is no field to put them in.
    """

    model_config = ConfigDict(extra="forbid")

    income_usd: float = Field(gt=0, description="Gross annual income in dollars")
    loan_amount_usd: float = Field(gt=0, description="Requested loan amount in dollars")
    property_value_usd: float | None = Field(default=None, gt=0)
    cltv: float = Field(ge=0, le=200, description="Combined loan-to-value ratio, percent")
    dti_band: str = Field(description="HMDA debt-to-income band string")
    loan_term: float | None = Field(default=360, ge=0, le=600, description="Months")
    interest_only: bool = False
    balloon: bool = False
    neg_amortization: bool = False
    units_band: str = "1"
    preapproval: bool = False
    tract_to_msa_income_pct: float | None = Field(default=None, ge=0, le=1000)


class ApplicantContext(BaseModel):
    """Protected attributes and identifiers. Never reaches a feature vector.

    Held on the evaluation frame and on batch artifacts, keyed by row id, so that
    fairness can be measured without the model ever seeing these values.
    """

    model_config = ConfigDict(extra="forbid")

    derived_race: str | None = None
    derived_ethnicity: str | None = None
    derived_sex: str | None = None
    applicant_age: str | None = None
    external_id: str | None = None
    county_code: str | None = None
    census_tract: str | None = None


@dataclass(frozen=True)
class FitParams:
    """Transform parameters fitted on the training frame and pinned to the model."""

    income_cap: float
    feature_ranges: dict[str, tuple[float, float]]  # p1/p99 in MODEL units
    raw_ranges: dict[str, tuple[float, float]]  # p1/p99 in RAW units, for counterfactuals
    #: Training median of every model feature, used to impute missing values.
    #: See `_impute_missing` for why this is not optional.
    feature_medians: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["feature_ranges"] = {k: list(v) for k, v in self.feature_ranges.items()}
        d["raw_ranges"] = {k: list(v) for k, v in self.raw_ranges.items()}
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> FitParams:
        return cls(
            income_cap=float(d["income_cap"]),
            feature_ranges={k: (float(v[0]), float(v[1])) for k, v in d["feature_ranges"].items()},
            raw_ranges={k: (float(v[0]), float(v[1])) for k, v in d["raw_ranges"].items()},
            feature_medians={k: float(v) for k, v in d.get("feature_medians", {}).items()},
        )


#: Model feature -> the raw applicant field it is derived from. Used by the
#: counterfactual search so it can move a human-meaningful quantity.
FEATURE_SOURCE: dict[str, str] = {
    "log_income": "income_usd",
    "log_loan_amount": "loan_amount_usd",
    "cltv": "cltv",
    "dti_band_ord": "dti_band",
    "loan_term": "loan_term",
    "interest_only": "interest_only",
    "balloon": "balloon",
    "neg_amortization": "neg_amortization",
    "tract_to_msa_income_pct": "tract_to_msa_income_pct",
    "units_ord": "units_band",
    "preapproval": "preapproval",
}


def constrained_features() -> list[str]:
    """Features carrying a non-zero monotone constraint."""
    cfg = get_config()
    return [f.name for f in cfg.features if f.monotone != 0]


def _impute_missing(out: pd.DataFrame, params: FitParams) -> pd.DataFrame:
    """Fill missing values on every model feature at the training median.

    This is not cosmetic, and it is not the usual "trees handle NaN" question.
    LightGBM does not honour monotone constraints across the missing bin, and the
    failure is not confined to the feature that is missing. Verified empirically on
    synthetic data with a +1 constraint on feature 0:

        clean train, clean predict                      0 violations
        NaN in an UNCONSTRAINED feature, train only     0 violations
        NaN in an UNCONSTRAINED feature, train+predict  44 violations
        NaN in an unconstrained feature, predict only   0 violations

    Any feature carrying NaN during training creates missing-bin splits, and a row
    routed through one of those splits escapes the constraint bookkeeping for the
    *constrained* features too. NaN that appears only at prediction time is
    harmless, because without missing-bin splits it simply follows default branches.

    So the guarantee holds if and only if no feature reaches training with a missing
    value. Missingness here is at most 0.8% (debt-to-income band), so median
    imputation is a small, disclosed intervention; the alternative is a model that
    claims constraints it does not honour, which would also invalidate the
    counterfactual binary search.

    Disclosed in validation pack section 3 and docs/limitations.md.
    """
    if not params.feature_medians:
        return out
    for name, median in params.feature_medians.items():
        if name in out.columns:
            out[name] = out[name].fillna(median)
    return out


def fit_params(train: pd.DataFrame) -> FitParams:
    """Fit the transform parameters on the training frame."""
    income_cap = float(np.nanpercentile(train["income_usd"], 99))
    bare = FitParams(income_cap, {}, {}, {})
    X = feature_matrix(train, bare)

    medians = {
        name: float(np.nanmedian(X[name].to_numpy(dtype=float))) for name in X.columns
    }
    X = _impute_missing(X, FitParams(income_cap, {}, {}, medians))

    feature_ranges: dict[str, tuple[float, float]] = {}
    for col in X.columns:
        vals = X[col].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            feature_ranges[col] = (0.0, 1.0)
        else:
            feature_ranges[col] = (
                float(np.percentile(vals, 1)),
                float(np.percentile(vals, 99)),
            )

    raw_ranges: dict[str, tuple[float, float]] = {}
    for feat, src in FEATURE_SOURCE.items():
        if src in ("dti_band", "units_band"):
            # Ordinal: range is over the code set actually observed.
            order = (
                schema.DTI_BAND_ORDER if src == "dti_band" else schema.UNITS_BAND_ORDER
            )
            raw_ranges[feat] = (0.0, float(len(order) - 1))
        elif src in ("interest_only", "balloon", "neg_amortization", "preapproval"):
            raw_ranges[feat] = (0.0, 1.0)
        else:
            vals = pd.to_numeric(train[src], errors="coerce").to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            raw_ranges[feat] = (
                (float(np.percentile(vals, 1)), float(np.percentile(vals, 99)))
                if len(vals)
                else (0.0, 1.0)
            )

    return FitParams(
        income_cap=income_cap,
        feature_ranges=feature_ranges,
        raw_ranges=raw_ranges,
        feature_medians=medians,
    )


def missingness_report(train: pd.DataFrame, params: FitParams) -> dict[str, float]:
    """Share of each feature imputed, for validation pack section 3.

    The pack states what was imputed and by how much; it does not hide the intervention.
    """
    bare = FitParams(params.income_cap, {}, {}, {})
    X = feature_matrix(train, bare)
    return {c: float(X[c].isna().mean()) for c in X.columns if X[c].isna().any()}


def feature_matrix(df: pd.DataFrame, params: FitParams) -> pd.DataFrame:
    """Build the ordered model matrix from a cleaned frame.

    Raises if any protected column would end up in the output.
    """
    cfg = get_config()
    cap = params.income_cap if params.income_cap > 0 else float("inf")

    income = pd.to_numeric(df["income_usd"], errors="coerce").clip(upper=cap)
    loan = pd.to_numeric(df["loan_amount_usd"], errors="coerce")

    out = pd.DataFrame(index=df.index)
    out["log_income"] = np.log1p(income)
    out["log_loan_amount"] = np.log1p(loan)
    out["cltv"] = pd.to_numeric(df["cltv"], errors="coerce").clip(0, 200)
    out["dti_band_ord"] = pd.to_numeric(df["dti_band_ord"], errors="coerce")
    out["loan_term"] = pd.to_numeric(df["loan_term"], errors="coerce").clip(60, 480)
    out["interest_only"] = pd.to_numeric(df["interest_only"], errors="coerce")
    out["balloon"] = pd.to_numeric(df["balloon"], errors="coerce")
    out["neg_amortization"] = pd.to_numeric(df["neg_amortization"], errors="coerce")
    out["tract_to_msa_income_pct"] = pd.to_numeric(
        df["tract_to_msa_income_pct"], errors="coerce"
    ).clip(0, 300)
    out["units_ord"] = pd.to_numeric(df["units_ord"], errors="coerce")
    out["preapproval"] = pd.to_numeric(df["preapproval"], errors="coerce")

    # The excluded proxy, available only when a challenger explicitly asks for it.
    if "tract_minority_pct" in df.columns:
        out["tract_minority_pct"] = pd.to_numeric(
            df["tract_minority_pct"], errors="coerce"
        ).clip(0, 100)

    _assert_no_protected(out.columns)
    out = _impute_missing(out, params)
    ordered = [c for c in cfg.feature_names if c in out.columns]
    extra = [c for c in out.columns if c not in cfg.feature_names]
    return out[ordered + extra].astype(float)


def build_vector(applicant: RawApplicant, params: FitParams, feature_names: list[str]) -> np.ndarray:
    """One applicant -> one row, in the pinned feature order."""
    _assert_no_protected(feature_names)
    row = _applicant_to_row(applicant)
    frame = feature_matrix(pd.DataFrame([row]), params)
    missing = [f for f in feature_names if f not in frame.columns]
    if missing:
        raise KeyError(f"feature(s) {missing} could not be built from RawApplicant")
    return frame[feature_names].to_numpy(dtype=float)[0]


def _applicant_to_row(a: RawApplicant) -> dict[str, Any]:
    return {
        "income_usd": a.income_usd,
        "loan_amount_usd": a.loan_amount_usd,
        "property_value_usd": a.property_value_usd,
        "cltv": a.cltv,
        "dti_band": a.dti_band,
        "dti_band_ord": schema.DTI_BAND_TO_ORD.get(a.dti_band.strip()),
        "loan_term": a.loan_term,
        "interest_only": float(a.interest_only),
        "balloon": float(a.balloon),
        "neg_amortization": float(a.neg_amortization),
        "units_band": a.units_band,
        "units_ord": schema.UNITS_BAND_TO_ORD.get(a.units_band.strip()),
        "preapproval": float(a.preapproval),
        "tract_to_msa_income_pct": a.tract_to_msa_income_pct,
    }


def applicant_from_frame_row(row: pd.Series) -> RawApplicant:
    """Rebuild a RawApplicant from a cleaned-frame row, dropping everything protected."""
    return RawApplicant(
        income_usd=float(row["income_usd"]),
        loan_amount_usd=float(row["loan_amount_usd"]),
        property_value_usd=_opt_float(row.get("property_value_usd")),
        cltv=float(np.clip(_or(row.get("cltv"), 80.0), 0, 200)),
        dti_band=str(row.get("dti_band") or "30%-<36%"),
        loan_term=_or(row.get("loan_term"), 360.0),
        interest_only=bool(_or(row.get("interest_only"), 0.0)),
        balloon=bool(_or(row.get("balloon"), 0.0)),
        neg_amortization=bool(_or(row.get("neg_amortization"), 0.0)),
        units_band=str(row.get("units_band") or "1"),
        preapproval=bool(_or(row.get("preapproval"), 0.0)),
        tract_to_msa_income_pct=_opt_float(row.get("tract_to_msa_income_pct")),
    )


def context_from_frame_row(row: pd.Series) -> ApplicantContext:
    return ApplicantContext(
        derived_race=_opt_str(row.get("derived_race")),
        derived_ethnicity=_opt_str(row.get("derived_ethnicity")),
        derived_sex=_opt_str(row.get("derived_sex")),
        applicant_age=_opt_str(row.get("applicant_age")),
        external_id=_opt_str(row.get("row_id")),
        county_code=_opt_str(row.get("county_code")),
        census_tract=_opt_str(row.get("census_tract")),
    )


def _assert_no_protected(columns: Any) -> None:
    cfg = get_config()
    leaked = set(columns) & set(cfg.protected) | set(columns) & set(schema.PROTECTED_COLUMNS)
    if leaked:
        raise ProtectedLeakage(
            f"protected attribute(s) {sorted(leaked)} reached a feature vector. "
            "Protected attributes live only on the evaluation frame, keyed by row id."
        )


def _or(v: Any, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) else f


def _opt_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _opt_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None
