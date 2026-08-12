"""Protected class construction.

HMDA reports race and ethnicity as separate derived fields, each carrying categories
that do not identify a single protected class: `Joint` (applicant and co-applicant
differ), `Free Form Text Only`, and `Race Not Available`. Those rows are dropped
rather than pooled into a residual bucket, because a coefficient on "Race Not
Available" cannot be interpreted, let alone defended. How many rows were dropped is
always reported alongside the results.

Race and ethnicity are combined the way fair lending analysis conventionally does
it: the reference group is non-Hispanic White, and Hispanic or Latino is a focal
group in its own right regardless of reported race.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from tally.config import get_config
from tally.data import schema

RACE_REFERENCE: Final[str] = "NH_White"
SEX_REFERENCE: Final[str] = "Male"
AGE_REFERENCE: Final[str] = "35-64"

#: derived_race value -> focal group label.
_RACE_MAP: Final[dict[str, str]] = {
    "Black or African American": "Black",
    "Asian": "Asian",
    "American Indian or Alaska Native": "AIAN",
    "Native Hawaiian or Other Pacific Islander": "NHPI",
    "2 or more minority races": "TwoOrMoreMinority",
}

#: Ordered for stable presentation; the reference group comes first.
RACE_GROUP_ORDER: Final[tuple[str, ...]] = (
    "NH_White",
    "Black",
    "Hispanic",
    "Asian",
    "AIAN",
    "NHPI",
    "TwoOrMoreMinority",
)

SEX_GROUP_ORDER: Final[tuple[str, ...]] = ("Male", "Female")

#: ECOA protects applicants aged 62 and over. HMDA bins age, so the closest
#: available construction is the 65-and-over bands; `<25` is reported separately
#: because young applicants are a distinct and frequently raised concern.
AGE_GROUP_ORDER: Final[tuple[str, ...]] = ("35-64", "62_plus", "under_25")

ATTRIBUTES: Final[tuple[str, ...]] = ("race", "sex", "age")

REFERENCE: Final[dict[str, str]] = {
    "race": RACE_REFERENCE,
    "sex": SEX_REFERENCE,
    "age": AGE_REFERENCE,
}

GROUP_ORDER: Final[dict[str, tuple[str, ...]]] = {
    "race": RACE_GROUP_ORDER,
    "sex": SEX_GROUP_ORDER,
    "age": AGE_GROUP_ORDER,
}

ATTRIBUTE_LABEL: Final[dict[str, str]] = {
    "race": "Race and ethnicity",
    "sex": "Sex",
    "age": "Age",
}


def _race_group(race: str | None, ethnicity: str | None) -> str | None:
    race = (race or "").strip()
    ethnicity = (ethnicity or "").strip()

    if ethnicity == "Hispanic or Latino":
        return "Hispanic"
    if ethnicity in schema.AMBIGUOUS_DEMOGRAPHIC or not ethnicity:
        return None
    if race in schema.AMBIGUOUS_DEMOGRAPHIC or not race:
        return None
    if race == "White":
        # Reaching here means ethnicity is "Not Hispanic or Latino".
        return RACE_REFERENCE
    return _RACE_MAP.get(race)


def _sex_group(sex: str | None) -> str | None:
    sex = (sex or "").strip()
    if sex in ("Male", "Female"):
        return sex
    return None


def _age_group(age: str | None) -> str | None:
    age = (age or "").strip()
    if age in schema.AGE_NA_CODES or not age:
        return None
    if age in ("65-74", ">74"):
        return "62_plus"
    if age == "<25":
        return "under_25"
    if age in ("35-44", "45-54", "55-64"):
        return AGE_REFERENCE
    # 25-34 sits between the young and the reference band; excluded from the age
    # comparison rather than assigned arbitrarily.
    return None


def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Map the derived demographic columns onto analysis groups.

    Returns one column per protected attribute, with NaN where the row cannot be
    assigned to a single class. Never mutates the input.
    """
    race = [
        _race_group(r, e)
        for r, e in zip(
            df.get("derived_race", pd.Series([None] * len(df))),
            df.get("derived_ethnicity", pd.Series([None] * len(df))),
            strict=False,
        )
    ]
    return pd.DataFrame(
        {
            "race": pd.Series(race, index=df.index, dtype="object"),
            "sex": df.get("derived_sex", pd.Series([None] * len(df), index=df.index)).map(_sex_group),
            "age": df.get("applicant_age", pd.Series([None] * len(df), index=df.index)).map(_age_group),
        },
        index=df.index,
    )


def group_counts(groups: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Assigned and excluded counts per attribute, for disclosure in the report."""
    out: dict[str, dict[str, int]] = {"assigned": {}, "excluded": {}}
    for attr in ATTRIBUTES:
        col = groups[attr]
        out["assigned"][attr] = int(col.notna().sum())
        out["excluded"][attr] = int(col.isna().sum())
    return out


def exclusion_summary(groups: pd.DataFrame) -> dict[str, dict[str, float | int | str]]:
    """Why rows were excluded, phrased for the validation pack."""
    total = len(groups)
    counts = group_counts(groups)
    return {
        attr: {
            "assigned": counts["assigned"][attr],
            "excluded": counts["excluded"][attr],
            "excluded_share": round(counts["excluded"][attr] / total, 4) if total else 0.0,
            "note": (
                "Rows whose reported category does not identify a single protected "
                "class (Joint, Free Form Text Only, Not Available) are excluded "
                "rather than pooled, because a coefficient on such a category cannot "
                "be interpreted."
            ),
        }
        for attr in ATTRIBUTES
    }


def with_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of `df` with the three group columns attached."""
    out = df.copy()
    groups = assign_groups(df)
    for attr in ATTRIBUTES:
        out[f"group_{attr}"] = groups[attr]
    # `group` is the race/ethnicity construction, the primary axis for the LDA search.
    out["group"] = groups["race"]
    return out


def resolve_outcome(df: pd.DataFrame, outcome: str) -> np.ndarray:
    """Coerce the named outcome column to a 0/1 array."""
    if outcome not in df.columns:
        raise KeyError(
            f"outcome column {outcome!r} not present. Expected one of: "
            f"{[c for c in df.columns if 'approv' in c.lower()]}"
        )
    return df[outcome].astype(bool).to_numpy().astype(int)


def min_group_n() -> int:
    return int(get_config().fairness["min_group_n"])
