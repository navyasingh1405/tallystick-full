"""HMDA column names, codes, and encodings — verified by hand against the FFIEC
public LAR data dictionary, not guessed.

Two facts that cost people days if they are not written down:

1. The Data Browser API CSV and the Snapshot Public LAR file use DIFFERENT
   headers for the same columns. Most differences are hyphen-vs-underscore, but
   `loan_to_value_ratio` (Data Browser) and `combined_loan_to_value_ratio`
   (Snapshot) are the same column under two names. A dash-to-underscore
   normalisation will NOT fix that one.

2. Several columns that look numeric are binned strings with sentinel values.
   `debt_to_income_ratio` is the worst: exact integers for 36-49, bins either
   side, plus `NA` and `Exempt`.

References
    https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/public-lar-schema
    https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields/
"""

from __future__ import annotations

import re
from typing import Final

# --------------------------------------------------------------------------- naming

#: Data Browser header -> canonical (Snapshot) name, for the cases where a plain
#: hyphen-to-underscore rewrite is not enough.
HEADER_ALIASES: Final[dict[str, str]] = {
    "loan_to_value_ratio": "combined_loan_to_value_ratio",
}


def normalize_header(col: str) -> str:
    """Map either source's header spelling onto the canonical Snapshot name."""
    c = col.strip().lower().replace("-", "_")
    c = re.sub(r"__+", "_", c)
    return HEADER_ALIASES.get(c, c)


#: Every column the pipeline reads. Anything not here is dropped at the DuckDB stage.
#: Names are canonical (post-`normalize_header`).
COLUMNS_USED: Final[tuple[str, ...]] = (
    # identity / provenance
    "activity_year",
    "lei",
    "state_code",
    "county_code",
    "census_tract",
    "derived_msa_md",
    # filters
    "action_taken",
    "loan_purpose",
    "occupancy_type",
    "lien_status",
    "loan_type",
    "business_or_commercial_purpose",
    # model inputs
    "loan_amount",
    "property_value",
    "income",
    "combined_loan_to_value_ratio",
    "debt_to_income_ratio",
    "loan_term",
    "interest_only_payment",
    "balloon_payment",
    "negative_amortization",
    "total_units",
    "preapproval",
    "tract_to_msa_income_percentage",
    # excluded from the model, retained for the proxy scan
    "tract_minority_population_percent",
    # protected — evaluation frame only, never a feature
    "derived_race",
    "derived_ethnicity",
    "derived_sex",
    "applicant_age",
)

#: Held out of every feature vector. Enforced by tests/test_model_protected.py.
PROTECTED_COLUMNS: Final[tuple[str, ...]] = (
    "derived_race",
    "derived_ethnicity",
    "derived_sex",
    "applicant_age",
)

# --------------------------------------------------------------------------- codes

ACTION_TAKEN: Final[dict[int, str]] = {
    1: "Loan originated",
    2: "Application approved but not accepted",
    3: "Application denied",
    4: "Application withdrawn by applicant",
    5: "File closed for incompleteness",
    6: "Purchased loan",
    7: "Preapproval request denied",
    8: "Preapproval request approved but not accepted",
}

LOAN_PURPOSE: Final[dict[int, str]] = {
    1: "Home purchase",
    2: "Home improvement",
    31: "Refinancing",
    32: "Cash-out refinancing",
    4: "Other purpose",
    5: "Not applicable",
}

OCCUPANCY_TYPE: Final[dict[int, str]] = {
    1: "Principal residence",
    2: "Second residence",
    3: "Investment property",
}

LIEN_STATUS: Final[dict[int, str]] = {1: "Secured by a first lien", 2: "Secured by a subordinate lien"}

LOAN_TYPE: Final[dict[int, str]] = {
    1: "Conventional",
    2: "Federal Housing Administration insured",
    3: "Veterans Affairs guaranteed",
    4: "USDA Rural Housing Service or Farm Service Agency guaranteed",
}

PREAPPROVAL: Final[dict[int, str]] = {1: "Preapproval requested", 2: "Preapproval not requested"}

#: Appears on business_or_commercial_purpose, interest_only_payment,
#: balloon_payment, negative_amortization, open_end_line_of_credit,
#: reverse_mortgage. Means the filer claimed a partial exemption.
EXEMPT_CODE: Final[int] = 1111

#: Sentinel strings that mean "no value", not "zero".
NULL_TOKENS: Final[frozenset[str]] = frozenset({"", "na", "n/a", "exempt", "null", "none"})

# --------------------------------------------------------------------------- bands

#: Ordinal encoding of the HMDA debt-to-income band string.
#:
#: Public data reports exact integers for 36-49 and coarse bins either side, so
#: the encoding has to accept both. Values ascend with risk; the model constraint
#: on the encoded feature is -1.
DTI_BAND_ORDER: Final[tuple[str, ...]] = (
    "<20%",
    "20%-<30%",
    "30%-<36%",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "43",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50%-60%",
    ">60%",
)

DTI_BAND_TO_ORD: Final[dict[str, int]] = {b: i for i, b in enumerate(DTI_BAND_ORDER)}

#: Human-facing label for each band, used in adverse action notices and
#: counterfactual sentences. Never show an applicant "dti_band_ord = 8".
DTI_BAND_LABEL: Final[dict[str, str]] = {
    "<20%": "below 20%",
    "20%-<30%": "between 20% and 30%",
    "30%-<36%": "between 30% and 36%",
    "50%-60%": "between 50% and 60%",
    ">60%": "above 60%",
    **{str(v): f"{v}%" for v in range(36, 50)},
}

#: Upper edge of each band as a percentage, for the counterfactual search. The
#: exact-integer bands map to themselves.
DTI_BAND_UPPER: Final[dict[str, float]] = {
    "<20%": 20.0,
    "20%-<30%": 30.0,
    "30%-<36%": 36.0,
    "50%-60%": 60.0,
    ">60%": 70.0,
    **{str(v): float(v) for v in range(36, 50)},
}

UNITS_BAND_ORDER: Final[tuple[str, ...]] = (
    "1",
    "2",
    "3",
    "4",
    "5-24",
    "25-49",
    "50-99",
    "100-149",
    ">149",
)

UNITS_BAND_TO_ORD: Final[dict[str, int]] = {b: i for i, b in enumerate(UNITS_BAND_ORDER)}

AGE_BAND_ORDER: Final[tuple[str, ...]] = (
    "<25",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65-74",
    ">74",
)

#: applicant_age sentinels: 8888 = not provided, 9999 = no co-applicant.
AGE_NA_CODES: Final[frozenset[str]] = frozenset({"8888", "9999"})

# --------------------------------------------------------------------------- derived demographics

DERIVED_RACE_VALUES: Final[tuple[str, ...]] = (
    "American Indian or Alaska Native",
    "Asian",
    "Black or African American",
    "Native Hawaiian or Other Pacific Islander",
    "White",
    "2 or more minority races",
    "Joint",
    "Free Form Text Only",
    "Race Not Available",
)

DERIVED_ETHNICITY_VALUES: Final[tuple[str, ...]] = (
    "Hispanic or Latino",
    "Not Hispanic or Latino",
    "Joint",
    "Ethnicity Not Available",
    "Free Form Text Only",
)

DERIVED_SEX_VALUES: Final[tuple[str, ...]] = ("Male", "Female", "Joint", "Sex Not Available")

#: Categories that cannot be interpreted as a single protected class. Dropped from
#: the regression frame rather than lumped into a residual bucket — a coefficient on
#: "Race Not Available" is uninterpretable. The dropped count is always reported.
AMBIGUOUS_DEMOGRAPHIC: Final[frozenset[str]] = frozenset(
    {
        "Joint",
        "Free Form Text Only",
        "Race Not Available",
        "Ethnicity Not Available",
        "Sex Not Available",
    }
)

# --------------------------------------------------------------------------- units

#: HMDA reports gross annual income in thousands of dollars.
INCOME_MULTIPLIER: Final[int] = 1_000

#: loan_amount and property_value are rounded to the midpoint of the nearest
#: $10,000 interval, so every value ends in 5000. This caps LTV resolution and is
#: disclosed in docs/limitations.md.
AMOUNT_ROUNDING_INTERVAL: Final[int] = 10_000
