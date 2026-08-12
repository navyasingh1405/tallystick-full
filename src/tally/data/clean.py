"""Raw HMDA CSV -> modelling frame, via DuckDB.

Every column is read as VARCHAR and cast explicitly. This is not fussiness: the CSV
sniffer types `loan_to_value_ratio` as DOUBLE from the first few thousand rows and
then either errors or silently NULLs every `Exempt` further down the file.

Filters that the Data Browser API cannot express (occupancy, lien, loan type,
business purpose) are applied here, and the row count at each stage is recorded so
validation pack section 2 can state the lineage without anything being typed by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from tally.data import schema

#: Canonical name -> the name it carries in the Data Browser CSV header.
#: Only the divergences need an entry; everything else matches.
DATA_BROWSER_SOURCE: dict[str, str] = {
    "combined_loan_to_value_ratio": "loan_to_value_ratio",
    "derived_msa_md": "derived_msa-md",
}


@dataclass
class Stage:
    name: str
    rows: int
    note: str = ""


@dataclass
class CleanResult:
    frame_path: Path
    stages: list[Stage] = field(default_factory=list)

    @property
    def final_rows(self) -> int:
        return self.stages[-1].rows if self.stages else 0


def _src(canonical: str) -> str:
    """Quoted source column reference for the Data Browser header spelling."""
    return f'"{DATA_BROWSER_SOURCE.get(canonical, canonical)}"'


def _num_simple(canonical: str) -> str:
    """Numeric cast. TRY_CAST maps 'NA', 'Exempt', and '' to NULL, which is what
    we want — those are missing values, not zeros."""
    return f"TRY_CAST(TRIM({_src(canonical)}) AS DOUBLE)"


def _flag(canonical: str) -> str:
    """HMDA 1 = yes, 2 = no, 1111 = Exempt. Exempt becomes NULL, not False."""
    v = _num_simple(canonical)
    return f"CASE WHEN {v} = 1 THEN 1.0 WHEN {v} = 2 THEN 0.0 ELSE NULL END"


def build_frame(raw_csv: Path, out_parquet: Path, cfg_data: dict[str, Any]) -> CleanResult:
    """Filter and decode one state-year into a parquet modelling frame."""
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    filters = cfg_data["filters"]
    stages: list[Stage] = []

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2GB'")
    con.execute("PRAGMA threads=4")

    # all_varchar is mandatory — see module docstring.
    con.execute(
        f"""
        CREATE VIEW raw AS
        SELECT * FROM read_csv(
            '{raw_csv.as_posix()}',
            all_varchar = true,
            header = true,
            sample_size = -1
        )
        """
    )

    total = con.execute("SELECT count(*) FROM raw").fetchone()[0]
    stages.append(Stage("raw rows from API", total, f"{raw_csv.name}"))

    # The API already applied these two; assert rather than trust.
    _assert_in(con, "action_taken", filters["action_taken"])
    _assert_in(con, "loan_purpose", filters["loan_purpose"])
    stages.append(
        Stage(
            "action_taken in (1,3), purpose=1",
            total,
            "applied server-side; verified locally",
        )
    )

    # Filters the API cannot express. Applied one at a time so the lineage shows
    # what each one cost.
    local_filters = [
        ("occupancy_type", filters["occupancy_type"], "principal residence"),
        ("lien_status", filters["lien_status"], "first lien"),
        ("loan_type", filters["loan_type"], "conventional"),
        (
            "business_or_commercial_purpose",
            filters["business_or_commercial_purpose"],
            "not primarily commercial",
        ),
    ]
    where: list[str] = []
    for col, allowed, note in local_filters:
        vals = ", ".join(str(v) for v in allowed)
        where.append(f"TRY_CAST(TRIM({_src(col)}) AS INTEGER) IN ({vals})")
        n = con.execute(f"SELECT count(*) FROM raw WHERE {' AND '.join(where)}").fetchone()[0]
        stages.append(Stage(f"{col} in ({vals})", n, note))

    filtered_where = " AND ".join(where)

    con.execute(
        f"""
        CREATE VIEW filtered AS
        SELECT * FROM raw WHERE {filtered_where}
        """
    )

    # Decode into human units. Nothing is imputed except CLTV, which is derived
    # from the (midpoint-rounded) amounts when HMDA reports it as NA.
    dti_cases = "\n".join(
        f"            WHEN TRIM({_src('debt_to_income_ratio')}) = '{band}' THEN {ordinal}"
        for band, ordinal in schema.DTI_BAND_TO_ORD.items()
    )
    units_cases = "\n".join(
        f"            WHEN TRIM({_src('total_units')}) = '{band}' THEN {ordinal}"
        for band, ordinal in schema.UNITS_BAND_TO_ORD.items()
    )

    con.execute(
        f"""
        CREATE VIEW decoded AS
        SELECT
            row_number() OVER (
                ORDER BY {_src('lei')}, {_src('census_tract')}, {_src('loan_amount')},
                         {_src('income')}, {_src('action_taken')}, {_src('derived_race')},
                         {_src('derived_sex')}, {_src('applicant_age')}
            ) AS row_ordinal,

            TRY_CAST(TRIM({_src('activity_year')}) AS INTEGER)      AS activity_year,
            TRIM({_src('lei')})                                      AS lei,
            TRIM({_src('state_code')})                               AS state_code,
            TRIM({_src('county_code')})                              AS county_code,
            TRIM({_src('census_tract')})                             AS census_tract,
            TRIM({_src('derived_msa_md')})                           AS derived_msa_md,

            TRY_CAST(TRIM({_src('action_taken')}) AS INTEGER)         AS action_taken,
            (TRY_CAST(TRIM({_src('action_taken')}) AS INTEGER) = 1)   AS approved,

            {_num_simple('income')} * {schema.INCOME_MULTIPLIER}      AS income_usd,
            {_num_simple('loan_amount')}                              AS loan_amount_usd,
            {_num_simple('property_value')}                           AS property_value_usd,

            -- HMDA reports CLTV as a free string with NA/Exempt sentinels. When it
            -- is absent, derive it from the amounts; both are midpoint-rounded to
            -- $10,000 so the derived figure is coarse. Flagged for the lineage.
            COALESCE(
                {_num_simple('combined_loan_to_value_ratio')},
                100.0 * {_num_simple('loan_amount')}
                      / NULLIF({_num_simple('property_value')}, 0)
            )                                                         AS cltv_raw,
            ({_num_simple('combined_loan_to_value_ratio')} IS NULL)    AS cltv_derived,

            TRIM({_src('debt_to_income_ratio')})                      AS dti_band,
            CASE
{dti_cases}
                ELSE NULL
            END                                                       AS dti_band_ord,

            {_num_simple('loan_term')}                                AS loan_term,
            {_flag('interest_only_payment')}                          AS interest_only,
            {_flag('balloon_payment')}                                AS balloon,
            {_flag('negative_amortization')}                          AS neg_amortization,

            TRIM({_src('total_units')})                               AS units_band,
            CASE
{units_cases}
                ELSE NULL
            END                                                       AS units_ord,

            CASE
                WHEN TRY_CAST(TRIM({_src('preapproval')}) AS INTEGER) = 1 THEN 1.0
                WHEN TRY_CAST(TRIM({_src('preapproval')}) AS INTEGER) = 2 THEN 0.0
                ELSE NULL
            END                                                       AS preapproval,

            {_num_simple('tract_to_msa_income_percentage')}           AS tract_to_msa_income_pct,
            {_num_simple('tract_minority_population_percent')}        AS tract_minority_pct,

            TRIM({_src('derived_race')})                              AS derived_race,
            TRIM({_src('derived_ethnicity')})                         AS derived_ethnicity,
            TRIM({_src('derived_sex')})                               AS derived_sex,
            TRIM({_src('applicant_age')})                             AS applicant_age
        FROM filtered
        """
    )

    # Rows with no income or no loan amount cannot be scored on their two most
    # important features; they are dropped rather than imputed.
    essential = "income_usd IS NOT NULL AND income_usd > 0 AND loan_amount_usd IS NOT NULL AND loan_amount_usd > 0"
    n_essential = con.execute(f"SELECT count(*) FROM decoded WHERE {essential}").fetchone()[0]
    stages.append(Stage("income and loan amount present", n_essential, "not imputed"))

    year = con.execute("SELECT max(activity_year) FROM decoded").fetchone()[0]

    con.execute(
        f"""
        CREATE VIEW final AS
        SELECT
            printf('{year}-%07d', row_ordinal)                AS row_id,
            * EXCLUDE (row_ordinal, cltv_raw),
            LEAST(GREATEST(cltv_raw, 0.0), 200.0)             AS cltv
        FROM decoded
        WHERE {essential}
        """
    )

    n_final = con.execute("SELECT count(*) FROM final").fetchone()[0]
    n_cltv_derived = con.execute(
        "SELECT count(*) FROM final WHERE cltv_derived AND cltv IS NOT NULL"
    ).fetchone()[0]
    n_cltv_missing = con.execute("SELECT count(*) FROM final WHERE cltv IS NULL").fetchone()[0]
    stages.append(
        Stage(
            "modelling frame",
            n_final,
            f"CLTV derived for {n_cltv_derived:,}, still missing for {n_cltv_missing:,}",
        )
    )

    con.execute(f"COPY final TO '{out_parquet.as_posix()}' (FORMAT PARQUET)")
    con.close()

    return CleanResult(out_parquet, stages)


def _assert_in(con: duckdb.DuckDBPyConnection, col: str, allowed: list[int]) -> None:
    """Confirm the API actually applied a filter it claims to support."""
    vals = ", ".join(str(v) for v in allowed)
    bad = con.execute(
        f"SELECT count(*) FROM raw WHERE TRY_CAST(TRIM({_src(col)}) AS INTEGER) NOT IN ({vals})"
    ).fetchone()[0]
    if bad:
        raise ValueError(
            f"{bad:,} rows have {col} outside {allowed}. The Data Browser API drops "
            "unknown parameters silently, so this means the request was not filtered "
            "as intended. Check cal.data.download.build_params."
        )
