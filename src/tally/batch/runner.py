"""Batch portfolio review.

A fair lending team does not evaluate one application at a time; it reviews a book of
business quarterly. This scores an uploaded portfolio, writes every decision to the
ledger as one chained batch, and runs the same fairness engine on that batch rather
than on the static evaluation frame.

The protected-attribute boundary is the delicate part. A batch CSV must carry
protected attributes — fairness cannot be measured without them — while those columns
must never reach a feature vector. The parser therefore splits each row into a
`RawApplicant` and an `ApplicantContext` at the point of reading, and only the former
can be passed to `build_vector`. The context is stored on the batch artifact, never on
the decision row.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from tally.config import get_config
from tally.data.features import (
    ApplicantContext,
    RawApplicant,
    applicant_from_frame_row,
    context_from_frame_row,
    feature_matrix,
)
from tally.explain.reason_codes import derive_reason_codes
from tally.explain.shap_engine import get_engine
from tally.ledger.store import DecisionRecord, LedgerStore
from tally.model.registry import LoadedModel, load_current

#: Columns the parser will read as context. Everything else must map to RawApplicant.
CONTEXT_COLUMNS = set(ApplicantContext.model_fields)


@dataclass
class RowError:
    row: int
    column: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {"row": self.row, "column": self.column, "message": self.message}


@dataclass
class ParsedBatch:
    applicants: list[RawApplicant] = field(default_factory=list)
    contexts: list[ApplicantContext] = field(default_factory=list)
    source_rows: list[int] = field(default_factory=list)
    errors: list[RowError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_batch(frame: pd.DataFrame) -> ParsedBatch:
    """Split each row into what the model may see and what only fairness may see."""
    out = ParsedBatch()
    applicant_fields = set(RawApplicant.model_fields)

    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        payload = {
            k: v
            for k, v in row.items()
            if k in applicant_fields and not (isinstance(v, float) and np.isnan(v))
        }
        try:
            applicant = RawApplicant.model_validate(payload)
        except ValidationError as exc:
            for err in exc.errors():
                out.errors.append(
                    RowError(
                        row=position,
                        column=str(err["loc"][0]) if err["loc"] else "(row)",
                        message=err["msg"],
                    )
                )
            continue

        context_payload = {
            k: (None if pd.isna(v) else str(v))
            for k, v in row.items()
            if k in CONTEXT_COLUMNS
        }
        out.applicants.append(applicant)
        out.contexts.append(ApplicantContext.model_validate(context_payload))
        out.source_rows.append(position)

    return out


def score_batch(
    parsed: ParsedBatch,
    *,
    model: LoadedModel | None = None,
    ledger: LedgerStore | None = None,
    filename: str | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Score a parsed portfolio, record it, and analyse it.

    SHAP is called once on the whole matrix rather than per application. That is the
    difference between ten thousand rows in seconds and ten thousand rows in an hour;
    the ledger write is a single transaction and is never the bottleneck.
    """
    cfg = get_config()
    model = model or load_current()
    ledger = ledger or LedgerStore.default()

    if not parsed.applicants:
        raise ValueError("no valid applications to score")

    if progress:
        progress.update(progress=0.1, detail="Building feature vectors")

    rows = [a.model_dump() for a in parsed.applicants]
    raw_frame = pd.DataFrame(
        [_applicant_frame_row(a, c) for a, c in zip(parsed.applicants, parsed.contexts, strict=True)]
    )
    X = feature_matrix(raw_frame, model.fit_params)[model.feature_names]

    if progress:
        progress.update(progress=0.3, detail=f"Scoring {len(X):,} applications")

    probability = model.probability(X)
    decisions = np.where(probability > model.threshold, "APPROVED", "DECLINED")

    if progress:
        progress.update(progress=0.5, detail="Computing explanations")

    attributions = get_engine(model).attribute(X)

    if progress:
        progress.update(progress=0.75, detail="Writing to the decision ledger")

    records = [
        DecisionRecord(
            model_version=model.version,
            model_hash=model.model_hash,
            input_payload=payload,
            raw_score=float(p),
            threshold=model.threshold,
            decision=str(d),
            reason_codes=derive_reason_codes(attr, str(d)).as_list(),
        )
        for payload, p, d, attr in zip(rows, probability, decisions, attributions, strict=True)
    ]
    summary = ledger.append_batch(records, filename=filename)
    written = ledger.batch_rows(summary.batch_id)

    results = raw_frame.copy()
    # Attach the modelled feature columns. Without them the batch's controlled
    # regression silently loses the transformed features (log income, log loan
    # amount) and measures a different quantity from the one section 7 reports.
    for column in X.columns:
        if column not in results.columns:
            results[column] = X[column].to_numpy()
    results.insert(0, "decision_id", [r["decision_id"] for r in written])
    results["probability"] = probability
    results["decision"] = decisions
    results["model_version"] = model.version
    results["reason_codes"] = [
        "; ".join(c["code"] for c in json.loads(r["reason_codes_json"])) for r in written
    ]

    if progress:
        progress.update(progress=0.85, detail="Running fairness analysis on the batch")

    batch_dir = cfg.paths.artifact("batches", summary.batch_id)
    batch_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(batch_dir / "results.csv", index=False)

    fairness = _batch_fairness(results, model)
    (batch_dir / "fairness.json").write_text(
        json.dumps(fairness, indent=2, default=float), encoding="utf-8"
    )

    payload = {
        "batch_id": summary.batch_id,
        "filename": filename,
        "row_count": summary.row_count,
        "first_seq": summary.first_seq,
        "last_seq": summary.last_seq,
        "batch_root_hash": summary.batch_root_hash,
        "created_at": summary.created_at,
        "model_version": model.version,
        "model_hash": model.model_hash,
        "approval_rate": float((decisions == "APPROVED").mean()),
        "declined": int((decisions == "DECLINED").sum()),
        "errors": [e.as_dict() for e in parsed.errors],
        "fairness": fairness,
        "results_csv": str(batch_dir / "results.csv"),
    }
    (batch_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )
    if progress:
        progress.update(progress=1.0, detail="Complete")
    return payload


def _applicant_frame_row(a: RawApplicant, c: ApplicantContext) -> dict[str, Any]:
    """One row carrying model inputs and protected context side by side.

    They coexist on this frame because fairness analysis needs both. They never
    coexist in a feature vector: `feature_matrix` raises on any protected column.
    """
    from tally.data import schema

    return {
        **a.model_dump(),
        "dti_band_ord": schema.DTI_BAND_TO_ORD.get(a.dti_band.strip()),
        "units_ord": schema.UNITS_BAND_TO_ORD.get(a.units_band.strip()),
        "interest_only": float(a.interest_only),
        "balloon": float(a.balloon),
        "neg_amortization": float(a.neg_amortization),
        "preapproval": float(a.preapproval),
        **c.model_dump(),
    }


def _batch_fairness(results: pd.DataFrame, model: LoadedModel) -> dict[str, Any]:
    """The same three analyses, scoped to this batch."""
    from tally.fairness.controlled import controlled_disparity
    from tally.fairness.disparity import raw_disparity
    from tally.fairness.groups import assign_groups, exclusion_summary
    from tally.fairness.proxy import proxy_scan

    frame = results.copy()
    frame["model_approved"] = frame["decision"] == "APPROVED"
    groups = assign_groups(frame)

    if groups["race"].notna().sum() < 30:
        return {
            "status": "insufficient_demographics",
            "note": (
                "This batch does not carry enough usable race and ethnicity data to "
                "support a fairness analysis. Supply derived_race and "
                "derived_ethnicity columns to enable it."
            ),
            "n": int(len(frame)),
        }

    controls = [f for f in model.feature_names if f in frame.columns]
    missing = [f for f in model.feature_names if f not in frame.columns]
    return {
        "controls_missing": missing,
        "status": "ok",
        "n": int(len(frame)),
        "approval_rate": float(frame["model_approved"].mean()),
        "group_exclusions": exclusion_summary(groups),
        "raw_disparity": raw_disparity(frame, outcome="model_approved", groups=groups),
        "controlled": controlled_disparity(
            frame, outcome="model_approved", controls=controls, groups=groups
        ),
        "proxy": proxy_scan(frame, features=controls, groups=groups),
    }


def run_demo_batch(
    n: int = 2000,
    *,
    seed: int = 20260810,
    ledger: LedgerStore | None = None,
    model: LoadedModel | None = None,
) -> dict[str, Any]:
    """Sample a portfolio from the out-of-time frame and run it through the batch path.

    Gives the demo a realistic book of business without needing a file to hand, and
    exercises exactly the same code an uploaded CSV would. `ledger` is injectable so
    tests can exercise the full path without appending to the real decision ledger.
    """
    from tally.data.splits import load_split

    frame = load_split("oot").sample(n=min(n, 50_000), random_state=seed)
    parsed = ParsedBatch()
    for position, (_, row) in enumerate(frame.iterrows(), start=1):
        parsed.applicants.append(applicant_from_frame_row(row))
        parsed.contexts.append(context_from_frame_row(row))
        parsed.source_rows.append(position)

    return score_batch(
        parsed, model=model, ledger=ledger, filename=f"sampled_portfolio_{n}.csv"
    )


def batch_template() -> pd.DataFrame:
    """A blank CSV showing every column the parser accepts."""
    from tally.data.splits import load_split

    try:
        sample = load_split("oot").head(5)
        rows = [
            _applicant_frame_row(applicant_from_frame_row(r), context_from_frame_row(r))
            for _, r in sample.iterrows()
        ]
        frame = pd.DataFrame(rows)
    except FileNotFoundError:
        columns = list(RawApplicant.model_fields) + list(CONTEXT_COLUMNS)
        frame = pd.DataFrame(columns=columns)
    keep = [c for c in frame.columns if c in set(RawApplicant.model_fields) | CONTEXT_COLUMNS]
    return frame[keep]


def load_batch_summary(batch_id: str) -> dict[str, Any]:
    cfg = get_config()
    path = cfg.paths.artifact("batches", batch_id, "summary.json")
    if not path.exists():
        raise FileNotFoundError(f"no batch {batch_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def batch_results_path(batch_id: str) -> Path:
    cfg = get_config()
    path = cfg.paths.artifact("batches", batch_id, "results.csv")
    if not path.exists():
        raise FileNotFoundError(f"no results for batch {batch_id}")
    return path


def new_batch_id() -> str:
    return str(uuid.uuid4())
