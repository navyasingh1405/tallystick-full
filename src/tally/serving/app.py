"""The decision service.

One process, one port, one worker. It serves the public landing page, the sign-in
flow, the dashboard, and the API from the same origin, so there is no CORS, no build
step, and nothing to deploy separately.

Two invariants hold across every route below:

* **Every path that produces a decision writes it to the ledger.** That includes the
  landing page's public preview, which would otherwise be an advertised exception to
  the product's central claim.
* **Route protection is declared per route**, as a dependency, rather than by a
  prefix list somewhere else. A route is public only if its own signature says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from tally.auth.deps import current_user_optional, requires
from tally.auth.store import User
from tally.config import get_config
from tally.serving import auth_routes, public_routes
from tally.serving.deps import Job, jobs, state
from tally.serving.models import (
    AdoptRequest,
    HealthResponse,
    OverrideRequest,
    ScoreRequest,
)

app = FastAPI(
    title="Tallystick",
    version="0.1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# --------------------------------------------------------------------------- cors
#
# Only needed for the split deploy, where the marketing page is served from a CDN and
# calls this service on another origin. Two properties matter here:
#
# * `allow_credentials` is False. Cookies must never ride a cross-origin request —
#   that is what would turn a permissive CORS policy into a way for another site to
#   act as a signed-in user. Sign-in happens on this origin only; the landing page
#   navigates here rather than authenticating across origins.
# * Origins are an explicit allowlist from configuration, never `*`. A wildcard would
#   let any page on the internet drive the public scorer from a visitor's browser.
_ALLOWED_ORIGINS = [o for o in get_config().service.get("allowed_origins", []) if o]
if _ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        max_age=600,
    )

app.include_router(auth_routes.router)
app.include_router(public_routes.router)


# --------------------------------------------------------------------------- errors


def _error(code: str, message: str, status: int = 400, **details: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "details": details}},
    )


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
    return _error(
        getattr(exc, "code", "http_error"), str(exc.detail), status=exc.status_code
    )


@app.exception_handler(FileNotFoundError)
async def missing_artifact(_: Request, exc: FileNotFoundError) -> JSONResponse:
    return _error("artifact_missing", str(exc), status=503)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    return _error("internal_error", f"{type(exc).__name__}: {exc}", status=500)


# --------------------------------------------------------------------------- health


@app.get("/api/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    s = state()
    model = s.model_or_none()
    chain = s.chain_status()
    return {
        "ok": model is not None and chain.ok,
        "model_version": model.version if model else None,
        "model_hash": model.model_hash if model else None,
        "ledger_height": chain.rows_checked,
        "chain_ok": chain.ok,
        "artifacts": s.artifact_status(),
    }


@app.get("/api/model/current")
def current_model(user: Annotated[User, Depends(requires("read"))]) -> dict[str, Any]:
    s = state()
    model = s.model
    return {
        "version": model.version,
        "hash": model.model_hash,
        "created_at": model.manifest["created_at"],
        "threshold": model.threshold,
        "features": [
            {
                "name": f.name,
                "display_name": f.display_name,
                "monotone": f.monotone,
                "unit": f.unit,
                "rationale": " ".join(f.rationale.split()),
            }
            for f in s.config.features
        ],
        "protected_attributes": list(s.config.protected),
        "metrics": model.metrics,
        "training": model.training_fingerprint,
    }


# --------------------------------------------------------------------------- scoring


@app.post("/api/score")
def score(
    request: ScoreRequest,
    user: Annotated[User, Depends(requires("score"))],
) -> dict[str, Any]:
    from tally.data.features import RawApplicant
    from tally.serving.scoring import score_applicant

    s = state()
    applicant = RawApplicant.model_validate(request.model_dump())
    result = score_applicant(applicant, model=s.model, ledger=s.ledger)
    return result.as_dict()


@app.get("/api/decisions/{decision_id}")
def get_decision(
    decision_id: str, user: Annotated[User, Depends(requires("read"))]
) -> dict[str, Any]:
    s = state()
    row = s.ledger.get(decision_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no decision {decision_id}")

    neighbours = s.ledger.neighbours(decision_id)
    return {
        "seq": row["seq"],
        "decision_id": row["decision_id"],
        "created_at": row["created_at"],
        "model_version": row["model_version"],
        "model_hash": row["model_hash"],
        "decision": row["decision"],
        "probability": row["raw_score"],
        "threshold": row["threshold"],
        "input": json.loads(row["input_json"]),
        "reason_codes": json.loads(row["reason_codes_json"]),
        "supersedes": row["supersedes"],
        "superseded_by": s.ledger.superseded_by(decision_id),
        "override_actor": row["override_actor"],
        "override_reason": row["override_reason"],
        "batch_id": row["batch_id"],
        "prev_hash": row["prev_hash"],
        "row_hash": row["row_hash"],
        "neighbours": {
            key: (
                None
                if r is None
                else {"seq": r["seq"], "decision_id": r["decision_id"], "row_hash": r["row_hash"]}
            )
            for key, r in neighbours.items()
        },
    }


@app.post("/api/decisions/{decision_id}/verify")
def verify_decision(
    decision_id: str, user: Annotated[User, Depends(requires("verify"))]
) -> dict[str, Any]:
    """Walk the chain from genesis, then replay the decision under its pinned model."""
    from tally.ledger.verify import replay, verify_chain

    s = state()
    if s.ledger.get(decision_id) is None:
        raise HTTPException(status_code=404, detail=f"no decision {decision_id}")
    return {
        "chain": verify_chain(s.ledger).as_dict(),
        "replay": replay(s.ledger, decision_id).as_dict(),
    }


@app.post("/api/decisions/{decision_id}/override")
def override_decision(
    decision_id: str,
    request: OverrideRequest,
    user: Annotated[User, Depends(requires("override"))],
) -> dict[str, Any]:
    """Record a manual override.

    The actor is the authenticated user, never a value the caller supplies. Before
    sign-in existed this field was free text, which made the audit trail a claim
    rather than a record: anyone could write anyone else's name into it.
    """
    s = state()
    try:
        written = s.ledger.override(
            decision_id,
            decision=request.decision,
            actor=user.email,
            reason=request.reason,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no decision {decision_id}") from None
    return {
        "decision_id": written.decision_id,
        "supersedes": written.supersedes,
        "seq": written.seq,
        "row_hash": written.row_hash,
        "note": "Recorded as a new ledger entry. The original decision is unchanged.",
    }


@app.get("/api/decisions/{decision_id}/adverse-action.pdf")
def adverse_action(
    decision_id: str, user: Annotated[User, Depends(requires("read"))]
) -> FileResponse:
    from tally.report.render import generate_adverse_action

    s = state()
    try:
        result = generate_adverse_action(decision_id, s.ledger)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no decision {decision_id}") from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return FileResponse(
        result.pdf_path,
        media_type="application/pdf",
        filename=f"adverse_action_{decision_id[:8]}.pdf",
    )


# --------------------------------------------------------------------------- ledger


@app.get("/api/ledger/recent")
def recent(
    user: Annotated[User, Depends(requires("read"))],
    limit: int = Query(default=25, ge=1, le=200),
) -> dict[str, Any]:
    s = state()
    rows = s.ledger.recent(limit)
    return {
        "height": s.ledger.height(),
        "head_hash": s.ledger.head_hash(),
        "rows": [
            {
                "seq": r["seq"],
                "decision_id": r["decision_id"],
                "created_at": r["created_at"],
                "decision": r["decision"],
                "probability": r["raw_score"],
                "batch_id": r["batch_id"],
                "supersedes": r["supersedes"],
                "row_hash": r["row_hash"],
            }
            for r in rows
        ],
    }


@app.get("/api/ledger/verify")
def verify_ledger(user: Annotated[User, Depends(requires("verify"))]) -> dict[str, Any]:
    """Always a fresh walk from genesis. The cached status behind /api/health exists
    only so the poll is cheap; anything a user explicitly asks to verify is verified."""
    from tally.ledger.verify import verify_chain

    s = state()
    s.invalidate_chain_cache()
    return verify_chain(s.ledger).as_dict()


# --------------------------------------------------------------------------- analysis


@app.get("/api/fairness")
def fairness(user: Annotated[User, Depends(requires("read"))]) -> dict[str, Any]:
    from tally.fairness.report import load_fairness_report

    return load_fairness_report()


@app.get("/api/drift")
def drift(user: Annotated[User, Depends(requires("read"))]) -> dict[str, Any]:
    from tally.drift.report import load_drift_report

    return load_drift_report()


@app.get("/api/lda/frontier")
def lda_frontier(user: Annotated[User, Depends(requires("read"))]) -> dict[str, Any]:
    from tally.lda.search import load_lda_summary

    summary = load_lda_summary()
    return {
        "headline": summary["headline"],
        "headline_challenger": summary.get("headline_challenger"),
        "objective": summary["objective"],
        "incumbent": summary["incumbent"],
        "frontier": summary["frontier"],
        "grid_axes": summary["grid_axes"],
        "n_challengers": summary["n_challengers"],
        "doctrine_note": summary["doctrine_note"],
        "points": [
            {
                "id": r["id"],
                "subset": r["subset"],
                "auc": r["auc"],
                "disparity_ame": r["disparity_ame"],
                "approval_rate": r["approval_rate"],
                "n_features": r["n_features"],
                "fairness_weight": r["params"].get("fairness_weight"),
                "on_frontier": any(f["id"] == r["id"] for f in summary["frontier"]),
                "is_incumbent": r["id"] == "incumbent",
            }
            for r in summary["results"]
            if r["disparity_status"] == "ok"
        ],
    }


@app.get("/api/lda/frontier.png")
def lda_plot(user: Annotated[User, Depends(requires("read"))]) -> FileResponse:
    path = get_config().paths.artifact("lda", "lda_frontier.png")
    if not path.exists():
        raise HTTPException(status_code=503, detail="run: uv run tally lda")
    return FileResponse(path, media_type="image/png")


@app.post("/api/lda/adopt")
def adopt(
    request: AdoptRequest, user: Annotated[User, Depends(requires("adopt"))]
) -> dict[str, Any]:
    from tally.lda.adoption import adoption_report

    try:
        return adoption_report(
            request.challenger_id, hold_approval_rate=request.hold_approval_rate
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


# --------------------------------------------------------------------------- batch


@app.get("/api/batch/template.csv")
def batch_template(user: Annotated[User, Depends(requires("batch"))]) -> Response:
    from tally.batch.runner import batch_template as template

    return Response(
        content=template().to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="batch_template.csv"'},
    )


@app.post("/api/batch")
async def submit_batch(
    user: Annotated[User, Depends(requires("batch"))],
    file: UploadFile | None = None,
    partial: bool = Query(default=False),
    sample: int = Query(default=0, ge=0, le=20_000),
) -> Any:
    """Upload a portfolio, or pass `sample=N` to draw one from the evaluation frame."""
    from tally.batch.runner import parse_batch, run_demo_batch, score_batch

    s = state()

    if sample:
        def sampled(job: Job) -> dict[str, Any]:
            job.update(progress=0.05, detail=f"Sampling {sample:,} applications")
            return run_demo_batch(sample)

        return jobs.submit(sampled).as_dict()

    if file is None:
        raise HTTPException(status_code=400, detail="upload a CSV file, or pass sample=N")

    raw = await file.read()
    filename = file.filename or "upload.csv"
    try:
        frame = pd.read_csv(pd.io.common.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"could not read CSV: {exc}") from None

    parsed = parse_batch(frame)
    if parsed.errors and not partial:
        # All-or-nothing by default. Silently scoring the rows that happened to parse
        # would put a partial portfolio in the ledger while the operator believed the
        # whole file had been reviewed.
        return _error(
            "batch_validation_failed",
            f"{len(parsed.errors)} row error(s). Fix them, or resubmit with "
            "partial=true to score only the valid rows.",
            status=422,
            errors=[e.as_dict() for e in parsed.errors[:100]],
            valid_rows=len(parsed.applicants),
            total_rows=int(len(frame)),
        )

    if not parsed.applicants:
        raise HTTPException(status_code=422, detail="no valid rows to score")

    def run(job: Job) -> dict[str, Any]:
        return score_batch(parsed, model=s.model, ledger=s.ledger, filename=filename, progress=job)

    return jobs.submit(run).as_dict()


@app.get("/api/batch/{job_id}")
def batch_status(
    job_id: str, user: Annotated[User, Depends(requires("batch"))]
) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id}")
    return job.as_dict()


@app.get("/api/batches/{batch_id}/results.csv")
def batch_results(
    batch_id: str, user: Annotated[User, Depends(requires("batch"))]
) -> FileResponse:
    from tally.batch.runner import batch_results_path

    try:
        path = batch_results_path(batch_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"no batch {batch_id}") from None
    return FileResponse(path, media_type="text/csv", filename=f"batch_{batch_id[:8]}.csv")


@app.get("/api/batches/{batch_id}")
def batch_summary(
    batch_id: str, user: Annotated[User, Depends(requires("batch"))]
) -> dict[str, Any]:
    from tally.batch.runner import load_batch_summary

    try:
        return load_batch_summary(batch_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"no batch {batch_id}") from None


# --------------------------------------------------------------------------- pack


@app.post("/api/validation-pack")
def build_pack(user: Annotated[User, Depends(requires("pack"))]) -> dict[str, Any]:
    from tally.report.render import generate_validation_pack

    def run(job: Job) -> dict[str, Any]:
        job.update(progress=0.1, detail="Reading model registry and artifacts")
        result = generate_validation_pack()
        return {
            "pdf_url": f"/api/validation-pack/file/{result.pdf_path.name}",
            "filename": result.pdf_path.name,
            "pages": result.pages,
            "size_bytes": result.size_bytes,
            "seconds": round(result.seconds, 1),
        }

    return jobs.submit(run).as_dict()


@app.get("/api/validation-pack/{job_id}")
def pack_status(
    job_id: str, user: Annotated[User, Depends(requires("pack"))]
) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id}")
    return job.as_dict()


@app.get("/api/validation-pack/file/{filename}")
def pack_file(
    filename: str, user: Annotated[User, Depends(requires("pack"))]
) -> FileResponse:
    path = get_config().paths.artifact("packs", Path(filename).name)
    if not path.exists() or path.suffix != ".pdf":
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


# --------------------------------------------------------------------------- pages

_WEB = get_config().paths.web


def _page(name: str) -> FileResponse:
    path = _WEB / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} is missing from web/")
    # No-store on the HTML shells: there is no build step and no content hashing, so
    # a cached shell silently serves last week's markup after an edit.
    return FileResponse(path, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/", include_in_schema=False)
def landing() -> FileResponse:
    """Public. Explains what the system does and lets a visitor score against the
    real model before deciding whether to ask for access."""
    return _page("landing.html")


@app.get("/login", include_in_schema=False)
def login_page() -> FileResponse:
    return _page("login.html")


@app.get("/app", include_in_schema=False)
def dashboard(
    user: Annotated[User | None, Depends(current_user_optional)],
) -> Response:
    """The dashboard shell.

    A signed-out visitor is redirected rather than shown a 401 page: arriving at a
    sign-in form is the useful outcome. The shell carries no data of its own — every
    figure in it comes from an authenticated API call — so the redirect is a courtesy
    rather than the security boundary. The boundary is on the API routes.
    """
    if user is None:
        return RedirectResponse(url="/login?next=/app", status_code=303)
    return _page("app.html")


# --------------------------------------------------------------------------- static

# Assets are public: they are stylesheets, scripts, and fonts, and the landing page
# needs them before anyone signs in. Mounted on explicit prefixes rather than at "/"
# so they cannot shadow an API or page route.


class NoCacheStatic(StaticFiles):
    """Serve assets without browser caching.

    There is no build step and no content hashing in the asset filenames, so a cached
    stylesheet or module silently serves stale code after an edit — which costs far
    more than the bytes saved on a localhost demo.
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Any:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


for _prefix in ("css", "js", "assets"):
    _directory = _WEB / _prefix
    if _directory.exists():
        app.mount(f"/{_prefix}", NoCacheStatic(directory=str(_directory)), name=_prefix)
