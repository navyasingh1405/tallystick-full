"""The one public API surface: the landing page's live scorer.

A visitor who has not signed in can score a synthetic application and watch the
decision, the reasons, and the counterfactual come back from the real model.

The decision is written to the ledger like any other. That is deliberate. The
product's central claim is that no path through this service produces a decision
without recording it, and carving out an exception for the marketing page would make
that claim false in exactly the place it is being advertised.

Because it is unauthenticated, it is rate limited by source address, and the request
schema is the same `RawApplicant` used everywhere else — so it is structurally
incapable of accepting a protected attribute.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from tally.data.features import RawApplicant
from tally.serving.models import ScoreRequest

router = APIRouter(prefix="/api/public", tags=["public"])

#: Per-source budget for the landing page scorer.
_MAX_REQUESTS = 30
_WINDOW_SECONDS = 60.0

_hits: dict[str, deque[float]] = defaultdict(deque)
_lock = Lock()


def _rate_limit(request: Request) -> None:
    source = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _lock:
        bucket = _hits[source]
        while bucket and now - bucket[0] > _WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= _MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail="Slow down a moment — the public preview is rate limited.",
                headers={"Retry-After": str(int(_WINDOW_SECONDS))},
            )
        bucket.append(now)


@router.post("/score")
def public_score(payload: ScoreRequest, request: Request) -> dict[str, Any]:
    """Score an application against the deployed model, and record it."""
    from tally.serving.deps import state
    from tally.serving.scoring import score_applicant

    _rate_limit(request)
    s = state()
    if s.model_or_none() is None:
        raise HTTPException(status_code=503, detail="no model is registered yet")

    result = score_applicant(
        RawApplicant.model_validate(payload.model_dump()), model=s.model, ledger=s.ledger
    )
    body = result.as_dict()
    body["recorded"] = {
        "seq": result.ledger.seq if result.ledger else None,
        "row_hash": result.ledger.row_hash if result.ledger else None,
        "note": (
            "This decision was written to the append-only ledger, exactly as an "
            "authenticated one would be. Nothing scores here without being recorded."
        ),
    }
    return body


@router.get("/frontier.png")
def public_frontier() -> Any:
    """The alternative-search plot, served publicly for the landing page.

    It is an aggregate chart of model metrics — no applicant appears in it, and the
    same figure is already published in the sample validation pack. Keeping it behind
    the sign-in would mean the landing page had to show a screenshot, which is
    exactly the sort of stale hand-maintained artifact this project argues against.
    """
    from fastapi.responses import FileResponse

    from tally.config import get_config

    path = get_config().paths.artifact("lda", "lda_frontier.png")
    if not path.exists():
        raise HTTPException(status_code=503, detail="run: uv run tally lda")
    return FileResponse(path, media_type="image/png")


@router.get("/summary")
def public_summary() -> dict[str, Any]:
    """Headline figures for the landing page, read from the generated artifacts.

    The landing page states no number that is not in an artifact some command
    produced — the same rule the validation pack follows.
    """
    from tally.fairness.report import load_fairness_report
    from tally.lda.search import load_lda_summary
    from tally.model.metrics import load_metric_artifact
    from tally.report.render import latest_pack_pages
    from tally.serving.deps import state

    s = state()
    model = s.model_or_none()
    out: dict[str, Any] = {
        "model_version": model.version if model else None,
        "model_hash": model.model_hash if model else None,
        "ledger_height": s.ledger.height(),
        "chain_ok": s.chain_status().ok,
        "pack_pages": latest_pack_pages(),
    }

    try:
        performance = load_metric_artifact("performance")
        out["performance"] = {
            "test_auc": performance["test"]["auc"],
            "oot_auc": performance["oot"]["auc"],
            "oot_n": performance["oot"]["n"],
            "train_year": None,
        }
    except FileNotFoundError:
        out["performance"] = None

    try:
        fairness = load_fairness_report()
        race = {g["group"]: g for g in fairness["raw_disparity"]["race"]["groups"]}
        controlled = fairness["controlled"]["specifications"]["race"]["primary"]["groups"]
        flagged = [
            {
                "group": name,
                "ame_pp": entry["ame_pp"],
                "p_value": entry["p_value"],
                "n": entry["n_focal"],
                "raw_rate": race.get(name, {}).get("approval_rate"),
                "raw_ratio": race.get(name, {}).get("ratio"),
            }
            for name, entry in controlled.items()
            if entry.get("status") == "ok" and entry.get("significant")
        ]
        flagged.sort(key=lambda g: g["ame_pp"])
        out["fairness"] = {
            "evaluation_n": fairness["evaluation_frame"]["n"],
            "evaluation_year": fairness["evaluation_frame"]["activity_year"],
            "four_fifths_flags": fairness["summary"]["four_fifths_flags"],
            "significant": flagged[:3],
            "proxy_top": (fairness["proxy"]["ranked"] or [{}])[0],
            "proxy_floor": fairness["proxy"]["negative_control"]["max_nmi"],
            "set_level_auc": fairness["proxy"]["set_level"].get("macro_auc"),
        }
    except (FileNotFoundError, KeyError):
        out["fairness"] = None

    try:
        lda = load_lda_summary()
        out["lda"] = {
            "headline": lda["headline"],
            "n_challengers": lda["n_challengers"],
            "frontier_size": len(lda["frontier"]),
            "incumbent_auc": lda["incumbent"]["auc"],
            "incumbent_disparity": lda["incumbent"]["disparity_ame"],
            "focal_group": lda["objective"]["focal_group"],
        }
    except (FileNotFoundError, KeyError):
        out["lda"] = None

    return out


def reset_rate_limits() -> None:
    """Test hook."""
    with _lock:
        _hits.clear()
