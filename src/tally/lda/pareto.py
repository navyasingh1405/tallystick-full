"""Pareto frontier over (AUC up, |disparity| down).

Small enough to be obviously correct, important enough that it is tested against
hand-checked synthetic points. If dominated candidates leak onto the frontier, the
central claim of the whole feature — that a comparably effective, less
discriminatory alternative exists — is unsupported.

The rounding is the part that matters. Two challengers that differ by 1e-15 in AUC
are the same model for any practical purpose, but strict float comparison treats one
as dominating the other and manufactures a frontier point out of arithmetic noise.
Round first, then compare.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

AUC_DECIMALS = 6
DISPARITY_DECIMALS = 6


@dataclass(frozen=True)
class Point:
    """One challenger's position in the objective space."""

    id: str
    auc: float
    disparity: float  # signed; dominance uses the absolute value

    @property
    def abs_disparity(self) -> float:
        return abs(self.disparity)


def dominates(a: Point, b: Point) -> bool:
    """`a` dominates `b`: at least as good on both objectives, strictly better on one."""
    at_least_as_good = a.auc >= b.auc and a.abs_disparity <= b.abs_disparity
    strictly_better = a.auc > b.auc or a.abs_disparity < b.abs_disparity
    return at_least_as_good and strictly_better


def _rounded(points: list[Point]) -> list[Point]:
    return [
        replace(
            p,
            auc=round(p.auc, AUC_DECIMALS),
            disparity=round(p.disparity, DISPARITY_DECIMALS),
        )
        for p in points
        if math.isfinite(p.auc) and math.isfinite(p.disparity)
    ]


def frontier(points: list[Point]) -> list[Point]:
    """Non-dominated set, ordered by descending AUC.

    Sweep by descending AUC and keep a point only if it improves on the best
    absolute disparity seen so far. Ties on AUC are ordered by disparity so the
    better of two equal-AUC points is considered first.
    """
    pts = sorted(_rounded(points), key=lambda p: (-p.auc, p.abs_disparity))
    out: list[Point] = []
    best = math.inf
    for p in pts:
        if p.abs_disparity < best:
            out.append(p)
            best = p.abs_disparity
    return out


def frontier_ids(points: list[Point]) -> set[str]:
    return {p.id for p in frontier(points)}


@dataclass(frozen=True)
class Headline:
    text: str
    challenger_id: str | None
    auc_delta: float
    disparity_reduction: float


def headline(
    points: list[Point], incumbent_id: str, auc_tolerance: float = 0.01
) -> Headline:
    """Generate the money sentence from the frontier. Never hand-written.

    Among frontier points within `auc_tolerance` AUC of the incumbent, pick the one
    with the largest reduction in absolute disparity.
    """
    by_id = {p.id: p for p in _rounded(points)}
    incumbent = by_id.get(incumbent_id)
    if incumbent is None:
        return Headline("No incumbent model was present in the search results.", None, 0.0, 0.0)

    candidates = [
        p
        for p in frontier(points)
        if p.id != incumbent_id
        and incumbent.auc - p.auc <= auc_tolerance
        and p.abs_disparity < incumbent.abs_disparity
    ]
    if not candidates:
        return Headline(
            "No alternative in the searched space achieved lower measured disparity "
            f"within {auc_tolerance:.1%} AUC of the incumbent.",
            None,
            0.0,
            0.0,
        )

    best = min(candidates, key=lambda p: p.abs_disparity)
    auc_delta = incumbent.auc - best.auc
    reduction = (
        (incumbent.abs_disparity - best.abs_disparity) / incumbent.abs_disparity
        if incumbent.abs_disparity
        else 0.0
    )

    if auc_delta <= 0:
        text = (
            f"An alternative model achieves {reduction:.0%} lower controlled "
            "disparity with no loss of AUC."
        )
    else:
        text = (
            f"A model with {auc_delta / max(incumbent.auc, 1e-9):.1%} lower AUC "
            f"achieves {reduction:.0%} lower controlled disparity."
        )

    return Headline(text, best.id, auc_delta, reduction)
