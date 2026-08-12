"""5a — raw approval rate disparity.

The four-fifths rule is a screening heuristic borrowed from employment testing
(EEOC Uniform Guidelines). It is NOT the legal standard for lending, and presenting
it as authoritative is the fastest way to lose a knowledgeable reader. It is reported
here because it is a useful first screen and because examiners expect to see it, with
that caveat attached to the output itself rather than buried in prose.

The analysis that carries weight is the controlled regression in `controlled.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tally.config import get_config
from tally.fairness.groups import (
    ATTRIBUTE_LABEL,
    ATTRIBUTES,
    GROUP_ORDER,
    REFERENCE,
    assign_groups,
    min_group_n,
    resolve_outcome,
)

FOUR_FIFTHS_CAVEAT = (
    "The four-fifths rule is a screening heuristic drawn from the EEOC Uniform "
    "Guidelines on employee selection. It is not the legal standard for credit "
    "discrimination and a ratio above 0.80 does not establish compliance. It is "
    "reported as a first screen only; the controlled analysis is the substantive one."
)


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves sensibly for small n and rates near 0 or 1,
    where the normal approximation produces intervals outside [0, 1]."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def raw_disparity(
    df: pd.DataFrame, *, outcome: str = "model_approved", groups: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Approval rate per group, ratio to the highest-rate group, four-fifths flag."""
    cfg = get_config()
    threshold = float(cfg.fairness["four_fifths_threshold"])
    floor = min_group_n()

    y = resolve_outcome(df, outcome)
    groups = groups if groups is not None else assign_groups(df)

    report: dict[str, Any] = {
        "outcome": outcome,
        "n": int(len(df)),
        "overall_approval_rate": float(y.mean()),
        "min_group_n": floor,
        "four_fifths_threshold": threshold,
        "caveat": FOUR_FIFTHS_CAVEAT,
    }

    for attr in ATTRIBUTES:
        col = groups[attr]
        rows: list[dict[str, Any]] = []

        rates: dict[str, float] = {}
        for name in GROUP_ORDER[attr]:
            mask = (col == name).to_numpy()
            n = int(mask.sum())
            if n == 0:
                continue
            if n < floor:
                rows.append(
                    {
                        "group": name,
                        "n": n,
                        "status": "insufficient_sample",
                        "note": f"fewer than {floor} applicants; no rate is reported",
                        "approval_rate": None,
                        "ratio": None,
                        "four_fifths_flag": None,
                    }
                )
                continue
            approvals = int(y[mask].sum())
            rate = approvals / n
            rates[name] = rate
            lo, hi = _wilson_interval(approvals, n)
            rows.append(
                {
                    "group": name,
                    "n": n,
                    "status": "ok",
                    "approvals": approvals,
                    "approval_rate": rate,
                    "ci_low": lo,
                    "ci_high": hi,
                    "is_reference": name == REFERENCE[attr],
                }
            )

        # The four-fifths comparison is against the highest-rate group, which is how
        # the guideline is written — not against the designated reference group.
        best_rate = max(rates.values()) if rates else 0.0
        best_group = max(rates, key=rates.__getitem__) if rates else None
        for row in rows:
            if row["status"] != "ok":
                continue
            ratio = row["approval_rate"] / best_rate if best_rate else float("nan")
            row["ratio"] = ratio
            row["four_fifths_flag"] = bool(ratio < threshold)

        report[attr] = {
            "label": ATTRIBUTE_LABEL[attr],
            "reference_group": REFERENCE[attr],
            "highest_rate_group": best_group,
            "highest_rate": best_rate if rates else None,
            "groups": rows,
            "any_flag": any(r.get("four_fifths_flag") for r in rows),
            "n_insufficient": sum(1 for r in rows if r["status"] == "insufficient_sample"),
        }

    report["any_flag"] = any(report[a]["any_flag"] for a in ATTRIBUTES)
    return report
