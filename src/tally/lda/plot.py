"""The frontier plot.

Palette-matched to the application so it reads as part of the product rather than a
matplotlib default dropped into a dark page. Rendered server-side as PNG and embedded
in both the dashboard and the validation pack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from tally.config import get_config  # noqa: E402

BG = "#0B0D10"
SURFACE = "#12151A"
LINE = "#232830"
FG = "#E6E9EF"
MUTED = "#8B93A1"
ACCENT = "#4CC2D6"
INCUMBENT = "#E8B14C"
DOMINATED = "#4A5361"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "text.color": FG,
            "axes.labelcolor": FG,
            "axes.edgecolor": LINE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": LINE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Inter", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "figure.dpi": 160,
        }
    )


def plot_frontier(
    results: list[Any],
    frontier_ids: set[str],
    incumbent_id: str,
    headline: str,
) -> Path:
    cfg = get_config()
    _style()

    usable = [r for r in results if r.disparity_status == "ok"]
    fig, ax = plt.subplots(figsize=(9, 5.4))

    dominated = [r for r in usable if r.id not in frontier_ids and r.id != incumbent_id]
    front = [r for r in usable if r.id in frontier_ids and r.id != incumbent_id]
    incumbent = next((r for r in usable if r.id == incumbent_id), None)

    if dominated:
        ax.scatter(
            [abs(r.disparity_ame) for r in dominated],
            [r.auc for r in dominated],
            s=26,
            c=DOMINATED,
            alpha=0.35,
            linewidths=0,
            label=f"Dominated challengers ({len(dominated)})",
        )

    if front:
        ordered = sorted(front, key=lambda r: abs(r.disparity_ame))
        ax.plot(
            [abs(r.disparity_ame) for r in ordered],
            [r.auc for r in ordered],
            color=ACCENT,
            linewidth=1.4,
            alpha=0.85,
            zorder=3,
        )
        ax.scatter(
            [abs(r.disparity_ame) for r in ordered],
            [r.auc for r in ordered],
            s=54,
            c=ACCENT,
            linewidths=0,
            zorder=4,
            label=f"Pareto frontier ({len(ordered)})",
        )

    if incumbent is not None:
        x, y = abs(incumbent.disparity_ame), incumbent.auc
        ax.scatter(
            [x], [y], s=190, facecolors="none", edgecolors=INCUMBENT,
            linewidths=2.2, zorder=6, label="Incumbent (deployed)",
        )
        ax.annotate(
            f"Incumbent\nAUC {y:.4f} · {incumbent.disparity_ame:+.2f}pp",
            xy=(x, y),
            xytext=(26, -34),
            textcoords="offset points",
            color=INCUMBENT,
            fontsize=9,
            ha="left",
            arrowprops=dict(arrowstyle="-", color=INCUMBENT, linewidth=1.0, alpha=0.8),
        )

    ax.set_xlabel(
        "Controlled disparity, absolute average marginal effect (percentage points)\n"
        "lower is fairer  →",
        labelpad=8,
    )
    ax.set_ylabel("Discrimination (AUC on the out-of-time frame)\nhigher is better  →", labelpad=8)
    ax.set_title("Less discriminatory alternative search", pad=26, loc="left", color=FG)
    ax.text(
        0.0,
        1.045,
        headline,
        transform=ax.transAxes,
        fontsize=10.5,
        color=ACCENT,
        ha="left",
        va="bottom",
    )

    ax.grid(True, linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    legend = ax.legend(
        loc="lower right", frameon=True, fontsize=9, facecolor=SURFACE, edgecolor=LINE
    )
    for text in legend.get_texts():
        text.set_color(MUTED)

    fig.tight_layout()
    path = cfg.paths.artifact("lda", "lda_frontier.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path
