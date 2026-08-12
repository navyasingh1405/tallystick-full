"""Frontier maths, checked against hand-verified points.

If a dominated candidate reaches the frontier, the project's central claim — that a
comparably effective, less discriminatory alternative exists — is unsupported. This
is small enough to verify by hand, so it is.
"""

from __future__ import annotations

from tally.lda.pareto import Point, dominates, frontier, frontier_ids, headline


def P(id_: str, auc: float, disparity: float) -> Point:
    return Point(id=id_, auc=auc, disparity=disparity)


# --------------------------------------------------------------------------- dominance


def test_strictly_better_on_both_dominates():
    assert dominates(P("a", 0.90, -1.0), P("b", 0.85, -2.0))


def test_equal_on_one_and_better_on_the_other_dominates():
    assert dominates(P("a", 0.90, -1.0), P("b", 0.90, -2.0))
    assert dominates(P("a", 0.95, -1.0), P("b", 0.90, -1.0))


def test_identical_points_do_not_dominate_each_other():
    assert not dominates(P("a", 0.9, -1.0), P("b", 0.9, -1.0))


def test_a_tradeoff_is_not_domination():
    """Higher AUC but worse disparity: neither dominates. Both belong on the frontier."""
    assert not dominates(P("a", 0.95, -3.0), P("b", 0.90, -1.0))
    assert not dominates(P("b", 0.90, -1.0), P("a", 0.95, -3.0))


def test_dominance_uses_absolute_disparity_not_signed():
    """A +2pp disparity is as bad as -2pp. Sign carries direction, not severity."""
    assert not dominates(P("a", 0.90, 2.0), P("b", 0.90, -2.0))
    assert dominates(P("a", 0.90, 1.0), P("b", 0.90, -2.0))


# --------------------------------------------------------------------------- frontier


def test_dominated_candidates_are_excluded():
    points = [
        P("best_auc", 0.95, -3.0),
        P("balanced", 0.93, -1.5),
        P("fairest", 0.88, -0.4),
        P("dominated", 0.90, -2.0),  # worse than balanced on both
        P("also_dominated", 0.85, -1.6),  # worse than fairest on both
    ]
    assert frontier_ids(points) == {"best_auc", "balanced", "fairest"}


def test_frontier_is_ordered_by_descending_auc():
    points = [P("a", 0.88, -0.4), P("b", 0.95, -3.0), P("c", 0.93, -1.5)]
    assert [p.id for p in frontier(points)] == ["b", "c", "a"]


def test_a_strictly_dominating_chain_leaves_one_point():
    points = [P(f"p{i}", 0.9 - i * 0.01, -1.0 - i) for i in range(6)]
    assert frontier_ids(points) == {"p0"}


def test_exact_ties_on_auc_keep_only_the_lower_disparity():
    points = [P("keep", 0.90, -1.0), P("drop", 0.90, -2.0)]
    assert frontier_ids(points) == {"keep"}


def test_exact_ties_on_disparity_keep_only_the_higher_auc():
    points = [P("keep", 0.95, -1.0), P("drop", 0.90, -1.0)]
    assert frontier_ids(points) == {"keep"}


def test_float_noise_does_not_manufacture_frontier_points():
    """Rounding before dominance is the whole point. Without it, 1e-12 of arithmetic
    noise turns one model into fifty distinct frontier members."""
    points = [P("base", 0.900000, -1.0)] + [
        P(f"noise{i}", 0.900000 + i * 1e-12, -1.0 - i * 1e-12) for i in range(1, 50)
    ]
    assert len(frontier(points)) == 1


def test_differences_above_the_rounding_precision_are_kept():
    """1e-6 survives rounding to six decimals, so real dominance is still detected."""
    a, b = P("a", 0.900000, -1.000002), P("b", 0.900002, -1.000000)
    assert dominates(b, a)
    assert frontier_ids([a, b]) == {"b"}


def test_a_tradeoff_above_the_rounding_precision_keeps_both():
    """Higher AUC bought with higher disparity is a choice for a committee to make,
    not a domination. Both points stay on the frontier."""
    a, b = P("a", 0.900000, -1.000000), P("b", 0.900002, -1.000002)
    assert not dominates(b, a) and not dominates(a, b)
    assert frontier_ids([a, b]) == {"a", "b"}


def test_single_point_is_its_own_frontier():
    assert frontier_ids([P("only", 0.9, -1.0)]) == {"only"}


def test_empty_input_gives_an_empty_frontier():
    assert frontier([]) == []


def test_non_finite_candidates_are_dropped_not_ranked():
    """A challenger whose disparity could not be estimated must not be treated as
    perfectly fair."""
    points = [P("ok", 0.90, -1.0), P("nan", 0.99, float("nan")), P("inf", 0.99, float("inf"))]
    assert frontier_ids(points) == {"ok"}


# --------------------------------------------------------------------------- headline


def test_headline_reports_the_largest_reduction_within_tolerance():
    points = [
        P("incumbent", 0.9000, -4.0),
        P("close", 0.8964, -2.0),  # 0.4% lower AUC, 50% less disparity
        P("far", 0.8000, -0.5),  # much better on fairness, far outside tolerance
    ]
    result = headline(points, "incumbent", auc_tolerance=0.01)
    assert result.challenger_id == "close"
    assert "50%" in result.text
    assert result.disparity_reduction == 0.5


def test_headline_says_so_when_nothing_better_exists():
    points = [P("incumbent", 0.90, -1.0), P("worse", 0.80, -3.0)]
    result = headline(points, "incumbent")
    assert result.challenger_id is None
    assert "No alternative" in result.text


def test_headline_handles_a_free_lunch():
    """Lower disparity at no AUC cost should not be described as a tradeoff."""
    points = [P("incumbent", 0.90, -4.0), P("free", 0.90, -1.0)]
    result = headline(points, "incumbent")
    assert "no loss of AUC" in result.text


def test_headline_is_safe_when_the_incumbent_is_missing():
    assert headline([P("a", 0.9, -1.0)], "not_present").challenger_id is None
