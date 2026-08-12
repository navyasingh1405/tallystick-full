"""The challenger space, enumerated deterministically before anything is trained.

Writing the grid to disk before the search starts makes the search plan itself an
artifact: a reviewer can see what was considered, not just what was found. "We looked
and found nothing better" only means something if the space that was looked at is on
the record.

The full cross product of every axis is 1,440 challengers, which is more compute than
the finding justifies. Instead: a full factorial over the two axes that matter most
for the fairness question (which features are available, and how hard the objective
is pushed toward parity), plus a seeded random sample of the rest. Deterministic
either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tally.config import get_config
from tally.ledger.canonical import content_hash

#: The excluded geographic proxy, offered as a search axis so its disparity cost can
#: be measured rather than assumed.
PROXY_FEATURE = "tract_minority_pct"


@dataclass(frozen=True)
class Challenger:
    """One point in the search space."""

    id: str
    subset: str
    features: tuple[str, ...]
    num_leaves: int
    max_depth: int
    min_child_samples: int
    reg_lambda: float
    fairness_weight: float
    origin: str  # "factorial" or "random"

    def config(self) -> dict[str, Any]:
        """Everything that affects the fitted model. This is what gets hashed."""
        return {
            "features": list(self.features),
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "reg_lambda": self.reg_lambda,
            "fairness_weight": self.fairness_weight,
        }

    def cache_key(self, data_fingerprint: str) -> str:
        return content_hash({"config": self.config(), "data": data_fingerprint})

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subset": self.subset,
            "origin": self.origin,
            **self.config(),
        }


@dataclass
class Grid:
    challengers: list[Challenger] = field(default_factory=list)
    subsets: dict[str, list[str]] = field(default_factory=dict)
    axes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_challengers": len(self.challengers),
            "axes": self.axes,
            "feature_subsets": self.subsets,
            "challengers": [c.as_dict() for c in self.challengers],
        }


def feature_subsets(baseline: list[str], top_proxies: list[str]) -> dict[str, list[str]]:
    """Feature sets to search over.

    `top_proxies` comes from the proxy scan, so the search is informed by the
    measurement rather than by guesswork: the features most associated with a
    protected attribute are the ones worth trying to remove.
    """
    p1 = top_proxies[0] if top_proxies else None
    p2 = top_proxies[1] if len(top_proxies) > 1 else None

    subsets: dict[str, list[str]] = {"baseline": list(baseline)}

    if p1:
        subsets[f"drop_{p1}"] = [f for f in baseline if f != p1]
    if p2:
        subsets[f"drop_{p2}"] = [f for f in baseline if f != p2]
    if p1 and p2:
        subsets[f"drop_{p1}_and_{p2}"] = [f for f in baseline if f not in (p1, p2)]

    subsets["drop_geography"] = [f for f in baseline if f != "tract_to_msa_income_pct"]
    subsets["minimal"] = [
        f for f in ("log_income", "log_loan_amount", "cltv", "dti_band_ord") if f in baseline
    ]
    subsets["drop_cltv"] = [f for f in baseline if f != "cltv"]
    # Deliberately bad: include the known geographic proxy, to price what using it
    # would cost in measured disparity rather than asserting that it is unacceptable.
    subsets["add_known_proxy"] = [*baseline, PROXY_FEATURE]

    return {k: v for k, v in subsets.items() if len(v) >= 2}


def build_grid(baseline: list[str], top_proxies: list[str], *, quick: bool = False) -> Grid:
    cfg = get_config()
    seed = int(cfg.lda["seed"])
    rng = np.random.default_rng(seed)

    subsets = feature_subsets(baseline, top_proxies)
    leaves_axis = [7, 15, 31, 63]
    depth_axis = [-1, 4, 6]
    child_axis = [20, 100, 500]
    lambda_axis = [0.0, 1.0, 10.0]
    weight_axis = [0.0, 0.25, 0.5, 1.0, 2.0]

    defaults = dict(
        num_leaves=int(cfg.model["params"]["num_leaves"]),
        max_depth=int(cfg.model["params"]["max_depth"]),
        min_child_samples=int(cfg.model["params"]["min_child_samples"]),
        reg_lambda=float(cfg.model["params"]["reg_lambda"]),
    )

    challengers: list[Challenger] = []
    seen: set[tuple] = set()

    def add(subset: str, origin: str, **over: Any) -> None:
        params = {**defaults, "fairness_weight": 0.0, **over}
        key = (subset, *sorted(params.items()))
        if key in seen:
            return
        seen.add(key)
        challengers.append(
            Challenger(
                id=f"c{len(challengers):03d}",
                subset=subset,
                features=tuple(subsets[subset]),
                origin=origin,
                num_leaves=int(params["num_leaves"]),
                max_depth=int(params["max_depth"]),
                min_child_samples=int(params["min_child_samples"]),
                reg_lambda=float(params["reg_lambda"]),
                fairness_weight=float(params["fairness_weight"]),
            )
        )

    # Full factorial over the two axes that speak directly to the fairness question.
    for subset in subsets:
        for weight in weight_axis:
            add(subset, "factorial", fairness_weight=weight)

    n_random = int(cfg.lda["quick_total"] if quick else cfg.lda["n_random"])
    if quick:
        n_random = max(0, n_random - len(challengers))

    subset_names = list(subsets)
    for _ in range(n_random):
        add(
            str(rng.choice(subset_names)),
            "random",
            num_leaves=int(rng.choice(leaves_axis)),
            max_depth=int(rng.choice(depth_axis)),
            min_child_samples=int(rng.choice(child_axis)),
            reg_lambda=float(rng.choice(lambda_axis)),
            fairness_weight=float(rng.choice(weight_axis)),
        )

    return Grid(
        challengers=challengers,
        subsets=subsets,
        axes={
            "feature_subsets": list(subsets),
            "num_leaves": leaves_axis,
            "max_depth": depth_axis,
            "min_child_samples": child_axis,
            "reg_lambda": lambda_axis,
            "fairness_weight": weight_axis,
            "design": (
                "Full factorial over feature subset x fairness weight, plus a seeded "
                "random sample of the full hyperparameter space. The full cross "
                "product would be 1,440 challengers; this covers the axes that bear "
                "on the fairness question exhaustively and samples the rest."
            ),
            "seed": seed,
        },
    )


def fairness_weights(
    groups, approved_rate_by_group: dict[str, float], weight: float
) -> np.ndarray:
    """Sample weights that upweight groups with lower base approval rates.

        w_i = 1 + lambda * (r_max / r_g(i) - 1)

    Monotone in lambda, reduces to uniform at lambda = 0, and needs the protected
    attribute at fit time only — never at inference. That last property is what makes
    it a defensible in-processing method rather than disparate treatment: the
    deployed model still never sees a protected attribute.
    """
    if weight <= 0 or not approved_rate_by_group:
        return np.ones(len(groups), dtype=float)

    r_max = max(approved_rate_by_group.values())
    out = np.ones(len(groups), dtype=float)
    values = groups.to_numpy()
    for group, rate in approved_rate_by_group.items():
        if rate <= 0:
            continue
        out[values == group] = 1.0 + weight * (r_max / rate - 1.0)
    return out
