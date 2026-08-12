"""The only module in this project that calls SHAP.

Why the baseline matters
------------------------
Explaining an applicant against the dataset mean answers "why is this applicant
unusual". Regulation B asks a different question: what would have had to be
different for approval. That is a question about the decision boundary, so the
reference point has to be the boundary.

The naive implementation subtracts a constant from the base value and calls it a
threshold baseline. That is not attribution — it relabels an offset while the
contributions still sum to `f(x) - E[f]`. The correct move is to change the
*background dataset* to a sample of real applicants sitting at the cutoff, so that
`E_B[f] = f*` by construction and the contributions sum to the decision margin.

The maths
---------
LightGBM's binary objective works in raw margin (log-odds) space:

    f(x) = sum_j phi_j(x) + E_B[f]        p(x) = sigmoid(f(x))

The threshold `t` is expressed on probability, but sigmoid is strictly monotone, so
it maps exactly onto a raw threshold:

    f* = logit(t)        APPROVE  <=>  p(x) > t  <=>  f(x) > f*

So all attribution happens in raw space and the conversion to probability happens
once, at the display boundary. `model_output="probability"` is never used: it takes
a different and slower code path and destroys additivity relative to `f*`.

The trap
--------
`feature_perturbation="tree_path_dependent"` is SHAP's default for a bare Booster
and it *silently ignores* the `data=` argument. Ship that by accident and the model
uses a mean baseline while the validation pack claims a threshold baseline. The
residual assertion below is what catches it.
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from tally.config import get_config
from tally.model.registry import LoadedModel


class BaselineNotAnchored(RuntimeError):
    """The SHAP background does not sit at the decision threshold."""


@contextlib.contextmanager
def _quiet_progress():
    """Suppress SHAP's tqdm bar for the duration of one call.

    SHAP draws an interventional-explanation progress bar straight to stderr with no
    option to turn it off in this version. It interleaves with CLI output and with
    pytest capture. Only stderr is redirected, and only around the explainer call,
    so nothing else in the process loses its error stream.
    """
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stderr(devnull):
            yield


def raw_threshold(threshold: float) -> float:
    """logit(t) — the decision boundary in the space SHAP attributes in."""
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    return math.log(threshold / (1.0 - threshold))


def threshold_background(
    booster,
    X_train: pd.DataFrame,
    threshold: float,
    n: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a background of real applicants whose mean score IS the decision boundary.

    The result is a set of real marginal applicants, which is both a defensible
    reference point for an adverse action notice and far easier to explain to a
    reviewer than an arithmetic shift of a population mean.

    Simply taking the rows nearest the boundary is not enough. The score density is
    rarely symmetric around the cutoff — here approvals outnumber declines nine to
    one — so the nearest rows sit mostly on one side and their mean drifts off the
    boundary. Instead, walk the candidates in order of distance and take the next
    one from whichever side pulls the running mean back toward the cutoff. The rows
    are still the closest available, and the residual is now bounded by one row's
    distance divided by n rather than left to chance.
    """
    f = np.asarray(booster.predict(X_train, raw_score=True), dtype=float)
    f_star = raw_threshold(threshold)
    offset = f - f_star

    order = np.argsort(np.abs(offset), kind="stable")
    above = [i for i in order if offset[i] >= 0]
    below = [i for i in order if offset[i] < 0]

    picked: list[int] = []
    running = 0.0
    ai = bi = 0
    for _ in range(min(n, len(order))):
        want_below = running > 0
        if want_below and bi < len(below):
            idx = below[bi]
            bi += 1
        elif not want_below and ai < len(above):
            idx = above[ai]
            ai += 1
        elif bi < len(below):
            idx = below[bi]
            bi += 1
        elif ai < len(above):
            idx = above[ai]
            ai += 1
        else:  # pragma: no cover - exhausted both sides
            break
        picked.append(idx)
        running += offset[idx]

    # Sorted so the artifact is byte-stable across runs; `seed` is retained for
    # signature compatibility with challenger training paths.
    del seed
    return X_train.iloc[np.sort(picked)].reset_index(drop=True)


@dataclass
class Attribution:
    """One applicant's SHAP decomposition, in raw margin space."""

    feature_names: list[str]
    values: np.ndarray  # phi_j
    base_value: float  # E_B[f], approximately f*
    raw_score: float  # f(x)
    raw_threshold: float  # f*

    @property
    def margin(self) -> float:
        """f(x) - f*. Positive means approved."""
        return self.raw_score - self.raw_threshold

    @property
    def residual(self) -> float:
        """How far the background sits from the boundary. Reported, not hidden."""
        return self.base_value - self.raw_threshold

    def sorted_negative(self) -> list[tuple[str, float]]:
        """Features working against the applicant, most damaging first."""
        pairs = [
            (name, float(v))
            for name, v in zip(self.feature_names, self.values, strict=True)
            if v < 0
        ]
        pairs.sort(key=lambda kv: kv[1])
        return pairs

    def as_dict(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.feature_names, self.values, strict=True)}


class ShapEngine:
    """Holds one TreeExplainer per model version.

    Constructing the explainer is the expensive part; scoring is not. The engine is
    cached per version so the API does not rebuild it per request.
    """

    def __init__(self, model: LoadedModel) -> None:
        cfg = get_config()
        self.model = model
        self.f_star = raw_threshold(model.threshold)

        if not model.background_path.exists():
            raise FileNotFoundError(
                f"shap_background.parquet missing from {model.directory}. The "
                "background is part of the pinned model artifact, not a runtime "
                "accident — retrain to regenerate it."
            )
        self.background = pd.read_parquet(model.background_path)[model.feature_names]

        # interventional + an explicit background is the whole point; the default
        # tree_path_dependent would discard the background without saying so.
        self.explainer = shap.TreeExplainer(
            model.booster,
            data=self.background,
            feature_perturbation="interventional",
            model_output="raw",
        )
        self.base_value = float(np.asarray(self.explainer.expected_value).ravel()[0])

        # shap subsamples an interventional background to 100 rows without saying
        # so. If that happens, the anchoring computed in threshold_background is not
        # the anchoring in force, and the residual below is measuring a sample of a
        # construction rather than the construction.
        used = getattr(self.explainer, "data", None)
        if used is not None and len(used) != len(self.background):
            raise BaselineNotAnchored(
                f"shap is using {len(used)} of the {len(self.background)} background "
                "rows. Set explain.shap_background_n to the number shap will keep, so "
                "the background that is balanced at the threshold is the one used."
            )

        tol = float(cfg.explain["baseline_residual_tolerance"])
        if abs(self.base_value - self.f_star) > tol:
            raise BaselineNotAnchored(
                f"background expected value {self.base_value:.4f} is "
                f"{abs(self.base_value - self.f_star):.4f} away from the decision "
                f"threshold {self.f_star:.4f} (tolerance {tol}). Either the background "
                "was not sampled at the boundary, or SHAP fell back to the "
                "tree_path_dependent path and ignored it."
            )

    @property
    def residual(self) -> float:
        return self.base_value - self.f_star

    def attribute(self, X: np.ndarray | pd.DataFrame) -> list[Attribution]:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        frame = pd.DataFrame(arr, columns=self.model.feature_names)

        with _quiet_progress():
            values = np.asarray(self.explainer.shap_values(frame, check_additivity=False))
        if values.ndim == 3:  # some SHAP versions return (n, features, outputs)
            values = values[:, :, 0]
        raw = self.model.raw_score(arr)

        return [
            Attribution(
                feature_names=list(self.model.feature_names),
                values=values[i],
                base_value=self.base_value,
                raw_score=float(raw[i]),
                raw_threshold=self.f_star,
            )
            for i in range(len(arr))
        ]

    def attribute_one(self, x: np.ndarray) -> Attribution:
        return self.attribute(x)[0]

    def sign_agreement(self, X: pd.DataFrame) -> dict[str, dict[str, object]]:
        """Do constrained features contribute in the direction their constraint implies?

        Monotone constraints bound the *function*, not per-observation attributions:
        interactions can still produce a contribution whose sign contradicts the
        constraint direction. This is reported as a diagnostic rather than asserted,
        because a diagnostic you might fail is more credible than an assertion tuned
        until it passes.

        Only *material* deviations from the reference are scored. A row whose feature
        value sits at the reference has a near-zero contribution whose sign is
        numerical noise; including those measures floating-point dust, not model
        behaviour. The cut is the median absolute deviation for that feature.
        """
        attributions = self.attribute(X)
        phi = np.vstack([a.values for a in attributions])
        bg_mean = self.background.mean(axis=0).to_numpy(dtype=float)
        Xv = np.asarray(X, dtype=float)

        out: dict[str, dict[str, object]] = {}
        for j, (name, direction) in enumerate(self.model.monotone.items()):
            if direction == 0:
                continue
            deviation = Xv[:, j] - bg_mean[j]
            finite = np.isfinite(deviation) & np.isfinite(phi[:, j])
            magnitude = np.abs(deviation[finite])

            if magnitude.size == 0 or np.allclose(magnitude, 0.0):
                out[name] = {
                    "rate": None,
                    "n_scored": 0,
                    "status": "not exercised — feature is constant across this sample",
                }
                continue

            cut = float(np.median(magnitude[magnitude > 0])) if (magnitude > 0).any() else 0.0
            material = finite & (np.abs(deviation) >= cut) & (np.abs(deviation) > 0)
            expected = np.sign(direction * deviation[material])
            actual = np.sign(phi[material, j])
            scored = (expected != 0) & (actual != 0)

            if scored.sum() == 0:
                out[name] = {"rate": None, "n_scored": 0, "status": "not exercised"}
                continue

            out[name] = {
                "rate": float((expected[scored] == actual[scored]).mean()),
                "n_scored": int(scored.sum()),
                "status": "ok",
            }
        return out


_engines: dict[str, ShapEngine] = {}


def get_engine(model: LoadedModel) -> ShapEngine:
    if model.version not in _engines:
        _engines[model.version] = ShapEngine(model)
    return _engines[model.version]


def clear_engines() -> None:
    _engines.clear()
