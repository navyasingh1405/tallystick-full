# Methodology notes

Design decisions that took thought, and the reasoning behind them. Kept separate from
the generated validation pack, which states what the system does rather than why it was
built that way.

## The SHAP baseline is a background dataset, not an offset

Regulation B asks what would have had to be different for approval. That is a question
about the decision boundary, so the reference point has to be the boundary.

The tempting implementation subtracts a constant from the SHAP base value and calls it a
threshold baseline. It is not attribution — the contributions still sum to
`f(x) − E[f]`; only the label changed.

The correct move is to change the background. LightGBM's binary objective is additive in
raw margin (log-odds) space:

```
f(x) = Σⱼ φⱼ(x) + E_B[f]        p(x) = σ(f(x))
```

Since σ is strictly monotone, a probability threshold `t` maps exactly onto a raw
threshold `f* = logit(t)`. Build the background from real applications whose mean score
is `f*`, and the contributions sum to `f(x) − f*`: the applicant's distance from the
decision boundary.

Sampling the rows nearest the cutoff is not sufficient. The score density is asymmetric
around the threshold — approvals outnumber declines nine to one here — so the nearest
rows sit mostly on one side and their mean drifts off the boundary. The construction
walks candidates in order of distance and takes the next one from whichever side pulls
the running mean back toward the cutoff. The residual is then bounded by one row's
distance divided by n rather than left to chance; measured at 1.5e−2 log-odds.

**The trap:** `feature_perturbation="tree_path_dependent"` is SHAP's default for a bare
Booster and *silently ignores* the `data=` argument. Ship that by accident and the system
uses a population-mean baseline while the validation document claims otherwise. Guarded
by an assertion at explainer construction and by a test that the anchored base value
differs from the naive one — measured at 0.0 versus 2.91.

## shap silently subsamples the background

`TreeExplainer` with `feature_perturbation="interventional"` caps the background at
**100 rows**. Hand it three hundred and it keeps a hundred, without a warning.

That is fatal to the construction above, in a way that is invisible. The background is
not a sample of anything — it is built by walking candidates in order of distance from
the cutoff and taking each from whichever side pulls the running mean back, so that
`E_B[f] = f*` holds *by construction*. Sampling from that set destroys the balance,
because the property belongs to the set as a whole and to no subset of it.

The symptom was that the residual came out at −1.5e−2 rather than the ~1e−4 the
construction should produce, and passed its tolerance anyway. It was found by
reimplementing SHAP in JavaScript for the marketing page and discovering the two
disagreed by up to 8e−2 — the Python was averaging over a hundred rows while the
JavaScript averaged over the three hundred it had been given.

Two changes. `explain.shap_background_n` is now 100, so the set that is balanced is
the set that is used; the residual is now −6.1e−4. And `ShapEngine` raises if
`explainer.data` is shorter than the background it was handed, so a future change to
either the config or the library cannot quietly reintroduce it.

The general lesson is worth more than the specific bug: a library silently discarding
part of an input is indistinguishable from working, and the only reason this surfaced
is that something else computed the same quantity a different way and disagreed.

## Monotone constraints and missing values

See the table in the README. The short version: any feature carrying NaN at training
time creates missing-bin splits, and a row routed through one escapes monotone-constraint
enforcement on the *constrained* features too. The constraint is a property of the
function only if no feature reaches training incomplete.

This matters beyond tidiness. The counterfactual search is a binary search, and binary
search on a non-monotone function returns an arbitrary point. A model that claims
constraints it does not hold would produce counterfactuals that are wrong in a way
nothing downstream would catch.

Mitigations, in order of strength: impute every feature at the training median; raise at
training if any feature arrives incomplete; and sweep each constrained feature across its
observed range on 500 real applications after training, asserting the score never moves
against its constraint.

## Why the fairness frame is the out-of-time year

Measuring fairness on data the model has never seen is both the stronger claim and, here,
the larger sample: 51,038 applications against 13,012 in the in-time holdout. Two groups
fall below the minimum sample threshold on that frame and are reported as insufficient
rather than given a spuriously precise ratio — which is the behaviour worth demonstrating
on real data rather than only in a test.

## Why the controls are the model's own feature set

Not a hand-picked subset, and not a richer set the model never saw. Using the model's own
features makes the surviving disparity mean something precise: disparity the model
produces **beyond what its own stated credit factors explain**. Any other choice invites
the question of why those controls and not others.

## Why the alternative search holds its control set fixed

Each challenger has a different feature set, so it is tempting to control each one for
its own features. That is wrong twice over. The frontier's horizontal axis would no
longer be a single quantity, and — worse — a challenger that used a proxy would look
*fairer*, because the proxy would absorb group membership among the controls. Holding
the controls fixed measures every model against one definition of "explained by
legitimate credit factors".

This was caught during the build: an earlier version resolved controls from the raw frame,
which lacked the two transformed income features, so the search reported disparities that
did not match section 7. Both are now computed by one function on one frame.

## Why the search steers by one group

A Pareto frontier needs one scalar per axis. The configured focal group for Colorado is
Hispanic, not because it is the conventional choice but because it is the group with a
well-powered, statistically significant disparity in this state and vintage: 6,350
applicants at −2.7pp, p < 1e−21, against 664 Black applicants at −0.9pp, p = 0.22.
Searching against an estimate indistinguishable from zero would be optimising sampling
noise.

Every group is still reported in full. The search summary records whether the chosen
group's incumbent disparity is significant and carries a warning if it is not, so the
choice stays auditable rather than buried in configuration.

## Why the four-fifths rule is reported but qualified

It is a screening heuristic from the EEOC Uniform Guidelines on employee selection, not
the legal standard for credit discrimination. Examiners expect to see it and it is a
useful first screen, so it is computed — with the caveat attached to the output object
itself, not to prose elsewhere that someone can quote the table without.

## Why drift alerts on effect size and never on p-values

At fifty thousand observations on each side, every two-sample test is significant
regardless of whether the change matters. A p-value-driven alert fires constantly, and an
alert that always fires is an alert nobody reads. This is the mechanism by which drift
monitoring becomes decoration, so the thresholds are PSI bands and a KS statistic
magnitude, with no p-value anywhere in the drift report — asserted by a test.

## Why the proxy threshold is calibrated rather than principled

Normalised mutual information magnitudes depend on sample size and binning, so no
absolute cutoff is a natural constant. The threshold is set so a known geographic proxy
is flagged while ordinary credit features are not, and every scan reports each feature's
multiple of a seeded random negative control so a reader can judge the calibration
directly. Saying "this is a calibrated heuristic" is more useful than implying a
threshold has theoretical standing it does not have.

A guard worth noting: scikit-learn's `normalized_mutual_info_score` returns **1.0** when
both labellings have a single distinct value, because it treats two trivial partitions as
perfectly agreeing. Unguarded, every constant feature ranks as a perfect proxy — which it
did, until the guard was added.

## Why the canonical form stores text and hashes the stored bytes

The verification path never re-serialises a parsed object; it hashes the bytes read back
from SQLite. That removes the entire class of bugs where the write path and the verify
path disagree about how to render a float, and where the chain then "breaks" for a reason
that has nothing to do with tampering. One canonicaliser serves the ledger and the
alternative-search cache keys alike.
