# Monitoring plan and revalidation triggers

This file is the source for section 11 of the generated validation pack.

## What is measured, and how often

## Input distribution drift

Population Stability Index on every model feature, with bin edges frozen from the
training vintage and applied unchanged to each comparison period. Bands follow the
conventional reading: below 0.10 stable, 0.10 to 0.25 moderate, 0.25 and above
significant. Monthly in production; here it is computed between the training year and
the out-of-time year.

## Score distribution drift

Population Stability Index and the two-sample Kolmogorov-Smirnov statistic on the
predicted probability. A score distribution can shift while every input distribution
looks stable, which is why it is watched separately.

Alerting is on effect size, never on p-values. At fifty thousand observations every
comparison is statistically significant, so a p-value-driven alert fires constantly
and is then ignored — which is how drift monitoring becomes decoration.

## Outcome and fairness monitoring

Approval rate overall and by protected group, and the controlled disparity estimate
from section 7, recomputed on each monitoring window. The controlled figure is the one
that matters: a raw rate can move because the applicant mix moved.

## Decision integrity

The hash chain is verified from genesis on every run of `tally verify-ledger`. A break
identifies the first affected row by sequence number. Any break is an incident, not a
metric.

## Revalidation triggers

Revalidation is required when any of the following occurs, without waiting for the
scheduled cycle:

- Population Stability Index at or above 0.25 on any model feature, or on the score
  distribution.
- A change of 3 percentage points or more in the overall approval rate that is not
  explained by a deliberate policy change.
- The controlled disparity estimate for any protected group becoming statistically
  significant where it previously was not, or growing by more than 1 percentage point.
- Any four-fifths screening flag appearing for a group that did not previously carry
  one.
- A new data vintage becoming available, which changes what the out-of-time frame is.
- Any change to the feature set, the decision threshold, or the training population.
- Any break in the decision ledger hash chain.
- A change in the regulatory perimeter that affects what this documentation must
  contain.

## Scheduled cycle

Full revalidation annually, following the release of a new HMDA vintage. Fair lending
analysis quarterly. Drift and integrity checks monthly.

## What monitoring cannot tell you

Every measurement described here compares the model against its own past behaviour and
against a historical dataset. None of it detects a systematic problem that was present
from the beginning, because the baseline itself would carry it. That is what the
alternative search in section 8 and the adversarial test in the repository's test
suite are for: they interrogate the model against a standard other than its own
history.
