# Known limitations

This file is the single source for section 10 of the generated validation pack. It is
pulled verbatim rather than restated, so what a reader sees in the PDF is what the
repository maintains.

## What this model is, and what it is not

The model is trained on `action_taken`, the outcome the lender recorded: originated
or denied. It therefore learns to **replicate historical lender approval behaviour**,
including whatever bias that behaviour contained. It is **not a default-risk model**.
It has never seen a repayment outcome and cannot be used to estimate one.

This distinction matters for how every number in this document should be read. A
disparity measured here is a disparity in *approval behaviour*, not evidence about
creditworthiness. Improving a fairness metric means the model reproduces the
historical pattern less faithfully, which is a defensible goal for a lender but is
not the same as being more accurate.

## Public HMDA has no credit score

The single most predictive variable in consumer credit underwriting is absent from
the public Loan/Application Register. Performance figures here are therefore not
representative of what a production underwriting model would achieve, and the
apparent importance of the variables that *are* present is inflated, because they are
partly standing in for the one that is missing.

This is a property of the public data, not a defect in the pipeline. A production
deployment would have credit bureau data and would need this entire analysis redone
against it.

## Coarsened and rounded inputs

Public HMDA deliberately coarsens several fields to protect applicant privacy:

- `debt_to_income_ratio` arrives as a band, not a number. Exact integers are retained
  only between 36 and 49; everything else is bucketed. The model therefore sees a
  coarse ordinal, and a counterfactual on this factor can only ever name a band
  boundary rather than a precise ratio.
- `loan_amount` and `property_value` are rounded to the midpoint of the nearest
  $10,000 interval. Loan-to-value ratios derived from them inherit that granularity,
  which places a floor on how precisely any loan-to-value counterfactual can be
  quoted.
- `applicant_age` is banded, so the age analysis compares bands rather than years and
  cannot isolate the 62-year threshold that ECOA actually names.

## Derived demographic categories are reported, not observed

`derived_race` and `derived_ethnicity` are constructed by the FFIEC from what
applicants reported, with their own collection artefacts. Roughly one applicant in
six falls into a category that does not identify a single protected class — `Joint`,
`Free Form Text Only`, or `Not Available`. Those rows are excluded from the fairness
analysis rather than pooled, because a coefficient on "Race Not Available" cannot be
interpreted. The counts are reported in section 7, and the exclusion is not random:
non-response may itself correlate with group membership.

## Missing values are imputed to make the constraints real

LightGBM does not enforce monotone constraints for rows routed through a missing-value
split, and the failure is not confined to the feature that is missing — any feature
carrying missing values at training time can break the guarantee for the constrained
features as well. Every feature is therefore imputed at its training median before
fitting. The share imputed is under 1% for every feature and is reported in section 3,
but it is a real intervention: applicants with missing data are scored as though they
were typical on that factor.

## Clean metrics on a static dataset do not establish compliance

Everything in this document is measured on a fixed historical extract. That
establishes properties of a model artifact. It does not establish that a deployed
system is compliant, because deployment introduces population drift, operational
overrides, distribution channel effects, and interaction with the rest of the credit
process — none of which are visible here.

## Counterfactuals move one factor at a time

A counterfactual holds every other input constant. Real applicant characteristics do
not move independently: a smaller loan changes the loan-to-value ratio, and a higher
income usually accompanies other differences. The figures are therefore a correct
description of the model's decision surface and a simplification of the applicant's
actual options. They should be read as "this model would have decided differently",
not as financial advice.

## The alternative search covers a bounded space

The less discriminatory alternative search in section 8 evaluates a finite, enumerated
set of challenger models. Finding no better alternative within that space is not proof
that none exists. The space searched is recorded in `artifacts/lda/grid.json` so the
claim can be checked and extended rather than taken on trust.

The search also steers by one group's disparity, because a Pareto frontier needs a
single scalar per axis. Other groups are reported in full but are not optimised
against, and a model that improves the targeted metric may not improve theirs.

## Sign-in makes the audit trail attributable, not the system secure

Manual overrides now record the authenticated user rather than a name supplied in the
request, which is what makes the override trail worth keeping. That is a real
improvement to the governance story and a small one to the security posture.

The authentication here is local accounts with argon2id passwords and server-side
sessions. There is no federated identity, no second factor, no tenant isolation, and
the demo runs over plain HTTP on a loopback address. It is adequate for a
demonstration on public data and is not a model for protecting a real credit system.
What is and is not implemented is set out in `docs/security.md`.

## The ledger detects tampering; it does not prevent it

Append-only behaviour is enforced by SQLite triggers, and every row is chained by
hash. Together these stop accidental modification and detect after-the-fact edits.

They do not stop an adversary with write access to the database file, who can drop the
triggers and recompute every subsequent hash. Making the ledger genuinely
tamper-*proof* requires periodically anchoring the head hash somewhere outside the
system's control. That is out of scope here and is stated rather than implied.
