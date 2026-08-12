# Project context

**Tallystick.** A governance layer around a credit model. The governance layer is the
product; the ML model is scaffolding. Optimize accordingly.

Named for the medieval tally stick: a debt recorded by notching a stick, then splitting
it lengthwise so each party held a half. Neither half means anything alone, and the
grain of a split piece of hazel cannot be forged. A hash chain, built in 1110.

Python package is `tally`, CLI entry point is `tally`.

## Hard rules

- Protected attributes (race, ethnicity, sex, age) NEVER enter a feature vector.
  `RawApplicant` has no field for them and `feature_matrix` raises on any protected
  column. There are tests for both. Do not weaken them.
- The ledger is append-only, enforced by SQLite triggers. Never add an UPDATE or
  DELETE path. Corrections are new rows with `supersedes` set.
- `CANON_VERSION` in `ledger/canonical.py` is mixed into every row hash. Changing it
  invalidates every chain ever written. It has been bumped once, at naming. It is now
  frozen.
- The override actor comes from the authenticated session, never from the request
  body. `OverrideRequest` has no `actor` field and must not gain one.
- Never hand-write content into the validation PDF. Every value must be read from a
  generated artifact. `assert_no_empty_values` runs on the rendered HTML.
- HMDA column names come from `src/tally/data/schema.py`, verified by hand against the
  FFIEC dictionary. Do not guess column names or invent codes. The Data Browser API
  silently drops unknown query parameters, so a typo returns unfiltered data that
  looks correct.
- Fairness formulas are specified in SPEC.md §4 F5. Do not substitute a different
  metric because it is easier to compute. The controlled analysis must be a fitted
  regression reporting p-values and confidence intervals.
- Route protection is a per-route dependency, not a prefix list. A new `/api/` route
  must be added to the protected or public table in `tests/test_auth_api.py`, which
  fails otherwise.

## Front-end rules

- **Never hide content with CSS that only script can un-hide.** Reveal animations opt
  in via a `.js-reveal` class set by script that has confirmed it can remove it, with
  an unconditional backstop. A transition that never advances, or an
  IntersectionObserver that never fires, must not be able to leave the page blank.
- `requestAnimationFrame` does not run in a tab that is not painting. Any value shown
  to a user must be written before the animation starts, never only from inside a rAF
  callback.
- The applicant form renders once and never re-renders; only the results region
  updates. Re-rendering a form drops the caret mid-keystroke.

## Commands

```
uv run tally setup | doctor | data | train | evaluate
uv run tally fairness | lda | drift | pack | verify-ledger
uv run tally batch-demo | demo | test | all
uv run tally user create|list|role|passwd|disable
```

## Style

Type hints everywhere. Pydantic for all API boundaries. Pure functions in `fairness/`
and `explain/` — no I/O inside computation. Comments explain why, not what, and are
worth writing where a reader would otherwise assume the obvious-but-wrong thing.
