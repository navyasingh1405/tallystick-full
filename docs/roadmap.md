# Roadmap

Things deliberately left out, so a reviewer can see they were decisions rather than
oversights. Each one is a real gap; none of them is hard in a way that would have taught
anyone anything.

## Moved into scope

- **Authentication.** Originally listed here as out of scope. It was built because the
  override endpoint recorded an actor string it did not verify, which made the audit
  trail a claim rather than a record: anyone could write anyone else's name into it.
  The actor is now the authenticated user, taken from the session, and the request
  schema has no field for it at all. What the implementation does and does not provide
  is set out in `docs/security.md`.

## Out of scope, on purpose

- **Federated identity and multi-tenancy.** Accounts are local, with a three-role
  model and no group mapping. A bank would want SSO against its own directory, SCIM
  provisioning, and tenant isolation. Route protection is one dependency, so that is
  where a real identity provider hooks in.
- **Postgres.** SQLite is the correct choice for a single-writer append-only ledger on
  one machine, and it makes the append-only triggers and the WAL semantics easy to
  demonstrate. A multi-writer deployment changes the concurrency story entirely and
  would need the chain construction rethought, not just the driver swapped.
- **Fannie Mae loan-performance merge.** Joining HMDA to loan performance would give a
  genuine default-risk target instead of a replicate-the-lender target, which is the
  single biggest limitation of this build. It is also a substantial data engineering
  project with no clean join key.
- **BISG surname and geocode race proxying.** Standard practice for fair lending
  analysis where race is unobserved. Not needed here because HMDA reports derived race
  directly, and adding a probabilistic proxy would make every disparity estimate
  conditional on the proxy's own error structure.
- **Real-time streaming drift.** The comparison here is annual because the data is
  annual. Streaming would be theatre on a dataset with one date field.
- **Model retraining orchestration and scheduling.** The revalidation triggers in
  section 11 say what should cause a rebuild; wiring them to a scheduler is deployment
  work, not governance design.
- **Containerisation beyond a single process.** The service is one uvicorn worker
  serving both the API and the dashboard, deliberately: the ledger is a single-writer
  database, so horizontal scaling requires a different storage design first.

## Known gaps worth closing next

- **Anchor the ledger head hash externally.** The chain detects tampering but cannot
  prevent an adversary with file access from rewriting it wholesale. Periodically
  publishing the head hash somewhere outside the system's control closes that, and is
  the difference between tamper-evident and tamper-proof.
- **Search a wider alternative space.** The current grid is 226 challengers over feature
  subsets, tree complexity, regularisation, and a reweighting objective. Adding
  adversarial debiasing, calibrated equalised odds post-processing, and monotone
  constraint variants would make the "no better alternative exists" claim considerably
  stronger.
- **Optimise the frontier for more than one group.** A Pareto frontier needs one scalar
  per axis, so the search steers by a single focal group. Multi-objective search over
  several protected groups at once is the honest generalisation.
- **Segment the counterfactual search.** It currently moves one factor at a time. Joint
  counterfactuals over two or three factors would be more actionable and considerably
  more work to keep verifiable.
- **Record model lineage across versions.** The registry versions models but does not
  record which model superseded which, or why. That is the model inventory a validation
  function would actually keep.
