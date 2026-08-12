# What this repository is

A complete snapshot of Tallystick, taken 2026-08-12: source, data, the trained
model, every generated artifact, and the decision ledger.

It exists alongside two others:

| Repository | Holds | Visibility |
|---|---|---|
| `aayusharmaaa/tallystick` | Source and the built static site. Data, models, artifacts, and the ledger are gitignored as regenerable. | private |
| `aayusharmaaa/tallystick-showcase` | Screenshots, generated PDFs, and an explanation. No code. | public |
| **this one** | **All of the above plus the state the other two leave out.** | private |

## What the snapshot includes that the working repository does not

| Path | Size | What it is |
|---|---|---|
| `data/` | 70 MB | FFIEC HMDA source extract and the built feature frames |
| `models/` | 1.1 MB | Trained LightGBM model v002, registry, and fit parameters |
| `artifacts/` | 198 MB | Every generated artifact: 497 files across packs, batch runs, fairness, LDA, metrics, and drift |
| `ledger/decisions.db` | 8 MB | The decision ledger — over 6,000 hash-chained entries, chain intact |

## The one exclusion

`ledger/accounts.db` is **not** here. It holds argon2id password hashes and live
session tokens, and anything committed to git is permanent in its history — so
credentials stay out even of a private repository.

Nothing is lost by that. The file is created on first run:

```bash
uv run tally user create
```

## Reproducing it instead

Everything in `data/`, `models/`, and `artifacts/` can be rebuilt from the source in
this repository:

```bash
uv sync
uv run tally all
```

That is the point of keeping them out of the working repository — and the point of
keeping them in this one is that a rebuild takes real time, and a snapshot of the
exact state that produced the figures on the live site does not.

## Verifying the chain

The ledger arrives with its hash chain intact. Confirm it:

```bash
uv run tally verify-ledger
```

At the time of the snapshot: **6,048+ rows, chain intact.**
