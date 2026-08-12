# The sixty-second demo

Rehearse this. The whole point is that it never stalls.

## Before you start

```bash
uv run tally demo
```

The browser opens at `http://127.0.0.1:8787` on the **landing page**. Scroll it once
yourself before showing anyone: confirm the hero counters have populated, the
interactive scorer has returned a decline, and the fairness cards show numbers rather
than skeletons.

Then open the dashboard and check the left rail: model version present, chain dot
green. If the dot is amber or red, stop and run `uv run tally verify-ledger` before
showing anyone anything.

Seed the ledger so the recent-decisions table is not empty:

```bash
uv run tally batch-demo --n 2000
```

Create yourself a named account, so the override later has a real name against it:

```bash
uv run tally user create you@yourbank.com --role admin --name "Your Name"
```

**Two ways in, and the choice is part of the pitch.** If your audience wants to click
around themselves afterwards, show them the "open the demo" button — a guest session,
nothing to fill in. If you are driving, sign in as yourself, because the override
demonstration lands better when the audit trail says your name.

---

## 0:00 — The claim (from the landing page)

Start on `/`, not on the dashboard. Everything on that page is fetched live, which is
worth saying out loud once:

> "Every number on this page came from the running system a second ago. The decision
> counter, the disparity figure, the frontier plot. Nothing here is a screenshot,
> because a marketing page full of stale hand-maintained numbers is exactly the
> problem this thing exists to solve."

Drag a slider in the **Score an applicant** panel. The decision, the reasons, and the
counterfactual update — and the line underneath says which ledger entry it just
became.

> "That was the real model, and that decision is now in the append-only ledger. There
> is no path through this service that scores somebody without recording it —
> including this one, on the marketing page."

Then click **open the demo**, or sign in, and continue on the dashboard.

## 0:00 — The claim

> "A lender can build a credit model in a week and then spend six months getting it past
> model risk and fair lending review. This automates the second part. Four things that
> are normally separate projects — reproducibility, adverse action reasons, fairness
> testing, and the validation document — are byproducts of scoring one applicant."

---

## 0:08 — Score an applicant

Pane 1. Click **Stretched applicant**, then **Score application**.

Point at three things, in this order:

1. **The gauge.** "The score, and the threshold drawn as a tick. A decision is a number
   compared against a published cutoff."
2. **The reasons.** "Four principal reasons, in applicant-facing language. Regulation B
   requires specific reasons, and the CFPB has said model complexity is not an excuse
   for generic ones."
3. **The counterfactual.** "A debt-to-income ratio at or below 49% would have produced
   an approval. That number was found by binary search and then re-scored to verify it
   actually flips the decision before we showed it to you."

If asked how the reasons are computed:

> "SHAP, but the baseline is the interesting part. Explaining against the population mean
> answers 'why is this applicant unusual'. We need 'what would have had to be different',
> so the reference is a set of real applications sitting exactly at the decision
> threshold."

---

## 0:25 — Verify and replay

Click **Verify & replay →**. You land in pane 2 with the decision loaded.

Click **Verify & replay**. Let the three steps resolve. Say nothing until they finish.

Then:

> "It walked the hash chain from genesis, loaded the exact model version this decision
> was made with, and re-scored the stored application. The difference between the stored
> score and the recomputed score is exactly zero. That is 'explain the decision you made
> on this applicant in March' — answered in two seconds."

### The override, if you have thirty spare seconds here

Click **Override this decision**, choose APPROVED, and type a real reason. Then point
at the line above the button:

> "The name on this override comes from my session. There is no field in the API for
> it — if you try to send one, the request is rejected. Before sign-in existed this
> was free text, which meant the audit trail recorded who *claimed* to have made the
> decision. That is not an audit trail, it is a suggestion."

The original decision is untouched; the override is a new chained entry that
supersedes it. Both stay in the record, which is the point of recording either.

If someone asks whether the ledger can be edited: the append-only triggers are in the
database, corrections are new rows carrying `supersedes`, and — say this part —

> "The triggers stop accidents and the chain detects tampering. Neither stops someone
> with write access to the file who drops the triggers and recomputes every hash after
> it. Making it tamper-proof needs an external witness, and that is out of scope here.
> It's written down in the limitations."

That admission buys more credibility than any feature in the demo.

---

## 0:38 — Fairness

Press `g f`, or click **Fairness**.

Scroll to the controlled regression:

> "The four-fifths screen flags Hispanic applicants. But that's a screening heuristic
> borrowed from employment testing, and it isn't the legal standard for lending. This is
> the one that gets argued: a logistic regression of the model's decision on protected
> class plus the model's own credit factors. Hispanic applicants, minus 2.7 points of
> approval probability, p of 3.8 times ten to the minus 22. That disparity survives
> controlling for every credit factor the model uses."

Then the proxy scan:

> "The top-ranked feature is tract minority population percentage, which is not in the
> model — it's scanned to prove the detector works. And every scan carries a seeded
> random column so you can see the noise floor rather than take the threshold on trust."

---

## 0:48 — The alternative search

Scroll to the frontier plot.

> "Under disparate impact doctrine, a business-justified model is still challengeable if
> a comparably effective, less discriminatory alternative exists. Nobody can answer that
> by inspection. So: 226 challengers trained, 21 on the Pareto frontier. A model with
> 1.1% lower AUC achieves 51% lower controlled disparity."

Click **Impact of adopting** on a frontier row. While it trains:

> "A plot shows a trade exists. A committee needs to know who changes outcome."

When it lands:

> "Same approval volume — the challenger's threshold is recalibrated so it approves the
> same share, otherwise you're just measuring a calibration shift. 722 applications move
> in each direction. Hispanic applicants net plus 228. Disparity improves for every
> group. That is a decision a committee can actually take."

---

## 0:58 — The document

Press `g p`. Click **Generate validation pack**.

> "Twelve sections against the SR 11-7 validation pillars, cross-referenced to EU AI Act
> Annex IV. Every number in it is read from an artifact that some other command
> produced — nothing in it is typed. Retrain the model, regenerate, and it describes the
> new model with no editing."

Open the PDF. Scroll to section 1 and read the heading out loud:

> "'What this model does not do.' It does not estimate credit risk. It is trained on what
> the lender did, not on what the borrower did, so it replicates historical approval
> behaviour including whatever bias was in it. Public HMDA has no credit score. Every
> disparity in this document is a disparity in approval behaviour."

---

## If you have another thirty seconds

**The adversarial test.** This is the strongest thing in the repository.

```bash
uv run pytest tests/test_fairness_adversarial.py -v
```

> "A governance tool that has never been shown to catch anything is not a governance
> tool. That fixture builds a model with a deliberate disparity and an otherwise
> identical clean model. All three analyses flag the biased one. None fires on the clean
> one."

**Portfolio review.** Pane 5, sample 2,000. Two thousand applications scored, chained
into the ledger as one batch with a single receipt hash, and analysed — in about
thirteen seconds.

---

## Questions you will get

**"Is the model any good?"**
Test AUC 0.90, out-of-time 0.88. But that is not the interesting number, and without a
credit score it isn't comparable to a production underwriting model. The model is
scaffolding — the governance layer is the product.

**"Why Colorado?"**
Trained on 2024, holding 2025 out entirely. HMDA has no date field beyond the activity
year, so the following vintage is the only honest out-of-time frame, and it doubles as
the drift comparison.

**"Could you use this on a real model?"**
The ledger, the reason codes, the fairness engine, and the document generator are model
agnostic — they need a scorer and a feature contract. The parts that are HMDA-specific
are the data pipeline and the feature definitions.

**"What surprised you building it?"**
LightGBM silently stops honouring monotone constraints for any row routed through a
missing-value split, and it isn't limited to the feature that's missing. A model can
claim constraints it doesn't hold. There's a table in the README and a test that sweeps
every constrained feature across its range on real applications to check the property
rather than trusting the parameter.
