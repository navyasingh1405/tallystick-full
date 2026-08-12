# Regulatory grounding

Pointers to the frameworks this project is structured against. **These are not
quotations, and none of this is a claim of compliance.** Verify current text before
relying on any of it. The machine-readable version that drives appendix 12 of the
generated pack lives in [`config/regulatory_map.yaml`](../config/regulatory_map.yaml).

## ECOA / Regulation B — 12 CFR 1002

Section 1002.9 governs notification of action taken. A notice of adverse action must
contain:

- a statement of the action taken;
- the name and address of the creditor;
- the ECOA notice at §1002.9(b)(1), which this project reproduces verbatim in the
  generated adverse action notice;
- the name and address of the federal agency administering compliance for that creditor;
- either a statement of the **specific principal reasons** for the adverse action, or
  disclosure of the applicant's right to request them within 30 days.

§1002.9(b)(2) requires that the reasons be specific and indicate the principal reasons.
The CFPB has stated that the complexity of a model does not excuse generic reasons —
which is precisely the pressure this project's reason codes and counterfactuals are built
against, and why the SHAP baseline is anchored at the decision threshold rather than at a
population mean.

**Implemented by:** `src/tally/explain/reason_codes.py`, `src/tally/explain/counterfactual.py`,
`src/tally/report/templates/adverse_action.html.j2`.

## Fed SR 11-7 and OCC 2011-12 — model risk management

Supervisory guidance on model risk management, organised around three pillars:

1. **Conceptual soundness** of development — variable selection, design choices,
   documented assumptions and limitations, and consideration of alternative approaches.
2. **Ongoing monitoring** — process verification and benchmarking after deployment.
3. **Outcomes analysis** — comparing model output against actual outcomes, including by
   segment.

The generated validation pack's twelve sections are organised against these pillars.
Section 8, the alternative search, is written against the "consideration of alternative
approaches" element specifically, which is usually the thinnest part of a validation
document because nobody has a systematic way to produce it.

**Implemented by:** `src/tally/report/sections.py` and the pack template.

## EU AI Act

- **Annex III point 5(b)** classifies AI systems intended to evaluate creditworthiness
  or establish credit scores for natural persons as high-risk.
- **Article 11 and Annex IV** set the required content of technical documentation.
- **Article 12** covers automatic logging over the system's lifetime.
- **Article 14** covers human oversight.
- **Article 72** covers post-market monitoring.

Standalone Annex III obligations were **deferred to 2 December 2027** by the Digital
Omnibus, with Council final approval on 29 June 2026.

Frame that deferral correctly: it moved the deadline, not the workload. Conformity
assessment on a high-risk system runs closer to a year than a quarter, and the technical
documentation required by Annex IV has to be produced from evidence generated during
development — not reconstructed afterwards. Building the documentation pipeline before
the deadline is the point, not a response to it.

The decision ledger is the Article 12 logging story: an append-only, hash-chained record
of every decision with the model version and input that produced it. The override path is
the Article 14 story: human intervention is recorded as a first-class event with an
actor and a documented reason, not as a silent edit.

**Implemented by:** `src/tally/ledger/`, `config/regulatory_map.yaml`.

## FCRA — 15 U.S.C. 1681

Where a consumer report informs a credit decision, additional disclosure duties attach —
notably §615(a) for adverse action based on a consumer report, and §609(g) for credit
score disclosure.

**No consumer report or credit score is used here**, because public HMDA contains
neither. The generated notice states that explicitly and explains that those disclosures
are therefore inapplicable and deliberately omitted. Including boilerplate that does not
apply would be worse than omitting it, and explaining the absence is more informative
than doing either silently.

## Disparate impact and the less discriminatory alternative

The doctrinal basis for section 8. Under disparate impact analysis, a practice with a
discriminatory effect may be defended by business necessity — but the defence can be
overcome by showing that a **comparably effective alternative with less discriminatory
effect** was available.

That framing is the reason the alternative search exists as code rather than as a
paragraph. A lender cannot establish that no such alternative exists by inspection, and
a regulator cannot disprove it by inspection either. What can be produced is an explicit,
recorded search over a stated hypothesis space, with the frontier and the space both
preserved as artifacts.

The honest limit is stated in the pack and in `docs/limitations.md`: the absence of a
better alternative **within a bounded search space** is not proof that none exists.

**Implemented by:** `src/tally/lda/`.
