# How Claude Science Powered This Project

**Project:** *When Does a Perturbation Model Know Enough?*
**Author:** Yash Raj
**Code:** https://github.com/yashraj59/RefineRx · **Models:** https://huggingface.co/yraj/RefineRx
**Annotation data:** https://github.com/yashraj59/tcell-perturbed-gene-annotation

---

## The one-paragraph version

Claude Science was the main workbench for this project, from the first literature search to the
final compiled PDF. I used it to survey the field, choose a method, build the working model, and —
most importantly — to run the *negative* experiments fast enough that they became the paper's
contribution rather than a footnote. Every number in the paper comes from a saved table produced in
Claude Science, and the whole reproducibility trail (code, checkpoints, per-perturbation signature
tables, figures, annotation provenance) is preserved as versioned artifacts.

---

## 1. Survey first — a graftability catalog that chose my method for me

I did not start by picking a model; I started by asking Claude Science which models could even *host*
the idea. The idea was to repurpose adaptive-computation halting (ACT / PonderNet) as a
**measurement** — how many rounds of iterative refinement a model needs to reach a perturbation's
endpoint response — rather than as a compute-saving stop rule.

Using Claude Science I built a **graftability catalog of 43 published perturbation models**, cloning
and reading each repository to find where a learned-halting loop could actually be inserted into the
forward pass. That review is what pointed me away from bespoke architectures and toward a **frozen
foundation backbone**: most models had no natural refinement axis, and the ones that did (diffusion
samplers, iterative refiners) coupled depth to effect size. The catalog is now Section 2 of the
paper, and 20 of the 43 models are ranked LOW/LOW–MEDIUM graftability — a negative that directly
motivated the final design.

## 2. Building the working readout

Once the catalog pointed me at a frozen backbone, I built the working model in Claude Science on
ARC Institute's pretrained **STATE Transition** foundation model (four frozen Replogle cell-line
checkpoints). The pieces I implemented and trained there:

- an **oracle-supervised halt head** grafted on top of the frozen backbone, where a naive learned
  gate collapses to a per-seed constant;
- a **magnitude-free per-round target** (cosine-direction, not full-magnitude response) so the depth
  signal cannot smuggle effect size back in;
- a **logit-lens per-layer decode**, reading each transformer layer as a model-implied intermediate
  response state to locate the refinement layer at which the endpoint estimate converges.

This cleared both gates I set: on K562 (968 perturbations) the halting depth E[N] is reproducible
(split-half ρ = 0.76), effect-independent (partial ρ = −0.04 controlling for #DE and cell count),
and recovers its oracle target (ρ = 0.61). Across four lines it reproduces within each line
(mean ρ = 0.80).

## 3. Where Claude Science mattered most — falsification at speed

The paper is falsification-first, and this is where Claude Science earned its place. It let me run
the negatives fast enough that they *became* the contribution instead of a footnote. In one project
I was able to run, save, and audit:

- **the naive-gate collapse** — a free learned halt gate pins to a constant for every threshold;
- **effect-size proxy checks** — showing free-learned depth is otherwise just a re-encoding of
  response magnitude (|ρ| up to ~0.9);
- **the identifiability wall across three model families** — scDiff, TxPert, and my own refiners,
  plus a 12-config ponder/prior/budget sweep, none of which produced a signature that is both
  reproducible and non-redundant;
- **cross-line and cross-donor portability tests** — the signature reproduces within a context
  (mean ρ = 0.80) but does not port across cell types (ρ = 0.14) or across resting donors (ρ ≈ 0),
  porting only where the biology itself converges (ρ = +0.49 at the stimulated CD4 48 h endpoint);
- **network-topology baselines** — a directed gene-regulatory network (CollecTRI) and the STRING
  protein–protein interaction graph, establishing that no model-free graph statistic reproduces the
  per-perturbation ordering of E[N] (best |ρ| = 0.23, below a 0.3 novelty ceiling) — the result that
  makes the descriptor genuinely novel;
- **the from-scratch pseudobulk attempt**, reported honestly as a limitation: a CD4-native model
  trained from scratch fits the response but its halting depth collapses (E[N] → 6.0, oracle
  stopping round degenerate for 100% of perturbations), which I attribute to pseudobulk aggregation
  rather than claiming as a clean refutation.

Every one of these is backed by a saved table in the repository. That is the discipline Claude
Science made cheap: a negative you can regenerate and audit is a negative you can publish.

## 4. Claude Code for the heaviest data step

I used **Claude Code** for one specific, heavy job: processing the roughly **22-million-cell**
Marson/Pritchard CD4+ T-cell CRISPRi screen down to the **two-donor, single-cell substrate**
(50 cells per perturbation, ~3.9M cells) that the halting analysis actually runs on. This was the
single largest data-handling step in the project, and keeping it separate kept the analysis workbench
clean.

## 5. autocollect-bio — my own Claude Science skill for zero-fabrication annotation

For the drug- and toxicity-target layer I built and used **autocollect-bio**, a Claude Science skill
I developed for **bulk biological annotation with zero fabrication**. Its single invariant is that a
model prior never stands in for a retrieved value: every cell in the output table is either a value
that came back from a named source on a recorded date, or the explicit token `NOT_FOUND`.

I used it to build the **11,526-gene** CD4+ T-cell perturbed-gene annotation from public databases —
**Open Targets, ChEMBL, gnomAD, DepMap, Pharos, STRING, and FDA** — with a **source tag on every
value**, so the annotation stays auditable and every candidate reads as a hypothesis ("same class as
anchor X"), not a validated target. Inside that table sits a **~1,796-gene** druggable/tractable deep
subset — the rows carrying an actionable drug or tractability call. The full zero-fabrication
annotation table (11,526 perturbed genes, per-value provenance, `NOT_FOUND` vs `NOT_FETCHED` strictly
separated) is released as a standalone, reproducible dataset:
**github.com/yashraj59/tcell-perturbed-gene-annotation**.

This is the part of the project where fabrication would have been both easy and invisible — no reader
checks 10,000 rows by hand — so making fabrication *structurally impossible* was the point of the
skill.

## 6. The reproducibility spine

Claude Science tied the whole project together as durable, versioned artifacts:

- **Remote compute** — training and GPU work dispatched to a remote L40 host from inside the same
  workbench, with results harvested straight back as artifacts.
- **Sub-agent delegation** — parallel tracks (per-cell-line drug discovery, cell-state analysis, the
  GitHub and model-hub pushes) run as independent agents and merged back.
- **Versioned artifacts + lineage** — every figure, table, checkpoint, and the manuscript itself is a
  tracked artifact with its producing code, so any number in the paper can be traced to the cell that
  computed it.
- **Custom skills** — `autocollect-bio` (annotation) and `autoresearch-bio` (the anti-fabrication
  discipline it ports) encode the project's provenance rules once and apply them everywhere.

## 7. What Claude Science specifically made possible

A falsification-first paper is only viable if the negatives are cheap to run and impossible to fake.
Claude Science gave me both: fast enough iteration to treat an identifiability wall as a result
worth characterising across three model families, and enough provenance discipline (per-value source
tags, saved tables, versioned artifacts) that every negative is auditable. The honest from-scratch
collapse, reported as a bounded limitation rather than softened into a win, is the clearest example —
it is in the paper *because* the tooling made it cheap to run and safe to report.

---

## Deliverables produced in Claude Science

- **Paper:** *When Does a Perturbation Model Know Enough?* (28 pp) — https://github.com/yashraj59/RefineRx
- **Models + checkpoints:** https://huggingface.co/yraj/RefineRx
- **Zero-fabrication annotation dataset:** https://github.com/yashraj59/tcell-perturbed-gene-annotation
- **Reproducibility repo:** full project arc in 10 numbered stages (00 catalog → 09 paper), keeping
  the failed grafts as the evidence for the negative-result thesis.
