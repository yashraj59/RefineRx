# RefineRx — Adaptive-depth halting as a per-perturbation signature in perturbation biology

**Thesis question.** Is an adaptive-depth model's halting **E[N]** — the expected number of
refinement rounds it needs to reach a perturbation's endpoint response — a *reproducible,
effect-size-independent per-perturbation signature* that can be recovered from **single-endpoint**
Perturb-seq? And if so, does that signature carry biology useful for ranking **CD4+ T-cell drug
targets**?

**The reframe that makes this work.** Adaptive-computation halting (ACT/PonderNet) was invented to
*save compute* — an easy input exits early, a hard one ponders longer. We repurpose it as a
**measurement rather than a stop rule**: on a frozen backbone whose layers all execute, halting saves
no compute, and the layer at which the model's endpoint estimate converges is read as a
per-perturbation proxy for **response complexity** — explicitly *not* biological time or causal
depth, which a single endpoint cannot recover.

This repository narrates the full project arc **chronologically**, and deliberately keeps the
experiments that *failed*: the many halting grafts that hit an identifiability wall are the evidence
for the project's negative-result thesis, not detours to be hidden. The directory numbering `00 → 09`
is the timeline.

---

## Headline findings

1. **Free-learned halting is largely an effect-size proxy — the identifiability wall.**
   Grafting learned halting (ACT / PonderNet) onto single-endpoint models produces an E[N] that is
   either a near-constant (uninformative) or a re-encoding of response magnitude (redundant,
   |ρ(E[N], effect)| up to ~0.9). Removing the magnitude anchor decouples E[N] from effect size but
   makes it **seed-dependent rather than data-dependent** (cross-seed ρ ≈ 0.14). On single-endpoint
   data no purely-computational regime is both reproducible *and* non-redundant.

2. **A fixed biological network, or an oracle stopping round, recovers a real signature.**
   Fusing a **fixed STRING network** into the halting mechanism reaches the useful corner
   (cross-seed ρ = 0.96, non-redundant). And on ARC's pretrained **STATE Transition** foundation
   model, an **oracle-supervised learned ACT** recovers a genuine per-perturbation depth:
   **K562 split-half ρ ≈ 0.76**, and across four Replogle lines (K562/HepG2/Jurkat/RPE1)
   within-line split-half **ρ ≈ 0.72–0.87 (4-line mean ρ ≈ 0.80)**.

3. **The signature is a genuinely novel descriptor — non-redundant with network topology.**
   No model-free network statistic reproduces the per-perturbation ordering of E[N]: directed-GRN
   cascade depth, out-degree, downstream reach, and PPI degree all fall below a 0.3 novelty ceiling
   (**best |ρ| = 0.23**, K562 PPI degree; GRN cascade depth |ρ| = 0.01–0.14 per line). The adaptive
   signature captures per-perturbation structure the simple baselines miss — this is the load-bearing
   *positive* for the thesis.

4. **Depth organizes druggability, but is non-additive with a network prior for one ranking task.**
   Clustering perturbations on the depth signature (UMAP + Leiden) recovers functional gene classes in
   which approved/clinical drug targets concentrate — a **translation/ribosome cluster in every line**
   (per-cluster odds ratios 2.8–9.8). That is a genuine positive. The *only* negative is narrow and
   **task-level, not descriptor-level**: for one downstream target-*class* ranking task, adding |ΔE[N]|
   on top of a STRING baseline built from the same modules gives no AUC lift in 3 of 4 lines. Depth is
   **non-redundant** as a descriptor (finding 3) yet **non-additive** for that one classifier — two
   different claims; only the latter is negative, and it does not diminish the signature's novelty.

5. **CD4+ T cells (application, paper Part 2).** On a two-donor, 50-cell-per-perturbation subsample
   (3.9M cells) of the genome-wide (~22M-cell) CD4+ CRISPRi screen, at single-cell resolution the
   depth signature is reproducible within a donor and **sharpens with stimulation**: within-donor
   split-half **ρ = 0.62 → 0.68 → 0.75** for Rest → Stim 8h → Stim 48h. At the stimulated endpoint,
   **resting-sparing suppressors converge shallower** than damaging/generic perturbations
   (Kruskal–Wallis **p = 5.2×10⁻²⁸**). Depth is donor-specific in rest/early activation
   (cross-donor ρ ≈ −0.1) and becomes portable only once donors converge on the shared activation
   program (cross-donor ρ = +0.49 at 48h).

6. **CD4-native from-scratch: pseudobulk hides the signature (a completed negative).**
   Training a CD4-native STATE model with fused magnitude-free halting **from scratch on pseudobulk**
   trains successfully — the response head fits — but the halting depth **collapses to a constant**
   (E[N] → 6.0, across-perturbation std ≈ 1e-4). A head-free diagnostic locates the cause upstream of
   the halt head: the oracle stopping round r\* is degenerate (**r\* = max for 100% of perturbations**)
   and per-round convergence is a **step function** (only the final round converges). A 10× stronger
   ponder does not restore spread. We attribute this **primarily to pseudobulk aggregation** — the
   single-cell signature (finding 5) proves the signal exists at single-cell resolution; the
   from-scratch single-cell control was infeasible in the compute available, so this **bounds the
   method to single-cell substrates and does not, on its own, refute end-to-end halting.**

7. **Cell-state (exploratory).** Refinement depth is **invariant to basal cell-cycle state**
   (cross-state ρ ≥ 0.99 in all four Replogle lines; real cross-state variance below a
   state-label-permutation null). What it couples to is the **directional-correction geometry** of
   the model-implied response trajectory (standardized β = +0.14 to +0.33), not a late cell-cycle
   clock. This track is exploratory and is not part of the paper unless promoted.

---

## The arc, stage by stage

- **[00 — Perturbation-model catalog & deep code review](00_catalog_and_review/)**
  The starting point. Code-grounded notes from cloning and reading the actual repositories of
  single-cell perturbation models across five families (foundation-model / condition-adaptation,
  knowledge-graph / GRN, chemical / drug, optimal-transport / flow / diffusion, transfer / spatial),
  with each architectural claim tied to the file and class actually read. An **ACT graftability
  matrix** scores where a learned-halting refinement loop could attach on each backbone — this is
  what chose the models tried next.

- **[01 — First ACT/PonderNet halting grafts](01_graft_attempts_ACT_Ponder/)**  *(what we tried first)*
  The first halting grafts, on small in-house predictors and on scDiff. A 12-config sweep over
  ponder weight / prior / β — and a 5→10 round budget increase — tests whether the constant-depth
  collapse can be broken. It cannot: E[N] is either a near-global constant or an effect-size echo
  (|ρ| up to 0.90), and doubling the budget buys almost no per-perturbation resolution
  (widest spread ≈ 4% of budget). **Negative result, kept as evidence.**

- **[02 — scDiff + TxPert adaptive-depth grafts](02_scdiff_txpert_grafts/)**  *(architecture-invariance test)*
  The same halting idea on two more architectures, with a magnitude-**free** (cosine-direction)
  target added to remove the self-inflicted magnitude confound. This exposes the core
  **reproducibility–redundancy trade-off**: magnitude-anchored halting is reproducible but redundant
  (ρ(E[N], effect) = −0.70 / −0.87); magnitude-free halting is non-redundant but not reproducible
  (cross-seed ρ ≈ 0.14–0.15). The mechanism is **identifiability** — a single endpoint does not pin
  how many rounds were needed to reach it.

- **[03 — Biology-fused ACT](03_bio_fused_act/)**  *(STRING-propagation halt features)*
  The first design that escapes the wall by making refinement **propagation on a fixed STRING PPI
  network** (APPNP-style), with halting defined as biological cascade saturation. Because the graph
  is fixed biology, depth cannot collapse to an optimizer constant: cross-seed **ρ = 0.96** and
  non-redundant (R² = 0.038 from effect size + #DE + count + degree), decoupled from effect size
  (ρ = +0.11). Its limitation is a narrow dynamic range on the small 88-gene scaffold — which
  motivates moving to a large pretrained backbone.

- **[04 — STATE oracle-ACT: the method that worked](04_state_oracle_act/)**  *(Replogle, 4 lines)*
  On ARC Institute's pretrained **STATE Transition** model, three results separate cleanly: a
  **computed** per-layer-argmin readout is reproducible and effect-independent (ρ = 0.836);
  **naive** learned ACT still collapses to a per-seed global constant; but an **oracle-supervised**
  learned halt head recovers a real per-perturbation depth (ρ = 0.761, K562 split-half ≈ 0.76).
  Refit across all four Replogle lines gives within-line split-half **ρ ≈ 0.72–0.87 (mean ≈ 0.80)**.
  Trained halt-head weights for each line are included.

- **[05 — Biology validation & drug-target discovery](05_biology_and_drug_discovery/)**  *(4 lines)*
  Depth UMAP/Leiden structure plus a response-cosine + STRING-adjacency drug-target discovery
  pipeline. Two results must be held apart. **The positive:** clustering on the depth signature alone
  recovers functional classes where approved/clinical drug targets concentrate (translation/ribosome
  cluster, odds ratios 2.8–9.8 every line), and depth is **non-redundant with network topology**
  (Stage 4's GRN/PPI baselines, best |ρ| = 0.23 — see finding 3). **The narrow negative:** for one
  incremental target-*class* ranking task, adding |ΔE[N]| on top of a STRING baseline gives no AUC
  lift in 3/4 lines — task-level **non-additivity**, not descriptor-level redundancy. The ranking
  itself is useful — the safety-pass recurrent hits (led by **MIOS / LAMTOR1 → MTOR**) are
  mechanism-coherent — but cross-line recurrence must be read against an essentiality confound.
  The drug/toxicity annotation feeding this stage was collected with `autocollect-bio` (per-value
  source tags; `NOT_FOUND` vs `NOT_FETCHED` kept distinct) and merges the author's standalone CD4
  perturbed-gene annotation dataset, released separately at
  [tcell-perturbed-gene-annotation](https://github.com/yashraj59/tcell-perturbed-gene-annotation).

- **[06 — Cell-state-conditioned refinement](06_cellstate_exploratory/)**  *(EXPLORATORY side-analysis, not paper)*
  An exploratory test of whether depth reports the cell's starting state. It does not: E[N] is
  **invariant to basal G1/S/G2M cell-cycle state** (cross-state ρ ≥ 0.99 in all four lines, below a
  permutation null), and instead couples to the **directional-correction shape** of the response
  trajectory. This walls the CD4 activation-phenotype result off from a cell-cycle confound. **This
  stage is exploratory and is excluded from the paper unless promoted.**

- **[07 — CD4+ T-cell application](07_cd4_application/)**  *(paper Part 2)*
  The STATE oracle-ACT applied to the Marson/Pritchard genome-wide CD4+ CRISPRi Perturb-seq screen
  (two donors × Rest / Stim 8h / Stim 48h, single-cell). Within-donor split-half **ρ = 0.62 → 0.68 →
  0.75** rising with stimulation; cross-donor portability only at the converged 48h endpoint
  (ρ = +0.49). At Stim 48h, **resting-sparing suppressors converge shallower** than damaging/generic
  perturbations (Kruskal p = 5.2×10⁻²⁸) — the Part-2 discovery.

- **[08 — CD4-native STATE fused halting](08_cd4_native_state_halt/)**  *(completed — a negative)*
  A CD4-native STATE model with adaptive-depth halting fused in and trained **from scratch on
  pseudobulk** (rather than reading depth off a frozen backbone), with magnitude-free,
  jointly-calibrated halting (8-layer Llama backbone, hidden 336, 6 refinement rounds). **Outcome: the
  response head fits, but E[N] collapses to a constant** (→ 6.0, std ≈ 1e-4). A head-free diagnostic
  shows the collapse is upstream of the halt head — the oracle stopping round r\* is degenerate
  (r\* = max for 100% of perturbations) and per-round convergence is a step function; a 10× stronger
  ponder does not restore spread. We attribute this **primarily to pseudobulk aggregation** (the
  single-cell signature in Stage 7 proves the signal exists at single-cell resolution), so it bounds
  the method to single-cell substrates and does not, on its own, refute end-to-end halting. Training
  config, graft code, checkpoints, the r\* diagnostic, and driving notebooks are here. Checkpoints and
  the four frozen-backbone halt heads are also on HuggingFace (`yraj/RefineRx`).

- **[09 — Manuscript](09_paper/)**
  The paper draft (LaTeX + PDF) plus the provenance records: generation methods for the CD4 gene
  annotation deliverables and the annotation schema.

### Data & provenance — the annotation layer

The drug- and toxicity-target annotation used across the discovery stages was built with a
**zero-fabrication discipline**: every value in every table is paired with the named source it came
from and the date it was retrieved, and the token `NOT_FOUND` (queried, genuinely absent) is kept
strictly separate from `NOT_FETCHED` (deliberately not queried) — no cell is ever a guessed or
model-derived value. Fabrication is *structurally impossible* rather than merely discouraged. Sources:
**Open Targets, ChEMBL, gnomAD, DepMap, Pharos, STRING, FDA**.

The headline annotation product — the **CD4+ T-cell perturbed-gene table (11,526 genes**, with a
**~1,796-gene** druggable/tractable deep subset inside it) — is released as a **standalone dataset
repo**: **https://github.com/yashraj59/tcell-perturbed-gene-annotation**. (A separate, smaller
1,708-gene Replogle cell-line drug-discovery union, 74 fields, lives inside Stage 05 and is not that
repo.) Both layers were collected the same way, using
[`autocollect-bio`](skills/autocollect-bio/) — a skill the author created using Claude Science, and
included in this repo (under `skills/`) for reproducibility of the annotation layer.

---

## Interpretation & caveats

- **Halting E[N] is a computational proxy for response *complexity*, not biological time or causal
  depth.** Refinement rounds count how much iterative correction a fitted model needs to reach a
  perturbation's endpoint; they are not hours, pseudotime, or a mechanistic number of regulatory
  steps. Single-endpoint data cannot recover the latter, and no claim here should be read that way.

- **Depth is cell-type-specific, not a context-portable constant.** The signature is reproducible
  *within a fixed context* (a given backbone + cell type) and is genuinely novel there (non-redundant
  with network topology, finding 3), but it does **not** transfer across contexts: cross-line E[N]
  ρ ≈ 0.14 and cross-donor (resting) ρ ≈ 0. The one place portability appears is where the *biology
  itself* converges — cross-donor ρ = +0.49 only at the fully-stimulated CD4 48h endpoint. Treat E[N]
  as a per-perturbation property recovered *within a context*, not as an intrinsic
  context-invariant constant. (This is distinct from calling it "model-specific" in a dismissive
  sense — within its context it is a reproducible, effect-independent, non-redundant descriptor.)

- **The identifiability wall is the load-bearing negative result.** Free learned halting on
  single-endpoint data is either redundant with effect size or unstable across seeds; a reproducible,
  non-redundant signature requires either pinning depth to fixed external structure (Stage 3's STRING
  graph) or supervising it with an oracle stopping round on a backbone that fits the data (Stage 4).
  The from-scratch CD4-native run (Stage 8) is a further instance: fused halting on *pseudobulk*
  collapses because the substrate provides no smooth accuracy-vs-depth curve to calibrate against —
  the signal that *does* survive at single-cell resolution (Stage 7) is aggregated away.

- **Exploratory vs. paper.** **Stage 6 (cell-state)** and any **temporal / cross-time analyses** are
  **exploratory** and are not part of the manuscript unless explicitly promoted. Stages 04, 05, and
  07 are the paper-track results.

- **Discovery outputs are hypotheses, not validated targets.** Drug-target candidates are annotated
  automatically from public databases (Open Targets, ChEMBL, gnomAD/DepMap, Pharos, STRING, FDA) with
  per-value source tags; they are "same class as anchor X" hypotheses, not hand-curated calls.

---

## Repository layout

```
00_catalog_and_review/       model catalog + code review + ACT graftability matrix
01_graft_attempts_ACT_Ponder/  first ACT/PonderNet grafts (negative)
02_scdiff_txpert_grafts/     scDiff + TxPert grafts; reproducibility–redundancy trade-off (negative)
03_bio_fused_act/            STRING-propagation halting — first reproducible+non-redundant signature
04_state_oracle_act/         STATE oracle-ACT — the method that worked (4 Replogle lines)
05_biology_and_drug_discovery/  depth UMAP/Leiden + drug-target discovery (4 lines)
06_cellstate_exploratory/    cell-state invariance (EXPLORATORY, not paper)
07_cd4_application/          CD4+ T-cell single-cell application (paper Part 2)
08_cd4_native_state_halt/    CD4-native fused halting from scratch — pseudobulk E[N] collapse (negative)
09_paper/                    manuscript + provenance records
```

Each stage directory has its own `README.md`. See [`REQUIREMENTS.md`](REQUIREMENTS.md) for the
software environments and [`LICENSE`](LICENSE) (MIT).
