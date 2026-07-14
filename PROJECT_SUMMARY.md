# RefineRx — Project Summary

*Adaptive-depth halting as a per-perturbation signature in perturbation biology*
**Paper:** When Does a Perturbation Model Know Enough? · **Author:** Yash Raj
**Code:** github.com/yashraj59/RefineRx · **Models:** huggingface.co/yraj/RefineRx
**Annotation:** the 11,526-gene CD4+ T-cell perturbed-gene table (with a ~1,796-gene druggable/tractable
deep subset) is released as a standalone dataset — github.com/yashraj59/tcell-perturbed-gene-annotation —
collected with `autocollect-bio`, a zero-fabrication annotation skill the author created using Claude Science (every
value paired with its source and recorded date; `NOT_FOUND` kept strictly separate from a guess).

---

## 1. The question

Perturb-seq measures a single endpoint — the transcriptome of perturbed cells against unperturbed
controls. We ask whether an adaptive-depth model's **halting behaviour** — how many rounds of
iterative refinement it needs to predict a perturbation's endpoint response, summarized as an
expected depth **E[N]** — is a **reproducible, effect-size-independent, per-perturbation signature**
that can be recovered from that single endpoint alone. Throughout, halting is treated strictly as a
**computational proxy for response complexity**, never as biological time or causal cascade length,
which a single endpoint cannot recover. The applied half asks whether that signature carries biology
useful for ranking **CD4+ T-cell drug targets** — inhibitory targets that suppress stimulated
inflammatory programs while sparing the resting T-cell state.

## 2. The direct observation

A single endpoint is the quantity perturbation models are trained on, yet the biology that produces
it is heterogeneous. A knockdown of a structural ribosomal protein produces a large, immediate,
largely direct transcriptional shift; a knockdown of a signalling regulator may produce a comparable
endpoint only after feedback, compensation, and a cascade of secondary events. **The endpoint alone
does not distinguish these regimes** — two perturbations of very different mechanistic complexity can
land at responses of identical magnitude. If a model reaches those endpoints by iterative refinement,
the *amount* of refinement is a candidate readout of that hidden complexity.

## 3. The identifiability wall

Free-learned halting grafted onto single-endpoint models does not yield a usable signature. Across
architectures (in-house predictors, scDiff, TxPert) and a 12-config sweep over ponder weight / prior /
budget, E[N] is either a **near-global constant** (uninformative) or a **re-encoding of response
magnitude** (redundant, |ρ(E[N], effect)| up to ~0.9). Removing the magnitude anchor (a magnitude-free
cosine-direction target) decouples E[N] from effect size but makes it **seed-dependent rather than
data-dependent** (cross-seed ρ ≈ 0.14). Doubling the refinement budget (5 → 10 rounds) buys almost no
per-perturbation resolution. The mechanism is **identifiability**: a single endpoint does not pin how
many rounds were needed to reach it, so on single-endpoint data no purely-computational regime is both
reproducible *and* non-redundant. **This negative is the load-bearing result of the project.**

## 4. The working readout

Two routes clear the wall — both by supplying structure the endpoint cannot.

- **Fixed external biology.** Making refinement propagation on a *fixed* STRING PPI network, with
  halting defined as biological cascade saturation, reaches the useful corner: cross-seed **ρ = 0.96**,
  non-redundant (R² = 0.038 from effect size + #DE + count + degree). Because the graph is fixed, depth
  cannot collapse to an optimizer constant. Its limit is a narrow dynamic range on a small scaffold.

- **Oracle supervision on a backbone that fits the data.** On ARC Institute's pretrained **STATE
  Transition** model, an **oracle-supervised, magnitude-free learned halt head** recovers a genuine
  per-perturbation depth where a naive learned gate still collapses. On K562 (968 perturbations),
  **split-half ρ = 0.76**, effect-independent (**partial ρ = −0.04** controlling #DE and cell count),
  and it recovers its oracle target (**ρ = 0.61**).

The conceptual reframe that makes this legitimate: adaptive-computation halting was invented to *save
compute*. Here it is repurposed as a **measurement, not a stop rule** — on a frozen backbone whose
eight transformer layers always execute, halting saves no compute, and the layer at which the model's
endpoint estimate converges is read as the per-perturbation depth.

## 5. What the readout measures

E[N] is a **computational proxy for response complexity** — the number of refinement rounds a fitted
model needs to best capture a perturbation's response direction. It is **not** biological time (the
data are a single endpoint), and **not** causal cascade length (a single endpoint cannot recover it).
Mechanistically it couples to the **directional-correction geometry** of the model-implied response
trajectory (standardized β = +0.14 to +0.33 for the nonlinear-correction term), not to late program
activation. It is **invariant to basal cell-cycle state** (cross-state ρ ≥ 0.99 in all four Replogle
lines, real cross-state variance below a state-label-permutation null), which walls the downstream
biology off from a cell-cycle confound.

## 6. Where it reproduces — and where it does not

The signature is reproducible **within a fixed context** (a given backbone + cell type) and is
genuinely novel there, but it does **not** transfer across contexts.

- **Within context:** K562 split-half ρ = 0.76; across four Replogle lines (K562/HepG2/Jurkat/RPE1)
  within-line split-half **ρ ≈ 0.72–0.87, mean ρ ≈ 0.80**.
- **Across cell types:** cross-line E[N] **ρ = 0.14** — it does not port.
- **Across donors (resting):** cross-donor **ρ ≈ 0** — donor-specific.
- **The one place portability appears is where the biology itself converges:** cross-donor
  **ρ = +0.49** only at the fully-stimulated CD4 48 h endpoint.

The honest framing is therefore **cell-type-specific, recovered within a context — not a
context-invariant per-perturbation constant.**

## 7. T-cell biological application

Applied to the genome-wide Marson/Pritchard primary CD4+ T-cell CRISPRi Perturb-seq screen — a
two-donor, 50-cell-per-perturbation subsample (**3.9M cells** of the full **~22M-cell** screen), at
single-cell resolution:

- **Reproducible within a donor, sharpening with stimulation:** within-donor split-half
  **ρ = 0.62 → 0.68 → 0.75** for Rest → Stim 8h → Stim 48h.
- **The Part-2 discovery:** at the stimulated 48 h endpoint, **resting-sparing inflammatory-program
  suppressors converge shallower** (lower E[N]) than damaging and generic perturbations
  (**Cliff's δ = −0.16 to −0.24**; Kruskal–Wallis **p = 5.2×10⁻²⁸**).

A CD4-native STATE model with fused halting trained **from scratch on pseudobulk** trains
successfully — the response head fits — but the halting depth **collapses to a constant** (E[N] → 6.0,
across-perturbation std ≈ 1e-4). A head-free diagnostic locates the cause upstream of the halt head:
the oracle stopping round is degenerate (r\* = max for **100% of perturbations**) and per-round
convergence is a **step function**. We attribute this **primarily to pseudobulk aggregation** — the
single-cell result above proves the signal exists at single-cell resolution; the from-scratch
single-cell control was infeasible in the compute available. This **bounds the method to single-cell
substrates and does not, on its own, refute end-to-end halting.**

## 8. Drug-target and network boundary

**The positive.** Clustered on its own, E[N] organizes the target landscape: approved and clinical
drug targets concentrate in a **translation/ribosome depth cluster in every line** (per-cluster odds
ratios **2.8–9.8**). And the descriptor is **non-redundant with network topology** — no model-free
graph statistic (directed-GRN cascade depth, out-degree, downstream reach, PPI degree) reproduces the
per-perturbation ordering of E[N] (**best |ρ| = 0.23**, below a 0.3 novelty ceiling). The adaptive
signature genuinely captures per-perturbation structure the simple baselines miss.

**The boundary.** For one *incremental* target-**class** ranking task — adding |ΔE[N]| on top of a
STRING baseline built from the *same* functional modules — E[N] gives no AUC lift in three of four
lines. This is **task-level non-additivity, not descriptor-level redundancy**: STRING and
depth-clustering happen to surface the same functional strata that the target-class label tracks, so
for predicting *that one label* they overlap — but E[N] still carries information STRING does not
(the non-redundancy above). Non-redundancy as a descriptor and non-additivity for one classifier are
different claims; only the latter is negative, and it does not diminish the signature's novelty.
