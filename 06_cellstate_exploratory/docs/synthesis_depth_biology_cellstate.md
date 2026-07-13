# Coupling the CD4 biology with the cell-state study: one axis, two directions

**Status.** This note synthesizes two independent analyses to test whether they tell a
single coherent story. The CD4 biology is a main-thesis (Part 2) result. The
cell-state study across the four Replogle lines is the **exploratory / frozen-model
side-analysis** explicitly walled off from the paper unless promoted. Numbers below are
read directly from the saved result artifacts, not recalled.

---

## The claim

**Refinement depth E[N] measures how much *directional correction* a perturbation's
predicted response requires — an intrinsic property of the perturbation, decoupled
from the cell state it starts from.** The two analyses reach this from opposite
directions: the cell-state study shows what depth is *not* (a context readout), and the
CD4 biology shows what depth *is* (a response-complexity readout that tracks activation
program).

---

## Thread 1 — CD4 biology (main thesis, Part 2)

On the primary CD4+ T-cell CRISPRi Perturb-seq data, with per-condition frozen
backbones:

- **Reproducible within a donor, and rising with stimulation.** Split-half E[N]
  Spearman ρ = **0.615 (Rest) → 0.678 (Stim 8h) → 0.748 (Stim 48h)**
  (n = 6,524 / 6,857 / 6,649). Depth is a well-defined signature, and it sharpens as the
  cell is driven further from its resting transcriptome.
- **Resting-sparing suppressors converge faster (shallower) than damaging or
  inflammatory perturbations.** At Stim 48h, E[N] separates the four phenotype
  categories (Kruskal–Wallis **p = 5.2 × 10⁻²⁸**); median E[N] =
  **7.227 (resting-sparing suppressor) < 7.264 (inflammatory) < 7.366 (damaging) <
  7.471 (other)**. Suppressors that keep cells near the resting manifold demand the
  *least* refinement.
- Depth separates **activation phenotypes**, not pathway families — consistent with the
  ruling that depth is a computational property of the fitted model, reproducible within
  a context, not a portable biological constant.

## Thread 2 — cell-state study (exploratory, four Replogle lines)

Applying each line's frozen checkpoint with the basal context sampled from G1 / S / G2M
control cells, target held fixed:

- **Depth is invariant to basal cell-cycle state.** Cross-state ρ(E[N]) =
  **0.996 (K562) / 0.998 (HepG2) / 0.989 (Jurkat) / 0.999 (RPE1)**. Against a
  within-batch state-label permutation null, the real cross-state SD of E[N] is *smaller*
  than null in every line (K562 0.040 vs 0.132; HepG2 0.014 vs 0.048; Jurkat 0.009 vs
  0.027; RPE1 0.008 vs 0.028; empirical p(real ≥ null) = 1.0 throughout). Depth is a
  perturbation property, not a cell-state property.
- **What depth *does* couple to is the shape of the model-implied trajectory.** Deeper
  perturbations trace intermediate states that dip further from the endpoint and then
  recover — a **directional-correction / nonlinear-correction** signal (standardized
  β(E[N] → nonlinear correction) = **+0.14 (K562) / +0.28 (HepG2) / +0.33 (Jurkat) /
  +0.25 (RPE1)**, all p ≤ 1 × 10⁻¹⁰), replicated in all four lines. Once magnitude
  confounds are controlled, the signal is in the trajectory's **shape, not its length**
  (β for total path length is negative: deeper → straighter total path).
- The originally hypothesized late cell-cycle / DNA-damage program activation is **not**
  supported (combined β ≈ 0, per-line signs disagree). Depth is not a late-program clock.

---

## Where the threads meet

Both describe the **same axis**:

| | Thread 1 (CD4 biology) | Thread 2 (cell-state) |
|---|---|---|
| What raises depth | Stimulation — large transcriptional reorganization | More directional correction in the response trajectory |
| What lowers depth | Resting-sparing suppression — cell stays near baseline | Straighter, less-corrected trajectory |
| Depends on cell context? | No — signature is perturbation-specific | No — invariant to G1/S/G2M (ρ ≥ 0.99) |

A resting-sparing suppressor is **shallow** because its response requires little
directional re-orientation — it keeps the cell near where it started. A strong
stimulator is **deep** because its response demands a large, multi-step reorientation of
the transcriptome. "How activated the endpoint is" (Thread 1) and "how much the response
direction has to swing" (Thread 2) are two views of one quantity: **response-program
complexity, measured as correction load.**

The cell-state invariance is what makes this a *clean* signature rather than an artifact:
depth is not reporting the cell's starting phase, so the CD4 activation-phenotype
separation cannot be a cell-cycle confound.

---

## What this does *not* claim

- **Not biological time.** Refinement rounds are a computational proxy for response
  complexity, not hours or causal depth — single-endpoint data cannot recover the latter.
- **Not a portable constant.** Depth is a property of a *fitted model* in a *context*,
  reproducible within that context; it is not a context-invariant property of the
  perturbation transferable across arbitrary backbones. (Cross-donor CD4 transfer holds
  only at the converged stimulated endpoint, ρ = +0.49 at 48h, and is weak/negative at
  Rest and 8h.)
- **The cell-state track remains exploratory** and is not part of the paper unless
  promoted.

---

## The bridge still to build

Thread 1 (CD4) and Thread 2 (Replogle lines) are on *different backbones*. The
CD4-native fused-halting model now training end-to-end is the piece that would let us
test the **directional-correction geometry directly on a CD4 backbone**:

- **If** the CD4-native E[N] reproduces the directional-correction coupling and the
  suppressor-shallow / stimulator-deep ordering, the two threads fuse into a single
  claim on the application dataset, and the target-ranking cards inherit a mechanistic
  reading (shallow = low correction load = resting-sparing).
- **If** the CD4-native E[N] collapses (the pseudobulk risk — pseudobulk averages away
  the within-population variation the single-cell signal drew on), the honest story is
  "the signature is recoverable at single-cell resolution but not from pseudobulk," which
  is itself a finding consistent with the negative-result mandate.

Either outcome is reportable. The synthesis above stands on the two completed threads;
the CD4-native run decides only whether they merge on one backbone.
