# My own adaptive-depth model, inspired by your halting setup — the result

**Instruction followed:** get inspired by the rampert halting system, but **build my own model, don't
use theirs.** I did. Nothing below imports or runs rampert / `HybridMemoryRAMLite` / scDiff / TxPert.
The model is a standalone recurrent refiner I wrote (`act_refiner_conv.ConvRefiner`), tested on the
same Adamson 2016 UPR Perturb-seq data (88 knockdowns) so the numbers compare directly to the earlier
scDiff (−0.70) and TxPert (−0.87) collapse results.

## What I took from your setup (design lessons, not code)

1. **Anti-collapse depth ceiling** (`max_rounds` small) — your finding that 8 collapses to a uniform
   ~2 while 4 stays adaptive.
2. **Scale-invariant halt features** — magnitude-blind *by architecture*, not just via the loss.
3. **No PonderNet-KL** — your finding that the KL prior makes depth effect-coupled and fails
   non-redundancy; plain convergence/ACT is the cleaner test.

## Two model variants, and why the second is the honest one

**Variant 1 — learned ACT gate (scale-invariant, ACT-4):** collapsed. Depth = 2.000 for *every*
perturbation, at **every** ponder cost including τ=0 (no penalty). A free learned gate over an
expressive one-shot predictor has no reason to assign different depths — the recon-optimal solution
is "halt as early as allowed" for everyone.

**Variant 2 — convergence-based depth (no gate):** depth = #rounds until the predicted unit direction
stabilizes, driven by a shared residual dynamical operator. This *cannot* collapse to a free gate.
And at first it looked like the breakthrough:

| stage | reproducibility ρ | R²(covariates) | ρ(depth, effect) | depth spread |
|---|---|---|---|---|
| **@ 60 epochs (smoke)** | **0.73** | 0.036 | −0.14 | CV 0.08 |
| **@ 400 epochs (converged)** | collapsed | — | 0.00 | **CV 0.00** |

At 60 epochs the convergence model landed in the *useful corner* every prior experiment missed —
reproducible AND non-redundant. But that signal **disappears when the model finishes training.**

## The decisive diagnostic: it's a training-time transient

I checkpointed one model at increasing epochs and watched the depth spread:

| epoch | recon loss | depth CV | depth range |
|---|---|---|---|
| 5 | 0.504 | 0.182 | 3 |
| 20 | 0.190 | 0.199 | 1 |
| 80 | 0.029 | 0.053 | 1 |
| **160** | 0.007 | **0.000** | **0** |
| 640 | 0.0002 | 0.000 | 0 |

As reconstruction loss falls from 0.50 to 0.0002, per-perturbation depth variance decays
**monotonically to exactly zero**. The variable-depth "signature" exists *only* while the model is
underfit. A converged model assigns every perturbation the identical depth.

## The finding (sharper than "use time-resolved data")

**Adaptive-depth halting on single-endpoint Perturb-seq is a property of the optimizer's trajectory,
not of the perturbation biology.** Three independent model families now agree:

- **magnitude-anchored** (scDiff, TxPert): depth reproducible but = effect-size proxy (redundant).
- **magnitude-free learned gate**: depth non-redundant but seed-unstable (not reproducible).
- **convergence-based**: depth *looks* reproducible-and-non-redundant, but only transiently while
  underfit; it collapses to a constant at convergence.

The unifying mechanism is **identifiability**: a single endpoint per perturbation does not constrain
how many refinement steps were needed to reach it. Any well-trained model is therefore free to reach
every endpoint in the same number of steps, and does. The only way depth varies is if the model is
prevented from converging (underfitting, a KL prior, or a magnitude-coupled loss) — and each of those
makes the depth an artifact of *that* constraint, not of the biology.

**This is a clean, pre-registered negative with a demonstrated mechanism**, and it is now robust to:
2 backbones × 3 halting mechanisms (learned-gate, PonderNet-KL, convergence) × the full epoch sweep.
Under your thesis's own rule ("if the signature fails reproducibility or collapses to an effect-size
proxy, that negative result is the finding"), this is the finding — and it is stronger for having
tried the convergence route that briefly appeared to succeed.

## What this leaves for the thesis

The depth axis is closed on single-endpoint data. What remains reproducible is the **response
direction itself** (a data statistic, not a learned depth). The defensible positive deliverable is to
rank Marson/Pritchard CD4+ T-cell targets on direction-space structure (program specificity vs the
resting state), with halting depth reported as demonstrated-non-viable. I can build that next.
