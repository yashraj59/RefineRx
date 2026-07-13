# scDiff + PonderNet: end-to-end training on Adamson 2016 UPR Perturb-seq

**What ran:** the PonderNet adaptive-depth graft wired into the *real* OmicsML/scDiff
`ScDiff` model (not a sketch), trained end-to-end on the L40 GPU host. The ponder loss is
**added to** scDiff's native diffusion loss inside an overridden `p_losses`; nothing about
the base objective was removed.

## Setup (all verified, not recalled)
- **Data:** `/workspace/halt/data/preprocessed/adamson_2016_upr_perturb_seq.h5ad` —
  62,477 cells × 20,616 genes, CRISPRi, subset to the 2,000 highly-variable genes.
  Grouping = `obs.target_gene`: **92 conditions** = 90 named-gene knockdowns + `control` (7,260 cells)
  + one non-gene `*` grouping row (100 cells, excluded from the confound analysis below).
- **Model:** scDiff `DiffusionModel` (MAE-style transformer, x0-parameterization),
  depth 4, embed 256, **7.68 M params**. Conditioner vocabulary = the 92 target-gene classes.
- **Adaptive depth:** a K=5-round x0-refinement recurrence (t-schedule [800,612,425,237,50])
  with a PonderNet halting head over the rounds. Learned-halting, β=0.05, geometric prior λ=0.2.
- **Compute:** NVIDIA L40, 20 epochs, ~9–12 s/epoch (~5 min total), `TRAIN_EXIT=0`.

## Training — the graft trains end-to-end
| | epoch 0 | epoch 19 (final) |
|---|---|---|
| diffusion loss (native `loss_simple`) | 0.2859 | 0.1611 |
| VLB term (`loss_vlb`) | 0.0614 | 0.0317 |
| **total** (`loss`) | 0.3147 | 0.1759 |
| ponder loss (`loss_ponder`) | 0.2887 | 0.1483 |
| mean refinement rounds E[N] | 2.4383 | 2.2930 |
| halt entropy | 1.4916 | 1.4784 |

Both losses fall together and gradients reach the halt head — the integration is correct
and stable. This is the engineering result: **PonderNet halting can be grafted onto a
production diffusion perturbation model and trained without destabilizing the base model.**

## Per-perturbation signature — the scientific result (a negative one)
After training, each of the 90 named-gene knockdowns got a signature (mean over its cells):
`refinement_rounds` E[N], `halt_confidence`, `nonlinear_correction`, and a **control-relative**
`response_shift` = ‖mean_expr(pert) − mean_expr(control)‖ (the effect-size covariate).

**Stage-0 confound check (E[N] vs effect size), 90 named-gene knockdowns (control and the non-gene `*` row excluded):**
- Spearman ρ = **-0.704** (p = 9.6e-15)
- Pearson  r = **-0.761** (p = 3.3e-18)
- E[N] range = **[2.194, 2.278]** — a spread of
  only **0.084 rounds** out of 5 possible.

### Reading this honestly
This first, **untuned** run lands squarely in the failure regime the synthetic validation
(`act_ponder/VALIDATION.md`) predicted:
1. **The halting head has near-collapsed to a constant depth** — E[N] varies by <0.1 round
   across all 90 named-gene knockdowns. There is almost no per-perturbation signal to interpret.
2. **What little variation exists is an effect-size proxy, not a complexity signal** — E[N] is
   strongly *anti*-correlated with response magnitude (ρ≈−0.70). Under the thesis's own
   pre-registered rule ("if the signature collapses to an effect-size proxy, that negative
   result is the finding"), **this run is a negative result** — as it should be reported.

This is exactly why the thesis plan puts the ponder-weight / prior / n_refine sweep and the
ranking-stability check *before* any biological interpretation. The pipeline now exists to run
that sweep; this run is the baseline it starts from, not evidence about the biology.

## What this establishes
- A **working, reproducible harness**: real scDiff + PonderNet, one GPU, ~5 min/run, emitting
  `metrics.json`, per-perturbation `signature_fixed.csv`, and `confound_check_fixed.json`.
- The **confound check is wired in from the start** — every future run reports corr(E[N], effect size)
  automatically, so an effect-size collapse can never be mistaken for signal.

## Immediate next steps (to escape the collapse regime)
1. **Sweep** `ponder_weight ∈ {0.02, 0.1, 0.3, 1.0}`, `lambda_prior ∈ {0.1, 0.2, 0.4}`, and a genuine
   ongoing per-round compute cost — VALIDATION.md shows E[N] only tracks true depth in a middle regime.
2. **Deepen the refinement budget** (n_refine 5→10) and confirm E[N] actually uses the extra range.
3. Only once E[N] shows real spread AND survives effect-size regression: reproducibility across
   seeds/guides/donors, then the Marson/Pritchard application.
