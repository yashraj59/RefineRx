# Stage 8 — CD4-native STATE fused halting, trained from scratch (completed — a negative)

Rather than reading depth off a *frozen* Replogle backbone applied to CD4 data (Stage 7), this stage
trains a **CD4-native STATE model with adaptive-depth halting fused into the training loop and trained
from scratch**, so the refinement axis is learned on CD4 biology directly. The halting is
magnitude-free and **jointly calibrated** with the response objective (§4.4 oracle-r\* + §4.5
joint-calibration, KL = 0, ponder gated on after warmup; 8-layer Llama backbone, hidden **336**,
6 refinement rounds, confidence token). This is the design the earlier grafts showed is necessary to
avoid both the constant-depth collapse and the seed-instability.

## Outcome: the response head fits, but E[N] collapses on pseudobulk

Trained on CD4 **pseudobulk** (278,684 profiles × 2,001 HVG; Rest / Stim8hr / Stim48hr; 4 donors as
the batch covariate; `log1p(counts)`; control = NTC). The model **fits the perturbation response**
well, but the fused halting depth **collapses to a constant**:

- Post-warmup **E[N] → 6.0** (the round budget), across-perturbation **std ≈ 1e-4**.
- A 10× stronger ponder (halt γ 0.01 → 0.1, resumed from the step-3000 checkpoint) does **not** break
  the collapse (E[N] std stays ≈ 1e-4).

### Head-free diagnostic — the collapse is upstream of the halt head

Loading the checkpoint and running the refinement rounds on held-out perturbations shows the halt
head is not the problem — the *substrate* is:

- The head-free **oracle stopping round r\* is degenerate**: r\* = max (final round) for **100% of
  1,064 held-out perturbation sets**, std = 0.
- Per-round magnitude-free distance is a **step function** (1.04 → 1.23 → 1.20 → 1.24 → 0.99 → 0.03):
  refinement rounds 2–5 are no better than round 1, and only the *final* round converges. There is no
  smooth accuracy-vs-depth curve for any halt head to calibrate on.

## What we do and do not claim

We attribute the collapse **primarily to pseudobulk aggregation**, and we are careful not to
over-read it:

- **The signal exists at single-cell resolution.** The *frozen*-backbone single-cell CD4 analysis
  (Stage 7, two donors) recovers a reproducible within-donor depth signature (split-half ρ =
  0.62–0.75) and the suppressor-shallow ordering. So depth **is** recoverable from CD4 data — just not
  from aggregated pseudobulk.
- **We could not run the clean control.** Separating "pseudobulk hides it" from "the from-scratch
  x0-recurrence mechanism kills it" needs a CD4-native backbone trained from scratch **at single-cell
  resolution**. The full screen is ~22M cells; from-scratch single-cell training did not reach stable
  dataloader throughput in the compute available, so this control is left as future work.
- **Narrow, honest statement:** *on aggregated pseudobulk the from-scratch fused model fits the
  response but yields no usable depth signature, most plausibly because pseudobulk hides it.* This
  does not, on its own, refute end-to-end halting.

This is a **completed negative under the project's negative-result mandate**, not an in-progress
placeholder.

## Contents

- `code/` — the STATE transition-halt model (`state_transition_halt.py`) and the halting graft
  (`halting_graft.py`).
- `config/` — the training YAML (`state_halt.yaml`) and the CD4 run config (`cd4.toml`).
- `notebooks/` — the training and embedding-use notebooks.

**Checkpoints** (best.ckpt / last.ckpt, ~449 MB each) and the **four frozen-backbone oracle-ACT halt
heads** are on HuggingFace: [`yraj/RefineRx`](https://huggingface.co/yraj/RefineRx).
