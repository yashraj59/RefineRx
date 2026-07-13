# CD4-native fused halting — E[N] collapse diagnostic

Head-free diagnostic run on the CD4-native pseudobulk checkpoint (`last.ckpt`): load the model, run the
6 refinement rounds on held-out perturbations, and measure the spread of the head-free **oracle stopping
round r\*** and the per-round convergence. This localizes the collapse **upstream of the halt head** — it
is a property of the substrate + recurrence, not of halt-head calibration.

## Expected depth E[N] (post-warmup)

| Run | halt γ | E[N] | across-perturbation std |
|-----|--------|------|--------------------------|
| default          | 0.01 | 6.000 | ≈ 1.7e-4 |
| strong ponder (resume from step 3000) | 0.10 | 5.9999 | ≈ 5e-6 → 9e-4 |

E[N] pins to the round budget (6.0). A 10× stronger ponder does **not** restore spread.

## Head-free oracle r\* (the decisive number)

- **r\* = max (final round, 0-indexed 5) for 100% of 1,064 held-out perturbation sets; std = 0.**
- There is no per-perturbation variation in the optimal stopping round to calibrate a halt head against.

## Per-round magnitude-free distance D_r (step-function convergence)

| round r | D_r |
|---------|-----|
| 1 | 1.043 |
| 2 | 1.227 |
| 3 | 1.199 |
| 4 | 1.235 |
| 5 | 0.988 |
| 6 | 0.031 |

Rounds 2–5 are **no better than round 1**; only the final round converges (D_6 ≈ 0.03). The x0-recurrence
produces a step function, not a smooth accuracy-vs-depth curve — so every perturbation's oracle stopping
round is the last one, and E[N] has nothing to vary over.

## Interpretation

The response head fits; the depth signal does not exist on this substrate. Attributed **primarily to
pseudobulk aggregation** — the frozen-backbone single-cell analysis (Stage 7) recovers a reproducible
depth signature (split-half ρ = 0.62–0.75) on the same biology, proving the signal exists at single-cell
resolution. The from-scratch single-cell control (on the ~22M-cell screen) was infeasible in the compute
available, so this bounds the method to single-cell substrates and does not, on its own, refute
end-to-end halting.
