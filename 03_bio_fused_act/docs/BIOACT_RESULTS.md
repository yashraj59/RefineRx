# True biology-fused ACT — built exactly as specified, and the decisive result

You asked for the real thing, and I built it exactly:
- **Learned PonderNet halt head** — emits λ_t each round → halting distribution p_t = λ_t·Π(1−λ_j),
  E[N] = Σ p_t·t, trained through a **ponder loss** (expected task loss under p + KL to a geometric
  prior). Fully differentiable. This IS ACT.
- **Biological features feed the halt head** — front size, front growth, new-territory-per-hop, mean
  field level, all computed on a **fixed transcriptome-wide STRING graph** (1508 nodes, 9212 edges,
  all 90 perturbed genes as seeds). Biology enters through the halt head's inputs.
- **Magnitude-free ponder loss** — per-round task term is cosine distance `1 − cos(pred_t, target_dir)`
  on L2-normalized directions (scale-invariant; no effect-size leakage).

Files: `bio_act.py` (the model), `train_bioact.py`, `string_graph_big.json` (the scaffold).

## Result: it collapses — and this time I found exactly why

Trained on the 1508-node graph, E[N] = **2.83 for every perturbation** (std 0.0008, range 0.006 of 8
rounds) = the geometric-prior mean. The learned halt head ignored the biological features and defaulted
to the KL prior. I checked two obvious culprits and ruled them out:
- **Not the loss** — confirmed the ponder loss is genuinely magnitude-free (cosine distance).
- **Not a readout shortcut** — masking the seed node from the readout (so the prediction must come from
  the propagated cascade) did **not** break the collapse (E[N] range still 0.006).

## The decisive diagnostic — the learning signal is absent

I measured the per-round cosine distance to the true response directly (no halting weighting), across
three propagation rates:

| propagation | round 1 → round 8 | refinement helps? |
|---|---|---|
| learned α=0.46 (fast) | 0.092 → 0.089 (Δ **+0.003**) | **flat** — field saturates by round 2 |
| α=0.05 (slow) | 0.084 → 0.102 (Δ **−0.017**) | **no** — later rounds *hurt* |
| α=0.02 (very slow) | 0.082 → 0.119 (Δ **−0.037**) | **no** — round 1 already best |

**In no regime does refinement improve the prediction**, and for nearly every perturbation the best
prediction lands at round 1–2 (fast: 27% at round 1; very-slow: 61% at round 1). With fast propagation
the field saturates in ~2 hops so every round is equally good → the halt head gets no task-differential
signal → it collapses to the KL prior. With slow propagation round 1 is already best and more hops
dilute the response with distal noise → the model *should* halt at round 1 for everyone → depth is
constant = 1. Either way there is **no perturbation for which deep refinement is the right answer**, so
a *learned* halt head has nothing to key on.

## Why this is the sharpest form of the thesis's negative

This is not "the signature is noisy" or "it's an effect-size proxy." It is deeper: **the per-round task
loss — the only thing that can teach a halt head to use depth — has no structure that varies across
perturbations.** A single endpoint is a *static* target; propagating a perturbation on a fixed network
reaches the predictive neighborhood in one hop, and iterating does not progressively uncover more of
the target because there is no sequential structure in a single endpoint to refine toward. Learned
adaptive depth is therefore **unidentifiable from single-endpoint data** — demonstrated here at the
level of the learning signal itself, and now robust across:

- 2 backbones (scDiff, TxPert) × magnitude-anchored PonderNet → depth = effect-size proxy
- magnitude-free learned gate → seed-noise
- free-convergence depth → training-time transient (collapses at convergence)
- **true biology-fused ACT, magnitude-free, transcriptome-wide → no per-round signal to learn from**

Under your own pre-registration ("if the signature fails reproducibility or collapses to an effect-size
proxy, that negative result is the finding"), this is the finding, in its most mechanistic form.

## The one thing that DID work — and what it really is

The **fixed** biological stopping rule (`bio_halt.py`, earlier) produced a reproducible (ρ=0.96),
non-redundant (R²=0.038), effect-size-independent per-perturbation depth. The distinction is now crisp
and important:

- A **learned** halting signature (ACT/PonderNet) is **not identifiable** from single endpoints — no
  per-round signal exists to train it. (this result)
- A **fixed** cascade-saturation depth on the STRING network **is** a reproducible, effect-size-
  independent per-perturbation quantity — but it is a **deterministic graph statistic** (how fast the
  perturbation's front saturates on the fixed network), *computed* not *learned*. It is a legitimate
  biological feature of each perturbation; it is not "an adaptive-depth model's halting behavior."

**So the honest, complete answer to the thesis:** the halting behavior of a *learned* adaptive-depth
model does not carry a reproducible per-perturbation signal on single-endpoint Perturb-seq, because the
learning signal for depth does not exist in that data. What *is* reproducible and effect-size-
independent is a **network-cascade-depth statistic** you can compute directly from the perturbed gene's
position in a biological network — which may still be useful for ranking targets (direct vs.
cascade-mediated), just not as a learned halting signature.

The graph-statistic route is the defensible positive deliverable, and it is what I would point at the
Marson/Pritchard CD4+ T-cell targets.
