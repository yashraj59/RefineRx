# Biology-fused ACT grafted onto the real TxPert model — the capstone result

Built exactly as asked: the biology-fused, magnitude-free ACT halt head grafted **onto an existing
published perturbation model** (valence-labs/TxPert), not a standalone toy.

## What was grafted (`bio_act_txpert.py`, on top of the existing `adaptive_txpert.py`)

- **Host model: TxPert** — a graph neural network (GATv2 message passing over a biological gene graph).
  Not a foundation model; the KG/GNN family. Its message passing already runs on biology, so the
  halt head reads cascade features from TxPert's own hops.
- **Learned PonderNet halt head** reading **biological cascade features** from TxPert's message passing:
  per-hop embedding movement ‖h_k − h_{k-1}‖, signal norm ‖h_k‖, hop fraction. Biology enters through
  the halt head's inputs.
- **Magnitude-free ponder loss** — per-hop task term is cosine distance on the L2-normalized response
  direction (target − basal), no full-magnitude MSE.
- Full protocol: 4 seeds × 25 epochs, n=89 perturbations, 8-hop budget.

## Result — reproducible, but it reduces to an effect-size proxy

| metric | value | reading |
|---|---|---|
| reproducibility ρ (across seeds) | **0.709** | depth ranking is stable |
| ρ(E[N], effect size) | **-0.837** | strongly effect-coupled |
| partial ρ(E[N], effect \| degree, cells) | **-0.832** | coupling survives controls |
| R²(effect + cells + degree) | **0.640** | covariates explain most of E[N] |
| ρ(E[N], graph degree) | -0.002 | topology exonerated |
| E[N] range | 0.34 of 8 (4.3%) | near-constant depth |

The depth **is reproducible (ρ=0.71)** — but it is reproducible *because it has collapsed back to an
effect-size proxy*: ρ(E[N], effect) = −0.84, essentially identical to the earlier magnitude-anchored
TxPert (−0.87). Removing magnitude and fusing biology into the halt features **did not break the
coupling.**

## Why — the per-hop diagnostic explains it directly

Per-hop cosine distance to the true response direction (no halting weighting):
`[0.604, 0.63, 0.648, 0.66, 0.666, 0.672, 0.669, 0.674]`

**Hop 1 is the best prediction for 85 of 89 perturbations**; distance rises after hop 1 and stays
above the hop-1 value (0.604 → 0.674, with a negligible 0.002 dip at hop 7). So on a real graph model
too, **refinement does not help** — the message passing
reaches its best direction estimate in one hop, and additional hops add noise. The halt head correctly
learns "halt at hop 1," and the only thing left to modulate the tiny residual depth (4.3% of budget) is
effect size: larger-effect perturbations resolve their direction marginally faster, so E[N] tracks
effect size. That is the −0.84 coupling.

## The complete verdict across every architecture tried

| model | halting | reproducible? | non-redundant? | outcome |
|---|---|---|---|---|
| scDiff + PonderNet (mag-anchored) | learned | ✓ | ✗ (ρ_eff −0.70) | effect-size proxy |
| TxPert + PonderNet (mag-anchored) | learned | ✓ | ✗ (ρ_eff −0.87) | effect-size proxy |
| magnitude-free learned gate | learned | ✗ (0.14) | ✓ | seed-noise |
| free-convergence toy | emergent | ✗ (collapses) | — | training transient |
| bio-fused ACT standalone (APPNP) | learned | ✗ | — | collapses to KL prior |
| **bio-fused ACT on TxPert (mag-free)** | **learned** | **✓ (0.71)** | **✗ (ρ_eff −0.84)** | **effect-size proxy** |

**Every model with a *learned* halting signature either collapses to an effect-size proxy or is
seed-noise.** The single point that reaches the useful corner (reproducible AND non-redundant) is the
**fixed-rule** cascade-saturation graph statistic — which is *computed*, not learned, and is therefore
not "a model's halting behavior."

## The finding, in one sentence

On single-endpoint Perturb-seq, **a learned adaptive-depth model's halting behavior does not carry a
reproducible, effect-size-independent per-perturbation signal** — demonstrated across 2 backbones,
3 halting mechanisms, magnitude-anchored and magnitude-free losses, and now on a real published graph
model (TxPert) — because refinement does not improve a single static endpoint (hop 1 is optimal for
~96% of perturbations), so any surviving depth variation is the effect-size gradient bleeding through.
This is the thesis's pre-registered negative, in its most complete form.

## What remains usable

The **fixed** network-cascade-depth statistic (how fast a perturbation's front saturates on a
biological graph — `bio_halt.py`, ρ=0.96, effect-independent) is a legitimate, reproducible
per-perturbation feature. It is a graph statistic, not a learned halting signature, and it is the
defensible route for ranking direct- vs. cascade-mediated targets (e.g. the Marson/Pritchard CD4+
T-cell screen).
