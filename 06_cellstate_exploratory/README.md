# Stage 6 — Cell-state-conditioned refinement (exploratory side-analysis, 4 lines)

> **EXPLORATORY — not part of the paper unless promoted.** This is a frozen-model side-analysis, walled
> off from the manuscript.

The question here is whether refinement depth is really a perturbation property or just a readout of the
cell's *starting* state. Using each Replogle line's frozen Stage-4 checkpoint with the basal context
sampled from G1 / S / G2M control cells (target held fixed), the answer is that **depth is invariant to
basal cell-cycle state**: cross-state ρ(E[N]) = 0.996 / 0.998 / 0.989 / 0.999 across K562 / HepG2 /
Jurkat / RPE1, and the real cross-state variance is *smaller* than a state-label-permutation null in
every line. What depth *does* couple to is the **shape of the model-implied response trajectory** — a
directional / nonlinear-correction signal (standardized β = +0.14 to +0.33, all four lines) — while the
originally-hypothesized late cell-cycle / DNA-damage program is not supported. This invariance is what
makes the CD4 activation-phenotype separation (Stage 7) *clean*: depth cannot be a cell-cycle confound
if it does not read cell-cycle state at all. `results/` holds the per-line layerwise and conditioned
tables and interaction JSONs; `figures/` the per-line and combined figures; `docs/` the synthesis note
that connects this track to the CD4 biology. It remains exploratory precisely because it is on a
different backbone from the application; Stage 8 is the piece that would let the geometry be tested on a
CD4 backbone directly.
