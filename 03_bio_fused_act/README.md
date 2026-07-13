# Stage 3 — Biology-fused ACT (STRING-propagation halt features)

The turning point in the failed-graft sequence: instead of a freely-learned recurrence, refinement here
is **propagation on a FIXED biological network** — the STRING PPI/functional graph over the perturbed
genes (APPNP-style personalized propagation), with halting defined as **biological cascade saturation**
(stop when the activated front stops growing). Because the graph is fixed biology, depth is a function
of *where the perturbed gene sits in the network*, not a parameter the optimizer can drive to a
constant — so for the first time in the project the signature reaches the **useful corner**:
reproducible across seeds (**ρ = 0.96**) and non-redundant (R² = 0.038 from effect size + #DE + cell
count + graph degree combined), decoupled from effect size (ρ = +0.11), and stable at full training
(recon 0.0005) rather than collapsing like the free-recurrence toy. Its honest limitation, recorded in
`docs/`, is a **narrow dynamic range** on the small, dense 88-gene scaffold (most perturbations at ~1
round, a cascade-propagator tail at ~2). The `code/` holds the bio-ACT and bio-halt variants (including
a TxPert-fused version) and their training loops; `figures/` shows the result and the diagnosis. In the
arc this stage proves depth *can* be made identifiable when pinned to fixed external structure — which
motivates Stage 4's move to a large pretrained backbone where the depth axis has room to develop.
