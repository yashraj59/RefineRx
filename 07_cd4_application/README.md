# Stage 7 — CD4+ T-cell application (paper Part 2): single-cell discovery

The application the whole method was built toward: the Stage-4 STATE oracle-ACT applied to the
Marson/Pritchard genome-wide **CD4+ T-cell CRISPRi Perturb-seq** screen (two donors, three conditions —
Rest / Stim 8h / Stim 48h — at single-cell resolution, per-condition frozen backbones). Three results
carry Part 2 of the paper. **Reproducibility:** within-donor split-half ρ(E[N]) = **0.615 → 0.678 →
0.748** (Rest → 8h → 48h) — well above the pseudobulk donor floor and rising monotonically with
stimulation. **Portability:** the signature is donor-specific in rest/early activation (cross-donor
ρ ≈ −0.1) but becomes **portable at the converged stimulated endpoint** (cross-donor ρ = +0.49 at 48h),
mirroring the Replogle cross-line result and consistent with depth being context-specific rather than a
portable invariant. **Discovery:** at Stim 48h, E[N] separates four phenotype categories
(Kruskal–Wallis **p = 5.2×10⁻²⁸**), and **resting-sparing suppressors converge shallower** than damaging
(Cliff's δ = −0.16) and generic (δ = −0.24) perturbations — they need less iterative refinement to reach
their endpoint. Unlike the immortalized Replogle lines, no depth cluster is drug-target-enriched in CD4
(a cell-context-dependent negative). `code/` holds the HVG prep variants (log1p, donor-out, LODO), the
backbone, and the oracle training; `docs/` the discovery report; `results/` the per-perturbation
signature, cross-donor E[N], phenotype categories, and the app-ready perturbation map; `figures/` the
headline discovery panel, depth UMAPs, and the biology / drug-toxicity views.
