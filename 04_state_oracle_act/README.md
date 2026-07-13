# Stage 4 — STATE oracle-ACT: the method that worked (Replogle 4 lines)

The method that finally recovers a genuine *learned* per-perturbation depth. On ARC Institute's
pretrained **STATE Transition** foundation model (`arcinstitute/ST-SE-Replogle`, all weights frozen),
three results separate cleanly and together are the thesis: (1) a **computed** readout — which of the
8 frozen layers best predicts each perturbation (per-layer loss argmin) — is reproducible and
effect-independent (ρ = 0.836); (2) **naive** learned ACT still collapses to a single per-seed global
constant; but (3) an **oracle-supervised** learned halt head — a refinement-token gate kept
LayerNorm-unsaturated and trained against a per-perturbation oracle stopping round — produces a real
learned depth that is reproducible (ρ = 0.761, K562 split-half ≈ 0.76), effect-independent
(partial ρ = −0.04), and recovers both the oracle target and the computed argmin. Refit across all
four Replogle lines (K562, HepG2, Jurkat, RPE1) gives within-line split-half **ρ ≈ 0.72–0.87
(4-line mean ≈ 0.80)**. `code/` holds the ARC loaders, the adaptive-state refiner, the oracle-refine
training, signature extraction, cross-line split-half, and the four-line refit; `docs/` carries the
full ARC-STATE writeup and validation; `results/` has the per-line signatures and seed E[N] arrays;
`models/` ships the trained per-line halt-head weights (`halthead_{k562,hepg2,jurkat,rpe1}.pt`). This
is the positive result the whole arc was reaching for, and the backbone for the biology (Stage 5) and
the CD4 application (Stage 7).
