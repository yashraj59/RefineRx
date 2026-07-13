# Stage 0 — Perturbation-model catalog & deep code review (the starting point)

This is where the project begins: before grafting any halting mechanism, we surveyed the landscape of
single-cell perturbation models to decide *which* backbones could even host an adaptive-depth
refinement loop. The `docs/` here are **code-grounded** notes — each model was cloned or read via raw
GitHub blobs (weights never fetched), and every claim about its forward pass, training loop, iterative
structure, and inference API cites the file and class actually read, across five model families
(foundation-model / condition-adaptation, knowledge-graph / GRN, chemical / drug, optimal-transport /
flow / diffusion, and transfer / spatial + benchmarks). The `results/act_graftability_matrix.csv`
distills this into a scored matrix of *where a learned-halting (ACT) refinement loop would attach* on
each architecture — the "iterative_structure" lens that drove every model choice in Stages 1–4. In the
arc, this stage frames the hypothesis and picks the candidates; the graftability scores flagged the
diffusion and message-passing hosts (scDiff, TxPert) tried next, and later the STATE foundation model
that ultimately worked.
