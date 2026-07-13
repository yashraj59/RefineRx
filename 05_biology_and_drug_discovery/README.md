# Stage 5 — Biology validation, depth UMAP/leiden, drug-target discovery (4 lines)

With a working, reproducible depth signature from Stage 4, this stage asks what it is *for*: it maps
the depth structure (UMAP + Leiden) across the four Replogle lines and runs a drug-target discovery
pipeline that ranks undrugged perturbations by response cosine + STRING adjacency to approved/clinical
anchors.

**Two claims must be held apart** — collapsing them into "redundant with STRING" is the mistake this
section was rewritten to fix.

- **The positive (what the signature does).** Clustering perturbations on the depth signature alone
  recovers functional gene classes in which **approved/clinical drug targets concentrate** — a
  translation/ribosome cluster in *every* line, per-cluster odds ratios **2.8–9.8**. And that depth
  descriptor is **non-redundant with network topology**: no GRN/PPI graph statistic reproduces the
  per-perturbation ordering of E[N] (best |ρ| = 0.23, below a 0.3 novelty ceiling — see Stage 4's GRN
  baseline). The adaptive signature genuinely captures what the simple baselines miss.
- **The narrow negative (one downstream classifier).** For one incremental target-*class* ranking
  task — adding |ΔE[N]| on top of a STRING baseline built from the *same* functional modules — there
  is no AUC lift in 3 of 4 lines (RPE1 a weak, drug-class-specific tie-breaker). This is **task-level
  non-additivity, not descriptor-level redundancy**: STRING and depth-clustering happen to surface the
  same functional strata that the target-*class* label tracks, so for predicting *that one label* they
  overlap. It says nothing about E[N]'s information content, which remains non-redundant (above).

The ranking itself is useful: the defensible cross-line hits are the **safety-pass (non-essential)
recurrent set**, led by **MIOS / LAMTOR1 → MTOR**, with the important caveat that raw recurrence is
confounded by essentiality (core-essential genes recur everywhere and are exactly what you would not
inhibit). `docs/` records the biology validation and the annotation-source manifests; `results/` holds
the per-line candidate tables, the merged/consistency tables, and the validation summary; `figures/`
shows the depth UMAP/Leiden, functional clusters, drug-toxicity structure, and per-line discovery
panels. All candidates are hypotheses ("same class as anchor X"), automatically annotated from public
databases with source tags.
