# Stage 5 — Biology validation, depth UMAP/leiden, drug-target discovery (4 lines)

With a working, reproducible depth signature from Stage 4, this stage asks what it is *for*: it maps
the depth structure (UMAP + Leiden) across the four Replogle lines and runs a drug-target discovery
pipeline that ranks undrugged perturbations by response cosine + STRING adjacency to approved/clinical
anchors. The honest verdict on the halting axis is **3 of 4 lines negative**: adding E[N] does *not*
sharpen drug-target-class discrimination beyond response + network similarity (HepG2/Jurkat/K562
redundant or irrelevant; RPE1 a weak, drug-class-specific tie-breaker). E[N] encodes response
*complexity*, which is largely orthogonal to pharmacological *target class* — a robust finding that held
across two annotation versions. What the pipeline *does* deliver is a coherent candidate ranking; the
defensible cross-line hits are the **safety-pass (non-essential) recurrent set**, led by
**MIOS / LAMTOR1 → MTOR**, with the important caveat that raw recurrence is confounded by essentiality
(core-essential genes recur everywhere and are exactly what you would not inhibit). `docs/` records the
biology validation and the annotation-source manifests; `results/` holds the per-line candidate tables,
the merged/consistency tables, and the validation summary; `figures/` shows the depth UMAP/Leiden,
functional clusters, drug-toxicity structure, and per-line discovery panels. All candidates are
hypotheses ("same class as anchor X"), automatically annotated from public databases with source tags.
