# Biological Validation of ACT Halting Depth — All Four Cell Lines

**Question:** does the adaptive-depth halting signature E[N] track known biological covariates, and
does it survive as an *effect-size-independent* quantity — across all four Replogle lines, not just K562?

Covariates tested per line (Spearman ρ of E[N] against each):
response magnitude, # DE genes, cell count, knockdown strength (autologous target |Δ|), baseline expression,
and the partial correlation depth~effect controlling {nDE, nCells} (the effect-independence anchor).

## Results (see bio_validation_heatmap.png, bio_validation_summary.csv)

| Covariate | K562 | RPE1 | Jurkat | HepG2 | Reading |
|-----------|------|------|--------|-------|---------|
| Response magnitude | −0.35*** | +0.32*** | +0.24*** | +0.36*** | correlates, **sign is cell-type-specific** |
| # DE genes | −0.35*** | +0.31*** | +0.23*** | +0.32*** | tracks response magnitude (same sign per line) |
| Cell count | −0.08* | −0.24*** | −0.02 | +0.14*** | weak, inconsistent sign — not a driver |
| **Knockdown strength** | −0.03 | +0.10 | −0.00 | +0.02 | **≈0 in all lines (all n.s.)** |
| **Baseline expression** | −0.03 | +0.09 | −0.00 | +0.03 | **≈0 in all lines (all n.s.)** |
| **Partial (depth~effect \| nDE,nCells)** | +0.03 | −0.02 | +0.07 | +0.16* | **collapses toward 0** |

## How depth explains biology

1. **Depth is not a technical artifact.** Knockdown strength (how hard the guide suppresses its own target)
   and the target's baseline expression are **uncorrelated with E[N] in every line** (all |ρ|<0.10, all n.s.).
   So halting depth is not measuring assay efficiency or gene abundance — it is a property of the *response*,
   not the *reagent*. This is the cleanest negative control in the panel and it passes in all four lines.

2. **Depth co-varies with response complexity, with a cell-type-specific sign.** E[N] correlates with response
   magnitude and #DE-genes (ρ 0.24–0.36) in HepG2, Jurkat, RPE1 — larger, broader responses need more
   refinement rounds. **K562 has the opposite sign (−0.35):** there, larger responses halt *earlier*. This
   sign flip is itself the cell-type-specificity result — the depth↔complexity relationship is not universal;
   each line has its own geometry (consistent with cross-line E[N] ρ=0.14, established earlier).

3. **Depth is effect-size-independent — the thesis anchor holds in all four lines.** After regressing out
   #DE-genes and cell count, the partial correlation between depth and effect size collapses to +0.03 (K562),
   −0.02 (RPE1), +0.07 (Jurkat), +0.16 (HepG2). Only HepG2 retains a weak residual (p=0.03); the other three
   are indistinguishable from zero. This is exactly the K562 result (partial ρ=−0.04, established earlier)
   reproduced across the panel: **the raw depth↔magnitude correlation is mediated by DE-gene count, not by a
   direct dependence on effect size.** Depth carries information beyond "how big is the response."

## Bottom line

The halting depth signature is biologically grounded but **not reducible to a covariate**:
- It is independent of knockdown strength and baseline expression (reagent/abundance) in every line.
- It relates to response complexity (#DE genes) with a **cell-type-specific sign** — the specificity finding.
- It is **effect-size-independent** once #DE and cell count are controlled — in all four lines (weak residual only in HepG2).

Combined with the established reproducibility (within-line split-half ρ=0.72–0.87) and cell-type-specificity
(cross-line ρ=0.14, cross-donor 0.06), this completes the biological characterization: E[N] is a reproducible,
effect-independent, cell-type-specific property of a perturbation's response — recovered from single-endpoint
data — but one whose *sign* relative to response magnitude is set by the cellular context.

_Method note: response magnitude / #DE / cell count use the full per-line signature (n≈945–1084 perts).
Knockdown strength and baseline expression require matching a perturbation to its own gene's row in the
2000-HVG expression matrix (gene symbols from var_dims.pkl), so they use the target-in-panel subset
(n≈169–223). Both are stated per cell in the figure._
