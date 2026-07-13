# Cross-Cell-Line Synthesis — ACT Halting & Drug-Target Discovery

**Scope.** Four Replogle Perturb-seq lines (HepG2, Jurkat, K562, RPE1). For each, an ACT adaptive-depth
halting signature E[N] was read from a frozen ARC ST-SE backbone, and a drug-target discovery pipeline
ranked undrugged perturbations by response-similarity + STRING adjacency to approved/clinical anchors.
The core hypothesis tested: **does the ACT halting axis E[N] add drug-target-class discrimination beyond
response + network similarity?**

## 1. The ACT value-add verdict (the headline)

| Line | Verdict | ΔAUC from adding \|ΔE[N]\| | E[N] approved-vs-undrugged |
|------|---------|---------------------------|----------------------------|
| HepG2  | **redundant**  | −0.010 (gene-disjoint CV) | n.s. |
| Jurkat | **irrelevant** | −0.002 (p=0.77); STRING +0.024 (p=0.001) | n.s. |
| K562   | **redundant**  | ~0 out-of-sample beyond response+effect | separates in-sample only |
| RPE1   | **adds (weak)** | +0.027 bootstrap, CI excludes 0 (LR p=1.9e-7) | tie-breaker, drug-class-specific |

**Conclusion: 3 of 4 lines NEGATIVE.** The ACT halting depth is *not* a useful discriminator of drug-target
class beyond what the response vector and STRING network already provide. RPE1 is a weak, drug-class-specific
positive — E[N] acts as a tie-breaker among DNA-replication-machinery candidates, not a primary driver.

This is a **legitimate, robust thesis finding**, not a pipeline failure. It held identical across two
annotation versions (before/after merging the richer per-line essentiality + more drug anchors), so it is
not an artifact of anchor choice. It is consistent with the established property of the signature: E[N] is
reproducible within a cell type and effect-size-independent, but it encodes *response complexity*, which is
largely orthogonal to *pharmacological target class*.

Importantly, E[N] is NOT merely redundant with response magnitude (|ρ(E[N], effect_size)| is low, by design)
— it simply carries little **target-class** information. The two negatives ("redundant" vs "irrelevant")
differ only in whether E[N] separates approved-from-undrugged at all (irrelevant) versus separates them
in-sample but adds nothing out-of-sample (redundant).

## 2. What the discovery pipeline DOES deliver

The candidate **ranking** (response cosine + STRING adjacency to approved anchors), independent of the ACT
axis, produces coherent, mechanism-consistent hypotheses. Per-line highlights (undrugged → nearest approved
anchor):
- **HepG2:** RPA2/RPA1 → ATR (ssDNA-binding replication stress), BORA → PLK1 (mitotic kinase cofactor).
- **Jurkat:** TTI1/SEH1L → MTOR (mTOR complex scaffolding), UHRF1 → DNMT1 (maintenance-methylation partner).
- **K562:** DNA-replication (→ POLD1/POLE/PRIM1) and OXPHOS (→ NDUFB) obligate complex partners.
- **RPE1:** DNA-replication machinery (RPA/MCM/CDC6/ORC1/RFC/GINS4) anchored on the approved primase target PRIM1.

## 3. Cross-line consistency (candidate_consistency.csv, 1415 unique candidates)

Recurrence: **100 candidates in all 4 lines, 439 in 3, 585 in 2, 291 in 1.**

**Critical caveat — recurrence is confounded by essentiality.** 1026 / 1415 candidates are core-essential in
every nominating line (flagged `tox_essential_<line>` / `recurrence_note`). These recur precisely because
core-essential ribosome-biogenesis / DNA-replication / OXPHOS genes produce large, similar transcriptional
responses in *every* line — and they are exactly the genes you would NOT want to inhibit therapeutically.

**The defensible recurrent hypotheses are the 378 safety-pass candidates** (non-essential in all nominating
lines). The standout:
- **MIOS → MTOR** — all 4 lines, mean score-percentile 0.90, non-essential. MIOS is a GATOR2-complex member
  regulating mTORC1; it consistently maps near the approved mTOR-inhibitor target in every cell line.
- **LAMTOR1 → MTOR** — 3 lines, corroborating the same mTOR-regulatory axis (Ragulator complex).
- Others (≥3 lines, safety-pass): CENPJ, MPHOSPH6, BRIP1 → POLD1/SEM1 (replication), UFM1 → ribosomal.

## 4. Cell-type specificity

The ACT signature itself does **not** port across lines (established: cross-line E[N] ρ = 0.14, cross-donor
0.06; within-line split-half ρ = 0.72–0.87). Consistent with that, the *per-line* candidate rankings differ,
and cross-line recurrence — once essentiality is removed — is modest. The mTOR-regulatory hits (MIOS/LAMTOR1)
are the clearest exception: a genuinely cross-line-robust, non-toxic, mechanistically-coherent signal.

## 5. Bottom line for the thesis

1. Adaptive-depth halting E[N] is reproducible and cell-type-specific, but **does not sharpen drug-target-class
   identification** beyond response + network similarity (3/4 lines negative; RPE1 a weak tie-breaker).
2. The response+STRING ranking is still useful and yields sensible same-class hypotheses.
3. Cross-line consistency must be read against essentiality: the safety-pass recurrent set (led by
   MIOS/LAMTOR1 → MTOR) is the actionable output.

_All candidates are hypotheses ("same class as anchor X"), not validated targets. Annotation is
Claude-generated from public databases (Open Targets, ChEMBL, gnomAD, DepMap, Pharos, STRING, FDA), not
hand-curated; every value carries a source tag._
