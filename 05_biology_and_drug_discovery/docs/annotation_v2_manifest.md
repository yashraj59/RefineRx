# Annotation v2.2 — provided annotation merged (Claude-generated from public DBs) + DepMap common-essential

## Source & provenance
Merged 1660/1708 genes with annotation from the provided repo
github.com/yashraj59/tcell-perturbed-gene-annotation. IMPORTANT: that annotation is **Claude-generated
from public databases** (Open Targets, ChEMBL, gnomAD, DepMap, Pharos, STRING, FDA), NOT hand-curated —
every value carries a `_source` tag per the repo's GENERATION_METHODS.md (retrieved 2026-07-08).
It is combined here with the annotation this project built the same way; all values are public-DB-sourced.

## Cell-type-specific essentiality (the ONLY per-line field; everything else is gene-level)
The provided repo has exact DepMap per-line gene-effect only for lines in Open Targets' screen subset:
K562 (ACH000551) and RPE1 (clones) — used directly (public DepMap CRISPR Chronos).

HepG2 and Jurkat are NOT in OT's ~50-screen-per-gene subset, and the full DepMap CRISPRGeneEffect matrix
is unreachable from the analysis sandbox (figshare blocked; depmap.org behind a JS bot wall; GCS mirror on
the exfiltration denylist). Therefore:
  - K562, RPE1: per-line geneEffect (geneEffect<-1.0 = strongly essential), common-essential fallback where a pert isn't in the matrix.
  - HepG2, Jurkat: DepMap COMMON-ESSENTIAL flag (Open Targets target.isEssential, NOT screen-capped) — a pan-cell essentiality axis. Marked in essentiality_source_hepg2/jurkat. No per-line values fabricated.

Common-essential (OT isEssential) pulled for 1697/1707 union genes: 1245 essential, 452 not, 10 NA.

## Columns
- depmap_common_essential (bool); per line L in {k562,rpe1,hepg2,jurkat}: depmap_geneEffect_L (K562/RPE1 only),
  essentiality_call_L, essentiality_source_L (provenance), tox_essential_L (strongly essential per-line OR common-essential => likely toxic to inhibit).
- annotation_provenance: all values public-DB-sourced; Claude-generated, not hand-curated.
- drug_status_reconciled reconciles OT/ChEMBL with Pharos Tclin/Tchem:
  approved=121,
  clinical=29,
  drugged_other=124,
  undrugged=1434.
- Kept all OT/ChEMBL/gnomAD/FDA columns + Pharos TDL, nearest_string_neighbors, immune_disease_assoc,
  tractability, pathway/protein class, function (all public-DB-sourced).

## Per-line files
per_line_essentiality/essentiality_{k562,rpe1,hepg2,jurkat}.csv — resolved essentiality call + source per gene.
