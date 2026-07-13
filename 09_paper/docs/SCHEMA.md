# SCHEMA.md — Drug-target Discovery Results Bundle

Single entry point for the interactive visualization app. **Schema 1.2** adds the `cd4_single_cell` block (CD4+ T-cell single-cell refinement depth, 2 donors x 3 conditions) alongside the 4-Replogle-line drug-discovery block. All data files are Claude Science
**artifacts** referenced by `version_id` in `results_manifest.json`. Resolve a version_id to a URL/path
via the artifact store (e.g. `/artifacts/<version_id>` or `host.artifact_path(version_id)`).

## Top-level: `results_manifest.json`
```
schema_version, generated, project, description
thesis_context   : {act_signature, discovery_question, headline_verdict}
cell_lines       : ["hepg2","jurkat","k562","rpe1"]
verdicts         : {line -> "adds"|"redundant"|"irrelevant"}   # ACT value-add per line
per_line         : {line -> {verdict, n_candidates, n_anchors, n_clusters,
                             candidates_csv, candidates_json, clusters_json,
                             stats_json, figure_png, report_md}}   # each value = artifact version_id
shared           : {annotation_csv, annotation_json, annotation_manifest,
                    string_union_net, pert_union, essentiality_<line>, act_signatures{line->vid}}
consistency      : {consistency_csv, consistency_json, merged_long_csv, merged_long_json, summary_json}
node_key_map     : {line -> {canonical_key -> that line's actual node key}}   # normalize per-node coords
known_caveats    : [ ... ]
```

## Per-line files (per_line.<line>.*)

### candidates_csv / candidates_json  (ranked undrugged candidate targets)
One row per undrugged candidate. Core columns (present in all lines; names may vary — see notes):
- `rank` (int), `candidate` (gene symbol), `composite_score`
- `nearest_approved_anchor` / `nearest_anchor` — the approved/clinical target it resembles
- `anchor_target_class`, anchor mechanism/drugs (col name varies: anchor_moa/anchor_mechanism)
- `response_cosine` — cosine to nearest anchor's response vector
- `candidate_E_N` / `EN`, `candidate_halt_confidence`, `oracle_rstar`, `effect_size`, `n_de`, `n_cells`
- `delta_E_N` — |E[N] candidate − anchor| (the ACT axis feature)
- Safety: `gnomad_pli`, `gnomad_loeuf`, `lof_constraint_flag`, `n_safety_events`, `fda_boxed_warning`,
  per-line `tox_essential`/essentiality, `safety_flag`
- `cluster`, `x`,`y` (2D embedding), `rationale`
NOTE: schemas differ slightly across the 4 agents (51–64 cols). `candidate`, `rank`, `composite_score`,
`anchor_target_class`, `cluster` are guaranteed in all four. Use `candidates_merged_long.*` for a
normalized cross-line view (12 canonical cols).

### clusters_json  (per-perturbation embedding + cluster assignment)  — for interactive scatter
```
line, embedding (method note), n_clusters, modularity/leiden_resolution
nodes: [ {pert, cluster, x, y, (pca_x/pca_y), drug_status, role/is_anchor,
          EN, effect_size, target_class, ...}, ... ]   # ONE POINT PER PERTURBATION
cluster_summary / clusters: [ {cluster, n, n_approved, n_clinical, n_undrugged, dominant_pathway}, ...]
```
IMPORTANT — per-node key names vary by line. Use `node_key_map[line]` from the manifest to normalize:
canonical `pert,x,y,pca_x,pca_y,EN,drug_status,is_anchor,effect_size`. (e.g. jurkat uses `perturbation`,
rpe1 uses `E_N` and has no PCA coords.) All four HAVE `x,y` for the primary scatter.

### stats_json  (the ACT value-add test)
Per-line statistical result. Keys vary but always include `act_value_add_verdict` (or `ACT_value_add_verdict`)
and the supporting numbers: approved-vs-undrugged E[N] test (Mann-Whitney p, logistic coef+p controlling
effect_size+n_de), and the pairwise same-class discrimination test (gene-disjoint CV AUC cosine vs
cosine+|ΔE[N]|, bootstrap ΔAUC CI). `counts` sub-dict has n anchors/candidates/clusters.

### figure_png  (multi-panel discovery figure)
(i) 2D embedding colored by drug status w/ anchors marked, (ii) E[N] approved vs undrugged,
(iii) ACT value-add (AUC bars / partial-corr with CI), (iv) top candidates+anchors.

### report_md  (<=2pp per-line writeup)  — verdict, top ~15 candidates + safety, caveats.

## Shared files (shared.*)

- `annotation_csv/json` (annotation_union_v2, 1708 x 73) — druggability + toxicity for the pert union,
  per-value provenance. Key cols: `symbol`, `drug_status_reconciled` (approved_target/clinical_target/
  drugged_other/undrugged), `user_pharos_tdl`, `user_nearest_string_neighbors`, `user_immune_disease_assoc`,
  per-line `depmap_geneEffect_<line>` (K562/RPE1 exact), `essentiality_call_<line>`,
  `essentiality_source_<line>` (provenance), `tox_essential_<line>` (bool), `depmap_common_essential`,
  gnomAD pLI/LOEUF, OT liabilities, FDA boxed warnings. `annotation_manifest` = methods/provenance.
- `string_union_net` — STRING network over the 1708-gene union: {node_genes, edges, weights, pert_node_idx}.
- `pert_union` — the 1708-pert union + per_line membership lists.
- `essentiality_<line>` (4 CSVs) — resolved per-line essentiality call + source + LoF + drug status.
- `act_signatures{line->vid}` — per-line ACT signature CSV: perturbation, expected_rounds (E[N]),
  halt_confidence, oracle_rstar, argmin_depth, effect_size, n_de, n_cells, is_val.

## Consistency files (consistency.*)

- `consistency_csv/json` (candidate_consistency, 1415 rows) — one row per unique candidate across lines:
  `candidate, n_lines, lines, mean_score_pct, best_rank, mean_response_cosine, anchor_class_mode,
  anchor_class_consistency, anchors, mean_delta_E_N, any_tox_essential, all_tox_essential, recurrence_note`.
- `merged_long_csv/json` (candidates_merged_long, 3178 rows) — normalized long form, one row per
  (line, candidate): the 12 canonical columns. Best table for cross-line app views.
- `summary_json` (consistency_summary) — headline counts + top safety-pass 4-line candidates + caveat.

## CD4+ T-cell single-cell block (`cd4_single_cell.*`)  — NEW in schema 1.2

A separate application: STATE oracle-ACT refinement depth in primary human CD4+ T cells (Marson/Pritchard
GWCD4i CRISPRi Perturb-seq), single-cell resolution, **2 donors (D1, D4) x 3 conditions (Rest, Stim8hr,
Stim48hr)**. This is a per-cell-type / per-donor experiment, NOT part of the 4-Replogle-line drug-discovery
block above — keep it on its own app view.

`cd4_single_cell` manifest keys: `description, backbone, donors, conditions, headline_verdict, stats,
leiden, files{...->version_id}, node_key_map_cd4, key_findings`.

### files.perturbation_map_csv  (PRIMARY app-ready table, 4989 perturbations x 33 cols) — the CD4 scatter
One row per perturbation (gene) that is common+valid across all 3 conditions. Columns:
- `gene`, `umap_x`, `umap_y` (depth-signature UMAP), `leiden` (0-8, 9 clusters), `category`
  (resting_sparing_suppressor / inflammatory / damaging / other)
- 12-feature signature: `EN_<cond>`, `HC_<cond>`, `NL_<cond>`, `PE_<cond>` for cond in {Rest, Stim8hr, Stim48hr}
  (E[N]=expected rounds, HC=halt confidence, NL=nonlinear correction, PE=predicted error)
- `ncell_min`, `rest_engagement`, `stim48_axis_proj`, `stim8_axis_proj` (activation-axis projections)
- Annotation (from the user's tcell annotation): `drug_target_status`, `approved_drugs`, `pharos_tdl`,
  `tractability_modality`, `immune_disease_assoc`, `depmap_essentiality`, `gnomad_loeuf`, `function`,
  `pathway_family`, `protein_class_family`, `nearest_string_neighbors`, `is_deep_subset`
- Normalize node keys via `node_key_map_cd4`: pert=`gene`, x=`umap_x`, y=`umap_y`, cluster=`leiden`,
  EN_primary=`EN_Stim48hr`, drug_status=`drug_target_status`, essentiality=`depmap_essentiality`.

### files.crossdonor_EN_csv  (17659 rows: gene, condition, EN_D1, EN_D4) — cross-donor scatter per condition
### files.phenotype_categories_csv  (4989 x 5) — gene -> category + axis projections
### files.signature_annotated_csv  (4989 x 29) — full per-gene signature + annotation join
### files.results_summary_json  — all CD4 stats (stage1/stageX/stage2/stage3, medians, deltas, p-values)
### files.discovery_figure_png  — 4-panel headline (within vs cross-donor rho; cross-donor scatters; Stage-3 violins)
### files.depth_umap_png  — 3-panel depth UMAP (leiden / E[N]@Stim48h / phenotype category)
### files.report_md  — CD4 discovery short report
### files.signatures_raw_tgz  — raw per-condition signature npz + stage JSONs (checkpoint)

### CD4 app guardrails (surface in the app)
- Within-donor reproducibility is high (rho 0.62-0.75) but cross-donor depth is ~0 EXCEPT at Stim48h
  (rho=0.49). Default the cross-donor scatter to the Stim48hr condition to show the portable case.
- Thesis result lives at Stim48h: resting-sparing suppressors have LOWER E[N] than damaging/other
  (Cliff's d=-0.16 / -0.24). Color the Stage-3 view by `category`, y-axis `EN_Stim48hr`.
- No CD4 depth cluster is drug-enriched (~97% essential) — do NOT reuse the Replogle drug-cluster overlay here.

## Reading order for the app
1. `results_manifest.json` → cell_lines, verdicts, all artifact version_ids.
2. Per line: `clusters_json.nodes` (scatter, normalize via node_key_map) + `candidates_csv` (ranked table)
   + `stats_json` (verdict panel) + `figure_png` + `report_md`.
3. Cross-line: `consistency_csv` (recurrence table) + `summary_json` (headline) + `merged_long_csv`.
4. Detail lookups: `annotation_csv` joined on gene `symbol`; `act_signatures[line]` for E[N] per pert.
5. CD4 block (separate view): `cd4_single_cell.files.perturbation_map_csv` (scatter, normalize via
   `node_key_map_cd4`) + `crossdonor_EN_csv` (cross-donor scatter) + `results_summary_json` (stats) +
   `discovery_figure_png`/`depth_umap_png` + `report_md`.
6. Paper: `paper.pdf` (25pp full write-up; CD4 = Section 5.8).

## Interpretation guardrails (surface in the app)
- The ACT value-add verdict is NEGATIVE in 3/4 lines (redundant/irrelevant) and weak-positive in rpe1.
  Present E[N] as a secondary/tie-breaker axis, not the primary ranker.
- Cross-line recurrence is dominated by core-essential genes (flagged `tox_essential_<line>` / recurrence_note).
  Default the consistency view to `recurrence_note == "non_essential(safety-pass)"`.
- Candidates are HYPOTHESES ("same class as anchor X"), not validated targets.
