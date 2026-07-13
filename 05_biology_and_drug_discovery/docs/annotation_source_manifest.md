# Annotation Source Manifest — Druggability + Toxicity/Safety Table

**Entity set:** 1708-gene union (Replogle Perturb-seq CRISPRi knockdowns across hepg2/jurkat/k562/rpe1).
**Denominator for all coverage numbers: 1708 genes** (frozen; deduped).
**Retrieved:** 2026-07-12
**Discipline:** autocollect-bio — every value carries a named source + date, or the literal token `NOT_FOUND` (looked, absent) / `NOT_FETCHED` (deliberately not queried for that subset). No value is guessed or model-derived.

---

## Sources used (connector → method → what it supplied)

| Source | Connector · method | Fields supplied | Genes queried |
|---|---|---|---|
| **Open Targets Platform** (GraphQL, api.platform.opentargets.org/api/v4) | clinical-genomics · `open_targets_graphql` (+ `mapIds`) | symbol→Ensembl mapping; `targetClass`; `tractability` (SM/AB buckets); `drugAndClinicalCandidates` (drug, phase, MoA); `safetyLiabilities`; `geneticConstraint` (pLI/LOEUF, gnomAD v2.1.1) | all 1708 (mapping); 1707 (target data) |
| **ChEMBL** (EBI) | chembl · `target_search` + `get_mechanism` | direct single-protein `chembl_max_phase`; curated MoA; action types — **confirmation** of OT drug calls | 157 (OT-drugged set only) |
| **gnomAD r4** | variants · `gene_constraint` | pLI / LOEUF (`oe_lof_upper`) — **fallback** where OT constraint absent | 56 (genes missing OT constraint) |
| **openFDA** (Drugs@FDA SPL labels) | drug-regulatory · `search_drug_labels` | boxed warnings; top adverse events — for the drug matched to each drugged gene | 157 (drugged set only) |
| **Seed CSVs** (prior curation) | tcell (284) + myeloid (152) annotation tables | cross-check of Ensembl IDs + curated ChEMBL max_phase (Tier-3 reconciliation) | 35 union overlap |
| **Ensembl BioMart** | biomart · `batch_translate` / `get_translation` | ID-mapping cross-validation of alias remaps | 47 aliased symbols |

---

## ID mapping (Tier 1a)

- **1707/1708 (99.9%) symbols mapped to Ensembl gene IDs** via Open Targets `mapIds` (entity=target).
- 1 unmapped: `AC118549.1` (clone-based name; no Open Targets/BioMart target record) → `ensembl_id=NOT_FOUND`, all downstream fields NOT_FOUND.
- **141 ambiguous / alias terms disambiguated** by preferring exact `approvedSymbol`, then the HGNC-renamed `<sym>1` form confirmed against obsolete/synonym symbols. **2 genuine mis-maps caught & corrected**: `GARS`→GARS1 (not GART), `QARS`→QARS1 (not EPRS1). All 1707 resolved IDs are well-formed and unique (no symbol collisions).

---

## DRUGGABILITY coverage (denominator 1708)

| Field | Found | NOT_FOUND | NOT_FETCHED | % |
|---|---|---|---|---|
| target_class (OT) | 863 | 845 | – | 50.5% |
| ot_tractability_sm | 978 | 730 | – | 57.3% |
| ot_tractability_ab | 429 | 1279 | – | 25.1% |
| ot_max_phase (OT any drug) | 157 | 1551 | – | 9.2% |
| chembl_max_phase (direct mech) | 65 | 93 | 1550 | 3.8% |
| mechanism_of_action | 157 | 1551 | – | 9.2% |

**drug_status_bucket (derived, reconciled OT + ChEMBL):**

| Bucket | N | Definition |
|---|---|---|
| approved_target | 41 | ChEMBL direct mechanism + phase-4 drug |
| clinical_target | 24 | ChEMBL direct mechanism, max phase 1–3 |
| drugged_other | 92 | OT lists a drug, but **no ChEMBL direct single-protein mechanism** — drug hits the gene via a complex / fusion / pathway (e.g. ribosomal subunits→ribosome-binding drugs, BCR→BCR-ABL, ATP1A1→cardiac glycosides curated at complex level) |
| undrugged | 1551 | no drug in OT or ChEMBL |

- **157 genes are drugged** by Open Targets (any associated drug); of these **65 confirmed** by a ChEMBL direct single-protein mechanism.
- **Reconciliation rule (Tier 3):** OT `drugAndClinicalCandidates` is inclusive (association-level, incl. complexes/fusions); ChEMBL `get_mechanism` is strict (direct single-protein). Both phases are kept as separate columns (`ot_max_phase`, `chembl_max_phase`); the bucket privileges ChEMBL-confirmed direct targets. `chembl_max_phase=NOT_FETCHED` on the 1550 OT-undrugged genes (subset criterion: ChEMBL queried only where OT already indicated a drug — a stated depth-tier, not a coverage gap).
- **Caveat (documented, not hidden):** ChEMBL stores some mechanisms only at the protein-*complex* level, so a bona-fide target whose drug is curated against the complex (e.g. ATP1A1 / Na-K-ATPase, proteasome subunits) lands in `drugged_other` rather than `approved_target`. `ot_max_phase=4` + `approved_drugs` are retained on these rows so downstream agents can re-promote them.

---

## TOXICITY / SAFETY coverage (denominator 1708)

| Field | Found | NOT_FOUND | NOT_FETCHED | % |
|---|---|---|---|---|
| gnomad_pli | 1693 | 15 | – | 99.1% |
| gnomad_loeuf | 1693 | 15 | – | 99.1% |
| fda_boxed_warning | 29 | 98 | 1551 | 1.7% |
| fda_adverse_events_top | 59 | 98 | 1551 | 3.5% |

- **LoF constraint (pLI/LOEUF):** primary source = Open Targets `geneticConstraint` (gnomAD v2.1.1); **verified equal** to gnomAD's own pLI/LOEUF on a 6-gene spanning check (OT `score`=pLI, `oeUpper`=LOEUF). gnomAD r4 `gene_constraint` used as fallback for 56 genes OT lacked constraint for (X-linked/mitochondrial mostly); 54/56 resolved, 2 gnomAD errors. 15 genes end NOT_FOUND (1 unmapped + genes absent from both constraint tables).
- **lof_constraint_flag:** constrained if pLI≥0.9 **or** LOEUF<0.35. → **constrained 554, tolerant 1139, unknown 15.**
- **OT safety liabilities:** 57 genes carry ≥1 liability (149 liabilities total); stored as JSON array-of-objects `{event, effect, datasource}` (+ literature PMID where present). Sources include Bhatt/Urban toxicity target lists, PharmGKB, AOP-Wiki, Brennan et al. 2024.
- **FDA labels** (drugged genes only, 157 queried): 59 genes matched a live SPL label; **29 carry a boxed warning**. Undrugged genes = `NOT_FETCHED` for both FDA fields (not looked — no drug to look up).

---

## Validation (Tier 2) — all passed

- Row count = 1708, symbols unique, order identical to input union. ✓
- No blank/NaN cells anywhere (every cell is a value, NOT_FOUND, or NOT_FETCHED). ✓
- All Ensembl IDs well-formed `ENSG\d{11}` and unique. ✓
- `drug_status_bucket` ∈ {approved_target, clinical_target, drugged_other, undrugged}; `lof_constraint_flag` ∈ {constrained, tolerant, unknown}. ✓
- Internal consistency: undrugged↔empty drug lists; approved_target↔phase-4 signal; constraint flag matches numeric pLI/LOEUF. ✓
- **Live spot-check: 12/12 genes re-queried against Open Targets matched recorded pLI/LOEUF/max-phase/drug-count/liability-count exactly; GART vs GARS1 ID-distinctness confirmed.** ✓

---

## Systematic gaps (honest)

1. **mechanism_of_action 9.2%** and **ot_max_phase 9.2%** are low *by design* — only 157/1708 union genes have any drug; the rest are legitimately undrugged (this is a CRISPRi essentiality screen, not a drug-target panel). Not a retrieval failure.
2. **target_class 50.5%** — Open Targets ChEMBL-derived target-class tree only covers ~half the proteome (enzymes/receptors/channels well; adaptors, structural, ribosomal proteins often absent).
3. **15 genes lack pLI/LOEUF** — 1 unmapped clone name + a handful absent from gnomAD constraint (very short / non-canonical transcripts).
4. **chembl_max_phase / FDA fields NOT_FETCHED on undrugged genes** — deliberate depth-tiering, not a gap.
5. `AC118549.1` unresolved end-to-end (row present, all fields NOT_FOUND).

## Reproducibility
Re-runnable from this manifest: OT `mapIds`→`targets(ensemblIds)` batched (40/call) with the query in the frame log; ChEMBL/gnomAD/FDA per-gene on the derived drugged/fallback subsets. gnomAD & OT servers are rate-limited and shared — batch and pace (0.1–0.15s).
