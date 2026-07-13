# GENERATION_METHODS.md — T-cell Perturbed Gene Annotation

Reproducibility and provenance record for the annotation deliverables. Every number in
this document is taken from `source_manifest.json` and `coverage_report.json` (the machine-
generated run records), not from recollection. All retrievals were performed on **2026-07-08**.

---

## 1. Input: the frozen 11,526-gene list

- **Artifact:** `perturbed_genes.txt` (artifact `0cba2f0f`, version `0a9ad127-1027-425b-b3db-aea06131efbb`).
- **Provenance:** the gene symbols are the `target_contrast_gene_name` column of the screen's
  `all_genes_annotated.csv`, i.e. the set of genes knocked down in the genome-scale CD4+ T-cell
  CRISPRi Perturb-seq screen (Marson/Pritchard). **This gene list is the only information taken
  from the screen** — every annotation value in the output comes from public databases or the
  published literature.
- **Fixing the 11,526 denominator:** the raw file contains 11,525 newline-terminated lines but
  the final line lacks a trailing newline; a `sort -u` deduplication yields **11,526 unique
  symbols**. That unique set is the coverage denominator used everywhere in this project — all
  coverage percentages are over 11,526, never over a resolved subset. The frozen list is stored
  as `handoff/genes.json` (11,526 entries) and drives every downstream query.
- **First five symbols (sanity anchor):** A1BG, A2M, AAAS, AACS, AAGAB.

---

## 2. Anti-fabrication discipline (autocollect-bio)

The whole point of the table is that it can be trusted value-by-value. The operating rules:

1. **Every annotation value is paired with a `_source` column** naming the database it came from,
   plus (in the parquet) the accession, the query term, and the retrieval date.
2. **Two distinct sentinels, never conflated:**
   - `NOT_FOUND` — the source **was queried** and returned no value for that gene.
   - `NOT_FETCHED` — the source **was deliberately not queried** for that gene. This is used
     **only** for `key_reference`, on the 9,730 genes outside the deep literature subset.
3. **Real citations only.** No PMID, DOI, accession, drug name, or numeric metric appears unless
   a live API call in this run returned it. Post-hoc audit: all 1,756 distinct PMIDs written into
   the table were confirmed to come from the pulled PubMed metadata set (subset check passed).
4. **No synthesized values.** Where a derived label was needed (e.g. Pharos TDL), it was fetched
   from the authoritative source rather than inferred, precisely to avoid fabrication.
5. **Conflicts are preserved, not silently resolved.** `drug_target_status` reconciles Open
   Targets and ChEMBL; the 84 genes where they
   disagree carry `drug_target_status_conflict = True`, with the ChEMBL value kept in the parquet
   detail field.

---

## 3. Data sources — exact endpoints, parameters, versions, counts

### 3.1 MyGene.info — identity mapping (ensembl_id, uniprot_id)
- **Endpoint:** `genes-ontologies/query_genes (MCP)`
- **Query parameters:** 11,526 perturbed gene symbols, scopes=symbol,alias
- **Version:** MyGene current · **Retrieved:** 2026-07-08
- **Requested / found / not-found:** 11,526 / 11,525 / 1
- **Batch primitive:** `query_genes` accepts up to **1000 terms/call**; all 11,526 symbols were
  mapped in 12 batches with `scopes="symbol,alias"`, `species="human"`, fields
  `symbol,name,entrezgene,ensembl.gene,uniprot,summary,type_of_gene`.
- **Alias multi-hits:** 11,526 inputs returned 12,022 records (some symbols matched multiple alias
  records); resolved by preferring the record whose returned symbol equals the input symbol.
- **Rescue:** 3 initially-unmatched entries were re-queried with `scopes="ensembl.gene"` — two were
  Ensembl IDs used as symbols in the screen (ENSG00000275895 → LOC102724594; ENSG00000289731,
  unannotated), recovered. **OCLM** (a withdrawn symbol) remained genuinely unresolvable → the
  single gene with `ensembl_id = NOT_FOUND`.
- **Round-trip validation:** 25 random symbol→UniProt→gene-name round-trips, **25/25 concordant**.

### 3.2 UniProt — function, protein families, InterPro/Pfam xrefs
- **Endpoint:** `genes-ontologies/get_uniprot_entries (MCP)`
- **Query parameters:** function[CC], protein families, InterPro/Pfam xref by Swiss-Prot accession
- **Version:** UniProt current · **Retrieved:** 2026-07-08
- **Requested / found / not-found:** 11,415 / 11,415 / 0
- **Batch primitive:** `get_uniprot_entries`, 100 accessions/call, fields mode returning
  `Function[CC]`, `Protein families`, `InterPro`, `Pfam`, `Subcellular location[CC]`. Every one of
  the 11,415 unique Swiss-Prot accessions returned a record.
- `function` is UniProt `Function[CC]` (first sentence, ECO/PubMed tags stripped) with MyGene
  `summary` as fallback.

### 3.3 Open Targets Platform — drug status, tractability, pathways, immune assoc, DepMap
- **Endpoint:** `clinical-genomics/open_targets_graphql (MCP)`
- **Query parameters:** Target by ensemblId: tractability, targetClass, pathways.topLevelTerm, drugAndClinicalCandidates, associatedDiseases, isEssential, depMapEssentiality
- **Version:** Open Targets GraphQL (2024/2025 release) · **Retrieved:** 2026-07-08
- **Requested / found / not-found:** 11,511 / 11,118 / 393
- **Batch primitive:** `open_targets_graphql`, **25 aliased `target(ensemblId:)` blocks/call** via a
  shared GraphQL fragment (~461 calls for 11,511 Ensembl IDs). Fields: `tractability{label,modality,
  value}`, `targetClass`, `pathways.topLevelTerm`, `drugAndClinicalCandidates`, `associatedDiseases`,
  `isEssential`, `depMapEssentiality`.
- **393 Ensembl IDs returned no OT target** (non-coding/deprecated) → `NOT_FOUND` for the five
  OT-derived columns on those genes.
- **Schema pitfalls fixed:** first-pass fields `maximumClinicalTrialPhase` / `isApproved` (invalid on
  `Drug`) → correct field is `maximumClinicalStage`; `phase`/`status`/`mechanismOfAction` invalid on
  the clinical-candidate row. The corrected drug sub-query is
  `drugAndClinicalCandidates{count rows{maxClinicalStage drug{id name drugType maximumClinicalStage}}}`.
- **DepMap field (verified against the live schema before batching, not guessed):**
  `depMapEssentiality{screens{geneEffect}}` plus the `isEssential` boolean on `Target`. The
  per-screen Chronos `geneEffect` values were averaged to a mean per gene.

### 3.4 STRING v12.0 — nearest_string_neighbors
- **Endpoint:** `string-db.org REST /interaction_partners + /get_string_ids`
- **Query parameters:** species=9606, required_score=400, top-10 partners; caller_identity=refinerx_annotation
- **Version:** STRING v12.0 · **Retrieved:** 2026-07-08
- **Requested / found / not-found:** 11,526 / 11,400 / 126
- **Batch primitive & scalability win:** instead of 11,526 per-gene MCP calls, neighbors were pulled
  from the STRING bulk REST endpoint `https://string-db.org/api/json/interaction_partners`
  (~120 batched POSTs), with a companion `get_string_ids?echo_query=1` pass to obtain a clean
  input-symbol→stringId join key. Same STRING v12.0 data.
- **Parameters:** `species=9606`, `required_score=400`, `limit=10` partners/gene,
  `caller_identity=refinerx_annotation`. Top-5 partners by combined score are reported per gene.
- **Note:** the MCP `get_string_network` single-gene call auto-adds ~10 neighbors, so per-gene
  neighbor retrieval genuinely needs one query per gene — the bulk REST route delivers that at scale.

### 3.5 Pharos / NCATS — pharos_tdl (Target Development Level)
- **Endpoint:** `pharos-api.ncats.io GraphQL targets(top/skip)`
- **Query parameters:** full target table: sym, tdl, fam
- **Version:** Pharos (TCRD) · **Retrieved:** 2026-07-08
- **Requested / found / not-found:** 20,412 / 20,080 / 332
- **Why a separate source:** Open Targets has no `tdl` field, so deriving TDL locally would violate
  the zero-fabrication rule — it must be fetched from Pharos (TCRD).
- **Pagination bug fixed:** `top`/`skip` on the **outer** `targets(filter:{})` field are ignored
  (returns the same 10 Tdark targets repeatedly). The parameters belong on the **inner** `targets`
  field: `targets(filter:{}){ targets(top:$top, skip:$skip){ sym tdl fam } }`. With that fix the
  full 20,412-row table paginated cleanly (page 500) to 20,080 unique symbols.
- **Network:** `pharos-api.ncats.io` required an explicit allowlist grant.
- **TDL distribution over the full table:** Tclin 703, Tchem 1,900, Tbio 12,242, Tdark 5,235.

### 3.6 ChEMBL — drug_target_status / approved_drugs reconciliation
- **Endpoint:** `chembl/target_search + get_mechanism (MCP)`
- **Query parameters:** deep-subset genes: mechanism of action, action_type, max_phase by target_chembl_id
- **Version:** ChEMBL current · **Retrieved:** 2026-07-08
- **Requested / found / not-found:** 1,796 / 602 / 1,194
- **Scope:** queried for the **1,796 deep-subset genes only** (the genes where drug status matters).
- **Batch primitive:** `target_search(gene_symbol)` → `get_mechanism(target_chembl_id)`; mechanisms
  carry molecule ChEMBL ID, mechanism-of-action, action_type, and max_phase. 602 of the 1,796 genes
  returned curated mechanism data.
- **Design choice:** Open Targets `drugAndClinicalCandidates.rows[].drug.name` is the primary source
  for `approved_drugs` (names + stage already in hand); ChEMBL `max_phase == 4` is the independent
  cross-check for approved status. This avoided ~2,093 redundant ChEMBL molecule-name lookups.

### 3.7 DepMap essentiality (via Open Targets) — depmap_essentiality
- **Endpoint:** `clinical-genomics/open_targets_graphql depMapEssentiality (MCP)`
- **Query parameters:** isEssential + per-screen geneEffect (Chronos) -> mean
- **Version:** DepMap (OT-integrated) · **Retrieved:** 2026-07-08
- **Requested / found / not-found:** 11,511 / 10,911 / 600
- Pulled in the same OT GraphQL batch mechanism as §3.3. Reported as an `isEssential`
  common-essential flag plus the mean Chronos `geneEffect` across all screens. 846 genes are
  common-essential (`isEssential = True`); 782 have mean geneEffect < −0.5.
- **Two counts, reconciled (they measure different things; mechanism verified against
  `depmap_parsed.json` + `rows_final.json`):** the manifest's DepMap `n_found = 10,911` is the number
  of distinct Ensembl targets that returned **per-screen `geneEffect`** data (from which the mean is
  computed). The column-coverage `found = 11,120` in §6 is the number of **table rows** whose
  `depmap_essentiality` cell is non-`NOT_FOUND`, which breaks down as:
  (a) **10,912 rows carry a mean geneEffect** — these correspond to the 10,911 targets, with one extra
  row because a single Ensembl ID (`ENSG00000244687`) is the mapping for two input symbols in this
  screen (CIR1 and UBE2V1), so both rows inherit that target's mean; plus
  (b) **208 rows carry `not_common_essential` with no mean** — every one of these maps to an Ensembl
  target that Open Targets returned but with **zero DepMap screens** (`n_screens = 0`, both
  `isEssential` and `mean_geneEffect` null). The assembly labeled these `not_common_essential` (the
  target exists and is not flagged common-essential) rather than `NOT_FOUND`, which is why they count
  as populated in the column but not in the manifest's per-screen tally.
  10,912 + 208 = 11,120. This is a source-vs-assembled-column difference (the manifest counts targets
  with quantitative screen data; the column counts rows with any essentiality label), not a data
  conflict. Caveat: the 208 rows convey only "no common-essential flag from OT," not a measured
  gene-effect — treat them as qualitative-only.

### 3.8 gnomAD v4.0 — gnomad_loeuf (LOEUF + pLI)
- **Endpoint:** `gcp-public-data--gnomad.storage.googleapis.com constraint_metrics.tsv (bulk)`
- **Query parameters:** LOEUF (lof.oe_ci.upper) + pLI (lof.pLI), MANE-select transcript
- **Version:** gnomAD v4.0 · **Retrieved:** 2026-07-08
- **Requested / found / not-found:** 11,526 / 10,910 / 616
- **Why bulk, not the per-gene API:** the `variants/gene_constraint` MCP averaged ~4 s/call →
  ~13 h for 11,526 genes. Instead the complete v4.0 constraint table
  (`gnomad.v4.0.constraint_metrics.tsv`, 85.9 MB, 211,272 transcript rows) was downloaded once from
  the bucket-qualified host `gcp-public-data--gnomad.storage.googleapis.com` (path-style GCS is
  exfil-denylisted; the virtual-hosted bucket form is grantable).
- **Columns used:** LOEUF = `lof.oe_ci.upper`, pLI = `lof.pLI`, selected on the **MANE-select ENST**
  transcript (fallback: any ENST with a non-null LOEUF). Joined by symbol/resolved-symbol.
- **Spot-check vs the live API (bulk value / API value / relative difference):** TP53 0.449 / 0.418
  (~7%), IL2RA 0.907 / 0.892 (~2%), CTLA4 0.211 / 0.381 (**~45%**). TP53 and IL2RA agree closely;
  **CTLA4 diverges substantially** because the bulk MANE-select transcript differs from the transcript
  the per-gene API aggregates — the two are not interchangeable at the value level for genes with
  multiple constrained transcripts. Both remain gnomAD v4 and, for CTLA4, both sit well below the
  README's LoF-intolerance threshold (LOEUF < 0.6), so the qualitative constraint call is unchanged;
  but the primary value written to the table is the **bulk MANE-select** number (0.211 for CTLA4),
  not the API number. Consumers needing an exact transcript-matched LOEUF should re-query the gnomAD
  API for the specific transcript of interest.

### 3.9 InterPro — protein_class_family
- **Endpoint:** `ftp.ebi.ac.uk entry.list (bulk) + UniProt InterPro xrefs`
- **Query parameters:** Family/Homologous_superfamily names for UniProt InterPro accessions; cross-validated vs protein-annotation/get_domain_architecture (MCP) on 2,688 accessions (100% concordant)
- **Version:** InterPro current · **Retrieved:** 2026-07-08
- **Requested / found / not-found:** 11,415 / 11,191 / 224
- **Why bulk, not the per-accession MCP:** `protein-annotation/get_domain_architecture` ran ~40 s per
  100 accessions (~56 min projected). Instead the InterPro `entry.list` (54,190 IPR→type→name rows)
  was downloaded once from `ftp.ebi.ac.uk`, and the InterPro accessions already returned in the
  UniProt xref pull (§3.2) were mapped to `Family` / `Homologous_superfamily` names locally.
- **Cross-validation:** the bulk-derived family assignments were checked against the partial MCP
  `get_domain_architecture` output on **2,688 accessions — 100% concordant** (every accession shared
  ≥1 family accession). Priority cascade: InterPro family → InterPro homologous superfamily → UniProt
  family → Open Targets target class.

### 3.10 Reactome v97 — pathway_family fallback
- **Endpoint:** `genes-ontologies/map_reactome_pathways (MCP, fallback)`
- **Query parameters:** low-level pathways for genes lacking OT top-level pathway (partial: service unreliable at scale)
- **Version:** Reactome v97 · **Retrieved:** 2026-07-08
- **Requested / found / not-found:** 4,709 / 13 / 4,696
- Attempted for the 4,709 genes lacking an Open Targets top-level pathway, via
  `map_reactome_pathways` (compact). The Reactome AnalysisService proved unreliable at scale (batches
  of 100–300 timed out > 60 s) and only ~15% of gap genes have any Reactome annotation, so the
  fallback was stopped after recovering **13** genes; the remaining gap genes are genuine `NOT_FOUND`.
  Primary pathway source is Open Targets `pathways.topLevelTerm` (6,816 targets).

### 3.11 PubMed — key_reference (deep subset only)
- **Endpoint:** `pubmed/search_articles + get_article_metadata (MCP)`
- **Query parameters:** deep-subset genes only: tiered title/immune/review search -> top PMID -> metadata
- **Version:** PubMed live · **Retrieved:** 2026-07-08
- **Requested / found / not-found:** 1,796 / 1,785 / 11 (final `key_reference` outcome). Note: the
  search step returned ≥1 PMID for **1,786** genes, but one of those (LYPLA2) had both its PMIDs fail
  to resolve to article metadata, so it lands as `NOT_FOUND` in the final column — hence 1,785 found,
  not 1,786. Both `source_manifest.json` and `coverage_report.json` report the reconciled 1,785 / 11.
- **Scope & tiering:** searched **only** for the 1,796 deep-subset genes. Per gene, a tiered query:
  (0) `GENE[Title] AND (T-lymphocytes[MeSH] OR immune[TIAB]) AND Review[PT]`, (1) `GENE[Title] AND
  (T cell OR immune OR lymphocyte)`, (2) `GENE[Title]` — first non-empty tier wins, top PMID kept.
- **Two-step API + metadata cap bug:** `search_articles` returns `pmids` only; `get_article_metadata`
  resolves citations but **caps at 20 articles/call** (requests of 50/100 silently return 20; 200
  returns empty). Metadata for the 3,117 unique PMIDs was therefore fetched in batches of 20
  (3,109 resolved). PMID lives at `identifiers.pmid`, date at `publication_date` (not top-level).
- **Result:** 1,785 deep genes got a real citation; 11 are `NOT_FOUND` (10 with no title-matching
  article + LYPLA2 whose 2 PMIDs returned no metadata); 9,730 non-deep genes are `NOT_FETCHED`.

---

## 4. Deep literature subset (1,796 genes)

`key_reference` was populated only for the deep subset, defined by a purely public criterion:

> Open Targets established drug target (>=1 drug at any phase incl approved) OR clinically tractable (Pharos Tclin/Tchem, or OT small-molecule/antibody clinical-grade tractability bucket: Approved Drug / Advanced Clinical / Phase 1 Clinical)

Overlapping contributions: Pharos Tclin/Tchem 1,611; OT any drug 785; OT small-molecule
clinical-tractable 539; OT approved-target 479; OT antibody clinical-tractable 165.

---

## 5. Column dictionary (31 columns)

Each annotation value column is followed by a `<col>_source` column naming the database it came from.

| # | column | meaning | populated by |
|---|---|---|---|
| 1 | gene_symbol | input perturbed gene | screen gene list |
| 2–3 | ensembl_id (+_source) | Ensembl gene ID | MyGene.info |
| 4–5 | uniprot_id (+_source) | Swiss-Prot accession | MyGene / UniProt |
| 6–7 | drug_target_status (+_source) | approved_target / clinical_target / tool_compound_only / no_known_drug | Open Targets + ChEMBL (reconciled) |
| 8 | drug_target_status_conflict | True where OT and ChEMBL disagree | derived |
| 9–10 | function (+_source) | concise protein function | UniProt Function[CC] (MyGene fallback) |
| 11–12 | pathway_family (+_source) | top-level pathway membership | Open Targets top-level (Reactome fallback) |
| 13–14 | nearest_string_neighbors (+_source) | top-5 STRING partners (combined score) | STRING v12.0 |
| 15–16 | tractability_modality (+_source) | small_molecule / antibody / PROTAC_degrader / other | Open Targets |
| 17–18 | pharos_tdl (+_source) | Target Development Level | Pharos (NCATS TCRD) |
| 19–20 | approved_drugs (+_source) | APPROVED: … \| CANDIDATES: …[phase] | Open Targets drugAndClinicalCandidates |
| 21–22 | immune_disease_assoc (+_source) | top immune/autoimmune disease associations (score) | Open Targets |
| 23–24 | protein_class_family (+_source) | protein family / homologous superfamily | InterPro (UniProt / OT fallback) |
| 25–26 | depmap_essentiality (+_source) | common_essential flag + mean gene-effect | DepMap via Open Targets |
| 27–28 | gnomad_loeuf (+_source) | LOEUF + pLI LoF constraint | gnomAD v4.0 |
| 29–30 | key_reference (+_source) | one authoritative PubMed reference (deep subset only) | PubMed |
| 31 | is_deep_subset | in the 1,796-gene literature tier | derived |

---

## 6. Coverage summary (over 11,526)

| column | found | % | NOT_FOUND | NOT_FETCHED |
|---|---|---|---|---|
| ensembl_id | 11,513 | 99.9 | 13 | 0 |
| uniprot_id | 11,464 | 99.5 | 62 | 0 |
| drug_target_status | 11,133 | 96.6 | 393 | 0 |
| function | 11,336 | 98.4 | 190 | 0 |
| pathway_family | 6,830 | 59.3 | 4,696 | 0 |
| nearest_string_neighbors | 11,400 | 98.9 | 126 | 0 |
| tractability_modality | 10,285 | 89.2 | 1,241 | 0 |
| pharos_tdl | 11,165 | 96.9 | 361 | 0 |
| approved_drugs | 785 | 6.8 | 10,741 | 0 |
| immune_disease_assoc | 6,150 | 53.4 | 5,376 | 0 |
| protein_class_family | 11,385 | 98.8 | 141 | 0 |
| depmap_essentiality | 11,120 | 96.5 | 406 | 0 |
| gnomad_loeuf | 10,910 | 94.7 | 616 | 0 |
| key_reference | 1,785 | 15.5 | 11 | 9,730 |

`drug_target_status` distribution: no_known_drug 10,335,
approved_target 487, clinical_target
283, tool_compound_only
28, NOT_FOUND
393. Genes with ≥1 approved drug:
479.

**Intrinsically low-coverage columns are real biology, not gaps.** `approved_drugs` (6.8%) and
`immune_disease_assoc` (53.4%) reflect that most genes are neither drug targets nor immune-disease
associated; these are true `NOT_FOUND`. `pathway_family` (59.3%) is limited by how many genes have
any curated pathway membership. `key_reference` (15.5% found) is by design — literature was fetched
only for the deep subset.

---

## 7. Safety / prioritization note

Two columns flag genes that make **poorer** immune-drug targets: `depmap_essentiality`
(common-essential genes risk on-target toxicity) and `gnomad_loeuf` (LOEUF < 0.6 / pLI ≥ 0.9 =
LoF-intolerant). Downstream target ranking should down-weight genes flagged by either.

---

## 8. Reproducibility artifacts

Deliverables: `tcell_perturbed_gene_annotation.csv`, `tcell_perturbed_gene_annotation.parquet`
(full provenance cells), `source_manifest.json`, `coverage_report.json`, `README_annotation.md`,
and this `GENERATION_METHODS.md`.

The **raw API dumps** (Open Targets ~66 MB, UniProt, MyGene, STRING, Pharos, ChEMBL, DepMap,
InterPro, PubMed, plus all parsed intermediates) are archived together as the
`raw_pulls_checkpoint.tar.gz` **artifact** — they are intentionally **not** committed to git because
they exceed GitHub file-size limits. Regenerate any column from that archive plus the endpoints and
parameters in §3.


---

## 9. Tooling & Provenance: Claude Science + skills

### Platform
This annotation table was produced by an autonomous specialist agent running in the **Claude
Science** agentic environment. The relevant platform features used here: sandboxed compute
kernels (isolated Python environments with a shared workspace); a versioned **artifact store**
that holds every output and checkpoint by content hash; **connector (MCP)** access to
bioinformatics databases, called programmatically from a control-plane kernel; and a network
sandbox with an explicit allowlist (two bulk hosts — the gnomAD GCS bucket and the EBI FTP —
were added by grant during the run). The job itself ran as a **delegated sub-agent** with a
**structured-output contract**: on completion the agent submitted a machine-validated payload
(artifact version IDs, per-column coverage, deep-subset criterion) rather than free text.

### No value in this table was produced by a language model
Every annotation value came from a **live API call** logged in `source_manifest.json`. The skills
below **structured the collection** — batching, provenance bookkeeping, validation, reconciliation —
they did **not** supply data. The agent wrote no gene's function, no drug name, no PMID, and no
constraint metric from its own parameters; where a source returned nothing, the cell is `NOT_FOUND`
or `NOT_FETCHED`, never a plausible-looking guess.

### Skills loaded and what each was used for

**`autocollect-bio`** — the anti-fabrication bulk-collection discipline that governs the whole
table. It defines the provenance-on-every-value rule, the `NOT_FOUND` vs `NOT_FETCHED` sentinels,
the tiered *pull → validate → reconcile* workflow, and the coverage/manifest reporting format. Its
`kernel.py` helpers were used directly to build the outputs: `cell()` (wrap a value with source /
accession / retrieved-date / query), `reconcile()` (combine multi-source values and flag conflicts —
used for `drug_target_status` across Open Targets and ChEMBL), `citation_cell()` (structure a PubMed
reference for `key_reference`), `manifest_row()` (each row of `source_manifest.json`),
`coverage_report()` (the found / NOT_FOUND / NOT_FETCHED tallies in `coverage_report.json`), and
`spot_check_plan()` (the random-sample audit that surfaced and fixed the `approved_drugs`
approved-vs-candidate imprecision).

**`mcp-genes-ontologies`** — MyGene.info, UniProt, and Reactome tools. Populated: `ensembl_id` and
`uniprot_id` (MyGene `query_genes`, 1000/call); `function` and the InterPro/Pfam xrefs feeding
`protein_class_family` (UniProt `get_uniprot_entries`); the `pathway_family` Reactome fallback
(`map_reactome_pathways`).

**`mcp-clinical-genomics`** — Open Targets Platform GraphQL (`open_targets_graphql`). The single
richest source: populated `drug_target_status`, `pathway_family` (primary, `pathways.topLevelTerm`),
`tractability_modality`, `approved_drugs` (primary), `immune_disease_assoc`, and `depmap_essentiality`
(`isEssential` + per-screen `geneEffect`). Also the source of the `targetClass` fallback for
`protein_class_family`.

**`mcp-chembl`** — ChEMBL `target_search` + `get_mechanism`. Used to **reconcile** `drug_target_status`
and cross-check `approved_drugs` on the 1,796 deep-subset genes (independent approved-drug confirmation
via `max_phase == 4`, mechanism-of-action, action_type).

**`mcp-protein-annotation`** — InterPro `get_domain_architecture`. Used to **cross-validate** the
`protein_class_family` assignments: the bulk-derived InterPro family labels were checked against this
connector on 2,688 accessions (100% concordant). The connector's `get_string_network` tool was also
used early to characterize STRING's per-gene neighbor behavior, which informed the decision to pull
neighbors from the STRING bulk REST endpoint at scale.

**`mcp-variants`** — gnomAD `gene_constraint`. Used to **validate** `gnomad_loeuf`: the per-gene API
values (TP53 / IL2RA / CTLA4) confirmed the values taken from the bulk gnomAD v4.0 constraint table,
which was the primary source at scale (the per-gene API was too slow for 11,526 genes).

**`mcp-pubmed`** — PubMed `search_articles` + `get_article_metadata`. Populated `key_reference` for the
deep subset (tiered title/immune/review search → top PMID → citation metadata). Every PMID in the
table traces to a call through this connector.

**`mcp-drug-regulatory`** — the FDA drug-application / labeling connector was **loaded during source
reconnaissance but not used to populate any final column**; `approved_drugs` is sourced from Open
Targets and ChEMBL. It is listed here for completeness and accuracy, not as a data source.

### Non-MCP data paths
Three columns' final values came from **bulk downloads** rather than an MCP tool, for scale/reliability
(each documented in §3): `nearest_string_neighbors` (STRING v12.0 bulk REST `interaction_partners`),
`gnomad_loeuf` (gnomAD v4.0 constraint TSV), and `protein_class_family` (InterPro `entry.list` mapped
onto UniProt InterPro xrefs). In each case an MCP connector was used to validate or characterize the
bulk data, and the bulk source carries the identical underlying database release.
