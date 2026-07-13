---
name: autocollect-bio
description: Discipline for bulk biological data/annotation collection at scale (hundreds to tens of thousands of entities) with zero fabrication. Use when building a large annotation table, gene/protein/compound dossier set, or literature-evidence matrix from public databases and APIs (Open Targets, UniProt, Reactome, STRING, ChEMBL, PubMed, etc.). Ports autoresearch-bio's anti-fabrication invariants — provenance on every value, explicit NOT_FOUND instead of guessed values, real-citations-only, tiered pull/validate/reconcile gates, a reproducible source manifest — to data collection rather than experiment search.
---

# Autocollect-Bio

A **bio-first, agent-agnostic** discipline for turning a list of biological
entities (genes, proteins, compounds, variants) into a large, trustworthy,
reproducible annotation table by pulling from public databases and literature —
**without fabricating a single value**.

It is the data-collection sibling of `autoresearch-bio`. That skill governs
autonomous *experiment* loops; this one governs autonomous *retrieval* loops.
The shared invariant is identical and is the whole point:

> **A model prior never stands in for a retrieved value.** Every cell in the
> output is either a value that came back from a named source on a recorded
> date, or the explicit token `NOT_FOUND`. There is no third option.

Fabrication in a 10,000-row table is invisible by construction — no reader can
check 10,000 rows — so the discipline has to make fabrication *structurally
impossible*, not merely discouraged.

---

## When to use

- Building an annotation table across hundreds–tens of thousands of entities.
- Assembling per-entity dossiers (function, pathway, druggability, network,
  disease, references) from public sources.
- Any "make a table with a column for X, Y, Z" request where X/Y/Z are
  retrievable facts, not computed results.

## When NOT to use

- Computed results from the user's own data (that is analysis, not collection).
- A handful of entities you can annotate and verify by hand.
- Anything requiring wet-lab protocols or clinical recommendations.

---

## Core invariants (non-negotiable)

1. **Protected source list.** Name every data source up front (DB, endpoint,
   release/version). The set of sources is fixed before the pull begins; adding
   one mid-run is an explicit amendment, logged in the manifest.
2. **Provenance on every value.** Each retrieved value carries `{value, source,
   accession, retrieved_date, query}`. A bare value with no provenance is a bug.
3. **NOT_FOUND, never guessed.** If a source has no answer, the cell is
   `NOT_FOUND` (optionally `AMBIGUOUS` / `NOT_APPLICABLE`). The model's belief
   about the answer is irrelevant and must never be written.
4. **Real citations only.** A PMID / DOI / accession is written ONLY if it was
   returned by a real API/search call in this run. Never construct an
   identifier from memory; never "recall" a paper. If the search returned
   nothing, the reference cell is `NOT_FOUND`.
5. **Tiered gates.** Tier 1 = pull (id-map → batch fetch). Tier 2 = validate
   (schema, id round-trip, coverage floor, type checks). Tier 3 = reconcile
   (cross-source conflicts resolved with both values kept + a rule). No row is
   "done" until it has passed Tier 2; conflicts are surfaced, not hidden.
6. **Conflicts are data.** When two sources disagree, record BOTH with
   provenance and a `conflict=True` flag plus the resolution rule applied.
   Silent pick-one is forbidden.
7. **Checkpoint + resume.** Long pulls checkpoint partial progress (every N
   entities) so a crash resumes, never restarts. Re-runs are idempotent keyed
   on entity id.
8. **Reproducible manifest.** One `source_manifest` records, per source: the
   endpoint, the exact query/params, the release/version, the retrieval date,
   #requested, #found, #NOT_FOUND. The table is reproducible from the manifest
   alone.
9. **Coverage report, not cherry-picks.** Report per-column found / NOT_FOUND
   counts over the FULL entity set. Never present only the rows that resolved.
10. **Rate-limit politeness.** Prefer bulk/batch endpoints; respect shared
    rate limits; back off on 429. A shared server is shared with everyone.

---

## Golden path

1. **Freeze inputs.** Load the canonical entity list; dedupe; count it. This
   count is the denominator for every coverage number afterward.
2. **Declare the schema + sources.** For each output column: which source,
   which endpoint/method, which id space, batch or per-entity. Write it down
   before pulling. Map columns → sources explicitly.
3. **Id-map first (Tier 1a).** Resolve every entity to the id spaces each
   source needs (symbol→Ensembl→UniProt→ChEMBL target, etc.) via a batch
   mapper. Round-trip a sample to catch silent mis-maps. Unmapped entities are
   `NOT_FOUND` for that source, carried forward — never dropped.
4. **Batch-pull each source (Tier 1b).** Use the largest batch the API allows.
   Store each value as a provenance cell. Checkpoint every N.
5. **Validate (Tier 2).** Schema present, types right, coverage ≥ a stated
   floor per column; id round-trip holds; spot-check a random sample of cells
   against the live source.
6. **Reconcile (Tier 3).** Where columns come from multiple sources (e.g. drug
   status from Open Targets AND ChEMBL), compare; keep both; flag conflicts;
   apply and record a resolution rule.
7. **Coverage report + manifest.** Emit per-column coverage over the full set
   and the source manifest.
8. **Tier the depth, not the honesty.** It is fine to pull cheap DB columns for
   ALL entities and reserve expensive literature synthesis for a subset — but
   the subset criterion is stated, and the un-deepened rows say `NOT_FETCHED`
   for those columns (a truthful "we did not look", distinct from a checked
   `NOT_FOUND`).

---

## Anti-fabrication checklist (run before declaring done)

- [ ] Every value cell has a source + retrieved_date, or is NOT_FOUND.
- [ ] No identifier (PMID/DOI/accession) appears that was not returned by a
      logged API call this run.
- [ ] Coverage numbers are over the full frozen entity count, not the resolved
      subset.
- [ ] Conflicts between sources are flagged, not silently resolved.
- [ ] The manifest lets someone re-pull and get the same table.
- [ ] Spot-check sample (≥20 random cells) re-verified against live sources.
- [ ] NOT_FETCHED (not looked) is distinct from NOT_FOUND (looked, absent).

## Kernel helpers

Loading this skill defines (see `kernel.py`): `NOT_FOUND`, `NOT_FETCHED`,
`AMBIGUOUS`, `cell(...)`, `reconcile(...)`, `manifest_row(...)`,
`citation_cell(...)`, `coverage_report(...)`, `spot_check_plan(...)`.
Use `cell()` for every retrieved value; never write a raw value into the table.
