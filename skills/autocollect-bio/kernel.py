"""autocollect-bio kernel helpers: provenance-tracked data-collection primitives.

Every retrieved value goes through cell(); every source through manifest_row().
No value enters the output table without provenance or an explicit sentinel.
"""

NOT_FOUND = "NOT_FOUND"        # looked, source had no answer
NOT_FETCHED = "NOT_FETCHED"    # deliberately not looked (tiered depth)
AMBIGUOUS = "AMBIGUOUS"        # multiple non-reconcilable answers
NOT_APPLICABLE = "NOT_APPLICABLE"
SENTINELS = ("NOT_FOUND", "NOT_FETCHED", "AMBIGUOUS", "NOT_APPLICABLE")

def cell(value=None, source=None, accession=None, retrieved=None, query=None):
    """Build a provenance-tagged value cell.

    value=None -> NOT_FOUND. Pass a SENTINEL string to record why it's empty.
    source is REQUIRED for any real value (raises if a real value has no source).
    """
    if value is None:
        value = NOT_FOUND
    is_sentinel = isinstance(value, str) and value in SENTINELS
    if not is_sentinel and not source:
        raise ValueError("cell() with a real value requires source= (provenance is mandatory)")
    return {"value": value, "source": source, "accession": accession,
            "retrieved": retrieved, "query": query}

def is_empty(c):
    """True if a cell holds a sentinel (no real value)."""
    v = c.get("value") if isinstance(c, dict) else c
    return isinstance(v, str) and v in SENTINELS

def reconcile(records, rule="prefer_first", prefer_order=None):
    """Reconcile multiple provenance cells for the SAME field across sources.

    Keeps ALL non-empty values; flags conflict when they disagree.
    rule: 'prefer_first' | 'prefer_order' (needs prefer_order list of sources)
          | 'any' (any non-empty wins, conflict if disagree).
    Returns a cell dict augmented with {values:[...], conflict:bool, rule:...}.
    """
    real = [r for r in records if not is_empty(r)]
    if not real:
        return {"value": NOT_FOUND, "source": None, "values": [], "conflict": False, "rule": rule}
    distinct = []
    for r in real:
        if r["value"] not in [d["value"] for d in distinct]:
            distinct.append(r)
    conflict = len(distinct) > 1
    chosen = real[0]
    if rule == "prefer_order" and prefer_order:
        by_src = {r["source"]: r for r in real}
        for s in prefer_order:
            if s in by_src:
                chosen = by_src[s]; break
    out = dict(chosen)
    out["values"] = [{"value": r["value"], "source": r["source"]} for r in real]
    out["conflict"] = conflict
    out["rule"] = rule
    return out

def citation_cell(pmid=None, title=None, source=None, retrieved=None, doi=None):
    """Build a reference cell. An identifier is recorded ONLY if it came from a
    real search this run: pass what the API returned. No pmid -> NOT_FOUND.
    Never construct a pmid/doi from memory — this helper cannot invent one for you."""
    if not pmid and not doi:
        return {"value": NOT_FOUND, "source": source, "retrieved": retrieved}
    if not source:
        raise ValueError("citation_cell with an identifier requires source= (which API returned it)")
    return {"value": {"pmid": pmid, "doi": doi, "title": title}, "source": source,
            "accession": pmid or doi, "retrieved": retrieved}

def manifest_row(source, endpoint, query=None, version=None, retrieved=None,
                 n_requested=None, n_found=None):
    """One reproducibility record for a source pull."""
    n_missing = None
    if n_requested is not None and n_found is not None:
        n_missing = n_requested - n_found
    return {"source": source, "endpoint": endpoint, "query": query,
            "version": version, "retrieved": retrieved, "n_requested": n_requested,
            "n_found": n_found, "n_not_found": n_missing}

def coverage_report(rows, columns, total=None):
    """Per-column found / NOT_FOUND / NOT_FETCHED counts over the FULL set.

    rows: list of dict-of-cells (one per entity). total: frozen entity count
    (defaults to len(rows); pass explicitly if some entities were dropped so the
    denominator stays honest)."""
    if total is None:
        total = len(rows)
    rep = {}
    for col in columns:
        found = fetched_empty = not_fetched = 0
        for r in rows:
            c = r.get(col)
            v = c.get("value") if isinstance(c, dict) else c
            if v == NOT_FETCHED:
                not_fetched += 1
            elif isinstance(v, str) and v in SENTINELS:
                fetched_empty += 1
            else:
                found += 1
        rep[col] = {"found": found, "empty": fetched_empty, "not_fetched": not_fetched,
                    "total": total, "coverage": round(found / total, 4) if total else 0.0}
    return rep

def spot_check_plan(rows, n=20, seed=0):
    """Pick n random (row_index, column) cells to re-verify against live sources.
    Returns a list of (row_index, column, current_value) to check by hand/API."""
    import random
    rng = random.Random(seed)
    out = []
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    for i in idxs[:n]:
        cols = [k for k, c in rows[i].items() if not is_empty(c)]
        if cols:
            col = rng.choice(cols)
            out.append((i, col, rows[i][col]))
    return out
