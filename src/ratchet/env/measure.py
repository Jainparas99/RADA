"""measure(): explain + result hashes + data checksums for one workload (SPEC §5.3)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from bson import json_util
from bson.json_util import CANONICAL_JSON_OPTIONS
from pymongo.database import Database

from ratchet.contracts import ExplainSummary, Measurement, QuerySpec, Workload
from ratchet.env.explain import run_explain, summarize
from ratchet.env.scratch import ScratchEnv, data_checksum

# A rewritten query that returns more than this is cut off; the marker in the hash makes
# sure it can never equal a baseline hash (baseline results are <= 500 docs, validate.py).
MAX_RESULT_DOCS = 20_000
MISSING_HASH = "missing"


class MeasureError(RuntimeError):
    def __init__(self, query_id: str, message: str):
        super().__init__(f"{query_id}: {message}")
        self.query_id = query_id


def canonical(doc: Any) -> str:
    """Canonical Extended JSON with sorted keys: field order inside a document does not
    matter, value types (int32 vs int64 vs double) do."""
    return json_util.dumps(doc, sort_keys=True, json_options=CANONICAL_JSON_OPTIONS)


def result_hash(docs: Iterable[Any]) -> str:
    """Order-sensitive SHA-256 over the canonical JSON of each result document."""
    h = hashlib.sha256()
    n = 0
    for d in docs:
        if n >= MAX_RESULT_DOCS:
            h.update(b"\x00TRUNCATED")
            break
        h.update(canonical(d).encode())
        h.update(b"\n")
        n += 1
    return f"sha256:{n}:{h.hexdigest()}"


def run_query(db: Database, q: QuerySpec) -> Iterable[dict]:
    coll = db[q.collection]
    if q.kind == "find":
        cur = coll.find(q.filter or {}, q.projection)
        if q.sort:
            cur = cur.sort([(k, v) for k, v in q.sort])
        if q.limit is not None:
            cur = cur.limit(q.limit)
        return cur
    return coll.aggregate(q.pipeline or [])


def explain_query(env: ScratchEnv, q: QuerySpec) -> ExplainSummary:
    return summarize(q.id, run_explain(env.db, q))


def measure(
    env: ScratchEnv, workload: Workload, queries: list[QuerySpec] | None = None
) -> Measurement:
    """Measure ``queries`` (default: the workload's own) on the scratch copy.

    ``queries`` is how rewritten queries get measured; ids must be the workload's query
    ids. ``workload_cost`` always weights by the *workload's* frequencies, never by the
    frequency carried on a (possibly rewritten) query.
    """
    from ratchet.scoring import query_cost  # local import: scoring imports env

    qs = queries if queries is not None else workload.queries
    freqs = {q.id: q.freq for q in workload.queries}
    per_query: dict[str, ExplainSummary] = {}
    hashes: dict[str, str] = {}
    for q in qs:
        if q.id not in freqs:
            raise MeasureError(q.id, "query id not in workload")
        try:
            per_query[q.id] = explain_query(env, q)
            hashes[q.id] = result_hash(run_query(env.db, q))
        except Exception as e:  # surfaced as APPLY_ERROR / trace by callers
            raise MeasureError(q.id, f"{type(e).__name__}: {e}") from e
    cost = sum(freqs[qid] * query_cost(s) for qid, s in per_query.items())
    checksums = {c: data_checksum(env.db, c, env.checksum_mode) for c in workload.collections}
    indexes = {c: sorted(s["name"] for s in env.index_specs(c)) for c in workload.collections}
    return Measurement(
        workload_id=workload.id,
        per_query=per_query,
        workload_cost=float(cost),
        result_hashes=hashes,
        data_checksums=checksums,
        indexes=indexes,
    )
