"""Run explain("executionStats") and normalise it into ExplainSummary (SPEC §5.3).

Shapes handled (all verified against real mongod 8.x output, see tests/env/fixtures/):

* find, classic engine: ``queryPlanner.winningPlan`` is a stage tree.
* find / aggregate, SBE (``explainVersion: "2"``): ``winningPlan.queryPlan`` is the
  stage tree (classic stage names such as EQ_LOOKUP, GROUP, IXSCAN); the SBE
  ``executionStages`` tree is ignored.
* aggregate fully pushed down to the query layer: same shape as a find
  (``queryPlanner.optimizedPipeline: true``), no ``stages`` key.
* aggregate with a pipeline: ``stages[0].$cursor`` holds the query-layer explain;
  later entries are pipeline stages (``$lookup`` carries its own
  ``totalDocsExamined`` / ``totalKeysExamined`` / ``collectionScans`` / ``indexesUsed``).

Counting rule for $lookup work, so that each examined doc/key is counted once:
classic ``$lookup`` stages report their foreign-side work per stage, so it goes into
``lookup_examined``. An SBE ``EQ_LOOKUP`` already folds the foreign side into
``executionStats.totalDocsExamined/totalKeysExamined``, so its ``lookup_examined`` is 0.
"""

from __future__ import annotations

from typing import Any

from bson import json_util
from pymongo.database import Database

from ratchet.contracts import ExplainSummary, QuerySpec


class ExplainParseError(ValueError):
    pass


# Stage names that read index keys and carry an index name.
_INDEX_STAGES = {
    "IXSCAN",
    "COUNT_SCAN",
    "DISTINCT_SCAN",
    "EXPRESS_IXSCAN",
    "EXPRESS_CLUSTERED_IXSCAN",
    "TEXT_MATCH",
    "TEXT_OR",
    "GEO_NEAR_2D",
    "GEO_NEAR_2DSPHERE",
    "EQ_LOOKUP",
}
_ID_STAGES = {"IDHACK", "EXPRESS_IXSCAN", "EXPRESS_CLUSTERED_IXSCAN"}
_SORT_STAGES = {"SORT"}  # SORT_MERGE merges pre-sorted index streams: not an in-memory sort
_PLAN_KEEP = (
    "stage",
    "indexName",
    "keyPattern",
    "indexBounds",
    "isMultiKey",
    "direction",
    "filter",
    "sortPattern",
    "limitAmount",
    "transformBy",
    "foreignCollection",
    "localField",
    "foreignField",
    "asField",
    "strategy",
    "indexKeyPattern",
    "nReturned",
    "keysExamined",
    "docsExamined",
)
_FULL_BOUNDS = {"[MinKey, MaxKey]", "[MaxKey, MinKey]"}


# --------------------------------------------------------------------------- running


def to_bson(obj: Any) -> Any:
    """Query values as BSON types. Workloads and planner output are plain JSON, where an
    ObjectId or a date is written in Extended JSON (``{"$oid": ...}``, ``{"$date": ...}``);
    the server needs the real types. Already-converted values pass through unchanged."""
    return json_util.loads(json_util.dumps(obj))


def explain_command(q: QuerySpec) -> dict[str, Any]:
    """The inner command for ``explain`` (without the verbosity wrapper)."""
    if q.kind == "find":
        cmd: dict[str, Any] = {"find": q.collection, "filter": to_bson(q.filter or {})}
        if q.projection is not None:
            cmd["projection"] = q.projection
        if q.sort:
            cmd["sort"] = {k: v for k, v in q.sort}
        if q.limit is not None:
            cmd["limit"] = q.limit
        return cmd
    if q.pipeline is None:
        raise ExplainParseError(f"aggregate query {q.id} has no pipeline")
    return {"aggregate": q.collection, "pipeline": to_bson(q.pipeline), "cursor": {}}


def run_explain(db: Database, q: QuerySpec) -> dict[str, Any]:
    return db.command({"explain": explain_command(q), "verbosity": "executionStats"})


# --------------------------------------------------------------------------- plan trees


def _children(node: dict) -> list[dict]:
    out: list[dict] = []
    if isinstance(node.get("inputStage"), dict):
        out.append(node["inputStage"])
    for key in ("inputStages", "thenStage", "elseStage"):
        v = node.get(key)
        if isinstance(v, list):
            out.extend(x for x in v if isinstance(x, dict))
        elif isinstance(v, dict):
            out.append(v)
    return out


def _walk_post(node: dict) -> list[dict]:
    """Leaf-first (execution order) list of plan nodes."""
    out: list[dict] = []
    for ch in _children(node):
        out.extend(_walk_post(ch))
    out.append(node)
    return out


def _trim_plan(node: dict) -> dict:
    t = {k: node[k] for k in _PLAN_KEEP if k in node}
    kids = _children(node)
    if len(kids) == 1:
        t["inputStage"] = _trim_plan(kids[0])
    elif kids:
        t["inputStages"] = [_trim_plan(k) for k in kids]
    return t


def _winning_plan(query_planner: dict) -> dict:
    wp = query_planner.get("winningPlan")
    if not isinstance(wp, dict):
        raise ExplainParseError("queryPlanner.winningPlan missing")
    # SBE (explainVersion 2) nests the classic-named tree under queryPlan.
    return wp["queryPlan"] if isinstance(wp.get("queryPlan"), dict) else wp


def _is_full_range(bounds: Any) -> bool:
    if not isinstance(bounds, dict) or not bounds:
        return False
    return all(isinstance(v, list) and set(v) <= _FULL_BOUNDS for v in bounds.values())


def _split(raw: dict) -> tuple[dict | None, list[dict]]:
    """Return (query-layer explain, later pipeline stages)."""
    if "stages" in raw:
        stages = raw["stages"]
        if not isinstance(stages, list) or not stages:
            raise ExplainParseError("empty stages")
        first = stages[0]
        if isinstance(first, dict) and "$cursor" in first:
            return first["$cursor"], stages[1:]
        return None, stages
    if "queryPlanner" in raw:
        return raw, []
    if "shards" in raw:
        raise ExplainParseError("sharded explain output is not supported")
    raise ExplainParseError(f"unrecognised explain shape: keys={sorted(raw)[:8]}")


def _stage_name(stage: dict) -> str:
    for k in stage:
        if k.startswith("$"):
            return k
    raise ExplainParseError(f"pipeline stage without $name: {sorted(stage)}")


def _int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- summarise


def summarize(query_id: str, raw: dict) -> ExplainSummary:
    """Normalise one explain("executionStats") output into ExplainSummary."""
    query_layer, pipe = _split(raw)
    engine = "sbe" if str(raw.get("explainVersion", "1")) == "2" else "classic"

    plan_nodes: list[dict] = []
    trimmed_plan: dict | None = None
    es: dict = {}
    if query_layer is not None:
        qp = query_layer.get("queryPlanner") or {}
        root = _winning_plan(qp)
        plan_nodes = _walk_post(root)
        trimmed_plan = _trim_plan(root)
        es = query_layer.get("executionStats") or {}
        if not es:
            raise ExplainParseError("executionStats missing (explain run without executionStats?)")

    stages = [n.get("stage", "?") for n in plan_nodes]
    indexes_used: list[str] = []
    scans: list[dict] = []
    lookups: list[dict] = []
    for n in plan_nodes:
        st = n.get("stage")
        if st in _ID_STAGES:
            name = n.get("indexName") or "_id_"
        else:
            name = n.get("indexName")
        if st in _INDEX_STAGES or st == "IDHACK":
            if name and st != "EQ_LOOKUP" and name not in indexes_used:
                indexes_used.append(name)
        if st in ("IXSCAN", "COUNT_SCAN", "DISTINCT_SCAN", "COLLSCAN"):
            scans.append(
                {
                    "stage": st,
                    "index": n.get("indexName"),
                    "key_pattern": n.get("keyPattern"),
                    "full_range": st == "COLLSCAN" or _is_full_range(n.get("indexBounds")),
                }
            )
        if st == "EQ_LOOKUP":
            strategy = n.get("strategy")
            fc = str(n.get("foreignCollection", ""))
            lookups.append(
                {
                    "source": "sbe",
                    "from": fc.split(".", 1)[1] if "." in fc else fc,
                    "local_field": n.get("localField"),
                    "foreign_field": n.get("foreignField"),
                    "as": n.get("asField"),
                    "strategy": strategy,
                    "indexed": strategy == "IndexedLoopJoin",
                    "indexes_used": [n["indexName"]] if n.get("indexName") else [],
                    "collection_scans": None,
                    "docs_examined": None,  # folded into executionStats totals
                    "keys_examined": None,
                    "pipeline_form": False,
                }
            )

    has_sort = any(s in _SORT_STAGES for s in stages)
    keys = _int(es.get("totalKeysExamined"))
    docs = _int(es.get("totalDocsExamined"))
    n_returned = _int(es.get("nReturned"))
    exec_ms = _int(es.get("executionTimeMillis"))
    lookup_examined = 0

    # Docs an in-memory sort had to buffer: a query-layer SORT is charged
    # totalDocsExamined (SPEC §5.6); a pipeline $sort the docs that flow into it.
    sort_docs = docs if has_sort else 0
    prev_n = n_returned
    pipeline_summary: list[dict] = []
    for st in pipe:
        name = _stage_name(st)
        stages.append(name)
        spec = st[name]
        entry: dict[str, Any] = {"name": name, "spec": spec, "nReturned": st.get("nReturned")}
        pipeline_summary.append(entry)
        if name == "$sort":
            has_sort = True
            sort_docs += prev_n  # docs flowing out of the previous stage
        if "nReturned" in st:
            n_returned = _int(st["nReturned"])
        prev_n = n_returned
        exec_ms = max(exec_ms, _int(st.get("executionTimeMillisEstimate")))
        if name in ("$lookup", "$graphLookup", "$unionWith"):
            ld, lk = _int(st.get("totalDocsExamined")), _int(st.get("totalKeysExamined"))
            lookup_examined += ld + lk
            if name == "$lookup" and isinstance(spec, dict):
                used = [str(x) for x in st.get("indexesUsed", [])]
                scans_n = st.get("collectionScans")
                lookups.append(
                    {
                        "source": "classic",
                        "from": spec.get("from"),
                        "local_field": spec.get("localField"),
                        "foreign_field": spec.get("foreignField"),
                        "as": spec.get("as"),
                        "strategy": None,
                        "indexed": bool(used) and not _int(scans_n),
                        "indexes_used": used,
                        "collection_scans": None if scans_n is None else _int(scans_n),
                        "docs_examined": ld,
                        "keys_examined": lk,
                        "pipeline_form": "localField" not in spec,
                    }
                )

    raw_trim: dict[str, Any] = {
        "explainVersion": raw.get("explainVersion"),
        "winningPlan": trimmed_plan,
        "executionStats": {
            "nReturned": es.get("nReturned"),
            "executionTimeMillis": es.get("executionTimeMillis"),
            "totalKeysExamined": es.get("totalKeysExamined"),
            "totalDocsExamined": es.get("totalDocsExamined"),
        },
        "pipeline": [{"name": p["name"], "nReturned": p["nReturned"]} for p in pipeline_summary],
        "derived": {
            "engine": engine,
            "pushed_down": not pipe,
            "sort_docs": sort_docs,
            "plan_stages": [n.get("stage") for n in plan_nodes],
            "scans": scans,
            "lookups": lookups,
            "pipeline": pipeline_summary,
        },
    }
    return ExplainSummary(
        query_id=query_id,
        stages=stages,
        n_returned=n_returned,
        keys_examined=keys,
        docs_examined=docs,
        exec_ms=exec_ms,
        has_in_memory_sort=has_sort,
        indexes_used=indexes_used,
        lookup_examined=lookup_examined,
        raw=raw_trim,
    )


def derived(s: ExplainSummary) -> dict:
    """The parser-derived details (scans, lookups, optimised pipeline) stored in ``raw``."""
    return ((s.raw or {}).get("derived")) or {}
