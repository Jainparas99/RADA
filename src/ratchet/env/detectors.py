"""Deterministic detectors (SPEC §5.4). They only diagnose; thresholds in config/detectors.yaml.

Collection-level findings (REDUNDANT_INDEX, UNUSED_INDEX) use ``query_id="*"``.
``env`` only needs ``count(coll)`` and ``index_specs(coll)`` (ScratchEnv, or a fake in tests).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import yaml
from bson.regex import Regex

from ratchet.contracts import ExplainSummary, Finding, Measurement, QuerySpec, Workload
from ratchet.env.explain import derived

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "detectors.yaml"
_BLOCKING = {"$lookup", "$unwind", "$group"}
_SPECIAL_KEY_TYPES = {"text", "2dsphere", "2d", "hashed"}


class EnvLike(Protocol):
    def count(self, coll: str) -> int: ...
    def index_specs(self, coll: str) -> list[dict[str, Any]]: ...


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- helpers


def _filter_fields(f: Any, out: list[str]) -> None:
    if isinstance(f, dict):
        for k, v in f.items():
            if k in ("$and", "$or", "$nor") and isinstance(v, list):
                for sub in v:
                    _filter_fields(sub, out)
            elif not k.startswith("$") and k not in out:
                out.append(k)


def _regexes(obj: Any, path: str = "") -> list[tuple[str, str, str]]:
    """(field, pattern, options) for every regex predicate in a filter."""
    found: list[tuple[str, str, str]] = []
    if isinstance(obj, (Regex, re.Pattern)):
        opts = "i" if obj.flags & re.IGNORECASE else ""
        found.append((path, obj.pattern, opts))
    elif isinstance(obj, dict):
        if "$regex" in obj:
            pat = obj["$regex"]
            if isinstance(pat, (Regex, re.Pattern)):
                found.extend(_regexes(pat, path))
            else:
                found.append((path, str(pat), str(obj.get("$options", ""))))
        for k, v in obj.items():
            if k in ("$regex", "$options"):
                continue
            sub = path if k.startswith("$") else (f"{path}.{k}" if path else k)
            found.extend(_regexes(v, sub))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_regexes(v, path))
    return found


def _norm_keys(keys: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    out = []
    for k, v in keys:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = 1 if v > 0 else -1
        out.append((k, v))
    return out


def _match_roots(spec: Any) -> set[str]:
    fields: list[str] = []
    _filter_fields(spec, fields)
    return {f.split(".", 1)[0] for f in fields}


def late_matches(pipeline: list[dict]) -> list[dict]:
    """$match stages after $lookup/$unwind/$group whose fields existed before that stage."""
    out: list[dict] = []
    produced: set[str] = set()  # roots created by earlier stages
    unwound: set[str] = set()
    blocker: str | None = None
    opaque = False  # after $group/$replaceRoot/$project-with-exprs we can't track fields
    for i, st in enumerate(pipeline):
        if not isinstance(st, dict) or len(st) != 1:
            continue
        name, spec = next(iter(st.items()))
        if name == "$match" and blocker and not opaque:
            if isinstance(spec, dict) and "$expr" not in spec and "$text" not in spec:
                roots = _match_roots(spec)
                if roots and not (roots & produced):
                    out.append(
                        {
                            "stage_index": i,
                            "after": blocker,
                            "fields": sorted(roots),
                            "on_unwound": sorted(roots & unwound),
                        }
                    )
        elif name == "$lookup" and isinstance(spec, dict):
            blocker = blocker or "$lookup"
            if spec.get("as"):
                produced.add(str(spec["as"]).split(".", 1)[0])
        elif name == "$unwind":
            blocker = blocker or "$unwind"
            path = spec if isinstance(spec, str) else (spec or {}).get("path", "")
            unwound.add(str(path).lstrip("$").split(".", 1)[0])
            if isinstance(spec, dict) and spec.get("includeArrayIndex"):
                produced.add(spec["includeArrayIndex"])
        elif name == "$group":
            blocker = blocker or "$group"
            opaque = True
        elif name in ("$addFields", "$set") and isinstance(spec, dict):
            produced.update(k.split(".", 1)[0] for k in spec)
        elif (
            name == "$project"
            and isinstance(spec, dict)
            and all(v in (0, 1, True, False) for v in spec.values())
        ):
            pass  # pure inclusion/exclusion: kept fields are still the original fields
        elif name == "$project" and isinstance(spec, dict):
            produced.update(
                k.split(".", 1)[0] for k, v in spec.items() if v not in (0, 1, True, False)
            )
        elif name in (
            "$replaceRoot",
            "$replaceWith",
            "$facet",
            "$bucket",
            "$bucketAuto",
            "$sortByCount",
            "$count",
        ):
            opaque = True
    return out


# --------------------------------------------------------------------------- detectors


def _f(qid: str, det: str, sev: str, evidence: dict, hint: str) -> Finding:
    return Finding(query_id=qid, detector=det, severity=sev, evidence=evidence, hint=hint)


def _per_query(q: QuerySpec, s: ExplainSummary, env: EnvLike, cfg: dict) -> list[Finding]:
    out: list[Finding] = []
    d = derived(s)
    plan_stages = d.get("plan_stages") or s.stages
    n = max(1, s.n_returned)
    coll_count = env.count(q.collection)

    # COLLSCAN_LARGE (also a full-range index scan, e.g. walking _id_ only to sort)
    c = cfg["COLLSCAN_LARGE"]
    if coll_count > c["min_docs"]:
        for sc in d.get("scans", []):
            if sc.get("full_range"):
                kind = (
                    "COLLSCAN"
                    if sc["stage"] == "COLLSCAN"
                    else f"FULL_INDEX_SCAN({sc.get('index')})"
                )
                out.append(
                    _f(
                        q.id,
                        "COLLSCAN_LARGE",
                        c["severity"],
                        {
                            "scan": kind,
                            "collection_docs": coll_count,
                            "docs_examined": s.docs_examined,
                            "keys_examined": s.keys_examined,
                            "n_returned": s.n_returned,
                        },
                        "add an index on the filter fields (equality, then sort, then range)",
                    )
                )
                break

    # HIGH_EXAMINE_RATIO (only when an index is used: otherwise it's a COLLSCAN finding)
    c = cfg["HIGH_EXAMINE_RATIO"]
    examined = s.keys_examined + s.docs_examined + s.lookup_examined
    denom = n
    if q.kind == "aggregate":  # $group/$count shrink nReturned; use docs leaving the query layer
        es = (s.raw or {}).get("executionStats") or {}
        denom = max(1, int(es.get("nReturned") or n))
    ratio = examined / denom
    if s.indexes_used and ratio > c["ratio"]:
        out.append(
            _f(
                q.id,
                "HIGH_EXAMINE_RATIO",
                c["severity"],
                {
                    "ratio": round(ratio, 1),
                    "examined": examined,
                    "returned": denom,
                    "indexes_used": s.indexes_used,
                },
                "index is unselective or wrongly ordered: "
                "equality fields first, then sort, then range",
            )
        )

    # IN_MEMORY_SORT
    c = cfg["IN_MEMORY_SORT"]
    if s.has_in_memory_sort:
        in_plan = "SORT" in plan_stages
        pipe_names = [p["name"] for p in d.get("pipeline", [])]
        before_sort = pipe_names[: pipe_names.index("$sort")] if "$sort" in pipe_names else []
        after_group = not in_plan and ("$group" in before_sort or "GROUP" in plan_stages)
        after_join = (
            not in_plan
            and not after_group
            and (
                any(x in before_sort for x in ("$lookup", "$unwind")) or "EQ_LOOKUP" in plan_stages
            )
        )
        sev = c["severity_after_group"] if after_group else c["severity"]
        if after_group:
            hint = "sort runs after $group and cannot use an index; group fewer documents"
        elif after_join:
            hint = (
                "sort runs after $lookup/$unwind; if it sorts on local fields, move $sort "
                "(with any $limit) before the join so an index can provide the order"
            )
        else:
            hint = (
                "put the sort field in the index after the equality fields (equality, sort, range)"
            )
        out.append(
            _f(
                q.id,
                "IN_MEMORY_SORT",
                sev,
                {"docs_sorted": d.get("sort_docs", s.docs_examined), "in_query_layer": in_plan},
                hint,
            )
        )

    # NOT_COVERED
    c = cfg["NOT_COVERED"]
    if q.kind == "find" and q.projection and "FETCH" in plan_stages:
        inc = {k: v for k, v in q.projection.items() if v not in (0, False)}
        excludes_id = q.projection.get("_id") in (0, False)
        if (
            inc
            and all(v in (1, True) for v in inc.values())
            and len(inc) <= c["max_projection_fields"]
        ):
            needed: list[str] = []
            _filter_fields(q.filter, needed)
            for k, _ in q.sort or []:
                if k not in needed:
                    needed.append(k)
            for k in inc:
                if k not in needed:
                    needed.append(k)
            if not excludes_id and "_id" not in needed:
                needed.append("_id")
            out.append(
                _f(
                    q.id,
                    "NOT_COVERED",
                    c["severity"],
                    {"needed_fields": needed, "docs_examined": s.docs_examined},
                    "a compound index on all needed fields (filter, sort, projection) "
                    "would make it covered",
                )
            )

    # LATE_MATCH
    c = cfg["LATE_MATCH"]
    if q.kind == "aggregate" and q.pipeline:
        late = late_matches(q.pipeline)
        if late and d.get("engine") == "classic" and not d.get("pushed_down"):
            # the optimiser may already have moved it; keep only if a $match still follows
            optimised = [p["name"] for p in d.get("pipeline", [])]
            still = any(
                name == "$match" and any(b in optimised[:j] for b in _BLOCKING)
                for j, name in enumerate(optimised)
            )
            late = late if still else []
        for lm in late:
            out.append(
                _f(
                    q.id,
                    "LATE_MATCH",
                    c["severity"],
                    lm,
                    f"move $match on {lm['fields']} before {lm['after']}"
                    + (" (keep the original $match after $unwind too)" if lm["on_unwound"] else ""),
                )
            )

    # UNINDEXED_LOOKUP
    c = cfg["UNINDEXED_LOOKUP"]
    for lk in d.get("lookups", []):
        unindexed = (
            (lk.get("collection_scans") or 0) > 0
            if lk["source"] == "classic"
            else not lk.get("indexed")
        )
        if unindexed:
            out.append(
                _f(
                    q.id,
                    "UNINDEXED_LOOKUP",
                    c["severity"],
                    {
                        k: lk.get(k)
                        for k in (
                            "from",
                            "foreign_field",
                            "strategy",
                            "collection_scans",
                            "docs_examined",
                            "pipeline_form",
                        )
                    },
                    f"index {lk.get('from')}.{lk.get('foreign_field') or '<joined field>'}",
                )
            )

    # UNANCHORED_REGEX
    c = cfg["UNANCHORED_REGEX"]
    sources: list[Any] = (
        [q.filter]
        if q.kind == "find"
        else [st["$match"] for st in (q.pipeline or []) if isinstance(st, dict) and "$match" in st]
    )
    for src in sources:
        for field, pat, opts in _regexes(src):
            if not pat.startswith("^") or "i" in opts:
                out.append(
                    _f(
                        q.id,
                        "UNANCHORED_REGEX",
                        c["severity"],
                        {"field": field, "pattern": pat, "options": opts},
                        "anchor the regex with ^ (case-sensitive) if semantics allow, "
                        "or accept the scan; an index cannot bound an unanchored regex",
                    )
                )
    return out


def _collection_level(workload: Workload, m: Measurement, env: EnvLike, cfg: dict) -> list[Finding]:
    out: list[Finding] = []
    used: dict[str, set[str]] = {c: set() for c in workload.collections}
    coll_of = {q.id: q.collection for q in workload.queries}
    for qid, s in m.per_query.items():
        used.setdefault(coll_of.get(qid, ""), set()).update(s.indexes_used)
        for lk in derived(s).get("lookups", []):
            used.setdefault(lk.get("from") or "", set()).update(lk.get("indexes_used") or [])

    for coll in workload.collections:
        specs = [x for x in env.index_specs(coll) if x["name"] != "_id_"]
        plain = [
            x
            for x in specs
            if not any(x.get(o) for o in ("partialFilterExpression", "unique", "sparse", "hidden"))
            and not any(v in _SPECIAL_KEY_TYPES for _, v in x["keys"])
        ]
        c = cfg["REDUNDANT_INDEX"]
        for a in plain:
            ka = _norm_keys(a["keys"])
            for b in plain:
                kb = _norm_keys(b["keys"])
                if a is not b and len(ka) < len(kb) and kb[: len(ka)] == ka:
                    out.append(
                        _f(
                            "*",
                            "REDUNDANT_INDEX",
                            c["severity"],
                            {"collection": coll, "index": a["name"], "covered_by": b["name"]},
                            f"drop {a['name']}: it is a prefix of {b['name']}",
                        )
                    )
                    break
        c = cfg["UNUSED_INDEX"]
        rate = float(workload.write_rate.get(coll, 0.0))
        heavy = rate >= c["write_heavy_rate"]
        for x in specs:
            if x["name"] not in used.get(coll, set()):
                out.append(
                    _f(
                        "*",
                        "UNUSED_INDEX",
                        c["severity_write_heavy"] if heavy else c["severity"],
                        {"collection": coll, "index": x["name"], "write_rate": rate},
                        f"no query in this workload uses {x['name']}"
                        + ("; drop it, the collection is write-heavy" if heavy else ""),
                    )
                )
    return out


def run_detectors(
    env: EnvLike,
    workload: Workload,
    m: Measurement,
    queries: list[QuerySpec] | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[Finding]:
    cfg = cfg or load_config()
    findings: list[Finding] = []
    for q in queries or workload.queries:
        s = m.per_query.get(q.id)
        if s is not None:
            findings.extend(_per_query(q, s, env, cfg))
    findings.extend(_collection_level(workload, m, env, cfg))
    return findings
