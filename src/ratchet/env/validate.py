"""Workload loading and validation (SPEC §5.2). ``workloads/validate.py`` is the CLI.

Workload files are MongoDB Extended JSON (``{"$oid": ...}``, ``{"$date": ...}``), parsed
with ``bson.json_util`` so ObjectIds and dates reach the server as real BSON types.

Static checks need no DB. Live checks run every query read-only on the *source* data:
twice with the default plan, then with ``hint: {$natural: 1}`` and ``{$natural: -1}``.
All four result hashes must be equal: a query whose output depends on scan order (a
missing tie-break, ``$avg`` over doubles, an unordered ``$lookup`` array) would make an
honest index change look like RESULT_MISMATCH. Results must be 1..500 documents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bson import json_util
from pymongo.database import Database

from ratchet.contracts import QuerySpec, Workload
from ratchet.env.apply import _scan, validate_index
from ratchet.env.measure import result_hash, run_query

WORKLOAD_DIR = Path(__file__).resolve().parents[3] / "workloads"
QUERY_COUNTS = {"train": (4, 6), "holdout": (4, 6), "canary": (2, 4)}
SPLIT_COUNTS = {"train": 8, "holdout": 4, "canary": 3}
MAX_RESULTS = 500
_ORDER_BREAKERS = {"$group", "$bucket", "$bucketAuto", "$sortByCount", "$facet", "$unionWith"}


@dataclass
class Report:
    workload_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    live: dict[str, dict[str, Any]] = field(default_factory=dict)  # qid -> {n, hash, ok}

    @property
    def ok(self) -> bool:
        return not self.errors


# --------------------------------------------------------------------------- loading


def load_workload(path: str | Path) -> Workload:
    return Workload.model_validate(json_util.loads(Path(path).read_text()))


def load_workloads(directory: str | Path = WORKLOAD_DIR) -> list[Workload]:
    return [load_workload(p) for p in sorted(Path(directory).glob("*.json"))]


def find_workload(workload_id: str, directory: str | Path = WORKLOAD_DIR) -> Workload:
    p = Path(directory) / f"{workload_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"no workload {workload_id!r} in {directory}")
    return load_workload(p)


# --------------------------------------------------------------------------- static


def _total_order_problem(q: QuerySpec) -> str | None:
    """None if the query's output order is fully determined by the query itself."""
    if q.kind == "find":
        fid = (q.filter or {}).get("_id")
        if fid is not None and not isinstance(fid, dict):
            return None  # point lookup on _id: at most one document
        if not q.sort or q.sort[-1][0] != "_id":
            return "find must sort with _id as the last key (unique tie-break)"
        return None
    stages = q.pipeline or []
    last = stages[-1] if stages else {}
    if "$count" in last:
        return None
    if "$group" in last and last["$group"].get("_id") is None:
        return None
    # find the last $sort and check nothing after it can reorder or multiply docs
    idx = max((i for i, st in enumerate(stages) if "$sort" in st), default=-1)
    if idx < 0:
        return (
            "aggregate must end in $count, a $group with _id: null, "
            "or a $sort with a unique tie-break"
        )
    keys = list(stages[idx]["$sort"].keys())
    after = [next(iter(st)) for st in stages[idx + 1 :]]
    if any(a in _ORDER_BREAKERS or a == "$unwind" for a in after):
        return "a stage after the last $sort can reorder or multiply documents"
    before = [next(iter(st)) for st in stages[:idx]]
    grouped = any(b in ("$group", "$sortByCount", "$bucket") for b in before)
    if keys[-1] != "_id":
        return "last $sort must end with _id (unique tie-break)"
    if not grouped and any(b == "$unwind" for b in before):
        return "after $unwind, _id is not unique; sort on a unique key or group first"
    return None


def _lookup_array_problems(q: QuerySpec) -> list[str]:
    """A $lookup 'as' array has no defined order unless it joins on _id (<= 1 element)."""
    out = []
    stages = q.pipeline or []
    for i, st in enumerate(stages):
        lk = st.get("$lookup") if isinstance(st, dict) else None
        if not isinstance(lk, dict) or lk.get("foreignField") == "_id":
            continue
        ref = f"${lk.get('as', '')}"
        later = stages[i + 1 :]
        rest = json_util.dumps(later)
        reduced = (
            f'{{"$size": "{ref}"}}' in rest
            or any(next(iter(st)) in ("$count", "$group") for st in later if isinstance(st, dict))
            or any(
                st.get("$unwind") in (ref, {"path": ref}) for st in later if isinstance(st, dict)
            )
        )
        if not reduced:
            out.append(
                f"$lookup into {lk.get('from')} returns an unordered '{ref[1:]}' array; "
                f"reduce it ($size/$unwind+$sort/$count) or join on _id"
            )
    return out


def check_static(w: Workload) -> Report:
    r = Report(w.id)
    e = r.errors.append
    if not w.collections or len(set(w.collections)) != len(w.collections):
        e("collections must be a non-empty list without duplicates")
    lo, hi = QUERY_COUNTS[w.split]
    if not lo <= len(w.queries) <= hi:
        e(f"{w.split} workloads need {lo}-{hi} queries, has {len(w.queries)}")
    ids = [q.id for q in w.queries]
    if len(set(ids)) != len(ids):
        e("query ids must be unique")
    for c in w.collections:
        if c not in w.index_budget:
            e(f"index_budget missing for {c}")
        if c not in w.write_rate:
            e(f"write_rate missing for {c}")
    for c, ixs in w.baseline_indexes.items():
        if c not in w.collections:
            e(f"baseline_indexes for unknown collection {c}")
        names = [ix.name for ix in ixs]
        if any(n is None for n in names) or len(set(names)) != len(names):
            e(f"baseline indexes on {c} need unique explicit names")
        if len(ixs) > w.index_budget.get(c, 10**6):
            e(f"baseline on {c} already exceeds its index_budget")
        for ix in ixs:
            for msg in validate_index(ix):
                e(f"baseline index {ix.name}: {msg}")
    for q in w.queries:
        p = f"{q.id}: "
        if q.collection not in w.collections:
            e(p + f"collection {q.collection} not in workload collections")
        if q.freq <= 0:
            e(p + "freq must be > 0")
        if q.kind == "find":
            if q.pipeline is not None:
                e(p + "find query with a pipeline")
        else:
            if not q.pipeline:
                e(p + "aggregate query without a pipeline")
            if q.filter or q.sort or q.limit is not None or q.projection is not None:
                e(p + "aggregate query must express filter/sort/limit/projection as stages")
        found: list[str] = []
        _scan([q.filter, q.projection, q.pipeline], w, q.id, found)
        r.errors.extend(p + f for f in found)
        problem = _total_order_problem(q)
        if problem:
            e(p + problem)
        r.errors.extend(p + m for m in _lookup_array_problems(q))
    return r


def check_all_static(workloads: list[Workload]) -> tuple[list[Report], list[str]]:
    """Per-workload reports plus cross-workload errors."""
    reports = [check_static(w) for w in workloads]
    cross: list[str] = []
    ids = [w.id for w in workloads]
    if len(set(ids)) != len(ids):
        cross.append("workload ids must be unique")
    by_split: dict[str, set[str]] = {"train": set(), "holdout": set(), "canary": set()}
    owner: dict[str, str] = {}
    for w in workloads:
        for c in w.collections:
            by_split[w.split].add(f"{w.db}.{c}")
            prev = owner.setdefault(c, w.db)
            if prev != w.db:
                cross.append(
                    f"collection name {c!r} exists in {prev} and {w.db}; scratch "
                    f"namespaces need unique collection names"
                )
    overlap = by_split["train"] & by_split["holdout"]
    if overlap:
        cross.append(f"train and holdout share collections: {sorted(overlap)}")
    overlap = by_split["train"] & by_split["canary"]
    if overlap:
        cross.append(f"train and canary share collections: {sorted(overlap)}")
    counts = {s: sum(1 for w in workloads if w.split == s) for s in SPLIT_COUNTS}
    if counts != SPLIT_COUNTS:
        reports and reports[0].warnings.append(
            f"split counts {counts} differ from SPEC {SPLIT_COUNTS}"
        )
    return reports, cross


# --------------------------------------------------------------------------- live


def _run_hinted(db: Database, q: QuerySpec, hint: dict | None):
    coll = db[q.collection]
    if q.kind == "find":
        cur = coll.find(q.filter or {}, q.projection)
        if q.sort:
            cur = cur.sort([(k, v) for k, v in q.sort])
        if q.limit is not None:
            cur = cur.limit(q.limit)
        if hint:
            cur = cur.hint([(k, v) for k, v in hint.items()])
        return cur
    kwargs = {"hint": hint} if hint else {}
    return coll.aggregate(q.pipeline or [], **kwargs)


def check_live(w: Workload, source: Database, report: Report | None = None) -> Report:
    r = report or Report(w.id)
    for q in w.queries:
        try:
            docs = list(run_query(source, q))
            h1 = result_hash(docs)
            h2 = result_hash(run_query(source, q))
            h3 = result_hash(_run_hinted(source, q, {"$natural": 1}))
            h4 = result_hash(_run_hinted(source, q, {"$natural": -1}))
        except Exception as ex:  # noqa: BLE001 - report every failure
            r.errors.append(f"{q.id}: query failed: {type(ex).__name__}: {str(ex)[:200]}")
            continue
        n = len(docs)
        ok = True
        if len({h1, h2}) != 1:
            r.errors.append(f"{q.id}: non-deterministic (two default runs differ)")
            ok = False
        elif len({h1, h3, h4}) != 1:
            r.errors.append(f"{q.id}: result depends on scan order ($natural hints differ)")
            ok = False
        if n == 0:
            r.errors.append(f"{q.id}: empty result")
            ok = False
        if n > MAX_RESULTS:
            r.errors.append(f"{q.id}: {n} results > {MAX_RESULTS}")
            ok = False
        r.live[q.id] = {"n": n, "hash": h1, "ok": ok}
    return r


# --------------------------------------------------------------------------- leakage vocabulary


def _query_fields(q: QuerySpec) -> set[str]:
    out: set[str] = set()

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and not k.startswith("$"):
                    out.update(p for p in k.split(".") if p)
                walk(v)
        elif isinstance(x, list | tuple):
            for v in x:
                walk(v)
        elif isinstance(x, str) and x.startswith("$") and not x.startswith("$$"):
            out.update(p for p in x[1:].split(".") if p)

    walk([q.filter, q.projection, [k for k, _ in q.sort or []], q.pipeline])
    return out


# Generic words a lesson may legitimately use ("put $limit early", "unwind items"):
# never treated as holdout-specific even if only holdout queries use them as field names.
GENERIC_TERMS = {
    "limit",
    "items",
    "item",
    "month",
    "best",
    "n",
    "qty",
    "quantity",
    "age",
    "customer",
    "region",
    "channel",
    "priority",
    "products",
    "result",
    "date",
    "name",
    "type",
    "score",
    "scores",
    "status",
    "total",
    "city",
    "state",
    "address",
    "zip",
    "email",
    "id",
}


def holdout_terms(workloads: list[Workload]) -> dict[str, list[str]]:
    """Names the proposer and lesson memory must never mention (for Track A's
    HOLDOUT_LEAKAGE check): holdout/canary collection names and workload ids, and
    field names that appear in holdout/canary queries but in no train query."""
    train = [w for w in workloads if w.split == "train"]
    hidden = [w for w in workloads if w.split != "train"]
    train_fields = {f for w in train for q in w.queries for f in _query_fields(q)}
    train_colls = {c for w in train for c in w.collections}
    hidden_fields = {f for w in hidden for q in w.queries for f in _query_fields(q)}
    return {
        "collections": sorted({c for w in hidden for c in w.collections} - train_colls),
        "workload_ids": sorted(w.id for w in hidden),
        "fields": sorted(hidden_fields - train_fields - GENERIC_TERMS - {"_id"}),
    }
