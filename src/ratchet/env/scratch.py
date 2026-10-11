"""Scratch copies of workload collections (SPEC §5.1).

Infra, never the agent, copies each workload collection once from the source DB
(read with the RO user) into ``ratchet_scratch_<ns>.<coll>`` (written with the RW user).
Before every tuning run the scratch collection's indexes are reset to the workload's
``baseline_indexes``. The agent only creates/drops indexes and rewrites queries, so data
never needs re-copying; the data checksum in every Measurement proves it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from bson.codec_options import CodecOptions
from bson.raw_bson import RawBSONDocument
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import OperationFailure

from ratchet.contracts import IndexDef, QuerySpec, Workload
from ratchet.env import conn

META = "_ratchet_meta"
ChecksumMode = Literal["dbhash", "full", "ids"]
_RAW = CodecOptions(document_class=RawBSONDocument)
_COPY_BATCH = 1000
# Options we compare when deciding whether an existing index matches a baseline one.
_INDEX_OPTS = ("partialFilterExpression", "unique", "sparse", "expireAfterSeconds", "hidden")


class ScratchError(RuntimeError):
    pass


@dataclass
class ScratchEnv:
    ns: str
    workload_id: str
    source_db: str
    collections: list[str]
    db: Database  # scratch database, RW client
    source: Database  # source database, RO client
    checksum_mode: ChecksumMode = "full"
    copied: dict[str, int] = field(default_factory=dict)  # coll -> docs copied this call
    # Current (possibly rewritten) queries, like FakeScratch.queries. apply_plan updates
    # it; reset_indexes restores the workload's own queries.
    queries: dict[str, QuerySpec] = field(default_factory=dict)

    def current_queries(self, workload: Workload) -> list[QuerySpec]:
        """Workload order, rewritten where a plan rewrote a query."""
        return [self.queries.get(q.id, q) for q in workload.queries]

    @property
    def scratch_db(self) -> str:
        return self.db.name

    def count(self, coll: str) -> int:
        return self.db[coll].estimated_document_count()

    def index_specs(self, coll: str) -> list[dict[str, Any]]:
        return list_index_specs(self.db, coll)


# --------------------------------------------------------------------------- indexes


def list_index_specs(db: Database, coll: str) -> list[dict[str, Any]]:
    out = []
    for ix in db[coll].list_indexes():
        d = dict(ix)
        spec = {"name": d["name"], "keys": [(k, v) for k, v in d["key"].items()]}
        for opt in _INDEX_OPTS:
            if opt in d:
                spec[opt] = d[opt]
        if "weights" in d:
            spec["weights"] = dict(d["weights"])
        out.append(spec)
    return out


def _norm_dir(v: Any) -> int | str:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return 1 if v > 0 else -1
    return str(v)


def _is_text(keys: list[tuple[str, Any]]) -> bool:
    return any(v == "text" for _, v in keys) or any(k == "_fts" for k, _ in keys)


def index_matches(existing: dict[str, Any], want: IndexDef) -> bool:
    """True if an existing index (from list_index_specs) is the baseline index ``want``."""
    if want.name and existing["name"] != want.name:
        return False
    if any(existing.get(o) for o in ("unique", "sparse", "expireAfterSeconds", "hidden")):
        return False
    if (existing.get("partialFilterExpression") or None) != (want.partialFilterExpression or None):
        return False
    if _is_text(want.keys):
        # text indexes are stored as {_fts: "text", _ftsx: 1} plus weights
        want_fields = {k for k, v in want.keys if v == "text"}
        return _is_text(existing["keys"]) and set(existing.get("weights", {})) == want_fields
    ex = [(k, _norm_dir(v)) for k, v in existing["keys"]]
    return ex == [(k, _norm_dir(v)) for k, v in want.keys]


def create_index(db: Database, coll: str, ix: IndexDef) -> str:
    kwargs: dict[str, Any] = {}
    if ix.name:
        kwargs["name"] = ix.name
    if ix.partialFilterExpression:
        kwargs["partialFilterExpression"] = ix.partialFilterExpression
    return db[coll].create_index([(k, v) for k, v in ix.keys], **kwargs)


def reset_indexes(env: ScratchEnv, workload: Workload) -> dict[str, list[str]]:
    """Make every scratch collection's indexes exactly ``baseline_indexes`` (plus ``_id_``)
    and forget any query rewrites."""
    env.queries = {q.id: q for q in workload.queries}
    result: dict[str, list[str]] = {}
    for coll in workload.collections:
        want = list(workload.baseline_indexes.get(coll, []))
        existing = [s for s in env.index_specs(coll) if s["name"] != "_id_"]
        keep: set[str] = set()
        for spec in existing:
            match = next((w for w in want if index_matches(spec, w)), None)
            if match is not None:
                keep.add(spec["name"])
                want.remove(match)
            else:
                env.db[coll].drop_index(spec["name"])
        for ix in want:
            keep.add(create_index(env.db, coll, ix))
        result[coll] = sorted(keep)
    return result


# --------------------------------------------------------------------------- checksums


def probe_dbhash(db: Database, coll: str) -> bool:
    try:
        db.command("dbHash", collections=[coll])
        return True
    except OperationFailure:
        return False


def data_checksum(db: Database, coll: str, mode: ChecksumMode) -> str:
    """``mode:count:digest``. Changes if any document is added, removed or modified
    (``ids`` mode only detects added/removed documents)."""
    c = db.get_collection(coll, codec_options=_RAW)
    if mode == "dbhash":
        r = db.command("dbHash", collections=[coll])
        n = db[coll].count_documents({})
        return f"dbhash:{n}:{r['collections'].get(coll, '')}"
    h = hashlib.sha256()
    n = 0
    projection = {"_id": 1} if mode == "ids" else None
    for doc in c.find({}, projection, sort=[("_id", 1)], batch_size=2000):
        h.update(doc.raw)
        n += 1
    return f"{mode}:{n}:{h.hexdigest()}"


def _choose_checksum_mode(db: Database, coll: str, full_max_mb: float) -> ChecksumMode:
    if probe_dbhash(db, coll):
        return "dbhash"
    try:
        size_mb = db.command("collStats", coll).get("size", 0) / 1e6
    except OperationFailure:
        size_mb = 0.0
    return "full" if size_mb <= full_max_mb else "ids"


# --------------------------------------------------------------------------- prepare


def _copy_server_side(src_name: str, dst: Database, coll: str) -> int | None:
    """Copy inside the cluster with ``$out`` if the RW user can also read the source
    (e.g. one sandbox user for both roles). None if it can't; then we stream instead."""
    try:
        dst.client[src_name][coll].aggregate([{"$out": {"db": dst.name, "coll": coll}}])
    except OperationFailure:
        return None
    return dst[coll].count_documents({})


def _copy_collection(src: Database, dst: Database, coll: str) -> int:
    n = _copy_server_side(src.name, dst, coll)
    if n is not None:
        return n
    dst[coll].drop()
    batch: list[RawBSONDocument] = []
    n = 0
    for doc in src.get_collection(coll, codec_options=_RAW).find({}, sort=[("_id", 1)]):
        batch.append(doc)
        if len(batch) >= _COPY_BATCH:
            dst[coll].insert_many(batch, ordered=True)
            n += len(batch)
            batch = []
    if batch:
        dst[coll].insert_many(batch, ordered=True)
        n += len(batch)
    return n


def prepare_scratch(
    workload: Workload,
    ns: str | None = None,
    *,
    rw: MongoClient | None = None,
    ro: MongoClient | None = None,
    checksum_mode: ChecksumMode | Literal["auto"] | None = None,
    full_max_mb: float | None = None,
    reset: bool = True,
) -> ScratchEnv:
    """Copy each workload collection once into ``ratchet_scratch_<ns>``, then reset
    indexes to the workload's baseline. Idempotent: a collection whose copy is complete
    and whose doc count still matches the source is not copied again."""
    if checksum_mode is None or full_max_mb is None:
        from ratchet.scoring import load_config  # local import: scoring imports env

        ck = load_config().get("checksum", {})
        checksum_mode = checksum_mode or ck.get("mode", "auto")
        full_max_mb = full_max_mb if full_max_mb is not None else float(ck.get("full_max_mb", 50))
    ns = ns or conn.current_ns()
    rw = rw or conn.rw_client()
    ro = ro or conn.ro_client()
    scratch = rw[conn.scratch_db_name(ns)]
    source = ro[workload.db]
    if scratch.name == source.name:
        raise ScratchError("scratch and source database are the same")
    meta = scratch[META]
    copied: dict[str, int] = {}
    for coll in workload.collections:
        src_ns = f"{workload.db}.{coll}"
        m = meta.find_one({"_id": coll})
        if m and m.get("source") != src_ns:
            raise ScratchError(
                f"scratch collection {scratch.name}.{coll} already holds {m.get('source')}; "
                f"cannot also copy {src_ns} (collection names must be unique per namespace)"
            )
        src_count = source[coll].estimated_document_count()
        if src_count == 0:
            raise ScratchError(f"source collection {src_ns} is empty or missing")
        if m and m.get("complete") and scratch[coll].estimated_document_count() == m["count"]:
            continue
        meta.replace_one(
            {"_id": coll}, {"_id": coll, "source": src_ns, "complete": False}, upsert=True
        )
        n = _copy_collection(source, scratch, coll)
        meta.replace_one(
            {"_id": coll},
            {
                "_id": coll,
                "source": src_ns,
                "complete": True,
                "count": n,
                "copied_at": dt.datetime.now(dt.UTC),
            },
            upsert=True,
        )
        copied[coll] = n

    mode: ChecksumMode
    if checksum_mode == "auto":
        m = meta.find_one({"_id": "__checksum_mode__"})
        if m:
            mode = m["mode"]
        else:
            mode = _choose_checksum_mode(scratch, workload.collections[0], full_max_mb)
            meta.replace_one(
                {"_id": "__checksum_mode__"},
                {"_id": "__checksum_mode__", "mode": mode},
                upsert=True,
            )
    else:
        mode = checksum_mode

    env = ScratchEnv(
        ns=ns,
        workload_id=workload.id,
        source_db=workload.db,
        collections=list(workload.collections),
        db=scratch,
        source=source,
        checksum_mode=mode,
        copied=copied,
    )
    if reset:
        reset_indexes(env, workload)
    return env


def cleanup(ns: str | None = None, *, rw: MongoClient | None = None) -> None:
    """Drop the whole scratch database for a namespace."""
    ns = ns or conn.current_ns()
    rw = rw or conn.rw_client()
    rw.drop_database(conn.scratch_db_name(ns))
