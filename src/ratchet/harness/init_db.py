"""`ratchet init`: collections, indexes (incl. Atlas Vector Search) and the gen-0 harness."""

from __future__ import annotations

from pydantic import BaseModel
from pymongo import ASCENDING
from pymongo.database import Database
from pymongo.operations import SearchIndexModel

from ratchet import db as dbm
from ratchet.config import Settings
from ratchet.harness.gen0 import gen0_spec
from ratchet.harness.models import HarnessDoc
from ratchet.harness.store import HarnessStore

VECTOR_INDEX = "vector_index"
EMBEDDING_FIELD = "embedding"

# SPEC §12: collection → [(keys, options)]
INDEXES: dict[str, list[tuple[list[tuple[str, int]], dict]]] = {
    dbm.WORKLOADS: [([("split", ASCENDING)], {})],
    dbm.HARNESS_VERSIONS: [([("version", ASCENDING)], {"unique": True})],
    dbm.TUNE_RESULTS: [
        ([("harness_version", ASCENDING), ("split", ASCENDING)], {}),
        ([("harness_version", ASCENDING), ("workload_id", ASCENDING)], {}),
    ],
    dbm.MEMORIES: [([("status", ASCENDING)], {})],
    dbm.GATE_DECISIONS: [([("candidate", ASCENDING)], {})],
    dbm.GENERATIONS: [([("gen", ASCENDING)], {})],
    dbm.LLM_CACHE: [([("key", ASCENDING)], {"unique": True})],
    dbm.TRANSFER_RUNS: [([("harness_version", ASCENDING)], {})],
}

# collection → extra filter fields usable in $vectorSearch
VECTOR_COLLECTIONS: dict[str, list[str]] = {dbm.FAILURES: [], dbm.MEMORIES: ["status"]}


class InitReport(BaseModel):
    database: str
    created_collections: list[str] = []
    indexes: list[str] = []
    vector_indexes: dict[str, str] = {}  # collection → created | exists | failed: <why>
    head: str = ""
    head_created: bool = False


def vector_definition(dims: int, filters: list[str]) -> dict:
    fields: list[dict] = [
        {"type": "vector", "path": EMBEDDING_FIELD, "numDimensions": dims, "similarity": "cosine"}
    ]
    fields += [{"type": "filter", "path": f} for f in filters]
    return {"fields": fields}


def _ensure_vector_index(coll, dims: int, filters: list[str]) -> str:
    try:
        if any(ix.get("name") == VECTOR_INDEX for ix in coll.list_search_indexes()):
            return "exists"
        coll.create_search_index(
            SearchIndexModel(
                definition=vector_definition(dims, filters), name=VECTOR_INDEX, type="vectorSearch"
            )
        )
        return "created"
    except Exception as e:  # tier without search, or a non-Atlas deployment
        return f"failed: {type(e).__name__}: {str(e)[:120]}"


def init_database(
    db: Database, settings: Settings | None = None, vector: bool = True
) -> InitReport:
    rep = InitReport(database=db.name)
    existing = set(db.list_collection_names())
    for name in dbm.ALL_COLLECTIONS:
        if name not in existing:
            db.create_collection(name)
            rep.created_collections.append(name)
    for name, specs in INDEXES.items():
        for keys, opts in specs:
            rep.indexes.append(f"{name}.{db[name].create_index(keys, **opts)}")
    if vector:
        dims = settings.embed_dims if settings else 1024
        for name, filters in VECTOR_COLLECTIONS.items():
            rep.vector_indexes[name] = _ensure_vector_index(db[name], dims, filters)
    store = HarnessStore(db)
    rep.head_created = store.head_id() is None
    head: HarnessDoc = store.init_gen0(gen0_spec(settings))
    rep.head = head.id
    return rep
