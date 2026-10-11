"""MongoDB handles: RW client for `ratchet_<ns>` and scratch DBs, RO client for `sample_*`."""

from __future__ import annotations

from functools import lru_cache

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from ratchet.config import get_settings

# Collection names (SPEC §12)
WORKLOADS = "workloads"
HARNESS_VERSIONS = "harness_versions"
HARNESS_HEAD = "harness_head"
TUNE_RESULTS = "tune_results"
FAILURES = "failures"
MEMORIES = "memories"
GATE_DECISIONS = "gate_decisions"
GENERATIONS = "generations"
LLM_CACHE = "llm_cache"
TRANSFER_RUNS = "transfer_runs"

ALL_COLLECTIONS = [
    WORKLOADS,
    HARNESS_VERSIONS,
    HARNESS_HEAD,
    TUNE_RESULTS,
    FAILURES,
    MEMORIES,
    GATE_DECISIONS,
    GENERATIONS,
    LLM_CACHE,
    TRANSFER_RUNS,
]

_APP_NAME = "ratchet"


@lru_cache(maxsize=1)
def rw_client() -> MongoClient:
    return MongoClient(get_settings().require_rw_uri(), appname=_APP_NAME, tz_aware=True)


@lru_cache(maxsize=1)
def ro_client() -> MongoClient:
    return MongoClient(get_settings().require_ro_uri(), appname=_APP_NAME, tz_aware=True)


def db_name(ns: str | None = None) -> str:
    return f"ratchet_{ns or get_settings().ns}"


def scratch_db_name(ns: str | None = None) -> str:
    return f"ratchet_scratch_{ns or get_settings().ns}"


def get_db(ns: str | None = None) -> Database:
    """The `ratchet_<ns>` state database (versions, traces, memory, cache)."""
    return rw_client()[db_name(ns)]


def get_scratch_db(ns: str | None = None) -> Database:
    """The `ratchet_scratch_<ns>` database holding scratch copies of workload collections."""
    return rw_client()[scratch_db_name(ns)]


def get_source_db(name: str) -> Database:
    """A read-only handle on a source dataset (e.g. `sample_mflix`)."""
    return ro_client()[name]


def coll(name: str, ns: str | None = None) -> Collection:
    if name not in ALL_COLLECTIONS:
        raise KeyError(f"unknown ratchet collection: {name}")
    return get_db(ns)[name]


def ping() -> dict[str, bool]:
    """Connectivity check for both users. Never raises; never echoes URIs."""
    out: dict[str, bool] = {}
    for label, factory in (("rw", rw_client), ("ro", ro_client)):
        try:
            factory().admin.command("ping")
            out[label] = True
        except Exception:
            out[label] = False
    return out
