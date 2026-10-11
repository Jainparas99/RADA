"""`llm_cache`: every provider round-trip, keyed by sha256 of (model, messages, params).

`MongoCache` is the real store (collection `llm_cache` in `ratchet_<ns>`); `MemoryCache` is used in
tests and when no database is configured. A broken cache never breaks an LLM call: read and write
errors are logged and treated as a miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any, Protocol

from ratchet.llm.types import RawCompletion

log = logging.getLogger(__name__)


def cache_key(kind: str, model: str, payload: dict[str, Any]) -> str:
    """Stable sha256 over canonical JSON. `kind` separates completions from embeddings."""
    blob = json.dumps(
        {"kind": kind, "model": model, **payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class Cache(Protocol):
    def get(self, key: str) -> dict | None: ...

    def put(self, key: str, model: str, kind: str, value: dict) -> None: ...


class MemoryCache:
    def __init__(self) -> None:
        self.store: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict | None:
        v = self.store.get(key)
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, key: str, model: str, kind: str, value: dict) -> None:
        self.store[key] = value


class NullCache:
    def get(self, key: str) -> dict | None:
        return None

    def put(self, key: str, model: str, kind: str, value: dict) -> None:
        return None


class MongoCache:
    """Documents: {key, kind, model, value, created_at}. `key` is unique (index from `init`)."""

    def __init__(self, collection: Any) -> None:
        self.coll = collection

    def get(self, key: str) -> dict | None:
        try:
            doc = self.coll.find_one({"key": key}, {"_id": 0, "value": 1})
        except Exception as e:  # never let the cache take down a run
            log.warning("llm_cache read failed: %s", type(e).__name__)
            return None
        return doc["value"] if doc else None

    def put(self, key: str, model: str, kind: str, value: dict) -> None:
        try:
            self.coll.replace_one(
                {"key": key},
                {
                    "key": key,
                    "kind": kind,
                    "model": model,
                    "value": value,
                    "created_at": datetime.now(UTC),
                },
                upsert=True,
            )
        except Exception as e:
            log.warning("llm_cache write failed: %s", type(e).__name__)


def default_cache() -> Cache:
    """Mongo `llm_cache` when MONGODB_URI_RW is configured, else an in-process cache."""
    try:
        from ratchet.config import get_settings

        if get_settings().mongodb_uri_rw:
            from ratchet import db

            return MongoCache(db.coll(db.LLM_CACHE))
    except Exception as e:
        log.warning("llm_cache unavailable (%s); using in-memory cache", type(e).__name__)
    return MemoryCache()


def raw_from_cache(value: dict) -> RawCompletion:
    return RawCompletion.model_validate(value)
