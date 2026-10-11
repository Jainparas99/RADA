"""MongoDB clients for Track B code.

Reads ``MONGODB_URI_RW`` / ``MONGODB_URI_RO`` / ``RATCHET_NS`` from the environment
(``.env`` is loaded if present). If only one URI is set (local development), it is used
for both roles. Track A's ``ratchet.db`` can replace this once it lands; every env
function also accepts explicit clients.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pymongo import MongoClient

_loaded = False


def _load_dotenv() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover
        pass


def _uri(primary: str, fallback: str) -> str:
    _load_dotenv()
    uri = os.environ.get(primary) or os.environ.get(fallback) or os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError(f"{primary} is not set (see .env.example)")
    return uri


@lru_cache(maxsize=4)
def _client(uri: str) -> MongoClient:
    return MongoClient(uri, appname="ratchet", tz_aware=False)


def rw_client() -> MongoClient:
    return _client(_uri("MONGODB_URI_RW", "MONGODB_URI_RO"))


def ro_client() -> MongoClient:
    return _client(_uri("MONGODB_URI_RO", "MONGODB_URI_RW"))


def current_ns() -> str:
    _load_dotenv()
    ns = os.environ.get("RATCHET_NS", "dev").strip()
    if not ns or not ns.replace("_", "").isalnum():
        raise RuntimeError(f"RATCHET_NS must be alphanumeric, got {ns!r}")
    return ns


def scratch_db_name(ns: str) -> str:
    return f"ratchet_scratch_{ns}"
