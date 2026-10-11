"""MongoDB clients for Track B code: thin wrappers over Track A's ``ratchet.db`` / ``config``.

Every env function also accepts explicit clients (tests use a local mongod).
"""

from __future__ import annotations

from pymongo import MongoClient

from ratchet import db
from ratchet.config import get_settings


def rw_client() -> MongoClient:
    return db.rw_client()


def ro_client() -> MongoClient:
    return db.ro_client()


def current_ns() -> str:
    return get_settings().ns


def scratch_db_name(ns: str) -> str:
    return db.scratch_db_name(ns)
