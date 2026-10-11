"""Shared fixtures for Track B tests.

Unit tests use explain outputs captured from real servers (tests/env/fixtures/<label>/).
Integration tests (marker ``mongo``) need a MongoDB at MONGODB_URI_TEST (default
localhost); they are skipped if none is reachable. They never touch the Atlas sandbox.
"""

from __future__ import annotations

import os
import random
import uuid
from pathlib import Path
from typing import Any

import pytest
from bson import json_util
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from ratchet.contracts import ExplainSummary, Measurement, QuerySpec, Workload

FIXTURES = Path(__file__).parent / "fixtures"
LABELS = sorted(p.name for p in FIXTURES.iterdir() if p.is_dir())


def load_fixture(label: str, name: str) -> tuple[QuerySpec, dict]:
    d = json_util.loads((FIXTURES / label / f"{name}.json").read_text())
    return QuerySpec.model_validate(d["query"]), d["explain"]


class FakeEnv:
    """The two env methods detectors need."""

    def __init__(
        self, counts: dict[str, int], specs: dict[str, list[dict[str, Any]]] | None = None
    ):
        self.counts = counts
        self.specs = specs or {}

    def count(self, coll: str) -> int:
        return self.counts.get(coll, 0)

    def index_specs(self, coll: str) -> list[dict[str, Any]]:
        return [{"name": "_id_", "keys": [("_id", 1)]}] + self.specs.get(coll, [])


def summary(
    qid: str = "q1",
    *,
    keys: int = 0,
    docs: int = 0,
    n: int = 1,
    sort: bool = False,
    lookup: int = 0,
    stages: list[str] | None = None,
    used: list[str] | None = None,
) -> ExplainSummary:
    return ExplainSummary(
        query_id=qid,
        stages=stages or ["COLLSCAN"],
        n_returned=n,
        keys_examined=keys,
        docs_examined=docs,
        exec_ms=0,
        has_in_memory_sort=sort,
        indexes_used=used or [],
        lookup_examined=lookup,
    )


def measurement(
    w: Workload,
    per_query: dict[str, ExplainSummary],
    *,
    hashes: dict[str, str] | None = None,
    checksums: dict[str, str] | None = None,
    indexes: dict[str, list[str]] | None = None,
) -> Measurement:
    return Measurement(
        workload_id=w.id,
        per_query=per_query,
        workload_cost=0.0,
        result_hashes=hashes if hashes is not None else {q.id: f"h-{q.id}" for q in w.queries},
        data_checksums=checksums
        if checksums is not None
        else {c: f"ck-{c}" for c in w.collections},
        indexes=indexes if indexes is not None else {c: ["_id_"] for c in w.collections},
    )


@pytest.fixture
def small_workload() -> Workload:
    return Workload.model_validate(
        {
            "id": "t_w1",
            "split": "train",
            "db": "src",
            "collections": ["movies", "comments"],
            "baseline_indexes": {
                "movies": [
                    {"keys": [["year", 1]], "name": "year_1"},
                    {"keys": [["year", 1], ["title", 1]], "name": "year_1_title_1"},
                ]
            },
            "write_rate": {"movies": 2.0, "comments": 40.0},
            "index_budget": {"movies": 3, "comments": 2},
            "queries": [
                {
                    "id": "q1",
                    "kind": "find",
                    "collection": "movies",
                    "filter": {"genres": "Drama", "year": {"$gt": 2000}},
                    "sort": [["rating", -1], ["_id", 1]],
                    "limit": 10,
                    "projection": {"title": 1},
                    "freq": 50,
                },
                {
                    "id": "q2",
                    "kind": "aggregate",
                    "collection": "comments",
                    "pipeline": [
                        {
                            "$lookup": {
                                "from": "movies",
                                "localField": "movie_id",
                                "foreignField": "_id",
                                "as": "m",
                            }
                        },
                        {"$unwind": "$m"},
                        {"$match": {"m.year": 2012}},
                        {"$count": "n"},
                    ],
                    "freq": 5,
                },
            ],
        }
    )


# --------------------------------------------------------------------------- live mongo


@pytest.fixture(scope="session")
def mongo() -> MongoClient:
    uri = os.environ.get("MONGODB_URI_TEST", "mongodb://localhost:27017")
    if "mongodb.net" in uri:
        pytest.skip("integration tests never run against Atlas")
    client = MongoClient(uri, serverSelectionTimeoutMS=1500)
    try:
        client.admin.command("ping")
    except PyMongoError:
        pytest.skip(f"no MongoDB reachable at {uri}")
    return client


@pytest.fixture(scope="session")
def source_db(mongo: MongoClient):
    """A small movies/comments source DB (6000 movies so COLLSCAN_LARGE fires)."""
    name = f"ratchet_testsrc_{uuid.uuid4().hex[:8]}"
    db = mongo[name]
    rnd = random.Random(3)
    genres = ["Drama", "Comedy", "Action", "Horror"]
    db.movies.insert_many(
        [
            {
                "_id": i,
                "title": f"Movie {i:05d}",
                "year": 1990 + rnd.randrange(30),
                "genres": rnd.sample(genres, rnd.randint(1, 2)),
                "rating": rnd.randrange(1, 100) / 10,
                "director": f"D{rnd.randrange(300)}",
            }
            for i in range(6000)
        ]
    )
    db.comments.insert_many(
        [
            {
                "_id": i,
                "movie_id": rnd.randrange(6000),
                "user": f"u{rnd.randrange(50)}",
                "n": rnd.randrange(1000),
            }
            for i in range(3000)
        ]
    )
    yield db
    mongo.drop_database(name)


@pytest.fixture
def ns(mongo: MongoClient):
    value = f"t{uuid.uuid4().hex[:8]}"
    yield value
    mongo.drop_database(f"ratchet_scratch_{value}")


@pytest.fixture
def live_workload(source_db) -> Workload:
    return Workload.model_validate(
        {
            "id": "live_w1",
            "split": "train",
            "db": source_db.name,
            "collections": ["movies", "comments"],
            "baseline_indexes": {
                "movies": [
                    {"keys": [["year", 1]], "name": "year_1"},
                    {"keys": [["year", 1], ["title", 1]], "name": "year_1_title_1"},
                ]
            },
            "write_rate": {"movies": 2.0, "comments": 40.0},
            "index_budget": {"movies": 3, "comments": 1},
            "queries": [
                {
                    "id": "q1",
                    "kind": "find",
                    "collection": "movies",
                    "filter": {"genres": "Drama", "year": {"$gt": 2010}},
                    "sort": [["rating", -1], ["_id", 1]],
                    "limit": 10,
                    "projection": {"title": 1, "rating": 1},
                    "freq": 50,
                },
                {
                    "id": "q2",
                    "kind": "find",
                    "collection": "movies",
                    "filter": {"director": "D7"},
                    "sort": [["year", 1], ["_id", 1]],
                    "freq": 10,
                },
                {
                    "id": "q3",
                    "kind": "aggregate",
                    "collection": "movies",
                    "pipeline": [
                        {"$match": {"year": 1999}},
                        {
                            "$lookup": {
                                "from": "comments",
                                "localField": "_id",
                                "foreignField": "movie_id",
                                "as": "c",
                            }
                        },
                        {"$project": {"title": 1, "k": {"$size": "$c"}}},
                        {"$sort": {"_id": 1}},
                    ],
                    "freq": 5,
                },
            ],
        }
    )
