"""Capture raw explain("executionStats") outputs as test fixtures.

Copies sample_mflix.movies + comments into the scratch DB of RATCHET_NS (or --ns), sets
fixture indexes there, runs each fixture query's explain and writes the raw output to
tests/env/fixtures/<label>/<name>.json (host info stripped). Run it once against the
event sandbox so the normaliser tests cover the sandbox's MongoDB version:

    uv run python scripts/capture_explain_fixtures.py --label atlas-sandbox

Locally (any mongod with movies/comments-shaped data in --db):

    MONGODB_URI_RW=mongodb://localhost:27017 uv run python scripts/capture_explain_fixtures.py \
        --db ratchet_probe --label local-8.2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bson import json_util
from bson.json_util import RELAXED_JSON_OPTIONS

from ratchet.contracts import IndexDef, QuerySpec, Workload
from ratchet.env import conn
from ratchet.env.explain import run_explain
from ratchet.env.scratch import create_index, prepare_scratch

OUT = Path(__file__).resolve().parents[1] / "tests" / "env" / "fixtures"


def q(id: str, **kw) -> QuerySpec:
    return QuerySpec(id=id, **kw)


# Phase 1: baseline indexes movies{year_1, year_1_title_1}, comments{} (none on movie_id).
PHASE1 = [
    q(
        "find_sort",
        kind="find",
        collection="movies",
        filter={"genres": "Drama", "year": {"$gt": 2000}},
        sort=[("imdb.rating", -1), ("_id", 1)],
        limit=10,
        projection={"title": 1, "imdb.rating": 1},
    ),
    q(
        "find_projection_covered",
        kind="find",
        collection="movies",
        filter={"year": 1999},
        projection={"title": 1, "_id": 0},
    ),
    q(
        "find_collscan",
        kind="find",
        collection="movies",
        filter={"directors": "Christopher Nolan", "runtime": {"$gt": 0}},
    ),
    q(
        "find_full_id_scan",
        kind="find",
        collection="movies",
        filter={"directors": "Christopher Nolan"},
        sort=[("_id", 1)],
    ),
    q(
        "find_regex",
        kind="find",
        collection="movies",
        filter={"title": {"$regex": "night", "$options": "i"}},
        sort=[("_id", 1)],
        limit=5,
    ),
    q(
        "find_or",
        kind="find",
        collection="movies",
        filter={"$or": [{"year": 1999}, {"year": 2001}]},
        sort=[("_id", 1)],
        limit=20,
    ),
    q("find_express_id", kind="find", collection="movies", filter={"_id": 1}),
    q(
        "agg_lookup_unwind_match",
        kind="aggregate",
        collection="comments",
        pipeline=[
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
    ),
    q(
        "agg_lookup_unindexed",
        kind="aggregate",
        collection="movies",
        pipeline=[
            {"$match": {"year": 1999}},
            {
                "$lookup": {
                    "from": "comments",
                    "localField": "_id",
                    "foreignField": "movie_id",
                    "as": "c",
                }
            },
            {"$project": {"title": 1, "n": {"$size": "$c"}}},
            {"$sort": {"_id": 1}},
        ],
    ),
    q(
        "agg_lookup_pipeline_form",
        kind="aggregate",
        collection="movies",
        pipeline=[
            {"$match": {"year": 1999}},
            {
                "$lookup": {
                    "from": "comments",
                    "let": {"id": "$_id"},
                    "pipeline": [{"$match": {"$expr": {"$eq": ["$movie_id", "$$id"]}}}],
                    "as": "c",
                }
            },
            {"$sort": {"_id": 1}},
            {"$limit": 3},
        ],
    ),
    q(
        "agg_pushdown",
        kind="aggregate",
        collection="movies",
        pipeline=[
            {"$match": {"year": 1999}},
            {"$sort": {"_id": 1}},
            {"$limit": 5},
            {"$project": {"title": 1}},
        ],
    ),
    q(
        "agg_unwind_match_group_sort",
        kind="aggregate",
        collection="movies",
        pipeline=[
            {"$unwind": "$genres"},
            {"$match": {"genres": "Drama"}},
            {"$group": {"_id": "$year", "n": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ],
    ),
    q(
        "agg_group_indexed",
        kind="aggregate",
        collection="movies",
        pipeline=[
            {"$match": {"year": {"$gte": 1990}}},
            {"$group": {"_id": "$year", "n": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ],
    ),
    q(
        "agg_count_scan",
        kind="aggregate",
        collection="movies",
        pipeline=[{"$match": {"year": 1999}}, {"$count": "n"}],
    ),
]
# Phase 2: + comments{movie_id_1}, movies{genres_1_imdb.rating_-1__id_1_year_1}
PHASE2 = [
    q(
        "find_sort_esr_index",
        kind="find",
        collection="movies",
        filter={"genres": "Drama", "year": {"$gt": 2000}},
        sort=[("imdb.rating", -1), ("_id", 1)],
        limit=10,
        projection={"title": 1, "imdb.rating": 1},
    ),
    q(
        "agg_lookup_indexed",
        kind="aggregate",
        collection="movies",
        pipeline=[
            {"$match": {"year": 1999}},
            {
                "$lookup": {
                    "from": "comments",
                    "localField": "_id",
                    "foreignField": "movie_id",
                    "as": "c",
                }
            },
            {"$project": {"title": 1, "n": {"$size": "$c"}}},
            {"$sort": {"_id": 1}},
        ],
    ),
]


def _strip(raw: dict) -> dict:
    raw = dict(raw)
    raw.pop("serverParameters", None)
    si = raw.pop("serverInfo", None) or {}
    raw["serverInfo"] = {"version": si.get("version")}
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="sample_mflix")
    ap.add_argument("--label", required=True)
    ap.add_argument("--ns", default=None)
    args = ap.parse_args()

    w = Workload(
        id="fixtures",
        split="train",
        db=args.db,
        collections=["movies", "comments"],
        baseline_indexes={
            "movies": [
                IndexDef(keys=[("year", 1)], name="year_1"),
                IndexDef(keys=[("year", 1), ("title", 1)], name="year_1_title_1"),
            ]
        },
        queries=PHASE1 + PHASE2,
    )
    env = prepare_scratch(w, args.ns or conn.current_ns())
    out = OUT / args.label
    out.mkdir(parents=True, exist_ok=True)

    def dump(qs: list[QuerySpec]) -> None:
        for spec in qs:
            raw = _strip(run_explain(env.db, spec))
            doc = {"query": spec.model_dump(mode="json"), "explain": raw}
            text = json_util.dumps(doc, indent=1, json_options=RELAXED_JSON_OPTIONS)
            (out / f"{spec.id}.json").write_text(text + "\n")
            print("wrote", out / f"{spec.id}.json")

    dump(PHASE1)
    create_index(env.db, "comments", IndexDef(keys=[("movie_id", 1)], name="movie_id_1"))
    create_index(
        env.db,
        "movies",
        IndexDef(
            keys=[("genres", 1), ("imdb.rating", -1), ("_id", 1), ("year", 1)], name="esr_idx"
        ),
    )
    dump(PHASE2)
    version = env.db.client.server_info().get("version")
    (out / "_meta.json").write_text(
        json.dumps({"label": args.label, "version": version, "source_db": args.db}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
