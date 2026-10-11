"""Detectors (SPEC §5.4) on real explain fixtures: one firing and one quiet case each."""

import pytest

from ratchet.contracts import QuerySpec, Workload
from ratchet.env.detectors import late_matches, load_config, run_detectors
from ratchet.env.explain import summarize
from tests.env.conftest import LABELS, FakeEnv, load_fixture, measurement, summary

CFG = load_config()
LABEL = "atlas-8.0" if "atlas-8.0" in LABELS else LABELS[0]
BIG = {"movies": 21349, "comments": 41079}


def wl(*queries: QuerySpec, collections=("movies", "comments"), write_rate=None) -> Workload:
    return Workload(
        id="w",
        split="train",
        db="d",
        collections=list(collections),
        queries=list(queries),
        index_budget={c: 3 for c in collections},
        write_rate=write_rate or {},
    )


def run(names, env=None, w_kwargs=None):
    qs, per = [], {}
    for n in names:
        q, raw = load_fixture(LABEL, n)
        qs.append(q)
        per[q.id] = summarize(q.id, raw)
    w = wl(*qs, **(w_kwargs or {}))
    return run_detectors(env or FakeEnv(BIG), w, measurement(w, per), cfg=CFG)


def dets(findings, qid=None):
    return {f.detector for f in findings if qid is None or f.query_id == qid}


def test_collscan_large_fires_and_respects_size():
    assert "COLLSCAN_LARGE" in dets(run(["find_collscan"]))
    assert "COLLSCAN_LARGE" not in dets(run(["find_collscan"], FakeEnv({"movies": 4000})))


def test_full_id_scan_counts_as_collscan():
    f = [x for x in run(["find_full_id_scan"]) if x.detector == "COLLSCAN_LARGE"][0]
    assert f.evidence["scan"] == "FULL_INDEX_SCAN(_id_)"


def test_high_examine_ratio():
    assert "HIGH_EXAMINE_RATIO" in dets(run(["find_sort"]))
    assert "HIGH_EXAMINE_RATIO" not in dets(run(["find_sort_esr_index"]))


def test_in_memory_sort():
    assert "IN_MEMORY_SORT" in dets(run(["find_sort"]))
    assert "IN_MEMORY_SORT" not in dets(run(["find_sort_esr_index"]))


def test_sort_after_group_is_low_severity():
    f = [x for x in run(["agg_unwind_match_group_sort"]) if x.detector == "IN_MEMORY_SORT"][0]
    assert f.severity == "low"


def test_not_covered():
    assert "NOT_COVERED" in dets(run(["find_sort"]))
    assert "NOT_COVERED" not in dets(run(["find_projection_covered"]))


def test_unindexed_lookup_sbe_and_classic():
    assert "UNINDEXED_LOOKUP" in dets(run(["agg_lookup_unindexed"]))
    assert "UNINDEXED_LOOKUP" in dets(run(["agg_lookup_pipeline_form"]))
    assert "UNINDEXED_LOOKUP" not in dets(run(["agg_lookup_indexed"]))
    assert "UNINDEXED_LOOKUP" not in dets(run(["agg_lookup_unwind_match"]))  # joins on _id


def test_unanchored_regex():
    assert "UNANCHORED_REGEX" in dets(run(["find_regex"]))  # "night" with "i"
    q = QuerySpec(
        id="q1",
        kind="find",
        collection="movies",
        filter={"title": {"$regex": "^The"}},
        sort=[("_id", 1)],
    )
    w = wl(q)
    found = run_detectors(
        FakeEnv(BIG), w, measurement(w, {"q1": summary("q1", stages=["IXSCAN"])}), cfg=CFG
    )
    assert "UNANCHORED_REGEX" not in dets(found)


def test_redundant_and_unused_index():
    specs = {
        "movies": [
            {"name": "year_1", "keys": [("year", 1)]},
            {"name": "year_1_title_1", "keys": [("year", 1), ("title", 1)]},
            {"name": "p_1", "keys": [("p", 1)], "partialFilterExpression": {"p": 1}},
        ]
    }
    found = run(["find_sort"], FakeEnv(BIG, specs), {"write_rate": {"movies": 40}})
    red = [f for f in found if f.detector == "REDUNDANT_INDEX"]
    assert [(f.evidence["index"], f.evidence["covered_by"]) for f in red] == [
        ("year_1", "year_1_title_1")
    ]
    unused = {f.evidence["index"]: f.severity for f in found if f.detector == "UNUSED_INDEX"}
    assert unused == {
        "year_1_title_1": "medium",
        "p_1": "medium",
    }  # year_1 is used; write-heavy -> medium


def test_index_used_by_lookup_is_not_unused():
    specs = {"comments": [{"name": "movie_id_1", "keys": [("movie_id", 1)]}]}
    found = run(["agg_lookup_indexed"], FakeEnv(BIG, specs))
    assert not [
        f for f in found if f.detector == "UNUSED_INDEX" and f.evidence["index"] == "movie_id_1"
    ]


@pytest.mark.parametrize(
    "pipeline,expected",
    [
        (
            [
                {"$lookup": {"from": "m", "localField": "a", "foreignField": "_id", "as": "x"}},
                {"$match": {"date": 1}},
            ],
            [["date"]],
        ),
        (
            [
                {"$lookup": {"from": "m", "localField": "a", "foreignField": "_id", "as": "x"}},
                {"$match": {"x.year": 1}},
            ],
            [],
        ),  # needs the joined field: not late
        ([{"$unwind": "$g"}, {"$match": {"g": "Drama", "year": 1}}], [["g", "year"]]),
        ([{"$group": {"_id": "$y", "n": {"$sum": 1}}}, {"$match": {"n": {"$gt": 1}}}], []),
        (
            [
                {"$addFields": {"l": {"$first": "$gr"}}},
                {"$lookup": {"from": "m", "localField": "a", "foreignField": "_id", "as": "x"}},
                {"$match": {"cuisine": "A", "l.grade": "A"}},
            ],
            [],
        ),
        ([{"$match": {"a": 1}}, {"$unwind": "$b"}], []),
    ],
)
def test_late_matches(pipeline, expected):
    assert [lm["fields"] for lm in late_matches(pipeline)] == expected


def test_late_match_flags_unwound_prefilter():
    lm = late_matches([{"$unwind": "$g"}, {"$match": {"g": "Drama"}}])[0]
    assert lm["on_unwound"] == ["g"] and lm["after"] == "$unwind"


def test_late_match_suppressed_when_optimiser_already_moved_it():
    """Classic engine moved $match into the $cursor: the detector stays quiet."""
    q = QuerySpec(
        id="q1",
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
            {"$match": {"date": 1}},
            {"$count": "n"},
        ],
    )
    s = summary("q1", stages=["COLLSCAN", "$lookup", "$group"])
    s.raw = {
        "derived": {
            "engine": "classic",
            "pushed_down": False,
            "pipeline": [{"name": "$lookup"}, {"name": "$group"}, {"name": "$project"}],
        }
    }
    w = wl(q)
    assert "LATE_MATCH" not in dets(
        run_detectors(FakeEnv(BIG), w, measurement(w, {"q1": s}), cfg=CFG)
    )
    s.raw["derived"]["pipeline"] = [{"name": "$lookup"}, {"name": "$match"}, {"name": "$group"}]
    assert "LATE_MATCH" in dets(run_detectors(FakeEnv(BIG), w, measurement(w, {"q1": s}), cfg=CFG))
