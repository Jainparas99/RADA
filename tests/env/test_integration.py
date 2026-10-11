"""End-to-end on a real (local) mongod: prepare -> measure -> apply -> measure -> score.

Every inner violation is triggered by a real plan here, plus the Phase-3 hardening cases.
Skipped when no local MongoDB is reachable; never runs against Atlas.
"""

import pytest

from ratchet.contracts import TuningPlan
from ratchet.env import apply_plan, measure, prepare_scratch, reset_indexes, run_detectors
from ratchet.env.scratch import data_checksum
from ratchet.scoring import score

pytestmark = pytest.mark.mongo
CFG = {"alpha": 0.002, "beta": 0.01, "gamma": 0.02, "score_min": -1.0, "score_max": 1.0}


@pytest.fixture
def env(mongo, live_workload, ns):
    return prepare_scratch(
        live_workload, ns, rw=mongo, ro=mongo, checksum_mode="full", full_max_mb=50
    )


def run(env, w, plan):
    before = measure(env, w)
    rep = apply_plan(env, w, plan)
    after = measure(env, w, rep.effective_queries)
    tp = plan if isinstance(plan, TuningPlan) else TuningPlan(actions=[])
    return before, rep, after, score(w, before, after, tp, rep, CFG)


Q1 = {
    "id": "q1",
    "kind": "find",
    "collection": "movies",
    "filter": {"genres": "Drama", "year": {"$gt": 2010}},
    "sort": [["rating", -1], ["_id", 1]],
    "limit": 10,
    "projection": {"title": 1, "rating": 1},
}


def names(env, coll):
    return sorted(s["name"] for s in env.index_specs(coll))


# --------------------------------------------------------------------------- scratch


def test_prepare_copies_once_and_resets_indexes(mongo, live_workload, ns):
    e1 = prepare_scratch(live_workload, ns, rw=mongo, ro=mongo)
    assert e1.copied == {"movies": 6000, "comments": 3000} and e1.db.name == f"ratchet_scratch_{ns}"
    e1.db.movies.create_index("rating")
    e1.db.movies.drop_index("year_1")
    e2 = prepare_scratch(live_workload, ns, rw=mongo, ro=mongo)
    assert e2.copied == {}
    assert names(e2, "movies") == ["_id_", "year_1", "year_1_title_1"]
    assert names(e2, "comments") == ["_id_"]


def test_reset_replaces_index_with_same_name_but_different_keys(env, live_workload):
    env.db.movies.drop_index("year_1")
    env.db.movies.create_index([("year", -1)], name="year_1")
    reset_indexes(env, live_workload)
    spec = [s for s in env.index_specs("movies") if s["name"] == "year_1"][0]
    assert spec["keys"] == [("year", 1)]


def test_source_is_never_modified(mongo, env, live_workload, source_db):
    before = sorted(ix["name"] for ix in source_db.movies.list_indexes())
    run(
        env,
        live_workload,
        {
            "actions": [
                {
                    "type": "create_index",
                    "collection": "movies",
                    "index": {"keys": [["director", 1]]},
                }
            ]
        },
    )
    assert sorted(ix["name"] for ix in source_db.movies.list_indexes()) == before


def test_apply_refuses_non_scratch_db(env, live_workload, source_db):
    env2 = type(env)(**{**env.__dict__, "db": source_db})
    with pytest.raises(RuntimeError):
        apply_plan(env2, live_workload, {"actions": []})


@pytest.mark.parametrize("mode", ["full", "dbhash"])
def test_checksum_detects_modification(env, mode):
    a = data_checksum(env.db, "movies", mode)
    assert data_checksum(env.db, "movies", mode) == a
    env.db.movies.update_one({"_id": 3}, {"$set": {"title": "x"}})
    assert data_checksum(env.db, "movies", mode) != a


# --------------------------------------------------------------------------- measure


def test_measure_is_deterministic(env, live_workload):
    a, b = measure(env, live_workload), measure(env, live_workload)
    assert a.result_hashes == b.result_hashes and a.workload_cost == b.workload_cost
    assert a.data_checksums == b.data_checksums and set(a.per_query) == {"q1", "q2", "q3"}


def test_detectors_on_live_env(env, live_workload):
    m = measure(env, live_workload)
    found = {(f.query_id, f.detector) for f in run_detectors(env, live_workload, m)}
    assert ("q2", "COLLSCAN_LARGE") in found and ("q1", "IN_MEMORY_SORT") in found
    assert ("q3", "UNINDEXED_LOOKUP") in found and ("*", "REDUNDANT_INDEX") in found


# --------------------------------------------------------------------------- honest plan


def test_good_plan_scores_and_flips_plan(env, live_workload):
    plan = TuningPlan.model_validate(
        {
            "actions": [
                {
                    "type": "create_index",
                    "collection": "movies",
                    "index": {
                        "keys": [["genres", 1], ["rating", -1], ["_id", 1], ["year", 1]],
                        "name": "esr",
                    },
                },
                {
                    "type": "create_index",
                    "collection": "movies",
                    "index": {"keys": [["director", 1], ["year", 1]]},
                },
                {
                    "type": "create_index",
                    "collection": "comments",
                    "index": {"keys": [["movie_id", 1]]},
                },
                {"type": "drop_index", "collection": "movies", "index_name": "year_1"},
            ]
        }
    )
    before, rep, after, r = run(env, live_workload, plan)
    assert (
        rep.ok and r.violations == [] and r.improvement > 0.9 and 0 < r.score < r.improvement + 0.03
    )
    assert "COLLSCAN" not in after.per_query["q2"].stages
    assert not after.per_query["q1"].has_in_memory_sort
    assert after.result_hashes == before.result_hashes


# --------------------------------------------------------------------------- violations


def test_result_mismatch_filter_change(env, live_workload):
    new = {**Q1, "filter": {"genres": "Drama", "year": {"$gt": 2015}}}
    _, rep, _, r = run(
        env,
        live_workload,
        {"actions": [{"type": "rewrite_query", "query_id": "q1", "new_query": new}]},
    )
    assert r.violations == ["RESULT_MISMATCH", "FORBIDDEN_ACTION"] and r.score == 0


def test_rewrite_that_reorders_ties(env, live_workload):
    """Changing the tie-break direction looks harmless but reorders equal ratings."""
    new = {**Q1, "sort": [["rating", -1], ["_id", -1]], "limit": 10}
    _, rep, _, r = run(
        env,
        live_workload,
        {"actions": [{"type": "rewrite_query", "query_id": "q1", "new_query": new}]},
    )
    # refused up front as a sort change (SORT_CHANGED), and still measured: the hash differs too
    assert not rep.ok and r.violations == ["RESULT_MISMATCH", "FORBIDDEN_ACTION"]


def test_projection_reorder_is_fine(env, live_workload):
    new = {**Q1, "projection": {"rating": 1, "title": 1}}
    _, rep, _, r = run(
        env,
        live_workload,
        {"actions": [{"type": "rewrite_query", "query_id": "q1", "new_query": new}]},
    )
    assert rep.ok and r.violations == []


def test_partial_indexes_cannot_change_results(env, live_workload):
    plan = {
        "actions": [
            {"type": "drop_index", "collection": "movies", "index_name": "year_1"},
            {
                "type": "create_index",
                "collection": "movies",
                "index": {
                    "keys": [["genres", 1], ["rating", -1]],
                    "partialFilterExpression": {"year": {"$gt": 2015}},
                },
            },
            {
                "type": "create_index",
                "collection": "movies",
                "index": {
                    "keys": [["director", 1]],
                    "partialFilterExpression": {"year": {"$gte": 1990}},
                },
            },
        ]
    }
    before, rep, after, r = run(env, live_workload, plan)
    assert (
        rep.ok
        and after.result_hashes == before.result_hashes
        and "RESULT_MISMATCH" not in r.violations
    )


def test_data_mutated(env, live_workload):
    before = measure(env, live_workload)
    env.db.comments.delete_one({"_id": 1})  # a "delete the slow documents" cheat
    after = measure(env, live_workload)
    r = score(
        live_workload,
        before,
        after,
        TuningPlan(actions=[]),
        apply_plan(env, live_workload, {"actions": []}),
        CFG,
    )
    assert "DATA_MUTATED" in r.violations and r.score == 0


def test_index_budget_exceeded_nothing_built(env, live_workload):
    plan = {
        "actions": [
            {
                "type": "create_index",
                "collection": "comments",
                "index": {"keys": [["movie_id", 1]]},
            },
            {"type": "create_index", "collection": "comments", "index": {"keys": [["user", 1]]}},
        ]
    }
    _, rep, _, r = run(env, live_workload, plan)
    assert r.violations == ["INDEX_BUDGET_EXCEEDED"] and names(env, "comments") == ["_id_"]


@pytest.mark.parametrize(
    "index",
    [
        {"keys": [["director", 1]], "unique": True},
        {"keys": [["year", 1]], "expireAfterSeconds": 5},
        {"keys": [["director", 1]], "hidden": True},
        {"keys": [["director", 1]], "sparse": True},
    ],
)
def test_forbidden_index_options_not_built(env, live_workload, index):
    _, rep, _, r = run(
        env,
        live_workload,
        {"actions": [{"type": "create_index", "collection": "movies", "index": index}]},
    )
    assert r.violations == ["FORBIDDEN_ACTION"] and names(env, "movies") == [
        "_id_",
        "year_1",
        "year_1_title_1",
    ]


def test_drop_id_forbidden(env, live_workload):
    _, _, _, r = run(
        env,
        live_workload,
        {"actions": [{"type": "drop_index", "collection": "movies", "index_name": "_id_"}]},
    )
    assert r.violations == ["FORBIDDEN_ACTION"] and "_id_" in names(env, "movies")


def test_out_precompute_never_runs(env, live_workload):
    pipe = [{"$match": {"year": 1999}}, {"$out": "answers"}]
    _, _, _, r = run(
        env,
        live_workload,
        {
            "actions": [
                {
                    "type": "rewrite_query",
                    "query_id": "q3",
                    "new_query": {
                        "id": "q3",
                        "kind": "aggregate",
                        "collection": "movies",
                        "pipeline": pipe,
                    },
                }
            ]
        },
    )
    assert "FORBIDDEN_ACTION" in r.violations and "answers" not in env.db.list_collection_names()


def test_lookup_outside_workload(env, live_workload, mongo):
    env.db["secrets"].insert_one({"_id": 1})
    pipe = [
        {"$match": {"year": 1999}},
        {"$lookup": {"from": "secrets", "localField": "_id", "foreignField": "_id", "as": "c"}},
        {"$project": {"title": 1, "k": {"$size": "$c"}}},
        {"$sort": {"_id": 1}},
    ]
    _, _, _, r = run(
        env,
        live_workload,
        {
            "actions": [
                {
                    "type": "rewrite_query",
                    "query_id": "q3",
                    "new_query": {
                        "id": "q3",
                        "kind": "aggregate",
                        "collection": "movies",
                        "pipeline": pipe,
                    },
                }
            ]
        },
    )
    assert r.violations == ["FORBIDDEN_ACTION"]


def test_apply_error_missing_index(env, live_workload):
    _, rep, _, r = run(
        env,
        live_workload,
        {"actions": [{"type": "drop_index", "collection": "movies", "index_name": "nope_1"}]},
    )
    assert r.violations == ["APPLY_ERROR"]


def test_apply_error_invalid_rewrite(env, live_workload):
    pipe = [{"$match": {"year": 1999}}, {"$project": {}}, {"$sort": {"_id": 1}}]
    _, rep, _, r = run(
        env,
        live_workload,
        {
            "actions": [
                {
                    "type": "rewrite_query",
                    "query_id": "q3",
                    "new_query": {
                        "id": "q3",
                        "kind": "aggregate",
                        "collection": "movies",
                        "pipeline": pipe,
                    },
                }
            ]
        },
    )
    # the rewrite also drops the $lookup (DROPPED_LITERAL), so FORBIDDEN_ACTION rides along
    assert "APPLY_ERROR" in r.violations and rep.effective_queries == live_workload.queries


def test_rewrite_freq_cannot_discount_cost(env, live_workload):
    before = measure(env, live_workload)
    rep = apply_plan(
        env,
        live_workload,
        {
            "actions": [
                {"type": "rewrite_query", "query_id": "q1", "new_query": {**Q1, "freq": 0.0001}}
            ]
        },
    )
    after = measure(env, live_workload, rep.effective_queries)
    assert after.workload_cost == before.workload_cost


def test_env_is_stateful_like_fakeenv(env, live_workload):
    """apply_plan records rewrites on the env; measure(env, w) runs them; reset forgets them."""
    new = {**Q1, "sort": [["rating", -1], ["_id", -1]]}
    before = measure(env, live_workload)
    apply_plan(
        env,
        live_workload,
        {"actions": [{"type": "rewrite_query", "query_id": "q1", "new_query": new}]},
    )
    after = measure(env, live_workload)  # no queries= argument
    assert after.result_hashes["q1"] != before.result_hashes["q1"]
    reset_indexes(env, live_workload)
    assert measure(env, live_workload).result_hashes == before.result_hashes


def test_apply_report_has_fakeenv_fields():
    from ratchet.env.apply import ApplyReport
    from tests.fakes import FakeApplyReport

    assert set(FakeApplyReport.model_fields) <= set(ApplyReport.model_fields)
