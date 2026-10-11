"""X1 cross-review: Track A attacking Track B's `env/apply.py` and `scoring.py`.

Pure tests (no database). Each `xfail(strict=True)` test is a plan that should be refused or
score no better than doing nothing, and currently is not. They are strict so that when Track B
closes a hole the test XPASSes, fails the run, and tells us to delete the marker.
Notes for Track B: docs/XREVIEW_TRACK_B.md.
"""

import pytest

from ratchet.contracts import Action, IndexDef, Measurement, QuerySpec, TuningPlan
from ratchet.env.apply import ApplyReport, validate_index, validate_plan, validate_rewrite
from ratchet.harness.runner import load_workload_files
from ratchet.scoring import score

WORKLOADS = {w.id: w for w in load_workload_files()}
W = WORKLOADS["mflix_w02"]
Q = {q.id: q for q in W.queries}

hole = pytest.mark.xfail(strict=True, reason="open hole reported to Track B (X1)")


def problems(original: QuerySpec, **changes) -> list[str]:
    hard, soft = validate_rewrite(original, original.model_copy(update=changes), W)
    return [*hard, *soft]


def plan_codes(*actions: Action) -> list[str]:
    _, errors = validate_plan(W, TuningPlan(actions=list(actions)))
    return sorted({e.code for e in errors})


def empty_measurement() -> Measurement:
    return Measurement(
        workload_id=W.id,
        per_query={},
        workload_cost=0.0,
        result_hashes={q.id: "h" for q in W.queries},
        data_checksums={c: "d" for c in W.collections},
        indexes={c: ["_id_"] for c in W.collections},
    )


# ------------------------------------------------------------- 1. precomputed answers


@hole
def test_precomputed_count_built_from_free_numbers_is_refused():
    """q2 is `... | $count: n`. NEW_LITERAL lets 0/1/-1 through, and any integer can be built
    from them, so the answer can be hard-coded without a single "new" constant. The rewrite
    reads one document, returns the identical `{n: <count>}`, and scores ~100% improvement."""
    seven = {"$add": [{"$multiply": [{"$add": [1, 1, 1]}, {"$add": [1, 1]}]}, 1]}
    cheat = [{"$limit": 1}, {"$project": {"_id": 0, "n": seven}}]
    assert problems(Q["q2"], pipeline=cheat)


@hole
def test_constants_hidden_in_object_keys_are_refused():
    """`_atoms` skips dict keys, so a literal object can carry data in its keys."""
    three = {"$size": {"$objectToArray": {"$literal": {"a": 1, "b": 1, "c": 1}}}}
    cheat = [{"$limit": 1}, {"$project": {"_id": 0, "n": three}}]
    assert problems(Q["q2"], pipeline=cheat)


# ------------------------------------------------------------- 2. semantics changed, results equal


@hole
def test_find_rewrite_cannot_drop_the_sort_tie_break():
    """Dropping `_id` from the sort makes the top-10 non-deterministic. On today's data the
    order usually comes out the same, so the hash matches and a cheaper index satisfies it."""
    assert problems(Q["q1"], sort=[Q["q1"].sort[0]])


@hole
def test_find_rewrite_cannot_drop_the_sort_entirely():
    assert problems(Q["q1"], sort=None)


@hole
def test_aggregate_rewrite_cannot_drop_the_sort_tie_break():
    pipe = [dict(s) for s in Q["q5"].pipeline]
    assert pipe[3] == {"$sort": {"n": -1, "_id": 1}}
    pipe[3] = {"$sort": {"n": -1}}
    assert problems(Q["q5"], pipeline=pipe)


@hole
def test_aggregate_rewrite_cannot_drop_a_predicate():
    """A find rewrite must keep its filter; an aggregate rewrite may silently lose a $match
    condition. If that condition happens to be true for every matching document today, the
    result hash is unchanged and the semantic change goes through."""
    pipe = [dict(s) for s in Q["q5"].pipeline]
    assert pipe[0] == {"$match": {"year": 1999, "genres": "Horror"}}
    pipe[0] = {"$match": {"year": 1999}}
    assert problems(Q["q5"], pipeline=pipe)


# ------------------------------------------------------------- 3. index budget


@hole
@pytest.mark.parametrize("field", ["$**", "imdb.$**"])
def test_wildcard_index_is_refused(field):
    """One wildcard index is "index every field" for the price of one budget slot, which is
    exactly the over-indexing the budget (and the budget canary) exists to stop."""
    assert validate_index(IndexDef(keys=[(field, 1)]))


# ------------------------------------------------------------- 4. scorer


@hole
def test_dropping_and_recreating_the_same_index_scores_nothing():
    """Drop an unused baseline index and create it again under a new name: nothing changed,
    but the plan earns the γ credit, which outweighs α·write_rate + β·size on a quiet
    collection. A no-op plan should not score above an empty plan."""
    before = after = empty_measurement()
    after = after.model_copy(update={"indexes": {**after.indexes, "movies": ["_id_", "year_1b"]}})
    apply = ApplyReport(
        ok=True,
        created={"movies": ["year_1b"]},
        dropped={"movies": ["year_1"]},
        index_sizes_mb={"movies": {"year_1b": 0.2}},
    )
    plan = TuningPlan(
        actions=[
            Action(type="drop_index", collection="movies", index_name="year_1"),
            Action(
                type="create_index",
                collection="movies",
                index=IndexDef(keys=[("year", 1)], name="year_1b"),
            ),
        ]
    )
    report = score(W, before, after, plan, apply)
    assert report.violations == [] and report.score <= 0.0


# ------------------------------------------------------------- defended (these hold today)


@pytest.mark.parametrize(
    "extra",
    [
        {"unique": True},
        {"sparse": True},
        {"hidden": True},
        {"expireAfterSeconds": 1},
        {"collation": {"locale": "en", "strength": 1}},
    ],
)
def test_forbidden_index_options_survive_parsing_and_are_refused(extra):
    index = IndexDef.model_validate({"keys": [["title", 1]], **extra})
    assert plan_codes(Action(type="create_index", collection="movies", index=index)) == [
        "FORBIDDEN_ACTION"
    ]


def test_dropping_id_or_everything_is_refused():
    assert plan_codes(Action(type="drop_index", collection="movies", index_name="_id_")) == [
        "FORBIDDEN_ACTION"
    ]
    assert validate_index(IndexDef(keys=[("title", 1)], name="_id_"))


@pytest.mark.parametrize(
    "stage",
    [
        {"$out": "answers"},
        {"$merge": {"into": "answers"}},
        {"$lookup": {"from": "users", "localField": "a", "foreignField": "b", "as": "m"}},
        {"$unionWith": {"coll": "users"}},
        {"$match": {"$where": "true"}},
    ],
)
def test_forbidden_stages_and_foreign_collections_are_refused(stage):
    assert problems(Q["q2"], pipeline=[*Q["q2"].pipeline, stage])


def test_answer_ids_and_changed_filters_are_refused():
    assert problems(Q["q1"], filter={"genres": "Comedy", "year": {"$gte": 2015}})
    assert problems(Q["q1"], limit=5)
    smuggled = [{"$match": {"movie_id": {"$in": [12345, 67890]}}}, {"$count": "n"}]
    assert any("NEW_LITERAL" in p for p in problems(Q["q2"], pipeline=smuggled))


def test_actions_outside_the_workload_are_refused():
    idx = IndexDef(keys=[("x", 1)])
    assert plan_codes(Action(type="create_index", collection="users", index=idx)) == [
        "FORBIDDEN_ACTION"
    ]
    extra = Action.model_validate({"type": "create_index", "collection": "movies", "hint": "x"})
    assert "FORBIDDEN_ACTION" in plan_codes(extra)


def test_every_workload_collection_has_an_index_budget():
    for w in WORKLOADS.values():
        assert set(w.collections) <= set(w.index_budget), w.id
