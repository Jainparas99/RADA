"""Plan validation (SPEC §5.5), pure: no database."""

import pytest

from ratchet.contracts import TuningPlan
from ratchet.env.apply import effective_queries, validate_plan


def plan(*actions):
    return {"actions": list(actions)}


def codes(errors, hard=None):
    return [e.code for e in errors if hard is None or e.hard == hard]


def create(coll="movies", **index):
    return {"type": "create_index", "collection": coll, "index": {"keys": [["genres", 1]], **index}}


def rewrite(qid, **new):
    return {"type": "rewrite_query", "query_id": qid, "new_query": new}


Q1 = {
    "id": "q1",
    "kind": "find",
    "collection": "movies",
    "filter": {"genres": "Drama", "year": {"$gt": 2000}},
    "sort": [["rating", -1], ["_id", 1]],
    "limit": 10,
    "projection": {"title": 1},
}
Q2_PIPE = [
    {"$lookup": {"from": "movies", "localField": "movie_id", "foreignField": "_id", "as": "m"}},
    {"$unwind": "$m"},
    {"$match": {"m.year": 2012}},
    {"$count": "n"},
]


def test_valid_plan_passes(small_workload):
    p = plan(
        create(),
        {"type": "drop_index", "collection": "movies", "index_name": "year_1"},
        rewrite("q1", **{**Q1, "projection": {"title": 1}, "sort": [["rating", -1], ["_id", 1]]}),
        create("movies", keys=[["year", 1]], partialFilterExpression={"year": {"$gt": 2000}}),
    )
    parsed, errors = validate_plan(small_workload, p)
    assert parsed is not None and errors == []


@pytest.mark.parametrize(
    "opt,val",
    [
        ("unique", True),
        ("sparse", True),
        ("hidden", True),
        ("expireAfterSeconds", 60),
        ("collation", {"locale": "en"}),
    ],
)
def test_forbidden_index_options_in_raw_json(small_workload, opt, val):
    _, errors = validate_plan(small_workload, plan(create(**{opt: val})))
    assert codes(errors) == ["FORBIDDEN_ACTION"] and opt in errors[0].detail


def test_unknown_action_types_forbidden(small_workload):
    for t in ("run_command", "$out", "insert", "drop_collection", "hint"):
        _, errors = validate_plan(small_workload, plan({"type": t, "collection": "movies"}))
        assert "FORBIDDEN_ACTION" in codes(errors), t


def test_unknown_action_field_forbidden(small_workload):
    _, errors = validate_plan(small_workload, plan({**create(), "hint": {"year": 1}}))
    assert codes(errors) == ["FORBIDDEN_ACTION"]


def test_drop_id_forbidden(small_workload):
    _, errors = validate_plan(
        small_workload, plan({"type": "drop_index", "collection": "movies", "index_name": "_id_"})
    )
    assert codes(errors) == ["FORBIDDEN_ACTION"]


def test_collection_outside_workload(small_workload):
    _, errors = validate_plan(small_workload, plan(create("users")))
    assert codes(errors) == ["FORBIDDEN_ACTION"]


@pytest.mark.parametrize("keys", [[["a", 0]], [["a", "wildcard"]], [["$where", 1]], []])
def test_bad_index_keys(small_workload, keys):
    _, errors = validate_plan(
        small_workload,
        plan({"type": "create_index", "collection": "movies", "index": {"keys": keys}}),
    )
    assert "FORBIDDEN_ACTION" in codes(errors)


@pytest.mark.parametrize(
    "stage",
    [
        {"$out": "x"},
        {"$merge": {"into": "x"}},
        {"$documents": [{"n": 1}]},
        {"$sample": {"size": 1}},
        {"$collStats": {}},
        {"$currentOp": {}},
    ],
)
def test_forbidden_stages_hard(small_workload, stage):
    _, errors = validate_plan(
        small_workload,
        plan(
            rewrite(
                "q2", id="q2", kind="aggregate", collection="comments", pipeline=[stage, *Q2_PIPE]
            )
        ),
    )
    assert "FORBIDDEN_ACTION" in codes(errors, hard=True)


@pytest.mark.parametrize(
    "expr",
    [
        {"$where": "true"},
        {"$expr": {"$function": {"body": "", "args": [], "lang": "js"}}},
        {"$expr": {"$lt": [{"$rand": {}}, 1]}},
        {"$expr": {"$lt": ["$date", "$$NOW"]}},
    ],
)
def test_forbidden_operators_and_vars(small_workload, expr):
    _, errors = validate_plan(
        small_workload,
        plan(
            rewrite(
                "q2",
                id="q2",
                kind="aggregate",
                collection="comments",
                pipeline=[{"$match": expr}, *Q2_PIPE],
            )
        ),
    )
    assert "FORBIDDEN_ACTION" in codes(errors, hard=True)


def test_lookup_outside_workload_forbidden(small_workload):
    pipe = [
        {"$lookup": {"from": "users", "localField": "u", "foreignField": "_id", "as": "m"}},
        *Q2_PIPE[1:],
    ]
    _, errors = validate_plan(
        small_workload,
        plan(rewrite("q2", id="q2", kind="aggregate", collection="comments", pipeline=pipe)),
    )
    assert any("outside the workload" in e.detail for e in errors)
    pipe = [
        {
            "$lookup": {
                "from": {"db": "other", "coll": "movies"},
                "localField": "u",
                "foreignField": "_id",
                "as": "m",
            }
        },
        *Q2_PIPE[1:],
    ]
    _, errors = validate_plan(
        small_workload,
        plan(rewrite("q2", id="q2", kind="aggregate", collection="comments", pipeline=pipe)),
    )
    assert any("another database" in e.detail for e in errors)


def test_rewrite_structure_is_hard(small_workload):
    for new in (
        {**Q1, "collection": "comments"},
        {**Q1, "kind": "aggregate", "pipeline": [{"$match": {}}]},
        {**Q1, "id": "other"},
    ):
        _, errors = validate_plan(small_workload, plan(rewrite("q1", **new)))
        assert "FORBIDDEN_ACTION" in codes(errors, hard=True)


def test_filter_change_is_soft_so_result_mismatch_shows(small_workload):
    """SPEC §14: a rewrite that changes the filter must surface as RESULT_MISMATCH, so it is
    measured (soft FORBIDDEN_ACTION), not blocked."""
    new = {**Q1, "filter": {"genres": "Drama", "year": {"$gt": 2015}}}
    parsed, errors = validate_plan(small_workload, plan(rewrite("q1", **new)))
    assert parsed is not None and errors and all(not e.hard for e in errors)
    assert any("changes filter" in e.detail for e in errors) and any(
        "NEW_LITERAL" in e.detail for e in errors
    )


def test_precomputed_answers_blocked_by_new_literal(small_workload):
    """Same results via a smuggled list of answer ids -> NEW_LITERAL."""
    pipe = [{"$match": {"_id": {"$in": [101, 202, 303]}}}, *Q2_PIPE]
    _, errors = validate_plan(
        small_workload,
        plan(rewrite("q2", id="q2", kind="aggregate", collection="comments", pipeline=pipe)),
    )
    assert any("NEW_LITERAL" in e.detail for e in errors)


def test_moving_match_across_lookup_is_not_a_new_literal(small_workload):
    pipe = [
        {
            "$lookup": {
                "from": "movies",
                "localField": "movie_id",
                "foreignField": "_id",
                "pipeline": [{"$match": {"year": 2012}}],
                "as": "m",
            }
        },
        {"$unwind": "$m"},
        {"$count": "n"},
    ]
    _, errors = validate_plan(
        small_workload,
        plan(rewrite("q2", id="q2", kind="aggregate", collection="comments", pipeline=pipe)),
    )
    assert errors == []


def test_limit_change_soft(small_workload):
    _, errors = validate_plan(small_workload, plan(rewrite("q1", **{**Q1, "limit": 5})))
    assert errors and all(not e.hard for e in errors)


def test_effective_queries_keep_workload_freq(small_workload):
    parsed, _ = validate_plan(small_workload, plan(rewrite("q1", **{**Q1, "freq": 0.001})))
    eff = effective_queries(small_workload, parsed)
    assert [q.id for q in eff] == ["q1", "q2"] and eff[0].freq == 50


def test_typed_plan_with_extra_fields_detected(small_workload):
    """If Track A's IndexDef allows extras, a typed plan carrying unique=True is still caught."""
    from pydantic import ConfigDict

    from ratchet.contracts import IndexDef

    class LooseIndex(IndexDef):
        model_config = ConfigDict(extra="allow")

    tp = TuningPlan.model_validate(plan(create()))
    tp.actions[0].index = LooseIndex(keys=[("a", 1)], unique=True)
    _, errors = validate_plan(small_workload, tp)
    assert codes(errors) == ["FORBIDDEN_ACTION"]


def test_malformed_plan_is_apply_error(small_workload):
    _, errors = validate_plan(small_workload, {"nope": 1})
    assert codes(errors) == ["APPLY_ERROR"]
    _, errors = validate_plan(
        small_workload, plan({"type": "create_index", "collection": "movies"})
    )
    assert codes(errors) == ["APPLY_ERROR"]
