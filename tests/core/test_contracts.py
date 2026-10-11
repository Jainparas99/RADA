import json

import pytest
from pydantic import ValidationError

from ratchet import contracts as c
from tests.fakes import FakeEnv, sample_workload

SPEC_WORKLOAD = {
    "id": "mflix_w02",
    "split": "train",
    "db": "sample_mflix",
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
            "sort": [["imdb.rating", -1], ["_id", 1]],
            "limit": 10,
            "projection": {"title": 1, "imdb.rating": 1},
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
    "notes": "late $match after $lookup; redundant year_1 index",
}


def roundtrip(model):
    again = type(model).model_validate(json.loads(model.model_dump_json()))
    assert again == model
    return again


def test_all_contract_models_exist():
    for name in [
        "IndexDef",
        "QuerySpec",
        "Workload",
        "ExplainSummary",
        "Finding",
        "Action",
        "TuningPlan",
        "Measurement",
        "ScoreReport",
        "TuneResult",
        "PatchOp",
        "ProposerOutput",
        "GateDecision",
    ]:
        assert hasattr(c, name), name


def test_spec_workload_roundtrips():
    w = roundtrip(c.Workload.model_validate(SPEC_WORKLOAD))
    assert w.queries[0].sort == [("imdb.rating", -1), ("_id", 1)]
    assert w.baseline_indexes["movies"][1].keys == [("year", 1), ("title", 1)]


def test_plan_and_patch_roundtrip():
    plan = c.TuningPlan(
        actions=[
            c.Action(
                type="create_index",
                collection="movies",
                index=c.IndexDef(keys=[("genres", 1), ("imdb.rating", -1)]),
            ),
            c.Action(type="drop_index", collection="movies", index_name="year_1"),
        ],
        summary="ESR index, drop redundant prefix",
    )
    roundtrip(plan)
    roundtrip(
        c.ProposerOutput(
            diagnosis="d",
            hypothesis="h",
            patch=[c.PatchOp(op="add", path="/planner_rules/-", value="Equality, sort, range.")],
            expected_effect="e",
        )
    )
    roundtrip(
        c.GateDecision(
            candidate="h_0001",
            parent="h_0000",
            verdict="REJECT",
            reason_codes=["NO_HOLDOUT_GAIN"],
            metrics={"parent": {"holdout": 0.4}, "candidate": {"holdout": 0.41}},
        )
    )


def test_tune_result_roundtrips_from_fake_env():
    env, w = FakeEnv(), sample_workload()
    scratch = env.prepare_scratch(w)
    before = env.measure(scratch, w)
    plan = c.TuningPlan(actions=[])
    apply = env.apply_plan(scratch, w, plan)
    after = env.measure(scratch, w)
    result = c.TuneResult(
        workload_id=w.id,
        split=w.split,
        harness_version="h_0000",
        plan=plan,
        before=before,
        after=after,
        score=env.score(w, before, after, plan, apply),
        findings=env.run_detectors(scratch, w, before),
        tokens=0,
        rounds=1,
        trace=[{"step": "measure"}],
    )
    assert roundtrip(result).score.score == 0.0


def test_invalid_literals_rejected():
    with pytest.raises(ValidationError):
        c.Action(type="collMod")
    with pytest.raises(ValidationError):
        c.PatchOp(op="move", path="/x")
    with pytest.raises(ValidationError):
        c.Workload.model_validate({**SPEC_WORKLOAD, "split": "test"})


def test_unknown_plan_keys_survive_parsing_so_the_env_can_forbid_them():
    plan = c.TuningPlan.model_validate(
        {
            "actions": [
                {
                    "type": "create_index",
                    "collection": "movies",
                    "index": {"keys": [["title", 1]], "unique": True},
                    "hint": "x",
                }
            ]
        }
    )
    assert plan.actions[0].index.model_extra == {"unique": True}
    assert plan.actions[0].model_extra == {"hint": "x"}
    assert roundtrip(plan).model_dump()["actions"][0]["index"]["unique"] is True
    env, w = FakeEnv(), sample_workload()
    report = env.apply_plan(env.prepare_scratch(w), w, plan)
    assert report.violations == ["FORBIDDEN_ACTION"] and report.applied == []
