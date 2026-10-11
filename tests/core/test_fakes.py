import pytest

from ratchet.contracts import Action, IndexDef, TuningPlan
from tests.fakes import FakeEnv, FakeLLM, sample_workload


def run(plan: TuningPlan, mutate: bool = False):
    env, w = FakeEnv(), sample_workload()
    scratch = env.prepare_scratch(w)
    before = env.measure(scratch, w)
    apply = env.apply_plan(scratch, w, plan)
    if mutate:
        env.mutate_data(scratch, "movies")
    after = env.measure(scratch, w)
    return env.score(w, before, after, plan, apply), before, after


def create(keys, coll="movies"):
    return Action(type="create_index", collection=coll, index=IndexDef(keys=keys))


def test_baseline_has_findings_and_empty_plan_scores_zero():
    env, w = FakeEnv(), sample_workload()
    scratch = env.prepare_scratch(w)
    m = env.measure(scratch, w)
    detectors = {f.detector for f in env.run_detectors(scratch, w, m)}
    assert {"COLLSCAN_LARGE", "IN_MEMORY_SORT", "LATE_MATCH"} <= detectors
    report, _, _ = run(TuningPlan(actions=[]))
    assert report.score == 0.0 and report.violations == []


def test_good_index_improves_score_without_violations():
    report, before, after = run(
        TuningPlan(actions=[create([("genres", 1), ("rating", -1), ("year", 1)])])
    )
    assert report.violations == []
    assert after.workload_cost < before.workload_cost
    assert report.score > 0.5
    assert after.per_query["q1"].stages == ["IXSCAN", "FETCH"]


def test_reordering_rewrite_keeps_results_and_changed_filter_does_not():
    w = sample_workload()
    q2 = w.queries[1]
    moved = q2.model_copy(update={"pipeline": [q2.pipeline[2], *q2.pipeline[:2], q2.pipeline[3]]})
    ok, _, _ = run(
        TuningPlan(actions=[Action(type="rewrite_query", query_id="q2", new_query=moved)])
    )
    assert ok.violations == [] and ok.improvement > 0

    q1 = w.queries[0]
    cheat = q1.model_copy(update={"filter": {"genres": "Drama", "year": {"$gt": 2015}}})
    bad, _, _ = run(
        TuningPlan(actions=[Action(type="rewrite_query", query_id="q1", new_query=cheat)])
    )
    assert bad.violations == ["RESULT_MISMATCH"] and bad.score == 0.0


@pytest.mark.parametrize(
    ("plan", "mutate", "code"),
    [
        (TuningPlan(actions=[create([(f, 1)]) for f in "abcd"]), False, "INDEX_BUDGET_EXCEEDED"),
        (
            TuningPlan(actions=[Action(type="drop_index", collection="movies", index_name="_id_")]),
            False,
            "FORBIDDEN_ACTION",
        ),
        (
            TuningPlan(actions=[Action(type="drop_index", collection="movies", index_name="nope")]),
            False,
            "APPLY_ERROR",
        ),
        (TuningPlan(actions=[]), True, "DATA_MUTATED"),
    ],
)
def test_inner_violations(plan, mutate, code):
    report, _, _ = run(plan, mutate=mutate)
    assert code in report.violations
    assert report.score == 0.0


def test_out_stage_is_forbidden():
    q2 = sample_workload().queries[1]
    cheat = q2.model_copy(update={"pipeline": [*q2.pipeline, {"$out": "answers"}]})
    report, _, _ = run(
        TuningPlan(actions=[Action(type="rewrite_query", query_id="q2", new_query=cheat)])
    )
    assert "FORBIDDEN_ACTION" in report.violations


def test_query_shape_strips_values():
    shape = FakeEnv().query_shape(sample_workload().queries[0])
    assert shape == (
        "movies | filter {genres: EQ, year: RANGE} | sort {rating: DESC, _id: ASC}"
        " | proj {rating, title}"
    )
    assert "Drama" not in shape and "2000" not in shape


def test_fake_llm_scripts_by_role_and_records_calls():
    llm = FakeLLM({"planner": [{"actions": [], "summary": "noop"}]}, default={"approve": True})
    r = llm.complete("planner", "anthropic:x", [{"role": "user", "content": "plan please"}])
    assert TuningPlan.model_validate(r.data).summary == "noop"
    assert llm.complete("critic", "anthropic:y", []).data == {"approve": True}
    assert [c["role"] for c in llm.calls] == ["planner", "critic"]
    assert "plan please" in llm.prompts("planner")
    with pytest.raises(AssertionError):
        FakeLLM().complete("planner", "m", [])


def test_fake_embed_is_deterministic_and_similarity_aware():
    llm = FakeLLM()
    a, b, c = llm.embed(
        [
            "movies | filter {genres: EQ, year: RANGE}",
            "movies | filter {genres: EQ, year: RANGE} | sort {rating: DESC}",
            "orders | pipeline $lookup > $unwind",
        ]
    )

    def dot(x, y):
        return sum(i * j for i, j in zip(x, y, strict=True))

    assert llm.embed(["x y"]) == llm.embed(["x y"])
    assert dot(a, b) > dot(a, c)
