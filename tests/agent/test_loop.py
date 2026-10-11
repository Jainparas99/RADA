import json

from ratchet.agent import tune
from ratchet.agent.context import build_context
from ratchet.agent.harness_view import AgentHarness, view
from ratchet.config import get_settings
from tests.fakes import FakeEnv, FakeLLM, sample_workload

ESR = {
    "type": "create_index",
    "collection": "movies",
    "index": {"keys": [["genres", 1], ["rating", -1], ["year", 1]]},
    "rationale": "ESR",
}
YEAR = {"type": "create_index", "collection": "movies", "index": {"keys": [["year", 1]]}}


def harness(**over) -> AgentHarness:
    doc = AgentHarness().model_dump(mode="json")
    for path, value in over.items():
        node = doc
        *parents, leaf = path.split("__")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return AgentHarness.model_validate(doc)


def run(llm, h=None, memory=None, **kw):
    env, w = FakeEnv(), sample_workload()
    scratch = env.prepare_scratch(w)
    return tune(w, h or harness(), scratch, api=env, llm=llm, memory=memory, **kw)


def plan(*actions, summary="s"):
    return {"actions": list(actions), "summary": summary}


def moved_q2():
    q2 = sample_workload().queries[1]
    p = q2.pipeline
    return q2.model_copy(update={"pipeline": [p[2], *p[:2], p[3]]}).model_dump(mode="json")


# ---------------------------------------------------------------- end to end on FakeEnv


def test_gen0_runs_end_to_end_and_improves():
    llm = FakeLLM({"planner": [plan(ESR)]})
    r = run(llm, harness_version="h_0000")
    assert r.score.violations == [] and r.score.score > 0.5
    assert r.after.per_query["q1"].stages == ["IXSCAN", "FETCH"]
    assert r.rounds == 1 and r.harness_version == "h_0000"
    assert r.tokens > 0
    assert r.models_used["planner"] == AgentHarness().model_routing.planner
    steps = [t["step"] for t in r.trace]
    assert steps[:3] == ["start", "measure_before", "llm"]
    assert {"plan", "apply_measure"} <= set(steps)
    llm_entry = next(t for t in r.trace if t["step"] == "llm")
    assert llm_entry["role"] == "planner" and llm_entry["messages"][0]["role"] == "system"
    json.dumps(r.model_dump(mode="json"))  # trace is storable


def test_invalid_plan_gets_one_repair():
    llm = FakeLLM({"planner": ["this is not json", plan(ESR)]})
    r = run(llm)
    assert r.score.score > 0.5
    assert len(llm.calls) == 2
    assert "That reply was invalid" in llm.prompts("planner")


def test_invalid_twice_gives_empty_plan_not_exception():
    llm = FakeLLM({"planner": ["nope", {"actions": "not a list"}]})
    r = run(llm)
    assert r.plan.actions == [] and r.score.score == 0.0 and r.score.violations == []
    p = next(t for t in r.trace if t["step"] == "plan")
    assert p["ok"] is False and p["error"]


def test_retry_round_sees_previous_round_and_accumulates():
    rewrite = {"type": "rewrite_query", "query_id": "q2", "new_query": moved_q2()}
    llm = FakeLLM({"planner": [plan(YEAR), plan(rewrite), plan()]})
    h = harness(
        retry_policy__max_rounds=3,
        retry_policy__retry_if_improvement_below=0.99,
        retry_policy__show_explain_diff=True,
    )
    r = run(llm, h)
    assert r.rounds == 3  # third round proposes nothing more
    assert [a.type for a in r.plan.actions] == ["create_index", "rewrite_query"]
    second = json.dumps(llm.calls[1]["messages"])
    assert "Previous round" in second and "explain_diff_before_after" in second
    imp = [t["score"]["improvement"] for t in r.trace if t["step"] == "apply_measure"]
    assert imp[1] > imp[0] and r.score.improvement == imp[1]


def test_violation_stops_retries():
    many = [
        {"type": "create_index", "collection": "movies", "index": {"keys": [[f, 1]]}}
        for f in "abcd"
    ]
    llm = FakeLLM({"planner": [plan(*many), plan(ESR)]})
    r = run(llm, harness(retry_policy__max_rounds=3))
    assert r.rounds == 1 and "INDEX_BUDGET_EXCEEDED" in r.score.violations and r.score.score == 0
    assert any(t.get("reason") == "violation" for t in r.trace)


def test_no_improvement_stops():
    useless = {"type": "create_index", "collection": "comments", "index": {"keys": [["x", 1]]}}
    llm = FakeLLM({"planner": [plan(YEAR), plan(useless), plan(ESR)]})
    h = harness(retry_policy__max_rounds=3, retry_policy__retry_if_improvement_below=0.99)
    r = run(llm, h)
    assert r.rounds == 2 and len(llm.calls) == 2
    assert any(t.get("reason") == "no improvement" for t in r.trace)


def test_empty_plan_is_a_clean_stop():
    llm = FakeLLM({"planner": [plan()]})
    r = run(llm, harness(retry_policy__max_rounds=3))
    assert r.rounds == 1 and r.score.score == 0.0 and r.after == r.before


def test_action_outside_allowed_actions_is_a_forbidden_violation():
    drop = {"type": "drop_index", "collection": "movies", "index_name": "title_1"}
    llm = FakeLLM({"planner": [plan(ESR, drop)]})
    r = run(llm, harness(protected__allowed_actions=["create_index"]))
    assert [a.type for a in r.plan.actions] == ["create_index"]  # stripped, rest applied
    p = next(t for t in r.trace if t["step"] == "plan")
    assert p["forbidden"][0]["action"]["type"] == "drop_index"
    assert "FORBIDDEN_ACTION" in r.score.violations and r.score.score == 0.0
    assert "Allowed action types for this run: create_index." in llm.prompts("planner")


# ---------------------------------------------------------------- harness knobs


def test_context_policy_controls_what_planner_sees():
    llm = FakeLLM({"planner": [plan()]})
    run(llm)  # gen 0: no findings
    assert "Detector findings" not in llm.prompts()
    llm2 = FakeLLM({"planner": [plan()]})
    run(llm2, harness(context_policy__include_detector_findings=True))
    assert "COLLSCAN_LARGE" in llm2.prompts()


def test_context_is_capped_at_max_chars():
    env, w = FakeEnv(), sample_workload()
    s = env.prepare_scratch(w)
    m = env.measure(s, w)
    text = build_context(w, harness(context_policy__max_chars=700), m, [])
    assert len(text) <= 700 and text.endswith("max_chars]")


def test_planner_rules_reach_the_prompt():
    llm = FakeLLM({"planner": [plan()]})
    run(llm, harness(planner_rules=["Equality fields first, then sort, then range."]))
    assert "Equality fields first" in llm.prompts("planner")


class FakeMemory:
    def __init__(self):
        self.calls = []

    def recall(self, shape, policy):
        self.calls.append(shape)
        return [{"text": "Put the sort field right after equality fields."}]


def test_memory_injected_only_on_retry_when_policy_says_so():
    mem = FakeMemory()
    llm = FakeLLM({"planner": [plan(YEAR), plan()]})
    h = harness(
        memory_policy__inject__mode="on_retry",
        retry_policy__max_rounds=2,
        retry_policy__retry_if_improvement_below=0.99,
    )
    r = run(llm, h, memory=mem)
    first, second = (json.dumps(c["messages"]) for c in llm.calls)
    assert "sort field right after" not in first and "sort field right after" in second
    assert r.lessons_used == ["Put the sort field right after equality fields."]
    assert mem.calls  # shapes, not raw values
    assert all("Drama" not in s for s in mem.calls)


def test_memory_never_by_default():
    mem = FakeMemory()
    run(FakeLLM({"planner": [plan()]}), memory=mem)
    assert mem.calls == []


def test_accepts_track_a_style_doc_with_extra_fields():
    doc = AgentHarness().model_dump(mode="json")
    doc["memory_policy"]["write"] = {"enabled": True, "from": ["train_improved"]}
    doc["protected"]["scorer_ref"] = {"version": "1", "source_hash": "sha256:x"}
    h = view(doc)
    assert h.retry_policy.max_rounds == 1
    r = run(FakeLLM({"planner": [plan(ESR)]}), doc)
    assert r.score.score > 0.5


def test_unavailable_planner_model_falls_back_to_config_default():
    class NoKeys(FakeLLM):
        def available(self, model):
            return model == get_settings().planner_model

    llm = NoKeys({"planner": [plan()]})
    r = run(llm, harness(model_routing__planner="openai:gpt-x"))
    assert llm.calls[0]["model"] == get_settings().planner_model
    assert r.models_used["planner"] == get_settings().planner_model
    assert any(t["step"] == "routing" for t in r.trace)
