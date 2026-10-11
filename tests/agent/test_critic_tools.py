import json

import pytest

from ratchet.agent import tune
from ratchet.agent.critic import pick_critic_model
from ratchet.agent.harness_view import AgentHarness
from ratchet.agent.planner import forbidden_reason
from tests.fakes import FakeEnv, FakeLLM, sample_workload

ESR = {
    "type": "create_index",
    "collection": "movies",
    "index": {"keys": [["genres", 1], ["rating", -1], ["year", 1]]},
}
BAD_ORDER = {
    "type": "create_index",
    "collection": "movies",
    "index": {"keys": [["year", 1], ["genres", 1]]},
}


def plan(*actions, summary="s"):
    return {"actions": list(actions), "summary": summary}


def harness(**over) -> AgentHarness:
    doc = AgentHarness().model_dump(mode="json")
    for path, value in over.items():
        node = doc
        *parents, leaf = path.split("__")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return AgentHarness.model_validate(doc)


class MultiProviderLLM(FakeLLM):
    """FakeLLM that claims every model is callable (so a critic can be picked)."""

    def available(self, model):
        return True


def run(llm, h, **kw):
    env, w = FakeEnv(), sample_workload()
    return tune(w, h, env.prepare_scratch(w), api=env, llm=llm, memory=None, **kw)


# ---------------------------------------------------------------- critic


def test_critic_rejects_and_planner_revises():
    llm = MultiProviderLLM(
        {
            "planner": [plan(BAD_ORDER), plan(ESR)],
            "critic": [
                {"approve": False, "issues": ["range field year before equality genres (ESR)"]},
                {"approve": True, "issues": []},
            ],
        }
    )
    h = harness(critic_policy__enabled=True, critic_policy__max_revisions=1)
    r = run(llm, h)
    keys = [list(map(tuple, a.index.keys)) for a in r.plan.actions]
    assert keys == [[("genres", 1), ("rating", -1), ("year", 1)]]
    crit = [t for t in r.trace if t["step"] == "critic"]
    assert [c["approve"] for c in crit] == [False, True]
    assert "ESR" in crit[0]["issues"][0]
    assert "A reviewer rejected this plan" in llm.prompts("planner")
    assert r.models_used["critic"] != r.models_used["planner"]


def test_critic_revised_plan_is_used_directly():
    llm = MultiProviderLLM(
        {
            "planner": [plan(BAD_ORDER)],
            "critic": [
                {"approve": False, "issues": ["ESR"], "revised_plan": plan(ESR)},
                {"approve": True, "issues": []},
            ],
        }
    )
    r = run(llm, harness(critic_policy__enabled=True))
    assert r.plan.actions[0].index.keys[0] == ("genres", 1)
    assert len([c for c in llm.calls if c["role"] == "planner"]) == 1


def test_critic_cannot_smuggle_forbidden_revision():
    unique = {**ESR, "index": {**ESR["index"], "unique": True}}
    llm = MultiProviderLLM(
        {
            "planner": [plan(ESR), plan(ESR)],
            "critic": [
                {"approve": False, "issues": ["x"], "revised_plan": plan(unique)},
                {"approve": True, "issues": []},
            ],
        }
    )
    r = run(llm, harness(critic_policy__enabled=True))
    assert r.score.violations == []


def test_critic_gives_up_after_max_revisions_and_keeps_plan():
    llm = MultiProviderLLM(
        {
            "planner": [plan(ESR), plan(ESR)],
            "critic": [{"approve": False, "issues": ["meh"]}] * 2,
        }
    )
    r = run(llm, harness(critic_policy__enabled=True, critic_policy__max_revisions=1))
    assert r.plan.actions and any(t["step"] == "critic_unresolved" for t in r.trace)


def test_critic_disabled_in_gen0():
    llm = MultiProviderLLM({"planner": [plan(ESR)]})
    run(llm, harness())
    assert all(c["role"] != "critic" for c in llm.calls)


def test_critic_failure_fails_open():
    llm = MultiProviderLLM({"planner": [plan(ESR)], "critic": ["garbage", "garbage"]})
    r = run(llm, harness(critic_policy__enabled=True))
    assert r.score.score > 0.5
    assert any(t["step"] == "critic" and t.get("error") for t in r.trace)


def test_pick_critic_prefers_other_provider_and_never_the_planner():
    class Avail:
        def __init__(self, ok):
            self.ok = ok

        def available(self, m):
            return m in self.ok

    h = harness(model_routing__critic="openai:gpt-x")
    planner = "anthropic:claude-haiku-5-5"
    assert pick_critic_model(h, planner, Avail({"openai:gpt-x", planner})) == "openai:gpt-x"
    sonnet = "anthropic:claude-sonnet-5-5"
    assert pick_critic_model(h, planner, Avail({sonnet, planner})) == sonnet
    assert (
        pick_critic_model(harness(model_routing__critic=planner), planner, Avail({planner})) is None
    )


# ---------------------------------------------------------------- tools


def test_tool_calls_round_trip_and_self_diagnose():
    llm = FakeLLM(
        {
            "planner": [
                {
                    "tool_calls": [
                        {"name": "explain", "args": {"query_id": "q1"}},
                        {"name": "list_indexes", "args": {"collection": "movies"}},
                        {"name": "findings", "args": {"query_id": "q1"}},
                        {"name": "nope", "args": {}},
                    ]
                },
                {
                    **plan(ESR),
                    "self_diagnose": [{"category": "budget_too_tight", "note": "only 3 slots"}],
                },
            ]
        }
    )
    r = run(llm, harness())
    assert r.score.score > 0.5
    second = json.dumps(llm.calls[1]["messages"])
    assert "Tool results" in second and "COLLSCAN" in second and "title_1" in second
    assert "unknown tool" in second
    tools = [t for t in r.trace if t["step"] == "tool"]
    assert [t["name"] for t in tools] == ["explain", "list_indexes", "findings", "nope"]
    assert r.self_diagnostics == [
        {"category": "budget_too_tight", "note": "only 3 slots", "round": 1}
    ]


def test_tool_steps_are_capped():
    ask = {"tool_calls": [{"name": "explain", "args": {"query_id": "q1"}}]}
    llm = FakeLLM({"planner": [ask, ask, ask, ask, plan()]})
    r = run(llm, harness())
    assert len(llm.calls) == 4  # 3 tool steps, then the 4th reply is taken as the (empty) plan
    assert "No more tool calls" in json.dumps(llm.calls[3]["messages"])
    assert r.plan.actions == []


def test_self_diagnose_tool_validates_category():
    ask = {
        "tool_calls": [
            {"name": "self_diagnose", "args": {"category": "explain_confusing", "note": "?"}},
            {"name": "self_diagnose", "args": {"category": "bogus"}},
        ]
    }
    r = run(FakeLLM({"planner": [ask, plan()]}), harness())
    assert [d["category"] for d in r.self_diagnostics] == ["explain_confusing"]


class Mem:
    def recall(self, shape, policy):
        return ["Anchor regexes with ^ so they can use an index."]


def test_recall_lessons_tool_respects_memory_policy():
    ask = {"tool_calls": [{"name": "recall_lessons", "args": {"shape": "movies | find"}}]}
    env, w = FakeEnv(), sample_workload()
    llm = FakeLLM({"planner": [ask, plan()]})
    r = tune(w, harness(), env.prepare_scratch(w), api=env, llm=llm, memory=Mem())
    assert "disabled by harness" in json.dumps(llm.calls[1]["messages"]) and not r.lessons_used
    llm = FakeLLM({"planner": [ask, plan()]})
    h = harness(memory_policy__inject__mode="on_retry")
    r = tune(w, h, env.prepare_scratch(w), api=env, llm=llm, memory=Mem())
    assert r.lessons_used == ["Anchor regexes with ^ so they can use an index."]


# ---------------------------------------------------------------- forbidden scan


@pytest.mark.parametrize(
    "action",
    [
        {"type": "collMod", "collection": "movies"},
        {**ESR, "index": {**ESR["index"], "unique": True}},
        {**ESR, "index": {**ESR["index"], "expireAfterSeconds": 60}},
        {"type": "drop_index", "collection": "movies", "index_name": "_id_"},
        {
            "type": "rewrite_query",
            "query_id": "q2",
            "new_query": {
                "id": "q2",
                "kind": "aggregate",
                "collection": "comments",
                "pipeline": [{"$out": "answers"}],
            },
        },
    ],
)
def test_forbidden_planner_actions_score_zero_with_code(action):
    h = harness()
    assert forbidden_reason(action, h)
    r = run(FakeLLM({"planner": [plan(ESR, action)]}), h)
    assert "FORBIDDEN_ACTION" in r.score.violations and r.score.score == 0.0
    assert any(t["step"] == "violation" for t in r.trace)
