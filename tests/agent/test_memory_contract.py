import json

from ratchet.agent import tune
from ratchet.agent.harness_view import AgentHarness
from ratchet.llm.cache import MemoryCache
from ratchet.llm.client import LLMClient
from ratchet.llm.types import RawCompletion
from ratchet.memory.models import Lesson
from tests.fakes import FakeEnv, FakeLLM, sample_workload

TEXT = "Put the sort field right after the equality fields."


class Mem:
    """Records every call; only `recall` exists, so any write attempt would raise."""

    def __init__(self):
        self.calls = []

    def recall(self, shape, policy):
        self.calls.append(("recall", shape, policy))
        return [Lesson.model_construct(text=TEXT, similarity=0.9)]


def plan(*a):
    return {"actions": list(a), "summary": "s"}


def test_lessons_used_are_exact_injected_texts_and_policy_is_the_harness_dict():
    doc = AgentHarness().model_dump(mode="json")
    doc["memory_policy"]["inject"]["mode"] = "always"
    doc["memory_policy"]["consolidate"] = {"merge_similarity": 0.9, "max_items": 40}
    env, w, mem = FakeEnv(), sample_workload(), Mem()
    llm = FakeLLM({"planner": [plan()]})
    r = tune(w, doc, env.prepare_scratch(w), api=env, llm=llm, memory=mem)
    assert r.lessons_used == [TEXT]
    assert TEXT in llm.prompts("planner")
    assert {c[0] for c in mem.calls} == {"recall"}
    assert all(c[2] is doc["memory_policy"] for c in mem.calls)


def test_recall_tool_records_exact_texts():
    doc = AgentHarness().model_dump(mode="json")
    doc["memory_policy"]["inject"]["mode"] = "on_retry"
    ask = {"tool_calls": [{"name": "recall_lessons", "args": {"shape": "movies | find"}}]}
    env, w = FakeEnv(), sample_workload()
    r = tune(
        w,
        doc,
        env.prepare_scratch(w),
        api=env,
        llm=FakeLLM({"planner": [ask, plan()]}),
        memory=Mem(),
    )
    assert r.lessons_used == [TEXT]


class Canned:
    name = "anthropic"

    def call(self, model_id, system, messages, **kw):
        return RawCompletion(text=json.dumps({"query_id": "q1", "lesson": TEXT}))


def test_client_accepts_track_a_lesson_writer_call():
    schema = {
        "type": "object",
        "properties": {"query_id": {"type": "string"}, "lesson": {"type": "string"}},
        "required": ["query_id", "lesson"],
    }
    c = LLMClient(cache=MemoryCache(), providers={"anthropic": Canned()})
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "{}"}]
    resp = c.complete("memory", "anthropic:claude-haiku-5-5", msgs, json_schema=schema)
    assert resp.data == {"query_id": "q1", "lesson": TEXT}
