"""Planner tools (SPEC §7) and the planner's reply format.

Tools are requested in JSON (provider-agnostic, works with every model and with FakeLLM):

    {"tool_calls": [{"name": "explain", "args": {"query_id": "q1"}}]}

The loop runs them and sends the results back; the planner then calls more tools or replies with
the plan (`actions` + `summary`). Any reply may also carry `self_diagnose` notes (Raindrop pattern).
Tools are read-only: nothing here can change indexes, queries or data.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel

from ratchet.agent.context import brief
from ratchet.agent.harness_view import AgentHarness
from ratchet.contracts import Action, Finding, Measurement, Workload

SELF_DIAGNOSE_CATEGORIES = (
    "explain_confusing",
    "conflicting_queries",
    "budget_too_tight",
    "cannot_fix_without_semantics_change",
    "tool_error",
)
MAX_TOOL_STEPS = 3
MAX_CALLS_PER_STEP = 4
MAX_RESULT_CHARS = 3000
MAX_SAMPLE_DOCS = 5


class ToolCall(BaseModel):
    name: str
    args: dict = {}


class SelfDiagnosis(BaseModel):
    category: Literal[
        "explain_confusing",
        "conflicting_queries",
        "budget_too_tight",
        "cannot_fix_without_semantics_change",
        "tool_error",
    ]
    note: str = ""


class PlannerReply(BaseModel):
    """Either tool requests or a plan (a plain `TuningPlan` reply also validates)."""

    actions: list[Action] | None = None
    summary: str = ""
    tool_calls: list[ToolCall] | None = None
    self_diagnose: list[SelfDiagnosis] | None = None


TOOL_SPECS: list[dict[str, Any]] = [
    {"name": "explain", "args": {"query_id": "str"}, "does": "explain() summary of one query"},
    {"name": "list_indexes", "args": {"collection": "str"}, "does": "current index names"},
    {"name": "collection_stats", "args": {"collection": "str"}, "does": "count, size, indexes"},
    {"name": "sample_docs", "args": {"collection": "str", "n": "int<=5"}, "does": "sample docs"},
    {"name": "findings", "args": {"query_id": "str"}, "does": "detector findings for a query"},
    {"name": "recall_lessons", "args": {"shape": "str"}, "does": "lessons for a query shape"},
    {
        "name": "self_diagnose",
        "args": {"category": "|".join(SELF_DIAGNOSE_CATEGORIES), "note": "str"},
        "does": "report why you are stuck (recorded for the harness proposer)",
    },
]


def tools_prompt() -> str:
    lines = [f"- {t['name']}({json.dumps(t['args'])}): {t['does']}" for t in TOOL_SPECS]
    return (
        "Tools (optional, read-only). To use them reply ONLY with "
        '{"tool_calls":[{"name":...,"args":{...}}]} (at most '
        f"{MAX_CALLS_PER_STEP} per reply, {MAX_TOOL_STEPS} replies); you get the results back.\n"
        + "\n".join(lines)
        + '\nAny reply may include "self_diagnose":[{"category":...,"note":...}].'
    )


class ToolBox:
    def __init__(
        self,
        api: Any,
        env: Any,
        workload: Workload,
        measurement: Measurement,
        findings: list[Finding],
        harness: AgentHarness,
        memory: Any = None,
        memory_policy: Any = None,
    ) -> None:
        self.api, self.env, self.workload = api, env, workload
        self.measurement = measurement
        self.findings = findings
        self.h = harness
        self.memory = memory
        self.memory_policy = memory_policy if memory_policy is not None else harness.memory_policy
        self.self_diagnostics: list[dict] = []
        self.lessons_seen: list[str] = []
        self.calls: list[dict] = []

    # -- dispatch -------------------------------------------------------------------

    def run(self, call: ToolCall, round_no: int = 0) -> dict:
        fn = getattr(self, f"t_{call.name}", None)
        if fn is None:
            result: Any = {"error": f"unknown tool {call.name!r}"}
        else:
            try:
                result = fn(**call.args)
            except TypeError as e:
                result = {"error": f"bad arguments for {call.name}: {e}"}
            except Exception as e:  # a broken tool must not kill the run
                result = {"error": f"{call.name} failed: {type(e).__name__}"}
                self.record_diagnosis("tool_error", f"{call.name}: {type(e).__name__}", round_no)
        text = json.dumps(result, default=str)
        if len(text) > MAX_RESULT_CHARS:
            result = {"truncated": text[:MAX_RESULT_CHARS]}
        out = {"name": call.name, "args": call.args, "result": result}
        self.calls.append({**out, "round": round_no})
        return out

    def record_diagnosis(self, category: str, note: str, round_no: int = 0) -> dict:
        if category not in SELF_DIAGNOSE_CATEGORIES:
            return {"error": f"category must be one of {list(SELF_DIAGNOSE_CATEGORIES)}"}
        d = {"category": category, "note": str(note)[:500], "round": round_no}
        self.self_diagnostics.append(d)
        return {"recorded": True}

    # -- tools ------------------------------------------------------------------------

    def _coll(self, collection: str) -> str:
        if collection not in self.workload.collections:
            raise ValueError(f"unknown collection {collection!r}")
        return collection

    def t_explain(self, query_id: str) -> dict:
        s = self.measurement.per_query.get(query_id)
        if s is None:
            return {"error": f"unknown query {query_id!r}"}
        out: dict[str, Any] = brief(s)
        if self.h.context_policy.include_full_explain and s.raw:
            out["raw"] = s.raw
        return out

    def t_list_indexes(self, collection: str) -> dict:
        try:
            self._coll(collection)
        except ValueError as e:
            return {"error": str(e)}
        return {"collection": collection, "indexes": self.measurement.indexes.get(collection, [])}

    def t_collection_stats(self, collection: str) -> dict:
        try:
            self._coll(collection)
        except ValueError as e:
            return {"error": str(e)}
        fn = getattr(self.api, "collection_stats", None)
        if not callable(fn):
            return {"error": "collection_stats not available in this environment"}
        return {"collection": collection, "stats": fn(self.env, collection)}

    def t_sample_docs(self, collection: str, n: int = 1) -> dict:
        try:
            self._coll(collection)
        except ValueError as e:
            return {"error": str(e)}
        fn = getattr(self.api, "sample_docs", None)
        if not callable(fn):
            return {"error": "sample_docs not available in this environment"}
        n = max(1, min(int(n), MAX_SAMPLE_DOCS))
        return {"collection": collection, "docs": fn(self.env, collection, n)}

    def t_findings(self, query_id: str) -> dict:
        return {
            "query_id": query_id,
            "findings": [
                f.model_dump(mode="json") for f in self.findings if f.query_id == query_id
            ],
        }

    def t_recall_lessons(self, shape: str) -> dict:
        if self.memory is None or not callable(getattr(self.memory, "recall", None)):
            return {"lessons": [], "note": "memory not available"}
        if self.h.memory_policy.inject.mode == "never":
            return {"lessons": [], "note": "memory injection disabled by harness"}
        from ratchet.agent.loop import lesson_text

        lessons = self.memory.recall(shape, self.memory_policy) or []
        texts = [lesson_text(x) for x in lessons][: max(0, self.h.memory_policy.retrieve.k)]
        self.lessons_seen += [t for t in texts if t not in self.lessons_seen]
        return {"lessons": texts}

    def t_self_diagnose(self, category: str, note: str = "") -> dict:
        return self.record_diagnosis(category, note)
