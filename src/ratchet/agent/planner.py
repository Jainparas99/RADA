"""The planner: harness rules + context (+ tools) → a strict `TuningPlan` (SPEC §7).

The reply is validated as `PlannerReply` (a plan, or tool requests). Invalid → one repair turn →
else an empty plan, never an exception. Before validation the raw JSON is scanned for forbidden
actions (unknown action types, types outside `protected.allowed_actions`, forbidden index options
such as `unique`/`sparse`/TTL/`hidden`): they are removed and reported as `FORBIDDEN_ACTION`, which
the loop turns into a recorded violation with score 0. Contract models silently drop unknown keys,
so this scan is what keeps e.g. a `unique: true` index from being laundered into a plain one.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from ratchet.agent.harness_view import ALL_ACTIONS, AgentHarness
from ratchet.agent.tools import (
    MAX_CALLS_PER_STEP,
    MAX_TOOL_STEPS,
    PlannerReply,
    ToolBox,
    tools_prompt,
)
from ratchet.contracts import TuningPlan
from ratchet.llm.jsonutil import JSONExtractError, extract_json
from ratchet.llm.types import LLMError

ROLE = "planner"
FORBIDDEN_INDEX_OPTIONS = ("unique", "sparse", "expireAfterSeconds", "hidden")
FORBIDDEN_STAGES = ("$out", "$merge")

SYSTEM = """You tune MongoDB query workloads. You get a workload (queries with frequencies,
per-collection index budgets and write rates), explain() statistics and, depending on the harness,
existing indexes, detector findings and lessons. You reply with a tuning plan.

Allowed actions (nothing else exists):
- create_index: {"type":"create_index","collection":C,"index":{"keys":[[field,1|-1],...],
  "name":optional,"partialFilterExpression":optional},"rationale":...}
  No unique, sparse, TTL or hidden options.
- drop_index: {"type":"drop_index","collection":C,"index_name":NAME,"rationale":...} (never _id_)
- rewrite_query: {"type":"rewrite_query","query_id":Q,"new_query":<full query spec, same id,
  kind and collection>,"rationale":...} Only reorders pipeline stages or changes projection/sort-key
  order. It MUST return exactly the same documents in the same order. No $out, $merge, hint.

Hard constraints: never change any query result; never change data; the number of non-_id
indexes on a collection after your plan must stay within its index_budget (count existing ones);
every index costs writes, so add only indexes that the workload's queries will use.

Final reply: one JSON object {"actions":[...],"summary":"one or two sentences"}.
An empty "actions" list is a valid plan."""


def system_prompt(h: AgentHarness, with_tools: bool = True) -> str:
    parts = [SYSTEM]
    if h.planner_rules:
        parts.append("Harness rules:\n" + "\n".join(f"- {r}" for r in h.planner_rules))
    parts.append(f"Allowed action types for this run: {', '.join(h.protected.allowed_actions)}.")
    if with_tools:
        parts.append(tools_prompt())
    return "\n\n".join(parts)


def response_text(resp: Any) -> str:
    t = getattr(resp, "text", None)
    return t if isinstance(t, str) else json.dumps(getattr(resp, "data", None))


def response_data(resp: Any) -> Any:
    data = getattr(resp, "data", None)
    if data is None:
        data = extract_json(response_text(resp))
    return data


def forbidden_reason(action: Any, h: AgentHarness) -> str | None:
    """Why a raw (unvalidated) action is forbidden, or None."""
    if not isinstance(action, dict):
        return None
    t = action.get("type")
    if t not in ALL_ACTIONS:
        return f"unknown action type {t!r}"
    if t not in h.protected.allowed_actions:
        return f"action type {t!r} not allowed by harness"
    index = action.get("index")
    if isinstance(index, dict):
        bad = [o for o in FORBIDDEN_INDEX_OPTIONS if o in index]
        if bad:
            return f"forbidden index option(s) {bad}"
    if t == "drop_index" and action.get("index_name") == "_id_":
        return "dropping _id_"
    nq = action.get("new_query")
    if isinstance(nq, dict):
        stages = nq.get("pipeline") or []
        if any(isinstance(s, dict) and any(k in s for k in FORBIDDEN_STAGES) for s in stages):
            return "$out/$merge in rewritten pipeline"
        if "hint" in nq:
            return "hint in rewritten query"
    return None


def split_forbidden(data: Any, h: AgentHarness) -> tuple[Any, list[dict]]:
    """Remove forbidden actions from a raw reply dict; return (clean data, forbidden list)."""
    if not isinstance(data, dict) or not isinstance(data.get("actions"), list):
        return data, []
    keep, bad = [], []
    for a in data["actions"]:
        why = forbidden_reason(a, h)
        if why:
            bad.append({"action": a, "reason": why})
        else:
            keep.append(a)
    return {**data, "actions": keep}, bad


def parse_reply(resp: Any, h: AgentHarness) -> tuple[PlannerReply, list[dict]]:
    data, bad = split_forbidden(response_data(resp), h)
    return PlannerReply.model_validate(data), bad


def make_plan(
    llm: Any,
    model: str,
    h: AgentHarness,
    context: str,
    extra_messages: list[dict] | None = None,
    toolbox: ToolBox | None = None,
    round_no: int = 0,
) -> tuple[TuningPlan, dict]:
    """Returns (plan, info). info: ok, error, forbidden, tool_calls, messages (for revisions)."""
    messages = [
        {"role": "system", "content": system_prompt(h, with_tools=toolbox is not None)},
        {"role": "user", "content": context},
        *(extra_messages or []),
    ]
    info: dict[str, Any] = {
        "ok": False,
        "error": None,
        "forbidden": [],
        "tool_calls": 0,
        "repaired": False,
    }
    repairs = steps = 0
    while True:
        try:
            resp = llm.complete(ROLE, model, messages, None, PlannerReply)
        except LLMError as e:  # includes LLMJSONError after the client's own repairs
            info["error"] = f"{type(e).__name__}: {e}"[:500]
            break
        text = response_text(resp)
        try:
            reply, bad = parse_reply(resp, h)
        except (ValidationError, JSONExtractError, ValueError) as e:
            info["error"] = f"invalid plan: {e}"[:500]
            if repairs:
                break
            repairs, info["repaired"] = 1, True
            messages = [
                *messages,
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": f"That reply was invalid: {info['error']}\n"
                    'Reply with only {"actions":[...],"summary":"..."}.',
                },
            ]
            continue
        info["forbidden"] += bad
        if toolbox is not None:
            for d in reply.self_diagnose or []:
                toolbox.record_diagnosis(d.category, d.note, round_no)
        if reply.tool_calls and toolbox is not None and steps < MAX_TOOL_STEPS:
            steps += 1
            results = [toolbox.run(c, round_no) for c in reply.tool_calls[:MAX_CALLS_PER_STEP]]
            info["tool_calls"] += len(results)
            last = steps == MAX_TOOL_STEPS
            messages = [
                *messages,
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": "Tool results:\n"
                    + json.dumps(results, default=str)
                    + (
                        "\nNo more tool calls. Reply with the plan now."
                        if last
                        else "\nCall more tools or reply with the plan."
                    ),
                },
            ]
            continue
        plan = TuningPlan(actions=reply.actions or [], summary=reply.summary)
        info.update(ok=True, error=None)
        info["messages"] = [*messages, {"role": "assistant", "content": text}]
        return plan, info
    return TuningPlan(actions=[], summary="planner failed; empty plan"), info
