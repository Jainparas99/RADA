"""The planner: harness rules + context → a strict `TuningPlan` (SPEC §7).

Output is validated against `TuningPlan`. Invalid → one repair turn → else an empty plan (never
an exception), recorded in the trace. Actions outside `protected.allowed_actions` are dropped
here as a guardrail; the scorer remains the authority on violations.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from ratchet.agent.harness_view import AgentHarness
from ratchet.contracts import TuningPlan
from ratchet.llm.jsonutil import JSONExtractError, extract_json
from ratchet.llm.types import LLMError

ROLE = "planner"

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

Reply with one JSON object: {"actions":[...],"summary":"one or two sentences"}.
An empty "actions" list is a valid plan."""


def system_prompt(h: AgentHarness) -> str:
    parts = [SYSTEM]
    if h.planner_rules:
        parts.append("Harness rules:\n" + "\n".join(f"- {r}" for r in h.planner_rules))
    allowed = h.protected.allowed_actions
    parts.append(f"Allowed action types for this run: {', '.join(allowed)}.")
    return "\n\n".join(parts)


def _text(resp: Any) -> str:
    t = getattr(resp, "text", None)
    return t if isinstance(t, str) else json.dumps(getattr(resp, "data", None))


def parse_plan(resp: Any) -> TuningPlan:
    """TuningPlan from an LLM response (real `LLMResponse` or `FakeLLMResponse`)."""
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, TuningPlan):
        return parsed
    data = getattr(resp, "data", None)
    if data is None:
        data = extract_json(_text(resp))
    return TuningPlan.model_validate(data)


def guard(plan: TuningPlan, h: AgentHarness) -> tuple[TuningPlan, list[dict]]:
    allowed = set(h.protected.allowed_actions)
    keep = [a for a in plan.actions if a.type in allowed]
    dropped = [a.model_dump(mode="json") for a in plan.actions if a.type not in allowed]
    return plan.model_copy(update={"actions": keep}), dropped


def make_plan(
    llm: Any, model: str, h: AgentHarness, context: str, extra_messages: list[dict] | None = None
) -> tuple[TuningPlan, dict]:
    """Returns (plan, info). info: ok, error, dropped, messages (for critic revisions)."""
    messages = [
        {"role": "system", "content": system_prompt(h)},
        {"role": "user", "content": context},
        *(extra_messages or []),
    ]
    info: dict[str, Any] = {"ok": False, "error": None, "dropped": [], "repaired": False}
    for attempt in range(2):
        try:
            resp = llm.complete(ROLE, model, messages, None, TuningPlan)
        except LLMError as e:  # includes LLMJSONError after the client's own repairs
            info["error"] = f"{type(e).__name__}: {e}"[:500]
            break
        try:
            plan = parse_plan(resp)
        except (ValidationError, JSONExtractError, ValueError) as e:
            info["error"] = f"invalid plan: {e}"[:500]
            if attempt == 0:
                info["repaired"] = True
                messages = [
                    *messages,
                    {"role": "assistant", "content": _text(resp)},
                    {
                        "role": "user",
                        "content": f"That plan was invalid: {info['error']}\n"
                        'Reply with only {"actions":[...],"summary":"..."}.',
                    },
                ]
            continue
        plan, dropped = guard(plan, h)
        info.update(ok=True, error=None, dropped=dropped)
        info["messages"] = messages + [{"role": "assistant", "content": _text(resp)}]
        return plan, info
    return TuningPlan(actions=[], summary="planner failed; empty plan"), info
