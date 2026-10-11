"""The critic (SPEC §7): a different model reviews each plan against `critic_rubric`.

Verdict: {"approve": bool, "issues": [...], "revised_plan": TuningPlan | null}. On rejection the
critic's revised plan is used if it gives one, else the planner revises with the issues. Up to
`critic_policy.max_revisions` revisions; every verdict goes into the trace. The critic is advisory:
it can change the plan, never the score — the deterministic scorer has the last word.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from ratchet.agent.harness_view import AgentHarness
from ratchet.agent.planner import forbidden_reason, make_plan, response_data
from ratchet.agent.tools import ToolBox
from ratchet.config import DEFAULT_CRITIC_MODEL, get_settings
from ratchet.contracts import TuningPlan, Workload
from ratchet.llm.jsonutil import JSONExtractError
from ratchet.llm.providers import split_model
from ratchet.llm.types import LLMError

ROLE = "critic"

BASE_CHECKS = [
    "Every action is one of create_index / drop_index / rewrite_query, with no unique, sparse, "
    "TTL or hidden index options and no $out/$merge/hint.",
    "No rewrite can change which documents are returned or their order (filters, limits, sort "
    "keys and tie-breaks must be preserved).",
    "Index count per collection after the plan stays within index_budget.",
    "Compound index keys follow equality, then sort, then range (ESR) for the query they serve.",
    "Every new index is used by at least one workload query; prefer fewer indexes on "
    "write-heavy collections.",
]

SYSTEM = """You are the critic for a MongoDB query-tuning agent. Another model wrote the plan below.
Check it against the checklist. Approve only if every item holds. If not, list concrete issues and,
if you can, give a corrected plan in the same format (or null).

Reply with one JSON object:
{"approve": true|false, "issues": ["..."],
 "revised_plan": {"actions":[...],"summary":"..."} | null}"""


class CriticVerdict(BaseModel):
    approve: bool
    issues: list[str] = []
    revised_plan: TuningPlan | None = None


def pick_critic_model(h: AgentHarness, planner_model: str, rec: Any) -> str | None:
    """A callable model different from the planner; prefer another provider (SPEC §11.2)."""
    try:
        configured = get_settings().critic_model
    except Exception:
        configured = DEFAULT_CRITIC_MODEL
    candidates = list(dict.fromkeys([h.model_routing.critic, configured, DEFAULT_CRITIC_MODEL]))
    usable = [m for m in candidates if m != planner_model and rec.available(m)]
    if not usable:
        return None

    def other_provider(m: str) -> bool:
        try:
            return split_model(m)[0] != split_model(planner_model)[0]
        except Exception:
            return False

    return next((m for m in usable if other_provider(m)), usable[0])


def system_prompt(h: AgentHarness) -> str:
    rubric = [*BASE_CHECKS, *h.critic_rubric]
    return SYSTEM + "\n\nChecklist:\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(rubric, 1))


def review(rec: Any, model: str, h: AgentHarness, context: str, plan: TuningPlan) -> CriticVerdict:
    messages = [
        {"role": "system", "content": system_prompt(h)},
        {
            "role": "user",
            "content": context
            + "\n\n## Plan under review\n"
            + plan.model_dump_json(exclude_none=True),
        },
    ]
    resp = rec.complete(ROLE, model, messages, None, CriticVerdict)
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, CriticVerdict):
        return parsed
    data = response_data(resp)
    rp = data.get("revised_plan") if isinstance(data, dict) else None
    if isinstance(rp, dict) and any(forbidden_reason(a, h) for a in rp.get("actions", [])):
        data = {**data, "revised_plan": None}  # never let the critic smuggle in a forbidden action
    return CriticVerdict.model_validate(data)


def critique_and_revise(
    rec: Any,
    h: AgentHarness,
    workload: Workload,
    context: str,
    plan: TuningPlan,
    info: dict,
    trace: list[dict],
    planner_model: str,
    toolbox: ToolBox | None = None,
    round_no: int = 0,
) -> TuningPlan:
    model = pick_critic_model(h, planner_model, rec)
    if model is None:
        trace.append({"step": "critic", "round": round_no, "skipped": "no critic model available"})
        return plan
    max_rev = max(0, h.critic_policy.max_revisions)
    for revision in range(max_rev + 1):
        try:
            verdict = review(rec, model, h, context, plan)
        except (LLMError, ValidationError, JSONExtractError, ValueError) as e:
            trace.append(
                {
                    "step": "critic",
                    "round": round_no,
                    "revision": revision,
                    "model": model,
                    "error": f"{type(e).__name__}: {e}"[:300],
                    "approve": None,
                }
            )
            return plan  # fail open: the scorer still guards the result
        trace.append(
            {
                "step": "critic",
                "round": round_no,
                "revision": revision,
                "model": model,
                "approve": verdict.approve,
                "issues": verdict.issues,
                "revised_by": "critic" if verdict.revised_plan is not None else None,
            }
        )
        if verdict.approve or revision == max_rev:
            if not verdict.approve:
                trace.append({"step": "critic_unresolved", "round": round_no})
            return plan
        if verdict.revised_plan is not None:
            plan = verdict.revised_plan
            continue
        feedback = [
            *info.get("messages", [])[2:],
            {
                "role": "user",
                "content": "A reviewer rejected this plan:\n- "
                + "\n- ".join(verdict.issues or ["(no details)"])
                + "\nRevise it. Reply with the full corrected plan.",
            },
        ]
        if not feedback[0:1] or feedback[0].get("role") != "assistant":
            feedback = [
                {"role": "assistant", "content": json.dumps(plan.model_dump(mode="json"))},
                feedback[-1],
            ]
        revised, rinfo = make_plan(rec, planner_model, h, context, feedback, toolbox, round_no)
        info["forbidden"] = info.get("forbidden", []) + rinfo.get("forbidden", [])
        if rinfo["ok"]:
            plan, info["messages"] = revised, rinfo["messages"]
    return plan
