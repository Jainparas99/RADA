"""Inner loop (SPEC §7): diagnose → plan → critic → apply on scratch → measure → retry.

`tune(workload, harness, env, mode)` matches the SPEC §10 interface. The environment functions
(`measure`, `run_detectors`, `apply_plan`, `score`, optional `query_shape`, `collection_stats`,
`sample_docs`) come from `api`: Track B's `ratchet.env` module by default, or any object with
those methods (e.g. `tests.fakes.FakeEnv`). LLM calls go through `llm` (default `ratchet.llm`).
Every LLM message, plan, apply report and measurement is recorded in `TuneResult.trace`.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel

from ratchet.agent import context as ctx
from ratchet.agent.harness_view import AgentHarness, view
from ratchet.agent.planner import make_plan
from ratchet.config import get_settings
from ratchet.contracts import (
    Finding,
    Measurement,
    QuerySpec,
    ScoreReport,
    TuneResult,
    TuningPlan,
    Workload,
)

Mode = Literal["normal", "frozen_memory"]


# --------------------------------------------------------------------------- plumbing


class Recorder:
    """Wraps an LLM backend: records every call into the trace and sums tokens."""

    def __init__(self, llm: Any, trace: list[dict]) -> None:
        self.llm = llm
        self.trace = trace
        self.tokens = 0
        self.models: dict[str, str] = {}

    def available(self, model: str) -> bool:
        fn = getattr(self.llm, "available", None)
        return True if fn is None else bool(fn(model))

    def complete(self, role, model, messages, tools=None, json_schema=None):
        t0 = time.monotonic()
        entry: dict[str, Any] = {"step": "llm", "role": role, "model": model, "messages": messages}
        try:
            resp = self.llm.complete(role, model, messages, tools, json_schema)
        except Exception as e:
            entry.update(error=f"{type(e).__name__}: {e}"[:500])
            self.trace.append(entry)
            raise
        tokens = int(getattr(resp, "tokens", 0) or 0)
        self.tokens += tokens
        self.models[role] = model
        entry.update(
            response=getattr(resp, "text", None),
            tokens=tokens,
            cached=bool(getattr(resp, "cached", False)),
            ms=int((time.monotonic() - t0) * 1000),
        )
        self.trace.append(entry)
        return resp


def default_api() -> Any:
    import ratchet.env as env_api  # Track B

    return env_api


def default_llm() -> Any:
    import ratchet.llm as llm_api

    return llm_api


def default_memory() -> Any | None:
    """Track A's memory module if it exposes `recall`, else None (no lessons)."""
    try:
        import ratchet.memory as mem
    except ImportError:
        return None
    return mem if callable(getattr(mem, "recall", None)) else None


def merge_reports(a: Any, b: Any) -> Any:
    """Combine two apply reports (pydantic models with list / dict-of-list fields)."""
    if a is None:
        return b
    if not isinstance(b, BaseModel):
        return b
    upd: dict[str, Any] = {}
    for name in type(b).model_fields:
        va, vb = getattr(a, name, None), getattr(b, name)
        if isinstance(va, list) and isinstance(vb, list):
            upd[name] = va + [x for x in vb if name != "violations" or x not in va]
        elif isinstance(va, dict) and isinstance(vb, dict):
            merged = dict(va)
            for k, v in vb.items():
                old = merged.get(k)
                merged[k] = old + v if isinstance(old, list) and isinstance(v, list) else v
            upd[name] = merged
    return b.model_copy(update=upd)


def _report_dump(r: Any) -> Any:
    return r.model_dump(mode="json") if isinstance(r, BaseModel) else repr(r)


def _shape(api: Any, q: Any) -> str:
    fn = getattr(api, "query_shape", None)
    if callable(fn):
        return fn(q)
    return f"{q.collection} | {q.kind} | filter {sorted(q.filter)}"


def _lesson_text(lesson: Any) -> str:
    for attr in ("text", "lesson", "content"):
        v = getattr(lesson, attr, None) if not isinstance(lesson, dict) else lesson.get(attr)
        if isinstance(v, str):
            return v
    return str(lesson)


def recall_lessons(
    memory: Any, api: Any, workload: Workload, harness_spec: Any, h: AgentHarness
) -> list[str]:
    if memory is None:
        return []
    policy = getattr(harness_spec, "memory_policy", None) or h.memory_policy
    budget = max(0, h.memory_policy.inject.max_tokens) * 4  # ~4 chars per token
    out: list[str] = []
    for q in workload.queries:
        try:
            lessons = memory.recall(_shape(api, q), policy)
        except Exception:
            continue
        for lesson in lessons or []:
            t = _lesson_text(lesson)
            if t and t not in out and sum(map(len, out)) + len(t) <= budget:
                out.append(t)
    return out


def _current_queries(env: Any) -> dict[str, QuerySpec] | None:
    qs = getattr(env, "queries", None)
    if isinstance(qs, dict) and qs and all(isinstance(q, QuerySpec) for q in qs.values()):
        return qs
    return None


def _optional(api: Any, name: str, *args: Any) -> Any:
    fn = getattr(api, name, None)
    if not callable(fn):
        return None
    try:
        return fn(*args)
    except Exception:
        return None


# --------------------------------------------------------------------------- the loop


class RoundOutcome(BaseModel):
    round: int
    plan: TuningPlan
    after: Measurement
    score: ScoreReport


def tune(
    workload: Workload,
    harness: Any,
    env: Any,
    mode: Mode = "normal",
    *,
    api: Any | None = None,
    llm: Any | None = None,
    memory: Any | None = ...,  # type: ignore[assignment]
    harness_version: str = "",
    critic: Any | None = None,
) -> TuneResult:
    api = api if api is not None else default_api()
    llm = llm if llm is not None else default_llm()
    memory = default_memory() if memory is ... else memory
    h = view(harness)
    trace: list[dict] = [{"step": "start", "workload": workload.id, "mode": mode}]
    rec = Recorder(llm, trace)

    settings = get_settings()
    planner_model = h.model_routing.planner
    if not rec.available(planner_model):
        trace.append({"step": "routing", "note": f"{planner_model} unavailable, using default"})
        planner_model = settings.planner_model

    before = api.measure(env, workload)
    findings: list[Finding] = api.run_detectors(env, workload, before)
    trace.append(
        {
            "step": "measure_before",
            "measurement": before.model_dump(
                mode="json", exclude={"per_query": {"__all__": {"raw"}}}
            ),
            "findings": [f.model_dump(mode="json") for f in findings],
        }
    )
    stats = samples = None
    if h.context_policy.include_collection_stats:
        stats = {c: _optional(api, "collection_stats", env, c) for c in workload.collections}
    if h.context_policy.sample_docs > 0:
        samples = {
            c: _optional(api, "sample_docs", env, c, h.context_policy.sample_docs)
            for c in workload.collections
        }

    inject = h.memory_policy.inject.mode
    lessons_used: list[str] = []
    applied: list = []
    report: Any = None
    prev_score: float | None = None
    last: RoundOutcome | None = None
    budget = h.protected.token_budget_per_workload
    max_rounds = max(1, h.retry_policy.max_rounds)
    rounds = 0

    for rnd in range(1, max_rounds + 1):
        rounds = rnd
        lessons: list[str] = []
        if inject == "always" or (inject == "on_retry" and rnd > 1):
            lessons = recall_lessons(memory, api, workload, harness, h)
            lessons_used += [t for t in lessons if t not in lessons_used]
        retry = None
        if last is not None:
            retry = ctx.retry_section(
                TuningPlan(actions=applied),
                last.score.improvement,
                last.score.violations,
                before,
                last.after,
                h.retry_policy.show_explain_diff,
            )
        context = ctx.build_context(
            workload,
            h,
            before,
            findings,
            current_queries=_current_queries(env),
            stats=stats,
            samples=samples,
            lessons=lessons,
            retry=retry,
        )
        plan, info = make_plan(rec, planner_model, h, context)
        trace.append(
            {
                "step": "plan",
                "round": rnd,
                "plan": plan.model_dump(mode="json"),
                "ok": info["ok"],
                "error": info["error"],
                "dropped_actions": info["dropped"],
            }
        )
        if critic is not None and h.critic_policy.enabled and plan.actions:
            plan = critic(rec, h, workload, context, plan, info, trace)

        if not plan.actions:
            trace.append({"step": "stop", "round": rnd, "reason": "empty plan"})
            if last is None:
                last = RoundOutcome(
                    round=rnd,
                    plan=TuningPlan(actions=[]),
                    after=before,
                    score=api.score(
                        workload, before, before, plan, api.apply_plan(env, workload, plan)
                    ),
                )
            break

        rep = api.apply_plan(env, workload, plan)
        report = merge_reports(report, rep)
        applied += plan.actions
        total_plan = TuningPlan(actions=list(applied), summary=plan.summary)
        after = api.measure(env, workload)
        sc = api.score(workload, before, after, total_plan, report)
        last = RoundOutcome(round=rnd, plan=total_plan, after=after, score=sc)
        trace.append(
            {
                "step": "apply_measure",
                "round": rnd,
                "apply": _report_dump(rep),
                "measurement": after.model_dump(
                    mode="json", exclude={"per_query": {"__all__": {"raw"}}}
                ),
                "score": sc.model_dump(mode="json"),
            }
        )
        if sc.violations:
            trace.append({"step": "stop", "round": rnd, "reason": "violation"})
            break
        if prev_score is not None and sc.score <= prev_score:
            trace.append({"step": "stop", "round": rnd, "reason": "no improvement"})
            break
        prev_score = sc.score
        if sc.improvement >= h.retry_policy.retry_if_improvement_below:
            trace.append({"step": "stop", "round": rnd, "reason": "good enough"})
            break
        if rec.tokens >= budget:
            trace.append({"step": "stop", "round": rnd, "reason": "token budget"})
            break

    assert last is not None
    final = last
    models_used = {"planner": planner_model, **rec.models}
    return TuneResult(
        workload_id=workload.id,
        split=workload.split,
        harness_version=harness_version or str(getattr(harness, "version", "") or ""),
        plan=final.plan,
        before=before,
        after=final.after,
        score=final.score,
        findings=findings,
        tokens=rec.tokens,
        rounds=rounds,
        trace=trace,
        self_diagnostics=[t for t in trace if t.get("step") == "self_diagnose"],
        lessons_used=lessons_used,
        models_used=models_used,
    )
