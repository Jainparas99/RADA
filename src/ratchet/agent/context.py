"""Builds the planner's context from the workload, measurements and `context_policy` (SPEC §6, §7).

Sections are added in priority order and the whole text is capped at `max_chars`; low-priority
sections (sample docs, stats, raw explain) are the first to be cut.
"""

from __future__ import annotations

import json
from typing import Any

from ratchet.agent.harness_view import AgentHarness
from ratchet.contracts import ExplainSummary, Finding, Measurement, TuningPlan, Workload

TRUNCATED = "\n...[context truncated to max_chars]"


def _j(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str, sort_keys=False)


def brief(s: ExplainSummary) -> dict:
    return {
        "stages": s.stages,
        "n_returned": s.n_returned,
        "keys_examined": s.keys_examined,
        "docs_examined": s.docs_examined,
        "in_memory_sort": s.has_in_memory_sort,
        "indexes_used": s.indexes_used,
        **({"lookup_examined": s.lookup_examined} if s.lookup_examined else {}),
    }


def workload_section(w: Workload, current_queries: dict[str, Any] | None = None) -> str:
    queries = current_queries or {q.id: q for q in w.queries}
    return "## Workload\n" + _j(
        {
            "id": w.id,
            "db": w.db,
            "collections": w.collections,
            "index_budget": w.index_budget,
            "write_rate": w.write_rate,
            "queries": [q.model_dump(mode="json", exclude_none=True) for q in queries.values()],
        }
    )


def explain_section(m: Measurement, full: bool) -> str:
    rows = {qid: brief(s) for qid, s in m.per_query.items()}
    out = "## Explain (before)\n" + _j({"workload_cost": m.workload_cost, "per_query": rows})
    if full:
        raw = {qid: s.raw for qid, s in m.per_query.items() if s.raw}
        if raw:
            out += "\n## Raw explain\n" + _j(raw)
    return out


def indexes_section(m: Measurement) -> str:
    return "## Existing indexes\n" + _j(m.indexes)


def findings_section(findings: list[Finding]) -> str:
    return "## Detector findings\n" + _j([f.model_dump(mode="json") for f in findings] or "none")


def lessons_section(lessons: list[str]) -> str:
    return "## Lessons from earlier workloads\n" + "\n".join(f"- {t}" for t in lessons)


def explain_diff(before: Measurement, after: Measurement) -> dict:
    out = {}
    for qid, b in before.per_query.items():
        a = after.per_query.get(qid)
        if a is None:
            continue
        out[qid] = {
            "stages": [b.stages, a.stages],
            "docs_examined": [b.docs_examined, a.docs_examined],
            "keys_examined": [b.keys_examined, a.keys_examined],
            "in_memory_sort": [b.has_in_memory_sort, a.has_in_memory_sort],
        }
    return out


def retry_section(
    prev_plan: TuningPlan,
    improvement: float,
    violations: list[str],
    before: Measurement,
    after: Measurement,
    show_diff: bool,
) -> str:
    body: dict[str, Any] = {
        "applied_so_far": prev_plan.model_dump(mode="json", exclude_none=True)["actions"],
        "improvement": round(improvement, 4),
        "violations": violations,
        "workload_cost": [before.workload_cost, after.workload_cost],
        "indexes_now": after.indexes,
    }
    if show_diff:
        body["explain_diff_before_after"] = explain_diff(before, after)
    return (
        "## Previous round (already applied on scratch; do not repeat these actions)\n"
        + _j(body)
        + "\nPropose only ADDITIONAL actions that improve further, or an empty plan if done."
    )


def build_context(
    workload: Workload,
    harness: AgentHarness,
    before: Measurement,
    findings: list[Finding],
    *,
    current_queries: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
    samples: dict[str, list] | None = None,
    lessons: list[str] | None = None,
    retry: str | None = None,
) -> str:
    p = harness.context_policy
    sections = [workload_section(workload, current_queries), explain_section(before, False)]
    if p.include_existing_indexes:
        sections.append(indexes_section(before))
    if p.include_detector_findings:
        sections.append(findings_section(findings))
    if retry:
        sections.append(retry)
    if lessons:
        sections.append(lessons_section(lessons))
    if p.include_collection_stats and stats:
        sections.append("## Collection stats\n" + _j(stats))
    if p.include_full_explain:
        raw = explain_section(before, True).split("\n## Raw explain\n", 1)
        if len(raw) == 2:
            sections.append("## Raw explain\n" + raw[1])
    if p.sample_docs > 0 and samples:
        sections.append("## Sample documents\n" + _j(samples))
    text = "\n\n".join(sections)
    cap = max(500, p.max_chars)
    if len(text) > cap:
        text = text[: cap - len(TRUNCATED)] + TRUNCATED
    return text
