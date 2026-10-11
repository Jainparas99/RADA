"""What the proposer is allowed to see (SPEC §9–§10, CLAUDE.md invariant 5).

Only TRAIN results go in. Holdout and canary workloads, traces and scores, and the scorer and
gate source, never do: `build_context` refuses non-train results, gate feedback is reduced to
reason codes plus train-side numbers, and `redact` scrubs any held-out name that slipped into
free text.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from ratchet.config import load_yaml
from ratchet.contracts import GateDecision, PatchOp, TuneResult
from ratchet.gate.leakage import LeakTerms
from ratchet.harness.models import HarnessDoc
from ratchet.jsonsafe import jsonable

MAX_FAILURES = 8
MAX_GATE_DECISIONS = 5
MAX_TRACE_CHARS = 4000
LOW_SCORE = 0.5


class ContextLeakError(RuntimeError):
    """Something other than a train result was offered to the proposer."""


class FailureCase(BaseModel):
    workload_id: str
    score: float
    improvement: float
    index_penalty: float
    violations: list[str]
    detectors: list[str]
    self_diagnostics: list[dict]
    plan: dict
    tokens: int
    rounds: int
    trace: str  # raw trace, truncated


class GateFeedback(BaseModel):
    candidate: str
    verdict: str
    reason_codes: list[str]
    patch: list[PatchOp] = []
    train_before: float | None = None
    train_after: float | None = None


class ProposerContext(BaseModel):
    head_version: str
    spec: dict
    editable_paths: list[str]
    bounds: dict
    patch_limits: dict
    train_summary: dict
    failures: list[FailureCase]
    memory_stats: dict = {}
    gate_history: list[GateFeedback] = []
    redactions: int = 0


def _tags(r: TuneResult) -> set[str]:
    return set(r.score.violations) | {f.detector for f in r.findings}


def select_failures(results: list[TuneResult], limit: int = MAX_FAILURES) -> list[TuneResult]:
    """Worst train results first, preferring ones that add a detector/violation not yet covered."""
    pool = sorted(results, key=lambda r: (r.score.score, r.workload_id))
    pool = [r for r in pool if r.score.violations or r.score.score < LOW_SCORE] or pool
    chosen: list[TuneResult] = []
    covered: set[str] = set()
    while pool and len(chosen) < limit:
        best = max(pool, key=lambda r: len(_tags(r) - covered))  # stable: first (worst) wins ties
        chosen.append(best)
        covered |= _tags(best)
        pool.remove(best)
    return chosen


def _failure_case(r: TuneResult) -> FailureCase:
    trace = json.dumps(jsonable(r.trace), default=str)
    if len(trace) > MAX_TRACE_CHARS:
        trace = trace[:MAX_TRACE_CHARS] + f"... [truncated {len(trace) - MAX_TRACE_CHARS} chars]"
    return FailureCase(
        workload_id=r.workload_id,
        score=r.score.score,
        improvement=r.score.improvement,
        index_penalty=r.score.index_penalty,
        violations=r.score.violations,
        detectors=sorted({f.detector for f in r.findings}),
        self_diagnostics=r.self_diagnostics,
        plan=jsonable(r.plan, exclude_none=True),
        tokens=r.tokens,
        rounds=r.rounds,
        trace=trace,
    )


def gate_feedback(decision: GateDecision, patch: list[PatchOp] | None = None) -> GateFeedback:
    """Reason codes and train-side numbers only; holdout/canary metrics stay with the gate."""
    m = decision.metrics
    return GateFeedback(
        candidate=decision.candidate,
        verdict=decision.verdict,
        reason_codes=decision.reason_codes,
        patch=patch or [],
        train_before=(m.get("parent") or {}).get("train"),
        train_after=(m.get("candidate") or {}).get("train"),
    )


def redact(obj: Any, leak_terms: LeakTerms) -> tuple[Any, int]:
    """Replace held-out names in every string of a JSON-like object. Returns (clean, hits)."""
    hits = 0

    def clean(s: str) -> str:
        nonlocal hits
        if not leak_terms.find(s):
            return s
        for t in leak_terms.terms:
            s, n = re.subn(rf"(?i)(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", "[redacted]", s)
            hits += n
        if leak_terms.find(s):  # obfuscated spelling: drop the whole string
            hits += 1
            return "[redacted]"
        return s

    def walk(o: Any) -> Any:
        if isinstance(o, str):
            return clean(o)
        if isinstance(o, dict):
            return {clean(k) if isinstance(k, str) else k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o

    return walk(obj), hits


def build_context(
    head: HarnessDoc,
    train_results: list[TuneResult],
    gate_history: list[GateFeedback] | None = None,
    memory_stats: dict | None = None,
    leak_terms: LeakTerms | None = None,
    bounds: dict | None = None,
    gate_cfg: dict | None = None,
) -> ProposerContext:
    bad = sorted({f"{r.workload_id} ({r.split})" for r in train_results if r.split != "train"})
    if bad:
        raise ContextLeakError(f"non-train results offered to the proposer: {', '.join(bad)}")
    bounds = bounds if bounds is not None else load_yaml("bounds.yaml")
    static_cfg = (gate_cfg if gate_cfg is not None else load_yaml("gate.yaml")).get("static", {})
    n = len(train_results)
    ctx = ProposerContext(
        head_version=head.id,
        spec=head.spec.to_doc(),
        editable_paths=[f"/{r}" for r in bounds.get("editable_roots", [])],
        bounds={"lists": bounds.get("lists", {}), "numeric": bounds.get("numeric", {})},
        patch_limits={
            "max_ops": static_cfg.get("max_ops", 4),
            "max_added_chars": static_cfg.get("max_added_chars", 1200),
        },
        train_summary={
            "workloads": n,
            "mean_score": sum(r.score.score for r in train_results) / n if n else 0.0,
            "violations": sum(len(r.score.violations) for r in train_results),
            "tokens_per_workload": sum(r.tokens for r in train_results) / n if n else 0.0,
            "per_workload": [
                {
                    "workload_id": r.workload_id,
                    "score": round(r.score.score, 4),
                    "violations": r.score.violations,
                    "tokens": r.tokens,
                }
                for r in train_results
            ],
        },
        failures=[_failure_case(r) for r in select_failures(train_results)],
        memory_stats=memory_stats or {},
        gate_history=(gate_history or [])[-MAX_GATE_DECISIONS:],
    )
    if leak_terms is not None:
        clean, hits = redact(ctx.model_dump(mode="json"), leak_terms)
        ctx = ProposerContext.model_validate({**clean, "redactions": hits})
    return ctx
