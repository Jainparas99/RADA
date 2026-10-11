"""Deterministic scorer (SPEC §5.6). No LLM anywhere in this file.

    cost(q)       = totalKeysExamined + totalDocsExamined
                    (+ totalDocsExamined again if an in-memory SORT remains in the query
                       layer; for a pipeline $sort, + the docs flowing into it)
                    (+ $lookup stage examined counts for aggregates)
    workload_cost = Σ freq(q) × cost(q)               freq from the *workload*
    improvement   = clamp(1 − cost_after / cost_before, −1, 1)
    index_penalty = Σ_new α × write_rate[coll] + β × size_MB(new) − γ × Σ_dropped_redundant
    score         = improvement − index_penalty      (clamped; 0 on any violation)

Inner violations (any one -> score 0): RESULT_MISMATCH, DATA_MUTATED,
INDEX_BUDGET_EXCEEDED, FORBIDDEN_ACTION, APPLY_ERROR.

The scorer recomputes every cost from ``per_query`` and the workload's frequencies; it
never trusts ``Measurement.workload_cost`` or a frequency carried on a rewritten query.
``SOURCE_HASH`` pins this file, ``config/scoring.yaml`` and the explain normaliser
(``env/explain.py``), so the gate can detect tampering (``SCORER_TAMPERED``).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ratchet.contracts import ExplainSummary, Measurement, ScoreReport, TuningPlan, Workload
from ratchet.env.apply import ApplyReport

SCORER_VERSION = "1"

RESULT_MISMATCH = "RESULT_MISMATCH"
DATA_MUTATED = "DATA_MUTATED"
INDEX_BUDGET_EXCEEDED = "INDEX_BUDGET_EXCEEDED"
FORBIDDEN_ACTION = "FORBIDDEN_ACTION"
APPLY_ERROR = "APPLY_ERROR"
VIOLATION_ORDER = (
    RESULT_MISMATCH,
    DATA_MUTATED,
    INDEX_BUDGET_EXCEEDED,
    FORBIDDEN_ACTION,
    APPLY_ERROR,
)

_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = _ROOT / "config" / "scoring.yaml"
_HASHED_FILES = (
    Path(__file__).resolve(),
    CONFIG_PATH,
    Path(__file__).resolve().parent / "env" / "explain.py",
)


def compute_source_hash(paths: tuple[Path, ...] = _HASHED_FILES) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return "sha256:" + h.hexdigest()


SOURCE_HASH = compute_source_hash()


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- cost


def query_cost(s: ExplainSummary) -> int:
    cost = s.keys_examined + s.docs_examined + s.lookup_examined
    if s.has_in_memory_sort:
        sort_docs = ((s.raw or {}).get("derived") or {}).get("sort_docs")
        cost += s.docs_examined if sort_docs is None else int(sort_docs)
    return cost


def workload_cost(workload: Workload, m: Measurement) -> float:
    total = 0.0
    for q in workload.queries:
        s = m.per_query.get(q.id)
        if s is not None:
            total += q.freq * query_cost(s)
    return total


def improvement(cost_before: float, cost_after: float) -> float:
    if cost_before <= 0:
        return 0.0
    return max(-1.0, min(1.0, 1.0 - cost_after / cost_before))


# --------------------------------------------------------------------------- indexes


def _baseline_keys(workload: Workload) -> dict[str, dict[str, list[tuple[str, Any]]]]:
    return {
        c: {ix.name: [(k, v) for k, v in ix.keys] for ix in ixs if ix.name}
        for c, ixs in workload.baseline_indexes.items()
    }


def _is_prefix(a: list[tuple[str, Any]], b: list[tuple[str, Any]]) -> bool:
    return len(a) < len(b) and b[: len(a)] == a


def dropped_redundant(workload: Workload, before: Measurement, apply: ApplyReport) -> int:
    """Dropped baseline indexes that were a strict key prefix of another baseline index,
    or used by no query in the before-measurement."""
    base = _baseline_keys(workload)
    used: set[str] = set()
    for s in before.per_query.values():
        used.update(s.indexes_used)
        for lk in ((s.raw or {}).get("derived") or {}).get("lookups", []):
            used.update(lk.get("indexes_used") or [])
    n = 0
    for coll, names in apply.dropped.items():
        for name in names:
            keys = base.get(coll, {}).get(name)
            if keys is None:
                continue  # not a baseline index (created and dropped in the same plan)
            others = [k for nm, k in base[coll].items() if nm != name]
            if any(_is_prefix(keys, o) for o in others) or name not in used:
                n += 1
    return n


def index_penalty(
    workload: Workload, before: Measurement, apply: ApplyReport, cfg: dict[str, Any]
) -> float:
    alpha, beta, gamma = float(cfg["alpha"]), float(cfg["beta"]), float(cfg["gamma"])
    pen = 0.0
    for coll, names in apply.created.items():
        sizes = apply.index_sizes_mb.get(coll, {})
        for name in names:
            pen += alpha * float(workload.write_rate.get(coll, 0.0))
            pen += beta * float(sizes.get(name, 0.0))
    pen -= gamma * dropped_redundant(workload, before, apply)
    return pen


# --------------------------------------------------------------------------- violations


def violations(
    workload: Workload, before: Measurement, after: Measurement, apply: ApplyReport
) -> list[str]:
    found: set[str] = set(apply.violations)
    for q in workload.queries:
        b, a = before.result_hashes.get(q.id), after.result_hashes.get(q.id)
        if b is None or a is None or a != b:
            found.add(RESULT_MISMATCH)
            break
    if set(before.data_checksums) != set(after.data_checksums) or any(
        after.data_checksums[c] != v for c, v in before.data_checksums.items()
    ):
        found.add(DATA_MUTATED)
    for coll, names in after.indexes.items():
        budget = workload.index_budget.get(coll)
        if budget is not None and len([n for n in names if n != "_id_"]) > budget:
            found.add(INDEX_BUDGET_EXCEEDED)
    return [v for v in VIOLATION_ORDER if v in found]


# --------------------------------------------------------------------------- score


def score(
    workload: Workload,
    before: Measurement,
    after: Measurement,
    plan: TuningPlan,
    apply: ApplyReport,
    cfg: dict[str, Any] | None = None,
) -> ScoreReport:
    cfg = cfg or load_config()
    imp = improvement(workload_cost(workload, before), workload_cost(workload, after))
    pen = index_penalty(workload, before, apply, cfg)
    viol = violations(workload, before, after, apply)
    if viol:
        s = 0.0
    else:
        s = max(float(cfg["score_min"]), min(float(cfg["score_max"]), imp - pen))
    return ScoreReport(
        improvement=round(imp, 6), index_penalty=round(pen, 6), score=round(s, 6), violations=viol
    )
