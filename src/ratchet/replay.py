"""`ratchet replay`: re-run a recorded generation purely from `llm_cache` (demo backup, SPEC §15).

The LLM client is switched to cache-only: every planner/critic/proposer reply must come from
`llm_cache`, so a replay costs nothing and makes no provider calls. Measurements still run against
the (scratch) environment, so the numbers are recomputed, not copied. A cache miss is reported, not
papered over. `RATCHET_LLM_CACHE_ONLY=1` does the same for any command (e.g. `ratchet evolve`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from ratchet.contracts import TuneResult, Workload
from ratchet.llm.cache import Cache
from ratchet.llm.client import LLMClient, get_client, set_client


class ReplayRow(BaseModel):
    workload_id: str
    split: str
    score: float
    improvement: float
    violations: list[str]
    llm_calls: int
    cache_misses: int
    recorded_score: float | None = None


def resolve_version(db: Any, version: str | None, gen: int | None) -> str:
    from ratchet.db import GENERATIONS, HARNESS_HEAD

    if gen is not None:
        doc = db[GENERATIONS].find_one({"gen": gen})
        if not doc or not doc.get("candidate"):
            raise ValueError(f"generation {gen} has no recorded candidate")
        return str(doc["candidate"])
    if version:
        return version if version.startswith("h_") else f"h_{int(version):04d}"
    head = db[HARNESS_HEAD].find_one({"_id": "head"})
    if not head:
        raise ValueError("no harness head; pass --version or --gen")
    return str(head["version_id"])


def cache_only_client(cache: Cache | None = None) -> LLMClient:
    return LLMClient(cache=cache if cache is not None else get_client().cache, cache_only=True)


def _misses(r: TuneResult) -> tuple[int, int]:
    llm = [t for t in r.trace if t.get("step") == "llm"]
    misses = [t for t in llm if "CacheMiss" in str(t.get("error", ""))]
    return len(llm), len(misses)


def replay(
    workloads: list[Workload],
    tune_fn: Callable[[Workload], TuneResult],
    recorded: dict[str, float] | None = None,
    cache: Cache | None = None,
) -> list[ReplayRow]:
    """Run `tune_fn` for each workload with a cache-only client installed."""
    prev = get_client()
    set_client(cache_only_client(cache if cache is not None else prev.cache))
    rows: list[ReplayRow] = []
    try:
        for w in workloads:
            r = tune_fn(w)
            calls, misses = _misses(r)
            rows.append(
                ReplayRow(
                    workload_id=w.id,
                    split=w.split,
                    score=r.score.score,
                    improvement=r.score.improvement,
                    violations=r.score.violations,
                    llm_calls=calls,
                    cache_misses=misses,
                    recorded_score=(recorded or {}).get(w.id),
                )
            )
    finally:
        set_client(prev)
    return rows
