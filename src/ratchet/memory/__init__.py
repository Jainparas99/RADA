"""Track A lesson memory (SPEC §8).

Module-level `recall` / `write_lessons` are the SPEC §10 interface, bound to the `ratchet_<ns>`
database, `ratchet.llm` (Track C) for embeddings and distillation, and `ratchet.env.query_shape`
(Track B). The tuning agent calls `ratchet.memory.recall(shape, policy)`.
"""

from __future__ import annotations

from typing import Any

from ratchet.contracts import TuneResult
from ratchet.memory.models import Lesson
from ratchet.memory.store import MemoryFrozenError, MemoryStore, cosine, inject_text

__all__ = [
    "Lesson",
    "MemoryFrozenError",
    "MemoryStore",
    "cosine",
    "default_store",
    "inject_text",
    "recall",
    "set_default_store",
    "write_lessons",
]

_store: MemoryStore | None = None


def set_default_store(store: MemoryStore | None) -> None:
    """Use one shared store (so `evolve` can freeze the same object the agent reads)."""
    global _store
    _store = store


def default_store() -> MemoryStore:
    global _store
    if _store is None:
        from ratchet import llm  # Track C
        from ratchet.config import get_settings
        from ratchet.db import get_db
        from ratchet.env import query_shape  # Track B

        _store = MemoryStore(
            get_db(),
            embed=llm.embed,
            complete=llm.complete,
            shape_fn=query_shape,
            model=get_settings().planner_model,
        )
    return _store


def recall(shape: str, policy: Any) -> list[Lesson]:
    return default_store().recall(shape, policy)


def write_lessons(
    results: list[TuneResult], policy: Any, queries: dict | None = None
) -> list[Lesson]:
    return default_store().write_lessons(results, policy, queries)
