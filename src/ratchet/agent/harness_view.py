"""A lenient, read-only view of the harness spec (SPEC §6) for the tuning agent.

Track A owns the strict `HarnessSpec`. The agent accepts that model, its `to_doc()` dict, or a
plain dict, and reads it through this view: unknown keys are ignored and missing knobs fall back
to gen-0 values, so the agent keeps working while the harness schema evolves.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ratchet.config import DEFAULT_CRITIC_MODEL, DEFAULT_PLANNER_MODEL

ALL_ACTIONS = ["create_index", "drop_index", "rewrite_query"]


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class ContextView(_Lenient):
    include_detector_findings: bool = False
    include_full_explain: bool = False
    include_existing_indexes: bool = True
    include_collection_stats: bool = False
    sample_docs: int = 0
    max_chars: int = 6000


class RetryView(_Lenient):
    max_rounds: int = 1
    retry_if_improvement_below: float = 0.5
    show_explain_diff: bool = False


class CriticView(_Lenient):
    enabled: bool = False
    max_revisions: int = 1


class RoutingView(_Lenient):
    planner: str = DEFAULT_PLANNER_MODEL
    critic: str = DEFAULT_CRITIC_MODEL


class InjectView(_Lenient):
    mode: str = "never"
    max_tokens: int = 500


class RetrieveView(_Lenient):
    k: int = 3
    min_similarity: float = 0.75


class MemoryView(_Lenient):
    retrieve: RetrieveView = RetrieveView()
    inject: InjectView = InjectView()


class ProtectedView(_Lenient):
    model_allowlist: list[str] = []
    allowed_actions: list[str] = Field(default_factory=lambda: list(ALL_ACTIONS))
    token_budget_per_workload: int = 30_000


class AgentHarness(_Lenient):
    planner_rules: list[str] = [
        "Add an index for queries that scan the whole collection.",
        "Never change what a query returns.",
    ]
    critic_rubric: list[str] = []
    context_policy: ContextView = ContextView()
    retry_policy: RetryView = RetryView()
    critic_policy: CriticView = CriticView()
    model_routing: RoutingView = RoutingView()
    memory_policy: MemoryView = MemoryView()
    protected: ProtectedView = ProtectedView()


def harness_doc(h: Any) -> dict:
    """The harness as a plain dict (wire names), whatever form it came in."""
    if h is None:
        return {}
    if isinstance(h, AgentHarness):
        return h.model_dump(mode="json")
    if hasattr(h, "to_doc"):
        return h.to_doc()
    if isinstance(h, BaseModel):
        return h.model_dump(mode="json", by_alias=True)
    return dict(h)


def view(h: Any) -> AgentHarness:
    if isinstance(h, AgentHarness):
        return h
    return AgentHarness.model_validate(harness_doc(h))


def gen0() -> AgentHarness:
    """Gen-0 defaults as the agent sees them (Track A's `gen0_spec` is the canonical one)."""
    return AgentHarness()
