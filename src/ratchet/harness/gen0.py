"""Gen 0: a deliberately minimal baseline harness (SPEC §6)."""

from __future__ import annotations

from ratchet.config import (
    DEFAULT_CRITIC_MODEL,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_PROPOSER_MODEL,
    Settings,
)
from ratchet.gate.hashes import gate_ref, scorer_ref
from ratchet.harness.models import ALLOWED_ACTIONS, HarnessSpec

TOKEN_BUDGET_PER_WORKLOAD = 30_000

GEN0_PLANNER_RULES = [
    "Add an index for queries that scan the whole collection.",
    "Never change what a query returns.",
]


def model_allowlist(settings: Settings | None = None) -> list[str]:
    models = [DEFAULT_PLANNER_MODEL, DEFAULT_CRITIC_MODEL, DEFAULT_PROPOSER_MODEL]
    if settings is not None:
        models += [settings.planner_model, settings.critic_model, *settings.model_allowlist]
    return list(dict.fromkeys(models))


def gen0_spec(settings: Settings | None = None) -> HarnessSpec:
    """2 short rules, no rubric, critic off, no findings/sample docs in context, memory never
    injected, one round. Scorer and gate hashes are pinned to the current source."""
    planner = settings.planner_model if settings else DEFAULT_PLANNER_MODEL
    critic = settings.critic_model if settings else DEFAULT_CRITIC_MODEL
    return HarnessSpec.from_doc(
        {
            "planner_rules": list(GEN0_PLANNER_RULES),
            "critic_rubric": [],
            "context_policy": {
                "include_detector_findings": False,
                "include_full_explain": False,
                "include_existing_indexes": True,
                "include_collection_stats": False,
                "sample_docs": 0,
                "max_chars": 6000,
            },
            "retry_policy": {
                "max_rounds": 1,
                "retry_if_improvement_below": 0.5,
                "show_explain_diff": False,
            },
            "critic_policy": {"enabled": False, "max_revisions": 1},
            "model_routing": {"planner": planner, "critic": critic},
            "memory_policy": {
                "write": {
                    "enabled": True,
                    "from": ["train_improved", "train_violation"],
                    "lesson_max_chars": 300,
                },
                "retrieve": {"k": 3, "min_similarity": 0.75},
                "inject": {"mode": "never", "max_tokens": 500},
                "consolidate": {"merge_similarity": 0.92, "max_items": 40},
                "invalidate": {"after_harmful_uses": 2},
            },
            "protected": {
                "model_allowlist": model_allowlist(settings),
                "allowed_actions": list(ALLOWED_ACTIONS),
                "token_budget_per_workload": TOKEN_BUDGET_PER_WORKLOAD,
                "scorer_ref": scorer_ref().model_dump(),
                "gate_ref": gate_ref().model_dump(),
            },
        }
    )
