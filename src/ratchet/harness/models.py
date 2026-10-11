"""Harness document models (SPEC §6, §8).

Types only: numeric ranges and list limits live in `config/bounds.yaml` and are enforced by the
gate, so a bad value is reported as OUT_OF_BOUNDS rather than a schema error. Models are strict
and forbid extra keys so a patch cannot smuggle in new fields or coerce types.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ratchet.contracts import PatchOp

ALLOWED_ACTIONS = ["create_index", "drop_index", "rewrite_query"]
EDITABLE_ROOTS = [
    "planner_rules",
    "critic_rubric",
    "context_policy",
    "retry_policy",
    "critic_policy",
    "memory_policy",
    "model_routing",
]


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class ContextPolicy(_Strict):
    include_detector_findings: bool
    include_full_explain: bool
    include_existing_indexes: bool
    include_collection_stats: bool
    sample_docs: int
    max_chars: int


class RetryPolicy(_Strict):
    max_rounds: int
    retry_if_improvement_below: float
    show_explain_diff: bool


class CriticPolicy(_Strict):
    enabled: bool
    max_revisions: int


class ModelRouting(_Strict):
    planner: str
    critic: str


class MemoryWrite(_Strict):
    enabled: bool
    from_: list[Literal["train_improved", "train_violation"]] = Field(alias="from")
    lesson_max_chars: int

    model_config = ConfigDict(strict=True, extra="forbid", populate_by_name=True)


class MemoryRetrieve(_Strict):
    k: int
    min_similarity: float


class MemoryInject(_Strict):
    mode: Literal["never", "always", "on_retry"]
    max_tokens: int


class MemoryConsolidate(_Strict):
    merge_similarity: float
    max_items: int


class MemoryInvalidate(_Strict):
    after_harmful_uses: int


class MemoryPolicy(_Strict):
    write: MemoryWrite
    retrieve: MemoryRetrieve
    inject: MemoryInject
    consolidate: MemoryConsolidate
    invalidate: MemoryInvalidate


class SourceRef(_Strict):
    version: str
    source_hash: str


class Protected(_Strict):
    model_allowlist: list[str]
    allowed_actions: list[str]
    token_budget_per_workload: int
    scorer_ref: SourceRef
    gate_ref: SourceRef


class HarnessSpec(_Strict):
    planner_rules: list[str]
    critic_rubric: list[str]
    context_policy: ContextPolicy
    retry_policy: RetryPolicy
    critic_policy: CriticPolicy
    model_routing: ModelRouting
    memory_policy: MemoryPolicy
    protected: Protected

    def to_doc(self) -> dict:
        """Plain JSON-able dict with wire names (`memory_policy.write.from`)."""
        return self.model_dump(mode="json", by_alias=True)

    @classmethod
    def from_doc(cls, doc: dict) -> HarnessSpec:
        return cls.model_validate(doc)


Status = Literal["candidate", "accepted", "rejected"]


class Scores(BaseModel):
    train: float | None = None
    holdout: float | None = None
    canary_pass: float | None = None
    tokens_per_workload: float | None = None
    violations: int | None = None  # inner violations on train + holdout


class HarnessDoc(BaseModel):
    """One document in `harness_versions`."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="_id")
    version: int
    parent: str | None = None
    status: Status = "candidate"
    created_by: str = "proposer"
    rationale: str = ""
    patch: list[PatchOp] = []
    spec: HarnessSpec
    scores: Scores = Scores()
    reason_codes: list[str] = []
    models_used: dict[str, str] = {}
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_doc(self) -> dict:
        d = self.model_dump(mode="python", by_alias=True, exclude={"spec"})
        d["spec"] = self.spec.to_doc()
        return d

    @classmethod
    def from_doc(cls, doc: dict) -> HarnessDoc:
        return cls.model_validate(doc)


def version_id(v: int | str) -> str:
    """Normalise 7, "7" or "h_0007" to "h_0007"."""
    if isinstance(v, str) and v.startswith("h_"):
        v = v[2:]
    n = int(v)
    if n < 0:
        raise ValueError("version must be >= 0")
    return f"h_{n:04d}"
