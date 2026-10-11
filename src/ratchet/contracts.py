"""Shared contracts (SPEC §10). Everyone codes against this file; only Track A edits it."""

from typing import Any, Literal

from pydantic import BaseModel

Split = Literal["train", "holdout", "canary"]


class IndexDef(BaseModel):
    keys: list[tuple[str, int | str]]  # [("genres",1),("imdb.rating",-1)]
    name: str | None = None
    partialFilterExpression: dict | None = None  # noqa: N815 (MongoDB option name)


class QuerySpec(BaseModel):
    id: str
    kind: Literal["find", "aggregate"]
    collection: str
    filter: dict = {}
    projection: dict | None = None
    sort: list[tuple[str, int]] | None = None
    limit: int | None = None
    pipeline: list[dict] | None = None
    freq: float = 1.0


class Workload(BaseModel):
    id: str
    split: Split
    db: str
    collections: list[str]
    baseline_indexes: dict[str, list[IndexDef]] = {}
    write_rate: dict[str, float] = {}
    index_budget: dict[str, int] = {}
    queries: list[QuerySpec]
    notes: str = ""


class ExplainSummary(BaseModel):
    query_id: str
    stages: list[str]  # e.g. ["IXSCAN","FETCH"] or ["COLLSCAN","SORT"]
    n_returned: int
    keys_examined: int
    docs_examined: int
    exec_ms: int
    has_in_memory_sort: bool
    indexes_used: list[str]
    lookup_examined: int = 0
    raw: dict | None = None  # trimmed raw explain, for traces


class Finding(BaseModel):
    query_id: str
    detector: str
    severity: Literal["high", "medium", "low"]
    evidence: dict
    hint: str


class Action(BaseModel):
    type: Literal["create_index", "drop_index", "rewrite_query"]
    collection: str | None = None
    index: IndexDef | None = None  # create_index
    index_name: str | None = None  # drop_index
    query_id: str | None = None  # rewrite_query
    new_query: QuerySpec | None = None
    rationale: str = ""


class TuningPlan(BaseModel):
    actions: list[Action]
    summary: str = ""


class Measurement(BaseModel):
    workload_id: str
    per_query: dict[str, ExplainSummary]
    workload_cost: float
    result_hashes: dict[str, str]
    data_checksums: dict[str, str]
    indexes: dict[str, list[str]]


class ScoreReport(BaseModel):
    improvement: float
    index_penalty: float
    score: float
    violations: list[str]  # inner violation codes


class TuneResult(BaseModel):
    workload_id: str
    split: Split
    harness_version: str
    plan: TuningPlan
    before: Measurement
    after: Measurement
    score: ScoreReport
    findings: list[Finding]
    tokens: int
    rounds: int
    trace: list[dict]
    self_diagnostics: list[dict] = []
    lessons_used: list[str] = []
    models_used: dict[str, str] = {}


class PatchOp(BaseModel):
    op: Literal["replace", "add", "remove"]
    path: str
    value: Any = None


class ProposerOutput(BaseModel):
    diagnosis: str
    hypothesis: str
    patch: list[PatchOp]
    expected_effect: str


class GateDecision(BaseModel):
    candidate: str
    parent: str
    verdict: Literal["ACCEPT", "REJECT"]
    reason_codes: list[str]
    metrics: dict[str, Any]
