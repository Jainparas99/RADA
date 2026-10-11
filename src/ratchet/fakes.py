"""Network-free stand-ins for Track B (`FakeEnv`) and Track C (`FakeLLM`).

`FakeEnv` implements the Track B function interface from SPEC §10 with a small deterministic
cost model, so the agent loop, the gate and `evolve` can be built and tested before the real
environment lands. Numbers are canned, not realistic: 10k-doc collections, COLLSCAN unless an
index leads with a filter field.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from pydantic import BaseModel

from ratchet.contracts import (
    Action,
    ExplainSummary,
    Finding,
    IndexDef,
    Measurement,
    QuerySpec,
    ScoreReport,
    Split,
    TuneResult,
    TuningPlan,
    Workload,
)
from ratchet.harness.models import HarnessSpec

__all__ = [
    "INNER_VIOLATIONS",
    "FakeApplyReport",
    "FakeBackend",
    "FakeEnv",
    "FakeLLM",
    "FakeLLMResponse",
    "FakeScratch",
    "fake_tune",
    "fake_workloads",
    "heuristic_plan",
    "index_name",
    "sample_workload",
]

INNER_VIOLATIONS = [
    "RESULT_MISMATCH",
    "DATA_MUTATED",
    "INDEX_BUDGET_EXCEEDED",
    "FORBIDDEN_ACTION",
    "APPLY_ERROR",
]

_FORBIDDEN_STAGES = {"$out", "$merge"}
_RANGE_OPS = {"$gt", "$gte", "$lt", "$lte", "$ne", "$in", "$nin"}


def _sha(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def index_name(ix: IndexDef) -> str:
    return ix.name or "_".join(f"{k}_{d}" for k, d in ix.keys)


# --------------------------------------------------------------------------- Track B types


class FakeScratch(BaseModel):
    """Stand-in for Track B's `ScratchEnv`."""

    ns: str
    workload_id: str
    indexes: dict[str, list[IndexDef]]
    queries: dict[str, QuerySpec]
    data_version: dict[str, int] = {}


class FakeApplyReport(BaseModel):
    """Stand-in for Track B's `ApplyReport`."""

    applied: list[Action] = []
    created: dict[str, list[str]] = {}
    dropped: dict[str, list[str]] = {}
    violations: list[str] = []
    errors: list[str] = []


# --------------------------------------------------------------------------- FakeEnv


def _filter_fields(q: QuerySpec) -> list[str]:
    if q.kind == "find":
        return [k for k in q.filter if not k.startswith("$")]
    fields: list[str] = []
    for stage in q.pipeline or []:
        if "$match" in stage:
            fields += [k for k in stage["$match"] if not k.startswith("$")]
    return fields


def _sort_fields(q: QuerySpec) -> list[str]:
    if q.kind == "find":
        return [f for f, _ in (q.sort or []) if f != "_id"]
    out: list[str] = []
    for stage in q.pipeline or []:
        if "$sort" in stage:
            out += [f for f in stage["$sort"] if f != "_id"]
    return out


def _late_match(q: QuerySpec) -> bool:
    seen_blocker = False
    for stage in q.pipeline or []:
        if any(k in stage for k in ("$lookup", "$unwind", "$group")):
            seen_blocker = True
        if "$match" in stage and seen_blocker:
            return True
    return False


class FakeEnv:
    """Deterministic in-memory environment with the Track B interface."""

    def __init__(self, collection_docs: int = 10_000, index_cost: float = 0.02) -> None:
        self.collection_docs = collection_docs
        self.index_cost = index_cost
        self.prepared: list[str] = []

    # -- interface -----------------------------------------------------------------

    def prepare_scratch(self, workload: Workload, ns: str = "test") -> FakeScratch:
        self.prepared.append(workload.id)
        return FakeScratch(
            ns=ns,
            workload_id=workload.id,
            indexes={
                c: [ix.model_copy(deep=True) for ix in workload.baseline_indexes.get(c, [])]
                for c in workload.collections
            },
            queries={q.id: q.model_copy(deep=True) for q in workload.queries},
            data_version={c: 0 for c in workload.collections},
        )

    def measure(
        self, env: FakeScratch, workload: Workload, queries: list[QuerySpec] | None = None
    ) -> Measurement:
        current = queries if queries is not None else list(env.queries.values())
        per_query = {q.id: self._explain(env, q) for q in current}
        cost = 0.0
        for q in current:
            s = per_query[q.id]
            c = s.keys_examined + s.docs_examined + s.lookup_examined
            if s.has_in_memory_sort:
                c += s.docs_examined
            cost += q.freq * c
        return Measurement(
            workload_id=workload.id,
            per_query=per_query,
            workload_cost=cost,
            result_hashes={q.id: self._result_hash(q) for q in current},
            data_checksums={c: _sha([c, v]) for c, v in sorted(env.data_version.items())},
            indexes={
                c: ["_id_"] + [index_name(ix) for ix in ixs] for c, ixs in env.indexes.items()
            },
        )

    def run_detectors(self, env: FakeScratch, workload: Workload, m: Measurement) -> list[Finding]:
        out: list[Finding] = []
        for qid, s in m.per_query.items():
            q = env.queries.get(qid)
            if "COLLSCAN" in s.stages:
                out.append(
                    Finding(
                        query_id=qid,
                        detector="COLLSCAN_LARGE",
                        severity="high",
                        evidence={"docs_examined": s.docs_examined, "n_returned": s.n_returned},
                        hint="add index",
                    )
                )
            if s.has_in_memory_sort:
                out.append(
                    Finding(
                        query_id=qid,
                        detector="IN_MEMORY_SORT",
                        severity="medium",
                        evidence={"stages": s.stages},
                        hint="put sort field in index (equality → sort → range)",
                    )
                )
            if q is not None and _late_match(q):
                out.append(
                    Finding(
                        query_id=qid,
                        detector="LATE_MATCH",
                        severity="medium",
                        evidence={"lookup_examined": s.lookup_examined},
                        hint="move $match earlier",
                    )
                )
        return out

    def apply_plan(self, env: FakeScratch, workload: Workload, plan: TuningPlan) -> FakeApplyReport:
        rep = FakeApplyReport()
        for a in plan.actions:
            err = self._apply_one(env, workload, a, rep)
            if err is None:
                rep.applied.append(a)
            else:
                code, msg = err
                if code not in rep.violations:
                    rep.violations.append(code)
                rep.errors.append(f"{code}: {msg}")
        return rep

    def score(
        self,
        workload: Workload,
        before: Measurement,
        after: Measurement,
        plan: TuningPlan,
        apply: FakeApplyReport,
    ) -> ScoreReport:
        violations = list(apply.violations)
        if any(after.result_hashes.get(q) != h for q, h in before.result_hashes.items()):
            violations.append("RESULT_MISMATCH")
        if after.data_checksums != before.data_checksums:
            violations.append("DATA_MUTATED")
        for c, names in after.indexes.items():
            budget = workload.index_budget.get(c)
            if budget is not None and len([n for n in names if n != "_id_"]) > budget:
                violations.append("INDEX_BUDGET_EXCEEDED")
                break
        if before.workload_cost > 0:
            improvement = max(-1.0, min(1.0, 1 - after.workload_cost / before.workload_cost))
        else:
            improvement = 0.0
        n_new = sum(len(v) for v in apply.created.values())
        n_dropped = sum(len(v) for v in apply.dropped.values())
        penalty = self.index_cost * (n_new - n_dropped)
        violations = [v for v in INNER_VIOLATIONS if v in violations]
        return ScoreReport(
            improvement=improvement,
            index_penalty=penalty,
            score=0.0 if violations else improvement - penalty,
            violations=violations,
        )

    def query_shape(self, q: QuerySpec) -> str:
        def kind(v: Any) -> str:
            if isinstance(v, dict) and any(k in _RANGE_OPS for k in v):
                return "RANGE"
            if isinstance(v, dict) and "$regex" in v:
                return "REGEX"
            return "EQ"

        if q.kind == "find":
            flt = ", ".join(f"{k}: {kind(v)}" for k, v in sorted(q.filter.items()))
            srt = ", ".join(f"{f}: {'ASC' if d > 0 else 'DESC'}" for f, d in (q.sort or []))
            proj = ", ".join(sorted(q.projection or {}))
            return f"{q.collection} | filter {{{flt}}} | sort {{{srt}}} | proj {{{proj}}}"
        stages = " > ".join(next(iter(s)) for s in (q.pipeline or []))
        return f"{q.collection} | pipeline {stages}"

    # -- test helpers --------------------------------------------------------------

    def mutate_data(self, env: FakeScratch, collection: str) -> None:
        """Simulate a write to scratch data (should trigger DATA_MUTATED)."""
        env.data_version[collection] = env.data_version.get(collection, 0) + 1

    # -- internals -----------------------------------------------------------------

    def _apply_one(
        self, env: FakeScratch, workload: Workload, a: Action, rep: FakeApplyReport
    ) -> tuple[str, str] | None:
        if a.type == "create_index":
            if a.collection not in env.indexes or a.index is None or not a.index.keys:
                return "APPLY_ERROR", "create_index needs a workload collection and keys"
            name = index_name(a.index)
            if name in {index_name(ix) for ix in env.indexes[a.collection]}:
                return "APPLY_ERROR", f"index {name} already exists"
            env.indexes[a.collection].append(a.index.model_copy(update={"name": name}))
            rep.created.setdefault(a.collection, []).append(name)
            return None
        if a.type == "drop_index":
            if a.index_name == "_id_":
                return "FORBIDDEN_ACTION", "cannot drop _id_"
            ixs = env.indexes.get(a.collection or "", [])
            keep = [ix for ix in ixs if index_name(ix) != a.index_name]
            if a.collection not in env.indexes or len(keep) == len(ixs):
                return "APPLY_ERROR", f"no index {a.index_name} on {a.collection}"
            env.indexes[a.collection] = keep
            rep.dropped.setdefault(a.collection, []).append(a.index_name or "")
            return None
        # rewrite_query
        old = env.queries.get(a.query_id or "")
        new = a.new_query
        if old is None or new is None:
            return "APPLY_ERROR", f"unknown query {a.query_id} or missing new_query"
        if new.kind != old.kind or new.collection != old.collection:
            return "FORBIDDEN_ACTION", "rewrite may not change kind or collection"
        if any(k in _FORBIDDEN_STAGES for s in (new.pipeline or []) for k in s):
            return "FORBIDDEN_ACTION", "$out/$merge are not allowed"
        env.queries[old.id] = new.model_copy(update={"id": old.id, "freq": old.freq})
        return None

    def _explain(self, env: FakeScratch, q: QuerySpec) -> ExplainSummary:
        n = self.collection_docs
        n_ret = q.limit or 10
        ffields, sfields = _filter_fields(q), _sort_fields(q)
        best: IndexDef | None = None
        best_hits = 0
        for ix in env.indexes.get(q.collection, []):
            keys = [k for k, _ in ix.keys]
            if keys and keys[0] in ffields:
                hits = len([k for k in keys if k in ffields])
                if hits > best_hits:
                    best, best_hits = ix, hits
        if best is None:
            stages, keys_ex, docs_ex, used = ["COLLSCAN"], 0, n, []
            sort_in_mem = bool(sfields)
        else:
            covered_filter = best_hits >= len(set(ffields))
            keys_ex = docs_ex = n_ret if covered_filter else n_ret * 20
            stages, used = ["IXSCAN", "FETCH"], [index_name(best)]
            ix_keys = {k for k, _ in best.keys}
            sort_in_mem = bool(sfields) and not set(sfields) <= ix_keys
        if sort_in_mem:
            stages.append("SORT")
        lookup = 0
        if q.kind == "aggregate" and any("$lookup" in s for s in q.pipeline or []):
            lookup = n if _late_match(q) else n_ret
        return ExplainSummary(
            query_id=q.id,
            stages=stages,
            n_returned=n_ret,
            keys_examined=keys_ex,
            docs_examined=docs_ex,
            exec_ms=max(1, (keys_ex + docs_ex + lookup) // 1000),
            has_in_memory_sort=sort_in_mem,
            indexes_used=used,
            lookup_examined=lookup,
        )

    @staticmethod
    def _result_hash(q: QuerySpec) -> str:
        """Semantic fingerprint: stable under stage reordering, changes if any predicate does."""
        if q.kind == "find":
            return _sha([q.collection, q.filter, q.sort, q.limit, sorted(q.projection or {})])
        return _sha([q.collection, sorted(json.dumps(s, sort_keys=True) for s in q.pipeline or [])])


def sample_workload(
    wid: str = "fake_w01",
    split: str = "train",
    coll: str = "movies",
    eq: str = "genres",
    rng: str = "year",
    sort: str = "rating",
    child: str = "comments",
    fk: str = "movie_id",
    budget: int = 3,
) -> Workload:
    """A small workload with a missing index, an in-memory sort and a late $match."""
    return Workload.model_validate(
        {
            "id": wid,
            "split": split,
            "db": "fake_db",
            "collections": [coll, child],
            "baseline_indexes": {coll: [{"keys": [["title", 1]], "name": "title_1"}]},
            "write_rate": {coll: 2.0, child: 40.0},
            "index_budget": {coll: budget, child: 2},
            "queries": [
                {
                    "id": "q1",
                    "kind": "find",
                    "collection": coll,
                    "filter": {eq: "Drama", rng: {"$gt": 2000}},
                    "sort": [[sort, -1], ["_id", 1]],
                    "limit": 10,
                    "projection": {"title": 1, sort: 1},
                    "freq": 50,
                },
                {
                    "id": "q2",
                    "kind": "aggregate",
                    "collection": child,
                    "pipeline": [
                        {
                            "$lookup": {
                                "from": coll,
                                "localField": fk,
                                "foreignField": "_id",
                                "as": "m",
                            }
                        },
                        {"$unwind": "$m"},
                        {"$match": {fk: 7}},
                        {"$count": "n"},
                    ],
                    "freq": 5,
                },
            ],
            "notes": "fake workload for tests",
        }
    )


def fake_workloads() -> list[Workload]:
    """3 train, 2 holdout (different collections), 1 canary with a tight index budget."""
    return [
        sample_workload("fake_train_01", "train", "movies", "genres", "year", "rating"),
        sample_workload("fake_train_02", "train", "listings", "city", "price", "score", "reviews",
                        "listing_id"),
        sample_workload("fake_train_03", "train", "restaurants", "cuisine", "opened", "grade",
                        "inspections", "restaurant_id"),
        sample_workload("fake_holdout_01", "holdout", "sales", "store", "sold_at", "amount",
                        "line_items", "sale_id"),
        sample_workload("fake_holdout_02", "holdout", "accounts", "tier", "balance", "opened_on",
                        "transactions", "account_id"),
        sample_workload("fake_canary_01", "canary", "trips", "station", "duration", "started_at",
                        "stops", "trip_id", budget=1),
    ]  # fmt: skip


# --------------------------------------------------------------------------- FakeLLM


class FakeLLMResponse(BaseModel):
    """Stand-in for Track C's `LLMResponse`."""

    text: str
    data: Any = None  # parsed JSON, when the scripted reply was JSON
    role: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    tool_calls: list[dict] = []

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


Scripted = Any  # dict | list | str | BaseModel | Callable[[dict], Scripted]


class FakeLLM:
    """Returns scripted replies and records every call.

    `script` is either a FIFO list used for all roles, or `{role: [replies...]}`. A reply may be
    JSON-like (dict/list), a string, a pydantic model, or a callable taking the recorded call.
    When a queue runs dry `default` is returned; with no default, `complete` raises.
    """

    def __init__(
        self,
        script: list[Scripted] | dict[str, list[Scripted]] | None = None,
        default: Scripted | None = None,
        embed_dim: int = 32,
    ) -> None:
        self._by_role: dict[str, list[Scripted]] = {}
        self._fifo: list[Scripted] = []
        if isinstance(script, dict):
            self._by_role = {r: list(v) for r, v in script.items()}
        elif script:
            self._fifo = list(script)
        self.default = default
        self.embed_dim = embed_dim
        self.calls: list[dict] = []

    def complete(
        self,
        role: str,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        json_schema: dict | None = None,
    ) -> FakeLLMResponse:
        call = {
            "role": role,
            "model": model,
            "messages": messages,
            "tools": tools,
            "json_schema": json_schema,
        }
        self.calls.append(call)
        queue = self._by_role.get(role)
        if queue:
            reply = queue.pop(0)
        elif self._fifo:
            reply = self._fifo.pop(0)
        elif self.default is not None:
            reply = self.default
        else:
            raise AssertionError(f"FakeLLM: no scripted reply left for role {role!r}")
        if callable(reply) and not isinstance(reply, BaseModel):
            reply = reply(call)
        if isinstance(reply, BaseModel):
            reply = reply.model_dump(mode="json")
        if isinstance(reply, str):
            text, data = reply, None
            try:
                data = json.loads(reply)
            except ValueError:
                pass
        else:
            text, data = json.dumps(reply), reply
        return FakeLLMResponse(
            text=text,
            data=data,
            role=role,
            model=model,
            input_tokens=sum(len(json.dumps(m, default=str)) for m in messages) // 4,
            output_tokens=len(text) // 4,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Hashed bag-of-words, L2-normalised: similar texts get similar vectors."""
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self.embed_dim
            for tok in re.findall(r"[a-z0-9_.$]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.embed_dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out

    def prompts(self, role: str | None = None) -> str:
        """Everything sent to the model (optionally for one role), as one string."""
        calls = [c for c in self.calls if role is None or c["role"] == role]
        return "\n".join(json.dumps(c["messages"], default=str) for c in calls)


# --------------------------------------------------------------------------- fake agent


def _rules_mention(harness: HarnessSpec, *words: str) -> bool:
    text = " ".join(harness.planner_rules).casefold()
    return any(w.casefold() in text for w in words)


def heuristic_plan(workload: Workload, harness: HarnessSpec, scratch: FakeScratch) -> TuningPlan:
    """A scripted "planner" whose quality depends on the harness, so the outer loop has
    something to improve: ESR indexes need findings in context or a rule about sort order,
    $match push-down needs a rule about $match, and respecting budgets needs a rule about it."""
    knows_esr = harness.context_policy.include_detector_findings or _rules_mention(
        harness, "sort", "ESR"
    )
    knows_match = _rules_mention(harness, "$match")
    knows_budget = _rules_mention(harness, "budget")
    actions: list[Action] = []
    for q in workload.queries:
        fields = _filter_fields(q)
        if q.kind == "find" and fields:
            used = len(scratch.indexes.get(q.collection, []))
            budget = workload.index_budget.get(q.collection)
            if knows_budget and budget is not None and used >= budget:
                continue
            eq = [f for f in fields if not isinstance(q.filter[f], dict)]
            rng = [f for f in fields if isinstance(q.filter[f], dict)]
            keys = [*eq, *_sort_fields(q), *rng] if knows_esr else fields[:1]
            dirs = dict(q.sort or [])
            actions.append(
                Action(
                    type="create_index",
                    collection=q.collection,
                    index=IndexDef(keys=[(k, dirs.get(k, 1)) for k in keys]),
                    rationale="filter fields" + (" in ESR order" if knows_esr else ""),
                )
            )
        if q.kind == "aggregate" and knows_match and _late_match(q):
            pipe = q.pipeline or []
            early = [s for s in pipe if "$match" in s] + [s for s in pipe if "$match" not in s]
            actions.append(
                Action(
                    type="rewrite_query",
                    query_id=q.id,
                    new_query=q.model_copy(update={"pipeline": early}),
                    rationale="move $match before $lookup",
                )
            )
    return TuningPlan(actions=actions, summary=f"{len(actions)} action(s)")


def fake_tune(
    workload: Workload,
    harness: HarnessSpec,
    env: FakeEnv | None = None,
    mode: str = "normal",
    llm: FakeLLM | None = None,
) -> TuneResult:
    """Stand-in for Track C's `tune()`: diagnose → plan (via FakeLLM) → apply → measure → score."""
    env = env or FakeEnv()
    scratch = env.prepare_scratch(workload)
    llm = llm or FakeLLM(default=lambda call: heuristic_plan(workload, harness, scratch))
    before = env.measure(scratch, workload)
    findings = env.run_detectors(scratch, workload, before)
    context = {
        "rules": harness.planner_rules,
        "queries": [q.model_dump(mode="json") for q in workload.queries],
        "indexes": before.indexes if harness.context_policy.include_existing_indexes else None,
        "findings": [f.model_dump() for f in findings]
        if harness.context_policy.include_detector_findings
        else None,
    }
    messages = [{"role": "user", "content": json.dumps(context)}]
    reply = llm.complete("planner", harness.model_routing.planner, messages)
    plan = TuningPlan.model_validate(reply.data)
    apply = env.apply_plan(scratch, workload, plan)
    after = env.measure(scratch, workload)
    report = env.score(workload, before, after, plan, apply)
    return TuneResult(
        workload_id=workload.id,
        split=workload.split,
        harness_version="",
        plan=plan,
        before=before,
        after=after,
        score=report,
        findings=findings,
        tokens=reply.tokens,
        rounds=1,
        trace=[
            {"step": "context", "messages": messages},
            {"step": "plan", "plan": plan.model_dump(mode="json")},
            {"step": "apply", "report": apply.model_dump(mode="json")},
            {"step": "score", "report": report.model_dump()},
        ],
        models_used={"planner": harness.model_routing.planner},
    )


class FakeBackend:
    """`harness.runner.Backend` over FakeEnv + fake_tune. No network, no database."""

    name = "fake"

    def __init__(self, workloads: list[Workload] | None = None, env: FakeEnv | None = None) -> None:
        self._workloads = workloads if workloads is not None else fake_workloads()
        self.env = env or FakeEnv()
        self.modes: list[tuple[str, str]] = []  # (workload_id, mode) for every tune call

    def workloads(self, split: Split | None = None) -> list[Workload]:
        return [w for w in self._workloads if split is None or w.split == split]

    def tune(self, workload: Workload, harness: HarnessSpec, mode: str) -> TuneResult:
        self.modes.append((workload.id, mode))
        return fake_tune(workload, harness, self.env, mode)
