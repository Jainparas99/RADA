"""Validate and apply a TuningPlan on a scratch copy (SPEC §5.5).

The agent may only ``create_index``, ``drop_index`` and ``rewrite_query``. Everything
is validated before anything touches the database; a plan with any forbidden action is
not applied at all. Validation is pure (no DB) so it is cheap to test.

Rules beyond the SPEC table, each closing a concrete cheat:

* ``rewrite_query`` keeps ``id``, ``collection`` and ``kind``. A find rewrite may only
  change ``projection`` key order; ``filter``, ``sort`` and ``limit`` must be identical.
* A rewrite may not introduce constants that are not in the original query
  (``NEW_LITERAL``): that blocks smuggling precomputed answers in, e.g. a
  ``{$match: {_id: {$in: [<the 10 answer ids>]}}}`` rewrite that returns the same results.
* ``$lookup`` / ``$graphLookup`` / ``$unionWith`` may only reference workload collections.
* Frequencies always come from the workload; a rewrite's ``freq`` is ignored.

Added after Track A's cross-review (docs/XREVIEW_TRACK_B.md):

* ``NEW_OPERATOR``: a rewrite may only add the restructuring operators in
  ``RESTRUCTURING_OPERATORS``. Arithmetic, string and conversion operators, ``$literal``,
  ``$range`` and the like can build any constant out of 0/1 and existing text, which defeats
  ``NEW_LITERAL``.
* ``CONSTANT_FIELD``: a rewrite may not add a computed field with no field reference in it.
* ``DROPPED_LITERAL``: every constant of the original must still be in the rewrite, so a
  ``$match`` condition cannot quietly disappear.
* ``SORT_CHANGED``: a rewrite keeps every sort key; the ``_id`` tie-break cannot be dropped.
* Wildcard index fields (``$**``) are refused: one slot must not index every field.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError
from pymongo.database import Database
from pymongo.errors import PyMongoError

from ratchet.contracts import Action, IndexDef, QuerySpec, TuningPlan, Workload
from ratchet.env.explain import explain_command
from ratchet.env.measure import canonical
from ratchet.env.scratch import ScratchEnv, create_index

FORBIDDEN_ACTION = "FORBIDDEN_ACTION"
APPLY_ERROR = "APPLY_ERROR"
INDEX_BUDGET_EXCEEDED = "INDEX_BUDGET_EXCEEDED"

ALLOWED_ACTIONS = ("create_index", "drop_index", "rewrite_query")
ALLOWED_INDEX_FIELDS = {"keys", "name", "partialFilterExpression"}
FORBIDDEN_INDEX_OPTIONS = {
    "unique",
    "sparse",
    "hidden",
    "expireAfterSeconds",
    "collation",
    "prepareUnique",
    "clustered",
    "storageEngine",
    "wildcardProjection",
    "weights",
    "default_language",
    "language_override",
}
ALLOWED_KEY_VALUES: set[Any] = {1, -1, "text", "2dsphere", "2d", "hashed"}
FORBIDDEN_STAGES = {
    "$out",
    "$merge",
    "$documents",
    "$currentOp",
    "$collStats",
    "$indexStats",
    "$listSessions",
    "$listLocalSessions",
    "$planCacheStats",
    "$changeStream",
    "$changeStreamSplitLargeEvent",
    "$sample",
    "$search",
    "$searchMeta",
    "$vectorSearch",
    "$listSearchIndexes",
    "$queryStats",
    "$listSampledQueries",
    "$shardedDataDistribution",
}
FORBIDDEN_OPERATORS = {"$where", "$function", "$accumulator", "$rand", "$sampleRate"}
FORBIDDEN_VARS = {"$$NOW", "$$CLUSTER_TIME", "$$USER_ROLES", "$$SEARCH_META"}
# Operators a rewrite may introduce even if the original query does not use them: enough to
# move, split and narrow stages, and to turn a $lookup into its pipeline form.
RESTRUCTURING_OPERATORS = {
    "$match",
    "$sort",
    "$limit",
    "$skip",
    "$project",
    "$unset",
    "$addFields",
    "$set",
    "$unwind",
    "$lookup",
    "$expr",
    "$eq",
    "$and",
}
_COMPUTING_STAGES = ("$addFields", "$set", "$project")
_FOREIGN_KEYS = {"$lookup": "from", "$graphLookup": "from", "$unionWith": "coll"}
_FREE_NUMBERS = {0, 1, -1}


class ActionError(BaseModel):
    action_index: int
    code: str  # FORBIDDEN_ACTION | APPLY_ERROR | INDEX_BUDGET_EXCEEDED
    detail: str
    # hard: nothing is applied. soft (semantic rewrite checks): the rewritten query is
    # still measured, read-only, so a changed result also shows up as RESULT_MISMATCH.
    hard: bool = True


class ApplyReport(BaseModel):
    """Field names match tests/fakes.py FakeApplyReport (applied, created, dropped,
    violations, errors) plus the extras the scorer and traces need."""

    ok: bool
    applied: list[Action] = []
    created: dict[str, list[str]] = {}
    dropped: dict[str, list[str]] = {}
    violations: list[str] = []  # distinct inner violation codes
    errors: list[str] = []  # "CODE: detail"
    error_details: list[ActionError] = []
    rewritten: list[str] = []
    effective_queries: list[QuerySpec] = []  # what measure() runs next (also stored on env)
    index_sizes_mb: dict[str, dict[str, float]] = {}


# --------------------------------------------------------------------------- raw checks


def _raw_index_option_errors(index: Any) -> list[str]:
    if not isinstance(index, dict):
        return []
    bad = sorted(set(index) - ALLOWED_INDEX_FIELDS)
    return [f"index option {k!r} not allowed" for k in bad]


def parse_plan(plan: TuningPlan | dict) -> tuple[TuningPlan | None, list[ActionError]]:
    """Accept a TuningPlan or raw planner JSON. Raw JSON is checked for forbidden keys
    *before* pydantic parsing, because pydantic drops unknown fields such as ``unique``."""
    errors: list[ActionError] = []
    if isinstance(plan, TuningPlan):
        for i, a in enumerate(plan.actions):
            extra = dict(a.index.model_extra or {}) if a.index is not None else {}
            for k in extra:
                errors.append(
                    ActionError(
                        action_index=i,
                        code=FORBIDDEN_ACTION,
                        detail=f"index option {k!r} not allowed",
                    )
                )
            for k in a.model_extra or {}:
                errors.append(
                    ActionError(
                        action_index=i,
                        code=FORBIDDEN_ACTION,
                        detail=f"action field {k!r} not allowed",
                    )
                )
        return plan, errors
    if not isinstance(plan, dict) or not isinstance(plan.get("actions"), list):
        return None, [
            ActionError(action_index=-1, code=APPLY_ERROR, detail="plan has no actions list")
        ]
    for i, a in enumerate(plan["actions"]):
        if not isinstance(a, dict):
            errors.append(
                ActionError(action_index=i, code=APPLY_ERROR, detail="action is not an object")
            )
            continue
        if a.get("type") not in ALLOWED_ACTIONS:
            errors.append(
                ActionError(
                    action_index=i,
                    code=FORBIDDEN_ACTION,
                    detail=f"action type {a.get('type')!r} not allowed",
                )
            )
        extra = set(a) - set(Action.model_fields)
        for k in sorted(extra):
            errors.append(
                ActionError(
                    action_index=i, code=FORBIDDEN_ACTION, detail=f"action field {k!r} not allowed"
                )
            )
        for msg in _raw_index_option_errors(a.get("index")):
            errors.append(ActionError(action_index=i, code=FORBIDDEN_ACTION, detail=msg))
    if errors:
        return None, errors
    try:
        return TuningPlan.model_validate(plan), []
    except ValidationError as e:
        return None, [ActionError(action_index=-1, code=APPLY_ERROR, detail=f"invalid plan: {e}")]


# --------------------------------------------------------------------------- pipeline checks


def _scan(obj: Any, workload: Workload, where: str, out: list[str]) -> None:
    """Recursively look for forbidden stages, operators, variables and foreign collections."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_STAGES:
                out.append(f"forbidden stage {k} in {where}")
            if k in FORBIDDEN_OPERATORS:
                out.append(f"forbidden operator {k} in {where}")
            if k in _FOREIGN_KEYS and isinstance(v, dict):
                target = v.get(_FOREIGN_KEYS[k])
                if isinstance(target, dict):  # {$lookup: {from: {db, coll}}}
                    out.append(f"{k} into another database in {where}")
                elif target is not None and target not in workload.collections:
                    out.append(f"{k} into collection {target!r} outside the workload in {where}")
            _scan(v, workload, where, out)
    elif isinstance(obj, list):
        for v in obj:
            _scan(v, workload, where, out)
    elif isinstance(obj, str):
        for var in FORBIDDEN_VARS:
            if obj == var or obj.startswith(var + "."):
                out.append(f"forbidden variable {var} in {where}")


def _atoms(obj: Any, out: set[tuple[str, str]]) -> None:
    """Constants in value positions (keys are skipped). Strings are split into path
    components with leading ``$`` stripped, so moving ``$match`` across ``$lookup``
    (``m.year`` -> ``year``) does not count as a new constant."""
    if isinstance(obj, dict):
        for v in obj.values():
            _atoms(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _atoms(v, out)
    elif obj is None or isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        if obj not in _FREE_NUMBERS:
            out.add(("n", repr(float(obj))))
    elif isinstance(obj, str):
        for part in obj.lstrip("$").split("."):
            if part:
                out.add(("s", part))
    else:
        out.add(("o", canonical({"v": obj})))


def query_atoms(q: QuerySpec) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    _atoms([q.filter, q.projection, q.sort, q.limit, q.pipeline], out)
    return out


def _operators(obj: Any, out: set[str]) -> None:
    """Every ``$``-prefixed key (stage or operator) anywhere in a query."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("$"):
                out.add(k)
            _operators(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _operators(v, out)


def query_operators(q: QuerySpec) -> set[str]:
    out: set[str] = set()
    _operators([q.filter, q.projection, q.pipeline], out)
    return out


def _has_field_ref(obj: Any) -> bool:
    if isinstance(obj, str):
        return obj.startswith("$")
    if isinstance(obj, dict):
        return any(_has_field_ref(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_field_ref(v) for v in obj)
    return False


def _constant_fields(new: QuerySpec, original: QuerySpec) -> list[str]:
    """Fields a rewrite computes from constants alone, e.g. ``{$addFields: {n: 1}}`` or
    ``{$project: {n: {...no field reference...}}}``. Stages copied from the original are
    not looked at."""
    known = {canonical(st) for st in (original.pipeline or [])}
    out: list[str] = []
    for st in new.pipeline or []:
        if not isinstance(st, dict) or canonical(st) in known:
            continue
        for stage in _COMPUTING_STAGES:
            body = st.get(stage)
            if not isinstance(body, dict):
                continue
            for field, value in body.items():
                inclusion = stage == "$project" and (isinstance(value, bool) or value in (0, 1))
                if not inclusion and not _has_field_ref(value):
                    out.append(field)
    return out


def _sort_keys(q: QuerySpec) -> tuple[set[tuple[str, Any]], list[set[str]]]:
    """(all (field, direction) pairs sorted on, the field set of each sort) for a query."""
    pairs: set[tuple[str, Any]] = set()
    sorts: list[set[str]] = []
    if q.sort:
        pairs.update((k, v) for k, v in q.sort)
        sorts.append({k for k, _ in q.sort})

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            spec = obj.get("$sort")
            if isinstance(spec, dict):
                pairs.update((k, canonical({"v": v})) for k, v in spec.items())
                sorts.append(set(spec))
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    walk(q.pipeline or [])
    return pairs, sorts


def _sort_problems(original: QuerySpec, new: QuerySpec) -> list[str]:
    o_pairs, o_sorts = _sort_keys(original)
    n_pairs, n_sorts = _sort_keys(new)
    out: list[str] = []
    if original.kind == "find":
        if [tuple(x) for x in (new.sort or [])] != [tuple(x) for x in (original.sort or [])]:
            out.append("SORT_CHANGED: find rewrite changes the sort")
        return out
    missing = sorted(k for k, _ in o_pairs - n_pairs)
    if missing:
        out.append(f"SORT_CHANGED: rewrite drops sort key(s): {', '.join(missing)}")
    if any("_id" in s for s in o_sorts) and any("_id" not in s for s in n_sorts):
        out.append("SORT_CHANGED: rewrite has a $sort without the _id tie-break")
    return out


def validate_rewrite(
    original: QuerySpec, new: QuerySpec, workload: Workload
) -> tuple[list[str], list[str]]:
    """Return (hard, soft) problems. Hard: the rewrite must not run at all. Soft: it is
    read-only and safe to measure, but breaks a rewrite rule (FORBIDDEN_ACTION anyway)."""
    errs: list[str] = []
    soft: list[str] = []
    if new.id != original.id:
        errs.append(f"rewrite changes query id {original.id!r} -> {new.id!r}")
    if new.collection != original.collection:
        errs.append("rewrite changes collection")
    if new.kind != original.kind:
        errs.append("rewrite changes query kind")
    if original.kind == "find":
        if canonical(new.filter or {}) != canonical(original.filter or {}):
            soft.append("find rewrite changes filter")
        if new.limit != original.limit:
            soft.append("find rewrite changes limit")
        if new.pipeline is not None:
            errs.append("find rewrite has a pipeline")
    else:
        p = new.pipeline
        if not isinstance(p, list) or not p:
            errs.append("aggregate rewrite has an empty pipeline")
        else:
            for st in p:
                if not isinstance(st, dict) or len(st) != 1:
                    errs.append("pipeline stage must be a single-key object")
                    break
    _scan([new.filter, new.projection, new.pipeline], workload, f"rewrite of {original.id}", errs)
    old_atoms, cur_atoms = query_atoms(original), query_atoms(new)
    new_atoms = cur_atoms - old_atoms
    if new_atoms:
        shown = ", ".join(sorted(v for _, v in new_atoms)[:5])
        soft.append(f"NEW_LITERAL: rewrite introduces constants not in the original: {shown}")
    # A find rewrite's filter and limit are already pinned above; its projection may shrink.
    lost = old_atoms - cur_atoms if original.kind == "aggregate" else set()
    if lost:
        shown = ", ".join(sorted(v for _, v in lost)[:5])
        soft.append(f"DROPPED_LITERAL: rewrite loses constants of the original: {shown}")
    new_ops = query_operators(new) - query_operators(original) - RESTRUCTURING_OPERATORS
    if new_ops:
        soft.append(f"NEW_OPERATOR: rewrite introduces {', '.join(sorted(new_ops))}")
    constant = _constant_fields(new, original)
    if constant:
        soft.append(f"CONSTANT_FIELD: computed without any field: {', '.join(sorted(constant))}")
    soft.extend(_sort_problems(original, new))
    return errs, soft


def validate_index(ix: IndexDef) -> list[str]:
    errs: list[str] = []
    if not ix.keys:
        errs.append("index has no keys")
    for field, v in ix.keys:
        if v not in ALLOWED_KEY_VALUES or isinstance(v, bool):
            errs.append(f"index key value {v!r} for {field!r} not allowed")
        if not isinstance(field, str) or not field or field.startswith("$"):
            errs.append(f"index field {field!r} not allowed")
        elif "$**" in field:
            errs.append(f"wildcard index field {field!r} not allowed")
    if ix.name == "_id_":
        errs.append("index name _id_ is reserved")
    if ix.partialFilterExpression is not None:
        tmp: list[str] = []
        _scan(
            ix.partialFilterExpression,
            Workload(id="_", split="train", db="_", collections=[], queries=[]),
            "partialFilterExpression",
            tmp,
        )
        errs.extend(e for e in tmp if "outside the workload" not in e)
    return errs


def validate_plan(
    workload: Workload, plan: TuningPlan | dict
) -> tuple[TuningPlan | None, list[ActionError]]:
    """Pure validation of every action against SPEC §5.5. No DB access."""
    parsed, errors = parse_plan(plan)
    if parsed is None:
        return None, errors
    by_id = {q.id: q for q in workload.queries}
    for i, a in enumerate(parsed.actions):

        def bad(msg: str, code: str = FORBIDDEN_ACTION, _i: int = i, hard: bool = True) -> None:
            errors.append(ActionError(action_index=_i, code=code, detail=msg, hard=hard))

        if a.type == "create_index":
            if a.collection not in workload.collections:
                bad(f"create_index on collection {a.collection!r} outside the workload")
            if a.index is None:
                bad("create_index without index", APPLY_ERROR)
            else:
                for msg in validate_index(a.index):
                    bad(msg)
        elif a.type == "drop_index":
            if a.collection not in workload.collections:
                bad(f"drop_index on collection {a.collection!r} outside the workload")
            if not a.index_name:
                bad("drop_index without index_name", APPLY_ERROR)
            elif a.index_name == "_id_":
                bad("dropping _id_ is not allowed")
        elif a.type == "rewrite_query":
            orig = by_id.get(a.query_id or "")
            if orig is None:
                bad(f"rewrite of unknown query {a.query_id!r}")
            elif a.new_query is None:
                bad("rewrite_query without new_query", APPLY_ERROR)
            else:
                hard_errs, soft_errs = validate_rewrite(orig, a.new_query, workload)
                for msg in hard_errs:
                    bad(msg)
                for msg in soft_errs:
                    bad(msg, hard=False)
    return parsed, errors


# --------------------------------------------------------------------------- apply


def _index_sizes_mb(db: Database, coll: str) -> dict[str, float]:
    try:
        st = next(db[coll].aggregate([{"$collStats": {"storageStats": {}}}]))
        sizes = st["storageStats"].get("indexSizes", {})
    except (PyMongoError, StopIteration, KeyError):
        try:
            sizes = db.command("collStats", coll).get("indexSizes", {})
        except PyMongoError:
            sizes = {}
    return {k: float(v) / 1e6 for k, v in sizes.items()}


def effective_queries(
    workload: Workload, plan: TuningPlan, base: list[QuerySpec] | None = None
) -> list[QuerySpec]:
    """Workload queries with this plan's rewrites applied (last rewrite of a query wins).
    Frequencies are always the workload's."""
    qs = {q.id: q for q in (base or workload.queries)}
    freqs = {q.id: q.freq for q in workload.queries}
    for a in plan.actions:
        if a.type == "rewrite_query" and a.new_query is not None and a.query_id in qs:
            qs[a.query_id] = a.new_query.model_copy(update={"freq": freqs[a.query_id]})
    return [qs[q.id] for q in workload.queries]


def apply_plan(
    env: ScratchEnv,
    workload: Workload,
    plan: TuningPlan | dict,
    base_queries: list[QuerySpec] | None = None,
) -> ApplyReport:
    """Validate, then apply index actions to the scratch DB and record rewrites on ``env``.

    Like FakeEnv, the env is stateful: rewrites accumulate in ``env.queries`` and a later
    ``measure(env, workload)`` runs them. ``prepare_scratch``/``reset_indexes`` start over.

    A hard FORBIDDEN_ACTION or projected INDEX_BUDGET_EXCEEDED -> nothing is applied.
    Soft FORBIDDEN_ACTION (rewrite rule broken) -> applied and measured, still a violation.
    A failing index build or invalid rewrite -> APPLY_ERROR, application stops.
    """
    if not env.db.name.startswith("ratchet_scratch_"):
        raise RuntimeError(f"refusing to apply a plan outside a scratch DB: {env.db.name}")
    original = list(base_queries or env.current_queries(workload))
    parsed, errors = validate_plan(workload, plan)

    def report(**kw: Any) -> ApplyReport:
        codes = sorted({e.code for e in errors})
        return ApplyReport(
            ok=not errors,
            violations=codes,
            errors=[f"{e.code}: {e.detail}" for e in errors],
            error_details=list(errors),
            **kw,
        )

    if parsed is None or any(e.hard for e in errors):
        return report(effective_queries=original)

    # Pre-flight against the live scratch state: drops must exist, budget must hold.
    current = {c: {s["name"] for s in env.index_specs(c)} - {"_id_"} for c in workload.collections}
    projected = {c: set(v) for c, v in current.items()}
    for i, a in enumerate(parsed.actions):
        if a.type == "drop_index":
            if a.index_name not in projected[a.collection]:
                errors.append(
                    ActionError(
                        action_index=i,
                        code=APPLY_ERROR,
                        detail=f"index {a.index_name!r} does not exist on {a.collection}",
                    )
                )
            projected[a.collection].discard(a.index_name)
        elif a.type == "create_index":
            projected[a.collection].add(a.index.name or f"__new_{i}")
    for c, names in projected.items():
        budget = workload.index_budget.get(c)
        if budget is not None and len(names) > budget:
            errors.append(
                ActionError(
                    action_index=-1,
                    code=INDEX_BUDGET_EXCEEDED,
                    detail=f"{c}: {len(names)} indexes > budget {budget}",
                )
            )
    if any(e.hard for e in errors):
        return report(effective_queries=original)

    created: dict[str, list[str]] = {}
    dropped: dict[str, list[str]] = {}
    applied: list[Action] = []
    for i, a in enumerate(parsed.actions):
        try:
            if a.type == "drop_index":
                env.db[a.collection].drop_index(a.index_name)
                dropped.setdefault(a.collection, []).append(a.index_name)
            elif a.type == "create_index":
                name = create_index(env.db, a.collection, a.index)
                created.setdefault(a.collection, []).append(name)
            elif a.type == "rewrite_query":
                env.db.command(
                    {"explain": explain_command(a.new_query), "verbosity": "queryPlanner"}
                )
            applied.append(a)
        except PyMongoError as e:
            errors.append(
                ActionError(action_index=i, code=APPLY_ERROR, detail=f"{type(e).__name__}: {e}")
            )
            break

    failed = any(e.code == APPLY_ERROR for e in errors)
    eff = original if failed else effective_queries(workload, parsed, original)
    env.queries = {q.id: q for q in eff}
    sizes = {c: _index_sizes_mb(env.db, c) for c in created}
    return report(
        created=created,
        dropped=dropped,
        rewritten=[]
        if failed
        else [a.query_id for a in parsed.actions if a.type == "rewrite_query"],
        effective_queries=eff,
        index_sizes_mb=sizes,
        applied=applied,
    )
