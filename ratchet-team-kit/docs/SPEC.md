# Ratchet — Specification

> A self-improving harness for MongoDB query tuning. A tuning agent fixes slow workloads, the harness learns from every workload it tunes, and it is promoted only when it gets better on workloads it has never seen — without changing a single query result.

Hackathon: Harness Engineering & Model Wrangling (#SFTechWeek). Problem Statement 1 (Recursive Harnessing), with the memory policy and multi-model routing touching Statement 2 and the "Model Wrangling" theme.

---

## 1. Pitch (2 minutes)

- **Problem.** Apps and AI agents fire queries at MongoDB all day. Slow queries are usually a missing or badly ordered index, or a pipeline that filters too late. Today a human reads `explain()` output and fixes them one by one. Workloads now change faster than humans can tune them.
- **How we spot it.** `explain()` reports how much work a query did. A query that returns 10 documents but reads 23,000 is the problem. Deterministic detectors flag it: collection scans, in-memory sorts, late `$match`, unindexed `$lookup`.
- **What we built.** Inside: an agent diagnoses, plans a fix, a critic (a different model) reviews it, we apply it to a scratch copy and re-measure. Outside: the harness improves itself — rubric rules, context policy, lesson memory, model routing — and a new version is promoted only if it scores better on **held-out workloads**.
- **Can't cheat.** Faster isn't enough. If a result changes, data changes, or the index budget is blown, it's rejected. The same applies to the harness: it can't edit the scorer, budget or workloads.
- **Model wrangling.** A harness evolved with one model is tested on other models to show whether the learning transfers.
- **Judge question to prepare for:** "Doesn't Atlas Performance Advisor do this?" → It suggests indexes from fixed heuristics. Ratchet also rewrites pipelines, learns across workloads, measurably improves on unseen ones, and its improvement loop is gated against cheating. The tuning is the demo; the self-improving harness is the product.

## 2. Event constraints (must hold)

- **New work only.** All code written at the event. Planning docs written beforehand; README says so and has a "Built at this hackathon" section.
- **MongoDB Atlas Sandbox** cluster from the event is the database for everything (finalist requirement).
- **Public repo** at submission. 1-minute demo video in the submission form.
- **Banned:** Streamlit, dashboard-as-main-feature, basic RAG. Demo = terminal CLI (Typer + Rich) + generated PNG chart.
- Judging: Technical Demo 35%, Implementation Difficulty 30%, Impact 20%, Creativity 15%.

## 3. Honest claims

- Ratchet blocks a **defined set** of cheating patterns (listed in §9). It does not "solve reward hacking".
- Cheats shown in the demo come from a **seeded attack mode**, labelled as such.
- Results are raw numbers on our workloads ("holdout improvement 41% → 88%"), not benchmark claims.
- Gen 0 is a deliberately minimal baseline. Say so.
- Atlas Performance Advisor exists. Say so if asked; explain the difference (§1).

## 4. Architecture

```
            ┌──────────────── OUTER LOOP (harness evolves) ────────────────┐
            │                                                               │
  harness head ──► run tuning agent on TRAIN workloads ──► traces ──► Proposer (LLM)
  (Atlas doc)                                                         │ JSON patch
       ▲                                                              ▼
       │ promote         candidate harness ──► run on TRAIN + HOLDOUT + CANARY
       └──────────────── Gate (code): static checks + dynamic checks ◄┘
                              │ reject → reason codes back to proposer

            ┌──────────────── INNER LOOP (one workload) ───────────────────┐
  workload ─► scratch copy ─► measure (explain) ─► detectors ─► Planner (model X)
                                                                    │ TuningPlan
                              re-measure ◄─ apply on scratch ◄─ Critic (model Y)
                                  │
                              Scorer: improvement − index penalty; violations
```

**Two levels of cheating defence**
- **Inner (Scorer, Track B):** a tuning plan that changes results, mutates data, blows the index budget or uses a forbidden action scores 0 and records a violation.
- **Outer (Gate, Track A):** a harness patch that touches protected fields, leaks holdout data, overfits train, or fails canaries is rejected.

## 5. Environment and workloads (Track B)

### 5.1 Data
- Atlas sample datasets loaded in the sandbox (`sample_mflix`, `sample_airbnb`, `sample_restaurants`, `sample_supplies`, `sample_analytics`, `sample_training`, …). Verify which exist.
- Optional synthetic collection for dramatic numbers: `scripts/gen_synthetic.py` → `ratchet_data.orders` (~200k small docs, seeded). **Check the sandbox storage limit first.**
- **Scratch copies.** Infra (never the agent) copies each workload collection once into `ratchet_scratch_<ns>.<coll>`. Before every tuning run, indexes are **reset to the workload's `baseline_indexes`**. The agent can only create/drop indexes and rewrite pipelines, so data never needs re-copying; a data checksum proves it.
- Each developer sets `RATCHET_NS` (`a`, `b`, `c`) so the three of you never collide. The final demo run uses `RATCHET_NS=main`.

### 5.2 Workload document (`workloads/*.json`)

```json
{
  "id": "mflix_w02",
  "split": "train",
  "db": "sample_mflix",
  "collections": ["movies", "comments"],
  "baseline_indexes": {"movies": [{"keys": [["year", 1]], "name": "year_1"}, {"keys": [["year", 1], ["title", 1]], "name": "year_1_title_1"}]},
  "write_rate": {"movies": 2.0, "comments": 40.0},
  "index_budget": {"movies": 3, "comments": 2},
  "queries": [
    {"id": "q1", "kind": "find", "collection": "movies",
     "filter": {"genres": "Drama", "year": {"$gt": 2000}},
     "sort": [["imdb.rating", -1], ["_id", 1]], "limit": 10, "projection": {"title": 1, "imdb.rating": 1},
     "freq": 50},
    {"id": "q2", "kind": "aggregate", "collection": "comments",
     "pipeline": [{"$lookup": {"from": "movies", "localField": "movie_id", "foreignField": "_id", "as": "m"}},
                  {"$unwind": "$m"}, {"$match": {"m.year": 2012}}, {"$count": "n"}],
     "freq": 5}
  ],
  "notes": "late $match after $lookup; redundant year_1 index"
}
```

Rules:
- **Deterministic results.** Every sorted or limited query has a unique tie-break (`_id`). `workloads/validate.py` runs every query twice and rejects non-deterministic ones, empty results, or > 500 docs.
- Mix of problems per workload: missing index, wrong compound order (ESR), in-memory sort, not covered, late `$match`, unindexed `$lookup`, unanchored regex, redundant/unused baseline index.
- Splits use **different collections**: train on e.g. mflix + airbnb + restaurants; holdout on e.g. supplies + analytics + training (+ synthetic orders). Holdout is never seen by the proposer.

| Split | Count | Queries each | Purpose |
|---|---|---|---|
| `train` | 8 | 4–6 | Proposer learns from these traces; lessons written only from these |
| `holdout` | 4 | 4–6 | Promotion decision (gate only) |
| `canary` | 3 | 2–4 | Traps: queries whose only "speed-up" changes semantics; a collection where the tempting fix is indexing everything; a query with a regex that can't be indexed. Pass = zero violations |

### 5.3 Measurement
- `find`: `db.command("explain", {"find": ...}, verbosity="executionStats")`.
- `aggregate`: `db.command({"explain": {"aggregate": coll, "pipeline": p, "cursor": {}}, "verbosity": "executionStats"})`. Output shape differs (stages, `$cursor`, `$lookup` stats) — normalise into `ExplainSummary`. **Verify the exact output on the sandbox's MongoDB version first and write fixtures from real output.**
- Result hash: run the query, canonical JSON (sorted keys, `bson.json_util`), order-sensitive. SHA-256.
- Data checksum: `count` + hash of `_id`s per scratch collection (or `dbHash` if permitted on the tier).

### 5.4 Detectors (deterministic, `env/detectors.py`)

| ID | Fires when | Hint for planner |
|---|---|---|
| `COLLSCAN_LARGE` | `COLLSCAN` and collection > 5k docs | add index |
| `HIGH_EXAMINE_RATIO` | (keys+docs examined)/max(1,nReturned) > 50 with an index used | unselective or wrongly ordered index |
| `IN_MEMORY_SORT` | `SORT` stage present | put sort field in index (equality → sort → range) |
| `NOT_COVERED` | `FETCH` present and projection ⊆ small field set | covering index |
| `LATE_MATCH` | `$match` after `$unwind`/`$lookup`/`$group` on fields available earlier | move `$match` earlier |
| `UNINDEXED_LOOKUP` | `$lookup` with collection scans on `from` | index `foreignField` |
| `UNANCHORED_REGEX` | `$regex` not starting with `^` or with `i` flag | anchor, or accept |
| `REDUNDANT_INDEX` | index A is a prefix of index B | drop A |
| `UNUSED_INDEX` | baseline index used by no query in workload | drop (if write-heavy) |

Detectors only **diagnose**. Thresholds live in `config/detectors.yaml`.

### 5.5 Actions the agent may take (`env/apply.py`)

| Action | Allowed | Notes |
|---|---|---|
| `create_index` | ✓ | keys + optional `partialFilterExpression`, `name`. **Forbidden options:** `unique`, `sparse` changes that alter semantics, `hidden: false` toggles on others, TTL (`expireAfterSeconds`) |
| `drop_index` | ✓ | never `_id_` |
| `rewrite_query` | ✓ | aggregate pipelines and find projection/sort-key order only; must return identical results |
| anything else (`$out`, `$merge`, `collMod`, writes, `hint`, dropping collections, creating collections) | ✗ | `FORBIDDEN_ACTION` |

### 5.6 Scorer (`scoring.py`, Track B, pinned with a source hash)

```
cost(q)        = totalKeysExamined + totalDocsExamined (+ totalDocsExamined again if an in-memory SORT remains)
                 (+ $lookup stage examined counts for aggregates)
workload_cost  = Σ freq(q) × cost(q)
improvement    = clamp(1 − cost_after / cost_before, −1, 1)
index_penalty  = Σ_new_indexes α × write_rate[coll]  +  β × size_MB(new)  −  γ × Σ_dropped_redundant
score          = improvement − index_penalty        (α, β, γ in config/scoring.yaml)
```

**Inner violations** (any one → workload score = 0, recorded):

| Code | Fires when |
|---|---|
| `RESULT_MISMATCH` | any query's result hash differs from baseline |
| `DATA_MUTATED` | data checksum changed |
| `INDEX_BUDGET_EXCEEDED` | more indexes than `index_budget[coll]` |
| `FORBIDDEN_ACTION` | action outside §5.5 |
| `APPLY_ERROR` | index build or rewrite failed |

Execution time is recorded and shown, never scored (noisy on a shared cluster).

## 6. Harness document (Track A)

Collection `harness_versions`, one doc per version; `harness_head` points to the active one.

```json
{
  "_id": "h_0007", "version": 7, "parent": "h_0006", "status": "accepted",
  "created_by": "proposer", "rationale": "...", "patch": [ ... ],
  "spec": {
    "planner_rules": ["..."],
    "critic_rubric": ["..."],
    "context_policy": {
      "include_detector_findings": true, "include_full_explain": false,
      "include_existing_indexes": true, "include_collection_stats": true,
      "sample_docs": 1, "max_chars": 6000
    },
    "retry_policy": {"max_rounds": 2, "retry_if_improvement_below": 0.5, "show_explain_diff": true},
    "critic_policy": {"enabled": true, "max_revisions": 1},
    "model_routing": {"planner": "anthropic:claude-haiku-5-5", "critic": "openai:<model>"},
    "memory_policy": { "...": "§8" },
    "protected": {
      "model_allowlist": ["anthropic:claude-haiku-5-5", "anthropic:claude-sonnet-5-5", "openai:<model>", "google:<model>"],
      "allowed_actions": ["create_index", "drop_index", "rewrite_query"],
      "token_budget_per_workload": 30000,
      "scorer_ref": {"version": "1", "source_hash": "sha256:..."},
      "gate_ref": {"version": "1", "source_hash": "sha256:..."}
    }
  },
  "scores": {"train": 0.62, "holdout": 0.58, "canary_pass": 1.0, "tokens_per_workload": 9100},
  "models_used": {"planner": "...", "critic": "...", "proposer": "..."}
}
```

- **Editable** (within `config/bounds.yaml`): `planner_rules`, `critic_rubric` (≤ 15 items, ≤ 300 chars each), `context_policy/*`, `retry_policy/*`, `critic_policy/*`, `memory_policy/*`, `model_routing/*` (values must be in `model_allowlist`).
- **Protected** (any change → reject): everything under `protected`. Index budgets and workloads live outside the harness entirely.
- Gen 0: 2 short rules, no rubric, critic disabled, `sample_docs: 0`, no detector findings in context, memory inject `never`, `max_rounds: 1`.

## 7. Tuning agent (Track C) — the napkin, inside

```
Brief (workload) → Diagnose (measure + detectors, per context_policy)
                 → Plan (planner model) → Critic (critic model, may request ≤ N revisions)
                 → Apply (scratch) → Measure → Retry if improvement < threshold
                 → TuneResult
```

- Planner output: strict JSON `TuningPlan` (§10). Invalid → one repair retry → else empty plan.
- Critic output: `{"approve": bool, "issues": [...], "revised_plan": TuningPlan | null}` checked against `critic_rubric`.
- Tools available to the planner: `explain(query_id)`, `list_indexes(coll)`, `collection_stats(coll)`, `sample_docs(coll, n)`, `findings(query_id)`, `recall_lessons(shape)` (if memory allows), `self_diagnose(category, note)` (Raindrop pattern: categories `explain_confusing`, `conflicting_queries`, `budget_too_tight`, `cannot_fix_without_semantics_change`, `tool_error`).
- Temperature 0. All calls through the cached LLM client. Every message, tool call, plan, critic verdict and measurement goes into the trace.

## 8. Memory policy (Track A, evolvable)

Lessons are short facts learned on train workloads, e.g. *"Equality fields first, then the sort field, then range fields."* Stored in `memories` with an embedding of the **query shape** (values stripped):

```
movies | filter {genres: EQ, year: RANGE} | sort {imdb.rating: DESC} | proj {title, imdb.rating}
```

```json
"memory_policy": {
  "write":       {"enabled": true, "from": ["train_improved", "train_violation"], "lesson_max_chars": 300},
  "retrieve":    {"k": 3, "min_similarity": 0.75},
  "inject":      {"mode": "on_retry", "max_tokens": 500},
  "consolidate": {"merge_similarity": 0.92, "max_items": 40},
  "invalidate":  {"after_harmful_uses": 2}
}
```

| Knob | Bounds |
|---|---|
| `retrieve.k` | 0–8 |
| `retrieve.min_similarity` | 0.5–0.95 |
| `inject.mode` | `never` / `always` / `on_retry` |
| `inject.max_tokens` | 0–1500 |
| `consolidate.merge_similarity` | 0.8–0.98 |
| `invalidate.after_harmful_uses` | 1–5 |

Invariants: lessons written **only** from train runs; writes frozen during holdout/canary; lesson text may not contain holdout collection names, field names unique to holdout, or query IDs (`HOLDOUT_LEAKAGE`). Embeddings via Atlas automated Voyage embeddings if available in the sandbox, else the `voyageai` SDK (verify current APIs first).

## 9. Gate (Track A, deterministic code)

**Static (on the patch):** `PATCH_PATH_NOT_EDITABLE`, `PROTECTED_FIELD_CHANGED`, `MODEL_NOT_ALLOWED`, `SCHEMA_INVALID`, `OUT_OF_BOUNDS`, `PATCH_TOO_LARGE` (> 4 ops or > 1200 added chars), `HOLDOUT_LEAKAGE`, `SCORER_TAMPERED` / `GATE_TAMPERED` (source hash mismatch).

**Dynamic (after running the candidate):**

| Code | Fires when |
|---|---|
| `CANARY_FAILED` | any canary workload has an inner violation |
| `INNER_VIOLATIONS_UP` | total inner violations on train+holdout > parent's |
| `NO_HOLDOUT_GAIN` | holdout mean score < parent + 0.03, unless equal (±0.01) and tokens/workload drop ≥ 15% |
| `OVERFIT_DIVERGENCE` | train mean rises ≥ 0.10 while holdout ≤ parent |
| `TOKEN_BUDGET_EXCEEDED` | tokens/workload > 1.25 × parent |

`ACCEPT` only if no code fires. Every decision → `gate_decisions` with parent/candidate metrics. Thresholds in `config/gate.yaml`.

**Rollback:** `ratchet rollback <v>`. At each generation start, the head is re-run on canaries; failure → automatic rollback.

## 10. Contracts (`src/ratchet/contracts.py`, Track A writes FIRST, everyone codes against it)

```python
from typing import Literal, Any
from pydantic import BaseModel, Field

Split = Literal["train", "holdout", "canary"]

class IndexDef(BaseModel):
    keys: list[tuple[str, int | str]]           # [("genres",1),("imdb.rating",-1)]
    name: str | None = None
    partialFilterExpression: dict | None = None

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
    stages: list[str]                 # e.g. ["IXSCAN","FETCH"] or ["COLLSCAN","SORT"]
    n_returned: int
    keys_examined: int
    docs_examined: int
    exec_ms: int
    has_in_memory_sort: bool
    indexes_used: list[str]
    lookup_examined: int = 0
    raw: dict | None = None           # trimmed raw explain, for traces

class Finding(BaseModel):
    query_id: str
    detector: str
    severity: Literal["high", "medium", "low"]
    evidence: dict
    hint: str

class Action(BaseModel):
    type: Literal["create_index", "drop_index", "rewrite_query"]
    collection: str | None = None
    index: IndexDef | None = None     # create_index
    index_name: str | None = None     # drop_index
    query_id: str | None = None       # rewrite_query
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
    violations: list[str]             # inner violation codes

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
```

**Function interfaces (owners in brackets):**

```python
# [B] env/
prepare_scratch(workload: Workload, ns: str) -> ScratchEnv          # copy once, reset indexes to baseline
measure(env: ScratchEnv, workload: Workload, queries: list[QuerySpec] | None = None) -> Measurement
run_detectors(env: ScratchEnv, workload: Workload, m: Measurement) -> list[Finding]
apply_plan(env: ScratchEnv, workload: Workload, plan: TuningPlan) -> ApplyReport   # validates §5.5
score(workload: Workload, before: Measurement, after: Measurement, plan: TuningPlan, apply: ApplyReport) -> ScoreReport
query_shape(q: QuerySpec) -> str

# [C] agent/, llm/
tune(workload: Workload, harness: HarnessSpec, env: ScratchEnv, mode: Literal["normal","frozen_memory"]) -> TuneResult
complete(role: str, model: str, messages: list[dict], tools: list[dict] | None = None, json_schema: dict | None = None) -> LLMResponse   # cached
embed(texts: list[str]) -> list[list[float]]

# [A] harness/, gate/, proposer/, memory/, evolve.py
get_head() -> HarnessDoc ; save_candidate(...) ; promote(v) ; rollback(v)
check_static(parent: HarnessSpec, patch: list[PatchOp]) -> list[str]
check_dynamic(parent_scores, cand_scores) -> list[str]
propose(context: ProposerContext) -> ProposerOutput
recall(shape: str, policy: MemoryPolicy) -> list[Lesson] ; write_lessons(results: list[TuneResult], policy)
evolve(gens: int) -> None
```

Until a dependency lands, use the stubs in `tests/fakes.py` (`FakeEnv`, `FakeLLM`) — Track A creates them in Phase 0.

## 11. Multi-model features (Track C)

1. **Provider-agnostic client** (`llm/`): `anthropic`, `openai`, `google` providers behind `complete()`; model IDs as `provider:model`. Providers without keys are skipped gracefully. Every call cached in `llm_cache`, tokens and cost logged. (A library like LiteLLM is fine if it saves time — verify.)
2. **Cross-model critic:** the critic can be a different provider from the planner (`model_routing`). The harness itself can learn this via the proposer.
3. **Model routing as an evolvable knob:** proposer can switch planner/critic among the allowlist; the gate's token rule keeps it honest about cost.
4. **Transfer test:** `ratchet transfer --harness <best> --models a,b,c` runs gen 0 and the best harness on **holdout** with each planner model → table "harness evolved with model X lifts model Y from 38% → 79%". This is the "Model Wrangling" slide.
5. **Proposer model** configurable (`RATCHET_PROPOSER_MODEL`), default a strong Claude model.

## 12. Collections and indexes

| Collection | Notes | Indexes |
|---|---|---|
| `workloads` | loaded from `workloads/*.json` | `split` |
| `harness_versions`, `harness_head` | §6 | `version` unique |
| `tune_results` | one per (harness_version, workload) | `(harness_version, split)` |
| `failures` | train TuneResults with low score or violations, summary + embedding | vector index |
| `memories` | §8 | vector index + `status` |
| `gate_decisions` | §9 | `candidate` |
| `generations` | gen, head_before, candidate, verdict, duration | `gen` |
| `llm_cache` | key = hash(model, messages, tools, schema) | `key` unique |
| `transfer_runs` | §11.4 | `harness_version` |

## 13. CLI (`ratchet`)

| Command | Owner | Does |
|---|---|---|
| `init` | A | collections, indexes (incl. vector), gen-0 harness |
| `seed` | B | load + validate workloads, prepare scratch copies |
| `measure <workload>` | B | before-state explain table + detector findings |
| `tune <workload> [--version V]` | C | run the agent on one workload, show plan, critic verdict, before/after |
| `eval [--version V] [--split S]` | A | run all workloads of a split, results table |
| `evolve --gens N` | A | outer loop |
| `history`, `diff A B`, `rollback V` | A | versions, verdicts, reason codes |
| `memory` | A | lessons with helpful/harmful counts |
| `attack` | C | seeded cheats at both levels, each shown with its reason code |
| `transfer` | C | §11.4 |
| `chart` | C | PNG: train/holdout by generation, rejected candidates ✕ with code |
| `watch` | C | change streams on `harness_versions`, `gate_decisions` (verify support on the tier; fall back to polling) |
| `replay` | C | re-run a recorded evolution from `llm_cache` (backup demo) |

CLI structure avoids merge conflicts: `cli.py` (A) mounts `cli_core.py` (A), `cli_env.py` (B), `cli_demo.py` (C).

## 14. `ratchet attack` (seeded, labelled)

**Inner (plans, caught by Scorer):**
- rewrite that changes a filter (`year > 2000` → `year > 2015`) → `RESULT_MISMATCH`
- index every field → `INDEX_BUDGET_EXCEEDED`
- `$out` precomputed answers → `FORBIDDEN_ACTION`
- `unique: true` index → `FORBIDDEN_ACTION`

**Outer (harness patches, caught by Gate):**
- raise `token_budget_per_workload` or add a model to the allowlist → `PROTECTED_FIELD_CHANGED`
- rule "for collection `<holdout coll>` always index `<field>`" → `HOLDOUT_LEAKAGE`
- patch that overfits train → `OVERFIT_DIVERGENCE`
- route planner to a model not in the allowlist → `MODEL_NOT_ALLOWED`

## 15. Demo (≈ 3 minutes)

1. **Problem (0:00–0:25):** `ratchet measure mflix_w02` → "returns 10, reads 23,000". Detectors flag COLLSCAN + in-memory sort.
2. **Inside (0:25–1:00):** `ratchet tune mflix_w02` → plan, critic (different model) catches an ESR mistake, before/after: COLLSCAN → IXSCAN, cost −9x.
3. **Outside (1:00–1:50):** `ratchet watch` in a split pane + `ratchet evolve --gens 1` live; then `ratchet history` and `ratchet chart` from the full run: holdout improvement climbs, indexes per workload drop. `ratchet diff` on one accepted patch (a rubric rule or memory-policy change) with the proposer's diagnosis.
4. **Can't cheat (1:50–2:30):** `ratchet attack` → each cheat rejected with its code.
5. **Model wrangling (2:30–2:50):** `ratchet transfer` table.
6. **Close (2:50–3:00):** "Everything ran on Atlas: versions, traces, vector memory, change streams. All built today."

Backup: the 60-second recorded video and `ratchet replay`.

## 16. Stretch (after freeze-ready only)

- OTLP export of traces to Raindrop Workshop (`/v1/traces`) so traces are viewable there too (verify API). Mention in pitch for the Raindrop RSI prize either way.
- Workload drift: replay a changed workload and show the harness re-tune with recalled lessons.
- Proposer improves its own context builder (second-level loop).

## 17. References (verify before citing in README)

- Lilian Weng, "Harness Engineering for Self-Improvement" (July 2026)
- Lee et al., "Meta-Harness" (arXiv 2603.28052) — raw traces to the proposer, transfer across models
- "Self-Harness" (arXiv 2606.09498) — regression-tested harness edits
- Zhang et al., "Darwin Gödel Machine" (arXiv 2505.22954) — objective hacking
- "EvilGenie" (arXiv 2511.21654) — held-out tests as a hack signal
- MongoDB docs: `explain` results, ESR guideline, Atlas Vector Search
- Raindrop self-diagnostics and Workshop
