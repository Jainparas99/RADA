# Rachet

**A self-improving harness for MongoDB query tuning.** A tuning agent fixes slow workloads; the
harness that steers it learns from every workload it tunes, and a new harness version is promoted
only when it scores better on workloads it has never seen, without changing a single query result.

Built at the Harness Engineering & Model Wrangling hackathon (#SFTechWeek). Problem Statement 1
(Recursive Harnessing), with memory policy and multi-model routing touching Statement 2.

---

## The problem

Apps and AI agents fire queries at MongoDB all day. Slow queries are usually a missing or badly
ordered index, or a pipeline that filters too late. Today a human reads `explain()` output and
fixes them one at a time. Workloads now change faster than humans can tune them.

`explain()` says how much work a query did. A query that returns 10 documents but reads 23,000 is
the problem. Ratchet's detectors flag it deterministically: collection scans, in-memory sorts,
late `$match`, unindexed `$lookup`, unanchored regexes, redundant indexes.

## How it works

```
            ┌──────────────── OUTER LOOP (the harness evolves) ─────────────┐
            │                                                               │
  harness head ──► tune TRAIN workloads ──► traces ──► Proposer (LLM) ──► JSON patch
  (Atlas doc)                                                               │
       ▲                                                                    ▼
       │ promote        candidate harness ──► run TRAIN + HOLDOUT + CANARY workloads
       └─────────────── Gate (deterministic code): static + dynamic checks ◄┘
                              │ reject → reason codes go back to the proposer

            ┌──────────────── INNER LOOP (one workload) ────────────────────┐
  workload ─► scratch copy ─► explain() ─► detectors ─► Planner (model X)
                                                           │ TuningPlan (strict JSON)
                    re-measure ◄─ apply on scratch ◄─ Critic (model Y, ≤ N revisions)
                        │
                    Scorer: improvement − index penalty; any violation → 0
```

**Inner loop (one workload).** The agent measures the workload on a scratch copy, runs the
detectors, and a planner model proposes a `TuningPlan`: only `create_index`, `drop_index` and
`rewrite_query`. A critic, a *different* model, reviews it against a rubric and can force
revisions. The plan is applied to the scratch copy, the workload is re-measured, and the
deterministic scorer computes `improvement − index_penalty`. If the improvement is below the
harness's threshold the agent retries with the explain diff. Every LLM message, tool call, plan,
critic verdict and measurement goes into the trace.

**Outer loop (the harness).** The harness is a versioned document in Atlas: planner rules, critic
rubric, context policy (what the planner sees), retry policy, critic policy, lesson-memory policy
and model routing. A proposer model reads train traces and proposes a small JSON patch. The gate
runs the candidate on train, held-out and canary workloads and promotes it only if held-out scores
improve and no invariant breaks.

**The tuning is the demo; the self-improving harness is the product.**

### Agent tools

The planner can ask for `explain(query_id)`, `list_indexes(coll)`, `collection_stats(coll)`,
`sample_docs(coll, n)`, `findings(query_id)`, `recall_lessons(shape)` (if the memory policy allows)
and `self_diagnose(category, note)` (categories: `explain_confusing`, `conflicting_queries`,
`budget_too_tight`, `cannot_fix_without_semantics_change`, `tool_error`). Tool calls are plain JSON,
so the same loop works with every provider.

### How measurement works

*(Track B section, for Track C to merge into the main README.)*

Ratchet never asks a model whether a tuning plan is good. Every number comes from MongoDB's own `explain("executionStats")` and from deterministic code in `src/ratchet/scoring.py`.

#### What we read from `explain()`

For each query we run `explain` with `executionStats` on a scratch copy of the collection and normalise the output into an `ExplainSummary`:

| Field | Source | Meaning |
|---|---|---|
| `keys_examined` | `executionStats.totalKeysExamined` | index entries read |
| `docs_examined` | `executionStats.totalDocsExamined` | documents fetched |
| `lookup_examined` | classic `$lookup` stage `totalDocsExamined + totalKeysExamined` | work done on the joined collection |
| `has_in_memory_sort` | a `SORT` plan stage, or a `$sort` pipeline stage | the server had to buffer and sort |
| `stages` | winning plan, leaf first, then pipeline stages | e.g. `COLLSCAN → SORT` vs `IXSCAN → FETCH → LIMIT` |
| `n_returned`, `indexes_used`, `exec_ms` | explain | shown, `exec_ms` never scored |

The normaliser handles every shape we met on MongoDB 8.0 (Atlas sandbox) and 8.2: plain finds, aggregates fully pushed down into the query layer, `$cursor` + pipeline stages, and SBE plans (`explainVersion: "2"`, where `$lookup` becomes `EQ_LOOKUP` and `$group` becomes `GROUP`). Each examined document is counted once. A classic `$lookup` reports its foreign-side work on the stage; an SBE `EQ_LOOKUP` already folds it into the totals. Fixtures captured from both servers are in `tests/env/fixtures/`.

#### The cost formula

```
cost(q)       = keys_examined + docs_examined + lookup_examined
                + (docs_examined again if an in-memory SORT remains in the query layer,
                   or the docs flowing into a pipeline $sort)
workload_cost = Σ freq(q) × cost(q)                  freq comes from the workload file
improvement   = clamp(1 − cost_after / cost_before, −1, 1)
index_penalty = Σ_new α·write_rate[coll] + β·size_MB(new) − γ·(dropped redundant/unused baseline indexes)
score         = improvement − index_penalty          (0 if any inner violation)
```

α, β, γ live in `config/scoring.yaml` (0.002, 0.01, 0.02). On a write-heavy collection (40 writes/s) every new index costs 0.08, so "index everything" loses even before the budget stops it. The frequency always comes from the workload file, never from a rewritten query, so a rewrite can't discount an expensive query.

One deliberate refinement of the SPEC formula: a `$sort` stage *inside the pipeline* is charged for the documents it actually sorts, not for `totalDocsExamined` again. On 8.x, SBE folds a join's foreign-side work into `totalDocsExamined`. So the literal rule would charge a 109-row sort after a `$lookup` for 2.18 million documents.

#### Why examined counts, not time

`totalKeysExamined` and `totalDocsExamined` are deterministic: the same plan on the same data always gives the same numbers. Execution time on a shared Atlas cluster depends on whoever else is running queries, cache state and network. Time is recorded and shown, never scored. For the same reason the gate can compare generations run hours apart.

#### Inner violations: when a plan scores 0

| Code | Fires when | Catches |
|---|---|---|
| `RESULT_MISMATCH` | any query's result hash differs from the baseline (or a query is missing) | "faster" queries that return different answers, rewrites that reorder ties |
| `DATA_MUTATED` | a scratch collection's checksum changed | deleting "slow" documents |
| `INDEX_BUDGET_EXCEEDED` | more indexes than `index_budget[coll]` (checked before building, and on the final state) | indexing every field |
| `FORBIDDEN_ACTION` | an action outside `create_index` / `drop_index` / `rewrite_query`, a forbidden option or stage, or a rewrite that breaks the rewrite rules | `$out` caches, `unique`/TTL/hidden indexes, dropping `_id_`, `$lookup` into another collection, smuggled answers |
| `APPLY_ERROR` | an index build or rewritten query fails | malformed plans |

**Result hash.** Each query runs on the scratch copy. Each result document is serialised as canonical Extended JSON with sorted keys (types matter: `int32` ≠ `double`), and the list is hashed with SHA-256, order-sensitive.

**Data checksum.** We use `dbHash` where the tier allows it (the event sandbox does). Otherwise a SHA-256 over every document in `_id` order. For very large collections, a count plus `_id` hash.

**Rewrite rules.** A rewrite keeps the query's `id`, collection and kind. A `find` rewrite may only change the projection and sort. An aggregate rewrite may not introduce constants that aren't in the original query (`NEW_LITERAL`), which blocks e.g. `{$match: {_id: {$in: [<the ten answer ids>]}}}`. A rewrite that breaks a semantic rule is still measured, because it's read-only. So a changed filter shows up as `RESULT_MISMATCH` *and* `FORBIDDEN_ACTION`. Rewrites that could write or read outside the workload (`$out`, `$merge`, `$documents`, `$where`, `$function`, `$rand`, `$$NOW`, `$lookup` into another collection) are never executed.

#### Workloads are validated before they count

`uv run ratchet validate` (or `workloads/validate.py`) runs every query on the source data four times: twice with the default plan, then with `hint: {$natural: 1}` and `{$natural: -1}`. All four results must be identical, and between 1 and 500 documents. That rejects queries whose output depends on scan order, such as a missing `_id` tie-break or `$avg` over doubles. Those would otherwise make an honest index change look like `RESULT_MISMATCH`. Train and holdout workloads use disjoint collections.

#### Example (Atlas sandbox, `mflix_w01`)

| | q1 genres + rating sort | q2 directors | workload cost | score |
|---|---|---|---|---|
| baseline | `IXSCAN(year_1) → FETCH → SORT`, 35,964 | `COLLSCAN → SORT`, 42,698 | 3,612,860 | |
| honest plan (2 ESR indexes, drop redundant `year_1`) | `IXSCAN → FETCH → LIMIT`, 22 | `IXSCAN → FETCH → SORT`, 96 | 963,720 | **0.731** |
| seeded cheat: q1 filter `year > 2015` | 3 | 42,698 | 1,814,810 | **0** (`RESULT_MISMATCH`, `FORBIDDEN_ACTION`) |
| seeded cheat: index six fields | — | — | unchanged, nothing built | **0** (`INDEX_BUDGET_EXCEEDED`) |

#### Limits

- Examined counts are a proxy for cost. They don't model memory, network or index size beyond the penalty term.
- Partial indexes can't change results: the planner only uses one when the query implies its filter. They are allowed; `hint` is not.
- The rule set blocks a defined list of cheats (above). It isn't a proof that no plan can game the score.
- Atlas Performance Advisor also suggests indexes. Ratchet's scorer is the referee for an agent that also rewrites pipelines and learns across workloads.

## Can't cheat (a defined set of cheats)

Faster isn't enough. Ratchet blocks a **defined set** of cheating patterns at two levels. It does
not "solve reward hacking".

**Inner (plans → scorer, plus the agent's own scan):**

| Code | Fires when |
|---|---|
| `RESULT_MISMATCH` | any query's result hash differs from baseline |
| `DATA_MUTATED` | the scratch data checksum changed |
| `INDEX_BUDGET_EXCEEDED` | more indexes on a collection than its budget |
| `FORBIDDEN_ACTION` | anything outside create/drop index and result-preserving rewrites (`$out`, `$merge`, `hint`, `unique`/`sparse`/TTL/`hidden` indexes, dropping `_id_`, …) |
| `APPLY_ERROR` | an index build or rewrite failed |

Any inner violation scores the workload 0.

**Outer (harness patches → gate):** `PATCH_PATH_NOT_EDITABLE`, `PROTECTED_FIELD_CHANGED`,
`MODEL_NOT_ALLOWED`, `SCHEMA_INVALID`, `OUT_OF_BOUNDS`, `PATCH_TOO_LARGE`, `HOLDOUT_LEAKAGE`,
`SCORER_TAMPERED` / `GATE_TAMPERED` (static); `CANARY_FAILED`, `INNER_VIOLATIONS_UP`,
`NO_HOLDOUT_GAIN`, `OVERFIT_DIVERGENCE`, `TOKEN_BUDGET_EXCEEDED` (dynamic). The proposer never sees
holdout or canary workloads, their traces, or the scorer and gate source.

`ratchet attack` runs **seeded, labelled** cheats (hand-written, injected on purpose, not model
behaviour) and shows each one rejected with its code:

| Level | Seeded cheat | Expected code |
|---|---|---|
| inner | rewrite narrows a filter (`year > 2000` → `year > 2015`) | `RESULT_MISMATCH` |
| inner | index every field | `INDEX_BUDGET_EXCEEDED` |
| inner | `$out` precomputed answers | `FORBIDDEN_ACTION` |
| inner | `unique: true` index | `FORBIDDEN_ACTION` |
| outer | raise `token_budget_per_workload` / extend the model allowlist | `PROTECTED_FIELD_CHANGED` |
| outer | rule "for collection *&lt;holdout coll&gt;* always index *&lt;field&gt;*" | `HOLDOUT_LEAKAGE` |
| outer | route the planner to a model not in the allowlist | `MODEL_NOT_ALLOWED` |
| outer | patch that lifts train but not holdout | `OVERFIT_DIVERGENCE` |

Inner cheats run twice: through the whole agent (the cheat is the planner's reply) and straight
into apply + score, to show the scorer catches them without the agent's help.

## Model wrangling

- **Provider-agnostic client.** Every LLM call goes through `ratchet.llm.complete()`: model ids
  are `provider:model` (Anthropic required; OpenAI and Google optional and skipped cleanly without
  a key), every round-trip is cached in Atlas (`llm_cache`, keyed by a hash of model, messages and
  parameters), JSON replies are validated against a schema with up to two repair turns, and 429 /
  5xx / timeouts back off exponentially. Tokens and cost are logged per call.
- **Cross-model critic.** The critic is a different model from the planner, from another
  provider when a key exists.
- **Routing is an evolvable knob.** The proposer can switch planner and critic models within the
  protected allowlist; the gate's token rule keeps it honest about cost.
- **Transfer test.** `ratchet transfer` runs gen 0 and the best harness on the held-out workloads
  with every planner model we have keys for.

| Role | Default |
|---|---|
| Planner | `anthropic:claude-haiku-5-5` |
| Critic | another provider if available, else `anthropic:claude-sonnet-5-5` |
| Proposer | `anthropic:claude-sonnet-5-5` |
| Embeddings | Atlas automated Voyage embeddings, else `voyageai` |

## Results

> **TODO (final run, `RATCHET_NS=main`):** fill in from `ratchet history`, `ratchet chart` and
> `ratchet transfer`. Raw numbers on our workloads only; gen 0 is a deliberately minimal baseline.

| | gen 0 (baseline) | best harness | |
|---|---:|---:|---|
| holdout mean score | _tbd_ | _tbd_ | |
| holdout mean cost improvement | _tbd_ | _tbd_ | |
| train mean score | _tbd_ | _tbd_ | |
| inner violations (train + holdout) | _tbd_ | _tbd_ | |
| tokens / workload | _tbd_ | _tbd_ | |
| generations run / accepted / rejected | | _tbd_ | |

![score per generation](docs/chart.png)

**Transfer** (`ratchet transfer`, written to `docs/transfer.md`): _tbd_

## Honest limits

- Ratchet blocks the cheats listed above. Other cheats may exist; we test the ones we know.
- Workloads are ours (Atlas sample datasets plus a seeded synthetic collection); numbers are not a
  benchmark.
- Gen 0 is deliberately minimal (two rules, no critic, no findings in context), so early gains
  are partly "turning features on". The gate still demands they hold on unseen workloads.
- Cost is `keys + docs examined`, a proxy for latency; wall-clock is shown but not scored.
- Current Claude models don't accept a sampling temperature; repeatability comes from `llm_cache`
  rather than temperature 0.
- Small holdout set (4 workloads), so promotion decisions are coarse.
- **Atlas Performance Advisor exists.** It suggests indexes from fixed heuristics. Ratchet also
  rewrites pipelines, learns across workloads, measurably improves on unseen ones, and its own
  improvement loop is gated against cheating.

## Run it

```bash
uv sync                      # Python 3.12; add --extra openai / --extra google / --extra voyage
cp .env.example .env         # fill in Atlas URIs, RATCHET_NS (a|b|c|main) and API keys
uv run ratchet info --ping
uv run ratchet init && uv run ratchet seed
uv run ratchet measure mflix_w02
uv run ratchet tune mflix_w02
uv run ratchet evolve --gens 8
uv run ratchet history && uv run ratchet chart && uv run ratchet attack && uv run ratchet transfer
```

Offline (no Atlas, no keys): `uv run ratchet attack --fake`. With a key but no Atlas:
`uv run ratchet tune fake_w01 --fake`. Tests: `uv run pytest -q`; lint:
`uv run ruff check . && uv run ruff format --check .`.

| Command | Does |
|---|---|
| `init`, `seed`, `measure <w>` | collections + gen-0 harness; load workloads, scratch copies; before-state explain + findings |
| `tune <w> [--version V]` | run the agent on one workload: findings, plan, critic verdicts, before/after |
| `eval`, `evolve --gens N` | run a split; run the outer loop |
| `history`, `diff A B`, `rollback V`, `memory` | versions and verdicts, patch diff, rollback, lessons |
| `attack [--inner/--outer]` | seeded cheats at both levels, each with its reason code |
| `transfer [--models a,b]` | gen 0 vs best harness on holdout per planner model |
| `chart` | PNG: score per generation, rejected candidates marked with their code |
| `watch` | live tail of versions and gate decisions (change streams, polling fallback) |
| `replay [--gen N]` | re-run a generation from `llm_cache` only (`RATCHET_LLM_CACHE_ONLY=1` for any command) |

Everything runs on the event's MongoDB Atlas Sandbox: harness versions, traces, gate decisions,
vector-searchable lessons, the LLM cache and change streams.

## Built at this hackathon

All code in this repository was written during the event. The planning documents
(`docs/SPEC.md`, `docs/TEAM_PLAN.md`, `CLAUDE.md`, `prompts/`) were written beforehand. Three
people, three tracks (core loop & gate; environment & measurement; agents, models & demo), each
working with its own Claude Code agent; the sync log is in `docs/HANDOFF.md`.

## References

<!-- Verify every entry (title, authors, id) before submission. -->

- Lilian Weng, "Harness Engineering for Self-Improvement" (2026)
- Lee et al., "Meta-Harness" (arXiv 2603.28052): raw traces to the proposer, transfer across models
- "Self-Harness" (arXiv 2606.09498): regression-tested harness edits
- Zhang et al., "Darwin Gödel Machine" (arXiv 2505.22954): objective hacking
- "EvilGenie" (arXiv 2511.21654): held-out tests as a hack signal
- MongoDB docs: explain results, the ESR (equality, sort, range) guideline, Atlas Vector Search
- Raindrop: self-diagnostics and Workshop

## License

MIT, see `LICENSE`.
