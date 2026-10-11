# Ratchet

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

<!-- Track B: merge docs/README_measurement.md here (explain normalisation, result hashes, data
checksums, scratch copies, cost formula). -->

```
cost(q)       = keysExamined + docsExamined (+ docsExamined again if an in-memory SORT remains)
                (+ $lookup examined counts for aggregates)
workload_cost = Σ freq(q) × cost(q)
improvement   = clamp(1 − cost_after / cost_before, −1, 1)
score         = improvement − index_penalty      (penalty grows with write rate and index size)
```

Execution time is shown, never scored: it is too noisy on a shared cluster.

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
into apply + score, to show the scorer catches them without the agent's help. The one exception is
`unique: true`, which the shared contract can't even express; the agent's raw-JSON scan catches
it.

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
