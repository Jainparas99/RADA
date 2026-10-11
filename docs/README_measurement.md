# How measurement works

*(Track B section, for Track C to merge into the main README.)*

Ratchet never asks a model whether a tuning plan is good. Every number comes from MongoDB's own `explain("executionStats")` and from deterministic code in `src/ratchet/scoring.py`.

## What we read from `explain()`

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

## The cost formula

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

## Why examined counts, not time

`totalKeysExamined` and `totalDocsExamined` are deterministic: the same plan on the same data always gives the same numbers. Execution time on a shared Atlas cluster depends on whoever else is running queries, cache state and network. Time is recorded and shown, never scored. For the same reason the gate can compare generations run hours apart.

## Inner violations: when a plan scores 0

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

## Workloads are validated before they count

`uv run ratchet validate` (or `workloads/validate.py`) runs every query on the source data four times: twice with the default plan, then with `hint: {$natural: 1}` and `{$natural: -1}`. All four results must be identical, and between 1 and 500 documents. That rejects queries whose output depends on scan order, such as a missing `_id` tie-break or `$avg` over doubles. Those would otherwise make an honest index change look like `RESULT_MISMATCH`. Train and holdout workloads use disjoint collections.

## Example (Atlas sandbox, `mflix_w01`)

| | q1 genres + rating sort | q2 directors | workload cost | score |
|---|---|---|---|---|
| baseline | `IXSCAN(year_1) → FETCH → SORT`, 35,964 | `COLLSCAN → SORT`, 42,698 | 3,612,860 | |
| honest plan (2 ESR indexes, drop redundant `year_1`) | `IXSCAN → FETCH → LIMIT`, 22 | `IXSCAN → FETCH → SORT`, 96 | 963,720 | **0.731** |
| seeded cheat: q1 filter `year > 2015` | 3 | 42,698 | 1,814,810 | **0** (`RESULT_MISMATCH`, `FORBIDDEN_ACTION`) |
| seeded cheat: index six fields | — | — | unchanged, nothing built | **0** (`INDEX_BUDGET_EXCEEDED`) |

## Limits

- Examined counts are a proxy for cost. They don't model memory, network or index size beyond the penalty term.
- Partial indexes can't change results: the planner only uses one when the query implies its filter. They are allowed; `hint` is not.
- The rule set blocks a defined list of cheats (above). It isn't a proof that no plan can game the score.
- Atlas Performance Advisor also suggests indexes. Ratchet's scorer is the referee for an agent that also rewrites pipelines and learns across workloads.
