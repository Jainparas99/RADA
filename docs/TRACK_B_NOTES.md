# Track B notes for Tracks A and C

Branch `b/env-measurement`. Everything below is on that branch, tested (`uv run pytest -q`: 148 passed) and lint-clean.

## Provisional files (Track A: take yours on rebase)

`pyproject.toml`, `.gitignore`, `src/ratchet/contracts.py` (SPEC §10 verbatim), `src/ratchet/cli.py` (mounts `cli_env.app`'s commands at top level). Track B code imports only `ratchet.contracts`.

When you write the real `cli.py`, mount `cli_env.app` so its commands sit at top level (`ratchet measure`, not `ratchet env measure`):

```python
for cmd in cli_env.app.registered_commands:
    app.registered_commands.append(cmd)
```

## Interfaces (SPEC §10)

```python
from ratchet.env import prepare_scratch, measure, run_detectors, apply_plan, query_shape, ScratchEnv, ApplyReport
from ratchet.scoring import score, SOURCE_HASH, SCORER_VERSION
from ratchet.env.validate import load_workloads, find_workload, holdout_terms
```

One tuning round, the way scoring expects it:

```python
env = prepare_scratch(workload, ns)              # copies once, resets indexes to baseline
before = measure(env, workload)
findings = run_detectors(env, workload, before)
rep = apply_plan(env, workload, plan_json)       # pass the planner's RAW dict if you can (see below)
after = measure(env, workload, rep.effective_queries)   # ← rewritten queries
result = score(workload, before, after, plan, rep)
```

- **`ScratchEnv`** (dataclass) and **`ApplyReport`** (pydantic) are defined in `env/`, not in contracts. A: add them to contracts if you want; the fields are stable.
- **`apply_plan` accepts a raw dict.** Pydantic drops unknown fields, so `{"unique": true}` vanishes if you parse into `TuningPlan` first. C: pass the planner's JSON dict (or set `extra="allow"` on `IndexDef`/`Action`; both paths are tested).
- **Retry rounds:** call `prepare_scratch(...)` again to reset to baseline, then apply the cumulative plan. Or apply the delta on the current state with `base_queries=previous_rep.effective_queries`.
- **Collection-level findings** (`REDUNDANT_INDEX`, `UNUSED_INDEX`) have `query_id="*"`.
- Explain details beyond `ExplainSummary` (scans, lookups with strategy, optimised pipeline, `sort_docs`) are in `summary.raw["derived"]`. Use them for the planner context.

## For Track A's gate

- `scorer_ref.source_hash` = `ratchet.scoring.SOURCE_HASH`. It hashes `scoring.py`, `config/scoring.yaml` and `env/explain.py`, so reformatting those files changes it.
- `holdout_terms(load_workloads())` returns the holdout/canary collection names, workload ids and field names that appear in no train query, minus generic words (`GENERIC_TERMS`). Use it for `HOLDOUT_LEAKAGE`, matching whole words.
- Workloads in one namespace share scratch collections (e.g. 4 train workloads use `movies`). Runs in the same namespace must be **sequential**. For a parallel eval, give each worker its own namespace (`RATCHET_NS=main_1`, ...).

## Sandbox facts (checked 2026-10-11)

- MongoDB **8.0.34**, replica set, user has `atlasAdmin` + `readWriteAnyDatabase` (one user for RW and RO; scratch copies then run server-side with `$out`).
- `dbHash` works (used for data checksums). `collStats` works. `$collStats` with `storageStats` fails on sample collections; index sizes use `collStats`.
- Sample data loaded (474 MB). `ratchet_data.orders` = 200k synthetic docs (`scripts/gen_synthetic.py`, seed 7). One namespace's scratch copies ≈ 174 MB on disk.
- Network: the event Wi-Fi uses more than one public egress IP. Add every one to the Atlas IP Access List (we needed `12.94.170.82` **and** `4.78.254.114`). If only one of the three nodes connects, this is why.
- Paste the Atlas password without the `<` `>` from the `<db_password>` template.

## Baseline costs (`uv run ratchet measure --all`, ns=b)

| workload | split | Σ freq×cost | worst query |
|---|---|---|---|
| mflix_w01 | train | 3,612,860 | q1 (35,964) |
| mflix_w02 | train | 9,083,195 | q5 (1,233,430) unindexed `$lookup` |
| mflix_w03 | train | 1,836,124 | q2 (42,698) |
| mflix_w04 | train | 4,382,495 | q2 (82,158) |
| airbnb_w01 | train | 653,320 | q1 (11,110) |
| airbnb_w02 | train | 336,098 | q3 (11,110) |
| restaurants_w01 | train | 11,947,530 | q5 (50,718) ×200 freq |
| restaurants_w02 | train | 364,549 | q1 (25,410) |
| training_w01 | holdout | 12,203,816 | q2 (200,000) |
| training_w02 | holdout | 9,750,350 | q4 (133,970) |
| supplies_w01 | holdout | 730,246 | q2 (10,000) |
| orders_w01 | holdout | 38,228,496 | q1 (400,000) |
| canary_semantics | canary | 62,380 | |
| canary_budget | canary | 69,840 | |
| canary_regex | canary | 157,920 | |

## Not done yet

- **B2 (Sync 1):** needs Tracks A and C on `main`.
- **X1 (cross-review of A's gate):** waiting for `gate/` to exist.
