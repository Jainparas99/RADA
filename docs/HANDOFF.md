# Handoff log

Append one entry per sync: time, track, what landed on main, what's next, anything broken or blocked.

- (pre-build) Planning docs written: `docs/SPEC.md`, `docs/TEAM_PLAN.md`, `CLAUDE.md`, `prompts/`. No code yet. Next: Track A runs Prompt A1.

- (2026-10-11, Track A, after A1 + A2; A2 is on branch `person_A`, not yet on `main`)
  - **Landed (A1, on main):** uv scaffold, `contracts.py` (SPEC §10), `config.py`, `db.py`, CLI shell, `tests/fakes.py`.
  - **Landed (A2):** `harness/models.py` (`HarnessSpec`, `MemoryPolicy`, `HarnessDoc`), `harness/gen0.py`, `harness/store.py` (`save_candidate`, `get_head`, `promote`, `reject`, `rollback`, `history`, `repin`), `harness/patch.py`, `harness/init_db.py` + `ratchet init`, `gate/static.py` (all 9 static codes, positive + negative tests), `gate/leakage.py`, `gate/hashes.py`, `harness/runner.py` + `ratchet eval --split S [--fake]`, `config/bounds.yaml`, `config/gate.yaml`.
  - **Not verified against the sandbox:** `ratchet init` has only run on mongomock (no `.env` on this machine yet). Vector index creation uses `create_search_index(SearchIndexModel(..., type="vectorSearch"))` and reports a per-collection failure instead of aborting if the tier refuses.
  - **For Track B:** the gate hashes `src/ratchet/scoring.py` + `config/scoring.yaml` itself (`gate/hashes.py`). Any change to those files after `ratchet init` makes every patch fail with `SCORER_TAMPERED` until someone runs `ratchet pin --reason "..."`. Same for `gate/*.py`, `config/gate.yaml`, `config/bounds.yaml` → `GATE_TAMPERED`. The real eval imports `prepare_scratch` from `ratchet.env` (export it from `env/__init__.py`) and reads `workloads/*.json`.
  - **For Track C:** the real eval imports `tune` from `ratchet.agent` (export it from `agent/__init__.py`) with the SPEC §10 signature; the runner fills in `harness_version` afterwards. `HarnessSpec` lives in `ratchet.harness.models`; wire name for the memory write sources is `memory_policy.write.from` (python attribute `from_`). Vector indexes are named `vector_index` on field `embedding`, cosine, `RATCHET_EMBED_DIMS` (default 1024) — tell A if your embedding model differs.
  - **Fakes moved:** implementation is in `src/ratchet/fakes.py`; `tests/fakes.py` re-exports it, so `from tests.fakes import FakeEnv, FakeLLM` still works. New: `fake_tune`, `FakeBackend`, `fake_workloads`.
  - **Next (A):** proposer context + call, dynamic gate, `evolve`, `history`/`diff`/`rollback` — all buildable on fakes.

- (2026-10-11, Track A, after A4 on fakes; on branch `person_A`, not yet on `main`)
  - **Landed:** `gate/dynamic.py` (all 5 dynamic codes, positive + negative tests), `proposer/context.py` (train-only context, failure selection, redaction), `proposer/propose.py` (role `proposer`, strict `ProposerOutput`, one repair retry), `evolve.py` (canary re-check + auto-rollback, stored parent results reused, static → run → dynamic → promote/reject, `generations` + `gate_decisions` written), CLI `evolve`, `history`, `diff`, `rollback`.
  - **Only exercised on fakes.** `ratchet evolve --fake` uses an in-memory DB, `FakeEnv`, a fake agent and a *scripted* proposer; its numbers mean nothing. Nothing in A4 has touched Atlas, a real `tune()` or a real model yet (A3 is still to do once B and C merge).
  - **For Track C:** `propose()` calls `ratchet.llm.complete("proposer", model, messages, json_schema=...)` and reads the reply from `.data` (parsed dict) or `.text`; tokens from `.tokens` or `.input_tokens + .output_tokens`. Tell A if `LLMResponse` uses other names. `messages[0]` has role `system`.
  - **For Track C (`attack`, outer part):** use `ratchet.gate.static.static_report(parent_spec, patch, leak_terms=LeakTerms.from_workloads(all_workloads))` and `ratchet.gate.dynamic.check_dynamic(parent_metrics, cand_metrics)`.
  - **Next (A):** A3 integration when B and C are on main; A5 memory policy.

- (2026-10-11, Track B) Branch `b/env-measurement`, not yet on main. **Landed:** scratch copies, explain normaliser (fixtures from Atlas 8.0.34 + local 8.2), `measure`, `apply_plan` (SPEC §5.5 + NEW_LITERAL anti-precompute), scorer with all 5 inner violations and SOURCE_HASH, all 9 detectors, `query_shape`, 15 workloads (8/4/3) validated live on the sandbox, synthetic `ratchet_data.orders` (200k), `ratchet seed / measure [--all] / validate`, 166 tests (incl. Track A's). Rebased on A1; uses `ratchet.db`, touches no Track A file; the env is stateful like `FakeEnv`. **Next:** Sync 1 with A and C; X1 once `gate/` exists. **Watch out:** the event Wi-Fi has several egress IPs, so add them all to the Atlas IP Access List. Details in `docs/TRACK_B_NOTES.md`.

- (2026-10-11, Track A, Sync 1 check; `person_A` now contains `main` = Track B)
  - **Checked:** trial-merged `person_A` + `main` (B) + `origin/C` (at C4): one text conflict (`docs/HANDOFF.md`), tests green apart from one stale Track A test (fixed). Nothing has been run against Atlas or a real model from Track A's machine (no `.env` here).
  - **Fixed in Track A files:** `RealBackend` now passes `tune()` an `api` that has `score` (see bug note); gate's scorer hash covers `env/explain.py` like `scoring.SOURCE_HASH`; `config/gate.yaml` ignores generic held-out names (`limit`, `status`, `items`, …) so ordinary rules are not rejected as `HOLDOUT_LEAKAGE`; proposer turns `ratchet.llm` errors into `NO_PROPOSAL`.
  - **Bug note for B and C (not fixed, your files):** `agent/loop.py` calls `api.score(...)` and `cli_demo.py` passes `api = ratchet.env`, but `ratchet.env` has no `score` (it lives in `ratchet.scoring`). `ratchet tune <workload>` without `--fake` will raise `AttributeError` after the plan is applied. Either B adds `from ratchet.scoring import score` to `env/__init__.py`, or C builds the api object the way `harness/runner.py::_EnvApi` does. `ratchet eval` / `evolve` are not affected.
  - **Note for C:** `C` is still based on A1. Merge `main` and `person_A` before the next push; `attack --outer` imports `ratchet.gate.*` and `ratchet.harness.gen0`, which only exist on `person_A`.
  - **Note for B:** A's leak matcher is `gate/leakage.py::LeakTerms` (also folds unicode look-alikes); it does not use `holdout_terms`. X1 can start: `gate/` is on `person_A`.
  - **Still missing (A):** `ratchet.memory` (A5). `tune()` already calls `ratchet.memory.recall(shape, policy)` when it exists.

- (2026-10-11, Track A, after A5 code; on `person_A`)
  - **Landed:** `ratchet.memory` (SPEC §8): `recall(shape, policy)` and `write_lessons(results, policy, queries)` at module level, `MemoryStore` (LLM-distilled lessons from train results, shape + text embeddings, `$vectorSearch` retrieval with exact-cosine fallback, `k` / `min_similarity`, consolidation by `merge_similarity` + `max_items`, helpful/harmful accounting, invalidation after harmful uses, write freeze), `inject_text(lessons, max_tokens)`, `ratchet memory`. `evolve` writes lessons from the head's and newly accepted versions' TRAIN runs, freezes memory while holdout/canary run, feeds counts-only memory stats to the proposer, and the static gate returns `HOLDOUT_LEAKAGE` if any active lesson names a held-out collection/field.
  - **Only tested on fakes + mongomock.** The `$vectorSearch` call and the Atlas score → cosine conversion (`2*score - 1`) have not run against the sandbox. The full run (`RATCHET_NS=main`, `evolve --gens 8`) has NOT been done: `main` has no Track C code yet and this machine has no `.env`.
  - **For Track C:** `recall` returns `Lesson` objects with `.text` (and `.similarity`). Put the lesson *texts* you injected into `TuneResult.lessons_used`; helpful/harmful accounting matches on text. Lesson distillation calls `llm.complete("memory", <planner model>, messages, json_schema=<dict>)`; embeddings call `llm.embed(texts)` and `llm.embed([shape], input_type="query")`.
  - **Track A depends on Track C for:** `ratchet.llm.complete` (proposer, lesson writer), `ratchet.llm.embed` (memory), `ratchet.agent.tune` (`eval`, `evolve`). All are imported lazily; without them only the `--fake` paths run.

- (2026-10-11, Track A, reply to Track C's notes; on `person_A`)
  - **Contracts change (additive):** `IndexDef` and `Action` now use `extra="allow"`, so unknown keys such as `unique` survive `TuningPlan` parsing and reach `env.apply_plan` (which already reads `model_extra`). `FakeEnv` now returns `FORBIDDEN_ACTION` for any extra key.
  - **Windows:** every text read in Track A code and tests now passes `encoding="utf-8"`.
  - **Stable names for `attack --outer`:** `gate.static.check_static`, `gate.dynamic.check_dynamic`, `gate.dynamic.RunMetrics`, `harness.gen0.gen0_spec` will not be renamed.

- (2026-10-11, Track A, X1 cross-review of Track B; on `person_A`)
  - **For Track B:** `docs/XREVIEW_TRACK_B.md` lists 5 issues in `env/apply.py` / `scoring.py` (precomputed answers built from "free" constants, droppable sort tie-breaks, droppable `$match` predicates in aggregate rewrites, wildcard indexes, drop-and-recreate credit) plus what held. Tests are in `tests/core/test_xreview_track_b.py` as `xfail(strict=True)`: when you fix one, its test XPASSes and fails the run; delete the `@hole` marker then. Validators only, nothing was run on the sandbox.

- **Sync 1 — Track C (branch `C`)**
  - *Landed:* `ratchet.llm.complete(role, model, messages, tools=None, json_schema=None, *, system,
    temperature, max_tokens)` → `LLMResponse` (`text`, `data`, `parsed`, `tokens`, `input_tokens`,
    `output_tokens`, `cost_usd`, `cached`, `model`). `provider:model` ids; anthropic required,
    openai/google optional (`ProviderUnavailable` when no key; `llm.available(model)` to check).
    `{"role":"system"}` messages are folded into the system prompt. Mongo `llm_cache` keyed by
    sha256(model, system, messages, params) (in-memory when no URI). JSON mode accepts a dict schema
    or a pydantic class, validates, retries 2x with the validation error, then `LLMJSONError`.
    Backoff on 408/409/429/5xx/timeouts. `llm.embed(texts)` → Voyage (cached) or local fallback;
    `RATCHET_EMBED_MODE=atlas` flags Atlas automated embeddings. `ratchet.agent.tune(workload,
    harness, env, mode, *, api, llm, memory, harness_version, critic)` → `TuneResult` with full trace;
    runs end to end on `FakeEnv`. `ratchet tune <workload> [--version V] [--fake] [--json]`.
  - *Seams / asks:* **B** — `tune()` calls module functions on `ratchet.env`: `measure(env, w)`,
    `run_detectors(env, w, m)`, `apply_plan(env, w, plan)`, `score(w, before, after, plan, apply)`,
    `query_shape(q)`; optional `collection_stats(env, coll)`, `sample_docs(env, coll, n)` (used only
    if present). `measure(env, w)` must measure the *current* (rewritten) queries held by the scratch
    env. Apply reports should be pydantic with list / dict-of-list fields (rounds are merged).
    **A** — `RealBackend` already matches `tune(w, harness, scratch, mode)`. Memory: `tune()` calls
    `ratchet.memory.recall(shape, policy)` if it exists; lessons may be str, dict or objects with
    `.text`. `init` should keep the unique index on `llm_cache.key`.
  - *Next (C):* critic + revisions, model routing/retry policy, tools incl. `self_diagnose` /
    `recall_lessons`, inner attacks.
  - *Broken / not verified:* no live run yet — this checkout has no `.env` (needs `RATCHET_NS=c`,
    `ANTHROPIC_API_KEY`, Atlas URIs). OpenAI/Google adapters are written against current SDK docs
    but untested without keys. Current Claude models reject `temperature`; determinism comes from
    `llm_cache`, not temperature 0.
- **Sync 2 — Track C (branch `C`)**
  - *Landed:* critic (`agent/critic.py`): a model different from the planner (other provider
    preferred: harness `model_routing.critic` → `RATCHET_CRITIC_MODEL` → `anthropic:claude-sonnet-5-5`)
    reviews every plan against base checks + `critic_rubric`; uses its revised plan or has the
    planner revise, up to `critic_policy.max_revisions`; fails open; verdicts in the trace;
    `tune()` runs it by default whenever the harness enables it. Planner tools (`agent/tools.py`):
    `explain`, `list_indexes`, `collection_stats`, `sample_docs`, `findings`, `recall_lessons`
    (gated by `memory_policy.inject.mode`), `self_diagnose` (→ `TuneResult.self_diagnostics`),
    via a JSON tool protocol that works for every provider and FakeLLM. Forbidden-action scan on
    the planner's raw JSON (unknown types, types outside `allowed_actions`, unique/sparse/TTL/hidden,
    `_id_` drop, `$out`/`$merge`/`hint`) → `FORBIDDEN_ACTION`, score 0. `ratchet attack --inner`
    (`--fake` works offline): all four SPEC §14 inner cheats rejected via the agent path; three of
    four also caught by the scorer alone.
  - *Asks:* **A** (contracts, additive): `IndexDef` drops unknown keys, so `unique: true` can't reach
    Track B's apply; the agent scan catches it today. Consider `model_config = ConfigDict(extra="allow")`
    on `IndexDef` so B's apply can also reject it (`FORBIDDEN_ACTION`) on the scorer path.
    **A**: `ratchet attack --outer` will call your static gate (`check_static(parent, patch)`) and
    dynamic gate (`check_dynamic(parent_scores, cand_scores)`); please keep those names stable.
  - *Next (C):* attack outer part, transfer, chart, watch, replay, README + DEMO.md.
  - *Broken / not verified:* still no live run (no `.env` in this checkout). OpenAI/Google adapters
    untested without keys.
- **Phase 3 / freeze-ready — Track C (branch `C`)**
  - *Landed:* `ratchet attack` now has both levels: 4 inner cheats (agent path + scorer-only path)
    and 5 outer cheats through Track A's `check_static` / `check_dynamic` (loaded lazily; shown as
    `n/a` until the gate is on main). Verified locally against `origin/person_A` merged into a
    throwaway worktree: **all 12 cheats rejected** with the expected codes, all Track C tests pass.
    `ratchet transfer` (gen 0 vs best on holdout per callable planner model → `transfer_runs` +
    `docs/transfer.md`), `ratchet chart` (`docs/chart.png` from `generations`, fallback
    `harness_versions`), `ratchet watch` (change streams, polling fallback), `ratchet replay
    [--gen N | --version V]` (cache-only LLM client; misses reported), `RATCHET_LLM_CACHE_ONLY=1`
    for any command. README (results table left as TODO for the final run) and `docs/DEMO.md`
    (3-min script, pre-flight, fallbacks, 60-s video script).
  - *Asks:* **A** — `tests/core/test_evolve.py::test_proposer_never_sees_holdout_canary_or_gate_source`
    fails on Windows: a `read_text()` without `encoding="utf-8"` hits cp1252 on `gate/leakage.py`.
    Merging `person_A` into `C` conflicts only in `docs/HANDOFF.md` (both append; keep both).
    **B** — please drop `docs/README_measurement.md`; I'll merge it into the README's "How
    measurement works" section.
  - *Next (C):* live `ratchet tune` on the sandbox once `.env` exists; real numbers into README
    after A's full run; README references need verifying; record video.
  - *Broken / not verified:* nothing live yet (no `.env` here). OpenAI/Google adapters untested.
- **Sync with Track A — Track C (branch `C`)**
  - *Landed:* `unique: true` now reaches the scorer (contracts `extra="allow"`), so `attack` drops the "contract strips it" branch: **13/13 cheats rejected** with `--fake`. `embed()` vectors are `RATCHET_EMBED_DIMS` wide (default 1024): local fallback uses it and Voyage is called with `output_dimension` (voyage-3.5 default is 1024); wrong-width replies raise. Memory contract checked: `lessons_used` holds exact `Lesson.text`s (auto-inject and `recall_lessons` tool), `tune()` only calls `recall`, and `recall` gets the harness's own `memory_policy`. Lesson writer calls (`complete("memory", ..., json_schema=dict)`, `embed(..., input_type="query")`) work with the client.
  - *Asks:* **A** — `tests/core/test_gate_static.py:294` `write_text(...)` needs `encoding="utf-8"` (fails on Windows: cp1252 cannot encode `→` in `config/*.yaml`).
  - *Next (C):* live run on the sandbox (`info --ping`, `init`, `seed`, `tune mflix_w02`, `eval --split train`).
