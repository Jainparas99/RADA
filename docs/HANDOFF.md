# Handoff log

Append one entry per sync: time, track, what landed on main, what's next, anything broken or blocked.

- (pre-build) Planning docs written: `docs/SPEC.md`, `docs/TEAM_PLAN.md`, `CLAUDE.md`, `prompts/`. No code yet. Next: Track A runs Prompt A1.
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
