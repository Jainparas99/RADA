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
