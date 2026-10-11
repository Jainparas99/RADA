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
