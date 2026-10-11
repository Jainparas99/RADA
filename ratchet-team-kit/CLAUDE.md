# CLAUDE.md — Ratchet

Ratchet is a self-improving harness for MongoDB query tuning, built by a team of three at the Harness Engineering & Model Wrangling hackathon. Inner loop: an agent diagnoses a slow workload with `explain()`, plans index changes and pipeline rewrites, a critic reviews, we apply on a scratch copy and re-measure. Outer loop: a proposer patches the harness from train traces; a deterministic gate promotes a patch only if held-out workloads improve and no invariant breaks.

**Read before writing code:** `docs/SPEC.md` (source of truth), `docs/TEAM_PLAN.md` (who owns what). Ask your human which track you're on (A, B or C) if they haven't said. **Only edit files your track owns.** If you need a change elsewhere, stop and tell your human what to ask for.

## Invariants (never violate; ask before changing)

1. Code against `src/ratchet/contracts.py`. Don't redefine those models elsewhere. Only Track A edits it.
2. The agent's only actions are `create_index`, `drop_index`, `rewrite_query` (SPEC §5.5), applied **only to scratch copies**, never to source sample data.
3. A tuning plan that changes any query result, changes data, blows the index budget or uses a forbidden action gets workload score 0 and a recorded violation.
4. Scorer and gate are deterministic code with pinned source hashes. No LLM in any accept/reject or scoring path.
5. The proposer outputs only a JSON patch over editable harness paths. It never sees holdout or canary workloads, their traces or results, or scorer/gate source. A test enforces this.
6. Protected harness fields (`protected/*`) never change via a patch. `model_routing` values must be in the allowlist.
7. Memory lessons are written only from train runs; writes frozen during holdout/canary evaluation.
8. Every tuning run stores a full trace: context shown, every LLM message and tool call, plan, critic verdict, before/after measurements, models used, tokens.
9. All LLM calls go through `llm.complete()` with `llm_cache`.
10. The only database is the event's MongoDB Atlas Sandbox. Use your `RATCHET_NS` namespace; never touch another person's namespace or `main` unless doing the final run.

## Hackathon rules that shape code

- New work only; all code written at the event.
- Terminal CLI (Typer + Rich) + PNG chart. **No Streamlit, no web dashboard.**
- Vector search is for retrieving lessons and failures, not a document chatbot.
- Public repo at submission.

## Stack

Python 3.12, `uv`, pydantic v2, `pymongo`, `anthropic`, optional `openai` / `google-genai`, `voyageai` (fallback for embeddings), Typer, Rich, matplotlib, `jsonpatch`, pytest, ruff. **Verify current SDK and MongoDB APIs on first use** (especially explain output shape for aggregates on the sandbox's MongoDB version) instead of assuming. Build test fixtures from real sandbox output.

Env vars (`.env`, see `.env.example`): `MONGODB_URI_RW`, `MONGODB_URI_RO`, `RATCHET_NS`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (optional), `GOOGLE_API_KEY` (optional), `VOYAGE_API_KEY` (optional), `RATCHET_PLANNER_MODEL`, `RATCHET_CRITIC_MODEL`, `RATCHET_PROPOSER_MODEL`.

## Layout

```
src/ratchet/
  contracts.py config.py db.py                 [A]
  harness/ gate/ proposer/ memory/ evolve.py   [A]
  cli.py cli_core.py                           [A]
  env/ (scratch, explain, detectors, apply)    [B]
  scoring.py cli_env.py                        [B]
  llm/ agent/ attack.py transfer.py viz/       [C]
  cli_demo.py                                  [C]
config/  bounds.yaml gate.yaml [A]  scoring.yaml detectors.yaml [B]
workloads/  *.json validate.py                 [B]
scripts/  gen_synthetic.py                     [B]
tests/  core/ [A]  env/ [B]  agent/ [C]  fakes.py [A]
docs/   SPEC.md TEAM_PLAN.md HANDOFF.md DEMO.md[C]
```

## Commands

- `make setup` → `uv sync`
- `make test` → `uv run pytest -q`
- `make lint` → `uv run ruff check . && uv run ruff format --check .`
- `make seed` → `uv run ratchet init && uv run ratchet seed`
- `make evolve` → `uv run ratchet evolve --gens 8`
- `make demo` → `uv run ratchet history && uv run ratchet attack && uv run ratchet chart`

## Conventions

- Type hints everywhere; pydantic for anything stored or crossing a module boundary.
- Thresholds, bounds and weights in `config/*.yaml`.
- Every inner violation and gate reason code has a test that triggers it and one that doesn't.
- Use `tests/fakes.py` (`FakeEnv`, `FakeLLM`) to test without the network.
- Small commits on your track branch (`a/*`, `b/*`, `c/*`); `make test lint` before every merge to `main`.
- Append to `docs/HANDOFF.md` at each sync.
- If behind, cut in the order in `docs/TEAM_PLAN.md`. Never cut scorer violations, gate tests or held-out evaluation.

## Honesty rules (README and demo)

- "Blocks a defined set of cheating patterns", never "solves reward hacking".
- Attack-mode cheats are seeded and labelled.
- Report raw numbers on our workloads. Gen 0 is a deliberately minimal baseline.
- Acknowledge Atlas Performance Advisor exists; explain the difference.
- README has a "Built at this hackathon" section.
