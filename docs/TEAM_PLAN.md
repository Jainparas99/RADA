# Team plan — 3 people, 3 Claude Code agents

Each person runs their own Claude Code session on their own track, in the same public repo. Tracks are split by directory so agents never edit the same files. Everyone codes against `src/ratchet/contracts.py` (SPEC §10), so nobody waits on anybody.

## Roles

| | Person A — **Core loop & gate** (lead) | Person B — **Environment & measurement** | Person C — **Agents, models & demo** |
|---|---|---|---|
| Owns | `contracts.py`, `config.py`, `db.py`, `harness/`, `gate/`, `proposer/`, `memory/`, `evolve.py`, `cli.py`, `cli_core.py`, `config/{bounds,gate}.yaml`, `tests/core/`, `tests/fakes.py` | `env/` (scratch, explain, detectors, apply), `scoring.py`, `workloads/`, `scripts/gen_synthetic.py`, `cli_env.py`, `config/{scoring,detectors}.yaml`, `tests/env/` | `llm/` (providers, cache, embed), `agent/` (planner, critic, tools, loop), `attack.py`, `transfer.py`, `viz/` (chart, watch), `replay`, `cli_demo.py`, `tests/agent/`, `README.md`, `docs/DEMO.md` |
| SPEC sections | §6, §8, §9, §10, §12 | §5 | §7, §11, §13 (demo cmds), §14, §15 |
| Biggest risk | gate correctness | explain output parsing on the sandbox's MongoDB version | LLM JSON reliability, provider keys |
| First deliverable | contracts + fakes + repo scaffold (≤ 20 min) | Atlas users + data verified, `measure` on 1 real workload | `llm.complete()` + cache (others depend on it) |

Suggested fit: A = whoever is strongest on systems/safety; B = strongest on MongoDB; C = strongest on LLM APIs and presenting.

## Ground rules

- **One owner per directory.** Need a change in someone else's area? Ask in the team chat; they (or their agent) make it.
- `contracts.py` changes only by A, announced in chat, and only additive after Sync 1.
- Branches: `a/*`, `b/*`, `c/*`. Small PRs, merged to `main` by the owner after `make test lint` passes. Rebase on `main` at every sync.
- Each person sets `RATCHET_NS` (`a` / `b` / `c`) — your own scratch namespace and `ratchet_<ns>` DB in the shared sandbox. Final run uses `main`.
- Append to `docs/HANDOFF.md` at every sync: what landed, what's next, what's broken.
- Don't print or commit secrets. `.env` is gitignored.

## Timeline

| Time | A | B | C |
|---|---|---|---|
| **Kickoff (≈15 min)** | Create repo, scaffold (Prompt A1), push contracts + fakes | Atlas: load sample data, create RW + RO users, note tier limits (storage, index count, change streams, `dbHash`), share URIs privately | Collect API keys (Anthropic, + OpenAI/Google if any, Voyage), confirm model IDs, share privately |
| **Phase 1 → Sync 1** | Harness store, gen-0 harness, static gate checks + tests, `init`, `eval` skeleton on fakes | Scratch copies, explain normaliser (fixtures from real output), scorer + inner violations, 3 workloads, `measure` | `llm/` providers + cache + `embed`, planner + apply loop on `FakeEnv`, `tune` command |
| **Sync 1** | **Goal: `ratchet tune <real workload>` runs end to end with gen-0 harness on the real sandbox.** Merge all three, fix seams. | | |
| **Phase 2 → Sync 2** | Proposer context + call + patch apply, dynamic gate, `evolve`, `history`, `diff`, `rollback`, canary auto-rollback | All 15 workloads (8/4/3) + `validate.py`, all detectors, synthetic collection, `query_shape()` | Critic + revisions, retry policy, tools incl. `self_diagnose` and `recall_lessons`, model routing, `attack` (inner part) |
| **Sync 2** | **Goal: `ratchet evolve --gens 2` produces at least one ACCEPT or a clean REJECT with reason codes.** | | |
| **Phase 3 → Freeze** | Memory policy (write/retrieve/inject/consolidate/invalidate), leakage check, `memory`; then the **full run** (8–10 gens, `RATCHET_NS=main`) | Hardening: edge cases in explain parsing, scorer tests, canary traps behaving; then help with README "How measurement works" | `attack` (outer part), `transfer`, `chart`, `watch`, `replay`; README + DEMO.md draft |
| **Feature freeze** | No new features after this. Only fixes. | | |
| **Submission** | Final results numbers → README table; tag `v0.1.0` | Clean-clone check: `uv sync && make test lint && make demo` | Record 60-s video, rehearse 3-min demo ×2, submit, repo public, all members added |

Event anchors: hacking starts 10:30, submissions due **5:00 PM**, judging from 5:00. Leave the last 45–60 minutes for video, README and rehearsal. If you're behind at Sync 2, cut (in order): `watch` → `replay` → consolidation → `transfer` (keep it as a 2-model table if possible) → invalidation. **Never cut:** scorer violations, gate tests, held-out evaluation.

## Critical path

`contracts` → `llm.complete` + `measure` → `tune` (Sync 1) → `propose` + gate → `evolve` (Sync 2) → full run → chart/demo.

The full evolution run takes real time (≈ 15 workloads × N gens). Start it as soon as Sync 2 passes, on `RATCHET_NS=main`, and keep building features against your own namespace while it runs. `llm_cache` means a re-run after a fix is cheap.

## Using the different models

| Role | Default | Why |
|---|---|---|
| Planner (inner) | `anthropic:claude-haiku-5-5` | cheap, fast, many calls |
| Critic (inner) | a different provider if available (`openai:*` / `google:*`), else `anthropic:claude-sonnet-5-5` | independent check, not self-grading |
| Proposer (outer) | `anthropic:claude-sonnet-5-5` (or Opus if budget allows) | reads many traces, few calls |
| Embeddings | Atlas automated Voyage embeddings, else `voyageai` | sponsor stack |
| Transfer test | every planner model you have keys for | Model Wrangling slide |

Also use your Claude Code agents differently: A's agent writes adversarial tests against B's scorer and C's planner output; B's agent reviews A's gate for bypasses; C's agent reviews B's explain parser against real outputs. Do this once in Phase 3 (Prompt X1).

## Definition of done (whole project)

- [ ] `ratchet evolve` full run on `main` with ≥ 1 accepted and ≥ 1 rejected candidate
- [ ] `ratchet attack` shows every cheat in SPEC §14 rejected with the right code
- [ ] `ratchet chart` PNG committed; `ratchet transfer` table in README
- [ ] All tests pass; lint clean; clean-clone check passes
- [ ] README: pitch, how it works, measurement, gate codes, results, honest limits, Built at this hackathon, references
- [ ] 60-s video recorded; 3-min live demo rehearsed twice; repo public; submission form done
