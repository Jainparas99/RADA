# Person A — Core loop & gate (lead)

Paste these into your Claude Code session one at a time. Wait for each to finish, run, and commit before the next.

## A1 — Kickoff: scaffold + contracts (do this first; the others are waiting)

```
You are on Track A. Read CLAUDE.md, docs/SPEC.md and docs/TEAM_PLAN.md fully.
Scaffold the repo with uv (Python 3.12): pyproject (ratchet package, `ratchet` console
script), ruff, pytest, Makefile per CLAUDE.md, MIT LICENSE, .gitignore (incl .env), and keep
the existing .env.example. Create the full directory layout from CLAUDE.md with empty
__init__.py files so other tracks can start. Write src/ratchet/contracts.py EXACTLY as in
SPEC §10 (fix only syntax if needed). Write config.py (env loading, RATCHET_NS), db.py
(RW and RO clients, `ratchet_<ns>` database, collection handles). Write tests/fakes.py
with FakeEnv (implements the Track B interface from SPEC §10 using canned Measurements)
and FakeLLM (returns scripted JSON). Write cli.py mounting cli_core.py, cli_env.py,
cli_demo.py (the latter two as empty Typer apps). Add a test that contracts import and
round-trip. Run make test and make lint, commit to main, and tell me when to tell the
team to pull. Keep it fast; aim for 20 minutes.
```

## A2 — Phase 1: harness store, gen 0, static gate

```
Track A, Phase 1. Implement: HarnessSpec / MemoryPolicy pydantic models (SPEC §6, §8)
in harness/models.py; config/bounds.yaml and config/gate.yaml with SPEC values;
harness/store.py (save_candidate, get_head, promote, rollback, history); the gen-0
harness exactly as SPEC §6 describes; `ratchet init` creating every collection and index
in SPEC §12 (vector indexes too — verify the current pymongo/Atlas API for search
indexes first); gate/static.py with every static reason code from SPEC §9 including
source-hash checks for scorer and gate; tests with a positive and negative case per
code. Add `ratchet eval --split S` that runs tune() for each workload of a split and
prints a Rich table — use FakeEnv/FakeLLM until Tracks B and C land. Commit on a/phase1,
merge to main when green. Then append to docs/HANDOFF.md.
```

## A3 — Sync 1 integration

```
Sync 1. Pull main (Tracks B and C have merged). Wire eval to the real env (Track B) and
real tune() (Track C). With RATCHET_NS from .env, run `ratchet init`, `ratchet seed`,
then `ratchet eval --split train` on the real sandbox. Fix any seam bugs in my files; for
bugs in other tracks' files, write me a precise bug note to pass on instead of editing
them. Show me the gen-0 table.
```

## A4 — Phase 2: proposer, dynamic gate, evolve

```
Track A, Phase 2. Implement: proposer/context.py building the context from SPEC §9-§10
(current spec, editable paths + bounds, up to 8 TRAIN failures chosen to cover distinct
detectors/violations with their raw traces, memory stats, last 5 gate decisions) and a
test proving no holdout/canary workload, trace, result or scorer/gate source can reach the
prompt; proposer/propose.py calling llm.complete with role "proposer" and strict
ProposerOutput JSON (one repair retry); patch application with jsonpatch restricted to
editable paths; gate/dynamic.py with every dynamic code in SPEC §9 and tests; evolve.py
(eval head → propose → static gate → run candidate on train+holdout+canary with memory
frozen for holdout/canary → dynamic gate → promote/reject → store generation and
GateDecision), canary re-check with auto-rollback at gen start; reuse stored parent
results instead of re-running the parent. CLI: evolve, history, diff (coloured spec diff),
rollback. Run `ratchet evolve --gens 2` and show me history plus one diff. Commit, merge.
```

## A5 — Phase 3: memory policy, then the full run

```
Track A, Phase 3. Implement memory/ per SPEC §8: lesson writer (LLM distils a lesson from
train TuneResults that improved or had violations; uses query_shape() from Track B),
embed via llm.embed (Track C), vector search retrieval with k/min_similarity, injection
per mode with token cap, consolidation by merge_similarity, invalidation after harmful
uses, helpful/harmful accounting from later results, HOLDOUT_LEAKAGE check on lesson text
wired into the static gate, write-freeze test. Make the proposer context include memory
stats so it can evolve memory_policy. Add `ratchet memory`. Commit, merge.
Then do the full run: set RATCHET_NS=main, `ratchet init && ratchet seed`, then
`ratchet evolve --gens 8`. While it runs, report progress every generation. At the end,
show history and tell me the headline numbers for the README.
```

## X1 — Cross-review (once, in Phase 3)

```
Review Track B's scoring.py and env/apply.py purely as an attacker: try to construct
TuningPlans that improve the score while changing semantics or evading the budget
(partial indexes that change results? rewrites that reorder ties? collation tricks?
dropping _id_? hidden indexes?). Write failing tests for anything that gets through and
a short note for Track B. Do not edit their files.
```
