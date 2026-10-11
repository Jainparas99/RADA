# Person C — Agents, models & demo

Paste these into your Claude Code session one at a time. Wait for each to finish, run, and commit before the next.

## Before coding (while A scaffolds)

- Collect keys: Anthropic (required), OpenAI and/or Google (optional, for the cross-model critic and transfer test), Voyage (only if Atlas automated embeddings aren't available). Share privately.
- Confirm the exact model IDs you have access to for each provider and post them in team chat; they go into `model_allowlist`.

## C1 — Phase 1: LLM client first, then the planner loop

```
You are on Track C. Read CLAUDE.md, docs/SPEC.md (§7, §11, §13-§15 are yours) and
docs/TEAM_PLAN.md. Pull main. On branch c/phase1, FIRST (others depend on it, merge it
as soon as it works): llm/ with complete(role, model, messages, tools, json_schema) and
embed(texts); providers anthropic (required), openai and google (optional, skipped if no
key); model IDs as "provider:model"; Mongo-backed llm_cache keyed by hash of
(model, messages, tools, schema); token + cost logging; strict-JSON helper with one repair
retry. Verify current SDK APIs before writing. Embeddings: try Atlas automated Voyage
embeddings if Track B confirms availability, else voyageai SDK. Tests with FakeLLM.
Merge that to main and tell me.
THEN: agent/ per SPEC §7 — context builder driven by harness.context_policy, planner call
producing a TuningPlan, tools (explain, list_indexes, collection_stats, sample_docs,
findings, self_diagnose), apply → measure → retry per retry_policy, full trace capture,
returning TuneResult. Develop against FakeEnv from tests/fakes.py. Add `ratchet tune
<workload>` in cli_demo.py printing findings, the plan, and a before/after table
(stages, docs examined, cost, improvement, violations). Commit, merge when green.
```

## C2 — Sync 1

```
Sync 1. Pull main. Run `ratchet tune <real workload>` on the real sandbox with my
RATCHET_NS. Fix issues in my files; write precise bug notes for other tracks. Show me
the output.
```

## C3 — Phase 2: critic, routing, inner attacks

```
Track C, Phase 2. Add the critic per SPEC §7 (separate model via model_routing, checks the
plan against critic_rubric, may request up to critic_policy.max_revisions revisions),
recall_lessons tool calling Track A's memory.recall (stub if not merged yet),
model_routing support for planner/critic, models_used recorded in TuneResult. Then
attack.py inner part (SPEC §14): seeded, clearly labelled cheating TuningPlans pushed
through Track B's apply+score, each printed with its violation code. `ratchet attack
--inner`. Tests. Commit, merge.
```

## C4 — Phase 3: demo surface + README

```
Track C, Phase 3. Implement: attack.py outer part (seeded harness patches through Track A's
gate, SPEC §14) so `ratchet attack` shows both levels; transfer.py + `ratchet transfer`
(gen-0 vs best harness on HOLDOUT for each available planner model, results stored in
transfer_runs, Rich table); viz/chart.py + `ratchet chart` (PNG: train and holdout mean
score per generation, rejected candidates marked ✕ with reason code, accepted versions
annotated); viz/watch.py + `ratchet watch` (change streams on harness_versions and
gate_decisions; fall back to polling if not supported on the tier); `ratchet replay` from
llm_cache. Then draft README.md (pitch, problem, how it works with a text diagram, inner
vs outer loop, gate codes, multi-model, results table placeholder, honest limits, Built at
this hackathon, reproduce commands, references from SPEC §17) and docs/DEMO.md (the
3-minute script from SPEC §15 with exact commands). Merge Track B's
docs/README_measurement.md into the README. Commit, merge.
```

## C5 — Submission

```
Final. Pull main after Track A's full run. Run `ratchet history`, `ratchet chart`,
`ratchet attack`, `ratchet transfer` on RATCHET_NS=main and put the real numbers into the
README results table (raw numbers, honest). Do a clean-clone check in a temp dir:
uv sync, make test, make lint, make demo. Fix anything broken in my files. Give me a
60-second video script with the exact commands and what to say over each.
```

## X1 — Cross-review (once, in Phase 3)

```
Review Track B's env/explain.py against the real explain fixtures and against 3 new
queries I'll run now on the sandbox (a $group, a covered query, a $facet). Report any
field that is mis-parsed or any stage the normaliser misses, as failing tests plus a
short note for Track B. Do not edit their files.
```
