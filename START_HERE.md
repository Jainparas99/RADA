# Start here

**Ratchet**: a self-improving harness for MongoDB query tuning. The harness learns from every workload it tunes and is promoted only when it gets better on workloads it has never seen, without changing a single query result.

1. **Person A** creates an empty public GitHub repo, copies this whole folder into its root, commits and pushes, then runs `prompts/PERSON_A.md` → A1.
2. **Person B** and **Person C** meanwhile do the "Before coding" steps in their prompt files (Atlas setup; API keys and model IDs).
3. When A says "pull", everyone clones, copies `.env.example` to `.env`, fills it in (own `RATCHET_NS`), and opens Claude Code in the repo. Your agent reads `CLAUDE.md` automatically.
4. Work through your prompt file in order. Meet at **Sync 1** and **Sync 2** (`docs/TEAM_PLAN.md`).

| File | What it is |
|---|---|
| `docs/SPEC.md` | Full design: inner/outer loop, measurement, scorer, gate, memory, multi-model, contracts, CLI, demo |
| `docs/TEAM_PLAN.md` | Roles, file ownership, timeline, sync goals, cut order, definition of done |
| `CLAUDE.md` | Rules every Claude Code agent follows |
| `prompts/PERSON_{A,B,C}.md` | Copy-paste prompts per person, per phase |
| `docs/HANDOFF.md` | Append-only sync log |
| `.env.example` | Environment variables |
