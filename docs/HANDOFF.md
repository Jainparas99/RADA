# Handoff log

Append one entry per sync: time, track, what landed on main, what's next, anything broken or blocked.

- (pre-build) Planning docs written: `docs/SPEC.md`, `docs/TEAM_PLAN.md`, `CLAUDE.md`, `prompts/`. No code yet. Next: Track A runs Prompt A1.
- (2026-10-11, Track B) Branch `b/env-measurement`, not yet on main. **Landed:** scratch copies, explain normaliser (fixtures from Atlas 8.0.34 + local 8.2), `measure`, `apply_plan` (SPEC §5.5 + NEW_LITERAL anti-precompute), scorer with all 5 inner violations and SOURCE_HASH, all 9 detectors, `query_shape`, 15 workloads (8/4/3) validated live on the sandbox, synthetic `ratchet_data.orders` (200k), `ratchet seed / measure [--all] / validate`, 148 tests. **Provisional:** pyproject, contracts.py, cli.py (Track A: take yours). **Next:** Sync 1 with A and C; X1 once `gate/` exists. **Watch out:** the event Wi-Fi has several egress IPs, so add them all to the Atlas IP Access List. Details in `docs/TRACK_B_NOTES.md`.
