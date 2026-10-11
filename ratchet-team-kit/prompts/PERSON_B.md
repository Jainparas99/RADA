# Person B — Environment & measurement

Paste these into your Claude Code session one at a time. Wait for each to finish, run, and commit before the next.

## Before coding (while A scaffolds)

In the Atlas UI for the event's sandbox:
- Load the sample dataset. Note which `sample_*` DBs exist and their sizes.
- Create two DB users: **RW** (readWrite on `ratchet_*` and `ratchet_scratch_*` DBs) and **RO** (read on `sample_*`). Share both URIs privately with the team.
- Note the tier and its limits: storage, max indexes per collection, whether change streams and `dbHash` are available, the MongoDB version.
- Post those findings in team chat; they decide what's possible.

## B1 — Phase 1: scratch, explain, scorer, 3 workloads

```
You are on Track B. Read CLAUDE.md, docs/SPEC.md (§5 is yours) and docs/TEAM_PLAN.md.
Pull main (Track A has scaffolded and written contracts.py). On branch b/phase1 implement:
env/scratch.py (prepare_scratch: copy each workload collection once into
ratchet_scratch_<ns> using the RW user, reset indexes to baseline_indexes before every run,
cleanup); env/explain.py: FIRST run explain("executionStats") on the real sandbox for one
find with sort, one find with projection, one aggregate with $lookup+$unwind+$match, save
the raw outputs as test fixtures, THEN write the normaliser to ExplainSummary for both find
and aggregate shapes (including $lookup examined counts); measure() with order-sensitive
result hashes (bson.json_util canonical JSON, SHA-256) and data checksums; env/apply.py
validating SPEC §5.5 (forbidden options and actions) and applying to scratch only;
scoring.py exactly per SPEC §5.6 with config/scoring.yaml and a source-hash constant;
tests for every inner violation code (positive and negative). Write 3 real workloads
(2 train, 1 holdout) per SPEC §5.2 and `ratchet measure <workload>` in cli_env.py printing a
Rich table (stages, nReturned, keys/docs examined, ratio). Commit, merge to main when green.
```

## B2 — Sync 1

```
Sync 1. Pull main. Run `ratchet seed` and `ratchet measure` on the real sandbox with my
RATCHET_NS. Help Track C's `ratchet tune` run end to end on one workload; fix issues in my
files, and write precise bug notes for issues in other tracks' files.
```

## B3 — Phase 2: full workload set, detectors, synthetic data

```
Track B, Phase 2. Implement all detectors in SPEC §5.4 (env/detectors.py,
config/detectors.yaml) with tests built from real explain fixtures; query_shape() (values
stripped, operators kept, stable string); workloads/validate.py (run each query twice,
reject non-deterministic, empty or >500-doc results; check each sorted/limited query has a
unique tie-break); the full workload set: 8 train, 4 holdout, 3 canary per SPEC §5.2 table,
with train and holdout on DIFFERENT collections and a good mix of problems per workload;
canary traps per SPEC §5.2. If the tier's storage allows, scripts/gen_synthetic.py for a
seeded ~200k-doc ratchet_data.orders collection and 1-2 holdout workloads on it. Run
validate on everything, and `ratchet measure` on each, and show me a table of baseline
workload costs. Commit, merge.
```

## B4 — Phase 3: hardening + README section

```
Track B, Phase 3. Read any bug notes or failing tests from Track A's cross-review and fix
them in my files. Add tests for: rewrites that reorder ties, partial indexes that would
change results, attempts to drop _id_, unique/TTL/hidden options, $lookup into a
collection outside the workload. Then write a "How measurement works" section for the
README (explain fields, cost formula, why examined-counts not time, inner violations) in
docs/README_measurement.md for Track C to merge in. Commit, merge.
```

## X1 — Cross-review (once, in Phase 3)

```
Review Track A's gate/ as an attacker: craft ProposerOutput patches that should be
rejected but pass (path tricks like /protected/../, escaped JSON pointers, adding keys
under protected via 'add', model IDs with different casing, leaking holdout names through
unicode lookalikes). Write failing tests and a short note for Track A. Do not edit their files.
```
