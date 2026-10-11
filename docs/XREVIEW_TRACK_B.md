# X1 cross-review: Track A attacking Track B's scorer and apply

Scope: `src/ratchet/env/apply.py` and `src/ratchet/scoring.py`, read as an attacker trying to
raise the score while changing semantics or dodging the index budget. Validators only (no
database): nothing below was run against the sandbox. Tests: `tests/core/test_xreview_track_b.py`.
The open holes are `xfail(strict=True)`, so the suite stays green now and a fix makes the test
XPASS and fail, at which point delete the marker. No Track B file was edited.

## Open holes (9 failing checks, 5 issues)

1. **Precomputed answers without a "new" literal (high).** `NEW_LITERAL` treats `0`, `1`, `-1`,
   booleans and `null` as free, and any integer can be built from those
   (`{$add: [1, 1, 1]}`), so `mflix_w02.q2` (`… | $count: n`) can be rewritten to
   `[{$limit: 1}, {$project: {_id: 0, n: <expression equal to the count>}}]`. Same result hash,
   one document read. Strings can be assembled the same way from characters of strings already
   in the query (`$substrCP` + `$concat`), and ObjectIds from those via `$toObjectId`.
   - Tests: `test_precomputed_count_built_from_free_numbers_is_refused`,
     `test_constants_hidden_in_object_keys_are_refused` (`_atoms` skips dict keys, so
     `{$literal: {"a": 1, "b": 1, "c": 1}}` carries data in its keys).
   - Suggested fix: listing constants cannot win. Either (a) refuse rewrites whose output
     fields are computed only from constants (`$literal`, or an operator with no field
     reference underneath), or better (b) a metamorphic check: run the original and the
     rewrite on a deterministic subset of the collection (an extra leading `$match` on a hash
     of `_id`) and require equal hashes there too. A hard-coded answer fails on the subset.

2. **Sort tie-break can be dropped (medium).** Removing `_id` from a find `sort`, removing the
   sort altogether, or removing `_id` from a pipeline `$sort` is accepted. The result is then
   only equal by accident of storage order, and a cheaper index can satisfy it.
   - Tests: `test_find_rewrite_cannot_drop_the_sort_tie_break`,
     `test_find_rewrite_cannot_drop_the_sort_entirely`,
     `test_aggregate_rewrite_cannot_drop_the_sort_tie_break`.
   - Suggested fix: a find rewrite's sort must be the same key/direction list; in a pipeline,
     every `$sort` that precedes a `$limit` must still end in `_id`.

3. **Aggregate rewrites can lose a predicate (medium).** A find rewrite must keep its filter,
   but an aggregate rewrite may drop a `$match` condition. If that condition is true for all
   matching documents today the hash is unchanged.
   - Test: `test_aggregate_rewrite_cannot_drop_a_predicate`.
   - Suggested fix: the mirror image of `NEW_LITERAL`: every constant of the original must
     still appear in the rewrite (`query_atoms(original) - query_atoms(new)` must be empty).
     The metamorphic check in (1) also narrows this.

4. **Wildcard indexes are allowed (medium).** `{"$**": 1}` and `{"a.$**": 1}` pass
   `validate_index`: one budget slot indexes every field, which is the over-indexing the
   budget and `canary_budget` exist to stop. Only the size term of the penalty pushes back.
   - Test: `test_wildcard_index_is_refused`.

5. **Drop-and-recreate earns credit (low).** Dropping an unused baseline index and creating the
   same keys under a new name changes nothing, but earns `γ` (0.02) against
   `α·write_rate + β·size` (0.006 for a 0.2 MB index on `movies`), so a no-op plan scores
   about +0.014.
   - Test: `test_dropping_and_recreating_the_same_index_scores_nothing`.
   - Suggested fix: no credit for a dropped index whose key pattern is re-created in the plan.

## Tried and held

- `unique`, `sparse`, `hidden`, `expireAfterSeconds`, `collation` on an index: refused, now also
  when the plan arrives as a parsed `TuningPlan` (contracts keep unknown keys since `8e9446f`).
- Dropping `_id_`, naming an index `_id_`, `drop_index "*"` (rejected as a missing index).
- `$out`, `$merge`, `$where`, `$lookup` / `$unionWith` into a collection outside the workload.
- Changed find filter or limit; answer ids smuggled in a `$match … $in`.
- Indexes or extra action fields outside the workload / contract.
- Laundering constants across retry rounds: rewrites are always compared with the workload's
  original query, not the previous round's.
- Frequency inflation on a rewritten query: the scorer uses the workload's frequencies.
- Every collection in all 15 workloads has an index budget (a missing one would mean unlimited).
- Partial indexes: could not find a way for one to change results; the planner only uses it
  when the query implies the filter.

## Not checked

- Whether `explain` under-reports work for some stage shapes on the sandbox (for example a
  `$facet` or a correlated sub-pipeline). The normaliser counts `$lookup`, `$graphLookup` and
  `$unionWith`; confirming the rest needs a live run.
