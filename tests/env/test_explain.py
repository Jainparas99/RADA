"""Explain normaliser against real explain outputs (local 8.2 and the Atlas 8.0 sandbox)."""

import pytest

from ratchet.env.explain import ExplainParseError, derived, explain_command, summarize
from ratchet.scoring import query_cost
from tests.env.conftest import LABELS, load_fixture


def S(label, name):
    q, raw = load_fixture(label, name)
    return summarize(q.id, raw)


@pytest.mark.parametrize("label", LABELS)
class TestFind:
    def test_sort_in_memory_with_wrong_index(self, label):
        s = S(label, "find_sort")
        assert s.stages[:2] == ["IXSCAN", "FETCH"] and "SORT" in s.stages
        assert s.has_in_memory_sort and s.indexes_used == ["year_1"]
        assert s.n_returned == 10 and s.keys_examined > 100 * s.n_returned
        assert derived(s)["sort_docs"] == s.docs_examined

    def test_esr_index_removes_sort(self, label):
        bad, good = S(label, "find_sort"), S(label, "find_sort_esr_index")
        assert not good.has_in_memory_sort and good.indexes_used == ["esr_idx"]
        assert query_cost(good) * 100 < query_cost(bad)

    def test_covered_projection_reads_no_docs(self, label):
        s = S(label, "find_projection_covered")
        assert "PROJECTION_COVERED" in s.stages and s.docs_examined == 0
        assert s.indexes_used == ["year_1_title_1"]

    def test_collscan(self, label):
        s = S(label, "find_collscan")
        assert s.stages == ["COLLSCAN"] and s.keys_examined == 0 and s.docs_examined > 5000
        assert s.indexes_used == [] and derived(s)["scans"][0]["full_range"]

    def test_full_id_index_scan_is_flagged_full_range(self, label):
        s = S(label, "find_full_id_scan")
        assert s.indexes_used == ["_id_"]
        assert derived(s)["scans"] == [
            {"stage": "IXSCAN", "index": "_id_", "key_pattern": {"_id": 1}, "full_range": True}
        ]

    def test_express_id_path(self, label):
        s = S(label, "find_express_id")
        assert s.stages == ["EXPRESS_IXSCAN"] and s.indexes_used == ["_id_"]

    def test_or_plan(self, label):
        s = S(label, "find_or")
        assert "SORT" in s.stages and "year_1" in s.indexes_used


@pytest.mark.parametrize("label", LABELS)
class TestAggregate:
    def test_classic_lookup_counts_foreign_work_once(self, label):
        s = S(label, "agg_lookup_unwind_match")
        d = derived(s)
        assert d["engine"] == "classic" and "$lookup" in s.stages
        lk = d["lookups"][0]
        assert lk["from"] == "movies" and lk["indexed"] and lk["collection_scans"] == 0
        assert s.lookup_examined == lk["docs_examined"] + lk["keys_examined"] > 0

    def test_sbe_unindexed_lookup(self, label):
        s = S(label, "agg_lookup_unindexed")
        d = derived(s)
        assert d["engine"] == "sbe" and "EQ_LOOKUP" in s.stages
        lk = d["lookups"][0]
        assert lk["strategy"] == "NestedLoopJoin" and not lk["indexed"]
        # SBE folds foreign-side work into the totals; nothing extra in lookup_examined
        assert s.lookup_examined == 0 and s.docs_examined > 100_000

    def test_sbe_indexed_lookup_is_far_cheaper(self, label):
        bad, good = S(label, "agg_lookup_unindexed"), S(label, "agg_lookup_indexed")
        lk = derived(good)["lookups"][0]
        assert (
            lk["strategy"] == "IndexedLoopJoin"
            and lk["indexed"]
            and lk["indexes_used"] == ["movie_id_1"]
        )
        assert query_cost(good) * 100 < query_cost(bad)

    def test_pipeline_form_lookup(self, label):
        s = S(label, "agg_lookup_pipeline_form")
        lk = derived(s)["lookups"][0]
        assert lk["pipeline_form"] and lk["collection_scans"] > 0 and not lk["indexed"]
        assert s.lookup_examined > 0

    def test_full_pushdown_looks_like_find(self, label):
        s = S(label, "agg_pushdown")
        d = derived(s)
        assert d["pushed_down"] and "SORT" in s.stages and s.n_returned == 5
        assert not any(x.startswith("$") for x in s.stages)

    def test_sort_after_group_charged_by_docs_sorted(self, label):
        s = S(label, "agg_unwind_match_group_sort")
        assert s.stages[-4:] == ["$unwind", "$match", "$group", "$sort"]
        group = [p for p in derived(s)["pipeline"] if p["name"] == "$group"][0]
        assert derived(s)["sort_docs"] == group["nReturned"]
        assert query_cost(s) == s.keys_examined + s.docs_examined + group["nReturned"]

    def test_covered_group(self, label):
        s = S(label, "agg_group_indexed")
        assert "GROUP" in s.stages and s.docs_examined == 0 and s.has_in_memory_sort

    def test_count_scan(self, label):
        s = S(label, "agg_count_scan")
        assert "COUNT_SCAN" in s.stages and s.n_returned == 1 and s.docs_examined == 0


def test_trimmed_raw_is_small():
    s = S(LABELS[0], "agg_lookup_unindexed")
    assert len(str(s.raw)) < 6000  # trace-friendly: SBE slot plans are dropped


def test_unrecognised_shapes_raise():
    with pytest.raises(ExplainParseError):
        summarize("q", {})
    with pytest.raises(ExplainParseError):
        summarize("q", {"shards": {}})
    with pytest.raises(ExplainParseError):
        summarize(
            "q", {"queryPlanner": {"winningPlan": {"stage": "COLLSCAN"}}}
        )  # no executionStats


def test_pipeline_without_cursor():
    s = summarize(
        "q", {"stages": [{"$documents": [], "nReturned": 2}, {"$limit": 1, "nReturned": 1}]}
    )
    assert s.stages == ["$documents", "$limit"] and s.n_returned == 1


def test_explain_command_shapes():
    from ratchet.contracts import QuerySpec

    f = QuerySpec(
        id="a",
        kind="find",
        collection="c",
        filter={"x": 1},
        sort=[("y", -1), ("_id", 1)],
        limit=3,
        projection={"x": 1},
    )
    assert explain_command(f) == {
        "find": "c",
        "filter": {"x": 1},
        "projection": {"x": 1},
        "sort": {"y": -1, "_id": 1},
        "limit": 3,
    }
    a = QuerySpec(id="b", kind="aggregate", collection="c", pipeline=[{"$match": {}}])
    assert explain_command(a) == {"aggregate": "c", "pipeline": [{"$match": {}}], "cursor": {}}
