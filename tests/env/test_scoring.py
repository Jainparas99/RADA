"""Scorer (SPEC §5.6): cost formula, penalty, and every inner violation (+ and -)."""

import pytest

from ratchet import scoring
from ratchet.contracts import TuningPlan
from ratchet.env.apply import ActionError, ApplyReport
from tests.env.conftest import measurement, summary

CFG = {"alpha": 0.002, "beta": 0.01, "gamma": 0.02, "score_min": -1.0, "score_max": 1.0}
EMPTY = TuningPlan(actions=[])


def ok_apply(**kw) -> ApplyReport:
    return ApplyReport(ok=True, **kw)


def before_after(w, before_cost=(1000, 100), after_cost=(10, 1)):
    b = measurement(
        w, {"q1": summary("q1", docs=before_cost[0]), "q2": summary("q2", docs=before_cost[1])}
    )
    a = measurement(
        w, {"q1": summary("q1", docs=after_cost[0]), "q2": summary("q2", docs=after_cost[1])}
    )
    return b, a


# --------------------------------------------------------------------------- cost


def test_query_cost_formula():
    assert scoring.query_cost(summary(keys=5, docs=7, lookup=11)) == 23
    assert scoring.query_cost(summary(keys=5, docs=7, sort=True)) == 5 + 7 + 7  # spec: docs again
    s = summary(keys=5, docs=7, sort=True)
    s.raw = {"derived": {"sort_docs": 2}}
    assert scoring.query_cost(s) == 5 + 7 + 2  # pipeline $sort: docs flowing into it


def test_workload_cost_uses_workload_freq_not_rewritten(small_workload):
    m = measurement(small_workload, {"q1": summary("q1", docs=10), "q2": summary("q2", docs=4)})
    assert scoring.workload_cost(small_workload, m) == 50 * 10 + 5 * 4


def test_improvement_clamped():
    assert scoring.improvement(100, 10) == pytest.approx(0.9)
    assert scoring.improvement(100, 500) == -1.0
    assert scoring.improvement(0, 5) == 0.0


def test_clean_improvement_scores(small_workload):
    b, a = before_after(small_workload)
    r = scoring.score(small_workload, b, a, EMPTY, ok_apply(), CFG)
    assert (
        r.violations == []
        and r.improvement == pytest.approx(1 - 505 / 50500)
        and r.score == r.improvement
    )


def test_index_penalty_write_rate_size_and_redundant_credit(small_workload):
    b, a = before_after(small_workload)
    rep = ok_apply(
        created={"comments": ["movie_id_1"], "movies": ["g_r"]},
        index_sizes_mb={"comments": {"movie_id_1": 2.0}, "movies": {"g_r": 1.0}},
        dropped={"movies": ["year_1"]},
    )  # year_1 is a prefix of year_1_title_1
    pen = scoring.index_penalty(small_workload, b, rep, CFG)
    assert pen == pytest.approx(0.002 * 40 + 0.01 * 2 + 0.002 * 2 + 0.01 * 1 - 0.02)


def test_dropping_a_used_non_redundant_index_gets_no_credit(small_workload):
    b = measurement(
        small_workload, {"q1": summary("q1", used=["year_1_title_1"]), "q2": summary("q2")}
    )
    rep = ok_apply(dropped={"movies": ["year_1_title_1"]})
    assert scoring.dropped_redundant(small_workload, b, rep) == 0
    assert (
        scoring.dropped_redundant(small_workload, b, ok_apply(dropped={"movies": ["year_1"]})) == 1
    )


def test_source_hash_pins_files(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("a")
    h1 = scoring.compute_source_hash((f,))
    f.write_text("b")
    assert scoring.compute_source_hash((f,)) != h1
    assert (
        scoring.SOURCE_HASH.startswith("sha256:")
        and scoring.SOURCE_HASH == scoring.compute_source_hash()
    )


# --------------------------------------------------------------------------- violations


def test_result_mismatch(small_workload):
    b, a = before_after(small_workload)
    a.result_hashes["q1"] = "different"
    r = scoring.score(small_workload, b, a, EMPTY, ok_apply(), CFG)
    assert r.violations == ["RESULT_MISMATCH"] and r.score == 0.0
    assert r.improvement > 0  # still reported, but not scored


def test_result_mismatch_negative(small_workload):
    b, a = before_after(small_workload)
    assert (
        "RESULT_MISMATCH"
        not in scoring.score(small_workload, b, a, EMPTY, ok_apply(), CFG).violations
    )


def test_missing_query_counts_as_mismatch(small_workload):
    """Dropping a slow query from the after-measurement can't make it free."""
    b, a = before_after(small_workload)
    del a.per_query["q2"]
    del a.result_hashes["q2"]
    assert scoring.score(small_workload, b, a, EMPTY, ok_apply(), CFG).violations == [
        "RESULT_MISMATCH"
    ]


def test_data_mutated(small_workload):
    b, a = before_after(small_workload)
    a.data_checksums["comments"] = "changed"
    assert scoring.score(small_workload, b, a, EMPTY, ok_apply(), CFG).violations == [
        "DATA_MUTATED"
    ]
    b2, a2 = before_after(small_workload)
    assert scoring.score(small_workload, b2, a2, EMPTY, ok_apply(), CFG).violations == []


def test_index_budget_exceeded_from_actual_state(small_workload):
    b, a = before_after(small_workload)
    a.indexes["comments"] = ["_id_", "a_1", "b_1", "c_1"]  # budget 2
    assert scoring.score(small_workload, b, a, EMPTY, ok_apply(), CFG).violations == [
        "INDEX_BUDGET_EXCEEDED"
    ]
    a.indexes["comments"] = ["_id_", "a_1", "b_1"]  # _id_ doesn't count
    assert scoring.score(small_workload, b, a, EMPTY, ok_apply(), CFG).violations == []


@pytest.mark.parametrize("code", ["FORBIDDEN_ACTION", "APPLY_ERROR", "INDEX_BUDGET_EXCEEDED"])
def test_apply_report_violations_propagate(small_workload, code):
    b, a = before_after(small_workload)
    rep = ApplyReport(
        ok=False, violations=[code], errors=[ActionError(action_index=0, code=code, detail="x")]
    )
    r = scoring.score(small_workload, b, a, EMPTY, rep, CFG)
    assert r.violations == [code] and r.score == 0.0


def test_violation_order_is_stable(small_workload):
    b, a = before_after(small_workload)
    a.result_hashes["q1"] = "x"
    a.data_checksums["movies"] = "y"
    rep = ApplyReport(ok=False, violations=["FORBIDDEN_ACTION"])
    assert scoring.score(small_workload, b, a, EMPTY, rep, CFG).violations == [
        "RESULT_MISMATCH",
        "DATA_MUTATED",
        "FORBIDDEN_ACTION",
    ]


def test_score_clamped(small_workload):
    b, a = before_after(small_workload)
    rep = ok_apply(dropped={"movies": ["year_1"] * 100})
    assert scoring.score(small_workload, b, a, EMPTY, rep, CFG).score == 1.0
