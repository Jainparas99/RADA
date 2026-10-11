"""Workload validation (static) + every committed workload passes static checks."""

import pytest

from ratchet.contracts import Workload
from ratchet.env.validate import WORKLOAD_DIR, check_all_static, check_static, load_workloads


def w(**kw) -> Workload:
    base = {
        "id": "x",
        "split": "canary",
        "db": "d",
        "collections": ["c"],
        "index_budget": {"c": 2},
        "write_rate": {"c": 1.0},
        "queries": [
            {
                "id": "q1",
                "kind": "find",
                "collection": "c",
                "filter": {"a": 1},
                "sort": [["b", 1], ["_id", 1]],
            },
            {"id": "q2", "kind": "find", "collection": "c", "filter": {"_id": 5}},
        ],
    }
    base.update(kw)
    return Workload.model_validate(base)


def errs(workload):
    return check_static(workload).errors


def test_valid_minimal():
    assert errs(w()) == []


def test_committed_workloads_pass_static():
    workloads = load_workloads(WORKLOAD_DIR)
    reports, cross = check_all_static(workloads)
    assert cross == [] and all(r.ok for r in reports), [(r.workload_id, r.errors) for r in reports]
    splits = [x.split for x in workloads]
    assert (splits.count("train"), splits.count("holdout"), splits.count("canary")) == (8, 4, 3)


@pytest.mark.parametrize(
    "q,needle",
    [
        (
            {"id": "q1", "kind": "find", "collection": "c", "filter": {"a": 1}},
            "_id as the last key",
        ),
        (
            {"id": "q1", "kind": "find", "collection": "c", "sort": [["_id", 1], ["b", 1]]},
            "_id as the last key",
        ),
        (
            {
                "id": "q1",
                "kind": "aggregate",
                "collection": "c",
                "pipeline": [{"$group": {"_id": "$a"}}],
            },
            "must end in $count",
        ),
        (
            {
                "id": "q1",
                "kind": "aggregate",
                "collection": "c",
                "pipeline": [{"$sort": {"_id": 1}}, {"$unwind": "$a"}],
            },
            "after the last $sort",
        ),
        (
            {
                "id": "q1",
                "kind": "aggregate",
                "collection": "c",
                "pipeline": [{"$unwind": "$a"}, {"$sort": {"_id": 1}}],
            },
            "after $unwind",
        ),
        (
            {
                "id": "q1",
                "kind": "aggregate",
                "collection": "c",
                "pipeline": [
                    {"$lookup": {"from": "c", "localField": "a", "foreignField": "b", "as": "x"}},
                    {"$sort": {"_id": 1}},
                ],
            },
            "unordered",
        ),
        (
            {
                "id": "q1",
                "kind": "aggregate",
                "collection": "c",
                "pipeline": [{"$sample": {"size": 3}}, {"$count": "n"}],
            },
            "forbidden stage",
        ),
        ({"id": "q1", "kind": "find", "collection": "zz", "sort": [["_id", 1]]}, "not in workload"),
    ],
)
def test_query_rules(q, needle):
    bad = w(queries=[q, {"id": "q2", "kind": "find", "collection": "c", "filter": {"_id": 1}}])
    assert any(needle in e for e in errs(bad)), errs(bad)


def test_lookup_reduced_by_size_is_fine():
    q = {
        "id": "q1",
        "kind": "aggregate",
        "collection": "c",
        "pipeline": [
            {"$lookup": {"from": "c", "localField": "a", "foreignField": "b", "as": "x"}},
            {"$project": {"n": {"$size": "$x"}}},
            {"$sort": {"_id": 1}},
        ],
    }
    assert (
        errs(w(queries=[q, {"id": "q2", "kind": "find", "collection": "c", "filter": {"_id": 1}}]))
        == []
    )


def test_budget_write_rate_and_counts():
    assert any("index_budget missing" in e for e in errs(w(index_budget={})))
    assert any("write_rate missing" in e for e in errs(w(write_rate={})))
    assert any("queries" in e for e in errs(w(split="train")))  # train needs 4-6
    many = [
        {"keys": [["a", 1]], "name": "a"},
        {"keys": [["b", 1]], "name": "b"},
        {"keys": [["d", 1]], "name": "d"},
    ]
    assert any("exceeds its index_budget" in e for e in errs(w(baseline_indexes={"c": many})))


def test_cross_split_collections_must_differ():
    a = w(id="a", split="canary")
    tr = w(id="t", split="train", queries=a.model_dump()["queries"] * 2)
    tr.queries = [q.model_copy(update={"id": f"q{i}"}) for i, q in enumerate(tr.queries)]
    ho = tr.model_copy(update={"id": "h", "split": "holdout"})
    _, cross = check_all_static([tr, ho])
    assert any("train and holdout share" in e for e in cross)
    other_db = ho.model_copy(update={"db": "other"})
    _, cross = check_all_static([tr, other_db])
    assert any("unique collection names" in e for e in cross)
