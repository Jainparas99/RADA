import pytest
from rich.console import Console
from typer.testing import CliRunner

from ratchet import cli_core
from ratchet import db as dbm
from ratchet.cli import app
from ratchet.contracts import PatchOp
from ratchet.evolve import Evolver
from ratchet.gate.static import check_static
from ratchet.harness import runner
from ratchet.harness.init_db import init_database
from ratchet.harness.patch import apply_patch
from ratchet.memory import MemoryFrozenError, MemoryStore, cosine, inject_text
from ratchet.memory.models import Lesson
from ratchet.proposer.propose import propose
from tests.fakes import (
    FakeBackend,
    FakeEnv,
    FakeLLM,
    fake_memory_llm,
    fake_workloads,
    scripted_proposer_llm,
)

QUERIES = {w.id: w.queries for w in fake_workloads()}
SHAPE = FakeEnv().query_shape


def make_store(db, llm=None, leak_terms=None):
    llm = llm or fake_memory_llm()
    store = MemoryStore(
        db, embed=llm.embed, complete=llm.complete, shape_fn=SHAPE, model="fake:m",
        leak_terms=leak_terms,
    )  # fmt: skip
    return store, llm


@pytest.fixture
def runs(spec):
    b = FakeBackend()
    return {s: runner.evaluate(b, spec, "h_0000", s) for s in ("train", "holdout", "canary")}


@pytest.fixture
def policy(spec):
    return spec.memory_policy


def violating(r):
    sc = r.score.model_copy(update={"score": 0.0, "violations": ["INDEX_BUDGET_EXCEEDED"]})
    return r.model_copy(update={"score": sc})


# ------------------------------------------------------------------------------- write


def test_lessons_are_written_from_improved_train_results(db, runs, policy):
    store, llm = make_store(db)
    new = store.write_lessons(runs["train"], policy, QUERIES)
    assert len(new) == 3 and db[dbm.MEMORIES].count_documents({"status": "active"}) == 3
    x = new[0]
    assert x.source == "train_improved" and x.source_workload == "fake_train_01"
    assert x.shape == SHAPE(QUERIES["fake_train_01"][0]) and "Drama" not in x.shape
    assert x.harness_version == "h_0000" and len(x.embedding) == 32 and x.status == "active"
    assert [c["role"] for c in llm.calls] == ["memory"] * 3
    brief = llm.prompts("memory")
    assert "query_shapes" in brief and "fake_holdout" not in brief


@pytest.mark.parametrize("split", ["holdout", "canary"])
def test_lessons_are_never_written_from_holdout_or_canary(db, runs, policy, split):
    store, llm = make_store(db)
    with pytest.raises(MemoryFrozenError, match=split):
        store.write_lessons([*runs["train"], *runs[split]], policy, QUERIES)
    assert db[dbm.MEMORIES].count_documents({}) == 0 and llm.calls == []


def test_writes_are_frozen_during_holdout_canary_evaluation(db, runs, policy):
    store, _ = make_store(db)
    store.write_lessons(runs["train"][:1], policy, QUERIES)
    before = list(db[dbm.MEMORIES].find())
    with store.frozen():
        assert store.is_frozen
        for write in (
            lambda: store.write_lessons(runs["train"], policy, QUERIES),
            lambda: store.record_outcomes(runs["train"], {}, policy),
            lambda: store.consolidate(policy),
        ):
            with pytest.raises(MemoryFrozenError):
                write()
        # reads still work while frozen
        assert store.recall(SHAPE(QUERIES["fake_train_01"][0]), policy)
        assert store.stats()["active"] == 1
    assert not store.is_frozen and list(db[dbm.MEMORIES].find()) == before
    assert len(store.write_lessons(runs["train"][1:], policy, QUERIES)) == 2


def test_write_sources_follow_the_policy(db, runs, policy):
    mixed = [runs["train"][0], violating(runs["train"][1])]
    store, _ = make_store(db)
    got = {x.source for x in store.write_lessons(mixed, policy, QUERIES)}
    assert got == {"train_improved", "train_violation"}

    only_violations = policy.model_copy(
        update={"write": policy.write.model_copy(update={"from_": ["train_violation"]})}
    )
    db[dbm.MEMORIES].delete_many({})
    assert [x.source for x in store.write_lessons(mixed, only_violations, QUERIES)] == [
        "train_violation"
    ]

    disabled = policy.model_copy(
        update={"write": policy.write.model_copy(update={"enabled": False})}
    )
    db[dbm.MEMORIES].delete_many({})
    assert store.write_lessons(mixed, disabled, QUERIES) == []


def test_unimproved_results_teach_nothing(db, runs, policy):
    flat = runs["train"][0]
    sc = flat.score.model_copy(update={"improvement": 0.0, "score": 0.0})
    store, llm = make_store(db)
    assert store.write_lessons([flat.model_copy(update={"score": sc})], policy, QUERIES) == []
    assert llm.calls == []


def test_lesson_text_is_capped_and_bad_replies_are_skipped(db, runs, policy):
    long = FakeLLM(default={"query_id": "q1", "lesson": "Always think. " * 100})
    store, _ = make_store(db, llm=long)
    assert len(store.write_lessons(runs["train"][:1], policy, QUERIES)[0].text) <= 300
    for bad in ("not json", {"lesson": ""}, {"query_id": "q1"}):
        db[dbm.MEMORIES].delete_many({})
        store, _ = make_store(db, llm=FakeLLM(default=bad))
        assert store.write_lessons(runs["train"][:1], policy, QUERIES) == []


def test_leaky_lessons_are_never_stored(db, runs, policy, leak_terms):
    leaky = FakeLLM(default={"query_id": "q1", "lesson": "On sales, index sold_at first."})
    store, _ = make_store(db, llm=leaky, leak_terms=leak_terms)
    assert store.write_lessons(runs["train"], policy, QUERIES) == []
    assert db[dbm.MEMORIES].count_documents({}) == 0


def test_static_gate_rejects_when_memory_holds_a_leaky_lesson(spec, leak_terms):
    ok = [PatchOp(op="replace", path="/memory_policy/inject/mode", value="always")]
    clean = ["Put equality fields first, then the sort field, then range fields."]
    assert check_static(spec, ok, leak_terms=leak_terms, lessons=clean) == []
    dirty = [*clean, "For the ѕаlеѕ collection index sold_at."]
    assert check_static(spec, ok, leak_terms=leak_terms, lessons=dirty) == ["HOLDOUT_LEAKAGE"]


# ------------------------------------------------------------------------------- retrieve


def test_recall_respects_k_and_min_similarity(db, runs, policy):
    store, _ = make_store(db)
    store.write_lessons(runs["train"], policy, QUERIES)
    shape = SHAPE(QUERIES["fake_train_01"][0])
    hits = store.recall(shape, policy)
    assert hits[0].source_workload == "fake_train_01" and hits[0].similarity == pytest.approx(1.0)
    assert all(h.similarity >= policy.retrieve.min_similarity for h in hits)
    assert store.last_search == "exact"  # mongomock has no $vectorSearch

    def with_retrieve(**kw):
        return policy.model_copy(update={"retrieve": policy.retrieve.model_copy(update=kw)})

    assert len(store.recall(shape, with_retrieve(k=8, min_similarity=0.5))) >= len(hits)
    assert len(store.recall(shape, with_retrieve(k=1, min_similarity=0.5))) == 1
    assert store.recall(shape, with_retrieve(k=0)) == []
    assert store.recall("orders | pipeline $group", with_retrieve(k=8, min_similarity=0.95)) == []


def test_recall_accepts_dict_and_view_policies(db, runs, policy):
    store, _ = make_store(db)
    store.write_lessons(runs["train"], policy, QUERIES)
    shape = SHAPE(QUERIES["fake_train_02"][0])
    as_dict = {"retrieve": {"k": 1, "min_similarity": 0.5}}

    class View:  # shaped like Track C's MemoryView
        class retrieve:  # noqa: N801
            k = 1
            min_similarity = 0.5

    assert [h.id for h in store.recall(shape, as_dict)] == [h.id for h in store.recall(shape, View)]
    assert store.recall(shape, None) == []


def test_atlas_vector_search_scores_are_converted_to_cosine(db, policy):
    lesson = Lesson(
        id="m_1", text="t", shape="s", embedding=[1.0, 0.0], source="train_improved",
        source_workload="w",
    )  # fmt: skip
    seen = {}

    class AtlasColl:
        def aggregate(self, pipeline):
            seen["stage"] = pipeline[0]["$vectorSearch"]
            return [
                {**lesson.to_doc(), "_vs": 0.95},
                {**lesson.to_doc(), "_id": "m_2", "_vs": 0.80},
            ]

    store, _ = make_store(db)
    store.coll = AtlasColl()
    hits = store.recall("some | shape", policy)  # min_similarity 0.75
    assert store.last_search == "atlas"
    assert [(h.id, round(h.similarity, 2)) for h in hits] == [("m_1", 0.9)]  # 0.80 → cos 0.6
    stage = seen["stage"]
    assert stage["index"] == "vector_index" and stage["path"] == "embedding"
    assert stage["limit"] == policy.retrieve.k and stage["filter"] == {"status": "active"}


def test_inject_text_respects_the_token_cap(db, runs, policy):
    store, _ = make_store(db)
    lessons = store.write_lessons(runs["train"], policy, QUERIES)
    assert inject_text(lessons, 0) == ""
    assert inject_text(lessons, 20).count("\n") == 0  # 80 chars: one lesson fits
    assert inject_text(lessons, 500).count("- ") == 3


# ------------------------------------------------------------------------------- consolidate


def test_duplicate_lessons_are_merged_not_duplicated(db, runs, policy):
    store, _ = make_store(db)
    store.write_lessons(runs["train"], policy, QUERIES)
    assert store.write_lessons(runs["train"], policy, QUERIES) == []
    docs = list(db[dbm.MEMORIES].find())
    assert len(docs) == 3 and all(d["support"] == 2 for d in docs)


def test_consolidate_merges_near_duplicates_and_evicts_the_weakest(db, policy):
    def add(i, vec, helpful=0, harmful=0):
        db[dbm.MEMORIES].insert_one(
            Lesson(
                id=f"m_{i}", text=f"lesson {i}", shape=f"s{i}", embedding=vec, text_embedding=vec,
                source="train_improved", source_workload="w", helpful=helpful, harmful=harmful,
            ).to_doc()
        )  # fmt: skip

    add(1, [1.0, 0.0, 0.0], helpful=3)
    add(2, [0.99, 0.05, 0.0], helpful=1)  # near-duplicate of 1
    add(3, [0.0, 1.0, 0.0], helpful=2)
    add(4, [0.0, 0.0, 1.0], harmful=1)
    store, _ = make_store(db)
    tight = policy.model_copy(
        update={"consolidate": policy.consolidate.model_copy(update={"max_items": 2})}
    )
    assert store.consolidate(tight) == {"merged": 1, "evicted": 1}
    by_id = {d["_id"]: d for d in db[dbm.MEMORIES].find()}
    assert by_id["m_2"]["status"] == "merged" and by_id["m_2"]["merged_into"] == "m_1"
    assert by_id["m_1"]["helpful"] == 4 and by_id["m_1"]["support"] == 2
    assert by_id["m_3"]["status"] == "active" and by_id["m_4"]["status"] == "evicted"
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0 and cosine([], [1.0]) == 0.0


# ------------------------------------------------------------------------------- accounting


def test_helpful_harmful_accounting_and_invalidation(db, runs, policy):
    store, _ = make_store(db)
    lessons = store.write_lessons(runs["train"], policy, QUERIES)
    text = lessons[0].text  # the fake writer gives all three the same text
    base = {r.workload_id: r.score.score for r in runs["train"]}

    def used(r, delta=0.0):
        sc = r.score.model_copy(update={"score": r.score.score + delta})
        return r.model_copy(update={"lessons_used": [text], "score": sc})

    t = runs["train"]
    counts = store.record_outcomes([used(t[0], +0.2), used(t[1]), t[2]], base, policy)
    assert counts == {"helpful": 1, "harmful": 0, "invalidated": 0}
    doc = db[dbm.MEMORIES].find_one({"_id": lessons[0].id})
    assert (doc["uses"], doc["helpful"], doc["harmful"]) == (2, 1, 0)

    store.record_outcomes([used(t[0], -0.2)], base, policy)
    assert store.stats()["active"] == 3  # one harmful use is below the limit of 2
    counts = store.record_outcomes([violating(used(t[0]))], base, policy)
    assert counts["invalidated"] == 3 and store.stats()["lessons"] == {"invalidated": 3}
    assert store.recall(SHAPE(QUERIES["fake_train_01"][0]), policy) == []
    assert store.stats() == {
        "lessons": {"invalidated": 3}, "active": 0, "uses": 12, "helpful": 3, "harmful": 6,
    }  # fmt: skip


def test_holdout_results_never_move_counters(db, runs, policy):
    store, _ = make_store(db)
    text = store.write_lessons(runs["train"], policy, QUERIES)[0].text
    hold = [violating(r.model_copy(update={"lessons_used": [text]})) for r in runs["holdout"]]
    assert store.record_outcomes(hold, {}, policy) == {"helpful": 0, "harmful": 0, "invalidated": 0}
    assert store.stats()["uses"] == 0


# ------------------------------------------------------------------------------- evolve


class WritingBackend(FakeBackend):
    """A misbehaving agent that tries to write a lesson on every tune call."""

    def __init__(self, store_ref):
        super().__init__()
        self.store_ref, self.blocked, self.allowed = store_ref, [], []

    def tune(self, workload, harness, mode):
        r = super().tune(workload, harness, mode)
        try:
            self.store_ref[0].consolidate(harness.memory_policy)
            self.allowed.append(workload.split)
        except MemoryFrozenError:
            self.blocked.append(workload.split)
        return r


def test_evolve_freezes_memory_for_holdout_and_canary_and_learns_from_train(db):
    init_database(db, vector=False)
    ref: list = []
    backend = WritingBackend(ref)
    store, _ = make_store(db)
    ref.append(store)
    llm = scripted_proposer_llm()
    ev = Evolver(
        db, backend, lambda c: propose(c, complete=llm.complete, model="fake:p"), memory=store
    )
    ev.run(2)
    assert set(backend.blocked) == {"holdout", "canary"} and set(backend.allowed) == {"train"}
    assert store.leak_terms is ev.leak_terms
    lessons = store.all()
    assert len(lessons) == 3 and {x.source_workload for x in lessons} == {
        "fake_train_01", "fake_train_02", "fake_train_03",
    }  # fmt: skip
    # h_0000's lessons, re-taught by the accepted h_0002; the rejected h_0001 taught nothing
    assert all(x.support == 2 for x in lessons)
    # the proposer saw memory stats (counts only) from generation 1 on
    first = llm.calls[0]["messages"][1]["content"]
    assert '"memory_stats":{"lessons":{"active":3}' in first
    assert lessons[0].text not in llm.prompts("proposer")


def test_proposer_can_evolve_memory_policy_within_bounds(spec, leak_terms):
    patch = [
        PatchOp(op="replace", path="/memory_policy/inject/mode", value="on_retry"),
        PatchOp(op="replace", path="/memory_policy/retrieve/k", value=5),
    ]
    assert check_static(spec, patch, leak_terms=leak_terms) == []
    assert apply_patch(spec, patch).memory_policy.retrieve.k == 5
    too_many = [PatchOp(op="replace", path="/memory_policy/retrieve/k", value=50)]
    assert check_static(spec, too_many) == ["OUT_OF_BOUNDS"]


def test_cli_memory_lists_lessons(db, runs, policy, monkeypatch):
    store, _ = make_store(db)
    store.write_lessons(runs["train"], policy, QUERIES)
    monkeypatch.setattr(cli_core, "console", Console(width=240))
    monkeypatch.setattr(cli_core, "_memory_store", lambda: store)
    out = CliRunner().invoke(app, ["memory"])
    assert out.exit_code == 0, out.output
    assert "3 active" in out.output and "fake_train_02" in out.output and "helpful" in out.output
