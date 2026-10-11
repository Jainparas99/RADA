import pytest

from ratchet import db as dbm
from ratchet.contracts import PatchOp
from ratchet.gate import hashes
from ratchet.harness.init_db import INDEXES, init_database, vector_definition
from ratchet.harness.models import HarnessDoc, HarnessSpec, Scores, version_id
from ratchet.harness.patch import PatchError, apply_patch
from ratchet.harness.store import HarnessStore, StoreError


def test_gen0_is_the_minimal_baseline(spec):
    assert len(spec.planner_rules) == 2 and spec.critic_rubric == []
    assert spec.critic_policy.enabled is False
    assert spec.context_policy.sample_docs == 0
    assert spec.context_policy.include_detector_findings is False
    assert spec.memory_policy.inject.mode == "never"
    assert spec.retry_policy.max_rounds == 1
    assert spec.protected.allowed_actions == ["create_index", "drop_index", "rewrite_query"]
    assert spec.model_routing.planner in spec.protected.model_allowlist
    assert spec.protected.gate_ref.source_hash == hashes.gate_hash()
    assert spec.protected.scorer_ref.source_hash == hashes.scorer_hash()


def test_spec_roundtrips_with_wire_names(spec):
    doc = spec.to_doc()
    assert doc["memory_policy"]["write"]["from"] == ["train_improved", "train_violation"]
    assert HarnessSpec.from_doc(doc) == spec


def test_spec_is_strict_and_forbids_extras(spec):
    doc = spec.to_doc()
    for mutate in (
        lambda d: d.update(extra=1),
        lambda d: d["retry_policy"].update(max_rounds="2"),
        lambda d: d["critic_policy"].update(enabled=1),
        lambda d: d["memory_policy"]["inject"].update(mode="sometimes"),
    ):
        bad = HarnessSpec.from_doc(doc).to_doc()
        mutate(bad)
        with pytest.raises(ValueError):
            HarnessSpec.from_doc(bad)


def test_version_id():
    assert version_id(7) == version_id("7") == version_id("h_0007") == "h_0007"
    with pytest.raises(ValueError):
        version_id("abc")


def test_apply_patch(spec):
    out = apply_patch(spec, [PatchOp(op="add", path="/planner_rules/-", value="Equality first.")])
    assert out.planner_rules[-1] == "Equality first." and len(spec.planner_rules) == 2
    with pytest.raises(PatchError):
        apply_patch(spec, [PatchOp(op="remove", path="/planner_rules/9")])
    with pytest.raises(PatchError):
        apply_patch(spec, [PatchOp(op="replace", path="/retry_policy/max_rounds", value="x")])


def test_init_creates_collections_indexes_and_gen0(db):
    rep = init_database(db)
    assert set(dbm.ALL_COLLECTIONS) <= set(db.list_collection_names())
    assert sorted(rep.created_collections) == sorted(dbm.ALL_COLLECTIONS)
    assert len(rep.indexes) == sum(len(v) for v in INDEXES.values())
    assert db[dbm.HARNESS_VERSIONS].index_information()["version_1"]["unique"] is True
    assert db[dbm.LLM_CACHE].index_information()["key_1"]["unique"] is True
    assert "harness_version_1_split_1" in db[dbm.TUNE_RESULTS].index_information()
    assert rep.head == "h_0000" and rep.head_created
    # mongomock has no Atlas Search; init reports it and carries on
    assert all(v.startswith("failed") for v in rep.vector_indexes.values())
    again = init_database(db)
    assert again.created_collections == [] and not again.head_created
    assert db[dbm.HARNESS_VERSIONS].count_documents({}) == 1


def test_vector_definition():
    d = vector_definition(1024, ["status"])
    assert d["fields"][0] == {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 1024,
        "similarity": "cosine",
    }
    assert d["fields"][1] == {"type": "filter", "path": "status"}


@pytest.fixture
def store(db, spec):
    s = HarnessStore(db)
    s.init_gen0(spec)
    return s


def test_store_head_candidate_promote(store, spec):
    assert store.get_head().id == "h_0000" and store.get_head().status == "accepted"
    patch = [PatchOp(op="add", path="/planner_rules/-", value="Equality, sort, range.")]
    cand = store.save_candidate("h_0000", apply_patch(spec, patch), patch=patch, rationale="ESR")
    assert (cand.id, cand.version, cand.parent, cand.status) == ("h_0001", 1, "h_0000", "candidate")
    assert store.get_head().id == "h_0000"
    store.set_scores(cand.id, Scores(train=0.6, holdout=0.5, canary_pass=1.0))
    promoted = store.promote(1)
    assert promoted.status == "accepted" and promoted.scores.holdout == 0.5
    assert store.get_head().id == "h_0001"
    assert store.get_head().patch == patch
    assert isinstance(store.get(1), HarnessDoc)


def test_store_reject_keeps_head_and_records_codes(store, spec):
    cand = store.save_candidate(0, spec)
    rejected = store.reject(cand.id, ["NO_HOLDOUT_GAIN"])
    assert rejected.status == "rejected" and rejected.reason_codes == ["NO_HOLDOUT_GAIN"]
    assert store.get_head().id == "h_0000"
    with pytest.raises(StoreError):
        store.promote(cand.id)


def test_store_refuses_stale_promotion(store, spec):
    a = store.save_candidate(0, spec)
    b = store.save_candidate(0, spec)
    store.promote(a.id)
    with pytest.raises(StoreError, match="head is h_0001"):
        store.promote(b.id)


def test_store_rollback_and_history(store, spec):
    a = store.save_candidate(0, spec)
    store.promote(a.id)
    rej = store.save_candidate(a.id, spec)
    store.reject(rej.id, ["CANARY_FAILED"])
    with pytest.raises(StoreError):
        store.rollback(rej.id)
    store.rollback(0, reason="canary")
    assert store.get_head().id == "h_0000"
    assert [(d.id, d.status) for d in store.history()] == [
        ("h_0000", "accepted"),
        ("h_0001", "accepted"),
        ("h_0002", "rejected"),
    ]
    assert [e["action"] for e in store.head_log()] == ["init", "promote", "rollback:canary"]
    with pytest.raises(StoreError):
        store.get(99)


def test_repin_creates_human_version_with_new_hashes(store):
    doc = store.repin(
        {"version": "1", "source_hash": "sha256:new-scorer"},
        {"version": "1", "source_hash": "sha256:new-gate"},
        "scorer landed",
    )
    assert doc.id == "h_0001" and doc.created_by == "human" and doc.status == "accepted"
    assert store.get_head().spec.protected.scorer_ref.source_hash == "sha256:new-scorer"
