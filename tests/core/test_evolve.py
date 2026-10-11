from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import ratchet
from ratchet import cli_core
from ratchet import db as dbm
from ratchet.cli import app
from ratchet.contracts import PatchOp
from ratchet.evolve import Evolver
from ratchet.harness.init_db import init_database
from ratchet.harness.patch import apply_patch
from ratchet.harness.store import HarnessStore
from ratchet.proposer.propose import propose
from tests.fakes import (
    SCRIPTED_PROPOSALS,
    FakeBackend,
    FakeLLM,
    fake_workloads,
    scripted_proposer_llm,
)

HIDDEN = {w.id for w in fake_workloads() if w.split != "train"}


def make(db, llm=None, backend=None):
    init_database(db, vector=False)
    llm = llm or scripted_proposer_llm()
    backend = backend or FakeBackend()
    ev = Evolver(
        db,
        backend,
        lambda ctx: propose(ctx, complete=llm.complete, model="fake:scripted"),
        proposer_model="fake:scripted",
    )
    return ev, llm, backend


@pytest.fixture
def full_run(db):
    ev, llm, backend = make(db)
    return ev, llm, backend, ev.run(7)


def test_scripted_run_accepts_improvements_and_rejects_each_cheat(full_run):
    ev, _, _, recs = full_run
    assert [(r.gen, r.verdict, r.reason_codes) for r in recs] == [
        (1, "REJECT", ["NO_HOLDOUT_GAIN", "OVERFIT_DIVERGENCE"]),
        (2, "ACCEPT", []),
        (3, "REJECT", ["PROTECTED_FIELD_CHANGED"]),
        (4, "REJECT", ["HOLDOUT_LEAKAGE"]),
        (5, "REJECT", ["CANARY_FAILED", "INNER_VIOLATIONS_UP", "NO_HOLDOUT_GAIN"]),
        (6, "ACCEPT", []),
        (7, "REJECT", ["NO_HOLDOUT_GAIN"]),
    ]
    assert [r.head_after for r in recs] == ["h_0000", "h_0002", *["h_0002"] * 3, "h_0006", "h_0006"]
    head = ev.store.get_head()
    assert head.id == "h_0006" and head.parent == "h_0002"
    assert head.scores.holdout > ev.store.get(2).scores.holdout > ev.store.get(0).scores.holdout
    assert head.models_used["proposer"] == "fake:scripted"
    assert [d.status for d in ev.store.history()] == [
        "accepted", "rejected", "accepted", "rejected", "rejected", "rejected", "accepted",
        "rejected",
    ]  # fmt: skip


def test_generations_and_gate_decisions_are_recorded(full_run, db):
    gens = list(db[dbm.GENERATIONS].find().sort("gen", 1))
    assert [g["gen"] for g in gens] == list(range(1, 8))
    assert gens[1]["verdict"] == "ACCEPT" and gens[1]["head_before"] == "h_0000"
    assert gens[1]["candidate"] == "h_0002" and gens[1]["duration_s"] >= 0
    decisions = list(db[dbm.GATE_DECISIONS].find().sort("candidate", 1))
    assert len(decisions) == 7
    static = db[dbm.GATE_DECISIONS].find_one({"candidate": "h_0003"})
    assert static["metrics"]["stage"] == "static" and static["verdict"] == "REJECT"
    dyn = db[dbm.GATE_DECISIONS].find_one({"candidate": "h_0002"})
    assert dyn["metrics"]["candidate"]["holdout"] > dyn["metrics"]["parent"]["holdout"]


def test_statically_rejected_patch_never_becomes_a_spec_or_runs(full_run, db):
    ev, _, _, _ = full_run
    for vid in ("h_0003", "h_0004"):
        doc = ev.store.get(vid)
        assert doc.status == "rejected" and doc.patch
        assert doc.spec == ev.store.get(doc.parent).spec
        assert db[dbm.TUNE_RESULTS].count_documents({"harness_version": vid}) == 0


def test_parent_results_are_reused_not_rerun(full_run):
    _, _, backend, _ = full_run
    runs = {}
    for wid, _mode in backend.modes:
        runs[wid] = runs.get(wid, 0) + 1
    # train/holdout: once for gen 0 + once per dynamically evaluated candidate (5 of them)
    assert runs["fake_train_01"] == runs["fake_holdout_01"] == 6
    # canary: the head re-check at the start of each of the 7 generations + the 5 candidates
    assert runs["fake_canary_01"] == 7 + 5


def test_memory_is_frozen_for_holdout_and_canary(full_run):
    _, _, backend, _ = full_run
    modes = {(wid, mode) for wid, mode in backend.modes}
    assert {m for w, m in modes if w.startswith("fake_train")} == {"normal"}
    assert {m for w, m in modes if w in HIDDEN} == {"frozen_memory"}


def test_proposer_never_sees_holdout_canary_or_gate_source(full_run, db):
    ev, llm, _, _ = full_run
    prompts = llm.prompts("proposer")
    assert len(llm.calls) == 7 and "fake_train_01" in prompts
    # no held-out workload, collection, field or query name
    assert ev.leak_terms.find(prompts) == []
    for wid in HIDDEN:
        assert wid not in prompts
    # no held-out results or metrics (the fake's rejected candidates score 0.0 on holdout only
    # through violations, so the violation code must not appear either)
    assert db[dbm.GATE_DECISIONS].count_documents({"metrics.stage": "dynamic"}) == 5
    assert "INDEX_BUDGET_EXCEEDED" not in prompts
    assert '"holdout"' not in prompts and "canary_violations" not in prompts
    # gate feedback did arrive, as reason codes
    assert "OVERFIT_DIVERGENCE" in prompts and "PROTECTED_FIELD_CHANGED" in prompts
    # no scorer or gate source
    src = Path(ratchet.__file__).parent
    for f in [*(src / "gate").glob("*.py"), src / "scoring.py"]:
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if len(line) > 40 and not line.startswith(("#", '"""')):
                assert line not in prompts, f"{f.name}: {line}"


def test_no_proposal_is_recorded_and_leaves_head_alone(db):
    ev, _, _ = make(db, llm=FakeLLM({"proposer": ["garbage", "more garbage"]}))
    rec = ev.run_generation()
    assert rec.verdict == "NO_PROPOSAL" and rec.candidate is None and rec.head_after == "h_0000"
    assert db[dbm.GENERATIONS].count_documents({}) == 1
    assert db[dbm.GATE_DECISIONS].count_documents({}) == 0
    assert len(ev.store.history()) == 1


def test_canary_recheck_rolls_back_a_bad_head(db, spec):
    ev, _, _ = make(db, llm=FakeLLM({"proposer": [SCRIPTED_PROPOSALS[1]]}))
    store = HarnessStore(db)
    greedy = apply_patch(
        spec, [PatchOp(op="add", path="/planner_rules/-", value="Index every field.")]
    )
    bad = store.save_candidate("h_0000", greedy, rationale="slipped past an older gate")
    store.promote(bad.id)
    events = []
    ev.on_event = lambda kind, data: events.append((kind, data))
    rec = ev.run_generation()
    assert rec.rolled_back_from == "h_0001" and rec.head_before == "h_0000"
    assert ("rollback", {"from": "h_0001", "to": "h_0000"}) in events
    assert store.head_log()[-2]["action"] == "rollback:canary"
    # the generation then carried on from the restored head
    assert rec.verdict == "ACCEPT" and store.get_head().parent == "h_0000"


def test_canary_recheck_with_no_clean_ancestor_stops(db, spec):
    backend = FakeBackend()
    greedy = apply_patch(
        spec, [PatchOp(op="add", path="/planner_rules/-", value="Index every field.")]
    )
    HarnessStore(db).init_gen0(greedy)
    ev, _, _ = make(db, backend=backend)
    with pytest.raises(RuntimeError, match="fails canaries"):
        ev.run_generation()


def test_gate_tamper_blocks_promotion(db, monkeypatch):
    ev, _, _ = make(db, llm=FakeLLM({"proposer": [SCRIPTED_PROPOSALS[1]]}))
    monkeypatch.setattr("ratchet.gate.hashes.gate_hash", lambda: "sha256:edited-after-pin")
    rec = ev.run_generation()
    assert rec.verdict == "REJECT" and rec.reason_codes == ["GATE_TAMPERED"]
    assert ev.store.get_head().id == "h_0000"


# ------------------------------------------------------------------------------- CLI


@pytest.fixture
def wide(monkeypatch):
    monkeypatch.setattr(cli_core, "console", Console(width=220))


def test_cli_evolve_fake(wide):
    out = CliRunner().invoke(app, ["evolve", "--gens", "2", "--fake"])
    assert out.exit_code == 0, out.output
    assert "FAKE run" in out.output and "OVERFIT_DIVERGENCE" in out.output
    assert "ACCEPT" in out.output and "h_0002" in out.output
    assert "include_detector_findings" in out.output  # the diff


def test_cli_history_diff_rollback(wide, full_run, monkeypatch):
    ev = full_run[0]
    monkeypatch.setattr(cli_core, "_store", lambda: ev.store)
    cli = CliRunner()
    hist = cli.invoke(app, ["history"]).output
    assert "head = h_0006" in hist and "HOLDOUT_LEAKAGE" in hist and "rejected" in hist
    d = cli.invoke(app, ["diff", "0", "2"]).output
    assert '-    "include_detector_findings": false' in d
    assert '+    "include_detector_findings": true' in d and "why:" in d
    assert "identical" in cli.invoke(app, ["diff", "2", "3"]).output
    assert cli.invoke(app, ["rollback", "2"]).exit_code == 0 and ev.store.get_head().id == "h_0002"
    assert cli.invoke(app, ["rollback", "5"]).exit_code == 1  # rejected version
    assert cli.invoke(app, ["diff", "0", "99"]).exit_code == 1
