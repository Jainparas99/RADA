import json

from ratchet import replay as rp
from ratchet import transfer as tr
from ratchet.agent import tune
from ratchet.agent.harness_view import AgentHarness
from ratchet.contracts import ScoreReport, TuneResult, TuningPlan
from ratchet.llm.cache import MemoryCache
from ratchet.llm.client import LLMClient, get_client, set_client
from ratchet.llm.types import RawCompletion
from ratchet.viz import chart, watch
from tests.fakes import FakeEnv, sample_workload

ESR_PLAN = {
    "actions": [
        {
            "type": "create_index",
            "collection": "movies",
            "index": {"keys": [["genres", 1], ["rating", -1], ["year", 1]]},
        }
    ],
    "summary": "ESR",
}


def workloads():
    return [
        sample_workload("t1", "train"),
        sample_workload("h1", "holdout", "orders"),
        sample_workload("h2", "holdout", "films"),
    ]


# ---------------------------------------------------------------- transfer


def fake_tune(w, doc, label):
    env = FakeEnv()
    before = env.measure(env.prepare_scratch(w), w)
    good = "ESR" in " ".join(doc["planner_rules"])
    small = doc["model_routing"]["planner"].endswith("small")
    imp = (0.8 if good else 0.4) - (0.1 if small else 0.0)
    return TuneResult(
        workload_id=w.id,
        split=w.split,
        harness_version=label,
        plan=TuningPlan(actions=[]),
        before=before,
        after=before,
        findings=[],
        tokens=1000,
        rounds=1,
        trace=[],
        score=ScoreReport(improvement=imp, index_penalty=0.0, score=imp, violations=[]),
    )


def test_transfer_runs_holdout_only_per_model_and_renders_markdown():
    gen0 = AgentHarness().model_dump(mode="json")
    best = AgentHarness(planner_rules=["Use ESR order."]).model_dump(mode="json")
    best["model_routing"]["planner"] = "anthropic:big"
    seen = []
    rep = tr.run_transfer(
        workloads(),
        gen0,
        best,
        "h_0007",
        ["anthropic:big", "openai:small"],
        fake_tune,
        on_result=lambda m, label, r: seen.append((m, label, r.workload_id)),
    )
    assert {w for _, _, w in seen} == {"h1", "h2"}  # holdout only
    rows = {r.model: r for r in rep.rows}
    assert abs(rows["anthropic:big"].lift - 0.4) < 1e-9
    assert abs(rows["openai:small"].best.mean_score - 0.7) < 1e-9
    md = tr.to_markdown(rep)
    assert "| `anthropic:big` (evolved with) | 0.40 | 0.80 | +0.40 |" in md
    assert "`openai:small`" in md and "30% → 70%" in md


def test_transfer_skips_failing_model():
    def boom(w, doc, label):
        if doc["model_routing"]["planner"] == "x:bad":
            raise RuntimeError("no key")
        return fake_tune(w, doc, label)

    doc = AgentHarness().model_dump(mode="json")
    rep = tr.run_transfer(workloads(), doc, doc, "h_1", ["x:bad", "a:ok"], boom)
    assert [r.model for r in rep.rows] == ["a:ok"] and "x:bad" in rep.skipped
    assert "skipped" in tr.to_markdown(rep)


def test_callable_models_dedupes_and_filters():
    assert tr.callable_models(["a", "b", "a", ""], lambda m: m != "b") == ["a"]


# ---------------------------------------------------------------- chart

GENS = [
    {
        "gen": 1,
        "verdict": "REJECT",
        "candidate": "h_0001",
        "reason_codes": ["NO_HOLDOUT_GAIN"],
        "parent_metrics": {"train": 0.30, "holdout": 0.28},
        "candidate_metrics": {"train": 0.35, "holdout": 0.29},
        "head_before": "h_0000",
    },
    {
        "gen": 2,
        "verdict": "ACCEPT",
        "candidate": "h_0002",
        "reason_codes": [],
        "parent_metrics": {"train": 0.30, "holdout": 0.28},
        "candidate_metrics": {"train": 0.50, "holdout": 0.47},
    },
    {
        "gen": 3,
        "verdict": "REJECT",
        "candidate": "h_0003",
        "reason_codes": ["PROTECTED_FIELD_CHANGED"],
        "parent_metrics": {"train": 0.50, "holdout": 0.47},
        "candidate_metrics": {},
    },
]


def test_points_from_generations_tracks_head():
    pts = chart.points_from_generations(GENS)
    assert [p.gen for p in pts] == [0, 1, 2, 3]
    assert [p.head_holdout for p in pts] == [0.28, 0.28, 0.47, 0.47]
    assert pts[1].verdict == "REJECT" and pts[1].reason_codes == ["NO_HOLDOUT_GAIN"]
    assert pts[3].cand_holdout is None  # statically rejected: never ran


def test_points_from_versions_fallback():
    docs = [
        {
            "_id": "h_0000",
            "version": 0,
            "status": "accepted",
            "scores": {"train": 0.3, "holdout": 0.2},
        },
        {
            "_id": "h_0001",
            "version": 1,
            "status": "rejected",
            "scores": {"holdout": 0.1},
            "reason_codes": ["CANARY_FAILED"],
        },
    ]
    pts = chart.points_from_versions(docs)
    assert [p.head_holdout for p in pts] == [0.2, 0.2] and pts[1].verdict == "REJECT"


def test_render_chart_writes_png(tmp_path):
    out = chart.render_chart(chart.points_from_generations(GENS), tmp_path / "c.png")
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 10_000


# ---------------------------------------------------------------- watch


class Cursor(list):
    def sort(self, *a, **k):
        return Cursor(sorted(self, key=lambda d: str(d["_id"]), reverse=True))

    def limit(self, n):
        return Cursor(self[:n])


class Coll:
    def __init__(self):
        self.docs = []

    def find(self, *a, **k):
        return Cursor(self.docs)


class DB(dict):
    def __missing__(self, key):
        self[key] = Coll()
        return self[key]

    def watch(self, *a, **k):
        raise NotImplementedError("no change streams on this tier")


def test_format_event_lines():
    gate = {"verdict": "REJECT", "reason_codes": ["X"]}
    assert "REJECT" in watch.format_event("gate_decisions", gate)
    version = {"_id": "h_0002", "status": "accepted"}
    assert "h_0002" in watch.format_event("harness_versions", version)
    assert "gen 3" in watch.format_event("generations", {"gen": 3, "verdict": "ACCEPT"})


def test_watch_falls_back_to_polling_and_prints_only_new_events():
    db = DB()
    db["harness_versions"].docs.append({"_id": "h_0000", "status": "accepted"})  # history
    lines = []

    def sleep(_):
        if not db["gate_decisions"].docs:
            db["gate_decisions"].docs.append({"_id": 1, "verdict": "ACCEPT", "candidate": "h_0001"})
            db["harness_versions"].docs.append({"_id": "h_0001", "status": "accepted"})

    mode = watch.watch(db, lines.append, interval=0, sleep=sleep, max_polls=3)
    assert mode == "poll"
    assert any("polling" in s for s in lines)
    body = [s for s in lines if "polling" not in s and "watching" not in s]
    assert len(body) == 2 and not any("h_0000" in s for s in body)


def test_status_change_is_a_new_event():
    db, seen = DB(), set()
    doc = {"_id": "h_0001", "status": "candidate"}
    db["harness_versions"].docs.append(doc)
    assert len(watch.poll_once(db, seen)) == 1
    doc["status"] = "rejected"
    assert len(watch.poll_once(db, seen)) == 1
    assert watch.poll_once(db, seen) == []


# ---------------------------------------------------------------- replay


class Canned:
    name = "anthropic"

    def __init__(self):
        self.calls = 0

    def call(self, model_id, system, messages, **kw):
        self.calls += 1
        return RawCompletion(text=json.dumps(ESR_PLAN), input_tokens=10, output_tokens=5)


def test_replay_uses_only_the_cache_and_reports_misses(monkeypatch):
    # Hermetic: the planner fallback reads settings; a developer's .env (e.g. a groq: planner)
    # must not change which model the replay looks up in the cache.
    from ratchet import config

    defaults = config.load_settings(dotenv=False).model_copy(
        update={
            "planner_model": config.DEFAULT_PLANNER_MODEL,
            "critic_model": config.DEFAULT_CRITIC_MODEL,
        }
    )
    monkeypatch.setattr("ratchet.agent.loop.get_settings", lambda: defaults)
    monkeypatch.setattr("ratchet.agent.critic.get_settings", lambda: defaults)
    cache, prov = MemoryCache(), Canned()
    env = FakeEnv()
    h = AgentHarness().model_dump(mode="json")

    def tune_fn(w):
        return tune(w, h, env.prepare_scratch(w), api=env, memory=None)

    prev = get_client()
    set_client(LLMClient(cache=cache, providers={"anthropic": prov}))
    try:
        live = tune_fn(sample_workload("t1"))  # records into the cache
        rows = rp.replay(
            [sample_workload("t1"), sample_workload("t2", coll="films")],
            tune_fn,
            recorded={"t1": live.score.score},
            cache=cache,
        )
    finally:
        set_client(prev)
    assert prov.calls == 1  # replay made no provider calls
    t1, t2 = rows
    assert t1.cache_misses == 0 and t1.score == t1.recorded_score == live.score.score
    assert t2.cache_misses == 1 and t2.score == 0.0


def test_cache_only_env_flag(monkeypatch):
    monkeypatch.setenv("RATCHET_LLM_CACHE_ONLY", "1")
    prev = get_client()
    set_client(None)
    try:
        assert get_client().cache_only is True
    finally:
        set_client(prev)
