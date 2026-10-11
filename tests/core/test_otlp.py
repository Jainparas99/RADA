import http.server
import json
import threading

import pytest
from typer.testing import CliRunner

from ratchet import cli_core, otlp
from ratchet.cli import app
from ratchet.harness import runner
from ratchet.harness.init_db import init_database
from tests.fakes import FakeBackend


@pytest.fixture
def results(spec):
    rs = runner.evaluate(FakeBackend(), spec, "h_0000", "train")
    llm_step = {
        "step": "llm",
        "role": "planner",
        "model": "groq:openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "plan please"}],
        "response": '{"actions": []}',
        "tokens": 1234,
        "cached": False,
        "ms": 250,
    }
    return [r.model_copy(update={"trace": [r.trace[0], llm_step, *r.trace[1:]]}) for r in rs]


def spans_of(payload):
    return payload["resourceSpans"][0]["scopeSpans"][0]["spans"]


def attrs(span):
    return {a["key"]: next(iter(a["value"].values())) for a in span["attributes"]}


def test_payload_is_valid_otlp_json(results):
    payload = otlp.to_otlp(results, now_ns=1_000)
    resource = payload["resourceSpans"][0]["resource"]["attributes"]
    assert resource == [{"key": "service.name", "value": {"stringValue": "ratchet"}}]
    spans = spans_of(payload)
    assert len(spans) == sum(1 + len(r.trace) for r in results)
    for s in spans:
        assert len(s["traceId"]) == 32 and len(s["spanId"]) == 16
        int(s["traceId"], 16), int(s["spanId"], 16)
        assert int(s["endTimeUnixNano"]) > int(s["startTimeUnixNano"])
    json.dumps(payload)  # plain JSON all the way down


def test_one_trace_per_run_with_children_under_the_root(results):
    spans = spans_of(otlp.to_otlp(results[:1], now_ns=0))
    root, children = spans[0], spans[1:]
    assert root["name"] == "tune fake_train_01" and "parentSpanId" not in root
    assert {c["traceId"] for c in children} == {root["traceId"]}
    assert {c["parentSpanId"] for c in children} == {root["spanId"]}
    a = attrs(root)
    assert a["ratchet.harness_version"] == "h_0000" and a["ratchet.split"] == "train"
    assert a["ratchet.score"] == pytest.approx(results[0].score.score)
    assert len({s["traceId"] for s in spans_of(otlp.to_otlp(results))}) == len(results)


def test_llm_calls_use_genai_attribute_names(results):
    llm = next(s for s in spans_of(otlp.to_otlp(results[:1])) if s["name"].startswith("chat "))
    a = attrs(llm)
    assert llm["kind"] == 3
    assert a["gen_ai.system"] == "groq" and a["gen_ai.request.model"] == "openai/gpt-oss-120b"
    assert a["gen_ai.usage.total_tokens"] == "1234"
    # prompt and completion under each naming scheme Raindrop Workshop reads
    assert "plan please" in a["gen_ai.input.messages"] and "plan please" in a["ai.prompt.messages"]
    assert a["gen_ai.prompt.0.role"] == "user" and a["gen_ai.prompt.0.content"] == "plan please"
    assert a["ai.response.text"] == a["gen_ai.completion.0.content"] == '{"actions": []}'
    assert json.loads(a["gen_ai.output.messages"])[0]["role"] == "assistant"
    assert int(llm["endTimeUnixNano"]) - int(llm["startTimeUnixNano"]) == 250_000_000


def test_violations_mark_the_run_as_an_error(results):
    bad = results[0].model_copy(
        update={"score": results[0].score.model_copy(update={"violations": ["RESULT_MISMATCH"]})}
    )
    root = spans_of(otlp.to_otlp([bad]))[0]
    assert root["status"] == {"code": 2, "message": "RESULT_MISMATCH"}
    assert spans_of(otlp.to_otlp(results[:1]))[0]["status"]["code"] == 1


@pytest.fixture
def collector():
    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, self.headers["Content-Type"], json.loads(body)))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1/traces", received
    server.shutdown()


def test_cli_posts_stored_traces_to_the_endpoint(db, spec, collector, monkeypatch, tmp_path):
    url, received = collector
    init_database(db, vector=False)
    runner.evaluate(FakeBackend(), spec, "h_0000", "train", db=db)
    monkeypatch.setattr("ratchet.db.get_db", lambda ns=None: db)
    out = tmp_path / "traces.json"
    res = CliRunner().invoke(app, ["export-traces", "--endpoint", url, "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "3 tuning runs" in res.output and "HTTP 200" in res.output
    path, ctype, body = received[0]
    assert path == "/v1/traces" and ctype == "application/json"
    assert body == json.loads(out.read_text(encoding="utf-8"))


def test_cli_explains_when_workshop_is_not_running(db, spec, monkeypatch):
    init_database(db, vector=False)
    runner.evaluate(FakeBackend(), spec, "h_0000", "train", db=db)
    monkeypatch.setattr("ratchet.db.get_db", lambda ns=None: db)
    res = CliRunner().invoke(app, ["export-traces", "--endpoint", "http://127.0.0.1:9/v1/traces"])
    assert res.exit_code == 1 and "raindrop workshop" in res.output
    assert cli_core.app is not None
