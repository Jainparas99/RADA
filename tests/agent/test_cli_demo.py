import json

import pytest
from typer.testing import CliRunner

from ratchet import config
from ratchet.cli import app
from ratchet.llm.cache import MemoryCache
from ratchet.llm.client import LLMClient, set_client
from ratchet.llm.types import RawCompletion

ESR_PLAN = {
    "actions": [
        {
            "type": "create_index",
            "collection": "movies",
            "index": {"keys": [["genres", 1], ["rating", -1], ["year", 1]]},
            "rationale": "equality, sort, range",
        }
    ],
    "summary": "ESR index for q1",
}


class Canned:
    name = "anthropic"

    def __init__(self, reply: dict):
        self.reply = json.dumps(reply)
        self.calls = 0

    def call(self, model_id, system, messages, **kw):
        self.calls += 1
        return RawCompletion(text=self.reply, input_tokens=500, output_tokens=80)


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(ns="c"))
    import ratchet.cli_demo as cli_demo

    monkeypatch.setattr(cli_demo, "get_settings", lambda: config.Settings(ns="c"))
    prov = Canned(ESR_PLAN)
    set_client(LLMClient(cache=MemoryCache(), providers={"anthropic": prov}))
    yield prov
    set_client(None)


def test_tune_fake_prints_findings_plan_and_before_after(offline):
    res = CliRunner().invoke(app, ["tune", "fake_w01", "--fake"])
    assert res.exit_code == 0, res.output
    out = res.output
    assert "COLLSCAN_LARGE" in out and "create_index" in out
    assert "Before -> after" in out and "COLLSCAN" in out and "IXSCAN" in out
    assert "violations none" in out
    assert offline.calls == 1


def test_tune_json_output(offline):
    res = CliRunner().invoke(app, ["tune", "fake_w01", "--fake", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output[res.output.index("{") :])
    assert data["workload_id"] == "fake_w01" and data["score"]["score"] > 0.5


def test_tune_without_key_exits_cleanly(monkeypatch):
    import ratchet.cli_demo as cli_demo

    monkeypatch.setattr(cli_demo, "get_settings", lambda: config.Settings(ns="c"))
    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(ns="c"))
    set_client(LLMClient(cache=MemoryCache()))
    try:
        res = CliRunner().invoke(app, ["tune", "fake_w01", "--fake"])
    finally:
        set_client(None)
    assert res.exit_code == 1 and "not callable" in res.output
