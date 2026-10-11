import importlib

import pytest

from ratchet import config
from ratchet.contracts import TuningPlan
from ratchet.llm.cache import MemoryCache, cache_key
from ratchet.llm.client import LLMClient, is_retryable
from ratchet.llm.jsonutil import JSONExtractError, extract_json, validate
from ratchet.llm.providers import AnthropicProvider, build_provider, split_model
from ratchet.llm.types import (
    CacheMiss,
    LLMCallError,
    LLMJSONError,
    ModelSpecError,
    ProviderUnavailable,
    RawCompletion,
)

embed_mod = importlib.import_module("ratchet.llm.embed")  # `ratchet.llm.embed` is the function


class HTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class ScriptedProvider:
    """Replies (or raises) from a FIFO list; records each call."""

    name = "anthropic"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def call(self, model_id, system, messages, **kw):
        self.calls.append({"model": model_id, "system": system, "messages": messages, **kw})
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return RawCompletion(text=r, input_tokens=100, output_tokens=20)


def client(*replies, **kw) -> tuple[LLMClient, ScriptedProvider, MemoryCache]:
    prov, cache = ScriptedProvider(*replies), MemoryCache()
    sleeps: list[float] = []
    c = LLMClient(cache=cache, providers={"anthropic": prov}, sleep=sleeps.append, **kw)
    c.sleeps = sleeps  # type: ignore[attr-defined]
    return c, prov, cache


SCHEMA = {
    "type": "object",
    "properties": {"approve": {"type": "boolean"}, "issues": {"type": "array"}},
    "required": ["approve", "issues"],
    "additionalProperties": False,
}
MSGS = [{"role": "user", "content": "review"}]
M = "anthropic:claude-haiku-5-5"


# ---------------------------------------------------------------- JSON reliability


def test_json_repair_retries_with_validation_error_then_succeeds():
    c, prov, _ = client(
        "not json at all", '{"approve": "yes", "issues": []}', '{"approve": true, "issues": []}'
    )
    r = c.complete("critic", M, MSGS, json_schema=SCHEMA)
    assert r.data == {"approve": True, "issues": []} and r.parsed == r.data
    assert r.attempts == 3 and len(prov.calls) == 3
    # the validation error is fed back to the model in the repair turn
    repair = prov.calls[2]["messages"][-1]["content"]
    assert "$.approve: expected boolean" in repair
    assert prov.calls[1]["messages"][-1]["content"].startswith("Your reply was not valid")
    # tokens summed over all attempts
    assert r.input_tokens == 300 and r.output_tokens == 60


def test_json_gives_up_with_typed_error_after_two_repairs():
    c, prov, _ = client("nope", "still nope", '{"approve": 1}')
    with pytest.raises(LLMJSONError) as ei:
        c.complete("critic", M, MSGS, json_schema=SCHEMA)
    assert ei.value.attempts == 3 and len(prov.calls) == 3
    assert ei.value.last_text == '{"approve": 1}'
    assert any("issues" in e for e in ei.value.errors)


def test_pydantic_schema_returns_model_and_strips_fences():
    reply = '```json\n{"actions": [], "summary": "noop"}\n```'
    c, _, _ = client(reply)
    r = c.complete("planner", M, MSGS, json_schema=TuningPlan)
    assert isinstance(r.parsed, TuningPlan) and r.parsed.summary == "noop"
    assert r.data == {"actions": [], "summary": "noop"}


def test_system_messages_are_folded_and_schema_is_in_system_prompt():
    c, prov, _ = client('{"approve": true, "issues": []}')
    c.complete("critic", M, [{"role": "system", "content": "SYS"}, *MSGS], json_schema=SCHEMA)
    call = prov.calls[0]
    assert call["system"].startswith("SYS") and "JSON Schema" in call["system"]
    assert all(m["role"] != "system" for m in call["messages"])
    assert call["json_mode"] is True


def test_extract_json_variants():
    assert extract_json('Here you go: {"a": {"b": "}"}} thanks') == {"a": {"b": "}"}}
    assert extract_json("[1, 2]") == [1, 2]
    with pytest.raises(JSONExtractError):
        extract_json("no json here")


def test_validator_handles_pydantic_refs_tuples_and_optionals():
    schema = TuningPlan.model_json_schema()
    good = {
        "actions": [
            {"type": "create_index", "collection": "m", "index": {"keys": [["a", 1], ["b", -1]]}}
        ]
    }
    assert validate(good, schema) == []
    bad = {"actions": [{"type": "collMod", "index": {"keys": [["a"]]}}]}
    errs = validate(bad, schema)
    assert any("collMod" in e for e in errs)
    assert validate({"summary": "x"}, schema) == ["$: missing required field 'actions'"]


# ---------------------------------------------------------------- cache


def test_cache_miss_then_hit_skips_provider():
    c, prov, cache = client('{"approve": true, "issues": []}')
    a = c.complete("critic", M, MSGS, json_schema=SCHEMA)
    b = c.complete("critic", M, MSGS, json_schema=SCHEMA)
    assert len(prov.calls) == 1
    assert (a.cached, b.cached) == (False, True)
    assert a.data == b.data and b.cost_usd == 0.0 and a.cost_usd > 0
    assert cache.hits == 1 and cache.misses == 1


def test_cache_key_depends_on_model_messages_and_params():
    base = {"system": None, "messages": MSGS, "max_tokens": 10}
    k = cache_key("complete", M, base)
    assert k == cache_key("complete", M, dict(base))
    assert k != cache_key("complete", "anthropic:other", base)
    assert k != cache_key("complete", M, {**base, "max_tokens": 11})
    assert k != cache_key("complete", M, {**base, "messages": [{"role": "user", "content": "x"}]})
    assert len(k) == 64


def test_cache_only_mode_raises_on_miss():
    c, prov, _ = client(cache_only=True)
    with pytest.raises(CacheMiss):
        c.complete("planner", M, MSGS)
    assert prov.calls == []


def test_broken_cache_is_a_miss_not_a_crash():
    from ratchet.llm.cache import MongoCache

    class Boom:
        def find_one(self, *a, **k):
            raise RuntimeError("db down")

        def replace_one(self, *a, **k):
            raise RuntimeError("db down")

    prov = ScriptedProvider("hello")
    c = LLMClient(cache=MongoCache(Boom()), providers={"anthropic": prov})
    assert c.complete("x", M, MSGS).text == "hello"


# ---------------------------------------------------------------- retries / timeouts


def test_backoff_on_429_and_5xx_then_success():
    c, prov, _ = client(HTTPError(429), HTTPError(529), TimeoutError(), "ok")
    assert c.complete("planner", M, MSGS).text == "ok"
    assert c.sleeps == [1.0, 2.0, 4.0]  # exponential


def test_non_retryable_error_fails_fast():
    c, prov, _ = client(HTTPError(400), "never")
    with pytest.raises(LLMCallError):
        c.complete("planner", M, MSGS)
    assert len(prov.calls) == 1 and c.sleeps == []


def test_retries_exhausted_raise_call_error():
    c, _, _ = client(*[HTTPError(503)] * 3, max_attempts=3)
    with pytest.raises(LLMCallError):
        c.complete("planner", M, MSGS)
    assert c.sleeps == [1.0, 2.0]


def test_is_retryable_classification():
    class APITimeoutError(Exception):
        pass

    assert is_retryable(HTTPError(429)) and is_retryable(HTTPError(500))
    assert is_retryable(APITimeoutError()) and is_retryable(ConnectionError())
    assert not is_retryable(HTTPError(401)) and not is_retryable(ValueError())


# ---------------------------------------------------------------- providers / keys


def test_model_spec_parsing():
    assert split_model("anthropic:claude-haiku-5-5") == ("anthropic", "claude-haiku-5-5")
    for bad in ("claude", "anthropic:", "mistral:x"):
        with pytest.raises(ModelSpecError):
            split_model(bad)


def test_missing_key_is_skippable(monkeypatch):
    monkeypatch.setattr(config, "get_settings", lambda: config.Settings())
    with pytest.raises(ProviderUnavailable):
        build_provider("anthropic")
    with pytest.raises(ProviderUnavailable):
        build_provider("openai")
    with pytest.raises(ProviderUnavailable):
        AnthropicProvider(None)
    c = LLMClient(cache=MemoryCache())
    assert not c.available("anthropic:claude-haiku-5-5")
    assert not c.available("google:gemini-x")
    assert not c.available("bogus")
    with pytest.raises(ProviderUnavailable):
        c.complete("planner", M, MSGS)


def test_available_when_key_present(monkeypatch):
    monkeypatch.setattr(config, "get_settings", lambda: config.Settings(anthropic_api_key="k"))
    assert LLMClient(cache=MemoryCache()).available(M)


# ---------------------------------------------------------------- embeddings


def test_local_embed_is_deterministic_and_similarity_aware(monkeypatch):
    monkeypatch.delenv("RATCHET_EMBED_MODE", raising=False)
    monkeypatch.setattr(embed_mod, "_voyage_key", lambda: None)
    assert embed_mod.embedding_mode() == "local"
    a, b, c = embed_mod.embed(
        [
            "movies | filter {genres: EQ, year: RANGE}",
            "movies | filter {genres: EQ, year: RANGE} | sort {rating: DESC}",
            "orders | pipeline $lookup > $unwind",
        ]
    )

    def dot(x, y):
        return sum(i * j for i, j in zip(x, y, strict=True))

    assert dot(a, b) > dot(a, c)
    assert embed_mod.embed([]) == []


def test_voyage_embed_is_cached_per_text(monkeypatch):
    calls: list[list[str]] = []

    def fake(texts, model, input_type):
        calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]

    monkeypatch.setattr(embed_mod, "_voyage_key", lambda: "k")
    monkeypatch.setattr(embed_mod, "_voyage_embed", fake)
    cache = MemoryCache()
    assert embed_mod.embedding_mode() == "voyage"
    assert embed_mod.embed(["ab", "abc"], cache=cache) == [[2.0, 1.0], [3.0, 1.0]]
    assert embed_mod.embed(["abc", "abcd"], cache=cache) == [[3.0, 1.0], [4.0, 1.0]]
    assert calls == [["ab", "abc"], ["abcd"]]


def test_atlas_mode_flag(monkeypatch):
    monkeypatch.setenv("RATCHET_EMBED_MODE", "atlas")
    assert embed_mod.embedding_mode() == "atlas"
