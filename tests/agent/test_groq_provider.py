"""groq:<model> runs through the OpenAI-compatible adapter with Groq's base URL (no network)."""

from types import SimpleNamespace

import pytest

from ratchet.llm import providers
from ratchet.llm.types import ProviderUnavailable


class _FakeOpenAI:
    calls: list[dict] = []

    def __init__(self, **kwargs):
        self.init = kwargs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        _FakeOpenAI.calls.append({"init": self.init, "create": kwargs})
        msg = SimpleNamespace(content='{"actions": []}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )


@pytest.fixture
def fake_sdk(monkeypatch):
    import openai

    _FakeOpenAI.calls = []
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    return _FakeOpenAI


def _settings(monkeypatch, **kw):
    from ratchet import config

    s = config.Settings(**kw)
    monkeypatch.setattr(config, "get_settings", lambda: s)


def test_groq_is_a_known_provider():
    assert providers.split_model("groq:openai/gpt-oss-120b") == ("groq", "openai/gpt-oss-120b")


def test_groq_uses_base_url_and_key(monkeypatch, fake_sdk):
    _settings(monkeypatch, groq_api_key="gsk_test")
    p = providers.build_provider("groq")
    assert p.name == "groq"
    out = p.call(
        "openai/gpt-oss-120b",
        "sys",
        [{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=50,
        temperature=0.0,
        timeout=5.0,
        json_mode=True,
    )
    call = fake_sdk.calls[0]
    assert call["init"]["base_url"] == providers.GROQ_BASE_URL
    assert call["init"]["api_key"] == "gsk_test"
    assert call["create"]["model"] == "openai/gpt-oss-120b"
    assert call["create"]["response_format"] == {"type": "json_object"}
    assert out.text == '{"actions": []}' and out.input_tokens == 11 and out.output_tokens == 7


def test_openai_keeps_default_endpoint(monkeypatch, fake_sdk):
    _settings(monkeypatch, openai_api_key="sk_test")
    p = providers.build_provider("openai")
    p.call(
        "gpt-x",
        None,
        [{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=5,
        temperature=None,
        timeout=5.0,
        json_mode=False,
    )
    assert p.name == "openai" and fake_sdk.calls[0]["init"]["base_url"] is None


def test_missing_groq_key_is_skipped(monkeypatch):
    _settings(monkeypatch)
    with pytest.raises(ProviderUnavailable, match="GROQ_API_KEY"):
        providers.build_provider("groq")
