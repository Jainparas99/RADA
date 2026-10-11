"""Provider adapters behind `complete()`. Model ids are `provider:model`.

Anthropic is required; OpenAI and Google are optional extras (`uv sync --extra openai|google`).
A provider without a key (or SDK) raises `ProviderUnavailable`, which callers treat as "skip".
SDK retries are disabled: `client.py` owns timeouts and backoff so behaviour is uniform.
"""

from __future__ import annotations

from typing import Any, Protocol

from ratchet.llm.types import ModelSpecError, ProviderUnavailable, RawCompletion

PROVIDERS = ("anthropic", "openai", "google")

# USD per 1M tokens (input, output). Logged for cost visibility only; unknown models cost 0.
PRICES: dict[str, tuple[float, float]] = {
    "anthropic:claude-haiku-5-5": (0.10, 0.50),
    "anthropic:claude-sonnet-5-5": (2.00, 10.00),
    "anthropic:claude-opus-5-5": (4.00, 20.00),
    "anthropic:claude-fable-5-1": (10.00, 50.00),
}


def split_model(model: str) -> tuple[str, str]:
    provider, sep, model_id = model.partition(":")
    if not sep or not model_id:
        raise ModelSpecError(f"model must look like 'provider:model', got {model!r}")
    if provider not in PROVIDERS:
        raise ModelSpecError(f"unknown provider {provider!r} (known: {', '.join(PROVIDERS)})")
    return provider, model_id


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


class Provider(Protocol):
    name: str

    def call(
        self,
        model_id: str,
        system: str | None,
        messages: list[dict],
        *,
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float | None,
        timeout: float,
        json_mode: bool,
    ) -> RawCompletion: ...


def _merge_turns(messages: list[dict]) -> list[dict]:
    """Collapse consecutive same-role turns (providers want user/assistant alternation)."""
    out: list[dict] = []
    for m in messages:
        content = m["content"] if isinstance(m["content"], str) else str(m["content"])
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + content}
        else:
            out.append({"role": m["role"], "content": content})
    return out


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None, effort: str | None = None) -> None:
        if not api_key:
            raise ProviderUnavailable("ANTHROPIC_API_KEY is not set")
        import anthropic

        self._sdk = anthropic
        self._key = api_key
        self.effort = effort

    def call(
        self,
        model_id: str,
        system: str | None,
        messages: list[dict],
        *,
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float | None,  # not accepted by current Claude models; determinism = cache
        timeout: float,
        json_mode: bool,
    ) -> RawCompletion:
        client = self._sdk.Anthropic(api_key=self._key, max_retries=0, timeout=timeout)
        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": _merge_turns(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        resp = client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        calls = [
            {"id": b.id, "name": b.name, "input": b.input}
            for b in resp.content
            if getattr(b, "type", "") == "tool_use"
        ]
        return RawCompletion(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            stop_reason=resp.stop_reason,
            tool_calls=calls,
        )


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise ProviderUnavailable("OPENAI_API_KEY is not set")
        try:
            import openai
        except ImportError as e:
            raise ProviderUnavailable("openai SDK not installed (uv sync --extra openai)") from e
        self._sdk = openai
        self._key = api_key

    def call(
        self,
        model_id: str,
        system: str | None,
        messages: list[dict],
        *,
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float | None,
        timeout: float,
        json_mode: bool,
    ) -> RawCompletion:
        client = self._sdk.OpenAI(api_key=self._key, max_retries=0, timeout=timeout)
        msgs = ([{"role": "system", "content": system}] if system else []) + _merge_turns(messages)
        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": msgs,
            "max_completion_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = resp.usage
        return RawCompletion(
            text=choice.message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            stop_reason=choice.finish_reason,
        )


class GoogleProvider:
    name = "google"

    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise ProviderUnavailable("GOOGLE_API_KEY is not set")
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ProviderUnavailable("google-genai not installed (uv sync --extra google)") from e
        self._genai, self._types = genai, types
        self._key = api_key

    def call(
        self,
        model_id: str,
        system: str | None,
        messages: list[dict],
        *,
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float | None,
        timeout: float,
        json_mode: bool,
    ) -> RawCompletion:
        t = self._types
        client = self._genai.Client(
            api_key=self._key, http_options=t.HttpOptions(timeout=int(timeout * 1000))
        )
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in _merge_turns(messages)
        ]
        cfg: dict[str, Any] = {"max_output_tokens": max_tokens}
        if system:
            cfg["system_instruction"] = system
        if temperature is not None:
            cfg["temperature"] = temperature
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        resp = client.models.generate_content(
            model=model_id, contents=contents, config=t.GenerateContentConfig(**cfg)
        )
        um = getattr(resp, "usage_metadata", None)
        return RawCompletion(
            text=resp.text or "",
            input_tokens=getattr(um, "prompt_token_count", 0) or 0,
            output_tokens=getattr(um, "candidates_token_count", 0) or 0,
        )


def build_provider(name: str) -> Provider:
    """Construct a provider from settings. Raises ProviderUnavailable when it can't run."""
    from ratchet.config import get_settings

    s = get_settings()
    if name == "anthropic":
        return AnthropicProvider(s.anthropic_api_key)
    if name == "openai":
        return OpenAIProvider(s.openai_api_key)
    if name == "google":
        return GoogleProvider(s.google_api_key)
    raise ModelSpecError(f"unknown provider {name!r}")
