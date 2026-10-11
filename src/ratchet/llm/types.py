"""Response and error types for the LLM client (SPEC §10, §11)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class LLMError(RuntimeError):
    """Base class for every error raised by `ratchet.llm`."""


class ProviderUnavailable(LLMError):
    """No API key (or SDK) for this provider. Callers may skip the model cleanly."""


class ModelSpecError(LLMError):
    """A model id that is not `provider:model`, or names an unknown provider."""


class LLMCallError(LLMError):
    """The provider call failed after all retries (or with a non-retryable error)."""


class CacheMiss(LLMError):
    """`cache_only` mode (replay) and the request is not in `llm_cache`."""


class LLMJSONError(LLMError):
    """The model did not return JSON matching the schema, even after the repair retries."""

    def __init__(self, message: str, errors: list[str], last_text: str, attempts: int) -> None:
        super().__init__(message)
        self.errors = errors
        self.last_text = last_text
        self.attempts = attempts


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """What `complete()` returns. Field names match `tests/fakes.FakeLLMResponse`.

    `data` is the parsed JSON (when the request had a schema or the reply was JSON); `parsed`
    is the validated pydantic instance when the schema was given as a model class, else `data`.
    Token counts and cost are summed over every attempt (JSON repair retries included).
    """

    model_config = {"arbitrary_types_allowed": True}

    text: str
    data: Any = None
    parsed: Any = None
    role: str = ""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    attempts: int = 1
    tool_calls: list[dict] = []
    stop_reason: str | None = None

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def usage(self) -> Usage:
        return Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=self.cost_usd,
        )


class RawCompletion(BaseModel):
    """One provider round-trip, exactly what is stored in `llm_cache`."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    tool_calls: list[dict] = []
