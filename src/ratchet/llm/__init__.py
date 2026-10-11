"""Track C LLM layer: `complete()` (cached, retried, JSON-validated) and `embed()`."""

from ratchet.llm.client import LLMClient, available, complete, get_client, set_client
from ratchet.llm.embed import embed, embedding_mode
from ratchet.llm.types import (
    CacheMiss,
    LLMCallError,
    LLMError,
    LLMJSONError,
    LLMResponse,
    ModelSpecError,
    ProviderUnavailable,
)

__all__ = [
    "CacheMiss",
    "LLMCallError",
    "LLMClient",
    "LLMError",
    "LLMJSONError",
    "LLMResponse",
    "ModelSpecError",
    "ProviderUnavailable",
    "available",
    "complete",
    "embed",
    "embedding_mode",
    "get_client",
    "set_client",
]
