"""Embeddings for lesson / failure retrieval (SPEC §8). Cached in `llm_cache`.

Modes (`embedding_mode()`):
- `atlas`  — Atlas automated Voyage embeddings (`RATCHET_EMBED_MODE=atlas`): the vector index
             embeds text server-side, so callers store/query raw text. `embed()` is still usable.
- `voyage` — the `voyageai` SDK with `VOYAGE_API_KEY` (`uv sync --extra voyage`).
- `local`  — deterministic hashed bag-of-words. Offline fallback for tests and dev only.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Literal

from ratchet.llm import cache as cache_mod
from ratchet.llm.types import ProviderUnavailable

DEFAULT_VOYAGE_MODEL = "voyage-3.5"
VOYAGE_DIMS = (256, 512, 1024, 2048)  # voyage-3.5 output_dimension options; default 1024
Mode = Literal["atlas", "voyage", "local"]


def _voyage_key() -> str | None:
    from ratchet.config import get_settings

    return get_settings().voyage_api_key


def embedding_mode() -> Mode:
    if (os.environ.get("RATCHET_EMBED_MODE") or "").strip().lower() == "atlas":
        return "atlas"
    return "voyage" if _voyage_key() else "local"


def embed_dims() -> int:
    """Vector width; must match Track A's vector indexes (RATCHET_EMBED_DIMS, default 1024)."""
    from ratchet.config import get_settings

    return int(get_settings().embed_dims)


def voyage_model() -> str:
    return (os.environ.get("RATCHET_EMBED_MODEL") or "").strip() or DEFAULT_VOYAGE_MODEL


def local_embed(texts: list[str], dim: int | None = None) -> list[list[float]]:
    """Hashed bag-of-words, L2-normalised: similar shapes get similar vectors."""
    dim = dim or embed_dims()
    out: list[list[float]] = []
    for t in texts:
        v = [0.0] * dim
        for tok in re.findall(r"[a-z0-9_.$]+", t.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


def _voyage_embed(texts: list[str], model: str, input_type: str, dims: int) -> list[list[float]]:
    key = _voyage_key()
    if not key:
        raise ProviderUnavailable("VOYAGE_API_KEY is not set")
    try:
        import voyageai
    except ImportError as e:
        raise ProviderUnavailable("voyageai not installed (uv sync --extra voyage)") from e
    client = voyageai.Client(api_key=key, max_retries=3, timeout=60)
    res = client.embed(texts, model=model, input_type=input_type, output_dimension=dims)
    return [list(v) for v in res.embeddings]


def embed(
    texts: list[str],
    input_type: Literal["document", "query"] = "document",
    cache: cache_mod.Cache | None = None,
    mode: Mode | None = None,
) -> list[list[float]]:
    """Vectors for `texts`, in order. Cached per (model, input_type, text)."""
    if not texts:
        return []
    mode = mode or embedding_mode()
    if mode == "local" or (mode == "atlas" and not _voyage_key()):
        return local_embed(texts)
    model = voyage_model()
    dims = embed_dims()
    if cache is None:
        from ratchet.llm.client import get_client

        cache = get_client().cache
    keys = [
        cache_mod.cache_key(
            "embed", f"voyage:{model}", {"input_type": input_type, "dims": dims, "text": t}
        )
        for t in texts
    ]
    found: dict[int, list[float]] = {}
    for i, k in enumerate(keys):
        hit = cache.get(k)
        if hit is not None:
            found[i] = hit["vector"]
    missing = [i for i in range(len(texts)) if i not in found]
    if missing:
        vecs = _voyage_embed([texts[i] for i in missing], model, input_type, dims)
        if any(len(v) != dims for v in vecs):
            raise ValueError(f"{model} returned vectors that are not {dims}-dim")
        for i, v in zip(missing, vecs, strict=True):
            found[i] = v
            cache.put(keys[i], f"voyage:{model}", "embed", {"vector": v})
    return [found[i] for i in range(len(texts))]
