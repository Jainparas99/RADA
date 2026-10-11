"""Environment loading. Secrets come from `.env` (gitignored) and are never printed."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field, field_validator

_NS_RE = re.compile(r"^[a-z0-9_]{1,24}$")

DEFAULT_PLANNER_MODEL = "anthropic:claude-haiku-5-5"
DEFAULT_CRITIC_MODEL = "anthropic:claude-sonnet-5-5"
DEFAULT_PROPOSER_MODEL = "anthropic:claude-sonnet-5-5"


class ConfigError(RuntimeError):
    """A required setting is missing or malformed."""


class Settings(BaseModel):
    mongodb_uri_rw: str | None = Field(default=None, repr=False)
    mongodb_uri_ro: str | None = Field(default=None, repr=False)
    ns: str = "a"
    anthropic_api_key: str | None = Field(default=None, repr=False)
    openai_api_key: str | None = Field(default=None, repr=False)
    google_api_key: str | None = Field(default=None, repr=False)
    voyage_api_key: str | None = Field(default=None, repr=False)
    planner_model: str = DEFAULT_PLANNER_MODEL
    critic_model: str = DEFAULT_CRITIC_MODEL
    proposer_model: str = DEFAULT_PROPOSER_MODEL
    model_allowlist: list[str] = []  # extra models allowed in model_routing (gen-0 pin)
    embed_dims: int = 1024  # vector index dimensions (Voyage default)

    @field_validator("ns")
    @classmethod
    def _valid_ns(cls, v: str) -> str:
        if not _NS_RE.match(v):
            raise ValueError("RATCHET_NS must match [a-z0-9_]{1,24} (e.g. a, b, c, main)")
        return v

    @property
    def db_name(self) -> str:
        return f"ratchet_{self.ns}"

    @property
    def scratch_db_name(self) -> str:
        return f"ratchet_scratch_{self.ns}"

    def require_rw_uri(self) -> str:
        if not self.mongodb_uri_rw:
            raise ConfigError("MONGODB_URI_RW is not set (copy .env.example to .env)")
        return self.mongodb_uri_rw

    def require_ro_uri(self) -> str:
        if not self.mongodb_uri_ro:
            raise ConfigError("MONGODB_URI_RO is not set (copy .env.example to .env)")
        return self.mongodb_uri_ro


def _env(name: str) -> str | None:
    """Value of an env var, with inline `# comments` and blanks treated as unset."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.split(" #", 1)[0].strip()
    if value.startswith("#"):
        return None
    return value or None


def load_settings(dotenv: bool = True) -> Settings:
    if dotenv:
        load_dotenv(find_dotenv(usecwd=True), override=False)
    try:
        return Settings(
            mongodb_uri_rw=_env("MONGODB_URI_RW"),
            mongodb_uri_ro=_env("MONGODB_URI_RO"),
            ns=_env("RATCHET_NS") or "a",
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            openai_api_key=_env("OPENAI_API_KEY"),
            google_api_key=_env("GOOGLE_API_KEY"),
            voyage_api_key=_env("VOYAGE_API_KEY"),
            planner_model=_env("RATCHET_PLANNER_MODEL") or DEFAULT_PLANNER_MODEL,
            critic_model=_env("RATCHET_CRITIC_MODEL") or DEFAULT_CRITIC_MODEL,
            proposer_model=_env("RATCHET_PROPOSER_MODEL") or DEFAULT_PROPOSER_MODEL,
            model_allowlist=[
                m.strip() for m in (_env("RATCHET_MODEL_ALLOWLIST") or "").split(",") if m.strip()
            ],
            embed_dims=int(_env("RATCHET_EMBED_DIMS") or 1024),
        )
    except ValueError as e:
        raise ConfigError(str(e)) from e


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    """`config/` at the repo root, or `RATCHET_CONFIG_DIR` if set."""
    override = os.environ.get("RATCHET_CONFIG_DIR")
    return Path(override) if override else repo_root() / "config"


def load_yaml(name: str) -> dict[str, Any]:
    with (config_dir() / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
