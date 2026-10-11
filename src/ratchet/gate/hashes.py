"""Pinned source hashes for the scorer and the gate (CLAUDE.md invariant 4).

A harness lineage records these under `protected` when it is created. If the scorer or gate
source differs from what the head pinned, nothing is promoted until a human re-pins
(`ratchet pin`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ratchet.config import config_dir
from ratchet.harness.models import SourceRef

SCORER_VERSION = "1"
GATE_VERSION = "1"

_PKG = Path(__file__).resolve().parents[1]  # src/ratchet


def scorer_files() -> list[Path]:
    return [_PKG / "scoring.py", config_dir() / "scoring.yaml"]


def gate_files() -> list[Path]:
    gate_src = sorted((_PKG / "gate").glob("*.py"))
    return [*gate_src, config_dir() / "gate.yaml", config_dir() / "bounds.yaml"]


def hash_files(paths: list[Path]) -> str:
    """SHA-256 over file names and contents; a missing file hashes as absent, not as an error."""
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode() + b"\0")
        h.update(p.read_bytes() if p.is_file() else b"<missing>")
        h.update(b"\0")
    return "sha256:" + h.hexdigest()


def scorer_hash() -> str:
    return hash_files(scorer_files())


def gate_hash() -> str:
    return hash_files(gate_files())


def scorer_ref() -> SourceRef:
    return SourceRef(version=SCORER_VERSION, source_hash=scorer_hash())


def gate_ref() -> SourceRef:
    return SourceRef(version=GATE_VERSION, source_hash=gate_hash())
