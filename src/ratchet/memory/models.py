"""Lesson documents in the `memories` collection (SPEC §8)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LessonStatus = Literal["active", "invalidated", "merged", "evicted"]
LessonSource = Literal["train_improved", "train_violation"]


class Lesson(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="_id")
    text: str
    shape: str  # query shape the lesson was learned on (values stripped)
    embedding: list[float] = Field(default_factory=list, repr=False)  # of `shape`
    text_embedding: list[float] = Field(default_factory=list, repr=False)
    status: LessonStatus = "active"
    source: LessonSource
    source_workload: str
    harness_version: str = ""
    support: int = 1  # how many train results taught (or re-taught) this lesson
    uses: int = 0
    helpful: int = 0
    harmful: int = 0
    merged_into: str | None = None
    similarity: float | None = None  # set on recall only, never stored
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_doc(self) -> dict:
        return self.model_dump(mode="python", by_alias=True, exclude={"similarity"})
