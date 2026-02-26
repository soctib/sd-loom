from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 — Pydantic needs this at runtime

from pydantic import BaseModel, field_validator


class Prompt(BaseModel):
    positive: str
    negative: str = ""


class LoomSpec(BaseModel):
    prompt: Prompt
    model: str
    width: int
    height: int
    steps: int
    cfg_scale: float
    seed: int
    count: int
    scheduler: str
    vram: str
    output_dir: str | Path

    @field_validator("prompt", mode="before")
    @classmethod
    def _coerce_prompt(cls, v: object) -> object:
        """Allow plain strings: ``--set prompt='a cat'`` → ``Prompt(positive='a cat')``."""
        if isinstance(v, str):
            return Prompt(positive=v)
        return v


@dataclass
class GenerationResult:
    image_path: Path
    seed: int
    elapsed_seconds: float
    workflow: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
