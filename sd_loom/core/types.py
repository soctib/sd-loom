from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 — Pydantic needs this at runtime
from typing import TYPE_CHECKING

from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    from PIL import Image


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
    clip_skip: int
    model_hash: str
    vae: str
    loras: list[tuple[str, float]]
    vram: str
    rng: str
    output_dir: str | Path
    input_image: str

    @field_validator("prompt", mode="before")
    @classmethod
    def _coerce_prompt(cls, v: object) -> object:
        """Allow plain strings: ``--set prompt='a cat'`` → ``Prompt(positive='a cat')``."""
        if isinstance(v, str):
            return Prompt(positive=v)
        return v


@dataclass
class LoomData:
    image: Image.Image | None = None
    seed: int = 0
    elapsed_seconds: float = 0.0
    workflow: str = ""
    text: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
