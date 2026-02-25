from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 — Pydantic needs this at runtime

from pydantic import BaseModel


class Prompt(BaseModel):
    prompt: str
    negative_prompt: str
    model: str
    width: int
    height: int
    steps: int
    cfg_scale: float
    seed: int
    scheduler: str
    vram: str
    output_dir: str | Path


@dataclass
class GenerationResult:
    image_path: Path
    seed: int
    elapsed_seconds: float
    workflow: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
