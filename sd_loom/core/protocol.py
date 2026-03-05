from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from sd_loom.core.types import Prompt


@runtime_checkable
class SpecProtocol(Protocol):
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
