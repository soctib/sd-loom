from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from sd_loom.core.types import GenerationResult, Prompt


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
    vae: str
    loras: list[tuple[str, float]]
    vram: str
    output_dir: str | Path


@runtime_checkable
class Workflow(Protocol):
    def run(self, spec: SpecProtocol) -> list[GenerationResult]: ...
