from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from sd_loom.core.types import GenerationResult


@runtime_checkable
class PromptSpec(Protocol):
    prompt: str
    negative_prompt: str
    model: str
    width: int
    height: int
    steps: int
    cfg_scale: float
    seed: int
    scheduler: str
    output_dir: str | Path


@runtime_checkable
class Workflow(Protocol):
    def run(self, spec: PromptSpec) -> GenerationResult: ...
