from __future__ import annotations

from pathlib import Path  # noqa: TC003 — Pydantic needs this at runtime

from sd_loom.core.types import Prompt


class DefaultPrompt(Prompt):
    negative_prompt: str = ""
    model: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg_scale: float = 7.0
    seed: int = -1
    count: int = 1
    scheduler: str = "euler"
    vram: str = "low"
    output_dir: str | Path = "outputs"
