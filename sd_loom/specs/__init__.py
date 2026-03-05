from __future__ import annotations

from pathlib import Path  # noqa: TC003 — Pydantic needs this at runtime

from sd_loom.core.types import LoomSpec, Prompt  # noqa: TC003


class DefaultSpec(LoomSpec):
    prompt: Prompt = Prompt(positive="")
    model: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg_scale: float = 7.0
    seed: int = -1
    count: int = 1
    scheduler: str = "euler"
    clip_skip: int = 1
    model_hash: str = ""
    vae: str = ""
    loras: list[tuple[str, float]] = []
    vram: str = "low"
    rng: str = "gpu"
    output_dir: str | Path = "outputs"
