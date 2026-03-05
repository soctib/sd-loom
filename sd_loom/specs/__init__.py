from __future__ import annotations

from pathlib import Path  # noqa: TC003 — Pydantic needs this at runtime

from sd_loom.core.types import LoomSpec, Prompt  # noqa: TC003, F401

# ---------------------------------------------------------------------------
# Resolution presets — (width, height) tuples for common SDXL aspect ratios
# ---------------------------------------------------------------------------

square = (1024, 1024)
landscape = (1216, 832)
portrait = (832, 1216)
wide = (1344, 768)
tall = (768, 1344)
ultrawide = (1536, 640)


def lora(name: str, weight: float = 1.0) -> tuple[str, float]:
    """Shorthand for a LoRA entry: ``lora("detail")`` → ``("detail", 1.0)``."""
    return (name, weight)


class DefaultSpec(LoomSpec):  # type: ignore[metaclass]
    prompt: Prompt = Prompt(positive="")
    model: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg_scale: float = 7.0
    seed: int = -1
    scheduler: str = "euler"
    clip_skip: int = 1
    model_hash: str = ""
    vae: str = ""
    loras: list[tuple[str, float]] = []
    vram: str = "low"
    rng: str = "gpu"
    output_dir: str | Path = "outputs"
    input_image: str = ""
    tag: str = ""
