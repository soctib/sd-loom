"""SDXL txt2img with default diffusers prompt encoding."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sd_loom.core.types import LoomData
from sd_loom.workflows.sdxl_common import generate, load_pipeline

if TYPE_CHECKING:
    from sd_loom.core.protocol import SpecProtocol


def run(spec: SpecProtocol) -> list[LoomData]:
    """SDXL txt2img with diffusers default encoding and VRAM-aware batching."""
    pipe, clip_skip = load_pipeline(spec)

    prompt_kwargs: dict[str, Any] = {
        "prompt": spec.prompt.positive,
        "negative_prompt": spec.prompt.negative or None,
        "clip_skip": clip_skip if clip_skip > 1 else None,
    }

    return generate(pipe, spec, prompt_kwargs, __name__.split(".")[-1])
