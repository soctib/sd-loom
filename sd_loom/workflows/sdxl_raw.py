"""SDXL txt2img with default diffusers prompt encoding."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sd_loom.workflows.sdxl_common import SdxlBase

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sd_loom.core.protocol import SpecProtocol
    from sd_loom.core.types import LoomData


class SdxlRaw(SdxlBase):
    """SDXL txt2img with diffusers default encoding."""

    def run(
        self, spec: SpecProtocol, data: Iterator[LoomData] | None = None,
    ) -> Iterator[LoomData]:
        pipe, clip_skip = self._load_pipeline(spec)

        prompt_kwargs: dict[str, Any] = {
            "prompt": spec.prompt.positive,
            "negative_prompt": spec.prompt.negative or None,
            "clip_skip": clip_skip if clip_skip > 1 else None,
        }

        yield from self._generate(pipe, spec, prompt_kwargs, __name__.split(".")[-1])
