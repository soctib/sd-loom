"""SDXL txt2img with compel encoding (A1111-compatible prompt handling)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sd_loom.workflows.sdxl_common import SdxlBase

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sd_loom.core.protocol import SpecProtocol
    from sd_loom.core.types import LoomData


class SdxlDiffusers(SdxlBase):
    """SDXL txt2img with compel encoding and diffusers scheduler."""

    def run(
        self, spec: SpecProtocol, data: Iterator[LoomData] | None = None,
    ) -> Iterator[LoomData]:
        pipe, _clip_skip = self._load_pipeline(spec)
        prompt_kwargs = _encode_prompt(pipe, spec)
        yield from self._generate(pipe, spec, prompt_kwargs, __name__.split(".")[-1])


def _has_cpu_offload(pipe: Any) -> bool:
    """Check if the pipeline has accelerate CPU offload hooks installed."""
    return hasattr(pipe, "_all_hooks") and len(pipe._all_hooks) > 0


def _encode_prompt(pipe: Any, spec: SpecProtocol) -> dict[str, Any]:
    """Encode prompts using compel for A1111-compatible text conditioning."""
    from compel import Compel, ReturnedEmbeddingsType

    offloaded = _has_cpu_offload(pipe)
    if offloaded:
        # Remove hooks temporarily so we can move encoders manually.
        pipe.remove_all_hooks()

    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")

    try:
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )

        prompt_embeds, pooled = compel(spec.prompt.positive)  # pyright: ignore[reportAssignmentType]
        neg_embeds, neg_pooled = compel(spec.prompt.negative or "")  # pyright: ignore[reportAssignmentType]
    finally:
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()
        if offloaded:
            pipe.enable_model_cpu_offload()

    return {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled,
        "negative_prompt_embeds": neg_embeds,
        "negative_pooled_prompt_embeds": neg_pooled,
    }
