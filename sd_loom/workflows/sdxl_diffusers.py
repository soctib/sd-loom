"""SDXL txt2img with compel encoding (A1111-compatible prompt handling)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sd_loom.workflows.sdxl_common import generate, load_pipeline

if TYPE_CHECKING:
    from sd_loom.core.protocol import SpecProtocol
    from sd_loom.core.types import LoomData


def run(spec: SpecProtocol) -> list[LoomData]:
    """SDXL txt2img with compel encoding and VRAM-aware batching."""
    pipe, _clip_skip = load_pipeline(spec)

    # Encode prompts with compel before applying VRAM profile,
    # since CPU offload breaks direct text encoder access.
    prompt_kwargs = _encode_prompt(pipe, spec)

    return generate(pipe, spec, prompt_kwargs, __name__.split(".")[-1])


def _encode_prompt(pipe: Any, spec: SpecProtocol) -> dict[str, Any]:
    """Encode prompts using compel for A1111-compatible text conditioning."""
    from compel import Compel, ReturnedEmbeddingsType

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

        prompt_embeds, pooled = compel(spec.prompt.positive)
        neg_embeds, neg_pooled = compel(spec.prompt.negative or "")
    finally:
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

    return {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled,
        "negative_prompt_embeds": neg_embeds,
        "negative_pooled_prompt_embeds": neg_pooled,
    }
