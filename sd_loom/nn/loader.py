"""Load ldm UNet weights directly from safetensors checkpoint files."""
from __future__ import annotations

from typing import TYPE_CHECKING

import safetensors.torch

from sd_loom.nn.unet import IntegratedUNet2DConditionModel

if TYPE_CHECKING:
    from pathlib import Path

# Fixed config for all SDXL models (same as Forge/ComfyUI).
SDXL_UNET_CONFIG: dict[str, object] = {
    "in_channels": 4,
    "model_channels": 320,
    "out_channels": 4,
    "num_res_blocks": [2, 2, 2],
    "channel_mult": [1, 2, 4],
    "num_head_channels": 64,
    "context_dim": 2048,
    "adm_in_channels": 2816,
    "num_classes": "sequential",
    "use_spatial_transformer": True,
    "use_linear_in_transformer": True,
    "transformer_depth": [0, 0, 2, 2, 10, 10],
    "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
    "transformer_depth_middle": 10,
}


def load_ldm_unet(safetensors_path: Path) -> IntegratedUNet2DConditionModel:
    """Load ldm UNet directly from a full SDXL safetensors checkpoint.

    Extracts the ``model.diffusion_model.*`` keys and loads them into the
    IntegratedUNet2DConditionModel (original ldm architecture).
    """
    sd = safetensors.torch.load_file(str(safetensors_path), device="cpu")
    prefix = "model.diffusion_model."
    unet_sd = {
        k[len(prefix):]: v
        for k, v in sd.items()
        if k.startswith(prefix)
    }
    del sd  # free memory

    if not unet_sd:
        raise ValueError(
            f"No UNet weights found in {safetensors_path.name}. "
            f"Expected keys starting with '{prefix}'."
        )

    model = IntegratedUNet2DConditionModel(**SDXL_UNET_CONFIG)  # type: ignore[arg-type]
    model.load_state_dict(unet_sd)
    del unet_sd

    return model.half().eval()
