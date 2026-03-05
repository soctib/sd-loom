from __future__ import annotations

from typing import TYPE_CHECKING

from sd_loom.core.types import LoomData

if TYPE_CHECKING:
    from sd_loom.core.protocol import SpecProtocol


def run(spec: SpecProtocol) -> list[LoomData]:
    """Debug workflow. Prints all spec fields, no GPU required."""
    lines = [
        "--- debug ---",
        f"  prompt:          {spec.prompt.positive!r}",
        f"  negative:        {spec.prompt.negative!r}",
        f"  model:           {spec.model!r}",
        f"  size:            {spec.width}x{spec.height}",
        f"  steps:           {spec.steps}",
        f"  cfg_scale:       {spec.cfg_scale}",
        f"  seed:            {spec.seed}",
        f"  scheduler:       {spec.scheduler}",
        f"  clip_skip:       {spec.clip_skip}",
        f"  vae:             {spec.vae!r}",
        f"  loras:           {spec.loras!r}",
        f"  vram:            {spec.vram}",
        f"  rng:             {spec.rng}",
        f"  output_dir:      {spec.output_dir}",
        f"  input_image:     {spec.input_image!r}",
    ]
    return [LoomData(text="\n".join(lines))]
