from __future__ import annotations

import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
)

from sd_loom.core.resolve import resolve_model
from sd_loom.core.types import GenerationResult

if TYPE_CHECKING:
    from sd_loom.core.protocol import PromptSpec

SCHEDULERS: dict[str, tuple[type[Any], dict[str, Any]]] = {
    "euler": (EulerDiscreteScheduler, {}),
    "euler_a": (EulerAncestralDiscreteScheduler, {}),
    "dpm++_2m": (DPMSolverMultistepScheduler, {}),
    "dpm++_2m_karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
}

def run(spec: PromptSpec) -> GenerationResult:
    """SDXL txt2img workflow."""
    model_path = resolve_model(spec.model)
    click.echo(f"Loading {model_path.name} ...")

    pipe: Any = StableDiffusionXLPipeline.from_single_file(
        str(model_path),
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.config.force_upcast = False
    pipe.enable_vae_tiling()

    pipe.scheduler = _make_scheduler(spec.scheduler, pipe.scheduler.config)

    seed = spec.seed if spec.seed >= 0 else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    click.echo(
        f"Generating {spec.width}x{spec.height}, {spec.steps} steps, "
        f"cfg {spec.cfg_scale}, seed {seed}, scheduler {spec.scheduler}"
    )

    t0 = time.perf_counter()
    result: Any = pipe(
        prompt=spec.prompt,
        negative_prompt=spec.negative_prompt or None,
        width=spec.width,
        height=spec.height,
        num_inference_steps=spec.steps,
        guidance_scale=spec.cfg_scale,
        generator=generator,
    )
    elapsed = time.perf_counter() - t0

    output_dir = Path(str(spec.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    image_path = output_dir / f"{timestamp}_{seed}.png"
    result.images[0].save(image_path)

    click.echo(f"Saved: {image_path} ({elapsed:.1f}s)")

    return GenerationResult(
        image_path=image_path,
        seed=seed,
        elapsed_seconds=elapsed,
    )


def _make_scheduler(name: str, config: dict[str, Any]) -> Any:
    """Instantiate a scheduler by name, using the pipeline's existing config."""
    if name not in SCHEDULERS:
        available = ", ".join(sorted(SCHEDULERS))
        raise ValueError(f"Unknown scheduler '{name}'. Available: {available}")
    cls, kwargs = SCHEDULERS[name]
    return cls.from_config(config, **kwargs)
