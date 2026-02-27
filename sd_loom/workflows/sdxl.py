from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any

import click
import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
)

from sd_loom.core.resolve import resolve_lora, resolve_model, resolve_vae
from sd_loom.core.save import save_image
from sd_loom.core.types import GenerationResult

if TYPE_CHECKING:
    from sd_loom.core.protocol import SpecProtocol

SCHEDULERS: dict[str, tuple[type[Any], dict[str, Any]]] = {
    "euler": (EulerDiscreteScheduler, {}),
    "euler_a": (EulerAncestralDiscreteScheduler, {}),
    "dpm++_2m": (DPMSolverMultistepScheduler, {}),
    "dpm++_2m_karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
}

VRAM_BATCH_SIZE: dict[str, int] = {"low": 1, "medium": 2, "high": 4}


def run(spec: SpecProtocol) -> list[GenerationResult]:
    """SDXL txt2img workflow with VRAM-aware batching."""
    model_path = resolve_model(spec.model)
    click.echo(f"Loading {model_path.name} ...")

    pipe: Any = StableDiffusionXLPipeline.from_single_file(
        str(model_path),
        torch_dtype=torch.float16,
    )

    # VAE swap
    if spec.vae:
        from diffusers import AutoencoderKL

        vae_path = resolve_vae(spec.vae)
        click.echo(f"Loading VAE {vae_path.name} ...")
        pipe.vae = AutoencoderKL.from_single_file(
            str(vae_path), torch_dtype=torch.float16,
        )

    # LoRA loading
    if spec.loras:
        for lora_name, weight in spec.loras:
            lora_path = resolve_lora(lora_name)
            click.echo(f"Loading LoRA {lora_path.name} (weight={weight}) ...")
            pipe.load_lora_weights(str(lora_path), adapter_name=lora_name)
        names = [name for name, _ in spec.loras]
        weights = [w for _, w in spec.loras]
        pipe.set_adapters(names, adapter_weights=weights)

    _apply_vram_profile(pipe, spec.vram)

    pipe.scheduler = _make_scheduler(spec.scheduler, pipe.scheduler.config)

    base_seed = spec.seed if spec.seed >= 0 else random.randint(0, 2**32 - 1)
    seeds = [base_seed + i for i in range(spec.count)]
    max_batch = VRAM_BATCH_SIZE.get(spec.vram, 1)
    workflow_name = __name__.split(".")[-1]
    results: list[GenerationResult] = []

    for chunk_start in range(0, len(seeds), max_batch):
        chunk_seeds = seeds[chunk_start : chunk_start + max_batch]
        generators = [
            torch.Generator(device="cpu").manual_seed(s) for s in chunk_seeds
        ]

        click.echo(
            f"Generating {spec.width}x{spec.height}, {spec.steps} steps, "
            f"cfg {spec.cfg_scale}, seeds {chunk_seeds}, scheduler {spec.scheduler}"
        )

        t0 = time.perf_counter()
        pipe_result: Any = pipe(
            prompt=spec.prompt.positive,
            negative_prompt=spec.prompt.negative or None,
            width=spec.width,
            height=spec.height,
            num_inference_steps=spec.steps,
            guidance_scale=spec.cfg_scale,
            num_images_per_prompt=len(chunk_seeds),
            generator=generators,
        )
        elapsed = time.perf_counter() - t0

        for i, seed in enumerate(chunk_seeds):
            image_path = save_image(
                pipe_result.images[i], spec, workflow_name, seed, elapsed,
            )
            click.echo(f"Saved: {image_path} (seed={seed}, {elapsed:.1f}s)")
            results.append(GenerationResult(
                image_path=image_path,
                seed=seed,
                elapsed_seconds=elapsed,
                workflow=workflow_name,
            ))

    return results


VRAM_PROFILES = ("low", "medium", "high")


def _apply_vram_profile(pipe: Any, profile: str) -> None:
    """Configure pipeline memory optimizations based on VRAM profile."""
    if profile not in VRAM_PROFILES:
        available = ", ".join(VRAM_PROFILES)
        raise ValueError(f"Unknown VRAM profile '{profile}'. Available: {available}")

    if profile == "low":
        # ~8GB: CPU offload + VAE tiling + fp16 VAE
        pipe.enable_model_cpu_offload()
        pipe.vae.config.force_upcast = False
        pipe.enable_vae_tiling()
    elif profile == "medium":
        # ~12GB: model on GPU, VAE tiling, fp16 VAE (no CPU offload — faster)
        pipe.to("cuda")
        pipe.vae.config.force_upcast = False
        pipe.enable_vae_tiling()
    else:
        # high (~16GB+): everything on GPU, no tiling, allow VAE upcast
        pipe.to("cuda")


def _make_scheduler(name: str, config: dict[str, Any]) -> Any:
    """Instantiate a scheduler by name, using the pipeline's existing config."""
    if name not in SCHEDULERS:
        available = ", ".join(sorted(SCHEDULERS))
        raise ValueError(f"Unknown scheduler '{name}'. Available: {available}")
    cls, kwargs = SCHEDULERS[name]
    return cls.from_config(config, **kwargs)
