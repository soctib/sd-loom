"""Shared SDXL pipeline setup, scheduling, and generation loop."""
from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any

import click
import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
)

from sd_loom.core.metadata import read_safetensors_metadata
from sd_loom.core.resolve import resolve_lora, resolve_model, resolve_vae
from sd_loom.core.types import LoomData

if TYPE_CHECKING:
    from sd_loom.core.protocol import SpecProtocol

SCHEDULERS: dict[str, tuple[type[Any], dict[str, Any]]] = {
    "euler": (EulerDiscreteScheduler, {}),
    "euler_karras": (EulerDiscreteScheduler, {"use_karras_sigmas": True}),
    "euler_a": (EulerAncestralDiscreteScheduler, {}),
    "dpm++_2m": (DPMSolverMultistepScheduler, {}),
    "dpm++_2m_karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
    "dpm++_2m_sde": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"}),
    "dpm++_2m_sde_karras": (
        DPMSolverMultistepScheduler,
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    ),
    "dpm++_sde": (DPMSolverSDEScheduler, {}),
    "dpm++_sde_karras": (DPMSolverSDEScheduler, {"use_karras_sigmas": True}),
    "ddim": (DDIMScheduler, {}),
}

VRAM_BATCH_SIZE: dict[str, int] = {"low": 1, "medium": 2, "high": 4}
VRAM_PROFILES = ("low", "medium", "high")

# Pipeline cache — reuse across consecutive specs with the same model/VAE/LoRAs.
_pipe_cache_key: str = ""
_pipe_cache: tuple[Any, int] | None = None


def _make_pipe_key(spec: SpecProtocol) -> str:
    """Build a cache key from model, VAE, and LoRAs."""
    model_path = resolve_model_with_hash_fallback(spec)
    return f"{model_path}:{spec.vae}:{tuple(sorted(spec.loras))}"


def load_pipeline(spec: SpecProtocol) -> tuple[Any, int]:
    """Load model, VAE, and LoRAs. Returns (pipeline, clip_skip).

    Results are cached — consecutive calls with the same model/VAE/LoRAs
    skip loading entirely.
    """
    global _pipe_cache_key, _pipe_cache

    key = _make_pipe_key(spec)
    if key == _pipe_cache_key and _pipe_cache is not None:
        click.echo("Reusing cached pipeline")
        return _pipe_cache

    # Clear old cache
    _pipe_cache = None
    _pipe_cache_key = ""
    torch.cuda.empty_cache()

    model_path = resolve_model_with_hash_fallback(spec)
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
    clip_skip = spec.clip_skip
    if spec.loras:
        resolved_loras = [
            (name, resolve_lora(name), weight)
            for name, weight in spec.loras
        ]
        clip_skip = resolve_clip_skip(
            [(name, path) for name, path, _ in resolved_loras],
            spec.clip_skip,
        )
        adapter_names: list[str] = []
        adapter_weights: list[float] = []
        for _, lora_path, weight in resolved_loras:
            adapter_name = lora_path.stem.replace(".", "_")
            click.echo(f"Loading LoRA {lora_path.name} (weight={weight}) ...")
            pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
            adapter_names.append(adapter_name)
            adapter_weights.append(weight)
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    _pipe_cache_key = key
    _pipe_cache = (pipe, clip_skip)
    return pipe, clip_skip


def generate(
    pipe: Any,
    spec: SpecProtocol,
    prompt_kwargs: dict[str, Any],
    workflow_name: str,
) -> list[LoomData]:
    """Apply VRAM profile, scheduler, and run the batched generation loop."""
    apply_vram_profile(pipe, spec.vram)
    pipe.scheduler = make_scheduler(spec.scheduler, pipe.scheduler.config)

    base_seed = spec.seed if spec.seed >= 0 else random.randint(0, 2**32 - 1)
    seeds = [base_seed + i for i in range(spec.count)]
    max_batch = VRAM_BATCH_SIZE.get(spec.vram, 1)
    results: list[LoomData] = []

    for chunk_start in range(0, len(seeds), max_batch):
        chunk_seeds = seeds[chunk_start : chunk_start + max_batch]
        rng_device = "cpu" if spec.rng == "cpu" else "cuda"
        generators = [
            torch.Generator(device=rng_device).manual_seed(s) for s in chunk_seeds
        ]

        click.echo(
            f"Generating {spec.width}x{spec.height}, {spec.steps} steps, "
            f"cfg {spec.cfg_scale}, seeds {chunk_seeds}, scheduler {spec.scheduler}"
        )

        t0 = time.perf_counter()
        pipe_result: Any = pipe(
            **prompt_kwargs,
            width=spec.width,
            height=spec.height,
            num_inference_steps=spec.steps,
            guidance_scale=spec.cfg_scale,
            num_images_per_prompt=len(chunk_seeds),
            generator=generators,
        )
        elapsed = time.perf_counter() - t0

        for i, seed in enumerate(chunk_seeds):
            click.echo(f"Generated (seed={seed}, {elapsed:.1f}s)")
            results.append(LoomData(
                image=pipe_result.images[i],
                seed=seed,
                elapsed_seconds=elapsed,
                workflow=workflow_name,
            ))

    return results


def resolve_clip_skip(
    lora_paths: list[tuple[str, Any]], spec_clip_skip: int,
) -> int:
    """Determine clip_skip from LoRA metadata, auto-correcting if needed."""
    from pathlib import Path

    lora_clip_skips: dict[str, int] = {}
    for lora_name, lora_path in lora_paths:
        path = Path(lora_path)
        if path.suffix != ".safetensors":
            continue
        meta = read_safetensors_metadata(path).get("metadata", {})
        val = meta.get("ss_clip_skip")
        if val is None or val == "None":
            continue
        lora_clip_skips[lora_name] = int(val)

    if not lora_clip_skips:
        return spec_clip_skip

    unique_values = set(lora_clip_skips.values())
    if len(unique_values) > 1:
        detail = ", ".join(f"{n}: clip_skip={v}" for n, v in lora_clip_skips.items())
        raise ValueError(
            f"LoRAs require conflicting clip_skip values ({detail}). "
            f"These LoRAs may not be compatible with each other."
        )

    lora_clip_skip = unique_values.pop()
    if lora_clip_skip != spec_clip_skip:
        names = ", ".join(lora_clip_skips)
        click.echo(
            f"Auto-setting clip_skip={lora_clip_skip} "
            f"(LoRA {names} trained with clip_skip={lora_clip_skip})"
        )
        return lora_clip_skip

    return spec_clip_skip


def resolve_model_with_hash_fallback(spec: SpecProtocol) -> Any:
    """Resolve model by name, falling back to CivitAI hash lookup."""
    from pathlib import Path

    try:
        return resolve_model(spec.model)
    except FileNotFoundError:
        pass

    if spec.model_hash:
        civitai = _civitai_lookup(spec.model_hash)
        if civitai:
            for f in civitai.get("files", []):
                if f.get("primary"):
                    canonical = Path(f["name"]).stem
                    try:
                        resolved = resolve_model(canonical)
                        click.echo(
                            f"Model '{spec.model}' not found, "
                            f"but matched '{resolved.name}' by hash"
                        )
                        return resolved
                    except FileNotFoundError:
                        pass

            model_name = civitai.get("model", {}).get("name", "Unknown")
            version_name = civitai.get("name", "")
            base_model = civitai.get("baseModel", "")
            download_url = civitai.get("downloadUrl", "")
            filename = ""
            for f in civitai.get("files", []):
                if f.get("primary"):
                    filename = f.get("name", "")
                    break

            raise FileNotFoundError(
                f"No models matching '{spec.model}' found locally.\n\n"
                f"CivitAI match: {model_name} {version_name} ({base_model})\n"
                f"Download: {download_url}\n"
                f"Save as: models/sdxl/checkpoints/{filename}"
            )

    raise FileNotFoundError(
        f"No models matching '{spec.model}' found in models/"
    )


def apply_vram_profile(pipe: Any, profile: str) -> None:
    """Configure pipeline memory optimizations based on VRAM profile."""
    if profile not in VRAM_PROFILES:
        available = ", ".join(VRAM_PROFILES)
        raise ValueError(f"Unknown VRAM profile '{profile}'. Available: {available}")

    if profile == "low":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
    elif profile == "medium":
        pipe.to("cuda")
        pipe.enable_vae_tiling()
    else:
        pipe.to("cuda")
        pipe.enable_vae_tiling()


def make_scheduler(name: str, config: dict[str, Any]) -> Any:
    """Instantiate a scheduler by name with A1111-compatible timestep spacing."""
    if name not in SCHEDULERS:
        available = ", ".join(sorted(SCHEDULERS))
        raise ValueError(f"Unknown scheduler '{name}'. Available: {available}")
    cls, kwargs = SCHEDULERS[name]
    return cls.from_config(config, timestep_spacing="trailing", steps_offset=0, **kwargs)


def _civitai_lookup(model_hash: str) -> dict[str, Any] | None:
    """Look up a model by hash on CivitAI. Returns API response or None."""
    import json
    import urllib.request

    url = f"https://civitai.com/api/v1/model-versions/by-hash/{model_hash}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sd-loom/0.1"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            result: dict[str, Any] = json.loads(resp.read())
            return result
    except Exception:
        return None
