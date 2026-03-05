"""SDXL txt2img with k-diffusion sampling and ldm UNet (Forge-parity).

This is the primary SDXL workflow. The bare name 'sdxl' is an alias for this.
"""
from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any

import click
import k_diffusion as K
import torch

from sd_loom.core.types import LoomData
from sd_loom.nn.loader import load_ldm_unet
from sd_loom.nn.unet import timestep_embedding
from sd_loom.workflows.sdxl_common import load_pipeline, resolve_model_with_hash_fallback

if TYPE_CHECKING:
    from sd_loom.core.protocol import SpecProtocol

# Map scheduler names to (k-diffusion sampler function, use_karras_sigmas).
K_SAMPLERS: dict[str, tuple[str, bool]] = {
    "euler": ("sample_euler", False),
    "euler_karras": ("sample_euler", True),
    "euler_a": ("sample_euler_ancestral", False),
    "dpm++_2m": ("sample_dpmpp_2m", False),
    "dpm++_2m_karras": ("sample_dpmpp_2m", True),
    "dpm++_2m_sde": ("sample_dpmpp_2m_sde", False),
    "dpm++_2m_sde_karras": ("sample_dpmpp_2m_sde", True),
    "dpm++_sde": ("sample_dpmpp_sde", False),
    "dpm++_sde_karras": ("sample_dpmpp_sde", True),
}

# SDE samplers need BrownianTreeNoiseSampler for reproducible results.
_SDE_SAMPLERS = {"sample_dpmpp_2m_sde", "sample_dpmpp_sde"}


class _LdmUNetWrapper:
    """Bridge ldm UNet to k-diffusion's apply_model() convention.

    k-diffusion's CompVisDenoiser expects inner_model.apply_model(x, t, **kwargs)
    returning the noise prediction (eps). This wrapper translates that to the
    ldm UNet's forward(x, timesteps, context, y).
    """

    def __init__(self, unet: Any, alphas_cumprod: torch.Tensor) -> None:
        self.model = unet
        self.alphas_cumprod = alphas_cumprod

    def apply_model(
        self, x: torch.Tensor, t: torch.Tensor, **kwargs: Any,
    ) -> torch.Tensor:
        model_dtype = next(self.model.parameters()).dtype
        result: torch.Tensor = self.model(
            x.to(model_dtype), t,
            context=kwargs.get("cond"),
            y=kwargs.get("y"),
        )
        return result.float()


def run(spec: SpecProtocol) -> list[LoomData]:
    """SDXL txt2img with k-diffusion sampling and ldm UNet."""
    if spec.scheduler not in K_SAMPLERS:
        supported = ", ".join(sorted(K_SAMPLERS))
        raise ValueError(
            f"Scheduler '{spec.scheduler}' not supported by the native workflow. "
            f"Supported: {supported}. For ddim, use sdxl_diffusers."
        )

    # Load diffusers pipeline for text encoders, tokenizers, VAE, and alphas_cumprod.
    pipe, _clip_skip = load_pipeline(spec)

    # Load ldm UNet directly from safetensors (bypasses diffusers conversion).
    model_path = resolve_model_with_hash_fallback(spec)
    click.echo("Loading ldm UNet ...")
    ldm_unet = load_ldm_unet(model_path)

    if spec.loras:
        click.echo(
            "Warning: LoRA weights are applied to text encoders but NOT to the ldm UNet. "
            "UNet LoRA support for the ldm path is a follow-up task."
        )

    # --- Encode prompts with compel (Forge-compatible penultimate hidden states) ---
    prompt_embeds, neg_embeds, pooled, neg_pooled = _encode_prompt(pipe, spec)

    # Free the diffusers UNet — we only needed the pipeline for text encoders,
    # VAE, tokenizers, and alphas_cumprod.
    alphas_cumprod = pipe.scheduler.alphas_cumprod
    del pipe.unet
    torch.cuda.empty_cache()

    # --- Build k-diffusion denoiser from ldm UNet ---
    wrapper = _LdmUNetWrapper(ldm_unet, alphas_cumprod)
    denoiser: Any = K.external.CompVisDenoiser(wrapper, quantize=True)
    denoiser.to("cuda")  # moves internal sigma/log_sigma buffers

    # --- Sigma schedule ---
    sampler_name, use_karras = K_SAMPLERS[spec.scheduler]
    if use_karras:
        sigmas = K.sampling.get_sigmas_karras(
            n=spec.steps,
            sigma_min=float(denoiser.sigmas[0]),
            sigma_max=float(denoiser.sigmas[-1]),
            device="cuda",
        )
    else:
        sigmas = denoiser.get_sigmas(spec.steps).to("cuda")

    sampler_fn = getattr(K.sampling, sampler_name)

    # --- SDXL conditioning for CFG (negative first, positive second) ---
    # Build the y vector (pooled embeds + sinusoidal micro-conditioning), Forge-style.
    cond_embeds = torch.cat([neg_embeds, prompt_embeds]).to("cuda")
    time_values = torch.tensor(
        [spec.height, spec.width, 0, 0, spec.height, spec.width],
        dtype=torch.float32,
    )
    time_emb = torch.cat([
        timestep_embedding(v.unsqueeze(0), 256) for v in time_values
    ])  # [6, 256]
    time_emb = time_emb.flatten().unsqueeze(0).repeat(2, 1)  # [2, 1536]
    y = torch.cat([
        torch.cat([neg_pooled, pooled]),  # [2, 1280]
        time_emb.to(pooled),
    ], dim=1).to("cuda")  # [2, 2816]

    extra_args: dict[str, Any] = {
        "cond": cond_embeds,
        "y": y,
    }

    # --- CFG-handling model function for k-diffusion ---
    cfg_scale = spec.cfg_scale

    def model_fn(
        x: torch.Tensor, sigma: torch.Tensor, **kwargs: Any,
    ) -> torch.Tensor:
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        noise_pred: torch.Tensor = denoiser(x_in, sigma_in, **kwargs)
        uncond, cond = noise_pred.chunk(2)
        return uncond + cfg_scale * (cond - uncond)

    # --- Generation loop ---
    base_seed = spec.seed if spec.seed >= 0 else random.randint(0, 2**32 - 1)
    seeds = [base_seed + i for i in range(spec.count)]
    results: list[LoomData] = []
    workflow_name = __name__.split(".")[-1]
    latent_shape = (1, 4, spec.height // 8, spec.width // 8)

    # Move ldm UNet to CUDA (diffusers pipeline UNet is unused for denoising).
    ldm_unet.to("cuda")
    if spec.vram != "low":
        pipe.vae.to("cuda")
        pipe.vae.enable_tiling()

    for seed in seeds:
        click.echo(
            f"Generating {spec.width}x{spec.height}, {spec.steps} steps, "
            f"cfg {spec.cfg_scale}, seed {seed}, "
            f"scheduler {spec.scheduler} (k-diffusion, ldm UNet)"
        )

        # Initial noise — GPU matches Forge default, CPU for cross-platform reproducibility.
        rng_device = "cpu" if spec.rng == "cpu" else "cuda"
        generator = torch.Generator(device=rng_device).manual_seed(seed)
        latents = torch.randn(latent_shape, device=rng_device, generator=generator)
        latents = latents.to("cuda") * sigmas[0]

        sampler_kwargs: dict[str, Any] = {}
        if sampler_name in _SDE_SAMPLERS:
            sampler_kwargs["noise_sampler"] = K.sampling.BrownianTreeNoiseSampler(
                latents, sigma_min=sigmas[-2], sigma_max=sigmas[0], seed=seed,
            )

        t0 = time.perf_counter()
        with torch.no_grad():
            denoised = sampler_fn(
                model_fn, latents, sigmas,
                extra_args=extra_args, **sampler_kwargs,
            )

        # --- VAE decode (upcast to float32 to avoid NaN/black images) ---
        if spec.vram == "low":
            ldm_unet.to("cpu")
            torch.cuda.empty_cache()
            pipe.vae.to("cuda")
            pipe.vae.enable_tiling()

        pipe.vae.to(dtype=torch.float32)
        with torch.no_grad():
            decoded = pipe.vae.decode(
                denoised.float() / pipe.vae.config.scaling_factor,
                return_dict=False,
            )[0]
        pipe.vae.to(dtype=torch.float16)

        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        elapsed = time.perf_counter() - t0

        if spec.vram == "low":
            pipe.vae.to("cpu")
            torch.cuda.empty_cache()
            if seed != seeds[-1]:
                ldm_unet.to("cuda")

        click.echo(f"Generated (seed={seed}, {elapsed:.1f}s)")
        results.append(LoomData(
            image=image,
            seed=seed,
            elapsed_seconds=elapsed,
            workflow=workflow_name,
        ))

    return results


def _encode_prompt(
    pipe: Any, spec: SpecProtocol,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode prompts with compel. Returns (embeds, neg_embeds, pooled, neg_pooled).

    Uses PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED which matches Forge/A1111's
    encoding: hidden_states[-2] from both CLIP-L and CLIP-G, no final layer norm.
    Verified identical to direct transformer calls for simple prompts; compel adds
    support for emphasis (word:1.2) and BREAK syntax.
    """
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
        prompt_embeds, pooled = compel(spec.prompt.positive)  # pyright: ignore[reportAssignmentType]
        neg_embeds, neg_pooled = compel(spec.prompt.negative or "")  # pyright: ignore[reportAssignmentType]
    finally:
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

    return prompt_embeds, neg_embeds, pooled, neg_pooled  # pyright: ignore[reportReturnType]
