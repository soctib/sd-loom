# 20 — Native SDXL workflow (A1111/ComfyUI parity)

## Status: ready

## Problem

The current `sdxl` workflow uses diffusers for inference. Even with compel
encoding and trailing timestep spacing, there are subtle differences from
A1111/Forge/ComfyUI output — particularly in color distribution when prompts
don't specify explicit colors.

Root cause: diffusers is a ground-up reimplementation. A1111, ComfyUI, and
Forge all share the original Stability AI model code + k-diffusion samplers,
which is why they produce near-identical results.

## Proposal

Build a new `sdxl` workflow using the original inference primitives:

- **Model loading**: load safetensors directly into the original SDXL
  architecture (not diffusers' reimplementation)
- **Prompt encoding**: original SDXL dual-CLIP encoding (handles BREAK,
  long prompts, proper hidden state extraction)
- **Samplers**: k-diffusion (`sample_dpmpp_2m_sde`, `sample_euler_ancestral`,
  etc.) — already installed
- **LoRA**: apply LoRA weights directly to model state dict
- **VRAM**: CPU offload / tiling as needed

Reuse `sdxl_common.py` for model resolution, CivitAI lookup, scheduler
mapping, clip_skip autofix, and the generation loop.

## Workflow naming after implementation

| Name             | Engine                         |
|------------------|-------------------------------|
| `sdxl`           | Native (original SD + k-diff) |
| `sdxl_diffusers` | Diffusers + compel encoding   |
| `sdxl_raw`       | Diffusers default encoding    |

## Evidence

- Compel encoding fixed the desaturation (gray/white → correct color palette)
- k-diffusion samplers alone didn't help (diffusers' prompt encoding was the
  real issue)
- Remaining gap: same seed produces different color distribution (noise init
  differs), and subtle encoding differences persist
- img2img upscale support (issue 17 remainder) would also be easier with
  native code
