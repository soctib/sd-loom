# 21 — Spec audit/analysis command

## Status: design

## Idea

A `loom audit SPEC` command (not a workflow — no generation, just analysis).
Takes a spec and reports everything it can learn about the models, LoRAs,
and settings, plus flags anything that looks wrong.

## What it would check

### Local metadata (no network)
- Model: file size, tensor count, base architecture (SD 1.5 / SDXL / etc.)
- LoRAs: ss_clip_skip, ss_base_model_version, ss_sd_model_name, ss_tag_frequency (trigger words)
- VAE: architecture, scaling factor
- clip_skip vs LoRA training clip_skip (already have autofix — audit surfaces the info)
- Scheduler: is it valid? does it make sense for the model type?
- Resolution: is it a standard SDXL bucket size?

### API lookups (CivitAI, maybe HuggingFace)
- Model/LoRA: full description, recommended settings, example images
- Base model compatibility: does the LoRA match the checkpoint's base?
- Known issues or notes from the model page
- Trigger words from the CivitAI page (compare with prompt)

### Cross-checks
- LoRA trained on different base than checkpoint (e.g. SD 1.5 LoRA on SDXL model)
- Missing trigger words in prompt (from LoRA metadata or CivitAI)
- Unusual CFG / step count for the model type
- Resolution not matching model's training resolution

## Output format
Structured report — could be JSON or a human-readable summary. Start with
human-readable, add --json later.

## Future: LLM analysis
Later, optionally pass the gathered data to an LLM for deeper analysis
("this LoRA was trained on Illustrious v0.1 but your checkpoint is based on
Illustrious v1.0 — results may vary"). Out of scope for v1.
