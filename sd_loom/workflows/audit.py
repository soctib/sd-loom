"""Audit a spec for potential issues before generation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sd_loom.core.metadata import read_safetensors_metadata
from sd_loom.core.resolve import resolve_lora, resolve_model, resolve_vae
from sd_loom.core.types import LoomData

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from sd_loom.core.protocol import SpecProtocol

# Standard SDXL aspect-ratio buckets (~1 megapixel each).
SDXL_BUCKETS = {
    (640, 1536), (768, 1344), (832, 1216), (896, 1152),
    (1024, 1024),
    (1152, 896), (1216, 832), (1344, 768), (1536, 640),
}

KNOWN_SCHEDULERS = {
    "euler", "euler_karras", "euler_a",
    "dpm++_2m", "dpm++_2m_karras",
    "dpm++_2m_sde", "dpm++_2m_sde_karras",
    "dpm++_sde", "dpm++_sde_karras",
    "ddim",
}

_LEVEL_TAG = {"ok": "  OK", "info": "INFO", "warn": "WARN", "error": " ERR"}


@dataclass
class _Check:
    level: str  # ok, info, warn, error
    category: str
    message: str


# ---------------------------------------------------------------------------
# Workflow class
# ---------------------------------------------------------------------------

class Audit:
    """Analyse a spec and report potential issues."""

    def run(
        self, spec: SpecProtocol, data: Iterator[LoomData] | None = None,
    ) -> Iterator[LoomData]:
        checks: list[_Check] = []

        model_path = _check_model(spec, checks)
        _check_loras(spec, checks)
        _check_vae(spec, checks)
        _check_resolution(spec, checks)
        _check_scheduler(spec, checks)
        _check_params(spec, checks)
        _check_prompt(spec, checks)

        model_hash = spec.model_hash
        if not model_hash and model_path is not None:
            import click

            click.echo(f"Hashing {model_path.name} for CivitAI lookup ...")
            model_hash = _compute_autov2(model_path)
        if model_hash:
            _check_civitai(model_hash, spec, checks)

        yield LoomData(text=_format_report(spec, checks))


# ---------------------------------------------------------------------------
# Individual check groups
# ---------------------------------------------------------------------------

def _check_model(spec: SpecProtocol, checks: list[_Check]) -> Path | None:
    if not spec.model:
        checks.append(_Check("error", "Model", "No model specified"))
        return None

    try:
        path = resolve_model(spec.model)
    except (FileNotFoundError, ValueError) as e:
        checks.append(_Check("error", "Model", str(e).split("\n")[0]))
        return None

    checks.append(_Check("ok", "Model", f"{path.name}"))

    if path.suffix == ".safetensors":
        meta = read_safetensors_metadata(path)
        checks.append(_Check("info", "Model",
                             f"{meta['file_size_mb']} MB, {meta['tensor_count']} tensors"))
        arch = meta.get("metadata", {}).get("modelspec.architecture")
        if arch:
            checks.append(_Check("info", "Model", f"Architecture: {arch}"))

    return path


def _check_loras(spec: SpecProtocol, checks: list[_Check]) -> None:
    if not spec.loras:
        return

    for lora_name, weight in spec.loras:
        try:
            path = resolve_lora(lora_name)
        except (FileNotFoundError, ValueError) as e:
            checks.append(_Check("error", f"LoRA '{lora_name}'", str(e).split("\n")[0]))
            continue

        checks.append(_Check("ok", f"LoRA '{lora_name}'",
                             f"{path.name} (weight={weight})"))

        if weight <= 0:
            checks.append(_Check("warn", f"LoRA '{lora_name}'",
                                 "Weight <= 0 — LoRA has no effect"))
        elif weight > 1.5:
            checks.append(_Check("warn", f"LoRA '{lora_name}'",
                                 f"Weight {weight} > 1.5 — may cause artifacts"))

        if path.suffix != ".safetensors":
            continue

        md = read_safetensors_metadata(path).get("metadata", {})
        _check_lora_metadata(lora_name, md, spec, checks)


def _check_lora_metadata(
    lora_name: str, md: dict[str, str], spec: SpecProtocol, checks: list[_Check],
) -> None:
    cat = f"LoRA '{lora_name}'"

    rank = md.get("ss_network_dim")
    alpha = md.get("ss_network_alpha")
    if rank:
        checks.append(_Check("info", cat, f"Rank {rank}, alpha {alpha or 'N/A'}"))

    base = md.get("ss_base_model_version", "")
    if base:
        checks.append(_Check("info", cat, f"Trained on base: {base}"))

    trained_model = md.get("ss_sd_model_name", "")
    if trained_model:
        checks.append(_Check("info", cat, f"Training checkpoint: {trained_model}"))

    clip_skip_raw = md.get("ss_clip_skip")
    if clip_skip_raw and clip_skip_raw != "None":
        lora_cs = int(clip_skip_raw)
        if lora_cs != spec.clip_skip:
            checks.append(_Check("warn", cat,
                                 f"Trained with clip_skip={lora_cs}, "
                                 f"spec uses clip_skip={spec.clip_skip}"))

    tag_freq_raw = md.get("ss_tag_frequency")
    if tag_freq_raw:
        _check_trigger_words(lora_name, tag_freq_raw, spec, checks)


def _check_trigger_words(
    lora_name: str, raw: str, spec: SpecProtocol, checks: list[_Check],
) -> None:
    cat = f"LoRA '{lora_name}'"
    try:
        tag_freq: dict[str, Any] = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return

    # ss_tag_frequency keys are "{repeats}_{concept}" — the concept IS the
    # activation tag by kohya convention.
    activation_tags: list[str] = []
    all_tags: dict[str, int] = {}
    for folder_key, dataset_tags in tag_freq.items():
        parts = str(folder_key).split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            activation_tags.append(parts[1])
        if isinstance(dataset_tags, dict):
            for tag, count in dataset_tags.items():
                tag = str(tag).strip()
                all_tags[tag] = all_tags.get(tag, 0) + int(count)

    if not all_tags:
        return

    top = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[:5]
    checks.append(_Check("info", cat,
                         "Top tags: " + ", ".join(f"'{t}' ({c})" for t, c in top)))

    prompt_lower = spec.prompt.positive.lower()

    if activation_tags:
        missing = [t for t in activation_tags if t.lower() not in prompt_lower]
        if missing:
            tags_str = ", ".join(f"'{t}'" for t in missing)
            checks.append(_Check("warn", cat,
                                 f"Activation tag {tags_str} not in prompt"))


def _check_vae(spec: SpecProtocol, checks: list[_Check]) -> None:
    if not spec.vae:
        checks.append(_Check("info", "VAE", "Using model's built-in VAE"))
        return
    try:
        path = resolve_vae(spec.vae)
        checks.append(_Check("ok", "VAE", path.name))
    except (FileNotFoundError, ValueError) as e:
        checks.append(_Check("error", "VAE", str(e).split("\n")[0]))


def _check_resolution(spec: SpecProtocol, checks: list[_Check]) -> None:
    w, h = spec.width, spec.height

    if w % 8 != 0 or h % 8 != 0:
        checks.append(_Check("error", "Resolution",
                             f"{w}x{h} — must be divisible by 8"))
        return

    if (w, h) in SDXL_BUCKETS:
        checks.append(_Check("ok", "Resolution", f"{w}x{h} (standard SDXL bucket)"))
        return

    mp = (w * h) / 1_000_000
    if mp < 0.5:
        checks.append(_Check("warn", "Resolution",
                             f"{w}x{h} ({mp:.2f} MP) — very low for SDXL (trained at ~1 MP)"))
    elif mp > 2.0:
        checks.append(_Check("warn", "Resolution",
                             f"{w}x{h} ({mp:.2f} MP) — may cause OOM or quality loss"))
    else:
        checks.append(_Check("warn", "Resolution",
                             f"{w}x{h} — not a standard SDXL bucket"))


def _check_scheduler(spec: SpecProtocol, checks: list[_Check]) -> None:
    if spec.scheduler in KNOWN_SCHEDULERS:
        checks.append(_Check("ok", "Scheduler", spec.scheduler))
    else:
        available = ", ".join(sorted(KNOWN_SCHEDULERS))
        checks.append(_Check("error", "Scheduler",
                             f"Unknown '{spec.scheduler}'. Available: {available}"))


def _check_params(spec: SpecProtocol, checks: list[_Check]) -> None:
    # CFG scale
    if spec.cfg_scale < 1.0:
        checks.append(_Check("warn", "CFG",
                             f"{spec.cfg_scale} — effectively no guidance"))
    elif spec.cfg_scale > 15.0:
        checks.append(_Check("warn", "CFG",
                             f"{spec.cfg_scale} — very high, may cause saturation"))
    else:
        checks.append(_Check("ok", "CFG", str(spec.cfg_scale)))

    # Steps
    if spec.steps < 10:
        checks.append(_Check("warn", "Steps",
                             f"{spec.steps} — very few, likely poor quality"))
    elif spec.steps > 60:
        checks.append(_Check("warn", "Steps",
                             f"{spec.steps} — diminishing returns past ~40"))
    else:
        checks.append(_Check("ok", "Steps", str(spec.steps)))



def _check_prompt(spec: SpecProtocol, checks: list[_Check]) -> None:
    pos = spec.prompt.positive.strip()
    if not pos:
        checks.append(_Check("error", "Prompt", "Empty positive prompt"))
        return

    words = len(pos.split())
    checks.append(_Check("info", "Prompt", f"{words} words"))

    if not spec.prompt.negative.strip():
        checks.append(_Check("info", "Prompt", "No negative prompt"))


def _check_civitai(
    model_hash: str, spec: SpecProtocol, checks: list[_Check],
) -> None:
    civitai = _civitai_lookup(model_hash)
    if not civitai:
        checks.append(_Check("info", "CivitAI", "No match for model hash"))
        return

    name = civitai.get("model", {}).get("name", "Unknown")
    version = civitai.get("name", "")
    base = civitai.get("baseModel", "")
    checks.append(_Check("info", "CivitAI", f"{name} — {version} (base: {base})"))

    trained_words: list[str] = civitai.get("trainedWords", [])
    if trained_words:
        checks.append(_Check("info", "CivitAI",
                             "Trigger words: " + ", ".join(trained_words)))

    images: list[dict[str, Any]] = civitai.get("images", [])
    examples = [img["meta"] for img in images if img.get("meta")]
    if examples:
        _check_civitai_examples(examples, spec, checks)


def _check_civitai_examples(
    examples: list[dict[str, Any]], spec: SpecProtocol, checks: list[_Check],
) -> None:
    """Compare spec settings against CivitAI example-image generation params."""
    from sd_loom.core.metadata import _a1111_sampler_to_scheduler

    cat = "CivitAI"

    # --- CFG ---
    cfgs = [float(e["cfgScale"]) for e in examples if "cfgScale" in e]
    if cfgs:
        lo, hi = min(cfgs), max(cfgs)
        cfg_str = str(lo) if lo == hi else f"{lo}-{hi}"
        checks.append(_Check("info", cat, f"Examples use CFG {cfg_str}"))
        median_cfg = sorted(cfgs)[len(cfgs) // 2]
        if abs(spec.cfg_scale - median_cfg) > 1.5:
            checks.append(_Check("warn", cat,
                                 f"Spec CFG {spec.cfg_scale} differs significantly "
                                 f"from examples ({cfg_str})"))

    # --- Steps ---
    steps = [int(e["steps"]) for e in examples if "steps" in e]
    if steps:
        lo, hi = min(steps), max(steps)
        steps_str = str(lo) if lo == hi else f"{lo}-{hi}"
        checks.append(_Check("info", cat, f"Examples use {steps_str} steps"))
        median_steps = sorted(steps)[len(steps) // 2]
        if spec.steps < median_steps * 0.5 or spec.steps > median_steps * 2:
            checks.append(_Check("warn", cat,
                                 f"Spec steps {spec.steps} differs significantly "
                                 f"from examples ({steps_str})"))

    # --- Scheduler ---
    sched_set: set[str] = set()
    for e in examples:
        sampler = e.get("sampler", "")
        schedule_type = e.get("Schedule type", "")
        if sampler:
            sched_set.add(_a1111_sampler_to_scheduler(sampler, schedule_type))
    if sched_set:
        sched_str = ", ".join(sorted(sched_set))
        checks.append(_Check("info", cat, f"Examples use scheduler: {sched_str}"))
        if spec.scheduler not in sched_set:
            checks.append(_Check("warn", cat,
                                 f"Spec scheduler '{spec.scheduler}' not used in "
                                 f"any example ({sched_str})"))

    # --- VAE ---
    vaes: set[str] = {e["VAE"] for e in examples if e.get("VAE")}
    if vaes:
        vae_str = ", ".join(sorted(vaes))
        checks.append(_Check("info", cat, f"Examples use VAE: {vae_str}"))
        if spec.vae:
            # Check if spec's VAE name fuzzy-matches any example VAE
            spec_vae_lower = spec.vae.lower()
            if not any(spec_vae_lower in v.lower() for v in vaes):
                checks.append(_Check("info", cat,
                                     f"Spec VAE '{spec.vae}' differs from examples"))
        else:
            checks.append(_Check("info", cat,
                                 "Spec uses built-in VAE; examples use external"))


def _compute_autov2(path: Path) -> str:
    """Compute CivitAI AutoV2 hash (SHA256 truncated to 10 hex chars)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:10]


def _civitai_lookup(model_hash: str) -> dict[str, Any] | None:
    """Look up a model version by hash on CivitAI."""
    import urllib.request

    url = f"https://civitai.com/api/v1/model-versions/by-hash/{model_hash}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sd-loom/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            result: dict[str, Any] = json.loads(resp.read())
            return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _format_report(spec: SpecProtocol, checks: list[_Check]) -> str:
    lines: list[str] = []
    tag = f" [{spec.tag}]" if spec.tag else ""
    lines.append(f"Audit: {spec.model}{tag}")
    lines.append("")

    prev_cat = ""
    for c in checks:
        if c.category != prev_cat:
            prev_cat = c.category
        lines.append(f"  [{_LEVEL_TAG[c.level]}] {c.category}: {c.message}")

    errors = sum(1 for c in checks if c.level == "error")
    warns = sum(1 for c in checks if c.level == "warn")
    lines.append("")
    if errors:
        lines.append(f"  {errors} error(s), {warns} warning(s)")
    elif warns:
        lines.append(f"  {warns} warning(s)")
    else:
        lines.append("  All checks passed")

    return "\n".join(lines)
