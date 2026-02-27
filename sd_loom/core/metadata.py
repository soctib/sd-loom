from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from PIL.PngImagePlugin import PngInfo

if TYPE_CHECKING:
    from pathlib import Path

    from sd_loom.core.protocol import SpecProtocol


def build_png_metadata(
    spec: SpecProtocol,
    workflow_name: str,
    seed: int,
    elapsed: float,
) -> PngInfo:
    """Build PNG tEXt metadata from generation parameters."""
    from pydantic import BaseModel

    if not isinstance(spec, BaseModel):
        raise TypeError("spec must be a Pydantic BaseModel")
    data: dict[str, Any] = spec.model_dump()
    data["seed"] = seed
    data["workflow"] = workflow_name
    data["elapsed_seconds"] = round(elapsed, 2)
    # Stringify Path so it's JSON-serializable
    data["output_dir"] = str(data["output_dir"])

    info = PngInfo()
    info.add_text("sd-loom", json.dumps(data))
    info.add_text("parameters", _a1111_format(data))
    return info


def read_image_metadata(path: Path) -> dict[str, Any]:
    """Read generation metadata from an image file (PNG, JPG, WebP, etc.).

    Tries multiple sources in order:
    1. sd-loom JSON (PNG text chunk)
    2. A1111 "parameters" (PNG text chunk)
    3. EXIF UserComment (JPEG/WebP — where A1111 stores params)
    """
    from PIL import Image

    with Image.open(path) as img:
        # 1. Our own format (PNG text chunk)
        raw = img.info.get("sd-loom")
        if raw is not None:
            result: dict[str, Any] = json.loads(raw)
            return result

        # 2. A1111 parameters in PNG text chunk
        params = img.info.get("parameters")
        if params is not None:
            return parse_a1111(str(params))

        # 3. EXIF UserComment (JPEG/WebP)
        exif = img.getexif()
        if exif:
            # UserComment is EXIF tag 0x9286 (37510)
            user_comment = exif.get(0x9286)
            if user_comment:
                text = user_comment if isinstance(user_comment, str) else user_comment.decode("utf-8", errors="replace")
                return parse_a1111(text)

    raise ValueError(f"No generation metadata found in {path}")


def read_png_metadata(path: Path) -> dict[str, Any]:
    """Read sd-loom metadata from a PNG file.

    Deprecated — use ``read_image_metadata`` instead.
    """
    return read_image_metadata(path)


def read_safetensors_metadata(path: Path) -> dict[str, Any]:
    """Read ``__metadata__`` from a safetensors file header (no weight loading)."""
    import struct

    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header: dict[str, Any] = json.loads(f.read(header_size))

    meta: dict[str, str] = header.get("__metadata__", {})
    return {
        "file": path.name,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 1),
        "tensor_count": len([k for k in header if k != "__metadata__"]),
        "metadata": meta,
    }


def parse_a1111(text: str) -> dict[str, Any]:
    """Parse an A1111-format parameters string into a dict.

    Format::

        positive prompt
        Negative prompt: negative prompt
        Steps: 30, Sampler: DPM++ 2M SDE, CFG scale: 5, Seed: 42, Size: 1024x1536, Model: name, Schedule type: Karras
    """
    import re

    lines = text.strip().split("\n")

    # Find the "Negative prompt:" line and the params line (last line with "Steps:")
    neg_idx: int | None = None
    params_idx: int | None = None
    for i, line in enumerate(lines):
        if line.startswith("Negative prompt:"):
            neg_idx = i
        if re.match(r"Steps:\s*\d", line):
            params_idx = i

    # Extract sections
    if params_idx is not None:
        params_line = lines[params_idx]
        if neg_idx is not None:
            positive = "\n".join(lines[:neg_idx]).strip()
            negative = "\n".join(lines[neg_idx:params_idx]).strip()
            negative = re.sub(r"^Negative prompt:\s*", "", negative)
        else:
            positive = "\n".join(lines[:params_idx]).strip()
            negative = ""
    else:
        positive = text.strip()
        negative = ""
        params_line = ""

    # Extract inline LoRAs from prompt: <lora:name:weight>
    loras: list[tuple[str, float]] = []
    for m in re.finditer(r"<lora:([^:>]+):([\d.]+)>", positive):
        loras.append((m.group(1), float(m.group(2))))
    # Remove LoRA tags from prompt text
    clean_positive = re.sub(r"\s*<lora:[^>]+>\s*", " ", positive).strip()
    # Collapse BREAK markers (keep them, they're meaningful for SDXL)
    clean_positive = re.sub(r"\s*BREAK\s*", " BREAK ", clean_positive).strip()

    # Parse key-value params
    params: dict[str, str] = {}
    if params_line:
        for m in re.finditer(r"(\w[\w\s]*?):\s*([^,]+?)(?:,|$)", params_line):
            params[m.group(1).strip()] = m.group(2).strip()

    # Build scheduler name from Sampler + Schedule type
    sampler = params.get("Sampler", "")
    schedule_type = params.get("Schedule type", "")
    scheduler = _a1111_sampler_to_scheduler(sampler, schedule_type)

    # Parse size
    size = params.get("Size", "")
    width, height = 1024, 1024
    if "x" in size:
        parts = size.split("x")
        width, height = int(parts[0]), int(parts[1])

    result: dict[str, Any] = {
        "prompt": {"positive": clean_positive, "negative": negative},
        "model": params.get("Model", ""),
        "model_hash": params.get("Model hash", ""),
        "width": width,
        "height": height,
        "steps": int(params.get("Steps", "0")),
        "cfg_scale": float(params.get("CFG scale", "0")),
        "seed": int(params.get("Seed", "-1")),
        "scheduler": scheduler,
        "loras": loras,
    }

    # Include extra A1111 params we don't map but are useful to see
    known_keys = {"Steps", "Sampler", "CFG scale", "Seed", "Size", "Model",
                  "Model hash", "Schedule type", "Negative prompt"}
    extras = {k: v for k, v in params.items() if k not in known_keys}
    if extras:
        result["a1111_extra"] = extras

    return result


def _a1111_sampler_to_scheduler(sampler: str, schedule_type: str) -> str:
    """Map A1111 sampler name + schedule type to our scheduler name."""
    # Normalize
    s = sampler.lower().replace(" ", "_").replace("++", "++")
    # Common mappings
    mapping: dict[str, str] = {
        "euler": "euler",
        "euler_a": "euler_a",
        "dpm++_2m": "dpm++_2m",
        "dpm++_2m_sde": "dpm++_2m_sde",
        "dpm++_sde": "dpm++_sde",
        "ddim": "ddim",
    }
    base = mapping.get(s, s)
    if schedule_type.lower() == "karras":
        base += "_karras"
    return base


def _a1111_format(data: dict[str, Any]) -> str:
    """Build an A1111-compatible parameters string."""
    prompt_data = data.get("prompt", {})
    if isinstance(prompt_data, dict):
        positive = prompt_data.get("positive", "")
        negative = prompt_data.get("negative", "")
    else:
        positive = str(prompt_data)
        negative = ""

    lines: list[str] = [positive]

    if negative:
        lines.append(f"Negative prompt: {negative}")

    params = (
        f"Steps: {data.get('steps')}, "
        f"Sampler: {data.get('scheduler')}, "
        f"CFG scale: {data.get('cfg_scale')}, "
        f"Seed: {data.get('seed')}, "
        f"Size: {data.get('width')}x{data.get('height')}, "
        f"Model: {data.get('model')}"
    )

    vae = data.get("vae", "")
    if vae:
        params += f", VAE: {vae}"

    loras: list[list[Any]] = data.get("loras", [])
    for lora_name, weight in loras:
        params += f", <lora:{lora_name}:{weight}>"

    lines.append(params)
    return "\n".join(lines)
