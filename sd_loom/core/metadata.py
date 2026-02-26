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


def read_png_metadata(path: Path) -> dict[str, Any]:
    """Read sd-loom metadata from a PNG file."""
    from PIL import Image

    with Image.open(path) as img:
        raw = img.info.get("sd-loom")
        if raw is None:
            raise ValueError(f"No sd-loom metadata in {path}")
        result: dict[str, Any] = json.loads(raw)
        return result


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
    lines.append(params)
    return "\n".join(lines)
