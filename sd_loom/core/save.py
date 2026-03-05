from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from sd_loom.core.metadata import build_png_metadata

if TYPE_CHECKING:
    from PIL import Image

    from sd_loom.core.protocol import SpecProtocol


def save_image(
    image: Image.Image,
    spec: SpecProtocol,
    workflow_name: str,
    seed: int,
    elapsed: float,
    *,
    run_timestamp: str,
    spec_name: str,
) -> Path:
    """Save a generated image with structured path and embedded metadata.

    Path: ``{output_dir}/{workflow}/{spec_name}/{run_timestamp}_{tag}_{seed}.png``
    """
    parts = [run_timestamp]
    if spec.tag:
        parts.append(spec.tag)
    parts.append(str(seed))
    filename = "_".join(parts) + ".png"

    output_dir = Path(str(spec.output_dir)) / workflow_name / spec_name
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / filename

    pnginfo = build_png_metadata(spec, workflow_name, seed, elapsed)
    save_kwargs: dict[str, Any] = {"pnginfo": pnginfo}
    image.save(image_path, **save_kwargs)

    return image_path
