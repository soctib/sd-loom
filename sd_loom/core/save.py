from __future__ import annotations

from datetime import UTC, datetime
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
) -> Path:
    """Save a generated image with structured path and embedded metadata."""
    spec_name: str = spec.tag if spec.tag else type(spec).__module__.split(".")[-1]
    now = datetime.now(UTC)
    date_dir = now.strftime("%Y%m%d")
    time_stamp = now.strftime("%H%M%S")

    output_dir = Path(str(spec.output_dir)) / workflow_name / spec_name / date_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / f"{time_stamp}_{seed}.png"

    # Cast away the protocol type so mypy sees .model_dump()
    pnginfo = build_png_metadata(spec, workflow_name, seed, elapsed)
    save_kwargs: dict[str, Any] = {"pnginfo": pnginfo}
    image.save(image_path, **save_kwargs)

    return image_path
