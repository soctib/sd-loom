"""Show metadata from an image, safetensors file, or A1111 parameters text."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from sd_loom.core.types import LoomData

if TYPE_CHECKING:
    from sd_loom.core.protocol import SpecProtocol


def run(spec: SpecProtocol) -> list[LoomData]:
    """Read and display metadata from spec.input_image."""
    if not spec.input_image:
        raise SystemExit("info workflow requires an input file (image, safetensors, or txt).")

    path = Path(spec.input_image)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    if path.suffix == ".safetensors":
        from sd_loom.core.metadata import read_safetensors_metadata

        data = read_safetensors_metadata(path)
    elif path.suffix == ".txt":
        from sd_loom.core.metadata import parse_a1111

        data = parse_a1111(path.read_text())
    else:
        from sd_loom.core.metadata import read_image_metadata

        try:
            data = read_image_metadata(path)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    return [LoomData(text=json.dumps(data, indent=2))]
