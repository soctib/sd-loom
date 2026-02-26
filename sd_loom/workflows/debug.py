from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from sd_loom.core.protocol import SpecProtocol

from sd_loom.core.types import GenerationResult


def run(spec: SpecProtocol) -> list[GenerationResult]:
    """Debug workflow. Prints all spec fields and returns placeholder results."""
    click.echo("--- debug ---")
    click.echo(f"  prompt:          {spec.prompt!r}")
    click.echo(f"  negative_prompt: {spec.negative_prompt!r}")
    click.echo(f"  model:           {spec.model!r}")
    click.echo(f"  size:            {spec.width}x{spec.height}")
    click.echo(f"  steps:           {spec.steps}")
    click.echo(f"  cfg_scale:       {spec.cfg_scale}")
    click.echo(f"  seed:            {spec.seed}")
    click.echo(f"  count:           {spec.count}")
    click.echo(f"  scheduler:       {spec.scheduler}")
    click.echo(f"  vram:            {spec.vram}")
    click.echo(f"  output_dir:      {spec.output_dir}")

    import random

    base_seed = spec.seed if spec.seed >= 0 else random.randint(0, 2**32 - 1)
    workflow_name = __name__.split(".")[-1]
    results: list[GenerationResult] = []

    for i in range(spec.count):
        seed = base_seed + i
        output_path = Path(str(spec.output_dir)) / f"placeholder_{seed}.png"
        results.append(GenerationResult(
            image_path=output_path,
            seed=seed,
            elapsed_seconds=0.0,
            workflow=workflow_name,
        ))

    return results
