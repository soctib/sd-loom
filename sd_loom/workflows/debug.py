from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from sd_loom.core.protocol import PromptSpec

from sd_loom.core.types import GenerationResult


def run(spec: PromptSpec) -> GenerationResult:
    """Debug workflow. Prints all spec fields and returns a placeholder result."""
    click.echo("--- debug ---")
    click.echo(f"  prompt:          {spec.prompt!r}")
    click.echo(f"  negative_prompt: {spec.negative_prompt!r}")
    click.echo(f"  model:           {spec.model!r}")
    click.echo(f"  size:            {spec.width}x{spec.height}")
    click.echo(f"  steps:           {spec.steps}")
    click.echo(f"  cfg_scale:       {spec.cfg_scale}")
    click.echo(f"  seed:            {spec.seed}")
    click.echo(f"  scheduler:       {spec.scheduler}")
    click.echo(f"  output_dir:      {spec.output_dir}")

    output_path = Path(str(spec.output_dir)) / "placeholder.png"

    return GenerationResult(
        image_path=output_path,
        seed=spec.seed,
        elapsed_seconds=0.0,
        workflow=__name__.split(".")[-1],
    )
