from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from sd_loom.core.loader import load_prompt, load_workflow

if TYPE_CHECKING:
    from sd_loom.core.protocol import PromptSpec
    from sd_loom.core.types import GenerationResult


@click.group()
def main() -> None:
    """sd-loom: Stable Diffusion where everything is Python."""


@main.command()
@click.argument("workflow_name")
@click.argument("prompt_name")
def run(workflow_name: str, prompt_name: str) -> None:
    """Run a generation workflow with the given prompt spec."""
    spec: PromptSpec = load_prompt(prompt_name)
    workflow_mod = load_workflow(workflow_name)

    run_fn: Any = workflow_mod.run
    result: GenerationResult = run_fn(spec)

    click.echo(f"Done: {result.image_path} (seed={result.seed}, {result.elapsed_seconds:.1f}s)")


@main.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
def info(image_path: Path) -> None:
    """Show embedded generation metadata from a PNG image."""
    from sd_loom.core.metadata import read_png_metadata

    try:
        data = read_png_metadata(image_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    click.echo(json.dumps(data, indent=2))
