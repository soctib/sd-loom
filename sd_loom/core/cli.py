from __future__ import annotations

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
