from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from sd_loom.core.loader import load_spec, load_workflow

if TYPE_CHECKING:
    from sd_loom.core.protocol import SpecProtocol
    from sd_loom.core.types import GenerationResult


@click.group()
def main() -> None:
    """sd-loom: Stable Diffusion where everything is Python."""


@main.command()
@click.argument("workflow_name")
@click.argument("prompt_name")
@click.option("-s", "--set", "overrides", multiple=True,
              help="Override a prompt field: --set key=value")
@click.option("-n", "--count", type=int, default=None,
              help="Number of images to generate (shorthand for --set count=N)")
def run(workflow_name: str, prompt_name: str, overrides: tuple[str, ...],
        count: int | None) -> None:
    """Run a generation workflow with the given prompt spec."""
    all_overrides = list(overrides)
    if count is not None:
        all_overrides.append(f"count={count}")
    spec: SpecProtocol = load_spec(prompt_name, overrides=tuple(all_overrides))
    workflow_mod = load_workflow(workflow_name)

    run_fn: Any = workflow_mod.run
    results: list[GenerationResult] = run_fn(spec)

    if len(results) == 1:
        r = results[0]
        click.echo(f"Done: {r.image_path} (seed={r.seed}, {r.elapsed_seconds:.1f}s)")
    else:
        for r in results:
            click.echo(f"Saved: {r.image_path} (seed={r.seed}, {r.elapsed_seconds:.1f}s)")
        total = sum(r.elapsed_seconds for r in results)
        click.echo(f"Done: {len(results)} images in {total:.1f}s")


@main.command()
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
def info(file_path: Path) -> None:
    """Show metadata from an image, safetensors file, or A1111 parameters text."""
    if file_path.suffix == ".safetensors":
        from sd_loom.core.metadata import read_safetensors_metadata

        data = read_safetensors_metadata(file_path)
    elif file_path.suffix == ".txt":
        from sd_loom.core.metadata import parse_a1111

        data = parse_a1111(file_path.read_text())
    else:
        from sd_loom.core.metadata import read_image_metadata

        try:
            data = read_image_metadata(file_path)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    click.echo(json.dumps(data, indent=2))
