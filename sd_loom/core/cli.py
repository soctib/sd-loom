from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click

from sd_loom.core.loader import load_spec, load_workflow

if TYPE_CHECKING:
    from sd_loom.core.types import LoomData


@click.command()
@click.argument("workflow_name")
@click.argument("spec_name")
@click.option("-s", "--set", "overrides", multiple=True,
              help="Override a spec field: --set key=value")
@click.option("-n", "--count", type=int, default=None,
              help="Number of images to generate (shorthand for --set count=N)")
def main(workflow_name: str, spec_name: str, overrides: tuple[str, ...],
         count: int | None) -> None:
    """sd-loom: Stable Diffusion where everything is Python.

    \b
    Everything is a workflow. The spec can be a .py file, .json file,
    bare built-in name, or any other file (image, safetensors, etc.).
    """
    all_overrides = list(overrides)
    if count is not None:
        all_overrides.append(f"count={count}")
    spec = load_spec(spec_name, overrides=tuple(all_overrides))
    workflow_mod = load_workflow(workflow_name)

    run_fn: Any = workflow_mod.run
    results: list[LoomData] = run_fn(spec)

    # Print text output.
    for r in results:
        if r.text is not None:
            click.echo(r.text)

    # Save images and summarize.
    from sd_loom.core.save import save_image

    saved: list[tuple[Any, LoomData]] = []
    for r in results:
        if r.image is not None:
            path = save_image(r.image, spec, r.workflow, r.seed, r.elapsed_seconds)
            saved.append((path, r))

    if len(saved) == 1:
        path, r = saved[0]
        click.echo(f"Done: {path} (seed={r.seed}, {r.elapsed_seconds:.1f}s)")
    elif len(saved) > 1:
        for path, r in saved:
            click.echo(f"Saved: {path} (seed={r.seed}, {r.elapsed_seconds:.1f}s)")
        total = sum(r.elapsed_seconds for _, r in saved)
        click.echo(f"Done: {len(saved)} images in {total:.1f}s")
