from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from sd_loom.core.loader import load_spec, load_workflow

if TYPE_CHECKING:
    from sd_loom.core.protocol import SpecProtocol
    from sd_loom.core.types import LoomData


def _expand_count(specs: list[SpecProtocol], count: int) -> list[SpecProtocol]:
    """Expand each spec into *count* copies with sequential seeds."""
    from pydantic import BaseModel

    expanded: list[SpecProtocol] = []
    for spec in specs:
        base_seed = spec.seed if spec.seed >= 0 else random.randint(0, 2**32 - 1)
        for i in range(count):
            if isinstance(spec, BaseModel):
                copy: SpecProtocol = type(spec).model_validate(
                    {**spec.model_dump(), "seed": base_seed + i}
                )
            else:
                copy = spec
            expanded.append(copy)
    return expanded


@click.command()
@click.argument("workflow_name")
@click.argument("spec_name")
@click.option("-s", "--set", "overrides", multiple=True,
              help="Override a spec field: --set key=value")
@click.option("-n", "--count", type=int, default=None,
              help="Generate N images per spec (expands specs with sequential seeds)")
def main(workflow_name: str, spec_name: str, overrides: tuple[str, ...],
         count: int | None) -> None:
    """sd-loom: Stable Diffusion where everything is Python.

    \b
    Everything is a workflow. The spec can be a .py file, .json file,
    bare built-in name, or any other file (image, safetensors, etc.).
    """
    specs = load_spec(spec_name, overrides=overrides)
    if count is not None:
        specs = _expand_count(specs, count)
    workflow = load_workflow(workflow_name)
    run_timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    spec_stem = Path(spec_name).stem if "/" in spec_name or spec_name.endswith(".py") else spec_name

    from sd_loom.core.save import save_image

    saved: list[tuple[Any, LoomData]] = []
    for spec in specs:
        for r in workflow.run(spec):
            if r.text is not None:
                click.echo(r.text)
            if r.image is not None:
                path = save_image(
                    r.image, spec, r.workflow, r.seed, r.elapsed_seconds,
                    run_timestamp=run_timestamp, spec_name=spec_stem,
                )
                saved.append((path, r))

    if len(saved) == 1:
        path, r = saved[0]
        click.echo(f"Done: {path} (seed={r.seed}, {r.elapsed_seconds:.1f}s)")
    elif len(saved) > 1:
        for path, r in saved:
            click.echo(f"Saved: {path} (seed={r.seed}, {r.elapsed_seconds:.1f}s)")
        total = sum(r.elapsed_seconds for _, r in saved)
        click.echo(f"Done: {len(saved)} images in {total:.1f}s")
