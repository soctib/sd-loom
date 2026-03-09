"""Gradio GUI for sd-loom (issue 45)."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import click
import gradio as gr

from sd_loom.core.loader import _find_workflow_class, load_spec, load_workflow

# Internal modules that aren't runnable workflows.
_SKIP_MODULES = {"__init__", "sdxl_common"}
_CSS = "h3 { margin: 0.75rem 0 0.5rem 0.5rem !important; }"

# ── Discovery ──────────────────────────────────────────────────────────────


def discover_workflows() -> dict[str, str]:
    """Return {name: docstring} for all built-in workflows."""
    workflows_dir = Path(__file__).parent / "workflows"
    result: dict[str, str] = {}
    for py in sorted(workflows_dir.glob("*.py")):
        name = py.stem
        if name in _SKIP_MODULES:
            continue
        try:
            module = importlib.import_module(f"sd_loom.workflows.{name}")
            cls = _find_workflow_class(module, name)
            if cls is not None:
                doc = cls.__doc__ or "(no description)"
                result[name] = doc.strip()
        except Exception:
            continue
    return result


def discover_specs() -> list[str]:
    """Return spec file paths from ``specs/`` directory."""
    specs_dir = Path.cwd() / "specs"
    if not specs_dir.is_dir():
        return []
    return sorted(
        str(p.relative_to(Path.cwd()))
        for p in specs_dir.iterdir()
        if p.suffix in (".py", ".json")
    )


def preview_spec(spec_path: str) -> str:
    """Load a spec and return its fields as formatted JSON."""
    if not spec_path:
        return ""
    try:
        specs = load_spec(spec_path)
        if not specs:
            return "(empty)"
        from pydantic import BaseModel

        data = specs[0].model_dump() if isinstance(specs[0], BaseModel) else vars(specs[0])
        return json.dumps(data, indent=2, default=str)
    except Exception as exc:
        return f"Error loading spec: {exc}"


# ── App ────────────────────────────────────────────────────────────────────

# Workflow instance cache — reused across runs, cleared on workflow switch.
_cached_workflow: tuple[str, Any] | None = None


def _run_generation(
    workflow_name: str,
    spec_path: str,
    progress: gr.Progress = gr.Progress(),  # noqa: B008
) -> str:
    """Run the workflow in a background thread (Gradio handles threading)."""
    global _cached_workflow  # noqa: PLW0603

    if not workflow_name or not spec_path:
        return "Select both a workflow and a spec first."

    from datetime import UTC, datetime

    from sd_loom.core.save import save_image

    try:
        specs = load_spec(spec_path)
        # Reuse cached workflow instance if same name (preserves model cache).
        if _cached_workflow is not None and _cached_workflow[0] == workflow_name:
            workflow = _cached_workflow[1]
        else:
            workflow = load_workflow(workflow_name)
            _cached_workflow = (workflow_name, workflow)
    except Exception as exc:
        return f"Error: {exc}"

    run_timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    spec_stem = Path(spec_path).stem

    saved: list[str] = []
    texts: list[str] = []
    try:
        for spec in specs:
            for r in workflow.run(spec):
                if r.text is not None:
                    texts.append(r.text)
                if r.image is not None:
                    path = save_image(
                        r.image, spec, r.workflow, r.seed, r.elapsed_seconds,
                        run_timestamp=run_timestamp, spec_name=spec_stem,
                    )
                    saved.append(f"{path} (seed={r.seed}, {r.elapsed_seconds:.1f}s)")
    except SystemExit as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error during generation: {exc}"

    parts: list[str] = []
    if texts:
        parts.append("\n".join(texts))
    if saved:
        parts.append(f"Saved {len(saved)} image(s):")
        parts.extend(f"  {s}" for s in saved)
    return "\n".join(parts) if parts else "Done (no output)."


def _clear_workflow_cache(name: str) -> dict[str, Any]:
    """Clear the cached workflow when the selection changes."""
    global _cached_workflow  # noqa: PLW0603
    if _cached_workflow is not None and _cached_workflow[0] != name:
        _cached_workflow = None
    return gr.Textbox(value=discover_workflows().get(name, ""))  # type: ignore[return-value]


def create_app() -> Any:
    """Build and return the Gradio Blocks app."""
    workflows = discover_workflows()
    workflow_names = list(workflows.keys())
    spec_files = discover_specs()

    with gr.Blocks(title="sd-loom") as app:
        gr.Markdown("# sd-loom")

        # ── Workflow selector ──
        with gr.Group(elem_classes="group-with-padding"):
            gr.Markdown("### Workflow")
            with gr.Row():
                workflow_dd = gr.Dropdown(
                    choices=workflow_names,
                    value=None,
                    label="Workflow",
                    interactive=True,
                    filterable=False,
                    scale=1,
                )
                workflow_doc = gr.Textbox(
                    label="Description",
                    interactive=False,
                    scale=2,
                )

        # ── Spec selector ──
        with gr.Group(elem_classes="group-with-padding"):
            gr.Markdown("### Spec")
            with gr.Row():
                spec_dd = gr.Dropdown(
                    choices=spec_files,
                    value=None,
                    label="Spec",
                    interactive=True,
                    filterable=False,
                    scale=1,
                )
                spec_preview = gr.Textbox(
                    label="Fields",
                    interactive=False,
                    lines=10,
                    scale=2,
                )

        # ── Start ──
        start_btn = gr.Button("Start", variant="primary", interactive=False)
        status_box = gr.Textbox(label="Status", interactive=False, lines=5)

        # ── Events ──
        def on_spec_change(spec_path: str) -> str:
            return preview_spec(spec_path)

        def update_start_btn(workflow: str, spec: str) -> dict[str, Any]:
            enabled = bool(workflow) and bool(spec)
            return gr.Button(interactive=enabled)  # type: ignore[return-value]

        workflow_dd.change(_clear_workflow_cache, inputs=workflow_dd, outputs=workflow_doc)
        workflow_dd.change(update_start_btn, inputs=[workflow_dd, spec_dd], outputs=start_btn)

        spec_dd.change(on_spec_change, inputs=spec_dd, outputs=spec_preview)
        spec_dd.change(update_start_btn, inputs=[workflow_dd, spec_dd], outputs=start_btn)

        start_btn.click(
            _run_generation,
            inputs=[workflow_dd, spec_dd],
            outputs=status_box,
        )

    return app


def launch(share: bool = False, port: int = 7860) -> None:
    """Create and launch the Gradio app."""
    app = create_app()
    app.launch(share=share, server_port=port, theme=gr.themes.Soft(), css=_CSS)


@click.command()
@click.option("--share", is_flag=True, default=False, help="Enable Gradio public tunnel")
@click.option("--port", type=int, default=7860, help="Server port (default: 7860)")
def main(share: bool, port: int) -> None:
    """sd-loom GUI: Gradio web interface for sd-loom."""
    launch(share=share, port=port)
