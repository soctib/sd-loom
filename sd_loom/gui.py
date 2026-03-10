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
_CSS = "\n".join([
    "h3 { margin: 0.75rem 0 0.5rem 0.5rem !important; }",
    ".spec-preview textarea { font-family: monospace !important; font-size: 0.85em !important; }",
])

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
    """Load a spec and return its fields as formatted JSON (all specs shown)."""
    if not spec_path:
        return ""
    try:
        specs = load_spec(spec_path)
        if not specs:
            return "(empty)"
        from pydantic import BaseModel

        items = [
            spec.model_dump() if isinstance(spec, BaseModel) else vars(spec)
            for spec in specs
        ]
        data: Any = items[0] if len(items) == 1 else items
        return json.dumps(data, indent=2, default=str)
    except Exception as exc:
        return f"Error loading spec: {exc}"


_HISTORY_LIMIT = 200


def _output_dir(workflow: str, spec_path: str) -> Path | None:
    """Return the output directory for a workflow + spec combination, or None."""
    if not workflow or not spec_path:
        return None
    spec_stem = Path(spec_path).stem
    d = Path("outputs") / workflow / spec_stem
    return d if d.is_dir() else None


def load_history(workflow: str, spec_path: str) -> list[str]:
    """Return the most recent image paths for a workflow + spec (newest first)."""
    d = _output_dir(workflow, spec_path)
    if d is None:
        return []
    pngs = sorted(d.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in pngs[:_HISTORY_LIMIT]]


# ── App ────────────────────────────────────────────────────────────────────

# Workflow instance cache — reused across runs, cleared on workflow switch.
_cached_workflow: tuple[str, Any] | None = None


def _run_generation(
    workflow_name: str,
    spec_path: str,
    count: int,
    progress: gr.Progress = gr.Progress(),  # noqa: B008
) -> tuple[str, list[Any]]:
    """Run the workflow. Returns (status_text, images)."""
    global _cached_workflow  # noqa: PLW0603

    if not workflow_name or not spec_path:
        return "Select both a workflow and a spec first.", []

    from datetime import UTC, datetime

    from sd_loom.core.cli import _expand_count
    from sd_loom.core.save import save_image

    try:
        specs = load_spec(spec_path)
        if count > 1:
            specs = _expand_count(specs, count)
        # Reuse cached workflow instance if same name (preserves model cache).
        if _cached_workflow is not None and _cached_workflow[0] == workflow_name:
            workflow = _cached_workflow[1]
        else:
            workflow = load_workflow(workflow_name)
            _cached_workflow = (workflow_name, workflow)
    except Exception as exc:
        return f"Error: {exc}", []

    run_timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    spec_stem = Path(spec_path).stem

    saved: list[str] = []
    images: list[Any] = []
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
                    caption = f"seed={r.seed}, {r.elapsed_seconds:.1f}s"
                    saved.append(f"{path} ({caption})")
                    images.append((r.image, caption))
    except SystemExit as exc:
        return f"Error: {exc}", images
    except Exception as exc:
        return f"Error during generation: {exc}", images

    parts: list[str] = []
    if texts:
        parts.append("\n".join(texts))
    if saved:
        parts.append(f"Saved {len(saved)} image(s):")
        parts.extend(f"  {s}" for s in saved)
    status = "\n".join(parts) if parts else "Done (no output)."
    return status, images


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
        # Browser-persisted state for selections.
        saved_workflow = gr.BrowserState(None, storage_key="loom_workflow")
        saved_spec = gr.BrowserState(None, storage_key="loom_spec")
        saved_count = gr.BrowserState(1, storage_key="loom_count")

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
                with gr.Column(scale=1):
                    spec_dd = gr.Dropdown(
                        choices=spec_files,
                        value=None,
                        label="Spec",
                        interactive=True,
                        filterable=False,
                    )
                    count_spinner = gr.Number(
                        value=1,
                        label="Count",
                        minimum=1,
                        maximum=100,
                        step=1,
                        precision=0,
                        interactive=True,
                    )
                spec_preview = gr.Textbox(
                    label="Fields",
                    interactive=False,
                    lines=12,
                    scale=2,
                    elem_classes="spec-preview",
                )

        # ── Start ──
        start_btn = gr.Button("Start", variant="primary", interactive=False)
        status_box = gr.Textbox(label="Status", interactive=False, lines=5)
        output_gallery = gr.Gallery(
            label="Output", columns=4, visible=False, object_fit="contain",
        )

        # ── Events ──
        def on_spec_change(spec_path: str) -> str:
            return preview_spec(spec_path)

        def update_start_btn(workflow: str, spec: str) -> dict[str, Any]:
            enabled = bool(workflow) and bool(spec)
            return gr.Button(interactive=enabled)  # type: ignore[return-value]

        def update_history(workflow: str, spec: str) -> Any:
            history = load_history(workflow, spec)
            return gr.Gallery(value=history, visible=bool(history))

        # Save selections to browser state on change.
        workflow_dd.change(_clear_workflow_cache, inputs=workflow_dd, outputs=workflow_doc)
        workflow_dd.change(update_start_btn, inputs=[workflow_dd, spec_dd], outputs=start_btn)
        workflow_dd.change(lambda v: v, inputs=workflow_dd, outputs=saved_workflow)
        workflow_dd.change(update_history, inputs=[workflow_dd, spec_dd], outputs=output_gallery)

        spec_dd.change(on_spec_change, inputs=spec_dd, outputs=spec_preview)
        spec_dd.change(update_start_btn, inputs=[workflow_dd, spec_dd], outputs=start_btn)
        spec_dd.change(lambda v: v, inputs=spec_dd, outputs=saved_spec)
        spec_dd.change(update_history, inputs=[workflow_dd, spec_dd], outputs=output_gallery)

        count_spinner.change(lambda v: v, inputs=count_spinner, outputs=saved_count)

        # Restore selections from browser state on page load.
        def restore(
            wf: str | None, sp: str | None, cnt: float | None,
        ) -> tuple[Any, ...]:
            doc = discover_workflows().get(wf, "") if wf else ""
            preview = preview_spec(sp) if sp else ""
            enabled = bool(wf) and bool(sp)
            history = load_history(wf or "", sp or "")
            return (
                gr.Dropdown(value=wf),
                gr.Textbox(value=doc),
                gr.Dropdown(value=sp),
                preview,
                gr.Number(value=cnt or 1),
                gr.Button(interactive=enabled),
                gr.Gallery(value=history, visible=bool(history)),
            )

        app.load(
            restore,
            inputs=[saved_workflow, saved_spec, saved_count],
            outputs=[workflow_dd, workflow_doc, spec_dd, spec_preview,
                     count_spinner, start_btn, output_gallery],
        )

        def run_and_show(
            wf: str, sp: str, cnt: int,
            progress: gr.Progress = gr.Progress(),  # noqa: B008
        ) -> tuple[str, Any]:
            status, _new_images = _run_generation(wf, sp, cnt, progress)
            # Reload history from disk (includes the just-saved images).
            history = load_history(wf, sp)
            return status, gr.Gallery(value=history, visible=bool(history))

        start_btn.click(
            run_and_show,
            inputs=[workflow_dd, spec_dd, count_spinner],
            outputs=[status_box, output_gallery],
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
