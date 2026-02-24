# sd-loom

Stable Diffusion image generation where everything is Python — no JSON configs, no GUIs, no node graphs. Prompts and workflows are plain Python modules that you write, version, and share like any other code.

## Why

Most SD frontends bury settings in JSON files or drag-and-drop UIs that are hard to diff, review, or automate. sd-loom treats generation as a programmable pipeline: define what you want in a prompt spec, wire it through a workflow, and run it from the CLI.

## Quick start

```bash
pip install -e ".[dev]"
loom run prompts/example.py
```

Specify a different workflow:

```bash
loom run prompts/example.py --workflow custom_workflow
```

## Status

Early development. Targeting SDXL first.

## License

MIT
