# sd-loom

Stable Diffusion image generation where everything is Python — no JSON configs, no GUIs, no node graphs. Prompts and workflows are plain Python modules that you write, version, and share like any other code.

## Why

Most SD frontends bury settings in JSON files or drag-and-drop UIs that are hard to diff, review, or automate. sd-loom treats generation as a programmable pipeline: define what you want in a prompt spec, wire it through a workflow, and run it from the CLI.

## Quick start

```bash
pip install -e ".[dev]"
loom run sdxl prompts/example.py
```

## CLI

### Generate images

```bash
loom run sdxl prompts/example.py                    # SDXL workflow, user prompt
loom run debug prompts/example.py                   # debug workflow (prints spec, no GPU)
```

### Override prompt fields

Any field in the prompt spec can be overridden from the CLI with `--set`:

```bash
loom run sdxl prompts/example.py --set seed=42
loom run sdxl prompts/example.py --set steps=50 --set cfg_scale=5.0
loom run sdxl prompts/example.py --set vram=high
```

### Batch generation

Generate multiple images in one run. Batching is automatic based on your VRAM profile (low=1, medium=2, high=4 images per forward pass):

```bash
loom run sdxl prompts/example.py --count 4
loom run sdxl prompts/example.py --count 4 --set vram=high   # all 4 in one pass
```

Seeds are sequential from the base seed (random or explicit).

### Inspect metadata

Every generated PNG embeds its full generation parameters:

```bash
loom info outputs/sdxl/example_20260226_120000_42.png
```

### VRAM profiles

Control GPU memory usage with `--set vram=PROFILE`:

| Profile  | Behavior |
|----------|----------|
| `low`    | CPU offload + VAE tiling (~8GB) |
| `medium` | Full GPU + VAE tiling (~12GB) |
| `high`   | Full GPU, no tiling (~16GB+) |

### Model resolution

Models are resolved by fuzzy match against your local `.safetensors` files — case-insensitive, partial match:

```bash
loom run sdxl prompts/example.py --set model=illustrious
```

## Prompts

Prompts are Python modules that define a subclass of `DefaultPrompt`:

```python
from sd_loom.prompts import DefaultPrompt

class MyPrompt(DefaultPrompt):
    prompt = "a photo of a cat"
    negative_prompt = "blurry, low quality"
    model = "illustriousRealismBy_v10VAE"
```

`DefaultPrompt` provides sensible defaults for everything except `prompt`. See `prompts/example.py` for a starting point.

## Status

Early development. Targeting SDXL first.

## License

MIT
