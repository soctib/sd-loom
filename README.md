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

## Specs

Specs are Python modules that define a subclass of `DefaultSpec`:

```python
from sd_loom.core.types import Prompt
from sd_loom.specs import DefaultSpec

class MySpec(DefaultSpec):
    prompt = Prompt(positive="a photo of a cat", negative="blurry, low quality")
    model = "illustriousRealismBy_v10VAE"
```

`DefaultSpec` provides sensible defaults for everything except `prompt`. See `prompts/example.py` for a starting point.

## Styles

Built-in styles wrap a subject string into a styled `Prompt` with curated positive/negative templates:

```python
from sd_loom.specs import DefaultSpec
from sd_loom.styles import cinematic

class MySpec(DefaultSpec):
    prompt = cinematic("a photo of a cat")
    model = "illustriousRealismBy_v10VAE"
```

Available styles:

| Category | Styles |
|----------|--------|
| **SAI presets** | `enhance`, `cinematic`, `photographic`, `anime`, `digital_art`, `pixel_art`, `fantasy`, `three_d_model`, `analog_film`, `comic_book`, `craft_clay`, `isometric`, `line_art`, `low_poly`, `neon_punk`, `origami`, `texture` |
| **Fooocus** | `fooocus_sharp`, `fooocus_masterpiece`, `fooocus_photograph`, `fooocus_cinematic` |
| **Photography** | `portrait`, `landscape`, `street_photo`, `macro`, `fashion`, `food`, `product` |
| **Art** | `oil_painting`, `watercolor`, `ink`, `art_nouveau`, `pop_art`, `gothic`, `steampunk`, `dark_fantasy`, `manga` |
| **Model-specific** | `realistic` (RealVisXL etc.), `illustrious` (Illustrious XL), `pony` (Pony Diffusion) |

## Status

Early development. Targeting SDXL first.

## License

MIT
