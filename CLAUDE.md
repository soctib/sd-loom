# sd-loom Architecture

Everything is Python — no JSON configs, no GUI. Users define specs and workflows as code. See `README.md` for the project pitch and quick start. Licensed MIT (`LICENSE`).

## Components

### `sd_loom/core/`
- **protocol.py** — `SpecProtocol` and `Workflow` Protocols (the contracts)
- **types.py** — `Prompt` (positive + negative text), `LoomSpec` (Pydantic BaseModel, all fields required), `GenerationResult` (dataclass)
- **loader.py** — Dual-mode loader for specs and workflows. Bare names resolve to built-ins (`sd_loom.specs.*`, `sd_loom.workflows.*`); file paths are loaded dynamically.
- **resolve.py** — `resolve_model()`, `resolve_vae()`, `resolve_lora()` — fuzzy file resolution by name against `models/`, `models/vae/`, `models/sdxl/lora/`
- **cli.py** — Click CLI. Entry point: `loom WORKFLOW [ARGS]` (no subcommands — everything is a workflow)

### `sd_loom/specs/`
Built-in specs. Each module defines a single `LoomSpec` subclass; the loader finds and instantiates it automatically (class name doesn't matter). Extend `DefaultSpec` to inherit sensible defaults. User-contributed specs live anywhere on disk and are passed as file paths.
- **`__init__.py`** — `DefaultSpec(LoomSpec)` (sensible defaults for everything except `prompt`; extend this)
- Fields include `vae: str` (bare name, resolved in `models/vae/`) and `loras: list[tuple[str, float]]` (name/weight pairs, resolved in `models/sdxl/lora/`)

### `specs/` (project root)
User-contributed specs. Not part of the package. `example.py` is a starting point.

### `sd_loom/styles/`
Built-in prompt styles. Each style is a callable `_Style` instance that takes a subject string and returns a `Prompt`. SAI presets (17), Fooocus (4), photography genres (7), art styles (9), model-specific quality boosters (3) — 40 styles total.

### `sd_loom/workflows/`
Built-in workflows. Each module exports a `run()` function returning `list[GenerationResult]`. Generation workflows take `run(spec: SpecProtocol)`. Utility workflows (e.g. `info`) take other args. The CLI dispatches based on the first parameter name: `spec` → load spec, otherwise pass the raw argument. User-contributed workflows live anywhere on disk and are passed as file paths.

## Resolution Rules
- **File path** → user file: `specs/example.py` loaded dynamically
- **Bare name** → built-in: `foo` resolves to `sd_loom.specs.foo`
- Detection: contains `/`, `\`, or ends with `.py` → file path; otherwise → built-in

## Type Strictness
- mypy strict mode (`pyproject.toml`)
- Pydantic for runtime validation
- Protocols for structural typing (no inheritance required)

## CLI
```
loom sdxl specs/example.py            # SDXL generation workflow
loom debug specs/example.py           # debug workflow (prints spec, no GPU)
loom info outputs/sdxl_ldm/img.png    # show embedded metadata (utility workflow)
```

## Issue Tracking
Issues live in `./issues/` as markdown files. Naming convention:
```
{number}_{status}_{slug}.md
```
Statuses: `done`, `ready`, `blocked`, `design`, `future`
