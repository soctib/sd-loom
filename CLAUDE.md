# sd-loom Architecture

Everything is Python — no JSON configs, no GUI. Users define prompts and workflows as code. See `README.md` for the project pitch and quick start. Licensed MIT (`LICENSE`).

## Components

### `sd_loom/core/`
- **protocol.py** — `PromptSpec` and `Workflow` Protocols (the contracts)
- **types.py** — `Prompt` (Pydantic BaseModel, all fields required), `GenerationResult` (dataclass)
- **loader.py** — Dual-mode loader for prompts and workflows. Bare names resolve to built-ins (`sd_loom.prompts.*`, `sd_loom.workflows.*`); file paths are loaded dynamically.
- **cli.py** — Click CLI. Entry point: `loom run PROMPT [--workflow NAME]`

### `sd_loom/prompts/`
Built-in prompt specs. Each module defines a single `Prompt` subclass; the loader finds and instantiates it automatically (class name doesn't matter). Extend `DefaultPrompt` to inherit sensible defaults. User-contributed prompts live anywhere on disk and are passed as file paths.
- **`__init__.py`** — `DefaultPrompt(Prompt)` (sensible defaults for everything except `prompt`; extend this)

### `prompts/` (project root)
User-contributed prompts. Not part of the package. `example.py` is a starting point.

### `sd_loom/workflows/`
Built-in workflows. Each module exports a `run(spec: PromptSpec) -> GenerationResult` function. User-contributed workflows live anywhere on disk and are passed as file paths.

## Resolution Rules
- **File path** → user file: `prompts/example.py` loaded dynamically
- **Bare name** → built-in: `foo` resolves to `sd_loom.prompts.foo`
- Detection: contains `/`, `\`, or ends with `.py` → file path; otherwise → built-in

## Type Strictness
- mypy strict mode (`pyproject.toml`)
- Pydantic for runtime validation
- Protocols for structural typing (no inheritance required)

## CLI
```
loom run prompts/example.py                        # user prompt, default workflow
loom run prompts/example.py --workflow debug           # explicit workflow
```
