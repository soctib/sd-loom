from __future__ import annotations

import re
from pathlib import Path


def _normalize(name: str) -> str:
    """Strip non-alphanumeric characters and lowercase for fuzzy comparison."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _resolve(name: str, search_dir: Path, label: str) -> Path:
    """Find a file in *search_dir* matching *name*.

    Tries exact stem match first, then falls back to fuzzy matching
    (case-insensitive, ignoring non-alphanumeric characters, partial match).
    Raises ``FileNotFoundError`` if nothing matches and ``ValueError``
    if the name is ambiguous (multiple fuzzy matches).
    """
    if not search_dir.is_dir():
        raise FileNotFoundError(
            f"{label} directory not found: {search_dir}\n"
            f"Create a '{search_dir.relative_to(Path.cwd())}/' folder and place "
            f"your {label.lower()} files inside it."
        )

    all_files = [p for p in search_dir.rglob("*") if p.is_file()]

    # Exact stem match
    exact = [p for p in all_files if p.stem == name]
    if len(exact) == 1:
        return exact[0].resolve()
    if len(exact) > 1:
        paths = "\n  ".join(str(p) for p in exact)
        raise ValueError(
            f"Ambiguous {label.lower()} name '{name}' — multiple exact matches:\n  {paths}"
        )

    # Fuzzy match: normalized name must appear in normalized stem
    query = _normalize(name)
    fuzzy = [p for p in all_files if query in _normalize(p.stem)]

    if not fuzzy:
        raise FileNotFoundError(
            f"No {label.lower()} matching '{name}' found in {search_dir}"
        )
    if len(fuzzy) > 1:
        paths = "\n  ".join(str(p) for p in fuzzy)
        raise ValueError(
            f"Ambiguous {label.lower()} name '{name}' — multiple fuzzy matches:\n  {paths}"
        )

    return fuzzy[0].resolve()


def resolve_model(name: str) -> Path:
    """Find a model file in ``models/`` matching *name*."""
    return _resolve(name, Path.cwd() / "models", "Models")


def resolve_vae(name: str) -> Path:
    """Find a VAE file in ``models/vae/`` matching *name*."""
    return _resolve(name, Path.cwd() / "models" / "vae", "VAE")


def resolve_lora(name: str) -> Path:
    """Find a LoRA file in ``models/sdxl/lora/`` matching *name*."""
    return _resolve(name, Path.cwd() / "models" / "sdxl" / "lora", "LoRA")
