from __future__ import annotations

import re
from pathlib import Path


def _normalize(name: str) -> str:
    """Strip non-alphanumeric characters and lowercase for fuzzy comparison."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_model(name: str) -> Path:
    """Find a model file in ``models/`` matching *name*.

    Tries exact stem match first, then falls back to fuzzy matching
    (case-insensitive, ignoring non-alphanumeric characters, partial match).
    Raises ``FileNotFoundError`` if nothing matches and ``ValueError``
    if the name is ambiguous (multiple fuzzy matches).
    """
    models_dir = Path.cwd() / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(
            f"Models directory not found: {models_dir}\n"
            "Create a 'models/' folder and place your checkpoints inside it."
        )

    all_files = [p for p in models_dir.rglob("*") if p.is_file()]

    # Exact stem match
    exact = [p for p in all_files if p.stem == name]
    if len(exact) == 1:
        return exact[0].resolve()
    if len(exact) > 1:
        paths = "\n  ".join(str(p) for p in exact)
        raise ValueError(
            f"Ambiguous model name '{name}' — multiple exact matches:\n  {paths}"
        )

    # Fuzzy match: normalized name must appear in normalized stem
    query = _normalize(name)
    fuzzy = [p for p in all_files if query in _normalize(p.stem)]

    if not fuzzy:
        raise FileNotFoundError(
            f"No model matching '{name}' found in {models_dir}"
        )
    if len(fuzzy) > 1:
        paths = "\n  ".join(str(p) for p in fuzzy)
        raise ValueError(
            f"Ambiguous model name '{name}' — multiple fuzzy matches:\n  {paths}"
        )

    return fuzzy[0].resolve()
