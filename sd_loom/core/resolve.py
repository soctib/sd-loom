from __future__ import annotations

from pathlib import Path


def resolve_model(name: str) -> Path:
    """Find a model file in ``models/`` whose stem matches *name*.

    Searches recursively from the current working directory's ``models/``
    folder.  Raises ``FileNotFoundError`` if nothing matches and ``ValueError``
    if the name is ambiguous (multiple files with the same stem).
    """
    models_dir = Path.cwd() / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(
            f"Models directory not found: {models_dir}\n"
            "Create a 'models/' folder and place your checkpoints inside it."
        )

    matches = [p for p in models_dir.rglob("*") if p.is_file() and p.stem == name]

    if not matches:
        raise FileNotFoundError(
            f"No model file with stem '{name}' found in {models_dir}"
        )
    if len(matches) > 1:
        paths = "\n  ".join(str(p) for p in matches)
        raise ValueError(
            f"Ambiguous model name '{name}' — multiple matches:\n  {paths}"
        )

    return matches[0].resolve()
