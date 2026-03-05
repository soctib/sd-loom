from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from types import ModuleType

    from sd_loom.core.protocol import SpecProtocol


def _is_file_path(name_or_path: str) -> bool:
    """Return True if the argument looks like a file path rather than a bare module name."""
    return (
        "/" in name_or_path
        or "\\" in name_or_path
        or name_or_path.endswith((".py", ".json"))
    )


def _load_module_from_file(path: str, prefix: str) -> ModuleType:
    """Dynamically import a Python file as a module."""
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Module not found: {file_path}")

    module_name = f"{prefix}.{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_builtin_module(package: str, name: str) -> ModuleType:
    """Import a built-in module from the given package."""
    module_name = f"{package}.{name}"
    return importlib.import_module(module_name)


def _find_spec_class(module: ModuleType, name_or_path: str) -> type[Any]:
    """Find the single LoomSpec subclass defined in a module."""
    from sd_loom.core.types import LoomSpec

    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, LoomSpec) and obj.__module__ == module.__name__
    ]

    if not candidates:
        raise AttributeError(
            f"Spec module '{name_or_path}' must define a LoomSpec subclass"
        )
    if len(candidates) > 1:
        names = ", ".join(c.__name__ for c in candidates)
        raise AttributeError(
            f"Spec module '{name_or_path}' has multiple LoomSpec subclasses ({names});"
            " it must define exactly one"
        )

    return candidates[0]


def _parse_overrides(overrides: tuple[str, ...]) -> dict[str, str]:
    """Parse ``key=value`` pairs into a dict."""
    result: dict[str, str] = {}
    for item in overrides:
        if "=" not in item:
            raise click.BadParameter(
                f"Override must be key=value, got: {item!r}", param_hint="'--set'"
            )
        key, value = item.split("=", 1)
        result[key] = value
    return result


def _load_json_spec(path: Path) -> SpecProtocol:
    """Load a spec from a JSON file, filling missing fields from DefaultSpec."""
    import json

    from sd_loom.specs import DefaultSpec

    data = json.loads(path.read_text())
    # Create a named subclass so save_image picks up the file stem as spec name
    spec_name = path.stem
    cls = type(spec_name, (DefaultSpec,), {"__module__": f"sd_loom.user_specs.{spec_name}"})
    instance: SpecProtocol = cls.model_validate(data)  # type: ignore[attr-defined]
    return instance


def _load_file_spec(path: Path) -> SpecProtocol:
    """Create a spec from a file path (image, safetensors, etc.).

    Sets ``input_image`` to the resolved path. All other fields use defaults.
    """
    from sd_loom.specs import DefaultSpec

    resolved = str(path.resolve())
    spec_name = path.stem
    cls = type(spec_name, (DefaultSpec,), {"__module__": f"sd_loom.user_specs.{spec_name}"})
    instance: SpecProtocol = cls(input_image=resolved)
    return instance


def _find_specs_from_module(
    module: ModuleType, name_or_path: str
) -> list[SpecProtocol]:
    """Return specs from a module: a ``specs`` list or a single LoomSpec subclass."""
    specs_list = getattr(module, "specs", None)
    if specs_list is not None and isinstance(specs_list, list):
        return list(specs_list)
    cls = _find_spec_class(module, name_or_path)
    return [cls()]


def load_spec(
    name_or_path: str,
    *,
    overrides: tuple[str, ...] = (),
) -> list[SpecProtocol]:
    """Load specs by name (built-in), file path (.py/.json), or input file.

    Returns a list — a module may export multiple specs via a ``specs`` list.

    - ``.py`` files must define a ``LoomSpec`` subclass or a ``specs`` list.
    - ``.json`` files are loaded as data, missing fields filled from ``DefaultSpec``.
    - Other file paths (images, safetensors, etc.) become a ``DefaultSpec`` with
      ``input_image`` set to the resolved path.
    - Bare names resolve to built-in specs in ``sd_loom.specs.*``.
    """
    file_path = Path(name_or_path)

    if file_path.suffix == ".json":
        instances = [_load_json_spec(file_path.resolve())]
    elif file_path.suffix == ".py" and _is_file_path(name_or_path):
        module = _load_module_from_file(name_or_path, "sd_loom.user_specs")
        instances = _find_specs_from_module(module, name_or_path)
    elif _is_file_path(name_or_path):
        instances = [_load_file_spec(file_path)]
    else:
        module = _load_builtin_module("sd_loom.specs", name_or_path)
        instances = _find_specs_from_module(module, name_or_path)

    if overrides:
        from pydantic import BaseModel

        parsed = _parse_overrides(overrides)
        instances = [
            type(inst).model_validate({**inst.model_dump(), **parsed})
            if isinstance(inst, BaseModel)
            else inst
            for inst in instances
        ]

    return instances


def load_workflow(name_or_path: str) -> Any:
    """Load a workflow module by name (built-in) or file path (user-contributed).

    The module must export a ``run`` callable.
    """
    if _is_file_path(name_or_path):
        module = _load_module_from_file(name_or_path, "sd_loom.user_workflows")
    else:
        module = _load_builtin_module("sd_loom.workflows", name_or_path)

    if not hasattr(module, "run"):
        raise AttributeError(
            f"Workflow module '{name_or_path}' must export a 'run' function"
        )

    return module
