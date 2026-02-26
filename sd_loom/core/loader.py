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
    return "/" in name_or_path or "\\" in name_or_path or name_or_path.endswith(".py")


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


def load_spec(
    name_or_path: str,
    *,
    overrides: tuple[str, ...] = (),
) -> SpecProtocol:
    """Load a spec module by name (built-in) or file path (user-contributed).

    The module must define exactly one ``LoomSpec`` subclass. The loader
    finds it automatically and instantiates it.
    """
    if _is_file_path(name_or_path):
        module = _load_module_from_file(name_or_path, "sd_loom.user_specs")
    else:
        module = _load_builtin_module("sd_loom.specs", name_or_path)

    cls = _find_spec_class(module, name_or_path)
    instance: SpecProtocol = cls()

    if overrides:
        from pydantic import BaseModel

        if not isinstance(instance, BaseModel):
            raise TypeError("Spec must be a Pydantic BaseModel")
        parsed = _parse_overrides(overrides)
        instance = type(instance).model_validate({**instance.model_dump(), **parsed})

    return instance


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
