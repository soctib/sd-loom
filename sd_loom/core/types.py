from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 — Pydantic needs this at runtime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, field_validator

if TYPE_CHECKING:
    from PIL import Image


class _SpecMeta(type(BaseModel)):  # type: ignore[misc]
    """Auto-annotate fields overridden without type annotations in subclasses.

    Pydantic v2 requires annotations on all field overrides. This metaclass
    copies the parent's annotation so user specs can simply write
    ``prompt = "a cat"`` instead of ``prompt: Prompt = "a cat"``.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, object],
        **kwargs: object,
    ) -> _SpecMeta:
        parent_fields: dict[str, object] = {}
        for base in bases:
            for field_name, field_info in getattr(base, "model_fields", {}).items():
                parent_fields[field_name] = field_info.annotation

        annotations: dict[str, object] = namespace.get("__annotations__", {})  # type: ignore[assignment]
        for field_name, annotation in parent_fields.items():
            if field_name in namespace and field_name not in annotations:
                if "__annotations__" not in namespace:
                    namespace["__annotations__"] = {}
                namespace["__annotations__"][field_name] = annotation  # type: ignore[index]

        return super().__new__(mcs, name, bases, namespace, **kwargs)  # type: ignore[no-any-return]


class Prompt(BaseModel):
    positive: str
    negative: str = "blurry, low quality, deformed, ugly, worst quality"


class LoomSpec(BaseModel, metaclass=_SpecMeta):
    model_config = ConfigDict(validate_default=True)

    prompt: Prompt
    model: str
    width: int
    height: int
    steps: int
    cfg_scale: float
    seed: int
    count: int
    scheduler: str
    clip_skip: int
    model_hash: str
    vae: str
    loras: list[tuple[str, float]]
    vram: str
    rng: str
    output_dir: str | Path
    input_image: str
    tag: str

    @field_validator("prompt", mode="before")
    @classmethod
    def _coerce_prompt(cls, v: object) -> object:
        """Allow plain strings: ``prompt = "a cat"`` → ``Prompt(positive="a cat")``."""
        if isinstance(v, str):
            return Prompt(positive=v)
        return v

    @field_validator("loras", mode="before")
    @classmethod
    def _coerce_loras(cls, v: object) -> object:
        """Allow bare strings: ``loras = ["detail"]`` → ``[("detail", 1.0)]``."""
        if not isinstance(v, list):
            return v
        return [(item, 1.0) if isinstance(item, str) else item for item in v]


@dataclass
class LoomData:
    image: Image.Image | None = None
    seed: int = 0
    elapsed_seconds: float = 0.0
    workflow: str = ""
    text: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
