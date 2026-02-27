from sd_loom import styles
from sd_loom.core.types import Prompt
from sd_loom.specs import DefaultSpec


class ExamplePrompt(DefaultSpec):
    prompt: Prompt = styles.cinematic("woman, vision pro, skirt tug")
    model: str = "illustriousRealismBy_v10VAE"
    seed: int = 1
    loras: list[tuple[str, float]] = [
        ("skirt", 1)
    ]
