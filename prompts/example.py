from sd_loom.core.types import Prompt
from sd_loom.specs import DefaultSpec


class ExamplePrompt(DefaultSpec):
    prompt: Prompt = Prompt(
        positive="a photo of a cat",
        negative="blurry, low quality, deformed",
    )
    model: str = "illustriousRealismBy_v10VAE"
