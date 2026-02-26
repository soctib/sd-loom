from sd_loom.specs import DefaultSpec


class ExamplePrompt(DefaultSpec):
    prompt: str = "a photo of a cat"
    negative_prompt: str = "blurry, low quality, deformed"
    model: str = "illustriousRealismBy_v10VAE"
