from sd_loom.prompts import DefaultPrompt


class ExamplePrompt(DefaultPrompt):
    prompt: str = "a photo of a cat"
    negative_prompt: str = "blurry, low quality, deformed"
    model: str = "illustriousRealismBy_v10VAE"
