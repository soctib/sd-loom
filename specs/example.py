from sd_loom import styles
from sd_loom.specs import DefaultSpec


class ExamplePrompt(DefaultSpec):
    prompt = styles.cinematic("woman, vision pro, skirt tug")
    model = "illustriousRealismBy_v10VAE"
    seed = 1
    loras = ["skirt"]
