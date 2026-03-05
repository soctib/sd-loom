from sd_loom import styles
from sd_loom.specs import DefaultSpec, landscape, lora


class Example(DefaultSpec):
    prompt = styles.cinematic("woman, vision pro, skirt tug")
    model = "illustriousRealism"
    seed = 1
    width, height = landscape
    loras = [lora("skirt"), lora("vision_pro")]
