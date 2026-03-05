from sd_loom import styles
from sd_loom.specs import DefaultSpec

specs = [
    DefaultSpec(
        prompt=styles.cinematic("a cat in a garden"),
        model="illustriousRealism",
        cfg_scale=cfg,
        steps=steps,
        seed=1,
        tag=f"cfg{cfg}_steps{steps}",
    )
    for cfg in [3.0, 5.0, 7.0, 10.0]
    for steps in [15, 25, 40]
]
