from sd_loom import styles
from sd_loom.specs import DefaultSpec

specs = [
    DefaultSpec(
        prompt=s("a cat in a garden"),
        model="illustriousRealism",
        seed=1,
        tag=s.name,
    )
    for s in [styles.cinematic, styles.photographic, styles.anime, styles.fantasy,
              styles.oil_painting, styles.watercolor, styles.pixel_art]
]
