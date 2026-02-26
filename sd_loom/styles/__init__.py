"""Built-in prompt styles.

SAI styles are from Stability AI's SDXL presets (Apache-2.0).
Fooocus styles are from lllyasviel/Fooocus (GPLv3).
Community styles are curated from popular CivitAI workflows.
"""

from __future__ import annotations

from sd_loom.core.types import Prompt


class _Style:
    """A reusable prompt template that wraps a subject string into a styled Prompt."""

    def __init__(self, template: str, negative: str = "") -> None:
        self.negative = negative
        self._template = template

    def __call__(self, subject: str) -> Prompt:
        return Prompt(
            positive=self._template.format(prompt=subject),
            negative=self.negative,
        )

    def __repr__(self) -> str:
        return f"_Style({self._template!r})"


# ---------------------------------------------------------------------------
# SAI / Stability AI SDXL presets (Apache-2.0)
# ---------------------------------------------------------------------------

enhance = _Style(
    template="breathtaking {prompt}. award-winning, professional, highly detailed",
    negative="ugly, deformed, noisy, blurry, distorted, grainy",
)

cinematic = _Style(
    template="cinematic film still {prompt}. shallow depth of field, vignette, "
    "highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, "
    "film grain, grainy",
    negative="anime, cartoon, graphic, text, painting, crayon, graphite, abstract, "
    "glitch, deformed, mutated, ugly, disfigured",
)

photographic = _Style(
    template="cinematic photo {prompt}. 35mm photograph, film, bokeh, "
    "professional, 4k, highly detailed",
    negative="drawing, painting, crayon, sketch, graphite, impressionist, noisy, "
    "blurry, soft, deformed, ugly",
)

anime = _Style(
    template="anime artwork {prompt}. anime style, key visual, vibrant, "
    "studio anime, highly detailed",
    negative="photo, deformed, black and white, realism, disfigured, low contrast",
)

digital_art = _Style(
    template="concept art {prompt}. digital artwork, illustrative, painterly, "
    "matte painting, highly detailed",
    negative="photo, photorealistic, realism, ugly",
)

pixel_art = _Style(
    template="pixel-art {prompt}. low-res, blocky, pixel art style, 8-bit graphics",
    negative="sloppy, messy, blurry, noisy, highly detailed, ultra textured, "
    "photo, realistic",
)

fantasy = _Style(
    template="ethereal fantasy concept art of {prompt}. magnificent, celestial, "
    "ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
    negative="photographic, realistic, realism, 35mm film, dslr, cropped, frame, "
    "text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, "
    "closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, "
    "black and white",
)

three_d_model = _Style(
    template="professional 3d model {prompt}. octane render, highly detailed, "
    "volumetric, dramatic lighting",
    negative="ugly, deformed, noisy, low poly, blurry, painting",
)

analog_film = _Style(
    template="analog film photo {prompt}. faded film, desaturated, 35mm photo, "
    "grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, "
    "found footage",
    negative="painting, drawing, illustration, glitch, deformed, mutated, "
    "cross-eyed, ugly, disfigured",
)

comic_book = _Style(
    template="comic {prompt}. graphic illustration, comic art, graphic novel art, "
    "vibrant, highly detailed",
    negative="photograph, deformed, glitch, noisy, realistic, stock photo",
)

craft_clay = _Style(
    template="play-doh style {prompt}. sculpture, clay art, centered composition, "
    "Claymation",
    negative="sloppy, messy, grainy, highly detailed, ultra textured, photo",
)

isometric = _Style(
    template="isometric style {prompt}. vibrant, beautiful, crisp, detailed, "
    "ultra detailed, intricate",
    negative="deformed, mutated, ugly, disfigured, blur, blurry, noise, noisy, "
    "realistic, photographic",
)

line_art = _Style(
    template="line art drawing {prompt}. professional, sleek, modern, minimalist, "
    "graphic, line art, vector graphics",
    negative="anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, "
    "off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, "
    "mutated, realism, realistic, impressionism, expressionism, oil, acrylic",
)

low_poly = _Style(
    template="low-poly style {prompt}. low-poly game art, polygon mesh, jagged, "
    "blocky, wireframe edges, centered composition",
    negative="noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
)

neon_punk = _Style(
    template="neonpunk style {prompt}. cyberpunk, vaporwave, neon, vibes, vibrant, "
    "stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, "
    "dark purple shadows, high contrast, cinematic, ultra detailed, intricate, "
    "professional",
    negative="painting, drawing, illustration, glitch, deformed, mutated, "
    "cross-eyed, ugly, disfigured",
)

origami = _Style(
    template="origami style {prompt}. paper art, pleated paper, folded, origami art, "
    "pleats, cut and fold, centered composition",
    negative="noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
)

texture = _Style(
    template="texture {prompt} top down close-up",
    negative="ugly, deformed, noisy, blurry",
)

# ---------------------------------------------------------------------------
# Fooocus community styles (GPLv3, lllyasviel/Fooocus)
# ---------------------------------------------------------------------------

fooocus_sharp = _Style(
    template="cinematic still {prompt}. emotional, harmonious, vignette, 4k epic "
    "detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, "
    "moody, epic, gorgeous, film grain, grainy",
    negative="anime, cartoon, graphic, blur, blurry, bokeh, text, painting, crayon, "
    "graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
)

fooocus_masterpiece = _Style(
    template="(masterpiece), (best quality), (ultra-detailed), {prompt}, "
    "illustration, detailed eyes, perfect composition, intricate details",
    negative="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality",
)

fooocus_photograph = _Style(
    template="photograph {prompt}, 50mm. cinematic 4k epic detailed photograph "
    "shot on kodak detailed cinematic, 35mm photo, grainy, vignette, vintage, "
    "Kodachrome, Lomography, stained, highly detailed",
    negative="bokeh, depth of field, blurry, cropped, semi-realistic, cgi, 3d, "
    "render, sketch, cartoon, drawing, anime, text, cropped, out of frame, "
    "worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid",
)

fooocus_cinematic = _Style(
    template="cinematic still {prompt}. emotional, harmonious, vignette, "
    "highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, "
    "film grain, grainy",
    negative="anime, cartoon, graphic, text, painting, crayon, graphite, abstract, "
    "glitch, deformed, mutated, ugly, disfigured",
)

# ---------------------------------------------------------------------------
# Community styles — photography genres
# ---------------------------------------------------------------------------

portrait = _Style(
    template="professional portrait photograph of {prompt}. natural lighting, "
    "shallow depth of field, bokeh, 85mm lens, sharp focus on eyes, "
    "professional, 4k, highly detailed",
    negative="cartoon, illustration, anime, painting, drawing, deformed, ugly, "
    "blurry, bad anatomy, disfigured, poorly drawn face, mutation, extra limbs",
)

landscape = _Style(
    template="breathtaking landscape photograph of {prompt}. golden hour, "
    "dramatic sky, wide angle, sharp focus, vivid colors, professional photography, "
    "4k, highly detailed",
    negative="cartoon, illustration, anime, painting, drawing, blurry, ugly, "
    "deformed, noisy, low quality",
)

street_photo = _Style(
    template="street photography of {prompt}. candid, raw, authentic, 35mm film, "
    "urban, natural lighting, documentary style, gritty, high contrast",
    negative="studio, posed, cartoon, illustration, anime, painting, drawing, "
    "blurry, deformed, ugly",
)

macro = _Style(
    template="macro photography of {prompt}. extreme close-up, shallow depth of "
    "field, sharp focus on subject, bokeh background, highly detailed, "
    "professional, stunning detail",
    negative="cartoon, illustration, anime, painting, drawing, blurry, ugly, "
    "deformed, noisy, low quality, out of focus",
)

fashion = _Style(
    template="high fashion editorial photograph of {prompt}. studio lighting, "
    "vogue style, glamorous, professional makeup, high-end, luxury, sharp focus, "
    "4k, highly detailed",
    negative="casual, candid, cartoon, illustration, anime, painting, deformed, "
    "ugly, blurry, bad anatomy, low quality",
)

food = _Style(
    template="professional food photography of {prompt}. appetizing, styled plating, "
    "soft natural lighting, shallow depth of field, 4k, highly detailed, "
    "restaurant quality",
    negative="cartoon, illustration, anime, painting, ugly, deformed, blurry, "
    "low quality, unappetizing",
)

product = _Style(
    template="professional product photography of {prompt}. studio lighting, "
    "clean background, commercial, sharp focus, high-end, 4k, highly detailed",
    negative="cartoon, illustration, anime, painting, deformed, ugly, blurry, "
    "low quality, messy background",
)

# ---------------------------------------------------------------------------
# Community styles — art & illustration
# ---------------------------------------------------------------------------

oil_painting = _Style(
    template="oil painting of {prompt}. thick brushstrokes, rich colors, textured "
    "canvas, masterful technique, gallery quality, fine art",
    negative="photo, photorealistic, digital, blurry, ugly, deformed, low quality",
)

watercolor = _Style(
    template="watercolor painting of {prompt}. soft washes, delicate, translucent "
    "colors, wet-on-wet technique, flowing, artistic, fine art",
    negative="photo, photorealistic, digital, sharp, heavy lines, ugly, deformed, "
    "low quality",
)

ink = _Style(
    template="ink drawing of {prompt}. black ink on white paper, detailed linework, "
    "cross-hatching, pen and ink illustration, high contrast, artistic",
    negative="color, photo, photorealistic, blurry, ugly, deformed, low quality",
)

art_nouveau = _Style(
    template="art nouveau illustration of {prompt}. ornate, flowing organic lines, "
    "decorative, Alphonse Mucha style, floral motifs, elegant, detailed",
    negative="photo, photorealistic, modern, minimalist, ugly, deformed, low quality",
)

pop_art = _Style(
    template="pop art of {prompt}. bold colors, Ben-Day dots, high contrast, "
    "graphic, Andy Warhol style, Roy Lichtenstein, vibrant, stylized",
    negative="photo, photorealistic, muted, dull, ugly, deformed, realistic, "
    "low quality",
)

gothic = _Style(
    template="dark gothic artwork of {prompt}. brooding atmosphere, ornate details, "
    "dramatic chiaroscuro, dark romanticism, macabre beauty, highly detailed",
    negative="bright, cheerful, cartoon, cute, blurry, low quality, deformed",
)

steampunk = _Style(
    template="steampunk artwork of {prompt}. Victorian era, brass gears, clockwork "
    "mechanisms, steam-powered, copper pipes, industrial, ornate, highly detailed",
    negative="modern, futuristic, plastic, cartoon, blurry, ugly, deformed, "
    "low quality",
)

dark_fantasy = _Style(
    template="dark fantasy art of {prompt}. ominous, foreboding, grim atmosphere, "
    "epic, highly detailed, dramatic lighting, dark color palette, "
    "professional fantasy illustration",
    negative="bright, cheerful, cute, cartoon, photo, photorealistic, blurry, "
    "ugly, deformed, low quality",
)

manga = _Style(
    template="manga artwork {prompt}. black and white, screentone shading, "
    "dynamic composition, expressive, japanese manga style, highly detailed",
    negative="photo, realistic, color, western comic, blurry, ugly, deformed, "
    "low quality",
)

# ---------------------------------------------------------------------------
# Community styles — model-specific quality boosters
# ---------------------------------------------------------------------------

realistic = _Style(
    template="{prompt}, RAW photo, 8k uhd, dslr, high quality, film grain, "
    "Fujifilm XT3",
    negative="worst quality, low quality, illustration, 3d, 2d, painting, cartoons, "
    "sketch, open mouth",
)

illustrious = _Style(
    template="masterpiece, best quality, {prompt}, absurdres, highly detailed",
    negative="worst quality, bad quality, low quality, lowres, displeasing, "
    "very displeasing, bad anatomy, bad hands, scan artifacts, monochrome, "
    "greyscale, jpeg artifacts, sketch, extra digits, fewer digits",
)

pony = _Style(
    template="score_9, score_8_up, score_7_up, {prompt}",
    negative="score_6, score_5, score_4, worst quality, low quality, "
    "bad anatomy, bad hands",
)
