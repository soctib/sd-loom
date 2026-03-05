"""SDXL txt2img — alias for sdxl_ldm (k-diffusion + ldm UNet, Forge-parity)."""
from sd_loom.workflows.sdxl_ldm import run

__all__ = ["run"]
