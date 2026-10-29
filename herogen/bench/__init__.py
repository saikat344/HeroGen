"""herogen ground-truth benchmark: render a known 3D character into a run folder with exact cameras.

A run folder made here has the same layout as a stage-2 turnaround (`views/`, `masks/`, `views/meta.json`)
plus `heldout/` evaluation views, so a body built from it can be scored against the true shape
(`gt/body.glb`) on views it was never built from. See `herogen/bench/render.py` for the folder contents
and the checks.

This module and `cameras.py` / `rig.py` import only the standard library and numpy: they also run inside
Blender's bundled Python, which has no Pillow.
"""
from __future__ import annotations

# The contract frame (docs/ARCHITECTURE.md): feet at Z = 0, +Z up, facing -Y, the character's left at +X.
HEIGHT_STUDS = 5.5

# Azimuth in degrees, measured from the front (camera on -Y) toward the character's left (camera on +X).
BUILDING_AZIMUTHS = (0, 90, 180, 270, 45, 135, 225, 315)
EVALUATION_AZIMUTHS = (25, 335, 65, 295)
CARDINAL = {0: "front", 90: "left", 180: "back", 270: "right"}
ELEVATION_DEG = 0.0

DEFAULT_SIZE = 1024
BACKGROUND = (255, 255, 255)  # plain white, the background the turnaround prompts ask for


def view_name(azimuth: int) -> str:
    """`front`, `left`, `back`, `right` for the contract's four views; any other view is named by the
    nearest of front and back, the side it turns toward, and the degrees it turns away from that front or
    back view, the scheme of the pipeline's `heldout/front-left-34.png`: 25 -> `front-left-25`,
    335 -> `front-right-25`, 135 -> `back-left-45`, 225 -> `back-right-45`."""
    a = int(azimuth) % 360
    if a in CARDINAL:
        return CARDINAL[a]
    if a < 90:
        return f"front-left-{a}"
    if a < 180:
        return f"back-left-{180 - a}"
    if a < 270:
        return f"back-right-{a - 180}"
    return f"front-right-{360 - a}"
