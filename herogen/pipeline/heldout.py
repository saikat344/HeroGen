"""Held-out 3/4 view (stage 2h): one extra image of the character that no building stage reads.

Written after the turnaround passes: `heldout/front-left-34.png` and `heldout/front-left-34-mask.png`, on the
same square canvas as `views/` and at the shared pixel scale, plus the record `stage-2h.json`. Only scoring
reads `heldout/` (see docs/ARCHITECTURE.md, "Additions").

Scale: the image is an edit of the cover that keeps the cover's framing, so cover pixels are mapped to view
pixels by front-view figure height / cover figure height. The figure's own height is not used to scale it,
which keeps the height check a measurement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from herogen.pipeline import checks as C
from herogen.pipeline import imaging
from herogen.pipeline.client import IMAGE_MODEL, ModelClient
from herogen.pipeline.runfolder import RunFolder
from herogen.pipeline.stages import _load, _sam_or_threshold, _Stage, method_kind

VIEW = "front-left-34"
IMAGE_REL = f"heldout/{VIEW}.png"
MASK_REL = f"heldout/{VIEW}-mask.png"
RAW_REL = f"heldout/{VIEW}-raw.png"
ATTEMPTS = 3
# Orthographic views of one figure at one scale have equal heights; the same 0.95 as the turnaround sheet.
HEIGHT_AGREE_MIN = C.HEIGHT_AGREE_MIN

HELDOUT_PROMPT = (
    "The exact same character as in this image: same costume, colours, proportions and armour pieces. "
    "Full body from the top of the head to the feet, standing in an A-pose like a 3D model on a turntable: "
    "arms straight and angled out and down at 45 degrees, hands well away from the hips, legs straight and "
    "slightly apart. "
    "Three-quarter view from the front-left: the character is turned 45 degrees toward the left edge of the "
    "image, so we see its front and its left side, and its left arm is the nearer one. "
    "Keep the figure the same size and position in the frame as in this image: the head and the feet at the "
    "same heights. "
    "No wind: loose cloth such as scarves, capes, coat tails and straps hangs straight down close to the body. "
    "Orthographic projection, no perspective. "
    "Plain flat white background, no shadows, no text, no labels."
)


def figure_mask(client: ModelClient, img: Image.Image) -> tuple[np.ndarray, str]:
    """SAM 3.1 figure mask, the same rule as a turnaround view: kept unless SAM missed the figure (empty, or
    under half the background-threshold area), in which case the threshold mask is used."""
    sam, method, _ = _sam_or_threshold(client, img)
    if method_kind(method) != "sam-3.1":
        return sam, method
    thr = imaging.threshold_mask(img)
    if sam.sum() > 0 and sam.sum() >= 0.5 * thr.sum():
        return sam, method
    return thr, "threshold (sam under half the threshold area)"


def place_on_canvas(
    raw: Image.Image, mask: np.ndarray, scale: float, size: int, bottom_row: int
) -> tuple[Image.Image, np.ndarray]:
    """Crop the figure, scale it by ``scale``, and put it on the ``size`` square: centred horizontally, its
    lowest row on ``bottom_row`` (the front view's feet). Anything past the canvas is cut off, so an oversized
    figure measures short rather than being shrunk to fit."""
    bg = tuple(int(c) for c in imaging.background_colour(np.array(raw)))
    canvas = Image.new("RGB", (size, size), bg)
    mcanvas = Image.new("L", (size, size), 0)
    b = imaging.bbox(mask)
    if b is None:
        return canvas, np.zeros((size, size), dtype=bool)
    new = (max(1, round((b[2] - b[0]) * scale)), max(1, round((b[3] - b[1]) * scale)))
    crop = raw.crop(b).resize(new, Image.LANCZOS)
    cmask = imaging.to_mask_image(mask[b[1] : b[3], b[0] : b[2]]).resize(new, Image.BILINEAR)
    at = (int(round(size / 2 - new[0] / 2)), int(bottom_row - new[1]))
    canvas.paste(crop, at)
    mcanvas.paste(cmask, at)
    return canvas, imaging.from_mask_image(mcanvas)


def heldout_checks(heldout: np.ndarray, front: np.ndarray, sides: list[np.ndarray], raw_area_frac: float) -> list:
    """Figure found; height vs. the front view's at the shared scale; width between side and front widths."""
    out = [C.check("heldout_figure_area_frac", raw_area_frac, 0.01)]
    hw, hh = imaging.box_size(imaging.bbox(heldout) or (0, 0, 0, 0))
    fw, fh = imaging.box_size(imaging.bbox(front) or (0, 0, 0, 0))
    side_w = sum(imaging.box_size(imaging.bbox(m) or (0, 0, 0, 0))[0] for m in sides) / max(1, len(sides))
    out.append(C.check("heldout_height_agreement_front", C.agreement([hh, fh]), HEIGHT_AGREE_MIN))
    # Coarse angle check: turned 45 degrees, the figure is narrower than the front view and wider than the
    # profile. The yaw itself is found by scoring.
    out.append(C.check("heldout_width_over_front_width", hw / fw if fw else 9.0, 1.0, op="<"))
    out.append(C.check("heldout_width_over_side_width", hw / side_w if side_w else 0.0, 1.0, op=">"))
    # Info: the feet point image left when the character shows its left side (not gated: feet apart in a
    # 3/4 view make this noisy).
    out.append(C.check("heldout_toe_offset", imaging.facing_offset(heldout) if hh else 0.0, 0.0, op="<",
                       gate=False, note="negative: feet point image left (front-left view)"))
    return out


def stage2h_heldout(run: RunFolder, client: ModelClient, *, attempts: int = ATTEMPTS) -> dict[str, Any]:
    """Generate, mask and check the held-out view for a run whose stage 2 passed. Writes only `heldout/` and
    `stage-2h.json`; every earlier output is left as it is."""
    s2 = run.read_stage(2)
    if not s2 or not s2.get("ok"):
        raise FileNotFoundError("stage 2 has not passed in this run folder; run stage 2 first")
    meta = run.read_json("views/meta.json")
    size = int(meta["image_size"])
    cover = _load(run, "01-cover.png")
    cover_mask = imaging.from_mask_image(_load(run, "01-masks/cover.png"))
    front = imaging.from_mask_image(_load(run, "masks/front.png"))
    sides = [imaging.from_mask_image(_load(run, f"masks/{v}.png")) for v in ("left", "right")]
    cb, fb = imaging.bbox(cover_mask), imaging.bbox(front)
    if cb is None or fb is None:
        raise ValueError("the cover mask or the front mask is empty")
    view_per_cover_px = (fb[3] - fb[1]) / (cb[3] - cb[1])
    sketch = _load(run, "00-sketch.png") if run.exists("00-sketch.png") else cover
    req_size = f"{sketch.size[0]}x{sketch.size[1]}"  # the size stage 1 asked for; the cover keeps its aspect

    st = _Stage(run, client, "2h", "heldout")  # type: ignore[arg-type]
    tried = []
    for attempt in range(attempts):
        raw = client.edit_image(HELDOUT_PROMPT, [cover], size=req_size, attempt=attempt)
        mask, method = figure_mask(client, raw)
        scale = view_per_cover_px * cover.size[0] / raw.size[0]
        img, placed = place_on_canvas(raw, mask, scale, size, fb[3])
        checks = heldout_checks(placed, front, sides, float(mask.mean()))
        tried.append({
            "attempt": attempt,
            "raw_size": list(raw.size),
            "mask_method": method,
            "scale_view_px_per_raw_px": round(scale, 5),
            "bbox_px": list(imaging.bbox(placed) or ()),
            "checks": checks,
        })
        if C.gate_ok(checks):
            break
    st.save_image(RAW_REL, raw)
    st.save_image(IMAGE_REL, img)
    st.save_image(MASK_REL, imaging.to_mask_image(placed))
    st.checks = checks
    st.details = {
        "prompt": HELDOUT_PROMPT,
        "view": VIEW,
        "requested_size": req_size,
        "image_size": size,
        "mask_method": method_kind(method),
        "mask_method_detail": method,
        "scale_basis": "front-view figure height / cover figure height (the edit keeps the cover's framing)",
        "front_bbox_px": list(fb),
        "attempts": tried,
        "read_by": "scoring only; no building stage reads heldout/",
    }
    st.notes.append(f"Held-out {VIEW} view from {IMAGE_MODEL}, not used to build any body.")
    if len(tried) > 1:
        st.notes.append(f"Generated {len(tried)} times; the last one is kept.")
    return st.finish()
