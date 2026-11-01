"""Threefold orthographic detail cameras and calibrated turnaround crops.

Source meshes use their imported frame; joints and artwork use the stage-2 contract frame. A uniform height
mapping connects them, without fitting the generated artwork to the repaint or changing the source geometry.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import character as C
from .cameras import axes, camera, project, px_per_unit

RES = (1920, 1280)
ZOOM = 3.0
BORDER_PX = 96
ANGLES = (0, 45, -45, 180)
ART_ANGLES = {"front": 0, "right": 90, "back": 180, "left": -90}


def contract_scale(info, meta):
    try:
        units = float(meta["height_units"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("close-ups need positive views/meta.json height_units") from None
    if not math.isfinite(units) or units <= 0:
        raise ValueError("close-ups need positive views/meta.json height_units")
    return units / info["height"]


def cameras_for(info, joints=None, meta=None):
    """Four overlapping torso/leg bands, each at four yaws, at 3x body pixels per unit.

    Anatomical heights use SAM 3D Body joints when supplied. Implausible locations outside the mesh fail
    explicitly. Forearms crossing a band remain visible, but do not get additional dedicated cameras.
    """
    h, lo = info["height"], np.asarray(info["lo"], float)
    fractions = {"chest": .72, "waist": .51, "thighs": .31, "boots": .12}
    if joints is not None:
        scale = contract_scale(info, meta)
        points = joints.get("joints", {}) if isinstance(joints, dict) else {}

        def z(names):
            try:
                values = np.asarray([points[n] for n in names], float)
            except (KeyError, TypeError, ValueError):
                raise ValueError("close-up joints need finite shoulder, waist, knee and ankle points") from None
            if values.shape != (len(names), 3) or not np.isfinite(values).all():
                raise ValueError("close-up joints must be finite 3D points")
            fraction = float(values[:, 2].mean() / (scale * h))
            if not 0 <= fraction <= 1.1:
                raise ValueError("close-up joints fall outside the mesh height")
            return fraction

        waist = z(["Waist"])
        shoulder = z(["LeftShoulder", "RightShoulder"])
        knee = z(["LeftKnee", "RightKnee"])
        ankle = z(["LeftAnkle", "RightAnkle"])
        fractions.update(chest=(waist + shoulder) / 2 + .04, waist=waist,
                         thighs=(waist + knee) / 2, boots=(knee + ankle) / 2)
    body = camera("body", info["body_centre"], 0, h * C.BODY_FRAME, C.BODY_RES)
    ortho = max(RES) / (ZOOM * px_per_unit(body))
    # Limit centre separation so the 96px feathers of adjacent bands overlap at full weight.
    span = ortho * RES[1] / max(RES) / h
    max_step = span * (1 - 2 * BORDER_PX / min(RES))
    centres = [float(fractions[n]) for n in fractions]
    centres[-1] = min(centres[-1], span / 2 - BORDER_PX / min(RES) * span)
    for i in range(len(centres) - 2, -1, -1):
        centres[i] = min(centres[i], centres[i + 1] + max_step)
    cams = []
    for band, fraction in zip(fractions, centres):
        centre = list(info["body_centre"])
        centre[2] = float(lo[2] + fraction * h)
        for angle in ANGLES:
            cam = camera(f"close-{band}{angle:+d}", centre, angle, ortho, RES)
            cam.update(band=band, zoom=body["ortho"] / ortho, pixels_per_unit_gain=ZOOM, frame_border_px=BORDER_PX,
                       location_source="sam3d-body joints" if joints is not None else "mesh bounds")
            cams.append(cam)
    return cams


def nearest_view(angle):
    """Cardinal reference nearest this yaw; diagonal ties choose the front deterministically."""
    return min(ART_ANGLES, key=lambda v: abs((angle - ART_ANGLES[v] + 180) % 360 - 180))


def band_box(cam, info):
    """Eight corners of the camera's orthographic box through the mesh's full depth."""
    right, up, fwd = axes(cam)
    centre = np.asarray(cam["centre"])
    bounds = np.array(list(itertools.product(*zip(info["lo"], info["hi"]))))
    depth = (bounds - centre) @ fwd
    k = px_per_unit(cam)
    return np.array([centre + right * x + up * z + fwd * d
                     for x, z, d in itertools.product((-cam["res"][0] / (2 * k), cam["res"][0] / (2 * k)),
                                                      (-cam["res"][1] / (2 * k), cam["res"][1] / (2 * k)),
                                                      (depth.min(), depth.max()))])


def artwork_camera(run, info, view):
    """Reconstruct the stage-2 orthographic scale and centre from metadata and the saved mask.

    Legacy meta.json stores pixels_per_unit/height_units, with the centre implicit in each mask's bounding
    box (the same convention as blender.views). No assumption of a square image or centred padding is made.
    """
    from herogen.blender.views import bbox, load_mask

    run = Path(run)
    meta = json.loads((run / "views/meta.json").read_text())
    scale = contract_scale(info, meta)
    try:
        ppu = float(meta["pixels_per_unit"]) * scale
    except (KeyError, TypeError, ValueError):
        raise ValueError("close-ups need positive views/meta.json pixels_per_unit") from None
    if not math.isfinite(ppu) or ppu <= 0:
        raise ValueError("close-ups need positive views/meta.json pixels_per_unit")
    with Image.open(run / f"views/{view}.png") as image:
        w, h = image.size
    mask = load_mask(run / f"masks/{view}.png")
    box = bbox(mask)
    if mask.shape != (h, w) or box is None:
        raise ValueError(f"close-up reference {view} needs a nonempty, matching mask")
    r0, r1, c0, c1 = box
    cmid, feet = (c0 + c1 + 1) / 2, r1 + 1
    cam = camera(view, info["body_centre"], ART_ANGLES[view], max(w, h) / ppu, (w, h))
    right, _, _ = axes(cam)
    centre = np.asarray(info["body_centre"]) + right * ((w / 2 - cmid) / ppu)
    centre[2] = info["lo"][2] + (feet - h / 2) / ppu
    cam["centre"] = centre.tolist()
    return cam


def crop_box(cam, art_cam, info):
    points = band_box(cam, info)
    x, y, _ = project(art_cam, points)
    w, h = art_cam["res"]
    box = [max(0, math.floor(x.min())), max(0, math.floor(y.min())),
           min(w, math.ceil(x.max())), min(h, math.ceil(y.max()))]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("close-up reference box does not intersect the artwork")
    return box


def reference(run, cam, info, folder):
    view = nearest_view(cam["angle"])
    art_cam = artwork_camera(run, info, view)
    box = crop_box(cam, art_cam, info)
    with Image.open(Path(run) / f"views/{view}.png") as image:
        crop = image.convert("RGB").crop(box)
    crop.save(Path(folder) / "reference-crop.png")
    return crop, {"render": "input.png", "crop": "reference-crop.png", "artwork": f"views/{view}.png",
                  "cover": "01-cover.png", "crop_box_px": box, "artwork_camera": art_cam,
                  "box_world": band_box(cam, info).tolist()}
