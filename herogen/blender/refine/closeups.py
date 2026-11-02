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


def _yaw_distance(angle, view_angle):
    return abs((angle - view_angle + 180) % 360 - 180)


def views_for_angle(angle):
    """Turnaround views that condition a repaint camera at this yaw.

    A camera within 30 degrees of a cardinal sees essentially one face, so it gets that view alone.
    A camera farther from every cardinal (40/45/135-degree diagonals) sees two faces, so it gets both
    bracketing views: choosing only one of them would arbitrarily drop the other face's artwork, and
    +-135 in particular is equidistant from a side view and the back view.
    """
    ranked = sorted(ART_ANGLES, key=lambda v: (_yaw_distance(angle, ART_ANGLES[v]), v))
    if _yaw_distance(angle, ART_ANGLES[ranked[0]]) >= 30:
        return tuple(ranked[:2])
    return (ranked[0],)


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


def framed_artwork(run, cam, info, view):
    """Sample a cardinal design reference at the render's world scale and vertical framing.

    Yaw stays cardinal: this is design artwork, not a synthesized view. A head camera receives just
    the head region instead of a full body. Uniform scaling preserves the artwork's proportions;
    prompts prohibit transferring them onto the mesh. No repaint or gate measurement fits this crop.
    """
    art_cam = artwork_camera(run, info, view)
    with Image.open(Path(run) / f"views/{view}.png") as image:
        source = image.convert("RGB")
    ratio = px_per_unit(art_cam) / px_per_unit(cam)
    cx, cy, _ = project(art_cam, np.asarray(cam["centre"]))
    w, h = cam["res"]
    affine = (ratio, 0, float(cx - ratio * w / 2), 0, ratio, float(cy - ratio * h / 2))
    crop = source.transform((w, h), Image.Transform.AFFINE, affine,
                            resample=Image.Resampling.BICUBIC,
                            fillcolor=source.getpixel((0, 0)))
    return crop, {"artwork": f"views/{view}.png", "artwork_camera": art_cam,
                  "output_to_artwork_px": list(affine), "size": [w, h],
                  "mask_edge_inset_px": max(1, round(max(source.size) * .004))}


def framed_mask(run, view, framing):
    """The stage-2 figure mask sampled by exactly the design reference's pixel transform."""
    from scipy import ndimage

    with Image.open(Path(run) / f"masks/{view}.png") as image:
        mask = np.asarray(image.convert("L")) > 127
    # SAM boundary pixels can include the artwork's pale backdrop. Keep colour-extension seeds
    # inside the foreground so magnifying a crop does not stretch that halo into white streaks.
    mask = ndimage.binary_erosion(mask, iterations=framing["mask_edge_inset_px"])
    if not mask.any():
        raise ValueError("silhouette reference needs a nonempty inset artwork mask")
    return Image.fromarray(mask.astype(np.uint8) * 255).transform(
        tuple(framing["size"]), Image.Transform.AFFINE, framing["output_to_artwork_px"],
        resample=Image.Resampling.NEAREST, fillcolor=0)


def silhouette_reference(image, source_mask, render_mask):
    """Fit a close-up design reference to the mesh outline before requesting a repaint.

    A uniform height fit keeps artwork details intact. Where the mesh extends beyond the artwork,
    its nearest foreground edge colour extends the design. Only the reference is resampled; no
    generated edit, projection or gate mask is changed.
    """
    import cv2
    from scipy import ndimage

    source = np.asarray(source_mask) > 127
    target = np.asarray(render_mask, bool)
    if source.shape != target.shape or image.size != target.shape[::-1]:
        raise ValueError("silhouette reference needs matching image and mask dimensions")
    sy, sx = np.where(source)
    ty, tx = np.where(target)
    if not len(sy) or not len(ty):
        raise ValueError("silhouette reference needs nonempty artwork and render masks")
    ratio = (sy.max() - sy.min() + 1) / (ty.max() - ty.min() + 1)
    yy, xx = np.indices(target.shape, dtype=np.float32)
    mapx = (xx - (tx.min() + tx.max()) / 2) * ratio + (sx.min() + sx.max()) / 2
    mapy = (yy - ty.min() + .5) * ratio + sy.min() - .5
    mapx, mapy = mapx.astype(np.float32), mapy.astype(np.float32)
    pixels = cv2.remap(np.asarray(image.convert("RGB")), mapx, mapy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    sampled_mask = cv2.remap(source.astype(np.uint8), mapx, mapy, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    if not sampled_mask.any():
        raise ValueError("silhouette reference does not intersect the artwork")
    extend = target & ~sampled_mask
    if extend.any():
        nearest = ndimage.distance_transform_edt(~sampled_mask, return_distances=False, return_indices=True)
        pixels[extend] = pixels[nearest[0][extend], nearest[1][extend]]
    pixels[~target] = round(255 * C.BG)
    return Image.fromarray(pixels)
