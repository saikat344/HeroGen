"""Fast lane: face-only paint from a head+0 sample that only the outline check rejected.

When neither frontal pass (`head+0`, `body+0`) holds accepted paint, the face would otherwise keep the
stage-3 source texture. A `head+0` sample whose hair or silhouette moved, but whose face registered,
still carries a usable face. Its face may contribute paint only inside the render's SAM face mask,
feathered inward, outside both images' hair and the outline-disagreement band, and only when the existing
face-outside check passes. No sample is added; the masks reuse the existing SAM 3.1 queries.
"""
import shutil

import numpy as np
from PIL import Image
from scipy import ndimage

from . import character as C, imaging as I

FRONTAL = ("head+0", "body+0")
VIEW = "head+0"
FOLDER = "face-recovery"
FEATHER_PX = 12   # inward ramp per 1024 render pixels, the beauty face composition's boundary width
OUTLINE_REASON = "outline moved"
REASON = "only the outline check rejected it; no accepted frontal paint"


def due(records):
    """True once both frontal passes are resolved and neither holds accepted paint."""
    head = records.get(VIEW)
    if head is None or head.get("reason") == "head_not_found":
        return False
    frontal = [records[key] for key in FRONTAL if key in records]
    return (not any(r["accepted"] for r in frontal)
            and all(r["reason"] != "incomplete" for r in frontal))


def candidates(record):
    """Zero-based attempts the outline check rejected, lowest first. Other reasons never qualify."""
    return sorted({trial["try"] - 1 for trial in record.get("attempts", [])
                   if not trial.get("accepted") and trial.get("reason") == OUTLINE_REASON
                   and "checks" in trial})


def evaluate(client, folder, attempt, cam, body_cam):
    """Re-measure one rejected sample and build its face-only paint, or say why it does not qualify.

    The outline must still fail with unchanged limits and the fast measurement; the face-outside check
    must pass with a face in the repaint. Model errors propagate.
    """
    with Image.open(folder / f"{cam['name']}.png") as rgba:
        inp, alpha = C.composite(rgba.convert("RGBA"))
    render_rgb = np.asarray(inp, dtype=np.float32) / 255
    figure = alpha > 0.5
    with Image.open(folder / f"edit-{attempt}.png") as image:
        edit = image.convert("RGB")
    mapx, mapy, warped, _ = I.register(render_rgb, alpha, np.asarray(edit, dtype=np.float32) / 255)
    fig_e = I.figure_mask(warped, render_figure=figure)
    measured = I.figure_mask(warped, render_figure=figure, render_rgb=render_rgb)
    outline_ok, outline = I.outline_gate(figure, measured, closeup=I.is_closeup(cam, body_cam))
    result = {"attempt": attempt, "outline": {k: round(outline[k], 3) for k in ("iou", "median_px", "p90_px")}}
    if outline_ok:
        return {**result, "qualified": False, "reason": "outline check passes on re-measurement"}
    parts_r = I.part_masks(client, inp, figure.shape)
    parts_e = I.part_masks(client, edit, figure.shape, mapx, mapy)
    ok, parts, _ = I.part_gate(parts_r, parts_e, figure)
    result["parts"] = parts
    if parts["face"] is None:
        return {**result, "qualified": False, "reason": "no face in the repaint"}
    if not ok:
        return {**result, "qualified": False, "reason": "new face outside the render face"}
    render_hair = np.zeros(figure.shape, bool)
    for inst in client.segment(inp, "hair"):
        render_hair |= inst.mask
    edit_hair = I.part_masks_text(client, edit, "hair", mapx, mapy)
    band = ndimage.binary_dilation(figure ^ fig_e, iterations=4)
    region = parts_r["face"] & figure & ~render_hair & ~edit_hair & ~band
    width = max(1, round(FEATHER_PX * max(figure.shape) / 1024))
    inside = ndimage.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
    t = np.clip((inside - .5) / width, 0, 1).astype(np.float32)
    weight = t * t * (3 - 2 * t)
    if not (weight > 0).any():
        return {**result, "qualified": False, "reason": "no face pixels outside hair and the outline band"}
    refined = I.colour_match(render_rgb, warped, figure & fig_e)
    return {**result, "qualified": True, "reason": REASON, "mask_area_px": int((weight > 0).sum()),
            "face_px": int((parts_r["face"] & figure).sum()), "_refined": refined, "_weight": weight,
            "_face": parts_r["face"]}


def resolve(client, work, records, cam, body_cam, previous=None):
    """The `face_recovery` record for the current frontal records, or None when recovery is not due.

    Attempts are evaluated lowest first and the first qualifying attempt wins; earlier evaluations in
    `previous` are reused. The winning paint is saved in `work/face-recovery` beside its render.
    """
    if not due(records):
        return None
    head = records[VIEW]
    known = {row["attempt"]: row for row in (previous or {}).get("evaluated", [])}
    evaluated, chosen = [], None
    for attempt in candidates(head):
        row = known.get(attempt)
        folder = work / VIEW / f"attempt-{attempt}"
        if row is None or (row["qualified"] and not (work / FOLDER / "weight.png").is_file()):
            row = evaluate(client, folder, attempt, cam, body_cam)
        if row["qualified"] and "_weight" in row:
            target = work / FOLDER
            target.mkdir(parents=True, exist_ok=True)
            name = cam["name"]
            for suffix in (".png", ".pos.exr"):
                shutil.copy2(folder / f"{name}{suffix}", target / f"{name}{suffix}")
            for key, image in (("refined", row.pop("_refined")), ("weight", row.pop("_weight"))):
                Image.fromarray((np.clip(image, 0, 1) * 255).round().astype(np.uint8)).save(target / f"{key}.png")
            Image.fromarray(row.pop("_face").astype(np.uint8) * 255).save(target / "face.png")
        evaluated.append({k: v for k, v in row.items() if not k.startswith("_")})
        if row["qualified"]:
            chosen = evaluated[-1]
            break
    if chosen is None:
        return {"applied": False, "view": VIEW, "evaluated": evaluated,
                "reason": ("no head+0 sample was rejected only by the outline check" if not evaluated
                           else "no outline-rejected head+0 sample passed the face checks")}
    return {"applied": True, "view": VIEW, "attempt": chosen["attempt"], "reason": REASON,
            "mask_area_px": chosen["mask_area_px"], "face_px": chosen["face_px"],
            "folder": FOLDER, "camera": cam, "evaluated": evaluated}


def view(work, records, recovery, height, body_cam):
    """The saved recovery as a projectable view, or None when it must not paint."""
    if not (recovery and recovery.get("applied")):
        return None
    if any(records.get(key, {}).get("accepted") for key in FRONTAL):
        return None
    loaded, = I.load_accepted_views(work, {FOLDER: {"accepted": True, "camera": recovery["camera"]}},
                                    height, body_cam=body_cam)
    return loaded
