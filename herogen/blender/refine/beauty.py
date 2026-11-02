"""Gated studio-render references for the experimental texture recipe."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ANGLES = (0, 45, -45, 90, -90, 180)
STAGE2 = {0: "front", 90: "right", -90: "left", 180: "back"}
STYLE_IOU_MIN = .70
# Scalp detections stop at the jaw; the occluded neck gap plus the collar's top band lie within
# a further quarter of the scalp height on the saved r1 crops (s023 head+180 cut off the collar).
NECK_FRACTION = .25
REAR_QUARTER_DEG = 45
BT1 = (
    "Create a polished stylised 3D game-character beauty render at yaw {angle} degrees. The first image is "
    "the exact camera, silhouette and framing guide: preserve its outline, A-pose, proportions, full-body size "
    "and every costume shape without moving, cropping, rotating or adding parts. The second image is the "
    "original character artwork: preserve this character's identity, costume, colours and details. The third "
    "image is the nearest stage-2 turnaround design view; use it for costume details visible at this yaw. "
    "Use a plain uniform light background. Render crisp edges, finely resolved facial features and hair, "
    "clear leather grain and seams, woven cloth folds, restrained metal highlights, and soft skin. Bake soft "
    "consistent studio illumination: a broad light above and in front of the CHARACTER, fixed in character "
    "space at front yaw zero in every view, with gentle fill. Never put a new camera-facing key light on "
    "the sides or back. Darken occluded creases under the cape, between straps and inside boot cuffs; keep "
    "material colours distinct. No dramatic shadows, rim light, ground shadow, scenery, text or props."
)
BT1_SHA256 = hashlib.sha256(BT1.encode()).hexdigest()


def union(client, image, text):
    mask = np.zeros(image.size[::-1], bool)
    for instance in client.segment(image, text):
        mask |= instance.mask
    return mask


def angular_distance(a, b):
    """Degrees between two yaws on the circle, 0 to 180."""
    return abs((a - b + 180) % 360 - 180)


def nearest(angle):
    """Nearest reference yaw; ties follow the frozen ANGLES order."""
    return min(ANGLES, key=lambda yaw: angular_distance(angle, yaw))


def lateral(angle):
    """True when a horizontal mirror flips the yaw: every yaw except front (0) and back (180)."""
    return 0 < angular_distance(angle, 0) < 180


def rear_quarter(angle):
    """True for cameras that see the back quarter: off back by at most REAR_QUARTER_DEG."""
    return 0 < angular_distance(angle, 180) <= REAR_QUARTER_DEG


def shows_face(yaw):
    """True when a crop at this yaw carries an eye or nose; rear-facing yaws show the scalp."""
    return angular_distance(yaw, 180) > REAR_QUARTER_DEG


def _centroid(mask):
    yy, xx = np.nonzero(mask)
    if not len(yy):
        return None
    return round(float(yy.mean())), round(float(xx.mean()))


def _translate(mask, dy, dx):
    """Integer translation with zero fill, never wrapping mask pixels across the frame."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    y0, y1 = max(0, dy), min(h, h + dy)
    x0, x1 = max(0, dx), min(w, w + dx)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = mask[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def aligned_iou(mask, expected):
    """IoU after moving the mask's centroid onto the guide mask's centroid: facing, not offset.

    A retained prop the guide mask lacks (a forearm shield SAM's mesh does not carry) shifts
    the centroid without flipping facing, so unregistered IoU can score the mirrored outline
    closer. Empty masks keep their raw IoU because they have no centroid to align.
    """
    centre, start = _centroid(expected), _centroid(mask)
    if centre is None or start is None:
        return float((mask & expected).sum() / max(1, (mask | expected).sum()))
    moved = _translate(mask, centre[0] - start[0], centre[1] - start[1])
    return float((moved & expected).sum() / max(1, (moved | expected).sum()))


def with_neck(box, height):
    """Extend a head crop downward into the neck and collar, clamped to the image."""
    x0, y0, x1, y1 = box
    return [x0, y0, x1, min(height, y1 + round((y1 - y0) * NECK_FRACTION))]


def generate(run, client, out, diagonal_guides):
    """Save each paid attempt before testing its unregistered SAM outline; never substitute a rejected view."""
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "references.json"
    rec = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "prompt": "BT1", "text": BT1, "sha256": BT1_SHA256, "views": {}}
    if rec["sha256"] != BT1_SHA256:
        raise ValueError("beauty reference prompt hash differs")
    cover = Image.open(run / "01-cover.png").convert("RGB")
    start = len(client.log.calls)
    def save():
        rec["cost_usd"] = round(rec.get("prior_cost_usd", 0) + client.log.cost_usd(start), 6)
        manifest_path.write_text(json.dumps(rec, indent=2) + "\n")
    rec["prior_cost_usd"] = rec.get("cost_usd", 0)
    try:
        for angle in ANGLES:
            key = str(angle)
            view = rec["views"].setdefault(key, {"accepted": False, "attempts": []})
            if view["accepted"] or (len(view["attempts"]) >= 3 and all("iou" in t for t in view["attempts"])):
                continue
            stage2 = STAGE2.get(angle, "front")
            design = Image.open(run / f"views/{stage2}.png").convert("RGB")
            if angle in STAGE2:
                guide = design
                expected = np.asarray(Image.open(run / f"masks/{stage2}.png").convert("L")) > 127
                threshold = .90
            else:
                rgba = Image.open(diagonal_guides[angle]).convert("RGBA")
                expected = np.asarray(rgba)[..., 3] > 127
                guide = Image.new("RGB", rgba.size, (240, 240, 240))
                guide.paste(rgba, mask=rgba.getchannel("A"))
                threshold = .85
            view.update(threshold=threshold, guide_stage2=stage2)
            Image.fromarray(expected.astype(np.uint8) * 255).save(out / f"expected{angle:+d}.png")
            for attempt in range(3):
                if attempt < len(view["attempts"]) and "iou" in view["attempts"][attempt]:
                    continue
                image = client.edit_image(BT1.format(angle=angle), [guide, cover, design],
                                          size=f"{guide.width}x{guide.height}", attempt=attempt)
                path = out / f"beauty{angle:+d}-try{attempt + 1}.png"
                image.save(path)
                trial = {"try": attempt + 1, "image": path.name, "accepted": False}
                if attempt < len(view["attempts"]):
                    view["attempts"][attempt] = trial
                else:
                    view["attempts"].append(trial)
                save()
                raw_mask = union(client, image, "character")
                mask = raw_mask
                # Resolution changes are resampling only, without alignment, translation or shape fitting.
                mask = np.asarray(Image.fromarray(mask).resize((expected.shape[1], expected.shape[0]),
                                                              Image.Resampling.NEAREST), bool)
                iou = float((mask & expected).sum() / max(1, (mask | expected).sum()))
                trial.update(iou=iou, accepted=iou >= threshold)
                Image.fromarray(raw_mask.astype(np.uint8) * 255).save(out / f"mask{angle:+d}-try{attempt + 1}.png")
                if trial["accepted"]:
                    view.update(accepted=True, image=path.name, mask=f"mask{angle:+d}-try{attempt + 1}.png")
                save()
                if view["accepted"]:
                    break
        for view in rec["views"].values():
            if not view["accepted"] or "head_box" in view:
                continue
            image = Image.open(out / view["image"]).convert("RGB")
            yy, xx = np.where(union(client, image, "head"))
            if not len(xx):
                raise ValueError("no head detected in accepted beauty view")
            x0, x1, y0, y1 = int(xx.min()), int(xx.max() + 1), int(yy.min()), int(yy.max() + 1)
            pad = round(max(x1 - x0, y1 - y0) * .15)
            view["head_box"] = [max(0, x0 - pad), max(0, y0 - pad),
                                min(image.width, x1 + pad), min(image.height, y1 + pad)]
            save()
        rec["status"] = "complete" if all(v["accepted"] for v in rec["views"].values()) else "rejected"
        rec.pop("error_type", None)
        return rec
    except Exception as exc:
        rec.update(status="error", error_type=type(exc).__name__)
        raise
    finally:
        save()


def load(folder):
    """Select existing style guides at IoU >= STYLE_IOU_MIN, facing-checked only where mirrors flip.

    Recompute from saved pixels, without registration. Legacy generation acceptance flags do not
    govern style selection. An unavailable yaw is omitted so cameras can use the nearest usable yaw.
    Head crops must be supplied for the selected images; loading never makes model calls.

    Mirroring cannot flip the facing of a front (0) or back (180) view, so the mirror test runs
    only on lateral yaws, and both orientations are centroid-aligned onto the guide mask first so
    a retained prop cannot make the mirrored outline the closer fit. Selected head crops are
    extended into the neck and collar so close-up repaints keep collar materials.
    """
    folder = Path(folder)
    rec = json.loads((folder / "references.json").read_text())
    if rec["sha256"] != BT1_SHA256:
        raise ValueError("beauty reference prompt hash differs")
    rec["selection_policy"] = {"iou_min": STYLE_IOU_MIN, "must_beat_mirror": "lateral",
                               "mirror_iou": "centroid_aligned",
                               "head_box_neck_extension": NECK_FRACTION,
                               "rear_quarter_face_free": True}
    for angle in ANGLES:
        v = rec["views"].setdefault(str(angle), {"attempts": []})
        expected_path = folder / f"expected{angle:+d}.png"
        candidates = []
        if v.get("attempts"):
            expected = np.asarray(Image.open(expected_path).convert("L")) > 127
            v["guide_mask_sha256"] = hashlib.sha256(expected_path.read_bytes()).hexdigest()
        for trial in v.get("attempts", []):
            mask_name = trial.get("mask", f"mask{angle:+d}-try{trial['try']}.png")
            mask = Image.open(folder / mask_name).convert("L").resize(
                (expected.shape[1], expected.shape[0]), Image.Resampling.NEAREST)
            mask = np.asarray(mask) > 127
            def iou(value):
                return float((value & expected).sum() / max(1, (value | expected).sum()))
            score = iou(mask)
            facing_score, mirrored = aligned_iou(mask, expected), aligned_iou(mask[:, ::-1], expected)
            correct_facing = facing_score > mirrored if lateral(angle) else True
            usable = score >= STYLE_IOU_MIN and correct_facing
            trial.update(style_iou=score, facing_iou=facing_score, mirror_iou=mirrored,
                         correct_facing=correct_facing, style_usable=usable, mask=mask_name,
                         mask_sha256=hashlib.sha256((folder / mask_name).read_bytes()).hexdigest())
            if usable:
                candidates.append(trial)
        v["accepted"] = bool(candidates)
        if not candidates:
            for key in ("image", "mask", "head_box", "image_sha256"):
                v.pop(key, None)
            continue
        best = max(candidates, key=lambda t: t["style_iou"])
        box = rec.get("head_boxes", {}).get(best["image"])
        if box is None and best["image"] == v.get("image"):
            box = v.get("head_box")
        if box is None:
            raise ValueError(f"beauty reference yaw {angle} has no head crop")
        with Image.open(folder / best["image"]) as image:
            if (len(box) != 4 or not all(isinstance(n, int) for n in box)
                    or not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height)):
                raise ValueError(f"beauty reference yaw {angle} has invalid head crop")
            box = with_neck(box, image.height)
        v.update(image=best["image"], mask=best["mask"], head_box=box,
                 image_sha256=hashlib.sha256((folder / best["image"]).read_bytes()).hexdigest())
    rec["fallbacks"] = {str(angle): selected_yaw(rec, angle) for angle in ANGLES}
    return rec


def selected_yaw(rec, angle):
    """Nearest usable circular yaw, or None to request the standard recipe.

    Rear-quarter cameras only draw on rear-facing yaws: a profile crop carries its eye and
    nose onto the rear scalp, where the face gate must reject the painted result.
    """
    usable = [yaw for yaw in ANGLES if rec["views"].get(str(yaw), {}).get("accepted")]
    if rear_quarter(angle):
        usable = [yaw for yaw in usable if not shows_face(yaw)]
    return min(usable, key=lambda yaw: angular_distance(angle, yaw)) if usable else None


def reference(folder, rec, angle, *, head=False):
    """Select the nearest usable yaw and its neck-extended head crop for close-up requests.

    Rear-quarter requests never receive a crop that shows a face.
    """
    folder = Path(folder)
    yaw = selected_yaw(rec, angle)
    if yaw is None:
        raise ValueError("no usable beauty reference; use the standard recipe")
    v = rec["views"][str(yaw)]
    image = Image.open(folder / v["image"]).convert("RGB")
    if head:
        box = v.get("head_box")
        if box is None:
            raise ValueError(f"beauty reference yaw {yaw} has no head crop")
        image = image.crop(box)
    return image, {"yaw": yaw, "requested_yaw": angle, "fallback": yaw != nearest(angle),
                   "image_sha256": v["image_sha256"], "head_box": v.get("head_box") if head else None}


def facial_features(client, image, face=None):
    """Detect feature presence inside a face crop, excluding ornaments and costume marks.

    Presence is semantic, with no sharpness or pixel-similarity threshold. Occluded features
    absent from the reference are not required. Segmentation errors propagate to the caller.
    """
    if face is None:
        face = np.zeros(image.size[::-1], bool)
        for inst in client.segment(image, "face"):
            face |= inst.mask
    y, x = np.nonzero(face)
    if not len(x):
        return {name: False for name in ("eyes", "eyebrows", "mouth")}
    box = (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
    crop = image.crop(box)
    inside = face[box[1]:box[3], box[0]:box[2]]
    return {name: any((inst.mask & inside).any() for inst in client.segment(crop, name))
            for name in ("eyes", "eyebrows", "mouth")}


def frontal_features(client, image, face):
    """Require two separate eye regions and a mouth inside the frontal face.

    SAM can return both eyes in one instance. Count connected regions in their union rather
    than instances; one connected eye mark cannot establish that both eyes are present.
    """
    from scipy import ndimage

    y, x = np.nonzero(face)
    if not len(x):
        return {"both_eyes": False, "mouth": False, "eye_regions": 0}
    box = (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
    crop = image.crop(box)
    inside = face[box[1]:box[3], box[0]:box[2]]
    eyes = union(client, crop, "eyes") & inside
    labels, count = ndimage.label(eyes)
    centres = ndimage.center_of_mass(eyes, labels, range(1, count + 1)) if count else []
    both = any(x < crop.width / 2 for _, x in centres) and any(x >= crop.width / 2 for _, x in centres)
    boxes = []
    for side in (False, True):
        regions = [i for i, (_, x) in enumerate(centres, 1) if (x >= crop.width / 2) == side]
        if regions:
            largest = max(regions, key=lambda i: int((labels == i).sum()))
            ey, ex = np.nonzero(labels == largest)
            boxes.append([int(ex.min()) + box[0], int(ey.min()) + box[1],
                          int(ex.max()) + 1 + box[0], int(ey.max()) + 1 + box[1]])
    return {"both_eyes": bool(both), "eye_regions": int(count),
            "eye_boxes": boxes,
            "mouth": bool((union(client, crop, "mouth") & inside).any())}
