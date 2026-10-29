"""Stages 0 (input), 1 (cover art) and 2 (turnaround) of the sketch-to-hero pipeline."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from herogen.pipeline import checks as C
from herogen.pipeline import imaging
from herogen.pipeline.client import (
    COST_BASIS, IMAGE_MODEL, PRICE_PER_IMAGE_USD, SAM_MODEL, ModelClient, SamInstance,
)
from herogen.pipeline.runfolder import RunFolder, now_iso

SAM_TEXTS = ("character", "person")  # tried in order; "person" misses armoured heroes on some sheets
SIDE_VIEWS = ("front", "right", "back", "left")
ALL_VIEWS = SIDE_VIEWS + ("top",)
HEIGHT_UNITS = 5.5  # the figure's height in units; studs, matching the stage-4 sections.json example
VIEW_MARGIN = 0.08  # empty border around the largest view on its square canvas, as a fraction of its size

COVER_PROMPT = (
    "Turn this pencil sketch into a clean, finished full-colour illustration of the same character. "
    "Keep exactly the same pose, body proportions, silhouette, costume and armour pieces as the sketch. "
    "Full body visible from the top of the head to the feet, centred, the same framing as the sketch. "
    "Plain flat light-grey background, no ground shadow, no scenery, no text."
)

TURNAROUND_PROMPT = (
    "Character turnaround sheet of this exact character: same costume, colours and proportions. "
    "Four full-body views side by side in one horizontal row, all at exactly the same scale, feet on one "
    "shared baseline, heads at the same height, evenly spaced with wide empty gaps between the figures. "
    "Left to right: 1) front view; 2) right side view, a strict profile showing the character's right side, "
    "facing the right edge of the image; 3) back view; 4) left side view, a strict profile showing the "
    "character's left side, facing the left edge of the image. "
    "Draw it like a 3D model in an A-pose turned on a turntable: in the front and back views the arms are "
    "straight and angled out and down at 45 degrees, hands well away from the hips. Turned 90 degrees for "
    "the side views, those same arms point sideways toward and away from the viewer, so in profile they "
    "are hidden behind the body and only the near arm shows, running down beside the torso; nothing "
    "reaches forward. In each side view the head, chest, knees and toes all point the same way. "
    "Legs straight and slightly apart. "
    "No wind: loose cloth such as scarves, capes, coat tails and straps hangs straight down close to the "
    "body, the same in every view. "
    "Orthographic projection, no perspective. "
    "Plain flat white background, no shadows, no text, no labels, no grid lines."
)

TOP_PROMPT = (
    "Orthographic top view of this exact character, like the top camera of a 3D model viewer: the camera is "
    "directly above the head looking straight down. You see only the tops of things: the top of the helmet "
    "as a rounded shape in the centre, the tops of the shoulders, the A-pose arms reaching out to the left "
    "and right and angled down, and the front tips of the boots just peeking out toward the bottom edge of "
    "the image. The face and chest are not visible. Plain flat white background, no shadows, no text."
)


class GateFailed(RuntimeError):
    def __init__(self, stage: int, record: dict[str, Any]) -> None:
        failed = [c["name"] for c in record["checks"] if c.get("gate", True) and not c["pass"]]
        super().__init__(f"stage {stage} gate failed: {', '.join(failed)}")
        self.stage = stage
        self.record = record


class _Stage:
    """Collects one stage's record and writes `stage-<n>.json`."""

    def __init__(self, run: RunFolder, client: ModelClient | None, n: int, name: str) -> None:
        self.run, self.client, self.n, self.name = run, client, n, name
        self.started = now_iso()
        self.t0 = time.monotonic()
        self.calls0 = len(client.log.calls) if client else 0
        self.outputs: list[str] = []
        self.checks: list[dict[str, Any]] = []
        self.details: dict[str, Any] = {}
        self.notes: list[str] = []

    def save_image(self, rel: str, img: Image.Image) -> None:
        img.save(self.run.file(rel), format="PNG")
        self.outputs.append(rel)

    def finish(self) -> dict[str, Any]:
        calls = self.client.log.calls[self.calls0 :] if self.client else []
        live = [c for c in calls if not c["cached"]]
        record = {
            "stage": self.n,
            "name": self.name,
            "ok": C.gate_ok(self.checks),
            "started": self.started,
            "seconds": round(time.monotonic() - self.t0, 2),
            "cost_usd": round(sum(PRICE_PER_IMAGE_USD.get(c["model"], 0.0) for c in live), 6),
            "cost_basis": COST_BASIS + "; replayed (cached) calls cost 0",
            "models": self.client.log.models(self.calls0) if self.client else [],
            "outputs": self.outputs,
            "checks": self.checks,
            "notes": " ".join(self.notes),
            "calls": calls,
            "details": self.details,
        }
        self.run.write_stage(record)
        if not record["ok"]:
            raise GateFailed(self.n, record)
        return record


def _load(run: RunFolder, rel: str) -> Image.Image:
    if not run.exists(rel):
        raise FileNotFoundError(f"{rel} is missing from the run folder; run the earlier stage first")
    return Image.open(run.path / rel).convert("RGB")


def _figure_mask(instances: list[SamInstance], shape: tuple[int, int]) -> np.ndarray:
    """Union of the instances at least 20% the size of the largest (drops stray small detections)."""
    if not instances:
        return np.zeros(shape, dtype=bool)
    areas = [int(i.mask.sum()) for i in instances]
    keep = [i for i, a in zip(instances, areas, strict=True) if a >= 0.2 * max(areas)]
    out = np.zeros(shape, dtype=bool)
    for i in keep:
        out |= i.mask
    return out


def method_kind(method: str) -> str:
    """The contract's mask-method name for a detailed method string: "sam-3.1", "threshold" or "visual_hull"."""
    if method.startswith(SAM_MODEL):
        return SAM_MODEL
    if method.startswith("visual_hull"):
        return "visual_hull"
    return "threshold"


def _sam_or_threshold(client: ModelClient, img: Image.Image) -> tuple[np.ndarray, str, int]:
    """SAM 3.1 figure mask (first text prompt that finds something), else the background-threshold mask."""
    shape = (img.size[1], img.size[0])
    errors = []
    for text in SAM_TEXTS:
        try:
            inst = client.segment(img, text)
        except Exception as e:  # noqa: BLE001  (a SAM failure falls back to the threshold mask, recorded)
            errors.append(type(e).__name__)
            continue
        mask = _figure_mask(inst, shape)
        if mask.sum() > 0:
            return mask, f"{SAM_MODEL}:{text}", len(inst)
    why = f"sam error: {', '.join(errors)}" if errors else "sam found nothing"
    return imaging.threshold_mask(img), f"threshold ({why})", 0


# ---- stage 0 ----------------------------------------------------------------------------------------


def stage0_input(run: RunFolder, sketch_path: Path, label: str = "") -> dict[str, Any]:
    st = _Stage(run, None, 0, "input")
    img = Image.open(sketch_path).convert("RGB")  # re-encoded: drops EXIF and other metadata
    st.save_image("00-sketch.png", img)
    w, h = img.size
    st.details = {"width": w, "height": h, "source_name": Path(sketch_path).name, "label": label}
    st.checks = [C.check("min_side_px", min(w, h), 256)]
    if label:
        st.notes.append(f"Input is a {label}.")
    return st.finish()


# ---- stage 1 ----------------------------------------------------------------------------------------


def stage1_cover(run: RunFolder, client: ModelClient) -> dict[str, Any]:
    st = _Stage(run, client, 1, "cover_art")
    sketch = _load(run, "00-sketch.png")
    w, h = sketch.size
    cover = client.edit_image(COVER_PROMPT, [sketch], size=f"{w}x{h}")
    st.save_image("01-cover.png", cover)
    sketch_mask, sketch_method, n_s = _sam_or_threshold(client, sketch)
    cover_mask, cover_method, n_c = _sam_or_threshold(client, cover)
    st.save_image("01-masks/sketch.png", imaging.to_mask_image(sketch_mask))
    st.save_image("01-masks/cover.png", imaging.to_mask_image(cover_mask))
    st.checks, extra = C.cover_checks(sketch_mask, cover_mask)
    st.details = {
        "prompt": COVER_PROMPT,
        "cover_size": list(cover.size),
        "mask_method": {"sketch": sketch_method, "cover": cover_method},
        "sam_instances": {"sketch": n_s, "cover": n_c},
        **extra,
    }
    return st.finish()


# ---- stage 2 ----------------------------------------------------------------------------------------

FACING_MIN = 0.05  # |toe offset| below this is "can't tell" and the view is left as drawn
SHEET_SIZE = "2560x1024"  # aspect only (Muse Image fixes the resolution); room for four A-pose figures


def split_sheet(sheet: Image.Image, fig_mask: np.ndarray, thr_mask: np.ndarray, fig_method: str) -> dict[str, Any]:
    """Find the four figures' column extents and cut them out (`panel0`..`panel3`, left to right).

    The split uses the SAM mask (falls back to the threshold mask if SAM's columns don't give four
    figures). A view keeps the SAM mask unless SAM missed it (empty, or under half the threshold area in
    that column range), in which case its mask is the threshold mask and `mask_method` says so.
    """
    names = [f"panel{i}" for i in range(len(SIDE_VIEWS))]
    sam_ok = fig_method.startswith(SAM_MODEL)
    runs = imaging.split_columns(fig_mask, expected=len(names))
    source = fig_method
    if len(runs) != len(names) and sam_ok:
        runs = imaging.split_columns(thr_mask, expected=len(names))
        source = "threshold"
    out: dict[str, Any] = {"runs": runs, "split_source": source, "views": {}}
    if len(runs) != len(names):
        return out
    combined = np.zeros_like(thr_mask)
    methods = {}
    for (x0, x1), name in zip(runs, names, strict=True):
        s, t = fig_mask[:, x0:x1], thr_mask[:, x0:x1]
        if sam_ok and s.sum() > 0 and s.sum() >= 0.5 * t.sum():
            combined[:, x0:x1], methods[name] = s, fig_method
        else:
            combined[:, x0:x1], methods[name] = t, "threshold"
    # Side views share the sheet's row range, so their feet stay on one baseline after the crop.
    boxes = [imaging.bbox(combined[:, x0:x1]) for x0, x1 in runs]
    y0 = min(b[1] for b in boxes if b)
    y1 = max(b[3] for b in boxes if b)
    for (x0, x1), name in zip(runs, names, strict=True):
        out["views"][name] = {
            "image": sheet.crop((x0, y0, x1, y1)),
            "mask": combined[y0:y1, x0:x1],
            "mask_method": methods[name],
            "sheet_box": [x0, y0, x1, y1],
        }
    return out


def assign_views(panels: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Name the four panels. Front and back are the two widest figures (arms out), front the first of them;
    the two narrow ones are the profiles, `right` the one whose toes point image right (else the first).

    Models don't always keep the requested order (front, right, back, left), so the order is read from the
    figures rather than assumed. Front vs. back can't be told from a silhouette; front-first is assumed.
    """
    order = sorted(panels)
    width = {p: imaging.box_size(imaging.bbox(panels[p]["mask"]) or (0, 0, 0, 0))[0] for p in order}
    wide = sorted(sorted(order, key=lambda p: -width[p])[:2])
    narrow = [p for p in order if p not in wide]
    face = {p: imaging.facing_offset(panels[p]["mask"]) for p in narrow}
    if face[narrow[0]] < -FACING_MIN and face[narrow[1]] > FACING_MIN:
        right, left = narrow[1], narrow[0]
    else:
        right, left = narrow[0], narrow[1]
    names = {"front": wide[0], "back": wide[1], "right": right, "left": left}
    info = {
        "panel_for_view": names,
        "requested_order_kept": [names[v] for v in SIDE_VIEWS] == order,
        "panel_widths_px": width,
    }
    return {v: panels[p] for v, p in names.items()}, info


def square_views(views: dict[str, dict[str, Any]], bg: tuple[int, int, int]) -> tuple[int, dict[str, Any]]:
    """Pad every view to one square size at the shared pixel scale (translation only, never a rescale).

    The figure is centred horizontally on its box; vertically the shared row range is centred, so all
    feet sit on one row.
    """
    biggest = max(max(v["image"].size) for v in views.values())
    size = int(math.ceil(biggest * (1 + 2 * VIEW_MARGIN)))
    size += size % 2
    out = {}
    for name, v in views.items():
        b = imaging.bbox(v["mask"])
        cx = (b[0] + b[2]) / 2 if b else v["image"].size[0] / 2
        cy = v["image"].size[1] / 2
        img = imaging.paste_centred(v["image"], size, bg, cx, cy)
        mimg = imaging.paste_centred(imaging.to_mask_image(v["mask"]), size, 0, cx, cy)
        out[name] = {**v, "image": img, "mask": imaging.from_mask_image(mimg)}
    return size, out


def orient_sides(sq: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Make `right` face image right and `left` face image left (contract cameras), judged by the feet.

    A view whose toes point the wrong way is mirrored (an orthographic profile seen from the other side is
    its mirror image; asymmetric costume details are mirrored with it).
    """
    fr = imaging.facing_offset(sq["right"]["mask"])
    fl = imaging.facing_offset(sq["left"]["mask"])
    checks_ = [
        C.check("right_view_toe_offset_as_drawn", fr, 0.0, op=">", gate=False),
        C.check("left_view_toe_offset_as_drawn", fl, 0.0, op="<", gate=False),
    ]
    notes: list[str] = []
    fixes: dict[str, Any] = {}
    for name, off, want in (("right", fr, 1), ("left", fl, -1)):
        if off * want < -FACING_MIN:
            v = sq[name]
            v["image"] = v["image"].transpose(Image.FLIP_LEFT_RIGHT)
            v["mask"] = np.fliplr(v["mask"])
            notes.append(f"The {name} view faced the wrong way (feet); mirrored it.")
            fixes[f"mirrored_{name}"] = True
    return checks_, notes, fixes


def model_top_view(client: ModelClient, cover: Image.Image, sq: dict[str, Any], size: int, bg) -> dict[str, Any]:
    """Ask Muse Image for a top view and bring it to the shared scale (its width set to the front width)."""
    img = client.edit_image(TOP_PROMPT, [cover], size="1024x1024")
    mask, method, _ = _sam_or_threshold(client, img)
    tb = imaging.bbox(mask)
    fb = imaging.bbox(sq["front"]["mask"])
    if tb is None or fb is None:
        return {"raw": img, "mask": None, "mask_method": method}
    scale = (fb[2] - fb[0]) / max(1, tb[2] - tb[0])
    crop = img.crop(tb)
    cmask = imaging.to_mask_image(mask[tb[1] : tb[3], tb[0] : tb[2]])
    new = (max(1, round(crop.size[0] * scale)), max(1, round(crop.size[1] * scale)))
    if max(new) > size:
        return {"raw": img, "mask": None, "mask_method": method, "scale": round(scale, 4)}
    scaled_mask = cmask.resize(new, Image.NEAREST)
    tile = Image.new("RGB", new, bg)
    tile.paste(crop.resize(new, Image.LANCZOS), (0, 0), scaled_mask)
    square = imaging.paste_centred(tile, size, bg, new[0] / 2, new[1] / 2)
    smask = imaging.from_mask_image(imaging.paste_centred(scaled_mask, size, 0, new[0] / 2, new[1] / 2))
    return {"raw": img, "image": square, "mask": smask, "mask_method": method, "scale": round(scale, 4)}


def hull_top_view(sq: dict[str, Any], size: int, bg) -> dict[str, Any]:
    """Top silhouette from the four side views (plan projection of their visual hull), drawn flat grey."""
    m = imaging.top_from_hull(sq["front"]["mask"], sq["back"]["mask"], sq["right"]["mask"], sq["left"]["mask"])
    img = Image.new("RGB", (size, size), bg)
    img.paste(Image.new("RGB", (size, size), (110, 110, 118)), (0, 0), imaging.to_mask_image(m))
    return {"image": img, "mask": m, "mask_method": "visual_hull"}


def stage2_turnaround(
    run: RunFolder, client: ModelClient, *, height_units: float = HEIGHT_UNITS, attempts: int = 4
) -> dict[str, Any]:
    st = _Stage(run, client, 2, "turnaround")
    cover = _load(run, "01-cover.png")
    tried = []
    for attempt in range(attempts):
        sheet = client.edit_image(TURNAROUND_PROMPT, [cover], size=SHEET_SIZE, attempt=attempt)
        bg = tuple(int(c) for c in imaging.background_colour(np.array(sheet)))
        thr = imaging.threshold_mask(sheet)
        fig, fig_method, n_inst = _sam_or_threshold(client, sheet)
        split = split_sheet(sheet, fig, thr, fig_method)
        sq: dict[str, Any] = {}
        size = 0
        assignment: dict[str, Any] = {}
        orient: list[dict[str, Any]] = []
        orient_notes: list[str] = []
        if split["views"]:
            named, assignment = assign_views(split["views"])
            size, sq = square_views(named, bg)
            orient, orient_notes, assignment["side_fixes"] = orient_sides(sq)
        side_checks = C.side_view_checks(
            {n: v["mask"] for n, v in sq.items()}, figures_found=len(split["runs"])
        )
        tried.append({
            "attempt": attempt,
            "sam_instances": n_inst,
            "split_source": split["split_source"],
            "column_runs": [list(r) for r in split["runs"]],
            "assignment": assignment,
            "checks": side_checks,
        })
        if C.gate_ok(side_checks):
            break
    st.save_image("02-turnaround.png", sheet)
    st.details = {
        "prompt": TURNAROUND_PROMPT,
        "sheet_size": list(sheet.size),
        "attempts": tried,
        "view_assignment": assignment,
    }
    if len(tried) > 1:
        st.notes.append(f"Sheet generated {len(tried)} times; the last one is kept.")
    st.checks = list(side_checks) + orient
    st.notes += orient_notes
    if not sq:
        st.notes.append("Could not split the sheet into four figures.")
        return st.finish()
    if not C.gate_ok(side_checks):
        # Save what was cut so the failure can be inspected, and skip the top-view call.
        for name, v in sq.items():
            st.save_image(f"views/{name}.png", v["image"])
            st.save_image(f"masks/{name}.png", imaging.to_mask_image(v["mask"]))
        return st.finish()

    # Top view. A drawn top view comes from its own call (a plan view can't share a row with standing
    # figures at one scale). It is used if its depth/width agrees with the side views; otherwise the top
    # silhouette implied by the four side views is used instead.
    side_masks = {n: v["mask"] for n, v in sq.items()}
    drawn = model_top_view(client, cover, sq, size, bg)
    st.save_image("02-top-drawn.png", drawn["raw"])
    used = None
    if drawn.get("mask") is not None:
        drawn_checks = C.top_view_checks(
            side_masks, drawn["mask"], prefix="top_drawn", gate=False,
            note="drawn top view, width scaled to the front width",
        )
        st.checks += drawn_checks
        if all(c["pass"] for c in drawn_checks):
            used = drawn
            st.details["top_method"] = IMAGE_MODEL
            st.details["top_method_detail"] = f"{IMAGE_MODEL} separate call, scaled to the front width"
    if used is None:
        used = hull_top_view(sq, size, bg)
        st.details["top_method"] = "visual_hull"
        st.details["top_method_detail"] = "visual hull of the four side views"
        st.notes.append(
            "The drawn top view did not agree with the side views, so views/top.png is the top silhouette "
            "implied by the four side views (visual hull), drawn flat grey."
        )
    st.details["top_drawn_scale"] = drawn.get("scale")
    sq["top"] = {"image": used["image"], "mask": used["mask"], "mask_method": used["mask_method"]}
    hull_note = "visual hull: agrees by construction up to side-view disagreement"
    st.checks += C.top_view_checks(
        side_masks, used["mask"], note=hull_note if used["mask_method"] == "visual_hull" else ""
    )

    for name in ALL_VIEWS:
        st.save_image(f"views/{name}.png", sq[name]["image"])
        st.save_image(f"masks/{name}.png", imaging.to_mask_image(sq[name]["mask"]))
    front_h = imaging.box_size(imaging.bbox(sq["front"]["mask"]))[1]
    meta = {
        "pixels_per_unit": round(front_h / height_units, 4),
        "height_units": height_units,
        "image_size": size,
        "views": list(ALL_VIEWS),
        "mask_method": {n: method_kind(sq[n]["mask_method"]) for n in ALL_VIEWS},
        "mask_method_detail": {n: sq[n]["mask_method"] for n in ALL_VIEWS},
        "top_method": st.details["top_method"],
    }
    run.write_json("views/meta.json", meta)
    st.outputs.append("views/meta.json")
    st.details.update({
        "image_size": size,
        "mask_method": meta["mask_method"],
        "mask_method_detail": meta["mask_method_detail"],
        "bbox_sizes_px": C.bbox_sizes({n: v["mask"] for n, v in sq.items()}),
    })
    return st.finish()
