"""Stages 0 (input), 1 (cover art) and 2 (turnaround) of the sketch-to-hero pipeline."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from herogen.pipeline import checks as C
from herogen.pipeline import imaging, inputs
from herogen.pipeline.client import (
    COST_BASIS, IMAGE_MODEL, SAM_MODEL, ApiError, ModelClient, SamInstance, call_cost,
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


class InputRejected(RuntimeError):
    """The input can't be used (no character found in the sketch, say). The stage record, written before this
    is raised, has status "input_rejected" and the reason as its error."""

    def __init__(self, stage: int, record: dict[str, Any]) -> None:
        super().__init__(record["error"])
        self.stage = stage
        self.record = record


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

    def _record(self) -> dict[str, Any]:
        calls = self.client.log.calls[self.calls0 :] if self.client else []
        live = [c for c in calls if not c["cached"]]
        return {
            "stage": self.n,
            "name": self.name,
            "ok": C.gate_ok(self.checks),
            "started": self.started,
            "seconds": round(time.monotonic() - self.t0, 2),
            "cost_usd": round(sum(call_cost(c) for c in live), 6),
            "cost_basis": COST_BASIS + "; replayed (cached) calls cost 0",
            "models": self.client.log.models(self.calls0) if self.client else [],
            "outputs": self.outputs,
            "checks": self.checks,
            "notes": " ".join(self.notes),
            "calls": calls,
            "details": self.details,
        }

    def finish(self, adjust: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        """Write the record (after ``adjust`` edits it, if given); raise `GateFailed` (after writing) when a
        gated check failed."""
        record = self._record()
        if adjust is not None:
            adjust(record)
        self.run.write_stage(record)
        if not record["ok"]:
            raise GateFailed(self.n, record)
        return record

    def fail(self, error: str, status: str = "error") -> dict[str, Any]:
        """Write the record of a stage that stopped early (``status`` "error"), rejected its input
        ("input_rejected") or is still running ("running"): not ok, with ``error`` and every call made so
        far, so paid calls are never left unrecorded."""
        record = {**self._record(), "ok": False, "status": status, "error": error}
        self.run.write_stage(record)
        return record

    def reject_input(self, error: str) -> InputRejected:
        """Write the record of a stage whose input can't be used, and return the exception to raise."""
        return InputRejected(self.n, self.fail(error, status="input_rejected"))


def _stopping_on_api_error(body: Callable[[], dict[str, Any]], st: _Stage) -> dict[str, Any]:
    """Run a stage's ``body``. A Model API failure (`ApiError`, after the client's retries) stops the stage:
    its record is written with the error and every call made so far, and the error is raised."""
    try:
        return body()
    except ApiError as e:
        st.fail(f"the Model API failed: {e}"[:800])
        raise


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
    """The contract's mask-method name for a detailed method string: "sam-3.1" or "visual_hull". Records
    written by earlier versions can also hold "threshold" (a background-threshold fallback, never made now)."""
    if method.startswith(SAM_MODEL):
        return SAM_MODEL
    if method.startswith("visual_hull"):
        return "visual_hull"
    return "threshold"


SAM_NOTHING_FOUND = f"{SAM_MODEL}:none"  # the method of an empty mask: no prompt in SAM_TEXTS found anything


def sam_instances(client: ModelClient, img: Image.Image) -> tuple[list[SamInstance], str]:
    """SAM 3.1 instances (non-empty masks only) of the first text prompt in SAM_TEXTS that finds something, and
    that prompt; ([], "") when none does. A Model API failure, a replay miss or a corrupt cache entry raises."""
    for text in SAM_TEXTS:
        inst = [i for i in client.segment(img, text) if i.mask.any()]
        if inst:
            return inst, text
    return [], ""


def sam_figure_mask(client: ModelClient, img: Image.Image) -> tuple[np.ndarray, str, int]:
    """SAM 3.1 figure mask from the first text prompt in SAM_TEXTS that finds something: (mask, method,
    instance count). When no prompt finds anything the mask is empty and the method SAM_NOTHING_FOUND; the
    caller decides what that means (no character in a sketch is an input failure; an empty generated image
    fails its gate). No mask is ever made another way: a Model API failure, a replay miss or a corrupt cache
    entry raises."""
    shape = (img.size[1], img.size[0])
    inst, text = sam_instances(client, img)
    if not inst:
        return np.zeros(shape, dtype=bool), SAM_NOTHING_FOUND, 0
    return _figure_mask(inst, shape), f"{SAM_MODEL}:{text}", len(inst)


def sam_coverage(sam: np.ndarray, reference: np.ndarray) -> float:
    """SAM mask px over background-threshold px: how much of what differs from the background SAM covered.
    The threshold mask is only this reference, never a figure mask. 0 for an empty SAM mask."""
    s = int(sam.sum())
    return 0.0 if s == 0 else s / max(1, int(reference.sum()))


def degraded_masks(record: dict[str, Any]) -> list[str]:
    """The masks a stage record was measured on that SAM 3.1 did not make (records from earlier versions,
    which fell back to background-threshold masks): "<what>: <method>" for each. Such a record is not a
    measurement and never counts as passed, whatever its `ok` says. Empty for every record made now."""
    det = record.get("details") or {}
    out = []
    methods = det.get("mask_method_detail") or det.get("mask_method") or {}
    if isinstance(methods, dict):
        out += [f"{k}: {m}" for k, m in methods.items() if method_kind(str(m)) == "threshold"]
    attempts = det.get("attempts") if record.get("stage") == 2 else None
    if attempts:  # an earlier version split a sheet on the threshold mask when SAM's columns didn't
        kept = det.get("kept_attempt", len(attempts) - 1)
        source = str(attempts[kept].get("split_source") or SAM_MODEL)
        if method_kind(source) == "threshold":
            out.append(f"split: {source}")
    for name, v in (det.get("views") or {}).items():  # stage 2h
        m = str((v or {}).get("mask_method_detail") or (v or {}).get("mask_method") or SAM_MODEL)
        if method_kind(m) == "threshold":
            out.append(f"{name}: {m}")
    return out


# ---- stage 0 ----------------------------------------------------------------------------------------


def _finish_input_checks(st: _Stage) -> None:
    """Raise `InputRejected` (after writing the record) when an input check rejected the sketch; otherwise
    add each warning's message to the record's notes."""
    if inputs.rejections(st.checks):
        raise st.reject_input(inputs.rejection_error(st.checks))
    st.notes += [f"Warning: {c['message']}." for c in inputs.warnings(st.checks)]


def stage0_input(run: RunFolder, sketch_path: Path, label: str = "") -> dict[str, Any]:
    st = _Stage(run, None, 0, "input")
    img = Image.open(sketch_path).convert("RGB")  # re-encoded: drops EXIF and other metadata
    st.save_image("00-sketch.png", img)
    w, h = img.size
    st.details = {"width": w, "height": h, "source_name": Path(sketch_path).name, "label": label}
    if label:
        st.notes.append(f"Input is a {label}.")
    st.checks = [inputs.size_check(w, h), inputs.contrast_check(img)]
    _finish_input_checks(st)
    return st.finish()


# ---- stage 1 ----------------------------------------------------------------------------------------


NO_CHARACTER = ("no character found in the sketch: SAM 3.1 found nothing for any of {texts}. HeroGen needs "
                "one clearly drawn, full-body character")


def stage1_cover(run: RunFolder, client: ModelClient) -> dict[str, Any]:
    st = _Stage(run, client, 1, "cover_art")
    return _stopping_on_api_error(lambda: _stage1(st, run, client), st)


def _sketch_part_checks(client: ModelClient, sketch: Image.Image,
                        figure: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The input checks on the main figure's body parts, each found with its own SAM 3.1 prompt (recorded in
    ``sam_prompts``)."""
    prompts = [inputs.FOOT_TEXT]
    feet = inputs.parts_on(client.segment(sketch, inputs.FOOT_TEXT), figure)
    legs = None
    if not feet:
        prompts.append(inputs.LEG_TEXT)
        legs = inputs.parts_on(client.segment(sketch, inputs.LEG_TEXT), figure)
    checks = [inputs.feet_check(feet, legs)]
    near = imaging.dilate(figure, 4)  # as the held-out views' face cue
    prompts += [inputs.FACE_TEXT, inputs.HEAD_TEXT]
    face = inputs.best_instance(client.segment(sketch, inputs.FACE_TEXT), near)
    head = inputs.best_instance(client.segment(sketch, inputs.HEAD_TEXT), near)
    facing, facing_details = inputs.facing_check(face, head)
    checks.append(facing)
    return checks, {"sam_prompts": prompts, "feet_on_figure": feet, "legs_on_figure": legs, **facing_details}


def _stage1(st: _Stage, run: RunFolder, client: ModelClient) -> dict[str, Any]:
    sketch = _load(run, "00-sketch.png")
    w, h = sketch.size
    # The input checks run first: a sketch they reject costs no paid edit. The photo check comes before every
    # other model call, so no other model (no SAM 3.1 face or head prompt) ever sees a photo of a real person.
    photo, photo_details = inputs.photo_check(inputs.judge_photo(client, sketch))
    if photo["verdict"] == "reject":
        st.checks = [photo]
        st.details = {"input": {"photo": photo_details}}
        _finish_input_checks(st)
    sketch_inst, text = sam_instances(client, sketch)
    sketch_method = f"{SAM_MODEL}:{text}" if sketch_inst else SAM_NOTHING_FOUND
    input_checks, input_details = inputs.character_checks(
        sketch_inst, NO_CHARACTER.format(texts=", ".join(repr(t) for t in SAM_TEXTS)))
    input_checks.insert(0, photo)
    input_details["photo"] = photo_details
    if sketch_inst and not inputs.rejections(input_checks):
        # Only the main (largest) figure is the character; smaller figures are warned about.
        sketch_mask = inputs.by_area(sketch_inst)[0].mask
        more_checks, more_details = _sketch_part_checks(client, sketch, sketch_mask)
        input_checks += more_checks
        input_details.update(more_details)
    st.checks = input_checks
    st.details = {"mask_method": {"sketch": sketch_method}, "sam_texts": list(SAM_TEXTS),
                  "sam_instances": {"sketch": len(sketch_inst)}, "input": input_details}
    _finish_input_checks(st)
    cover = client.edit_image(COVER_PROMPT, [sketch], size=f"{w}x{h}")
    st.save_image("01-cover.png", cover)
    # An empty cover mask (SAM found nothing in the cover) fails the gated cover_mask_area_frac check.
    cover_mask, cover_method, n_c = sam_figure_mask(client, cover)
    st.save_image("01-masks/sketch.png", imaging.to_mask_image(sketch_mask))
    st.save_image("01-masks/cover.png", imaging.to_mask_image(cover_mask))
    cover_checks, extra = C.cover_checks(sketch_mask, cover_mask)
    st.checks = input_checks + cover_checks
    st.details = {
        "prompt": COVER_PROMPT,
        "cover_size": list(cover.size),
        "mask_method": {"sketch": sketch_method, "cover": cover_method},
        "sam_instances": {"sketch": len(sketch_inst), "cover": n_c},
        "input": input_details,
        **extra,
    }
    return st.finish()


# ---- stage 2 ----------------------------------------------------------------------------------------

FACING_MIN = 0.05  # |toe offset| below this is "can't tell" and the view is left as drawn
SHEET_SIZE = "2560x1024"  # aspect only (Muse Image fixes the resolution); room for four A-pose figures


def split_sheet(sheet: Image.Image, fig_mask: np.ndarray, thr_mask: np.ndarray, fig_method: str) -> dict[str, Any]:
    """Find the four figures' column extents in the SAM mask and cut them out (`panel0`..`panel3`, left to
    right), each with its SAM mask.

    Only the SAM mask is used: when its columns don't give four figures the sheet has no views (and fails
    `figures_found`). ``thr_mask`` (background threshold) is only the reference for `sam_coverage` per view,
    which a gated check compares with C.SAM_COVERAGE_MIN, so a figure SAM missed fails the gate instead of
    being measured on another mask.
    """
    names = [f"panel{i}" for i in range(len(SIDE_VIEWS))]
    runs = imaging.split_columns(fig_mask, expected=len(names)) if fig_mask.any() else []
    out: dict[str, Any] = {"runs": runs, "split_source": fig_method, "views": {}, "sam_coverage": {}}
    if len(runs) != len(names):
        return out
    combined = np.zeros_like(fig_mask)
    for (x0, x1), name in zip(runs, names, strict=True):
        combined[:, x0:x1] = fig_mask[:, x0:x1]
        out["sam_coverage"][name] = round(sam_coverage(fig_mask[:, x0:x1], thr_mask[:, x0:x1]), 4)
    # Side views share the sheet's row range, so their feet stay on one baseline after the crop.
    boxes = [imaging.bbox(combined[:, x0:x1]) for x0, x1 in runs]
    y0 = min(b[1] for b in boxes if b)
    y1 = max(b[3] for b in boxes if b)
    for (x0, x1), name in zip(runs, names, strict=True):
        out["views"][name] = {
            "image": sheet.crop((x0, y0, x1, y1)),
            "mask": combined[y0:y1, x0:x1],
            "mask_method": fig_method,
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
    """Ask Muse Image for a top view and bring it to the shared scale (its width set to the front width).
    Its mask is SAM 3.1's; when SAM finds nothing in it, ``mask`` is None and the drawn view is not used."""
    img = client.edit_image(TOP_PROMPT, [cover], size="1024x1024")
    mask, method, _ = sam_figure_mask(client, img)
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
    return _stopping_on_api_error(lambda: _stage2(st, run, client, height_units, attempts), st)


# Stage 3's carve of a sheet's views, gated by stage 3's own silhouette-IoU gate (see `carve_check`).
CARVE_GATE = "carve_silhouette_iou_min"
# Retry/ranking gates; after exhaustion, a carve-only failure does not fail stage 2.
# `attempt_rank` counts a missing gate as failed.
SHEET_GATES = ("figures_found", "height_agreement_4_views", "front_back_width_agreement", "left_right_mirror_iou",
               "view_mask_sam_coverage_min", CARVE_GATE)


def carve_check(masks: dict[str, np.ndarray], height_units: float, *, derived_top: bool = False) -> tuple[dict[str, Any], dict[str, float]]:
    """Carve ``masks`` exactly as stage 3 does and gate the worst view's silhouette IoU with stage 3's own gate
    (`herogen.blender.stage3.IOU_GATE`): (the check, every view's IoU). A derived top constrains the
    geometry but its own IoU is not gated, matching stage 3.

    Generated views that disagree (a staff in the back view only, arms angled differently front and back, a
    profile wider than the other) carve into a body that loses whatever any view lacks, and stage 3 then
    refuses a carve fallback. Checked here, such a sheet is retried; exhaustion does not block the hybrid
    when only this check fails.
    """
    from herogen.blender import stage3 as s3
    from herogen.blender.views import views_from_masks

    ious = s3.carve_view_ious(views_from_masks(masks, height_units), height_units)
    gated = {v: x for v, x in ious.items() if v != "top" or not derived_top}
    worst = min(gated, key=gated.get)
    note = f"worst: {worst}; stage 3's carve of these views, IoU per view " + ", ".join(
        f"{v} {x:.3f}" for v, x in ious.items())
    return C.check(CARVE_GATE, ious[worst], s3.IOU_GATE, note=note), ious


def _coverage_checks(split: dict[str, Any]) -> list[dict[str, Any]]:
    cov = split["sam_coverage"]
    if not cov:
        return []
    low = min(cov, key=cov.get)
    return [C.check("view_mask_sam_coverage_min", cov[low], C.SAM_COVERAGE_MIN,
                    note=f"lowest: {low}; SAM 3.1 px over background-threshold px in each figure's columns")]


def _sheet_top(client: ModelClient, cover: Image.Image, sq: dict[str, Any], size: int, bg,
               height_units: float) -> tuple:
    """Select the top and check the final masks before ranking or accepting a sheet."""
    top_checks: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    notes: list[str] = []
    # Top view. A drawn top view comes from its own call (a plan view can't share a row with standing
    # figures at one scale). It is used if its depth/width agrees with the side views; otherwise the top
    # silhouette implied by the four side views is used instead.
    side_masks = {n: v["mask"] for n, v in sq.items()}
    drawn = model_top_view(client, cover, sq, size, bg)
    used = None
    final_carve = None
    if drawn.get("mask") is not None:
        drawn_checks = C.top_view_checks(
            side_masks, drawn["mask"], prefix="top_drawn", gate=False,
            note="drawn top view, width scaled to the front width",
        )
        top_checks += drawn_checks
        if all(c["pass"] for c in drawn_checks):
            # Stage 3 carves with a drawn top view and gates its IoU too: keep it only if that carve still passes.
            with_top, ious = carve_check({**side_masks, "top": drawn["mask"]}, height_units)
            details["carve_ious_with_drawn_top"] = ious
            if with_top["pass"]:
                used = drawn
                final_carve = with_top
                details["carve_ious_with_saved_top"] = ious
                details["top_method"] = IMAGE_MODEL
                details["top_method_detail"] = f"{IMAGE_MODEL} separate call, scaled to the front width"
            else:
                notes.append(f"Stage 3's carve with the drawn top view fails its IoU gate ({with_top['note']}), "
                                "so the drawn top view is not used.")
    if used is None:
        used = hull_top_view(sq, size, bg)
        details["top_method"] = "visual_hull"
        details["top_method_detail"] = "visual hull of the four side views"
        notes.append(
            "The drawn top view did not agree with the side views, so views/top.png is the top silhouette "
            "implied by the four side views (visual hull), drawn flat grey."
        )
    details["top_drawn_scale"] = drawn.get("scale")
    hull_note = "visual hull: agrees by construction up to side-view disagreement"
    top_checks += C.top_view_checks(
        side_masks, used["mask"], note=hull_note if used["mask_method"] == "visual_hull" else ""
    )

    if final_carve is None:
        final_carve, ious = carve_check(
            {**side_masks, "top": used["mask"]}, height_units, derived_top=True
        )
        details["carve_ious_with_saved_top"] = ious
    top_checks.append(final_carve)
    return used, drawn["raw"], top_checks, details, notes


def _stage2(st: _Stage, run: RunFolder, client: ModelClient, height_units: float, attempts: int
            ) -> dict[str, Any]:
    cover = _load(run, "01-cover.png")
    tried = []
    kept: tuple = ()
    for attempt in range(attempts):
        sheet = client.edit_image(TURNAROUND_PROMPT, [cover], size=SHEET_SIZE, attempt=attempt)
        bg = tuple(int(c) for c in imaging.background_colour(np.array(sheet)))
        thr = imaging.threshold_mask(sheet)
        fig, fig_method, n_inst = sam_figure_mask(client, sheet)
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
        ) + _coverage_checks(split)
        if sq:  # every split sheet, so `attempt_rank` compares sheets on the same gates
            side_checks.append(carve_check({n: v["mask"] for n, v in sq.items()}, height_units)[0])
        top = None
        top_raw = None
        top_details: dict[str, Any] = {}
        top_notes: list[str] = []
        # Select and measure the saved top whenever the other side gates pass, even if the preliminary
        # carve fails. The hybrid still needs complete views if no retry produces a passing carve.
        if sq and C.gate_ok([c for c in side_checks if c["name"] != CARVE_GATE]):
            top, top_raw, top_checks, top_details, top_notes = _sheet_top(
                client, cover, sq, size, bg, height_units
            )
            # Replace the preliminary side-only carve with the carve of exactly the masks saved below.
            side_checks = [c for c in side_checks if c["name"] != CARVE_GATE] + top_checks
        tried.append({
            "attempt": attempt,
            "sam_instances": n_inst,
            "split_source": split["split_source"],
            "sam_coverage": split["sam_coverage"],
            "column_runs": [list(r) for r in split["runs"]],
            "assignment": assignment,
            **top_details,
            "checks": side_checks,
        })
        # Prefer sheets that can pass with an advisory carve. Within each group keep the fewest failed
        # gates, then the smallest summed shortfall; a tie keeps the earlier one.
        non_carve_rank = C.attempt_rank(
            [c for c in side_checks if c["name"] != CARVE_GATE],
            tuple(n for n in SHEET_GATES if n != CARVE_GATE),
        )
        rank = (non_carve_rank[0] > 0, *C.attempt_rank(side_checks, SHEET_GATES))
        tried[-1]["rank"] = {"fails_non_carve_gate": rank[0], "failed_gates": rank[1], "shortfall": rank[2]}
        if not kept or rank < kept[0]:
            kept = (rank, attempt, sheet, bg, sq, size, assignment, orient, orient_notes, side_checks,
                    top, top_raw, top_details, top_notes)
        if C.gate_ok(side_checks):
            break
    (_, kept_attempt, sheet, bg, sq, size, assignment, orient, orient_notes, side_checks,
     top, top_raw, top_details, top_notes) = kept
    st.save_image("02-turnaround.png", sheet)
    st.details = {
        "prompt": TURNAROUND_PROMPT,
        "sheet_size": list(sheet.size),
        "attempts": tried,
        "kept_attempt": kept_attempt,
        "view_assignment": assignment,
        **top_details,
    }
    if len(tried) > 1:
        st.notes.append(f"Sheet generated {len(tried)} times; generation {kept_attempt + 1} is kept (sheets with "
                        "no non-carve gate failures first, then fewest failed gates, then smallest shortfall).")
    # Attempts retain their gated measurements for retries and ranking. Only the kept stage record can
    # become advisory, after retrying is over, and only when no other gated check failed.
    st.checks = [dict(c) for c in side_checks + orient]
    failed = {c["name"] for c in st.checks if c.get("gate", True) and not c["pass"]}
    if failed == {CARVE_GATE}:
        note = ("The carve of these views would fail stage 3's IoU gate, so a carve fallback would be "
                "refused. Sheet attempts are exhausted; this check does not fail stage 2 for the hybrid.")
        for c in st.checks:
            if c["name"] == CARVE_GATE:
                c["gate"] = False
                c["note"] = c.get("note", "") + "; " + note
        st.notes.append(note)
    st.notes += orient_notes + top_notes
    if top is not None:
        st.save_image("02-top-drawn.png", top_raw)
        sq["top"] = {"image": top["image"], "mask": top["mask"], "mask_method": top["mask_method"]}
    if not sq:
        st.notes.append("Could not split the sheet into four figures.")
        return st.finish()
    if top is None:
        # Save what was cut so the failure can be inspected; no top was requested for this sheet.
        for name, v in sq.items():
            st.save_image(f"views/{name}.png", v["image"])
            st.save_image(f"masks/{name}.png", imaging.to_mask_image(v["mask"]))
        return st.finish()

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
    # Front-view part masks (face, head) for the later stages; imported here because parts uses the
    # held-out module, which imports this one.
    from herogen.pipeline.parts import write_part_masks

    st.details["part_masks"] = write_part_masks(run, client)
    st.outputs += [p["file"] for p in st.details["part_masks"].values() if p.get("file")] + ["masks/parts.json"]
    st.details.update({
        "image_size": size,
        "mask_method": meta["mask_method"],
        "mask_method_detail": meta["mask_method_detail"],
        "bbox_sizes_px": C.bbox_sizes({n: v["mask"] for n, v in sq.items()}),
    })
    return st.finish()
