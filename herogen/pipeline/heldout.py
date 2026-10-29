"""Held-out 3/4 views (stage 2h): two extra images of the character that no building stage reads.

After the turnaround passes, each of two views is its own edit of the cover, drawn in the turnaround's A-pose
(the front view, redrawn on a canvas the cover's size as `heldout/pose-reference.png`, is the second input
image; it sets the pose and the framing):

- `front-left-34`: seen from about 45 degrees front-left, the character facing image left;
- `front-right-34`: seen from about 45 degrees front-right, the character facing image right.

Each view is generated up to three times until it is accepted; the best generation is kept. Its image, mask
and raw generation go to `heldout/<view>.png`, `heldout/<view>-mask.png` and `heldout/<view>-raw.png` when
it is accepted, to `heldout/review/` when a check could not decide (needs human review), and to
`heldout/rejected/` when a gated check failed, so `heldout/<view>.png` only ever holds an accepted view. All
are on the same square canvas as `views/`, at the shared pixel scale. The record is `stage-2h.json`: each
view's status, found yaw and attempts under `details.views`, the building azimuths the views must stay clear
of, and `"reference": "generated"` (a score against a generated view measures consistency, not accuracy).
Only scoring reads `heldout/` (see docs/ARCHITECTURE.md, "Held-out views").

Pose and angle are checked camera-aware: silhouette landmarks (hands, head, knees, shins, soles) are placed in
3D from the front and profile views, projected at every yaw, and the best fit gives the found yaw and the
remaining error. Facing is three independent cues (that fit, the toes, SAM 3.1 face-in-head).

Scale: the prompt asks for the figure at the size and place it has in the pose reference, which is built
here, so reference pixels are mapped to view pixels by front-view figure height / reference figure height.
The generated figure's own height is not used to scale it, which keeps the height check a measurement.

The checks and how their thresholds were set are described at each threshold below. The synthetic numbers
come from exact 3/4 silhouettes of the stage-2 test humanoid (`tests/test_herogen_pipeline_heldout_synth.py`,
run directly to print the table); the real numbers from the turnaround and the earlier rejected 3/4 view of
one generated run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from herogen.pipeline import checks as C
from herogen.pipeline import imaging
from herogen.pipeline.client import (
    IMAGE_MODEL, SAM_MODEL, ApiError, ModelClient, SamInstance,
)
from herogen.pipeline.runfolder import RunFolder
from herogen.pipeline.stages import _load, _sam_or_threshold, _Stage, method_kind

ATTEMPTS = 3  # generations per view before it is rejected
NOMINAL_YAW_DEG = 45.0
# Orthographic views of one figure at one scale have equal heights; the same 0.95 as the turnaround sheet.
HEIGHT_AGREE_MIN = C.HEIGHT_AGREE_MIN
FIGURE_AREA_MIN = 0.01

# Two input images: the cover (the character's look) and the turnaround's front view redrawn on a canvas the
# cover's size (`pose_reference`: the pose and the framing). With the cover alone the edit keeps the cover's
# arm pose, which can differ from the turnaround's A-pose the bodies are built in (measured on a generated
# run: hands at 0.55-0.65 of the figure height in the cover, 0.475 in the front view; all six cover-only
# generations were rejected on pose). A cover can also crop the figure (that run's cover cuts the legs at
# the shins), so its framing is no scale basis; with the front view as it was, the edit's framing fell
# between the two inputs' (heights 0.91-0.95 of the front view's at the cover's scale).
POSE_REFERENCE = "views/front.png"
POSE_REFERENCE_OUT = "heldout/pose-reference.png"  # the second input image as sent
_COMMON = (
    "The exact same character as in the first image: same costume, colours, proportions and armour pieces. "
    "Full body from the top of the head to the feet, standing still in exactly the pose of the second "
    "image, an A-pose like a 3D model on a turntable: arms straight and angled out and down exactly as far "
    "as in the second image, hands well away from the hips; legs straight and parallel, slightly apart, "
    "both feet flat on the ground side by side, not walking and not striding. "
    "The second image only shows the pose, seen from the front; do not copy its viewing angle. "
    "{turn} The head and face turn with the body, looking the same way as the chest and the toes. "
    "Keep the figure the same size and position in the frame as in the second image: the head and the feet "
    "at the same heights. "
    "No wind: loose cloth such as scarves, capes, coat tails and straps hangs straight down close to the "
    "body. "
    "Orthographic projection, no perspective. "
    "Plain flat white background, no shadows, no text, no labels."
)


@dataclass(frozen=True)
class HeldoutView:
    name: str
    yaw_deg: float  # camera yaw (docs/ARCHITECTURE.md cameras): +90 is the left view, -90 the right view
    facing: int  # -1: the character faces image left; +1: image right
    prompt: str


VIEWS = (
    HeldoutView("front-left-34", NOMINAL_YAW_DEG, -1, _COMMON.format(turn=(
        "Three-quarter view from the front-left: the character is turned 45 degrees toward the left edge of "
        "the image, so we see its front and its left side; its left arm is the nearer one and its toes point "
        "toward the left edge of the image."))),
    HeldoutView("front-right-34", -NOMINAL_YAW_DEG, 1, _COMMON.format(turn=(
        "Three-quarter view from the front-right: the character is turned 45 degrees toward the right edge "
        "of the image, so we see its front and its right side; its right arm is the nearer one and its toes "
        "point toward the right edge of the image."))),
)
VIEW_BY_NAME = {v.name: v for v in VIEWS}
PROMPTS = {v.name: v.prompt for v in VIEWS}


# ---- measurements ------------------------------------------------------------------------------------

REACH_ROWS = (0.20, 0.70)  # rows searched for the hands: below the head, above the knees (A-pose hands ~45%)
ARM_PLANE_ROWS = (0.25, 0.45)  # profile rows whose mid-depth is taken as the plane the A-pose arms lie in
HEAD_ROWS = 0.06  # top of the head
# Leg bands, fractions of the figure height from the top: knees, shins, soles. Straight A-pose legs keep
# their front-view x and their profile depth; a stride spreads the lower bands in depth, which a 3/4 view
# shows as width.
LEG_BANDS = {"knee": (0.68, 0.74), "shin": (0.84, 0.90), "sole": (0.96, 1.0)}
ROUND_BANDS = ("knee", "shin")  # legs have a round cross-section there; feet are box-like


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    xs = np.flatnonzero(row)
    if xs.size == 0:
        return []
    br = np.flatnonzero(np.diff(xs) > 1)
    starts = np.r_[xs[0], xs[br + 1]]
    ends = np.r_[xs[br], xs[-1]] + 1
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


def _band(mask: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, int, int]:
    """Rows ``lo``..``hi`` (fractions of the figure height from the top), their first row and the height."""
    b = imaging.bbox(mask)
    h = b[3] - b[1]
    r0, r1 = b[1] + int(lo * h), b[1] + max(int(lo * h) + 1, int(hi * h))
    return mask[r0:r1], r0, h


def _extent(rows: np.ndarray) -> tuple[float, float]:
    cols = np.flatnonzero(rows.any(axis=0))
    if cols.size == 0:
        return 0.0, 0.0
    return float(cols[0]), float(cols[-1] + 1)


@dataclass
class Landmarks:
    """Silhouette landmarks of one view, in px on the view's canvas (x to the image right) and, for the
    hands, height above the soles as a fraction of the figure height."""

    hands: list[tuple[float, float]]  # (x, z) of the leftmost and rightmost points of the reach rows
    head_x: float  # centre of the head's top rows
    legs: dict[str, tuple[float, float]]  # left and right ends of each leg band
    leg_runs: dict[str, list[tuple[int, int]]]  # each leg band's runs (one per leg when the legs are apart)
    height: int


def landmarks(mask: np.ndarray) -> Landmarks | None:
    b = imaging.bbox(mask)
    if b is None or b[3] - b[1] < 20:
        return None
    rows, r0, h = _band(mask, *REACH_ROWS)
    hands = []
    for pick in (np.min, np.max):
        per_row = [(pick(np.flatnonzero(r)), i) for i, r in enumerate(rows) if r.any()]
        best = pick([x for x, _ in per_row])
        at = [i for x, i in per_row if abs(x - best) <= 1]
        z = (b[3] - (r0 + float(np.median(at)))) / h
        hands.append((float(best) + (1.0 if pick is np.max else 0.0), z))
    hx0, hx1 = _extent(_band(mask, 0.0, HEAD_ROWS)[0])
    legs, runs = {}, {}
    for name, (lo, hi) in LEG_BANDS.items():
        band = _band(mask, lo, hi)[0]
        legs[name] = _extent(band)
        runs[name] = _runs(band.any(axis=0))
    return Landmarks(hands=hands, head_x=(hx0 + hx1) / 2, legs=legs, leg_runs=runs, height=h)


LANDMARK_NAMES = (
    ["left hand x", "right hand x", "head x"]
    + [f"{band} {end}" for band in LEG_BANDS for end in ("left end", "right end")]
    + ["left hand z", "right hand z"]
)


@dataclass
class TurnaroundModel:
    """What the turnaround says about the landmarks in 3D (x from the front view, depth y from the left
    view, + toward the character's back), enough to predict them at any yaw."""

    front: Landmarks
    arm_y: float  # depth of the plane the A-pose arms lie in
    head_y: float
    leg_y: dict[str, tuple[float, float]]  # each leg band's front and back depth

    def predict(self, yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """Predicted horizontal positions (hands, head, leg band ends; px) and the hands' heights at
        ``yaw_deg``: u = x cos + y sin, the camera of docs/ARCHITECTURE.md turned by the yaw. A knee or
        shin band is one ellipse per front-view run (its width from the run, its depth from the profile),
        a sole band each front-view run swept over the profile's toe-to-heel depth."""
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        f = self.front
        u = [f.hands[0][0] * c + self.arm_y * s, f.hands[1][0] * c + self.arm_y * s,
             f.head_x * c + self.head_y * s]
        for name in LEG_BANDS:
            y0, y1 = self.leg_y[name]
            if name in ROUND_BANDS:  # elliptic cross-section: front half-width rx, profile half-depth ry
                ry, yc = (y1 - y0) / 2, (y0 + y1) / 2
                ends = []
                for a, b in f.leg_runs[name]:
                    half = math.hypot((b - a) / 2 * c, ry * s)
                    ends += [(a + b) / 2 * c + yc * s - half, (a + b) / 2 * c + yc * s + half]
                u += [min(ends), max(ends)]
            else:  # box-like (feet): each front run swept over the band's profile depth
                ys = sorted((y0 * s, y1 * s))
                u += [min(a * c for a, _ in f.leg_runs[name]) + ys[0],
                      max(b * c for _, b in f.leg_runs[name]) + ys[1]]
        return np.array(u), np.array([f.hands[0][1], f.hands[1][1]])


def turnaround_model(front: np.ndarray, left: np.ndarray, right: np.ndarray) -> TurnaroundModel | None:
    """Landmark model from the front view and the two profiles (the right one mirrored onto the left's
    axes, and the two averaged)."""
    fl = landmarks(front)
    if fl is None or not all(fl.leg_runs.values()):
        return None
    profiles = [p for p in (left, np.fliplr(right)) if imaging.bbox(p) is not None]
    if not profiles:
        return None
    arm, head = [], []
    legs: dict[str, list[tuple[float, float]]] = {n: [] for n in LEG_BANDS}
    for p in profiles:
        rows, _, _ = _band(p, *ARM_PLANE_ROWS)
        arm.append(float(np.median([sum(_extent(r[None, :])) / 2 for r in rows if r.any()])))
        h0, h1 = _extent(_band(p, 0.0, HEAD_ROWS)[0])
        head.append((h0 + h1) / 2)
        for name, (lo, hi) in LEG_BANDS.items():
            legs[name].append(_extent(_band(p, lo, hi)[0]))
    leg_y = {n: (float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v])))
             for n, v in legs.items()}
    return TurnaroundModel(front=fl, arm_y=float(np.mean(arm)), head_y=float(np.mean(head)), leg_y=leg_y)


YAW_GRID = np.arange(-85.0, 85.5, 1.0)


def landmark_errors(model: TurnaroundModel, seen: Landmarks, yaw_deg: float) -> np.ndarray:
    """Per-landmark error at ``yaw_deg`` (order of LANDMARK_NAMES), as fractions of the front figure
    height, after the best horizontal shift (the canvas centre is not a reliable origin once a figure
    turns)."""
    u, z = model.predict(yaw_deg)
    m = [seen.hands[0][0], seen.hands[1][0], seen.head_x]
    for name in LEG_BANDS:
        m += list(seen.legs[name])
    du = np.array(m) - u
    du -= du.mean()
    h = model.front.height
    dz = (np.array([seen.hands[0][1], seen.hands[1][1]]) - z) * seen.height
    return np.concatenate([du, dz]) / h


def fit_yaw(model: TurnaroundModel, seen: Landmarks) -> dict[str, Any]:
    """Search the yaw for the least RMS landmark error, separately with the character facing image left
    (yaw > 0) and image right (yaw < 0)."""
    errs = [landmark_errors(model, seen, y) for y in YAW_GRID]
    rms = np.array([float(np.sqrt(np.mean(e * e))) for e in errs])
    out: dict[str, Any] = {}
    for side, sel in (("left", YAW_GRID > 0), ("right", YAW_GRID < 0)):
        i = int(np.argmin(np.where(sel, rms, np.inf)))
        out[f"facing_{side}"] = {"yaw_deg": float(YAW_GRID[i]), "rms": float(rms[i]),
                                 "max": float(np.max(np.abs(errs[i]))),
                                 "worst": LANDMARK_NAMES[int(np.argmax(np.abs(errs[i])))]}
    return out


def toe_offset_h(mask: np.ndarray) -> float:
    """`imaging.facing_offset` normalised by the figure height instead of its width (an A-pose width moves
    with the arms; the height doesn't): + toes point image right, - image left."""
    b = imaging.bbox(mask)
    if b is None:
        return 0.0
    w, h = imaging.box_size(b)
    return imaging.facing_offset(mask) * w / h


def _best(instances: list[SamInstance], within: np.ndarray | None = None) -> SamInstance | None:
    """The most confident instance (largest when no confidence), keeping only those mostly on ``within``."""
    keep = [i for i in instances if i.mask.any()]
    if within is not None:
        keep = [i for i in keep if (i.mask & within).sum() >= 0.5 * i.mask.sum()]
    if not keep:
        return None
    return max(keep, key=lambda i: (i.confidence if i.confidence is not None else 0.0, int(i.mask.sum())))


def face_offset(face: np.ndarray, head: np.ndarray) -> float:
    """Which way the head is turned: (face centroid x - head centroid x) / head box width; + is image
    right."""
    hb = imaging.bbox(head)
    if hb is None or not face.any():
        return 0.0
    fx = float(np.nonzero(face)[1].mean())
    hx = float(np.nonzero(head)[1].mean())
    return (fx - hx) / max(1, hb[2] - hb[0])


FACE_TEXT, HEAD_TEXT = "face", "head"


def measure_face(client: ModelClient, raw: Image.Image, figure: np.ndarray) -> dict[str, Any]:
    """SAM 3.1 `face` and `head` on the raw image; the offset of the face inside the head, or None when
    either is not found on the figure (a full helmet, say).

    Only a Model API failure (`ApiError`, after the client's retries; cached, so a replay repeats it) is
    recorded, with ``api_error`` set. Anything else is raised: a replay miss (a partial cache must not quietly
    drop a vote), a corrupt cache entry, a bug."""
    out: dict[str, Any] = {"face_found": False, "head_found": False, "offset": None}
    near = imaging.dilate(figure, 4)
    try:
        face = _best(client.segment(raw, FACE_TEXT), near)
        head = _best(client.segment(raw, HEAD_TEXT), near)
    except ApiError as e:  # anything else (a replay miss, a corrupt cache entry, a bug) is raised
        out["error"] = f"{type(e).__name__}: {e}"[:300]
        out["api_error"] = True
        return out
    out["face_found"], out["head_found"] = face is not None, head is not None
    if face is not None and head is not None and (face.mask & head.mask).sum() >= 0.5 * face.mask.sum():
        out["offset"] = face_offset(face.mask, head.mask)
        out["face_box"], out["head_box"] = list(face.box), list(head.box)
    elif face is not None and head is not None:
        out["error"] = "face not inside head"
    return out


# ---- checks ------------------------------------------------------------------------------------------
#
# Thresholds come from exact 3/4 silhouettes of the stage-2 test humanoid at yaws 20-70 degrees, for both
# views, plain and clothed (hair, scarf, backpack, a scarf tail hanging behind one leg), with and without deep
# shoulder armour, in the turnaround's A-pose (arms 50 degrees from vertical) and distorted: arms lowered to
# 35 degrees, arms hanging, legs striding 8, 12 and 25 degrees (40 cases each).
# `python3 tests/test_herogen_pipeline_heldout_synth.py` prints the table; ranges below are its min and max.
#
# Pose: RMS landmark error (`landmark_errors`) at the best yaw on the view's side, as a fraction of the
# figure height. A-pose 0.0024-0.0116 (40 of 40 accepted, found yaw within 2 degrees); arms lowered to 35
# degrees 0.026-0.052, hanging 0.038-0.109, stride 8 degrees 0.018-0.054, 12 degrees 0.016-0.070,
# 25 degrees 0.035-0.120 (none accepted: 192 rejected, 8 of the smallest strides need review). Generated
# run: the earlier 3/4 view (a stride with lowered arms, drawn facing right) measures 0.030 as a front-right
# view and 0.034 as a front-left one, rejected either way. Up to 0.0135
# passes, above 0.020 fails, between needs human review.
POSE_RMS_PASS = 0.0135
POSE_RMS_FAIL = 0.020
# Facing, three cues, each a vote of +1 (the view's way), -1 (the other way) or 0 (can't tell):
# - landmarks: RMS of the best fit facing the other way minus the best fit facing the view's way. A-pose
#   +0.0057-0.0060 at yaw 20 rising to +0.018-0.020 at 70; every mirror image is rejected. Generated run:
#   the earlier 3/4 view, drawn facing right with a stride, measures -0.003 as a front-left view (no vote;
#   its toes vote against).
FIT_FACING_MARGIN = 0.004
# - toes: toe offset (height-normalised) with the view's sign over the profile views' toe offset. A-pose
#   0.34-0.36 at yaw 20, 0.69-0.71 at 45, 0.93-0.96 at 70; 0 at the front, 0.17-0.21 at 10-12. Generated
#   turnaround: front and back views 0.00-0.01 of the profiles'.
TOE_TURN_MIN = 0.25
# - head: SAM 3.1 `face` centroid inside `head`, with the view's sign, over the same offset in the profile
#   views. Synthetic (a face on the front of the head) 0.19 at yaw 10, 0.37-0.39 at 20, 0.74-0.76 at 45.
#   Generated turnaround: front 0.008 / 0.19 = 0.04; a body turned 45 degrees with the head left nearly
#   frontal 0.22. 0.30 asks for a head turned with the body by about 16 degrees or more.
FACE_TURN_MIN = 0.30
FACE_PROFILE_MIN = 0.05  # profile face offsets below this can't serve as the reference (front-on noise 0.008)
# Facing passes when at least one cue says the view's way and none says the other way; no deciding cue
# needs human review.
# Build/evaluation split: the building cameras' azimuths (docs/ARCHITECTURE.md views). A held-out view whose
# found yaw is within 10 degrees of one of them is not held out and is rejected.
BUILD_AZIMUTHS_DEG = (0.0, 90.0, 180.0, 270.0)
AZIMUTH_CLEARANCE_DEG = 10.0


def azimuth_clearance(yaw_deg: float) -> float:
    """Angular distance from ``yaw_deg`` to the nearest building azimuth, in degrees."""
    return min(abs((yaw_deg - a + 180.0) % 360.0 - 180.0) for a in BUILD_AZIMUTHS_DEG)


def _vote(value: float | None, limit: float) -> int:
    if value is None:
        return 0
    return 1 if value >= limit else (-1 if value <= -limit else 0)


def _review(c: C.Check) -> C.Check:
    """A non-gating check whose failure sends the view to human review."""
    return {**c, "gate": False, "review": True}


def view_checks(
    view: HeldoutView, heldout: np.ndarray, model: TurnaroundModel | None, profile_toe: float,
    raw_area_frac: float, face: dict[str, Any], profile_face: float | None,
) -> tuple[list[C.Check], dict[str, Any]]:
    """The checks for one held-out view (placed at the shared scale) and the measurements behind them.

    Gated checks reject the view; checks marked ``review`` send it to human review when they fail.
    """
    out = [C.check("figure_area_frac", raw_area_frac, FIGURE_AREA_MIN)]
    seen = landmarks(heldout)
    if model is None or seen is None:
        out.append(C.check("figure_measurable", 0.0, 1.0, op="==",
                           note="no figure landmarks" if seen is None else "no turnaround landmarks"))
        return out, {}
    fh = model.front.height
    out.append(C.check("height_agreement_front", C.agreement([seen.height, fh]), HEIGHT_AGREE_MIN))
    fit = fit_yaw(model, seen)
    side, other = ("left", "right") if view.facing < 0 else ("right", "left")
    best, wrong = fit[f"facing_{side}"], fit[f"facing_{other}"]
    out.append(C.check("pose_landmark_rms", best["rms"], POSE_RMS_FAIL, op="<=",
                       note=f"at the found yaw {best['yaw_deg']:+.0f}; worst landmark: {best['worst']}"))
    out.append(_review(C.check("pose_landmark_rms_clear", best["rms"], POSE_RMS_PASS, op="<=")))
    clearance = azimuth_clearance(best["yaw_deg"])
    out.append(C.check("found_yaw_clear_of_build_azimuths_deg", clearance, AZIMUTH_CLEARANCE_DEG, op=">",
                       note=f"found yaw {best['yaw_deg']:+.0f}; building azimuths "
                            f"{list(BUILD_AZIMUTHS_DEG)}"))

    toe = toe_offset_h(heldout)
    toe_turn = view.facing * toe / profile_toe if profile_toe > 0.01 else None
    face_turn = None
    if face.get("offset") is not None and profile_face is not None:
        face_turn = view.facing * face["offset"] / profile_face
    margin = wrong["rms"] - best["rms"]
    votes = {"landmarks": _vote(margin, FIT_FACING_MARGIN), "toes": _vote(toe_turn, TOE_TURN_MIN),
             "head": _vote(face_turn, FACE_TURN_MIN)}
    against = [k for k, v in votes.items() if v < 0]
    for_ = [k for k, v in votes.items() if v > 0]
    out.append(C.check("facing_cues_against", len(against), 0, op="==",
                       note=f"cues saying the other way: {', '.join(against) or 'none'}"))
    out.append(_review(C.check("facing_cues_for", len(for_), 1,
                               note=f"cues saying the view's way: {', '.join(for_) or 'none'}")))
    measured = {
        "found_yaw_deg": best["yaw_deg"],
        "fit": fit,
        "azimuth_clearance_deg": clearance,
        "facing_fit_margin": margin,
        "toe_offset_h": toe,
        "profile_toe_offset_h": profile_toe,
        "toe_turn": toe_turn,
        "face": face,
        "profile_face_offset": profile_face,
        "face_turn": face_turn,
        "facing_votes": votes,
    }
    return out, _rounded(measured)


def _rounded(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, dict):
        return {k: _rounded(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_rounded(x) for x in v]
    return v


def view_status(checks: list[C.Check]) -> str:
    """"accepted", "needs review" (a review check failed, no gated one did) or "rejected"."""
    if not C.gate_ok(checks):
        return "rejected"
    if any(c.get("review") and not c["pass"] for c in checks):
        return "needs review"
    return "accepted"


# ---- stage -------------------------------------------------------------------------------------------


# The reference figure's height over the canvas height (less if its width would pass 0.96 of the canvas).
# Generated run: with the reference filling 0.83 of the height, all six generations filled 0.883-0.893, so
# the reference is drawn the size the model draws a full figure anyway.
POSE_REFERENCE_FILL = 0.89
POSE_REFERENCE_MAX_WIDTH = 0.96


def pose_reference(front_img: Image.Image, front_mask: np.ndarray, size: tuple[int, int]
                   ) -> tuple[Image.Image, int]:
    """The front view's figure alone on a white canvas of ``size``, centred, scaled to fill
    POSE_REFERENCE_FILL of the canvas height (or POSE_REFERENCE_MAX_WIDTH of its width, if that is
    tighter). Returns the image and the figure's height on it."""
    canvas = Image.new("RGB", size, (255, 255, 255))
    fb = imaging.bbox(front_mask)
    if fb is None:
        return canvas, 0
    fw, fh = imaging.box_size(fb)
    scale = min(POSE_REFERENCE_MAX_WIDTH * size[0] / fw, POSE_REFERENCE_FILL * size[1] / fh)
    new = (max(1, round(fw * scale)), max(1, round(fh * scale)))
    crop = front_img.crop(fb).resize(new, Image.LANCZOS)
    cmask = imaging.to_mask_image(front_mask[fb[1] : fb[3], fb[0] : fb[2]]).resize(new, Image.BILINEAR)
    canvas.paste(crop, ((size[0] - new[0]) // 2, (size[1] - new[1]) // 2), cmask)
    return canvas, new[1]


def figure_mask(client: ModelClient, img: Image.Image) -> tuple[np.ndarray, str]:
    """SAM 3.1 figure mask, the same rule as a turnaround view: kept unless SAM missed the figure (empty, or
    under half the background-threshold area), in which case the threshold mask is used."""
    sam, method, _ = _sam_or_threshold(client, img)
    if method_kind(method) != SAM_MODEL:
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


_STATUS_DIR = {"accepted": "heldout", "needs review": "heldout/review", "rejected": "heldout/rejected"}
_RANK = {"accepted": 0, "needs review": 1, "rejected": 2}


def view_paths(view: HeldoutView, status: str = "accepted") -> dict[str, str]:
    """Where a view's image, mask and raw generation are written for its status."""
    d = _STATUS_DIR[status]
    return {"image": f"{d}/{view.name}.png", "mask": f"{d}/{view.name}-mask.png",
            "raw": f"{d}/{view.name}-raw.png"}


@dataclass
class _Context:
    """What every attempt of every view is measured against."""

    cover: Image.Image
    pose: Image.Image  # the turnaround's front view in the cover's frame, the pose to draw
    req_size: str
    size: int
    bottom_row: int
    view_per_pose_px: float  # front-view px per pose-reference px
    model: TurnaroundModel | None
    profile_toe: float
    profile_face: float | None


def _generate_view(st: _Stage, client: ModelClient, view: HeldoutView, ctx: _Context, attempts: int
                   ) -> tuple[str, list[C.Check], dict[str, Any]]:
    """Generate ``view`` until it is accepted, at most ``attempts`` times, and keep the best attempt
    (accepted, else needs review, else rejected; ties go to the lower landmark error)."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    tried: list[dict[str, Any]] = []
    kept: tuple = ()
    for attempt in range(attempts):
        raw = client.edit_image(view.prompt, [ctx.cover, ctx.pose], size=ctx.req_size, attempt=attempt)
        mask, method = figure_mask(client, raw)
        scale = ctx.view_per_pose_px * ctx.pose.size[0] / raw.size[0]
        img, placed = place_on_canvas(raw, mask, scale, ctx.size, ctx.bottom_row)
        face = measure_face(client, raw, mask) if mask.any() else {"offset": None, "error": "no figure"}
        checks, measured = view_checks(view, placed, ctx.model, ctx.profile_toe, float(mask.mean()), face,
                                       ctx.profile_face)
        status = view_status(checks)
        tried.append({
            "attempt": attempt,
            "status": status,
            "raw_size": list(raw.size),
            "mask_method": method,
            "scale_view_px_per_raw_px": round(scale, 5),
            "bbox_px": list(imaging.bbox(placed) or ()),
            "measured": measured,
            "checks": checks,
        })
        side = "left" if view.facing < 0 else "right"
        key = (_RANK[status], (measured.get("fit") or {}).get(f"facing_{side}", {}).get("rms", 9.0))
        if not kept or key < kept[0]:
            kept = (key, attempt, status, checks, raw, img, placed, method)
        if status == "accepted":
            break
    _, attempt, status, checks, raw, img, placed, method = kept
    files = view_paths(view, status)
    for other, folder in _STATUS_DIR.items():  # never leave an unchecked or stale image where scoring reads
        if other != status:
            for rel in view_paths(view, other).values():
                if st.run.exists(rel):
                    (st.run.path / rel).unlink()
            d = st.run.path / folder
            if other != "accepted" and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    st.save_image(files["raw"], raw)
    st.save_image(files["image"], img)
    st.save_image(files["mask"], imaging.to_mask_image(placed))
    details = {
        "status": status,
        "accepted": status == "accepted",
        "kept_attempt": attempt,
        "found_yaw_deg": tried[attempt]["measured"].get("found_yaw_deg"),
        "prompt": view.prompt,
        "nominal_yaw_deg": view.yaw_deg,
        "facing": "image left" if view.facing < 0 else "image right",
        "files": files,
        "mask_method": method_kind(method),
        "mask_method_detail": method,
        "attempts": tried,
    }
    return status, checks, details


def _profile_face(client: ModelClient, run: RunFolder) -> tuple[float | None, dict[str, Any]]:
    """Mean |face offset| of the two profile views: how far a face sits off the head centre fully turned."""
    found: dict[str, Any] = {}
    for v in ("left", "right"):
        if not (run.exists(f"views/{v}.png") and run.exists(f"masks/{v}.png")):
            found[v] = {"offset": None, "error": "view missing"}
            continue
        found[v] = measure_face(client, _load(run, f"views/{v}.png"),
                                imaging.from_mask_image(_load(run, f"masks/{v}.png")))
    offs = [abs(f["offset"]) for f in found.values() if f.get("offset") is not None]
    ref = float(np.mean(offs)) if len(offs) == 2 else None
    return (ref if ref is not None and ref >= FACE_PROFILE_MIN else None), found


def stage2h_heldout(run: RunFolder, client: ModelClient, *, attempts: int = ATTEMPTS) -> dict[str, Any]:
    """Generate, mask and check both held-out views for a run whose stage 2 passed. Writes only `heldout/`
    and `stage-2h.json`; every earlier output is left as it is. The gate passes when both views are
    accepted; a view that needs human review or was rejected fails it."""
    s2 = run.read_stage(2)
    if not s2 or not s2.get("ok"):
        raise FileNotFoundError("stage 2 has not passed in this run folder; run stage 2 first")
    meta = run.read_json("views/meta.json")
    cover = _load(run, "01-cover.png")
    front = imaging.from_mask_image(_load(run, "masks/front.png"))
    left, right = (imaging.from_mask_image(_load(run, f"masks/{v}.png")) for v in ("left", "right"))
    fb = imaging.bbox(front)
    if fb is None:
        raise ValueError("the front mask is empty")
    pose, pose_h = pose_reference(_load(run, POSE_REFERENCE), front, cover.size)
    sketch = _load(run, "00-sketch.png") if run.exists("00-sketch.png") else cover

    st = _Stage(run, client, "2h", "heldout")  # type: ignore[arg-type]
    profile_face, profile_face_found = _profile_face(client, run)
    ctx = _Context(
        cover=cover,
        pose=pose,
        # The size stage 1 asked for; the cover keeps its aspect.
        req_size=f"{sketch.size[0]}x{sketch.size[1]}",
        size=int(meta["image_size"]),
        bottom_row=fb[3],
        view_per_pose_px=(fb[3] - fb[1]) / pose_h,
        model=turnaround_model(front, left, right),
        profile_toe=float(np.mean([abs(toe_offset_h(m)) for m in (left, right)])),
        profile_face=profile_face,
    )
    st.save_image(POSE_REFERENCE_OUT, ctx.pose)
    per_view: dict[str, Any] = {}
    for view in VIEWS:
        status, checks, details = _generate_view(st, client, view, ctx, attempts)
        st.checks += [{**c, "name": f"{view.name}.{c['name']}"} for c in checks]
        if status == "needs review":  # not passed: the gate stays closed until a person looks
            st.checks.append(C.check(f"{view.name}.status_accepted", 0.0, 1.0, op="==",
                                     note="needs human review: a check could not decide"))
        per_view[view.name] = details
        n = len(details["attempts"])
        where = f" ({details['files']['image']})" if status != "accepted" else ""
        st.notes.append(f"{view.name}: {status} after {n} generation(s), kept generation "
                        f"{details['kept_attempt'] + 1}{where}.")
    st.details = {
        "views": per_view,
        "accepted_views": [v for v, d in per_view.items() if d["status"] == "accepted"],
        "needs_review_views": [v for v, d in per_view.items() if d["status"] == "needs review"],
        "reference": "generated",
        "reference_note": ("the held-out views are generated images that can invent detail, so a score "
                           "against them measures consistency with the generated character, not accuracy"),
        "build_azimuths_deg": list(BUILD_AZIMUTHS_DEG),
        "azimuth_clearance_deg": AZIMUTH_CLEARANCE_DEG,
        "profile_toe_offset_h": round(ctx.profile_toe, 4),
        "profile_face_offset": None if profile_face is None else round(profile_face, 4),
        "profile_face": _rounded(profile_face_found),
        "inputs": ["01-cover.png", POSE_REFERENCE_OUT],
        "pose_reference_basis": f"{POSE_REFERENCE} figure centred on a canvas the cover's size, "
                                f"{round(pose_h / cover.size[1], 3)} of its height",
        "requested_size": ctx.req_size,
        "image_size": ctx.size,
        "scale_basis": "front-view figure height / pose-reference figure height (the prompt asks for the "
                       "reference's size and place)",
        "front_bbox_px": list(fb),
        "read_by": "scoring only; no building stage reads heldout/",
    }
    st.notes.append(f"Held-out views from {IMAGE_MODEL}, not used to build any body.")
    return st.finish()
