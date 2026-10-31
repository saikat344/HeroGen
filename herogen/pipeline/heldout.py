"""Held-out turned views (stage 2h): four extra images of the character that no building stage reads.

After the turnaround passes, four views are made, each its own edit of the cover drawn in the turnaround's
A-pose, with a pose reference and a mannequin turn guide (see "Three input images" below):
`front-left-30`, `front-left-60`, `front-right-30` and `front-right-60`, named by their target yaw.
Front-left views face image left, front-right views image right. The stage writes into its record the
building azimuths the run uses: the four contract views for generated runs, or the building cameras in a
bench run's `cameras.json`. A view's found yaw comes from Sapiens 2 body keypoints fitted to the turnaround's
(`herogen/pipeline/keypoints.py`); every true yaw its measured error allows (`FIT_ERROR_DEG`, from renders at
known yaws) must clear every building azimuth by 10 degrees. The generated windows are 25-35 and 55-65 degrees,
and generations aim at their middle.

Each view is generated up to four times until it is accepted; the best generation is kept. Its found yaw
must land in its window; up to 5 degrees outside needs human review. Its image, mask and raw generation go
to `heldout/<view>.png`, `heldout/<view>-mask.png` and `heldout/<view>-raw.png` when it is accepted, to
`heldout/review/` when a check could not decide (needs human review), and to `heldout/rejected/` when a gated
check failed, so `heldout/<view>.png` only ever holds an accepted view. All are on the square canvas of
`views/`, at the shared pixel scale, the feet on the front view's baseline; a figure that would pass the top or
a side is never cut: the square is padded by the same amount on every side (`place_on_canvas`, recorded per
attempt as `canvas_pad_px`). A person decides views with `review_view` (the CLI's `review`
command), which moves the files and appends the decision to the record; scoring reads only accepted views.
`stage2h_heldout(views=...)` remakes some views and keeps the others and their decisions: a kept view must
come from the current turnaround, and is judged again (`recheck_view`) if the checks changed since; the
files of a remade view a person decided on are archived, never deleted.

The record is `stage-2h.json`: each view's status (and what the checks said, `machine_status`), found yaw
and attempts under `details.views`, the human decisions under `details.decisions`, the building azimuths and
clearance (`build_azimuths_deg`, `azimuth_clearance_deg`, `fit_error_deg`), each attempt's yaw ranges,
and `"reference": "generated"` (a score against a generated view measures consistency, not
accuracy). Each view also records the turnaround it was made from, the checks it was judged under and its
image's SHA-256; `accepted_views` gives a scorer the accepted views whose image and turnaround still match.
New files are staged and swapped in only when the stage finishes; until then, and if it stops early, the
earlier record and files stay (the record says why it stopped, with every call made). Only scoring reads
`heldout/` (see docs/ARCHITECTURE.md, "Held-out views").

Pose and angle are checked camera-aware from Sapiens 2 body keypoints: the turnaround's joints are placed in 3D
from its front view and profiles and fitted to the view's over yaw (head, torso and legs; `keypoints.fit_yaw`),
which gives the found yaw; each arm's angle below horizontal, which the turn doesn't change, gives the A-pose
check. Facing is two cues: the side the keypoint fit prefers, and SAM 3.1 face-in-head. The silhouette landmark
fit is still recorded, as information.

Scale: the prompt asks for the figure at the size and place of the turn guide, which is built here at the
same scale as the pose reference (`framing_scale`), so generated pixels are mapped to view pixels by one over
that scale. The generated figure's own height is not used to scale it, which keeps the height check a
measurement.

The checks and how their thresholds were set are described at each threshold below. The synthetic numbers
come from exact turned silhouettes of the stage-2 test humanoid
(`tests/test_herogen_pipeline_heldout_synth.py`, run directly to print the tables); the real numbers from
generated runs.

Replay: a generation's request (its prompt and input images) depends on the attempts before it, through the
yaw each one was found at and whether it passed, so a cache recorded with other fit or threshold constants
can miss partway through a view; the miss is an error, never a quiet change of result.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import posixpath
import re
import shutil
import time
from datetime import datetime
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PIL import Image, ImageDraw

from herogen.pipeline import checks as C
from herogen.pipeline import imaging
from herogen.pipeline.inputs import FACE_TEXT, HEAD_TEXT, best_instance, face_offset
from herogen.pipeline import keypoints as KP
from herogen.pipeline.sapiens import COST_BASIS as SAPIENS_COST_BASIS
from herogen.pipeline.sapiens import POSE_MODEL, PoseClient, PoseError
from herogen.pipeline.client import (
    IMAGE_MODEL, SAM_MODEL, ApiError, ModelClient, ReplayMiss, SamInstance,
)
from herogen.pipeline.runfolder import RunFolder, now_iso
from herogen.pipeline.stages import _load, _Stage, degraded_masks, method_kind, sam_coverage, sam_figure_mask

ATTEMPTS = 4  # generations per view before it is given up on
# Orthographic views of one figure at one scale have equal heights; the same 0.95 as the turnaround sheet.
HEIGHT_AGREE_MIN = C.HEIGHT_AGREE_MIN
FIGURE_AREA_MIN = 0.01

# Three input images per generation: the cover (the character's look), a pose reference (the turnaround's
# front view, or its nearer profile for large turns, redrawn on a canvas the cover's size; `pose_reference`)
# and a turn guide (a plain grey mannequin of the turnaround's pose at the requested yaw; `turn_guide`), the
# last two setting the framing too. Measured on a generated run (one generation per case, found yaw by the
# landmark fit):
# - cover only: the edit keeps the cover's arms, not the turnaround's A-pose (hands at 0.55-0.65 of the
#   figure height against 0.475; six of six rejected on pose); a cover can also crop the figure, so its
#   framing is no scale basis (heights 0.91-0.95 of the front view's at the cover's scale);
# - cover + front view, asked for 30 / 60 degrees: 28, -27 / 17 and -16, the 60-degree front-left one
#   facing the wrong way; plainer turn wording 24 (wrong way), -20 / 15 (wrong way), -40; adding the nearer
#   profile 19 (wrong way), -32 / 3, -33;
# - cover + mannequin: right way every time but the pose drifted (RMS 0.025-0.106);
# - cover + front view + mannequin: right way 4 of 4, RMS 0.007-0.013, turns 26, -43 / 29, -33;
# - cover + nearer profile, asked for 60: 42 (RMS 0.048), -49 (RMS 0.011).
# The model turns less than asked, so each new attempt asks for the target plus the shortfall the last
# attempt measured (`next_request`), with the profile as the pose reference once the request reaches
# PROFILE_REFERENCE_FROM_DEG. Live on that run with this scheme: the 60-degree views turned 56-67 degrees
# from the profile reference but with the arms lower than the A-pose (RMS 0.017-0.059), so the best of each
# went to human review (where a person rejected both: arms not in the A-pose). The 30-degree views turned 32
# and 31 degrees (found by the drift-robust fit; the plain fit read 28 and 26) with the arms a little raised
# (RMS 0.0142 and 0.0136, review); a further front-right generation was accepted at -26 (RMS 0.009; plain
# fit -23, yaw range -28 to -22), two further front-left ones turned only 19 degrees.
PROFILE_REFERENCE_FROM_DEG = 50.0
MAX_REQUEST_DEG = 85.0


@dataclass(frozen=True)
class HeldoutView:
    name: str
    yaw_deg: float  # target camera yaw (docs/ARCHITECTURE.md cameras): +90 is the left view, -90 the right
    window_deg: tuple[float, float]  # accepted |found yaw| range
    facing: int  # -1: the character faces image left; +1: image right

    @property
    def side(self) -> str:
        return "left" if self.facing < 0 else "right"

    @property
    def aim_deg(self) -> float:
        """The |yaw| generations ask for: the middle of the window (the target itself can lie outside it)."""
        return (self.window_deg[0] + self.window_deg[1]) / 2


_TAIL = (
    "Legs straight and parallel, slightly apart, both feet flat on the ground side by side, not walking and "
    "not striding. The head and face turn with the body, looking the same way as the chest and the toes. "
    "No wind: loose cloth such as scarves, capes, coat tails and straps hangs straight down close to the "
    "body. Orthographic projection, no perspective. Plain flat white background, no shadows, no text, no "
    "labels."
)


def view_prompt(view: HeldoutView, request_deg: float, reference: str) -> str:
    """The prompt for one generation of ``view``, asking for ``request_deg`` degrees with the pose shown by
    ``reference`` ("front" or "profile")."""
    side, deg = view.side, int(round(request_deg))
    pose = ("its pose seen from the front: an A-pose like a 3D model on a turntable, arms straight and "
            "angled out and down, hands well away from the hips" if reference == "front" else
            f"its pose seen exactly from the {side} side, 90 degrees from the front, in the A-pose of a 3D "
            "model on a turntable")
    prompt = (
        "The exact same character as in the first image: same costume, colours, proportions and armour "
        f"pieces. The second image shows {pose}. The third image is a plain grey mannequin that shows the "
        f"exact viewing angle wanted: {deg} degrees from the front, the toes and face toward the {side} edge "
        "of the image. Draw the full character from the top of the head to the feet in the pose of the "
        "second image, turned exactly like the mannequin with its feet pointing the same way, the same size "
        f"and position in the frame as the mannequin. We see its front and its {side} side; its {side} arm "
        "is the nearer one. " + _TAIL
    )
    return prompt


def next_request(view: HeldoutView, request_deg: float, found_deg: float | None) -> float:
    """The next attempt's requested turn: the last request plus the last attempt's shortfall (the view's
    `aim_deg` minus the found yaw, both as turns toward the view's side), kept between 5 and MAX_REQUEST_DEG. No found yaw on the
    view's side (the figure faced the other way or wasn't measurable): ask the same again."""
    if found_deg is None:
        return request_deg
    return float(min(MAX_REQUEST_DEG, max(5.0, request_deg + view.aim_deg - abs(found_deg))))


# Build/evaluation split. The azimuths a body may be built or coloured from, which this stage writes into its
# record (`build_azimuths_deg`): the four contract views for generated runs. No pipeline stage builds or
# colours from their diagonals; bench runs instead use their recorded building cameras. A held-out view's
# true yaw must be at least AZIMUTH_CLEARANCE_DEG from every one of them, so a
# score on it never stands in for a score on a view a body was built from. The true yaw isn't known, only the
# found yaw and the measured error of the fit that found it (`yaw_range`); the whole range of true yaws the
# found one allows must clear.
BUILD_AZIMUTHS_DEG = (0.0, 90.0, 180.0, 270.0)
AZIMUTH_CLEARANCE_DEG = 10.0
# Found |yaw| minus true |yaw| of the body-keypoint fit (`keypoints.fit_yaw`: head, torso and legs, scores of at
# least KP_MIN_SCORE), measured on the bench-gt renders at known yaws, flat and lit, each judged against the
# character's own front and profile renders as the turnaround: 21 characters at 25, 45 and 65 degrees either side,
# and 19 of them rendered again turned 5 degrees each way (20, 30, 40, 50, 60 and 70), 592 views at 20-70 degrees.
# The 5th and 95th percentiles; the median is -1.0, and per angle the 5th percentile runs from -5.0 (70 degrees)
# to -12.0 (20). The low tail comes from three characters (a long coat, a long dress, a sci-fi suit) whose turn the
# fit reads 4-16 degrees short at every angle. The silhouette fit (`fit_yaw`) is off by 14 degrees or more on one
# view in ten at 25, 30, 60 and 65 degrees, against 3.4-7.2 for this one. Not measured on generated images, which
# have no known yaw.
FIT_ERROR_DEG = (-8.7, 2.0)
YAW_WINDOW_HALF_DEG = 5.0
# Body keypoints (Sapiens 2) scoring less are left out of every keypoint fit (on the same renders, 0.5 instead
# reads the turn no better: 6.5 degrees off at the 90th percentile against 6.2).
KP_MIN_SCORE = 0.3


def building_azimuths(run: RunFolder) -> tuple[float, ...]:
    """The run's horizontal building views. Top is perpendicular to every held-out camera and adds no yaw
    constraint. Generated runs use the contract views; bench runs record the views they used in cameras.json.
    Missing bench camera data is an error, never permission to ignore its diagonals."""
    if not run.exists("cameras.json"):
        meta = run.read_json("views/meta.json")
        if meta.get("reference") == "ground-truth":
            raise ValueError("a ground-truth run needs cameras.json to identify its building views")
        return BUILD_AZIMUTHS_DEG
    data = run.read_json("cameras.json")
    cameras = {c["view"]: c for c in data["cameras"] if c["role"] == "building"}
    names = data["building"]
    if not names or any(n not in cameras for n in names):
        raise ValueError("cameras.json must identify every building view's camera")
    azimuths = tuple(sorted({float(cameras[n]["azimuth_deg"]) % 360.0 for n in names}))
    if not all(math.isfinite(a) for a in azimuths):
        raise ValueError("building azimuths must be finite")
    return azimuths


def yaw_window(target_deg: float, build_azimuths: tuple[float, ...] = BUILD_AZIMUTHS_DEG
               ) -> tuple[float, float]:
    """The accepted |found yaw| range for a signed target: target +/- YAW_WINDOW_HALF_DEG, cut back so every
    true yaw the measured fit error allows clears this run's building azimuths by AZIMUTH_CLEARANCE_DEG."""
    sign = 1.0 if target_deg >= 0 else -1.0
    target = abs(target_deg)
    lo, hi = target - YAW_WINDOW_HALF_DEG, target + YAW_WINDOW_HALF_DEG
    for azimuth in build_azimuths:
        # Unwrap each building camera onto this side, nearest the target (including across yaw zero).
        a = target + (sign * azimuth - target + 180.0) % 360.0 - 180.0
        if a <= target:
            lo = max(lo, a + AZIMUTH_CLEARANCE_DEG + FIT_ERROR_DEG[1])
        else:
            hi = min(hi, a - AZIMUTH_CLEARANCE_DEG + FIT_ERROR_DEG[0])
    if lo > hi:
        raise ValueError(f"no yaw window around {target_deg} keeps clear of the building azimuths")
    return lo, hi


# Two targets each side, between the front and profile; generated windows are 25-35 and 55-65 degrees.
VIEWS = tuple(
    HeldoutView(f"front-{side}-{deg}", sign * deg, yaw_window(sign * deg), -sign)
    for deg in (30, 60) for side, sign in (("left", 1), ("right", -1))
)
VIEW_BY_NAME = {v.name: v for v in VIEWS}


def view_for_build(view: HeldoutView, build_azimuths: tuple[float, ...]) -> HeldoutView:
    """Use this run's clearance window for generation, judging and rechecking alike."""
    return replace(view, window_deg=yaw_window(view.yaw_deg, build_azimuths))


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


# Arm-angle drift: a generated view often holds the arms a few degrees higher or lower than the turnaround's
# A-pose. A hand then moves along the circle about its shoulder, so its height changes with its distance from
# the body, and the fit, which reads the turn from how far apart the hands look, reads a wrong yaw (raised
# arms look wider, so less turned). The hands' heights don't change with the yaw, so the drift-robust fit
# moves each predicted hand along the tangent of that circle to the height it is seen at: the tangent is
# perpendicular to the forearm's outer edge in the front view (ARM_EDGE_ROWS above the hand), so no shoulder
# position is needed. Moves larger than ARM_DRIFT_MAX_H of the figure height (about 12 degrees of drift for an
# A-pose arm) are cut there, and an arm steeper than ARM_SLOPE_MIN (px across per px down; near-hanging) is
# not moved. The pose error is still measured against the turnaround's own arms (`fit_yaw`), so a drifted
# arm costs as much as before.
ARM_EDGE_ROWS = 0.15
ARM_DRIFT_MAX_H = 0.06
ARM_SLOPE_MIN = 0.2


def arm_slopes(front: np.ndarray, hands: list[tuple[float, float]]) -> list[float]:
    """Each hand's forearm direction in the front view as px across per px down (outward: - for the
    image-left hand, + for the image-right one), from a line fitted to the outer edge of the ARM_EDGE_ROWS of
    the figure height above the hand; 0 when it can't be measured."""
    b = imaging.bbox(front)
    h = b[3] - b[1]
    out = []
    for (hx, hz), pick in zip(hands, (np.min, np.max), strict=True):
        r_hand = int(round(b[3] - hz * h))
        rows = range(max(b[1], r_hand - int(ARM_EDGE_ROWS * h)), min(b[3], r_hand + 1))
        pts = [(r, float(pick(np.flatnonzero(front[r])))) for r in rows if front[r].any()]
        if len(pts) < 3:
            out.append(0.0)
            continue
        r_, x_ = np.array(pts).T
        slope = float(np.polyfit(r_, x_, 1)[0])
        out.append(slope if slope * (1 if pick is np.max else -1) >= ARM_SLOPE_MIN else 0.0)
    return out


@dataclass
class TurnaroundModel:
    """What the turnaround says about the landmarks in 3D (x from the front view, depth y from the left
    view, + toward the character's back), enough to predict them at any yaw."""

    front: Landmarks
    arm_y: float  # depth of the plane the A-pose arms lie in
    head_y: float
    leg_y: dict[str, tuple[float, float]]  # each leg band's front and back depth
    arm_slope: tuple[float, float] = (0.0, 0.0)  # `arm_slopes` of the image-left and image-right hands
    hand_z_morph: float = 0.0  # `hand_z_morph` of the front view (information only)

    def hand_xs(self, seen_z: tuple[float, float] | None = None) -> list[float]:
        """The front-view hands' x, or with ``seen_z`` (the hands' heights in a view, fractions of its
        figure height) each hand moved along its arm's tangent to that height (see ARM_DRIFT_MAX_H)."""
        xs = [x for x, _ in self.front.hands]
        if seen_z is None:
            return xs
        h = self.front.height
        for i, ((_, z), slope) in enumerate(zip(self.front.hands, self.arm_slope, strict=True)):
            if slope:
                rows_up = float(np.clip((seen_z[i] - z) * h, -ARM_DRIFT_MAX_H * h, ARM_DRIFT_MAX_H * h))
                xs[i] += rows_up / slope  # up the arm's tangent: outward for a raised arm
        return xs

    def predict(self, yaw_deg: float, seen_z: tuple[float, float] | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
        """Predicted horizontal positions (hands, head, leg band ends; px) and the hands' heights at
        ``yaw_deg``: u = x cos + y sin, the camera of docs/ARCHITECTURE.md turned by the yaw. A knee or
        shin band is one ellipse per front-view run (its width from the run, its depth from the profile),
        a sole band each front-view run swept over the profile's toe-to-heel depth. ``seen_z``: predict the
        hands for arms held at those heights (`hand_xs`); their predicted heights are then those too."""
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        f = self.front
        hx = self.hand_xs(seen_z)
        u = [hx[0] * c + self.arm_y * s, hx[1] * c + self.arm_y * s, f.head_x * c + self.head_y * s]
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
        z = seen_z if seen_z is not None else (f.hands[0][1], f.hands[1][1])
        return np.array(u), np.array(z)


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
    return TurnaroundModel(front=fl, arm_y=float(np.mean(arm)), head_y=float(np.mean(head)), leg_y=leg_y,
                           arm_slope=tuple(arm_slopes(front, fl.hands)), hand_z_morph=hand_z_morph(front, fl))


YAW_GRID = np.arange(-85.0, 85.5, 1.0)


def landmark_errors(model: TurnaroundModel, seen: Landmarks, yaw_deg: float, *, drift: bool = False
                    ) -> np.ndarray:
    """Per-landmark error at ``yaw_deg`` (order of LANDMARK_NAMES), as fractions of the front figure
    height, after the best horizontal shift (the canvas centre is not a reliable origin once a figure
    turns). ``drift``: the hands predicted for arms held at the heights seen (`TurnaroundModel.hand_xs`), so
    an arm-angle drift doesn't read as a turn; the hand heights' errors are then 0."""
    seen_z = (seen.hands[0][1], seen.hands[1][1])
    u, z = model.predict(yaw_deg, seen_z if drift else None)
    m = [seen.hands[0][0], seen.hands[1][0], seen.head_x]
    for name in LEG_BANDS:
        m += list(seen.legs[name])
    du = np.array(m) - u
    du -= du.mean()
    h = model.front.height
    dz = (np.array(seen_z) - z) * seen.height
    return np.concatenate([du, dz]) / h


def _rms(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e * e)))


def fit_yaw(model: TurnaroundModel, seen: Landmarks) -> dict[str, Any]:
    """Fit the yaw separately with the character facing image left (yaw > 0) and image right (yaw < 0).

    The found yaw (``yaw_deg``) is the least-RMS yaw of the drift-robust fit (`landmark_errors` with
    ``drift``); the pose error there (``rms``, ``max``, ``worst``) is measured against the turnaround's own
    arms, so it is never below the plain fit's minimum (``plain_rms``, at ``plain_yaw_deg``), which the
    facing margin compares."""
    plain = np.array([_rms(landmark_errors(model, seen, y)) for y in YAW_GRID])
    robust = np.array([_rms(landmark_errors(model, seen, y, drift=True)) for y in YAW_GRID])
    out: dict[str, Any] = {}
    for side, sel in (("left", YAW_GRID > 0), ("right", YAW_GRID < 0)):
        i = int(np.argmin(np.where(sel, robust, np.inf)))
        j = int(np.argmin(np.where(sel, plain, np.inf)))
        e = landmark_errors(model, seen, float(YAW_GRID[i]))
        out[f"facing_{side}"] = {"yaw_deg": float(YAW_GRID[i]), "rms": _rms(e),
                                 "max": float(np.max(np.abs(e))),
                                 "worst": LANDMARK_NAMES[int(np.argmax(np.abs(e)))],
                                 "drift_rms": float(robust[i]),
                                 "plain_yaw_deg": float(YAW_GRID[j]), "plain_rms": float(plain[j])}
    return out


# The silhouette fit above no longer decides a view's turn (see FIT_ERROR_DEG and `yaw_range`); it is recorded.
# How far the hands' heights move when a mask is eroded or dilated by HAND_Z_MORPH_PX is recorded for
# information only: it is close to zero by construction (a hand's lowest-reaching row hardly moves when its
# outline does), so it can't measure the error above.
HAND_Z_MORPH_PX = (1, 2)


def _hand_zs(lm: Landmarks) -> np.ndarray:
    return np.array([z for _, z in lm.hands])


def hand_z_morph(mask: np.ndarray, lm: Landmarks | None = None) -> float:
    """RMS change of the hands' heights (fractions of the figure height) when ``mask`` is eroded or dilated
    by each of HAND_Z_MORPH_PX pixels (information only; see above)."""
    lm = lm or landmarks(mask)
    if lm is None:
        return 0.0
    z0, dev = _hand_zs(lm), []
    for r in HAND_Z_MORPH_PX:
        for op in (imaging.erode, imaging.dilate):
            other = landmarks(op(mask, r))
            if other is not None:
                dev.append(_hand_zs(other) - z0)
    return float(np.sqrt(np.mean(np.square(dev)))) if dev else 0.0


def yaw_range(found_deg: float) -> dict[str, Any]:
    """The true-yaw range a found yaw allows, from the keypoint fit's measured error FIT_ERROR_DEG (found |yaw|
    minus true |yaw|): |found| minus the upper error to |found| minus the lower one, signed back to the found
    yaw's side."""
    sign = 1.0 if found_deg >= 0 else -1.0
    lo, hi = abs(found_deg) - FIT_ERROR_DEG[1], abs(found_deg) - FIT_ERROR_DEG[0]
    return {"found_deg": [found_deg, found_deg], "true_deg": sorted([sign * lo, sign * hi])}


def range_clearance(lo_deg: float, hi_deg: float, build_azimuths: tuple[float, ...] = BUILD_AZIMUTHS_DEG
                    ) -> float:
    """Angular distance from the yaw range ``lo_deg``..``hi_deg`` to the nearest building azimuth (0 when one
    is inside it), in degrees."""
    if any(math.ceil((lo_deg - a) / 360.0) <= math.floor((hi_deg - a) / 360.0) for a in build_azimuths):
        return 0.0
    return min(azimuth_clearance(lo_deg, build_azimuths), azimuth_clearance(hi_deg, build_azimuths))


def toe_offset_h(mask: np.ndarray) -> float:
    """`imaging.facing_offset` normalised by the figure height instead of its width (an A-pose width moves
    with the arms; the height doesn't): + toes point image right, - image left."""
    b = imaging.bbox(mask)
    if b is None:
        return 0.0
    w, h = imaging.box_size(b)
    return imaging.facing_offset(mask) * w / h


def measure_face(client: ModelClient, raw: Image.Image, figure: np.ndarray) -> dict[str, Any]:
    """SAM 3.1 `face` and `head` on the raw image; the offset of the face inside the head, or None when
    either is not found on the figure (a full helmet, say).

    Only a Model API failure (`ApiError`, after the client's retries; never cached, so a replay of the run has
    no entry for it and stops) is recorded, with ``api_error`` set, and `view_checks` then sends the view to
    human review rather than judge its facing without the cue: the one Model API failure that does not stop a
    stage, since it can only keep a view from being accepted. Anything else is raised: a replay miss (a
    partial cache must not quietly drop a vote), a corrupt cache entry, a bug."""
    out: dict[str, Any] = {"face_found": False, "head_found": False, "offset": None}
    near = imaging.dilate(figure, 4)
    try:
        face = best_instance(client.segment(raw, FACE_TEXT), near)
        head = best_instance(client.segment(raw, HEAD_TEXT), near)
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


def _box(mask: np.ndarray) -> list[float] | None:
    b = imaging.bbox(mask)
    return None if b is None else [float(v) for v in b]


def measure_keypoints(pose: PoseClient, img: Image.Image, mask: np.ndarray) -> dict[str, Any]:
    """Sapiens 2 keypoints of the figure in ``img`` (its person box: ``mask``'s bounding box): the body joints
    `keypoints.POSE_JOINTS` as [x, y, score] in ``img``'s px, and the mask's bottom edge row (the soles).

    A Sapiens failure (`PoseError`, after the client's tries) is recorded with ``api_error`` set instead of
    stopping the stage. Anything else is raised (a replay miss, a bug)."""
    box = _box(mask)
    if box is None:
        return {"joints": None, "error": "no figure"}
    try:
        kps = pose.keypoints(img, box)
    except PoseError as e:
        return {"joints": None, "error": f"{type(e).__name__}: {e}"[:300], "api_error": True}
    return {"joints": {n: [round(v, 3) for v in kps[n]] for n in KP.POSE_JOINTS if n in kps},
            "box": box, "bottom": box[3]}


def _seen(kp: dict[str, Any] | None, scale: float) -> dict[str, tuple[float, float, float]] | None:
    """A measured view's joints in turnaround px (`keypoints.to_uz`), the low-scoring ones dropped."""
    if not kp or not kp.get("joints"):
        return None
    return KP.to_uz({n: tuple(v) for n, v in kp["joints"].items()}, kp["bottom"], scale, KP_MIN_SCORE)


def body_model(turnaround_kp: dict[str, Any] | None) -> KP.BodyModel | None:
    """The turnaround's 3D joint model from its recorded keypoints (`turnaround_keypoints`); None when they are
    missing or too few."""
    if not turnaround_kp or any(not (turnaround_kp.get(v) or {}).get("joints") for v in ("front",)):
        return None
    f = turnaround_kp["front"]
    side = {v: ({n: tuple(x) for n, x in turnaround_kp[v]["joints"].items()}, turnaround_kp[v]["bottom"])
            if (turnaround_kp.get(v) or {}).get("joints") else (None, None) for v in ("left", "right")}
    return KP.body_model({n: tuple(x) for n, x in f["joints"].items()}, side["left"][0], side["right"][0],
                         front_bottom=f["bottom"], front_height=turnaround_kp["front_height"],
                         left_bottom=side["left"][1], right_bottom=side["right"][1], min_score=KP_MIN_SCORE)


def turnaround_keypoints(pose: PoseClient, views: dict[str, Image.Image], masks: dict[str, np.ndarray]
                         ) -> dict[str, Any]:
    """`measure_keypoints` of the turnaround's front, left and right views, and the front figure's height."""
    out: dict[str, Any] = {v: measure_keypoints(pose, views[v], masks[v]) for v in ("front", "left", "right")}
    fb = imaging.bbox(masks["front"])
    out["front_height"] = float(fb[3] - fb[1]) if fb else 0.0
    return out


# ---- checks ------------------------------------------------------------------------------------------
#
# Thresholds come from exact 3/4 silhouettes of the stage-2 test humanoid at yaws 20-70 degrees, for both
# views, plain and clothed (hair, scarf, backpack, a scarf tail hanging behind one leg), with and without deep
# shoulder armour, in the turnaround's A-pose (arms 50 degrees from vertical) and distorted: arms lowered to
# 35 degrees, arms hanging, legs striding 8, 12 and 25 degrees (40 cases each).
# `python3 tests/test_herogen_pipeline_heldout_synth.py` prints the table; ranges below are its min and max.
#
# A-pose: each arm's angle below horizontal (`keypoints.arm_drop_deg`: how far the wrist hangs below the shoulder
# over the arm's length in the front view, which a turn about the vertical axis doesn't change). The nearer arm
# (the view's side) must be found and inside ARM_DROP_DEG; the farther one too when both its shoulder and wrist
# score at least ARM_FAR_MIN_SCORE (at 60 degrees it is often hidden). Set on the tuning split of the labelled
# bench views (viewlabels-ai-v2, A-pose label; tuning s001-s028 and s044-s045, test s029-s043): false rejects
# 0.02 / 0.13 (30 degrees, tuning / test) and 0.00 / 0.14 (60), false accepts 0.60 / 0.67 (30: 10 and 3 views
# labelled not A-pose, mostly at ordinary arm angles, so for other reasons) and 0.07 / 0.03 (60), against
# 0.84 / 0.81 and 0.88 / 0.86 false rejects, 0.00 / 0.00 and 0.07 / 0.03 false accepts for the silhouette
# landmark RMS it replaces. On the bench-gt renders (every arm posed at 45 degrees) 8-10% of views fall outside
# the band (stylised figures whose wrist keypoints miss), none of them by a wide margin.
ARM_DROP_DEG = (35.0, 72.0)
ARM_FAR_MIN_SCORE = 0.5
# The keypoints' whole-pose RMS at the found yaw (`keypoints.pose_rms`, arms and legs, across and up) is recorded
# but not judged: no threshold separated the labelled views (at 0.06 of the height it falsely rejected 0.25 of
# A-pose views and accepted 0.50-0.78 of the others). The silhouette landmark fit (`fit_yaw`) is recorded too.
# Facing, two cues, each a vote of +1 (the view's way), -1 (the other way) or 0 (can't tell):
# - keypoints: RMS of the body-keypoint fit facing the other way minus facing the view's way (`keypoints.fit_yaw`,
#   a fraction of the figure height; the nose, eyes and toes ahead of the body give the side). Bench-gt renders
#   with the side known: 5th percentile +0.0034 at 25 degrees, +0.0115 at 45, +0.030 at 65; 1 of 217 below 0.
#   On the labelled bench views (viewlabels-ai-v2, facing label), with the head cue: false rejects 0.02 / 0.04
#   (30 degrees, tuning / test split) and 0.00 / 0.03 (60), false accepts 0.12 / 0.14 and 0.09 / 0.42 (5 of 12),
#   against 0.30 / 0.56 and 0.28 / 0.41 false rejects, 0.12 / 0.14 and 0.14 / 0.33 false accepts for the three
#   cues it replaces. The silhouette fit's margin and the toes are recorded only: the toes alone falsely
#   rejected 0.28-0.52 of the views facing the right way, and accepted 0.41-0.86 of those facing the wrong way.
KP_FACING_MARGIN = 0.003
# - head: SAM 3.1 `face` centroid inside `head`, with the view's sign, over the same offset in the profile
#   views. Synthetic (a face on the front of the head) 0.19 at yaw 10, 0.37-0.39 at 20, 0.74-0.76 at 45.
#   Generated turnaround: front 0.008 / 0.19 = 0.04; a body turned 45 degrees with the head left nearly
#   frontal 0.22. 0.30 asks for a head turned with the body by about 16 degrees or more.
FACE_TURN_MIN = 0.30
FACE_PROFILE_MIN = 0.05  # profile face offsets below this can't serve as the reference (front-on noise 0.008)
# Facing passes when at least one cue says the view's way and none says the other way; no deciding cue
# needs human review.
# Front or back: every view must also show a face. The keypoint fit only looks at the front half (-89 to +89
# degrees), and a back view whose left and right joints are named the other way round projects exactly like a
# front view (u = -x cos(180 - y) + d sin(180 - y) = x cos y + d sin y; the joint model gives each pair one
# depth), so a back view at 154 degrees with the names swapped reads +26 with no error, and one at 120 reads
# +60. Sapiens swaps the names on about half of back views. A face SAM 3.1 finds inside the head is the
# positive evidence that the view is the front: without one the view needs human review (`face_seen`). On the
# labelled bench views a face was found on 78% (30 degrees) and 76% (60) of those facing the right way.
# The view's own yaw window (`yaw_window`): inside it the view can be accepted; up to 5 degrees outside it
# needs human review; further out it is rejected (and generated again). The found yaw is the body-keypoint fit's
# on the view's side (the facing checks judge the side).
YAW_WINDOW_REVIEW_DEG = 5.0


def window_miss(view: HeldoutView, yaw_deg: float) -> float:
    """Degrees by which ``yaw_deg`` falls outside the view's window (0 inside); facing the other way is
    not counted here (the facing checks reject it)."""
    lo, hi = view.window_deg
    y = abs(yaw_deg)
    return max(0.0, lo - y, y - hi)


def azimuth_clearance(yaw_deg: float, build_azimuths: tuple[float, ...] = BUILD_AZIMUTHS_DEG) -> float:
    """Angular distance from ``yaw_deg`` to the nearest building azimuth, in degrees."""
    return min(abs((yaw_deg - a + 180.0) % 360.0 - 180.0) for a in build_azimuths)


def _vote(value: float | None, limit: float) -> int:
    if value is None:
        return 0
    return 1 if value >= limit else (-1 if value <= -limit else 0)


def _review(c: C.Check) -> C.Check:
    """A non-gating check whose failure sends the view to human review."""
    return {**c, "gate": False, "review": True}


def view_checks(
    view: HeldoutView, heldout: np.ndarray, model: TurnaroundModel | None, profile_toe: float,
    raw_area_frac: float, face: dict[str, Any], profile_face: float | None, *,
    profile_face_error: bool = False, mask_sam_coverage: float | None = None,
    body: KP.BodyModel | None = None, keypoints: dict[str, Any] | None = None, kp_scale: float = 1.0,
    build_azimuths: tuple[float, ...] = BUILD_AZIMUTHS_DEG,
) -> tuple[list[C.Check], dict[str, Any]]:
    """The checks for one held-out view (placed at the shared scale) and the measurements behind them.

    Gated checks reject the view; checks marked ``review`` send it to human review when they fail.
    ``body``: the turnaround's joint model (`body_model`); ``keypoints``: the view's `measure_keypoints`, in
    the px of the image they were measured on, ``kp_scale`` turnaround px per image px. Their fit
    (`keypoints.fit_yaw`) gives the found yaw; without either, or with too few joints, the turn can't be
    measured and the view needs review.
    ``profile_face_error``: the profile views' face cue failed on the Model API, so the head cue could not
    vote (needs review, like a failure on this view's face). ``mask_sam_coverage``: the figure mask's
    `sam_coverage`; under C.SAM_COVERAGE_MIN SAM missed the figure, the view is rejected and nothing else is
    measured on its mask (None only for a view recorded before coverage was, whose SAM mask met the same rule).
    """
    view = view_for_build(view, build_azimuths)
    out = [C.check("figure_area_frac", raw_area_frac, FIGURE_AREA_MIN)]
    if mask_sam_coverage is not None:
        out.append(C.check("figure_mask_sam_coverage", mask_sam_coverage, C.SAM_COVERAGE_MIN,
                           note="SAM 3.1 px over background-threshold px"))
        if mask_sam_coverage < C.SAM_COVERAGE_MIN:
            return out, {"mask_sam_coverage": round(mask_sam_coverage, 4)}
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
    kseen = _seen(keypoints, kp_scale)
    kfit = KP.fit_yaw(body, kseen) if body is not None and kseen is not None else None
    found = None
    if kfit is None:
        why = ((keypoints or {}).get("error") or ("no turnaround keypoints" if body is None else
                                                  "too few body keypoints found"))
        out.append(_review(C.check("turn_measured", 0.0, 1.0, op="==",
                                   note=f"the turn could not be measured: {why}")))
        rng, clearance, miss = None, None, None
    else:
        found = kfit[f"facing_{side}"]["yaw_deg"]
        rng = yaw_range(found)
        clearance = range_clearance(*rng["true_deg"], build_azimuths)
        lo_t, hi_t = rng["true_deg"]
        out.append(C.check("found_yaw_clear_of_build_azimuths_deg", clearance, AZIMUTH_CLEARANCE_DEG, op=">=",
                           note=f"found yaw {found:+.1f} (body keypoints), true yaw {lo_t:+.1f} to {hi_t:+.1f} "
                                f"(fit error {list(FIT_ERROR_DEG)}); building azimuths {list(build_azimuths)}"))
        miss = window_miss(view, found)
        lo, hi = view.window_deg
        note = f"found yaw {found:+.1f}, window {lo:.1f}-{hi:.1f}"
        out.append(C.check("found_yaw_window_miss_deg", miss, YAW_WINDOW_REVIEW_DEG, op="<=", note=note))
        out.append(_review(C.check("found_yaw_in_window", miss, 0.0, op="<=", note=note)))
    arms = arm_angles(body, kseen, side)
    lo_a, hi_a = ARM_DROP_DEG
    if arms["near"] is None:
        out.append(_review(C.check("arm_angle_measured", 0.0, 1.0, op="==",
                                   note=f"the {side} (nearer) arm's shoulder or wrist was not found")))
    for which in ("near", "far"):
        a = arms[which]
        if a is not None:
            out.append(C.check(f"arm_angle_{which}_off_band_deg", max(0.0, lo_a - a, a - hi_a), 0.0, op="<=",
                               note=f"{which}er arm {a:.0f} degrees below horizontal; "
                                    f"A-pose band {lo_a:.0f}-{hi_a:.0f}"))
    kpose = (KP.pose_rms(KP.pose_residuals(body, kseen, found))
             if found is not None and body is not None and kseen is not None else None)

    toe = toe_offset_h(heldout)
    toe_turn = view.facing * toe / profile_toe if profile_toe > 0.01 else None
    face_turn = None
    if face.get("offset") is not None and profile_face is not None:
        face_turn = view.facing * face["offset"] / profile_face
    margin = wrong["plain_rms"] - best["plain_rms"]
    kmargin = None if kfit is None else kfit[f"facing_{other}"]["rms"] - kfit[f"facing_{side}"]["rms"]
    votes = {"keypoints": _vote(kmargin, KP_FACING_MARGIN), "head": _vote(face_turn, FACE_TURN_MIN)}
    against = [k for k, v in votes.items() if v < 0]
    for_ = [k for k, v in votes.items() if v > 0]
    out.append(C.check("facing_cues_against", len(against), 0, op="==",
                       note=f"cues saying the other way: {', '.join(against) or 'none'}"))
    out.append(_review(C.check("facing_cues_for", len(for_), 1,
                               note=f"cues saying the view's way: {', '.join(for_) or 'none'}")))
    if face.get("api_error") or profile_face_error:
        out.append(_review(C.check("head_cue_measured", 0.0, 1.0, op="==",
                                   note="the Model API failed on the face or head, so the head cue could not "
                                        "vote")))
    face_seen = face.get("offset") is not None
    if not face_seen and not face.get("api_error"):
        out.append(_review(C.check("face_seen", 0.0, 1.0, op="==",
                                   note="no face found inside the head: nothing shows the view is the front "
                                        "rather than a back view (keypoints with left and right swapped fit a "
                                        "back view as a front one)")))
    measured = {
        "found_yaw_deg": found,
        "keypoint_fit": kfit,
        "arm_angles_deg": arms,
        "keypoint_pose_rms": kpose,
        "keypoints": keypoints,
        "fit": fit,
        "hand_z_morph": {"view": hand_z_morph(heldout, seen), "front": model.hand_z_morph,
                         "note": "information only; close to zero by construction"},
        "true_yaw_range_deg": rng["true_deg"] if rng else None,
        "azimuth_clearance_deg": clearance,
        "yaw_window_miss_deg": miss,
        "facing_fit_margin": margin,  # the silhouette fit's (information only)
        "facing_keypoint_margin": kmargin,
        "toe_offset_h": toe,
        "profile_toe_offset_h": profile_toe,
        "toe_turn": toe_turn,
        "face": face,
        "profile_face_offset": profile_face,
        "face_turn": face_turn,
        "facing_votes": votes,
        "face_seen": face_seen,
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


def arm_angles(body: KP.BodyModel | None, seen: dict[str, tuple[float, float, float]] | None, side: str
               ) -> dict[str, float | None]:
    """`keypoints.arm_drop_deg` of the nearer arm (``side``, the view's side) and of the farther one, which counts
    only when its shoulder and wrist both score at least ARM_FAR_MIN_SCORE; None for an arm not measured."""
    out: dict[str, float | None] = {"near": None, "far": None}
    if body is None or seen is None:
        return out
    other = "right" if side == "left" else "left"
    out["near"] = KP.arm_drop_deg(body, seen, side)
    if all(seen.get(f"{other}_{j}", (0, 0, 0))[2] >= ARM_FAR_MIN_SCORE for j in ("shoulder", "wrist")):
        out["far"] = KP.arm_drop_deg(body, seen, other)
    return out


def view_status(checks: list[C.Check]) -> str:
    """"accepted", "needs review" (a review check failed, no gated one did) or "rejected"."""
    if not C.gate_ok(checks):
        return "rejected"
    if any(c.get("review") and not c["pass"] for c in checks):
        return "needs review"
    return "accepted"


# ---- stage -------------------------------------------------------------------------------------------


# Framing: the pose reference and the turn guide are drawn at one scale (`framing_scale`, canvas px per
# turnaround px), the figure POSE_REFERENCE_FILL of the canvas height, less if the front figure, a profile or
# the guide at the requested yaw would pass POSE_REFERENCE_MAX_WIDTH of the canvas width. The model follows
# the guide's size and place, and the reference agrees with it, so a generated figure's px map back to the
# turnaround's by 1 / that scale. Generated run: with the reference filling 0.83 of the height, all six
# generations filled 0.883-0.893, so the reference is drawn the size the model draws a full figure anyway.
POSE_REFERENCE_FILL = 0.89
POSE_REFERENCE_MAX_WIDTH = 0.96


def framing_scale(front_h: int, widths: list[float], size: tuple[int, int]) -> float:
    """Canvas px per turnaround px for a figure ``front_h`` tall whose drawings are ``widths`` wide
    (turnaround px), on a canvas of ``size``."""
    return min([POSE_REFERENCE_FILL * size[1] / front_h]
               + [POSE_REFERENCE_MAX_WIDTH * size[0] / w for w in widths if w > 0])


def pose_reference(view_img: Image.Image, view_mask: np.ndarray, size: tuple[int, int], scale: float
                   ) -> tuple[Image.Image, int]:
    """A turnaround view's figure alone on a white canvas of ``size``, scaled by ``scale`` (`framing_scale`),
    centred. Returns the image and the figure's height on it."""
    canvas = Image.new("RGB", size, (255, 255, 255))
    fb = imaging.bbox(view_mask)
    if fb is None:
        return canvas, 0
    fw, fh = imaging.box_size(fb)
    new = (max(1, round(fw * scale)), max(1, round(fh * scale)))
    crop = view_img.crop(fb).resize(new, Image.LANCZOS)
    cmask = imaging.to_mask_image(view_mask[fb[1] : fb[3], fb[0] : fb[2]]).resize(new, Image.BILINEAR)
    canvas.paste(crop, ((size[0] - new[0]) // 2, (size[1] - new[1]) // 2), cmask)
    return canvas, new[1]


GUIDE_GREY = (150, 150, 150)
# (kind, points (u px, fraction of the height), width px)
Shape = tuple[str, list[tuple[float, float]], float]


def _aligned_profile(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The two profiles' common silhouette: the right one mirrored and shifted sideways onto the left one's
    box centre first (the rows already share the turnaround's baseline), so an offset doesn't shrink it.
    Falls back to whichever profile has a figure."""
    flipped = np.fliplr(right)
    bl, br = imaging.bbox(left), imaging.bbox(flipped)
    if bl is None or br is None:
        return left if bl is not None else flipped
    dx = int(round((bl[0] + bl[2]) / 2 - (br[0] + br[2]) / 2))
    shifted = np.zeros_like(flipped)
    if dx >= 0:
        shifted[:, dx:] = flipped[:, : flipped.shape[1] - dx]
    else:
        shifted[:, :dx] = flipped[:, -dx:]
    both = left & shifted
    return both if both.any() else left


def _row_at(mask: np.ndarray, frac: float) -> int:
    """The row ``frac`` of the figure height down from the top of ``mask``'s figure."""
    b = imaging.bbox(mask)
    return b[1] + min(int(frac * (b[3] - b[1])), b[3] - b[1] - 1)


def central_run(row: np.ndarray, cx: float) -> tuple[float, float]:
    """The body's run in a torso row: the run holding the figure's centre ``cx``. When the row is parted at
    the centre, the span from the nearest run left of it to the nearest run right of it (never an outer run,
    which is a hand or an arm); one side only when the other is empty; ``cx`` +-1 when the row is empty."""
    rr = _runs(row)
    hit = [r for r in rr if r[0] <= cx <= r[1]]
    if hit:
        return hit[0]
    lefts, rights = [r for r in rr if r[1] <= cx], [r for r in rr if r[0] > cx]
    if not lefts and not rights:
        return cx - 1, cx + 1
    return (lefts[-1] if lefts else rights[0])[0], (rights[0] if rights else lefts[-1])[1]


def _two(runs: list[tuple[int, int]], fallback: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    """The outline rule for a leg row: the outer two runs, one run split at its middle (legs together, or a
    long coat), or ``fallback`` split when the row is empty."""
    if len(runs) >= 2:
        return runs[0], runs[-1]
    a, b = runs[0] if runs else fallback
    m = (a + b) / 2
    return (a, m), (m, b)


@dataclass(frozen=True)
class GuideLegs:
    """Where the turn guide's legs are in the front view (x px): the hips' run, and the image-left and
    image-right leg's run at the knee and sole rows. ``method``: how they were found (`guide_legs`)."""

    ROWS: ClassVar[dict[str, float]] = {"hip": 0.52, "knee": 0.71, "sole": 0.98}  # fractions from the top
    hips: tuple[float, float]
    knees: tuple[tuple[float, float], tuple[float, float]]
    soles: tuple[tuple[float, float], tuple[float, float]]
    method: str


def legs_from_outline(front: np.ndarray) -> GuideLegs:
    """The legs when no feet were found: the outermost runs of the knee and sole rows (`_two`) and, for the
    hips, the hip row's run holding the figure box's centre (the span of the nearest runs either side of it
    when the row is parted there; `central_run`)."""
    fb = imaging.bbox(front)
    hips = central_run(front[_row_at(front, GuideLegs.ROWS["hip"])], (fb[0] + fb[2]) / 2)
    knees, soles = (_two(_runs(front[_row_at(front, GuideLegs.ROWS[n])]), hips) for n in ("knee", "sole"))
    return GuideLegs(hips=hips, knees=knees, soles=soles, method="outline")


def _overlap(p: tuple[float, float], q: tuple[float, float]) -> float:
    return min(p[1], q[1]) - max(p[0], q[0])


def _trace_leg(mask: np.ndarray, run: tuple[float, float], r_from: int, r_to: int) -> tuple[float, float]:
    """Follow one leg's ``run`` of row ``r_from`` up the outline to row ``r_to``. On each row above, of the
    runs overlapping the last one, keep the one overlapping ``run`` itself most (then the last one). The
    starting columns favour the leg when a touching prop parts from it again, but cannot separate a prop
    that still overlaps those columns. A row with no overlapping run keeps the last. Two legs that
    join into the pelvis below ``r_to`` both end on the run they join."""
    a, b = run
    for r in range(r_from - 1, r_to - 1, -1):
        over = [q for q in _runs(mask[r]) if q[0] < b and q[1] > a]
        if over:
            a, b = max(over, key=lambda q: (_overlap(q, run), _overlap(q, (a, b))))
    return a, b


def legs_from_feet(front: np.ndarray, feet: list[np.ndarray], method: str) -> GuideLegs | None:
    """The legs anchored on two detected feet (full-size masks, image-left first): each foot's leg is the
    sole-row run its pixels in that row overlap most, clipped to the foot's own columns. Each leg is
    followed up the outline (`_trace_leg`) to the knee row and on to the hip row; legs on one run at the
    knee row split it at its middle; the hips span from the image-left leg's left edge to the image-right
    leg's right edge (the run both reach, when they join below the hip row). ``method`` is recorded with
    them. None when a foot has no pixel on a run of the sole row."""
    rows = {n: _row_at(front, f) for n, f in GuideLegs.ROWS.items()}
    sole_runs = _runs(front[rows["sole"]])
    picked = []
    for foot in feet:
        cols = np.flatnonzero(foot[rows["sole"]] & front[rows["sole"]])
        if cols.size == 0:
            return None
        lo, hi = int(cols[0]), int(cols[-1]) + 1
        over = [(min(q[1], hi) - max(q[0], lo), q) for q in sole_runs if q[0] < hi and q[1] > lo]
        if not over:
            return None
        run = max(over)[1]
        picked.append((max(run[0], lo), min(run[1], hi)))
    soles = tuple(picked)
    knees = tuple(_trace_leg(front, run, rows["sole"], rows["knee"]) for run in soles)
    if knees[0] == knees[1]:
        a, b = knees[0]
        knees = ((a, (a + b) / 2), ((a + b) / 2, b))
    legs = [_trace_leg(front, run, rows["knee"], rows["hip"]) for run in knees]
    hips = (min(a for a, _ in legs), max(b for _, b in legs))
    return GuideLegs(hips=hips, knees=knees, soles=soles, method=method)


def find_feet(client: ModelClient, img: Image.Image, figure: np.ndarray,
              texts: tuple[str, ...] = ("foot", "shoe", "boot")) -> tuple[list[np.ndarray], dict[str, Any]]:
    """SAM 3.1 feet on a turnaround front view: each prompt of ``texts`` in turn until one finds two feet.
    A foot is an instance mostly on the figure (half its pixels or more within 4 px of it) with pixels on the
    figure's sole row (0.98 of the height); of overlapping instances (IoU 0.5 or more) the more confident
    is kept, then the two most confident, image-left first. Returns the feet ([] when no prompt found two)
    and what each prompt found. A Model API failure is raised, so the stage stops (a replay of a run made
    before feet were asked for misses here)."""
    near = imaging.dilate(figure, 4)
    sole = _row_at(figure, GuideLegs.ROWS["sole"])
    found: dict[str, Any] = {}
    for text in texts:
        keep = [i for i in client.segment(img, text) if i.mask.any()
                and (i.mask & near).sum() >= 0.5 * i.mask.sum() and (i.mask[sole] & figure[sole]).any()]
        keep.sort(key=lambda i: -(i.confidence if i.confidence is not None else 0.0))
        feet: list[SamInstance] = []
        for inst in keep:
            if all(imaging.iou(inst.mask, f.mask) < 0.5 for f in feet):
                feet.append(inst)
        found[text] = [{"box": list(f.box), "confidence": f.confidence} for f in feet]
        if len(feet) >= 2:
            two = sorted(feet[:2], key=lambda i: float(np.nonzero(i.mask)[1].mean()))
            return [f.mask for f in two], {"prompt": text, "found": found}
    return [], {"prompt": None, "found": found}


def guide_legs(front: np.ndarray, feet: list[np.ndarray], prompt: str | None = None) -> GuideLegs:
    """The guide's legs: anchored on the feet SAM found (`legs_from_feet`; method ``sam-3.1:<prompt>``), as a
    silhouette alone can't tell a leg from a tail, a staff or a cape.
    Without two feet on the sole row, the outline rule (`legs_from_outline`; method ``outline``)."""
    legs = legs_from_feet(front, feet, f"{SAM_MODEL}:{prompt}") if len(feet) == 2 else None
    return legs if legs is not None else legs_from_outline(front)


def find_guide_legs(client: ModelClient, img: Image.Image,
                    front: np.ndarray) -> tuple[GuideLegs, dict[str, Any]]:
    """The guide's legs (`guide_legs`) on the feet `find_feet` finds, and what it found. When no prompt found
    two usable feet, the outline rule draws them and ``fallback_reason`` records why. A Model API failure or a
    replay miss on the feet is raised, so the stage stops."""
    feet, found = find_feet(client, img, front)
    legs = guide_legs(front, feet, found["prompt"])
    if legs.method == "outline":
        found["fallback_reason"] = "two usable feet were not found on the sole row"
    return legs, found


def _guide_shapes(front: np.ndarray, left: np.ndarray, right: np.ndarray, yaw_deg: float,
                  legs: GuideLegs | None = None) -> tuple[list[Shape], int]:
    """The mannequin's shapes at ``yaw_deg`` in turnaround px (u relative to the front figure's centre) and
    the front figure's height; see `turn_guide`."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    profile = _aligned_profile(left, right)
    fb = imaging.bbox(front)
    h = fb[3] - fb[1]
    cx = (fb[0] + fb[2]) / 2
    rows, _, _ = _band(profile, *ARM_PLANE_ROWS)
    arm_y = float(np.median([sum(_extent(r[None, :])) / 2 for r in rows if r.any()]))

    def runs_at(mask: np.ndarray, frac: float) -> list[tuple[int, int]]:
        return _runs(mask[_row_at(mask, frac)])

    legs = legs or legs_from_outline(front)
    hip_row, knee_row, sole_row = (GuideLegs.ROWS[n] for n in ("hip", "knee", "sole"))
    hips = legs.hips

    def central(frac: float) -> tuple[float, float]:
        return hips if frac == hip_row else central_run(front[_row_at(front, frac)], cx)

    def depth(frac: float) -> tuple[float, float]:
        rr = runs_at(profile, frac)
        return (rr[0][0], rr[-1][1]) if rr else (arm_y - 1, arm_y + 1)

    def u(x: float, y: float) -> float:
        return (x - cx) * c + (y - arm_y) * s

    shapes: list[Shape] = []
    lw = 0.035 * h
    hx0, hx1 = _extent(_band(front, 0.0, 0.12)[0])
    pd0, pd1 = depth(0.06)
    hu = u((hx0 + hx1) / 2, (pd0 + pd1) / 2)
    half = math.hypot((hx1 - hx0) / 2 * c, (pd1 - pd0) / 2 * s)
    shapes.append(("ellipse", [(hu - half, 0.0), (hu + half, 0.12)], 0.0))
    shapes.append(("dot", [(u((hx0 + hx1) / 2, pd0), 0.06)], 0.6 * lw))  # the nose: the face's side
    edges_l, edges_r = [], []
    for frac in (0.14, 0.2, 0.3, 0.4, hip_row):
        a, b = central(frac)
        p0, p1 = depth(frac)
        m = u((a + b) / 2, (p0 + p1) / 2)
        half = math.hypot((b - a) / 2 * c, (p1 - p0) / 2 * s)
        edges_l.append((m - half, frac))
        edges_r.append((m + half, frac))
    shapes.append(("polygon", edges_l + edges_r[::-1], 0.0))
    fl = landmarks(front)
    for (hand_x, hand_z), shoulder_x in zip(fl.hands, central(0.2), strict=True):
        hand = (u(hand_x, arm_y), 1.0 - hand_z)
        shapes.append(("line", [(u(shoulder_x, arm_y), 0.2), hand], lw))
        shapes.append(("dot", [hand], lw))
    knees, soles = legs.knees, legs.soles
    ky = sum(depth(knee_row)) / 2
    toe_y, heel_y = depth(sole_row)
    hip_xs = (hips[0] + 0.25 * (hips[1] - hips[0]), hips[0] + 0.75 * (hips[1] - hips[0]))
    for hip_x, (ka, kb), (sa, sb) in zip(hip_xs, knees, soles, strict=True):
        ankle = (u((sa + sb) / 2, ky), 0.94)
        heel = (u((sa + sb) / 2, heel_y), 0.985)
        leg = [(u(hip_x, arm_y), hip_row), (u((ka + kb) / 2, ky), knee_row), ankle]
        shapes.append(("line", leg, 1.3 * lw))
        shapes.append(("line", [ankle, heel], lw))
        shapes.append(("line", [heel, (u((sa + sb) / 2, toe_y), 0.985)], 1.1 * lw))
    return shapes, h


def guide_width(shapes: list[Shape]) -> float:
    """The mannequin's width in turnaround px, strokes included."""
    us = [p[0] for _, pts, _ in shapes for p in pts]
    pad = max(w for _, _, w in shapes)
    return max(us) - min(us) + 2 * pad


def turn_guide(front: np.ndarray, left: np.ndarray, right: np.ndarray, yaw_deg: float, size: tuple[int, int],
               scale: float | None = None, legs: GuideLegs | None = None) -> tuple[Image.Image, int]:
    """A plain grey mannequin of the turnaround's pose seen at ``yaw_deg``, on a white canvas of ``size``,
    centred, at ``scale`` canvas px per turnaround px (default `framing_scale` for the front figure and the
    mannequin). Returns the image and the figure's height on it (the front figure's height times the scale).

    Only landmark positions come from the turnaround, never its outline: the head (an ellipse, its width
    from the front view and its depth from the profile, with a small nose on the face side), the torso (the
    body's central run per row as an ellipse, `central_run`, down to the hips), the arms (straight from the
    shoulders to the hand landmarks, in the frontal plane), the legs (from the hips to the knee and sole
    runs of ``legs``, `guide_legs`, by default the outline rule; one run split in two when the legs touch)
    and the feet (heel to toe from the profiles' soles), each projected with u = x cos + y sin.
    """
    shapes, h = _guide_shapes(front, left, right, yaw_deg, legs)
    if scale is None:
        fw = imaging.box_size(imaging.bbox(front))[0]
        scale = framing_scale(h, [fw, guide_width(shapes)], size)
    k = scale
    us = [p[0] for _, pts, _ in shapes for p in pts]
    ox = size[0] / 2 - (min(us) + max(us)) / 2 * k
    oy = (size[1] - h * k) / 2
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    for kind, pts, width in shapes:
        xy = [(ox + p[0] * k, oy + p[1] * h * k) for p in pts]
        if kind == "ellipse":
            d.ellipse([xy[0][0], xy[0][1], xy[1][0], xy[1][1]], fill=GUIDE_GREY)
        elif kind == "dot":
            r = width * k
            d.ellipse([xy[0][0] - r, xy[0][1] - r, xy[0][0] + r, xy[0][1] + r], fill=GUIDE_GREY)
        elif kind == "polygon":
            d.polygon(xy, fill=GUIDE_GREY)
        else:
            d.line(xy, fill=GUIDE_GREY, width=max(2, int(width * k)), joint="curve")
    return img, round(h * k)


def figure_mask(client: ModelClient, img: Image.Image) -> tuple[np.ndarray, str, float]:
    """SAM 3.1 figure mask, its method, and its `sam_coverage` of the background-threshold area (the gated
    `figure_mask_sam_coverage` check, the same rule as a turnaround view). The mask is always SAM's (empty
    when SAM found nothing); a Model API failure raises."""
    sam, method, _ = sam_figure_mask(client, img)
    return sam, method, sam_coverage(sam, imaging.threshold_mask(img))


def place_on_canvas(
    raw: Image.Image, mask: np.ndarray, scale: float, size: int, bottom_row: int
) -> tuple[Image.Image, np.ndarray, int]:
    """Crop the figure, scale it by ``scale``, and put it on the ``size`` square of the turnaround views:
    centred horizontally, its lowest row on ``bottom_row`` (the front view's feet). A figure that would pass
    the top or a side of the square is neither cut nor shrunk: the square is padded by the same ``pad`` px
    on every side, so it stays square and centred, the scale is unchanged, and the views' pixel (x, y) is
    (x + pad, y + pad) on it. An oversized figure therefore measures its full height. Returns the image, the
    mask and ``pad`` (0 when the figure fits)."""
    bg = tuple(int(c) for c in imaging.background_colour(np.array(raw)))
    b = imaging.bbox(mask)
    if b is None:
        return Image.new("RGB", (size, size), bg), np.zeros((size, size), dtype=bool), 0
    new = (max(1, round((b[2] - b[0]) * scale)), max(1, round((b[3] - b[1]) * scale)))
    crop = raw.crop(b).resize(new, Image.LANCZOS)
    cmask = imaging.to_mask_image(mask[b[1] : b[3], b[0] : b[2]]).resize(new, Image.BILINEAR)
    x0, y0 = int(round(size / 2 - new[0] / 2)), int(bottom_row - new[1])
    pad = max(0, -y0, -x0, x0 + new[0] - size, bottom_row - size)
    side = size + 2 * pad
    canvas = Image.new("RGB", (side, side), bg)
    mcanvas = Image.new("L", (side, side), 0)
    canvas.paste(crop, (x0 + pad, y0 + pad))
    mcanvas.paste(cmask, (x0 + pad, y0 + pad))
    return canvas, imaging.from_mask_image(mcanvas), pad


_STATUS_DIR = {"accepted": "heldout", "needs review": "heldout/review", "rejected": "heldout/rejected"}
_RANK = {"accepted": 0, "needs review": 1, "rejected": 2}


def view_paths(view: HeldoutView, status: str = "accepted") -> dict[str, str]:
    """Where a view's image, mask and raw generation are written for its status."""
    d = _STATUS_DIR[status]
    return {"image": f"{d}/{view.name}.png", "mask": f"{d}/{view.name}-mask.png",
            "raw": f"{d}/{view.name}-raw.png"}


def heldout_rel(rel: Any) -> str | None:
    """``rel`` normalised when it names a path under `heldout/` of a run folder, else None (`heldout/../x`
    and absolute paths included), so nothing outside `heldout/` is ever deleted or moved."""
    if not isinstance(rel, str) or rel.startswith("/") or "\\" in rel:
        return None
    norm = posixpath.normpath(rel)
    return norm if norm.startswith("heldout/") and ".." not in norm.split("/") else None


def _remove_empty_status_dirs(run: RunFolder) -> None:
    for status, folder in _STATUS_DIR.items():
        d = run.path / folder
        if status != "accepted" and d.is_dir() and not any(d.iterdir()):
            d.rmdir()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def move_files(run: RunFolder, moves: dict[str, str]) -> None:
    """Move run-folder files ``src -> dst`` (both under `heldout/`) all or none: every source must exist
    before anything moves, and a failure part-way moves the moved ones back."""
    pairs = []
    for src, dst in moves.items():
        a, b = heldout_rel(src), heldout_rel(dst)
        if a is None or b is None:
            raise ValueError(f"refusing to move {src!r} to {dst!r}: not under heldout/")
        if not (run.path / a).is_file():
            raise FileNotFoundError(f"{a} is missing from the run folder")
        if a != b:
            pairs.append((a, b))
    done: list[tuple[str, str]] = []
    try:
        for a, b in pairs:
            run.file(b)
            os.replace(run.path / a, run.path / b)
            done.append((a, b))
    except BaseException:
        for a, b in reversed(done):
            os.replace(run.path / b, run.path / a)
        raise


# Provenance. Each view records the stage-2 identity it was made from (the SHA-256 of the files below) and
# the checks version it was judged under (`checks_version`). A view whose turnaround changed is never kept;
# one judged under other checks is checked again (`recheck_view`) before it is kept or reviewed.
IDENTITY_FILES = ("01-cover.png", "views/meta.json", "views/front.png", "views/left.png", "views/right.png",
                  "masks/front.png", "masks/left.png", "masks/right.png")


def stage2_identity(run: RunFolder) -> str:
    h = hashlib.sha256()
    for rel in IDENTITY_FILES:
        h.update(rel.encode() + b"\0" + (file_sha256(run.path / rel).encode() if run.exists(rel) else b"-"))
    return h.hexdigest()


def checks_version() -> str:
    """A hash of everything a view is judged by: every upper-case constant of this module (thresholds,
    landmark bands, the yaw grid, the windows), and the source of the judging functions here and of the
    `checks` and `imaging` modules. A change to any of them changes the version without anyone bumping it,
    and the views are judged again (no model calls) before they are kept or reviewed."""
    consts = {}
    for name, value in sorted(globals().items()):
        if name.isupper() and isinstance(value, (int, float, str, tuple, list, dict, np.ndarray)):
            consts[name] = value.tolist() if isinstance(value, np.ndarray) else repr(value)
    sources = [inspect.getsource(f) for f in (
        _runs, _band, _extent, landmarks, arm_slopes, TurnaroundModel, turnaround_model, landmark_errors,
        _rms, fit_yaw, hand_z_morph, yaw_range, range_clearance, azimuth_clearance, window_miss, yaw_window,
        building_azimuths, view_for_build, toe_offset_h, face_offset, _vote, _review, view_checks, view_status, recheck_view, _seen, body_model, C,
        imaging, KP)]
    digest = hashlib.sha256(json.dumps([consts, sources], sort_keys=True).encode()).hexdigest()[:12]
    return f"c-{digest}"


@dataclass
class _Context:
    """What every attempt of every view is drawn from and measured against."""

    cover: Image.Image
    views: dict[str, Image.Image]  # the turnaround's front, left and right views
    masks: dict[str, np.ndarray]  # their masks
    req_size: str
    size: int
    bottom_row: int
    front_h: int
    model: TurnaroundModel | None
    profile_toe: float
    profile_face: float | None
    profile_face_error: bool = False  # the profiles' face cue failed on the Model API
    identity: str = ""  # `stage2_identity`
    build_azimuths: tuple[float, ...] = BUILD_AZIMUTHS_DEG
    legs: GuideLegs | None = None  # the turn guide's legs (`guide_legs`); None: the outline rule
    pose: PoseClient | None = None  # Sapiens 2 keypoints
    turnaround_kp: dict[str, Any] | None = None  # `turnaround_keypoints`
    body: KP.BodyModel | None = None  # `body_model` of them


def _inputs(view: HeldoutView, ctx: _Context, request_deg: float
            ) -> tuple[str, list[Image.Image], float, str]:
    """(prompt, input images, framing scale, reference name) for one generation: the pose reference and the
    turn guide share one `framing_scale`."""
    reference = "profile" if request_deg >= PROFILE_REFERENCE_FROM_DEG else "front"
    name = view.side if reference == "profile" else "front"
    m = ctx.masks
    shapes, _ = _guide_shapes(m["front"], m["left"], m["right"], -view.facing * request_deg, ctx.legs)
    widths = [imaging.box_size(b)[0] for b in (imaging.bbox(x) for x in m.values()) if b is not None]
    k = framing_scale(ctx.front_h, widths + [guide_width(shapes)], ctx.cover.size)
    ref, _ = pose_reference(ctx.views[name], m[name], ctx.cover.size, k)
    guide, _ = turn_guide(m["front"], m["left"], m["right"], -view.facing * request_deg, ctx.cover.size, k,
                          ctx.legs)
    return view_prompt(view, request_deg, reference), [ctx.cover, ref, guide], k, reference


# Overall time limit for one stage 2h run. Each Model API call can retry within its own deadline (client
# DEADLINE_S: 10 minutes for SAM 3.1, 15 for Muse Image), and a run makes up to 16 generations and 55 SAM calls;
# a typical live run takes 7-12 minutes. The limit is checked before each generation, so the generation in
# flight finishes first.
TIME_LIMIT_S = 1800.0


class StageTimeLimit(RuntimeError):
    """Stage 2h ran past its time limit."""


def _generate_view(st: _HeldoutStage, client: ModelClient, view: HeldoutView, ctx: _Context, attempts: int,
                   deadline: float | None = None) -> tuple[str, list[C.Check], dict[str, Any]]:
    """Generate ``view`` until it is accepted, at most ``attempts`` times, each attempt asking for the target
    plus the shortfall the last one measured, and keep the best attempt (accepted, else needs review, else
    rejected; ties go to the smaller window miss, then the lower landmark error). ``deadline``
    (`time.monotonic`): raise `StageTimeLimit` instead of starting a generation after it."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    tried: list[dict[str, Any]] = []
    kept: tuple = ()
    request = view.aim_deg
    first_sample = VIEWS.index(view) * attempts if view in VIEWS else 0
    for attempt in range(attempts):
        if deadline is not None and time.monotonic() > deadline:
            raise StageTimeLimit(f"stage 2h passed its {TIME_LIMIT_S:.0f} s time limit before {view.name} "
                                 f"generation {attempt + 1}")
        prompt, images, framing, reference = _inputs(view, ctx, request)
        # Two views can send the same request (the same prompt and images); each view draws its own samples.
        sample = first_sample + attempt
        try:
            raw = client.edit_image(prompt, images, size=ctx.req_size, attempt=sample)
        except ReplayMiss as e:
            raise ReplayMiss(
                f"{view.name} generation {attempt + 1}: {e}. A request (its prompt and input images) "
                "depends on the code that drew the inputs and on the yaws earlier generations were found at, "
                "so a cache recorded by another version can miss; views kept with --views need no new "
                f"generations, so make only this one again, live: --stages 2h --views {view.name} --live"
            ) from e
        mask, method, coverage = figure_mask(client, raw)
        # Raw px -> input-canvas px (the model may answer at another resolution) -> turnaround px.
        scale = ctx.cover.size[0] / raw.size[0] / framing
        img, placed, pad = place_on_canvas(raw, mask, scale, ctx.size, ctx.bottom_row)
        measurable = coverage >= C.SAM_COVERAGE_MIN
        face = measure_face(client, raw, mask) if measurable else {"offset": None, "error": "no SAM figure"}
        kps = measure_keypoints(ctx.pose, raw, mask) if measurable and ctx.body is not None else None
        checks, measured = view_checks(view, placed, ctx.model, ctx.profile_toe, float(mask.mean()), face,
                                       ctx.profile_face, profile_face_error=ctx.profile_face_error,
                                       mask_sam_coverage=coverage, body=ctx.body, keypoints=kps, kp_scale=scale,
                                       build_azimuths=ctx.build_azimuths)
        status = view_status(checks)
        tried.append({
            "attempt": attempt,
            "sample": sample,
            "status": status,
            "requested_yaw_deg": -view.facing * request,
            "pose_reference": f"views/{'front' if reference == 'front' else view.side}.png",
            "prompt": prompt,
            "raw_size": list(raw.size),
            "mask_method": method,
            "framing_scale_input_px_per_view_px": round(framing, 5),
            "scale_view_px_per_raw_px": round(scale, 5),
            "canvas_pad_px": pad,
            "bbox_px": list(imaging.bbox(placed) or ()),
            "measured": measured,
            "checks": checks,
        })
        fit = (measured.get("keypoint_fit") or {}).get(f"facing_{view.side}", {})
        miss = measured.get("yaw_window_miss_deg")
        key = (_RANK[status], 99.0 if miss is None else miss, fit.get("rms", 9.0))
        if not kept or key < kept[0]:
            kept = (key, attempt, status, checks, raw, img, placed, method, images)
        if status == "accepted":
            break
        facing_ok = (measured.get("facing_votes") or {}) and not [
            c for c in checks if c["name"] == "facing_cues_against" and not c["pass"]]
        request = next_request(view, request, measured.get("found_yaw_deg") if facing_ok else None)
    _, attempt, status, checks, raw, img, placed, method, images = kept
    files = view_paths(view, status)
    st.save_image(files["raw"], raw)
    st.save_image(files["image"], img)
    st.save_image(files["mask"], imaging.to_mask_image(placed))
    inputs = {"pose_reference": f"heldout/inputs/{view.name}-pose.png",
              "turn_guide": f"heldout/inputs/{view.name}-guide.png"}
    st.save_image(inputs["pose_reference"], images[1])
    st.save_image(inputs["turn_guide"], images[2])
    details = {
        "status": status,
        "machine_status": status,  # what the checks decided; `status` can change on human review
        "review": None,  # the latest human decision (`review_view`)
        "accepted": status == "accepted",
        "image_sha256": file_sha256(st.staged(files["image"])),
        "stage2_identity": ctx.identity,
        "checks_version": checks_version(),
        "run_started": st.started,
        "kept_attempt": attempt,
        "found_yaw_deg": tried[attempt]["measured"].get("found_yaw_deg"),
        "target_yaw_deg": view.yaw_deg,
        "yaw_window_deg": list(view.window_deg),
        "facing": "image left" if view.facing < 0 else "image right",
        "files": files,
        "kept_inputs": {"cover": "01-cover.png", **inputs},
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


def _view_outputs(d: dict[str, Any]) -> list[str]:
    """The files a view's record lists: its image, mask and raw generation and its kept inputs."""
    rels = list((d.get("files") or {}).values())
    rels += [v for k, v in (d.get("kept_inputs") or {}).items() if k != "cover"]
    return [r for r in rels if heldout_rel(r)]


def _status_checks(name: str, d: dict[str, Any]) -> list[C.Check]:
    """The gated `<view>.status_accepted` check for a view that is not accepted but whose own gated checks
    passed: one waiting for human review, or one rejected on review."""
    review = d.get("review") or {}
    if d["status"] == "needs review":
        return [C.check(f"{name}.status_accepted", 0.0, 1.0, op="==",
                        note="needs human review: a check could not decide")]
    if d["status"] == "rejected" and review.get("decision") == "rejected":
        why = review.get("note") or "no note"
        return [C.check(f"{name}.status_accepted", 0.0, 1.0, op="==",
                        note=f"rejected on human review by {review.get('by')}: {why}")]
    return []


def _summary(per_view: dict[str, Any]) -> dict[str, list[str]]:
    return {f"{key}_views": [v for v, d in per_view.items() if d["status"] == status]
            for key, status in (("accepted", "accepted"), ("needs_review", "needs review"),
                                ("rejected", "rejected"))}


def _ordered(per_view: dict[str, Any]) -> dict[str, Any]:
    order = {v.name: i for i, v in enumerate(VIEWS)}
    return dict(sorted(per_view.items(), key=lambda kv: order.get(kv[0], len(order))))


def _iso(t: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(t) if isinstance(t, str) else None
    except ValueError:
        return None


def _view_started(prev: dict[str, Any], name: str) -> Any:
    """When ``name`` was made: its own record, else the earlier record's note of kept views, else the
    record's start."""
    d = ((prev.get("details") or {}).get("views") or {}).get(name) or {}
    kept = (prev.get("details") or {}).get("kept_from_earlier_run") or {}
    if d.get("run_started"):
        return d["run_started"]
    if isinstance(kept.get("views"), list) and name in kept["views"]:  # written before views kept their own
        return kept.get("started")
    return prev.get("started")  # a stopped run leaves the record's start as the run that made the views


def check_provenance(run: RunFolder, prev: dict[str, Any], name: str, identity: str) -> None:
    """Raise unless view ``name`` of the earlier record was made from this run's current turnaround: its
    recorded `stage2_identity` matches, or (records written before identities) it was made after stages 1
    and 2 last ran."""
    d = prev["details"]["views"][name]
    remake = (f"remake it: run --stages 2h without --views (or with --views {name}) so every view comes "
              "from the current turnaround")
    if d.get("stage2_identity"):
        if d["stage2_identity"] != identity:
            raise ValueError(f"{name} was made from an earlier turnaround (stage 1 or 2 has run since); "
                             f"{remake}")
        return
    made = _iso(_view_started(prev, name))
    for n in (1, 2):
        ran = _iso((run.read_stage(n) or {}).get("started"))
        if made is None or ran is None or ran >= made:
            raise ValueError(f"{name} has no turnaround identity and can't be shown to be newer than stage "
                             f"{n}; {remake}")


def recheck_view(run: RunFolder, name: str, d: dict[str, Any], model: TurnaroundModel | None,
                 profile_toe: float, profile_face: float | None, profile_face_error: bool,
                 identity: str, body: KP.BodyModel | None = None
                 ) -> tuple[dict[str, Any], list[C.Check], dict[str, str]]:
    """Judge a recorded view's kept generation again under the current checks, from its files and recorded
    measurements, its keypoints included (no model calls). A human decision on the same image still holds (a
    rejection always; an
    acceptance unless a gated check now fails). Returns the updated record, its checks and the file moves to
    its new status's folder."""
    build_azimuths = building_azimuths(run)
    view = view_for_build(VIEW_BY_NAME[name], build_azimuths)
    kept = dict(d["attempts"][d["kept_attempt"]])
    rels = {k: heldout_rel(d["files"].get(k)) for k in ("image", "mask", "raw")}
    if None in rels.values():
        raise ValueError(f"{name}'s files {d['files']} are not all under heldout/")
    placed = imaging.from_mask_image(_load(run, rels["mask"]))
    area = next((c["value"] for c in kept["checks"] if c["name"] == "figure_area_frac"), 1.0)
    coverage = next((c["value"] for c in kept["checks"] if c["name"] == "figure_mask_sam_coverage"), None)
    if coverage is None and method_kind(str(kept.get("mask_method", SAM_MODEL))) != SAM_MODEL:
        coverage = 0.0  # an earlier version measured it on a threshold mask: not a measurement
    face = (kept.get("measured") or {}).get("face") or {"offset": None}
    kps = (kept.get("measured") or {}).get("keypoints")
    if kps is None:
        kps = {"joints": None, "error": "no keypoints recorded (made before they were measured)"}
    checks, measured = view_checks(view, placed, model, profile_toe, area, face, profile_face,
                                   profile_face_error=profile_face_error, mask_sam_coverage=coverage,
                                   body=body, keypoints=kps, build_azimuths=build_azimuths,
                                   kp_scale=float(kept.get("scale_view_px_per_raw_px", 1.0)))
    machine = view_status(checks)
    image_sha = d.get("image_sha256") or file_sha256(run.path / rels["image"])
    review = d.get("review")
    status = machine
    if review and review.get("image_sha256") == image_sha:
        if review["decision"] == "rejected" or machine != "rejected":
            status = review["decision"]
    files = view_paths(view, status)
    # Only the kept generation's files exist, so only it is judged again; the others keep their old
    # measurements, marked as judged under the earlier checks.
    attempts = [a if i == d["kept_attempt"] else
                {**a, "stale": True, "judged_under": a.get("judged_under") or d.get("checks_version")}
                for i, a in enumerate(d["attempts"])]
    attempts[d["kept_attempt"]] = {**kept, "status": machine, "checks": checks, "measured": measured}
    new = {**d, "status": status, "machine_status": machine, "accepted": status == "accepted", "files": files,
           "image_sha256": image_sha, "stage2_identity": identity, "checks_version": checks_version(),
           "found_yaw_deg": measured.get("found_yaw_deg"), "yaw_window_deg": list(view.window_deg),
           "attempts": attempts,
           "rechecked": {"at": now_iso(), "from_checks_version": d.get("checks_version"),
                         "machine_status_before": d.get("machine_status") or d["status"]}}
    return new, checks, {rels[k]: files[k] for k in files}


STAGING_DIR = "heldout/.staging"
ARCHIVE_DIR = "heldout/archive"


class _HeldoutStage(_Stage):
    """Writes its images under STAGING_DIR (at the same relative paths); `_commit` moves them into place
    once every view is made, so a run that stops early leaves the earlier files as they were."""

    def staged(self, rel: str) -> Path:
        p = self.run.path / STAGING_DIR / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_image(self, rel: str, img: Image.Image) -> None:
        img.save(self.staged(rel), format="PNG")
        self.outputs.append(rel)


def _commit(run: RunFolder, st: _HeldoutStage, prev: dict[str, Any], made: list[str],
            kept_outputs: list[str], moves: dict[str, str], decisions: list[dict[str, Any]]) -> None:
    """Swap a finished run's files in: the earlier files of the views made again go (to ARCHIVE_DIR when a
    person decided on them, never deleted), kept views move to their status's folder, staged files move in."""
    prev_views = (prev.get("details") or {}).get("views") or {}
    stamp = re.sub(r"[^0-9A-Za-z]+", "-", str(st.started)).strip("-")
    old = {heldout_rel(r) for r in prev.get("outputs", [])} - {None} - set(kept_outputs) - set(moves)
    for name in made:
        view = VIEW_BY_NAME[name]
        old |= {rel for status in _STATUS_DIR for rel in view_paths(view, status).values()}
    decided = {r for n, d in prev_views.items() if n in made and d.get("review") for r in _view_outputs(d)}
    for rel in sorted(old):
        if not (run.path / rel).is_file():
            continue
        if rel in decided:
            dst = f"{ARCHIVE_DIR}/{stamp}/{rel.removeprefix('heldout/')}"
            move_files(run, {rel: dst})
            for e in decisions:
                if e.get("image") == rel and not e.get("archived_to"):
                    e["archived_to"] = dst
        else:
            (run.path / rel).unlink()
    move_files(run, moves)
    for rel in st.outputs:
        if rel not in kept_outputs and (run.path / STAGING_DIR / rel).is_file():
            run.file(rel)
            os.replace(run.path / STAGING_DIR / rel, run.path / rel)
    shutil.rmtree(run.path / STAGING_DIR, ignore_errors=True)
    _remove_empty_status_dirs(run)


def stage2h_heldout(run: RunFolder, client: ModelClient, *, attempts: int = ATTEMPTS,
                    views: list[str] | None = None, time_limit_s: float | None = TIME_LIMIT_S,
                    pose: PoseClient | None = None) -> dict[str, Any]:
    """Generate, mask and check the held-out views for a run whose stage 2 passed: all four, or only
    ``views`` (by name), keeping the others, their files and their review decisions from the earlier record
    (each checked to come from the current turnaround, and judged again if the checks changed since).
    Writes only `heldout/` and `stage-2h.json`; every earlier output is left as it is. The gate passes when
    every view is accepted; a view that needs human review or was rejected fails it.

    New files are written to STAGING_DIR and swapped in only when every view is made. Meanwhile
    `stage-2h.json` keeps the earlier record, marked as running; if the stage stops early (an error, a replay
    miss, the time limit) it is written with the error and the calls made, and the earlier files stay."""
    s2 = run.read_stage(2)
    if not s2 or not s2.get("ok"):
        raise FileNotFoundError("stage 2 has not passed in this run folder; run stage 2 first")
    if degraded_masks(s2):
        raise ValueError(f"stage 2 passed on masks SAM 3.1 did not make ({'; '.join(degraded_masks(s2))}), "
                         "written by an earlier version; run stage 2 again")
    names = [v.name for v in VIEWS] if not views else list(dict.fromkeys(views))
    unknown = [n for n in names if n not in VIEW_BY_NAME]
    if unknown:
        raise ValueError(f"unknown held-out views {unknown}; the views are {[v.name for v in VIEWS]}")
    build_azimuths = building_azimuths(run)
    chosen = [view_for_build(v, build_azimuths) for v in VIEWS if v.name in names]
    prev = run.read_stage("2h") or {}  # type: ignore[arg-type]
    prev_views = (prev.get("details") or {}).get("views") or {}
    keep = {n: d for n, d in prev_views.items() if n not in names and n in VIEW_BY_NAME}
    missing = [v.name for v in VIEWS if v.name not in names and v.name not in keep]
    if views and missing:
        raise FileNotFoundError(f"no earlier stage 2h record of {missing} to keep; make all views first")
    identity = stage2_identity(run)
    try:
        for name in keep:
            check_provenance(run, prev, name, identity)
    except ValueError as e:  # the earlier record no longer describes this run's views: say so in it
        run.write_stage({**prev, "ok": False, "status": "error", "error": str(e)})
        raise
    meta = run.read_json("views/meta.json")
    cover = _load(run, "01-cover.png")
    front = imaging.from_mask_image(_load(run, "masks/front.png"))
    left, right = (imaging.from_mask_image(_load(run, f"masks/{v}.png")) for v in ("left", "right"))
    fb = imaging.bbox(front)
    if fb is None:
        raise ValueError("the front mask is empty")
    sketch = _load(run, "00-sketch.png") if run.exists("00-sketch.png") else cover
    masks = {"front": front, "left": left, "right": right}

    st = _HeldoutStage(run, client, "2h", "heldout")  # type: ignore[arg-type]
    own_pose = pose is None
    if pose is None:  # Sapiens 2 shares the Model API client's cache folder and mode
        pose = PoseClient(client.cache_dir, mode=client.mode)
    pose_calls0 = len(pose.log.calls)

    def pose_cost() -> float:
        return pose.log.cost_usd(pose_calls0)
    shutil.rmtree(run.path / STAGING_DIR, ignore_errors=True)  # a run that was killed
    decisions = [dict(e) for e in (prev.get("details") or {}).get("decisions") or []]
    kept_checks = [c for c in prev.get("checks", [])
                   if c["name"].split(".", 1)[0] in keep and not c["name"].endswith(".status_accepted")]
    kept_outputs = [r for d in keep.values() for r in _view_outputs(d)]
    per_view: dict[str, Any] = {n: {**d, "run_started": _view_started(prev, n)} for n, d in keep.items()}
    moves: dict[str, str] = {}

    def assemble() -> None:
        pv = _ordered(per_view)
        st.checks = list(kept_checks) + [c for c in st.checks if c["name"].split(".", 1)[0] not in keep]
        st.checks = [c for c in st.checks if not c["name"].endswith(".status_accepted")]
        for name, d in pv.items():
            st.checks += _status_checks(name, d)
        outs = kept_outputs + [o for o in st.outputs if o not in kept_outputs]
        st.outputs = list(dict.fromkeys(moves.get(o, o) for o in outs))
        st.details.update({"views": pv, **_summary(pv), "decisions": decisions,
                           "made_this_run": [v.name for v in chosen]})
        if keep:
            st.details["kept_from_earlier_run"] = {n: per_view[n]["run_started"] for n in keep}

    def stopped(error: str, status: str) -> dict[str, Any]:
        """The earlier record, its views, files, calls and cost untouched, with this run added to
        `stopped_runs` (its error, calls and cost) and its cost added to the total."""
        run_rec = st._record()
        base = dict(prev) if prev else {**run_rec, "ok": False, "outputs": [], "checks": [], "details": {},
                                        "calls": [], "cost_usd": 0.0, "run_cost_usd": 0.0}
        this = {"started": run_rec["started"], "seconds": run_rec["seconds"], "status": status,
                "error": error, "made": [v.name for v in chosen],
                "cost_usd": round(run_rec["cost_usd"] + pose_cost(), 6), "calls": run_rec["calls"],
                "keypoint_calls": pose.log.calls[pose_calls0:]}
        stops = [r for r in earlier_stops if r.get("started") != this["started"]] + [this]
        rec = {**base, "ok": False, "status": status, "error": error, "run_cost_usd": prev_run_cost,
               "stopped_runs": stops, "cost_usd": round(prev_run_cost + sum(r["cost_usd"] for r in stops), 6)}
        run.write_stage(rec)
        return rec

    # Costs: `run_cost_usd` is the run that wrote the record's views; the runs that stopped since add theirs
    # (`stopped_runs`), so `cost_usd` never drops a paid call.
    earlier_stops = list(prev.get("stopped_runs") or [])
    prev_run_cost = float(prev.get("run_cost_usd", prev.get("cost_usd") or 0.0)) if prev else 0.0
    if prev and "run_cost_usd" not in prev:  # a record written before stopped runs were kept
        prev_run_cost -= sum(r.get("cost_usd", 0.0) for r in earlier_stops)

    stopped("stage 2h is running, or stopped without writing its record; the views listed are the earlier "
            "run's", status="running")
    deadline = None if time_limit_s is None else st.t0 + time_limit_s
    try:
        profile_face, profile_face_found = _profile_face(client, run)
        front_view = _load(run, "views/front.png")
        legs, feet_found = find_guide_legs(client, front_view, front)
        ctx = _Context(
            cover=cover,
            views={v: _load(run, f"views/{v}.png") for v in masks},
            masks=masks,
            # The size stage 1 asked for; the cover keeps its aspect.
            req_size=f"{sketch.size[0]}x{sketch.size[1]}",
            size=int(meta["image_size"]),
            bottom_row=fb[3],
            front_h=fb[3] - fb[1],
            model=turnaround_model(front, left, right),
            profile_toe=float(np.mean([abs(toe_offset_h(m)) for m in (left, right)])),
            profile_face=profile_face,
            profile_face_error=any(f.get("api_error") for f in profile_face_found.values()),
            identity=identity,
            build_azimuths=build_azimuths,
            legs=legs,
            pose=pose,
        )
        ctx.turnaround_kp = turnaround_keypoints(pose, ctx.views, masks)
        ctx.body = body_model(ctx.turnaround_kp)
        version = checks_version()
        for name, d in keep.items():
            if (d.get("checks_version") != version
                    or (prev.get("details") or {}).get("build_azimuths_deg") != list(build_azimuths)):
                new, checks, mv = recheck_view(run, name, d, ctx.model, ctx.profile_toe, profile_face,
                                               ctx.profile_face_error, identity, ctx.body)
                per_view[name] = {**new, "run_started": per_view[name]["run_started"]}
                kept_checks[:] = [c for c in kept_checks if c["name"].split(".", 1)[0] != name]
                kept_checks += [{**c, "name": f"{name}.{c['name']}"} for c in checks]
                moves.update(mv)
                st.notes.append(f"{name}: checked again under the current checks: {new['status']}.")
        for view in chosen:
            status, checks, details = _generate_view(st, client, view, ctx, attempts, deadline)
            st.checks += [{**c, "name": f"{view.name}.{c['name']}"} for c in checks]
            per_view[view.name] = details
            n = len(details["attempts"])
            where = f" ({details['files']['image']})" if status != "accepted" else ""
            st.notes.append(f"{view.name}: {status} after {n} generation(s), kept generation "
                            f"{details['kept_attempt'] + 1}{where}.")
    except BaseException as e:
        shutil.rmtree(run.path / STAGING_DIR, ignore_errors=True)
        stopped(f"{type(e).__name__}: {e}"[:800], status="error")
        raise
    finally:
        if own_pose:
            pose.close()
    if keep:
        st.notes.append(f"Kept from earlier runs: {', '.join(keep)}.")
    assemble()
    st.details.update({
        "reference": "generated",
        "reference_note": ("the held-out views are generated images that can invent detail, so a score "
                           "against them measures consistency with the generated character, not accuracy"),
        "build_azimuths_deg": list(build_azimuths),
        "azimuth_clearance_deg": AZIMUTH_CLEARANCE_DEG,
        "fit_error_deg": list(FIT_ERROR_DEG),
        "checks_version": checks_version(),
        "stage2_identity": identity,
        "profile_toe_offset_h": round(ctx.profile_toe, 4),
        "profile_face_offset": None if profile_face is None else round(profile_face, 4),
        "profile_face": _rounded(profile_face_found),
        "guide_legs": {"method": legs.method, "hips": list(legs.hips), "knees": [list(k) for k in legs.knees],
                       "soles": [list(k) for k in legs.soles], "feet": feet_found},
        "turnaround_keypoints": ctx.turnaround_kp,
        "keypoint_model": POSE_MODEL,
        "keypoint_calls": pose.log.calls[pose_calls0:],
        # Live calls that ran on an ephemeral app instead of the deployment, and why (`PoseClient._open`).
        "keypoint_fallbacks": sorted({c["fallback"] for c in pose.log.calls[pose_calls0:] if c.get("fallback")}),
        "keypoint_cost_basis": SAPIENS_COST_BASIS,
        "inputs": ("the cover; a pose reference (views/front.png, or the nearer profile once the request "
                   f"reaches {PROFILE_REFERENCE_FROM_DEG:.0f} degrees) and a mannequin turn guide at the "
                   "requested yaw, both centred on a canvas the cover's size at one scale (the figure "
                   f"{POSE_REFERENCE_FILL} of its height, less if a drawing would pass "
                   f"{POSE_REFERENCE_MAX_WIDTH} of its width); each view keeps its kept generation's under "
                   "heldout/inputs/"),
        "requested_size": ctx.req_size,
        "image_size": ctx.size,
        "scale_basis": "1 / the framing scale the pose reference and the turn guide share (the prompt "
                       "asks for the guide's size and place), times input width / generated width",
        "front_bbox_px": list(fb),
        "read_by": "scoring only, accepted views only (`accepted_views`: status accepted and the image's "
                   "recorded SHA-256); no building stage reads heldout/",
    })
    st.notes.append(f"Held-out views from {IMAGE_MODEL}, not used to build any body.")
    try:
        _commit(run, st, prev, [v.name for v in chosen], kept_outputs, moves, decisions)
    except BaseException as e:
        stopped(f"swapping the new files in failed: {type(e).__name__}: {e}"[:800], status="error")
        raise
    # This run's cost plus the runs that stopped since the run the earlier record describes.
    since = [r for r in earlier_stops if str(r.get("started", "")) > str(prev.get("started", ""))]
    stops_cost = sum(r.get("cost_usd", 0.0) for r in since)
    run_cost = pose_cost()
    return st.finish(lambda rec: rec.update({"run_cost_usd": round(rec["cost_usd"] + run_cost, 6),
                                             "stopped_runs": since,
                                             "models": rec["models"] + [POSE_MODEL],
                                             "cost_usd": round(rec["cost_usd"] + run_cost + stops_cost, 6)}))


def accepted_views(run: RunFolder) -> dict[str, str]:
    """The held-out views a scorer may use, from `stage-2h.json` (never from the folder layout): view name
    -> image path, for each view whose status is accepted, that was made from the current turnaround
    (`stage2_identity`) and whose image still has its recorded SHA-256. None while stage 2h is running. A
    record from before per-view statuses (one `details.view`) counts when it passed."""
    rec = run.read_stage("2h")  # type: ignore[arg-type]
    if not rec or rec.get("status") == "running":
        return {}
    det = rec.get("details") or {}
    out = {}
    identity = stage2_identity(run) if det.get("views") else ""
    for name, d in (det.get("views") or {}).items():
        img = heldout_rel((d.get("files") or {}).get("image"))
        sha = d.get("image_sha256")
        if method_kind(str(d.get("mask_method_detail") or d.get("mask_method") or SAM_MODEL)) != SAM_MODEL:
            continue  # measured on a threshold mask by an earlier version: never counts as accepted
        if d.get("status") == "accepted" and img and sha and d.get("stage2_identity") == identity \
                and (run.path / img).is_file() and file_sha256(run.path / img) == sha:
            out[name] = img
    if not det.get("views") and isinstance(det.get("view"), str) and rec.get("ok"):
        img = f"heldout/{det['view']}.png"
        if img in rec.get("outputs", []) and run.exists(img):
            out[det["view"]] = img
    return out


# ---- human review --------------------------------------------------------------------------------------

DECISIONS = ("accepted", "rejected")


def _profile_values(run: RunFolder, rec: dict[str, Any]
                    ) -> tuple[TurnaroundModel | None, float, float | None, bool, KP.BodyModel | None]:
    """The turnaround models and profile cues a recorded run was judged with, without model calls."""
    m = {v: imaging.from_mask_image(_load(run, f"masks/{v}.png")) for v in ("front", "left", "right")}
    det = rec.get("details") or {}
    pf = det.get("profile_face") or {}
    return (turnaround_model(m["front"], m["left"], m["right"]),
            float(np.mean([abs(toe_offset_h(m[v])) for v in ("left", "right")])),
            det.get("profile_face_offset"),
            any(isinstance(f, dict) and f.get("api_error") for f in pf.values()),
            body_model(det.get("turnaround_keypoints")))


def recheck_heldout(run: RunFolder) -> dict[str, Any]:
    """Judge every recorded view again under the current checks (no model calls; each must come from the
    current turnaround), move files to their new status's folder and rewrite `stage-2h.json`."""
    rec = run.read_stage("2h")  # type: ignore[arg-type]
    if not rec or not (rec.get("details") or {}).get("views"):
        raise FileNotFoundError("this run folder has no stage 2h record; run stage 2h first")
    if rec.get("status") == "running":
        raise ValueError("stage 2h is running in this run folder; wait for it to finish")
    identity = stage2_identity(run)
    views = rec["details"]["views"]
    for name in views:
        if name not in VIEW_BY_NAME:
            raise ValueError(f"{name} is not a current held-out view; run stage 2h again")
        check_provenance(run, rec, name, identity)
    model, toe, face, face_err, body = _profile_values(run, rec)
    checks = [c for c in rec["checks"] if c["name"].split(".", 1)[0] not in views]
    moves: dict[str, str] = {}
    for name, d in list(views.items()):
        new, ch, mv = recheck_view(run, name, d, model, toe, face, face_err, identity, body)
        views[name] = {**new, "run_started": _view_started(rec, name)}
        checks += [{**c, "name": f"{name}.{c['name']}"} for c in ch] + _status_checks(name, views[name])
        moves.update(mv)
    move_files(run, moves)
    try:
        rec["checks"] = checks
        rec["ok"] = C.gate_ok(checks) and not rec.get("error")  # a stopped run's error stays
        rec["outputs"] = [moves.get(o, o) for o in rec.get("outputs", [])]
        rec["details"].update({**_summary(views), "checks_version": checks_version(),
                               "stage2_identity": identity, "fit_error_deg": list(FIT_ERROR_DEG),
                               "build_azimuths_deg": list(building_azimuths(run)),
                               "azimuth_clearance_deg": AZIMUTH_CLEARANCE_DEG})
        for old in ("yaw_uncertainty_deg", "found_yaw_clearance_deg", "hand_z_sigmas"):
            rec["details"].pop(old, None)
        rec["notes"] = (rec.get("notes", "") + f" Checked again under checks {checks_version()} at "
                        f"{now_iso()}: " + ", ".join(f"{n} {d['status']}" for n, d in views.items()) + ".")
        _remove_empty_status_dirs(run)
        run.write_stage(rec)
    except BaseException:
        move_files(run, {b: a for a, b in moves.items() if a != b})
        raise
    return rec


def review_view(run: RunFolder, view_name: str, decision: str, *, note: str = "", by: str) -> dict[str, Any]:
    """Record a person's decision on one held-out view and move its image, mask and raw generation to
    `heldout/` (accepted) or `heldout/rejected/` (rejected).

    The decision is appended to `details.decisions` in `stage-2h.json` (who, when, the decision, the note,
    the status before and after, the image's SHA-256), the view's `status` and `review` are updated and the
    gate is re-evaluated: it passes once every view is accepted. A view whose own gated checks failed can be
    rejected but not accepted: review decides what a check could not, it doesn't overrule a failed gate. The
    view must come from the current turnaround and have been judged under the current checks (`recheck`),
    and its image must still be the one judged. The files move all or none, and back if the record can't be
    written. Returns the updated record."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    if not by.strip():
        raise ValueError("say who decided (--by)")
    rec = run.read_stage("2h")  # type: ignore[arg-type]
    if not rec or not (rec.get("details") or {}).get("views"):
        raise FileNotFoundError("this run folder has no stage 2h record; run stage 2h first")
    if rec.get("status") == "running":
        raise ValueError("stage 2h is running in this run folder; wait for it to finish")
    views = rec["details"]["views"]
    if view_name not in views:
        raise ValueError(f"no held-out view {view_name!r} in this run; it has {list(views)}")
    if view_name not in VIEW_BY_NAME:
        raise ValueError(f"{view_name} is not a current held-out view; run stage 2h again")
    d = views[view_name]
    check_provenance(run, rec, view_name, stage2_identity(run))
    if d.get("checks_version") != checks_version():
        raise ValueError(f"{view_name} was judged under other checks ({d.get('checks_version')}); judge it "
                         f"again first: python3 -m herogen.pipeline recheck --run-id {run.run_id}")
    img = heldout_rel(d["files"].get("image"))
    if img is None or not (run.path / img).is_file():
        raise FileNotFoundError(f"{view_name}'s image {d['files'].get('image')!r} is missing")
    if file_sha256(run.path / img) != d.get("image_sha256"):
        raise ValueError(f"{img} is not the image that was checked (its SHA-256 differs); run stage 2h again")
    machine = d.get("machine_status") or d["status"]
    if decision == "accepted" and machine == "rejected":
        failed = [c["name"] for c in rec["checks"]
                  if c["name"].startswith(view_name + ".") and c.get("gate", True) and not c["pass"]]
        raise ValueError(f"{view_name} failed gated checks ({', '.join(failed)}); it can't be accepted on "
                         "review")
    target = view_paths(VIEW_BY_NAME[view_name], decision)
    old_files = dict(d["files"])
    moves = {old_files[k]: target[k] for k in target}
    move_files(run, moves)
    try:
        entry = {
            "view": view_name,
            "decision": decision,
            "by": by.strip(),
            "at": now_iso(),
            "note": note,
            "from_status": d["status"],
            "to_status": decision,
            "machine_status": machine,
            "image": target["image"],
            "image_sha256": d["image_sha256"],
        }
        d.update({"status": decision, "machine_status": machine, "accepted": decision == "accepted",
                  "files": target, "review": entry})
        details = rec["details"]
        details.setdefault("decisions", []).append(entry)
        details.update(_summary(views))
        rec["outputs"] = [moves.get(o, o) for o in rec.get("outputs", [])]
        rec["checks"] = [c for c in rec["checks"] if c["name"] != f"{view_name}.status_accepted"]
        rec["checks"] += _status_checks(view_name, d)
        rec["ok"] = C.gate_ok(rec["checks"]) and not rec.get("error")
        rec["notes"] = (rec.get("notes", "") + f" {view_name}: {decision} on human review by {entry['by']}"
                        + (f" ({note})." if note else ".")).strip()
        _remove_empty_status_dirs(run)
        run.write_stage(rec)
    except BaseException:
        move_files(run, {b: a for a, b in moves.items() if a != b})
        raise
    return rec
