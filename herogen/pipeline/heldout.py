"""Held-out turned views (stage 2h): four extra images of the character that no building stage reads.

After the turnaround passes, four views are made, each its own edit of the cover drawn in the turnaround's
A-pose, with a pose reference and a mannequin turn guide (see "Three input images" below):
`front-left-30`, `front-left-60`, `front-right-30` and `front-right-60`, named by their target yaw.
Front-left views face image left, front-right views image right. The stage writes into its record the
building azimuths a body may be built or coloured from (the four turnaround views and the diagonals between
them, every 45 degrees). A view's whole yaw range must clear every one by 10 degrees: from the drift-robust
fit's yaw to the plain fit's, widened by 2 degrees on each side, the larger bound of the fit's error measured on
synthetic outlines (-2 to +1 degrees; not yet measured on real images). The windows are 25-33 and 57-65
degrees.

Each view is generated up to four times until it is accepted; the best generation is kept. Its found yaw
must land in its window; up to 5 degrees outside needs human review. Its image, mask and raw generation go
to `heldout/<view>.png`, `heldout/<view>-mask.png` and `heldout/<view>-raw.png` when it is accepted, to
`heldout/review/` when a check could not decide (needs human review), and to `heldout/rejected/` when a gated
check failed, so `heldout/<view>.png` only ever holds an accepted view. All are on the same square canvas as
`views/`, at the shared pixel scale; scoring reads only accepted views.

The record is `stage-2h.json`: each view's status, found yaw and attempts under `details.views`, the
building azimuths and clearance (`build_azimuths_deg`, `azimuth_clearance_deg`, `fit_error_deg`), each attempt's yaw ranges,
and `"reference": "generated"` (a score against a generated view measures consistency, not
accuracy). Each view also records the turnaround it was made from, the checks it was judged under and its
image's SHA-256; `accepted_views` gives a scorer the accepted views whose image and turnaround still match.
New files are staged and swapped in only when the stage finishes; until then, and if it stops early, the
earlier record and files stay (the record says why it stopped, with every call made). Only scoring reads
`heldout/` (see docs/ARCHITECTURE.md, "Held-out views").

Pose and angle are checked camera-aware: silhouette landmarks (hands, head, knees, shins, soles) are placed in
3D from the front and profile views and projected at every yaw. A fit robust to the arms being held a few
degrees off the A-pose gives the found yaw; the pose error is measured there against the turnaround's own
arms. Facing is three independent cues (the fit, the toes, SAM 3.1 face-in-head).

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
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from herogen.pipeline import checks as C
from herogen.pipeline import imaging
from herogen.pipeline.client import (
    IMAGE_MODEL, SAM_MODEL, ApiError, ModelClient, ReplayMiss, SamInstance,
)
from herogen.pipeline.runfolder import RunFolder
from herogen.pipeline.stages import _load, _sam_or_threshold, _Stage, method_kind

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
    """The next attempt's requested turn: the last request plus the last attempt's shortfall (target minus
    found, both as turns toward the view's side), kept between 5 and MAX_REQUEST_DEG. No found yaw on the
    view's side (the figure faced the other way or wasn't measurable): ask the same again."""
    if found_deg is None:
        return request_deg
    return float(min(MAX_REQUEST_DEG, max(5.0, request_deg + abs(view.yaw_deg) - abs(found_deg))))


# Build/evaluation split. The azimuths a body may be built or coloured from, which this stage writes into its
# record (`build_azimuths_deg`) for later stages to keep to: the four turnaround views and the four diagonals
# between them. A held-out view's true yaw must be at least AZIMUTH_CLEARANCE_DEG from every one of them. The
# true yaw isn't known, only a range for it (`yaw_range`): from the drift-robust fit's yaw to the plain fit's,
# widened on both sides by FIT_ERROR_DEG's larger bound; the whole range must clear.
BUILD_AZIMUTHS_DEG = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
AZIMUTH_CLEARANCE_DEG = 10.0
# Found |yaw| minus true |yaw| on exact silhouettes, measured by the drift study (`drift_table` in
# tests/test_herogen_pipeline_heldout_synth.py; four synthetic figures in the turnaround's A-pose, the
# held-out arms drifted -8 to +8 degrees, drawn at the targets and at and half a degree past both edges of
# every window): -2.0 to +1.0 degrees with the drift-robust fit (-12.5 to +8.5 with the hands taken at their
# turnaround positions). Not symmetric: the 60-degree views read 0.5-2 degrees low. The hands cause it: a
# hand landmark is the outline's outermost point, and a hand is round, so its radius doesn't foreshorten with
# the turn while the fit foreshortens the whole front-view x (centre plus radius). At 60 degrees the seen
# hands are about 0.01 of the figure height each farther out than predicted, which reads as a smaller turn
# (without the hands the same fits read 2-4 degrees high instead). Measured on synthetic outlines only, not
# yet on real images. Not lowered to fit the windows.
FIT_ERROR_DEG = (-2.0, 1.0)
YAW_WINDOW_HALF_DEG = 5.0


def yaw_window(target_deg: float) -> tuple[float, float]:
    """The accepted |found yaw| range for a target: YAW_WINDOW_HALF_DEG either side, cut back to
    AZIMUTH_CLEARANCE_DEG plus the larger FIT_ERROR_DEG bound (2 degrees) from every building azimuth. The cut
    is the same on both sides: the fit error is measured on synthetic outlines only, so its asymmetry isn't
    used to move a window edge nearer an azimuth."""
    margin = AZIMUTH_CLEARANCE_DEG + max(abs(e) for e in FIT_ERROR_DEG)
    lo, hi = target_deg - YAW_WINDOW_HALF_DEG, target_deg + YAW_WINDOW_HALF_DEG
    for a in BUILD_AZIMUTHS_DEG:
        if a <= target_deg:
            lo = max(lo, a + margin)
        else:
            hi = min(hi, a - margin)
    if lo > hi:
        raise ValueError(f"no yaw window around {target_deg} keeps clear of the building azimuths")
    return lo, hi


# Two targets each side, between the front, the diagonal and the profile: 30 degrees (accepted 25-33) and
# 60 degrees (accepted 57-65).
VIEWS = tuple(
    HeldoutView(f"front-{side}-{deg}", sign * deg, yaw_window(deg), -sign)
    for deg in (30, 60) for side, sign in (("left", 1), ("right", -1))
)
VIEW_BY_NAME = {v.name: v for v in VIEWS}


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


# The yaw range a view is judged on. The drift-robust fit reads a change of both hands' heights from the
# turnaround's as arm drift, so a shared error in measuring them (both hands read high, or both low) moves its
# yaw too, about 2 degrees per 0.01 of the figure height at 30 degrees, and nothing measured on one image
# separates that error from real drift. Until the fit's error is measured on real images with known yaws, a
# view's yaw range spans both readings: the drift-robust fit's yaw and the plain fit's (hands where the
# turnaround has them, which is right when there is no drift), widened on both sides by the larger
# FIT_ERROR_DEG bound (2 degrees); the whole range must clear every building azimuth by AZIMUTH_CLEARANCE_DEG.
# With the arms where the turnaround has them the two fits agree; one run's generated 30-degree views read 4-5
# degrees apart. One corner remains (a sweep of 980 synthetic views drawn under 10
# degrees from an azimuth): arms raised 3-8 degrees and both hands measured 0.01-0.02 of the height low push
# both fits low, and 24 of the 980, 30-degree views drawn at 35.5-37 degrees, pass with a true clearance of
# 8-9.5 degrees. Left as a documented limitation until the fit is calibrated on real images.
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


def yaw_range(fit_side: dict[str, Any], side: str) -> dict[str, Any]:
    """From one side of `fit_yaw`: the found-yaw range spanning the drift-robust and the plain fit, and the
    true-yaw range it allows: that range widened on both sides by the larger FIT_ERROR_DEG bound (the error
    is measured on synthetic outlines only, so its asymmetry isn't used to let a view nearer an azimuth; the
    same margin as `yaw_window`), in |yaw|, signed back to the ``side``."""
    sign = 1.0 if side == "left" else -1.0
    found = [abs(fit_side["yaw_deg"]), abs(fit_side["plain_yaw_deg"])]
    margin = max(abs(e) for e in FIT_ERROR_DEG)
    lo, hi = min(found) - margin, max(found) + margin
    return {"found_deg": sorted([sign * min(found), sign * max(found)]),
            "true_deg": sorted([sign * lo, sign * hi])}


def range_clearance(lo_deg: float, hi_deg: float) -> float:
    """Angular distance from the yaw range ``lo_deg``..``hi_deg`` to the nearest building azimuth (0 when one
    is inside it), in degrees."""
    if any(lo_deg <= (a + 180.0) % 360.0 - 180.0 <= hi_deg for a in BUILD_AZIMUTHS_DEG):
        return 0.0
    return min(azimuth_clearance(lo_deg), azimuth_clearance(hi_deg))


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
    recorded, with ``api_error`` set, and `view_checks` then sends the view to human review rather than judge
    its facing without the cue. Anything else is raised: a replay miss (a partial cache must not quietly drop
    a vote), a corrupt cache entry, a bug."""
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
# Pose: RMS landmark error (`landmark_errors`, against the turnaround's own arms) at the found yaw, as a
# fraction of the figure height. A-pose 0.0024-0.0116 (40 of 40 accepted, found yaw 0-2 degrees under the
# drawn yaw); arms lowered to 35 degrees 0.027-0.052, hanging 0.039-0.110, stride 8 degrees 0.018-0.054,
# 12 degrees 0.016-0.070, 25 degrees 0.036-0.120 (none accepted: 192 rejected, 8 of the smallest strides
# need review). Arms drifted 5 degrees from the turnaround's: 0.011-0.017 (at the targets and window
# edges); 8 degrees 0.016-0.024. Generated runs: an earlier 3/4 view (a stride with lowered arms, drawn
# facing right) measured 0.030 as a front-right view and 0.034 as a front-left one with the plain fit (never
# less at the found yaw), rejected either way; on the run in the "Three input images" note, generations
# with the arms raised a little over the turnaround's measure 0.0136-0.0142 at the found yaw (0.0116-0.0133
# at the plain fit's yaw, which absorbed the arms into the turn). Up to 0.0135 passes, above 0.020 fails,
# between needs human review.
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
# The view's own yaw window (`yaw_window`): inside it the view can be accepted; up to 5 degrees outside it
# needs human review; further out it is rejected (and generated again). Synthetic (the yaw-window table, 4
# views x 4 figures, A-pose drawn 0, 3, 5 and 8 degrees either side of the target): at the target and 3
# degrees off accepted 40 of 48, the other 8 drawn at 57 and found at 56 (rejected: their yaw range comes
# within 10 degrees of the diagonal); 5 off: accepted at 65 (8), review at 25 (found 24; 8), rejected at 35 and
# 55 (16); 8 off: review at 22 and 68 (16), rejected at 38 and 52 (16). Distorted poses drawn at the target
# and 3 degrees either side: 0 of 240 accepted (238 rejected, 2 review).
YAW_WINDOW_REVIEW_DEG = 5.0


def window_miss(view: HeldoutView, yaw_deg: float) -> float:
    """Degrees by which ``yaw_deg`` falls outside the view's window (0 inside); facing the other way is
    not counted here (the facing checks reject it)."""
    lo, hi = view.window_deg
    y = abs(yaw_deg)
    return max(0.0, lo - y, y - hi)


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
    raw_area_frac: float, face: dict[str, Any], profile_face: float | None, *,
    profile_face_error: bool = False,
) -> tuple[list[C.Check], dict[str, Any]]:
    """The checks for one held-out view (placed at the shared scale) and the measurements behind them.

    Gated checks reject the view; checks marked ``review`` send it to human review when they fail.
    ``profile_face_error``: the profile views' face cue failed on the Model API, so the head cue could not
    vote (needs review, like a failure on this view's face).
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
    rng = yaw_range(best, side)
    clearance = range_clearance(*rng["true_deg"])
    lo_t, hi_t = rng["true_deg"]
    out.append(C.check("found_yaw_clear_of_build_azimuths_deg", clearance, AZIMUTH_CLEARANCE_DEG, op=">=",
                       note=f"found yaw {best['yaw_deg']:+.0f} (plain fit {best['plain_yaw_deg']:+.0f}), "
                            f"true yaw {lo_t:+.0f} to {hi_t:+.0f} (fit error {FIT_ERROR_DEG}); building azimuths "
                            f"{list(BUILD_AZIMUTHS_DEG)}"))
    miss = window_miss(view, best["yaw_deg"])
    lo, hi = view.window_deg
    note = f"found yaw {best['yaw_deg']:+.0f}, window {lo:.0f}-{hi:.0f}"
    out.append(C.check("found_yaw_window_miss_deg", miss, YAW_WINDOW_REVIEW_DEG, op="<=", note=note))
    out.append(_review(C.check("found_yaw_in_window", miss, 0.0, op="<=", note=note)))

    toe = toe_offset_h(heldout)
    toe_turn = view.facing * toe / profile_toe if profile_toe > 0.01 else None
    face_turn = None
    if face.get("offset") is not None and profile_face is not None:
        face_turn = view.facing * face["offset"] / profile_face
    margin = wrong["plain_rms"] - best["plain_rms"]
    votes = {"landmarks": _vote(margin, FIT_FACING_MARGIN), "toes": _vote(toe_turn, TOE_TURN_MIN),
             "head": _vote(face_turn, FACE_TURN_MIN)}
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
    measured = {
        "found_yaw_deg": best["yaw_deg"],
        "fit": fit,
        "hand_z_morph": {"view": hand_z_morph(heldout, seen), "front": model.hand_z_morph,
                         "note": "information only; close to zero by construction"},
        "found_yaw_range_deg": rng["found_deg"],
        "true_yaw_range_deg": rng["true_deg"],
        "azimuth_clearance_deg": clearance,
        "yaw_window_miss_deg": miss,
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


def _two(runs: list[tuple[int, int]], fallback: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    """The image-left and image-right leg's run in a row: the outer two runs, one run split at its middle
    (legs together, or a long coat), or ``fallback`` split when the row is empty."""
    if len(runs) >= 2:
        return runs[0], runs[-1]
    a, b = runs[0] if runs else fallback
    m = (a + b) / 2
    return (a, m), (m, b)


def _guide_shapes(front: np.ndarray, left: np.ndarray, right: np.ndarray, yaw_deg: float
                  ) -> tuple[list[Shape], int]:
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
        b = imaging.bbox(mask)
        return _runs(mask[b[1] + min(int(frac * (b[3] - b[1])), b[3] - b[1] - 1)])

    def central(frac: float) -> tuple[float, float]:
        rr = runs_at(front, frac)
        hit = [r for r in rr if r[0] <= cx <= r[1]]
        return hit[0] if hit else (rr[0] if rr else (cx - 1, cx + 1))

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
    for frac in (0.14, 0.2, 0.3, 0.4, 0.52):
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
    hips = central(0.52)
    knees, soles = _two(runs_at(front, 0.71), hips), _two(runs_at(front, 0.98), hips)
    ky = sum(depth(0.71)) / 2
    toe_y, heel_y = depth(0.98)
    hip_xs = (hips[0] + 0.25 * (hips[1] - hips[0]), hips[0] + 0.75 * (hips[1] - hips[0]))
    for hip_x, (ka, kb), (sa, sb) in zip(hip_xs, knees, soles, strict=True):
        ankle = (u((sa + sb) / 2, ky), 0.94)
        heel = (u((sa + sb) / 2, heel_y), 0.985)
        shapes.append(("line", [(u(hip_x, arm_y), 0.52), (u((ka + kb) / 2, ky), 0.71), ankle], 1.3 * lw))
        shapes.append(("line", [ankle, heel], lw))
        shapes.append(("line", [heel, (u((sa + sb) / 2, toe_y), 0.985)], 1.1 * lw))
    return shapes, h


def guide_width(shapes: list[Shape]) -> float:
    """The mannequin's width in turnaround px, strokes included."""
    us = [p[0] for _, pts, _ in shapes for p in pts]
    pad = max(w for _, _, w in shapes)
    return max(us) - min(us) + 2 * pad


def turn_guide(front: np.ndarray, left: np.ndarray, right: np.ndarray, yaw_deg: float, size: tuple[int, int],
               scale: float | None = None) -> tuple[Image.Image, int]:
    """A plain grey mannequin of the turnaround's pose seen at ``yaw_deg``, on a white canvas of ``size``,
    centred, at ``scale`` canvas px per turnaround px (default `framing_scale` for the front figure and the
    mannequin). Returns the image and the figure's height on it (the front figure's height times the scale).

    Only landmark positions come from the turnaround, never its outline: the head (an ellipse, its width
    from the front view and its depth from the profile, with a small nose on the face side), the torso (the
    body's central run per row as an ellipse), the arms (straight from the shoulders to the hand landmarks,
    in the frontal plane), the legs (hips to the knee and ankle runs; one run split in two when the legs
    touch) and the feet (heel to toe from the profiles' soles), each projected with u = x cos + y sin.
    """
    shapes, h = _guide_shapes(front, left, right, yaw_deg)
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


# Provenance. Each view records the stage-2 identity it was made from (the SHA-256 of the files below), the
# checks version it was judged under (`checks_version`) and its image's SHA-256, so a scorer can tell that an
# accepted view still belongs to the run (`accepted_views`).
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
    `checks` and `imaging` modules. A change to any of them changes the version without anyone bumping it."""
    consts = {}
    for name, value in sorted(globals().items()):
        if name.isupper() and isinstance(value, (int, float, str, tuple, list, dict, np.ndarray)):
            consts[name] = value.tolist() if isinstance(value, np.ndarray) else repr(value)
    sources = [inspect.getsource(f) for f in (
        _runs, _band, _extent, landmarks, arm_slopes, TurnaroundModel, turnaround_model, landmark_errors,
        _rms, fit_yaw, hand_z_morph, yaw_range, range_clearance, azimuth_clearance, window_miss, yaw_window,
        toe_offset_h, face_offset, _vote, _review, view_checks, view_status, C, imaging)]
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


def _inputs(view: HeldoutView, ctx: _Context, request_deg: float
            ) -> tuple[str, list[Image.Image], float, str]:
    """(prompt, input images, framing scale, reference name) for one generation: the pose reference and the
    turn guide share one `framing_scale`."""
    reference = "profile" if request_deg >= PROFILE_REFERENCE_FROM_DEG else "front"
    name = view.side if reference == "profile" else "front"
    m = ctx.masks
    shapes, _ = _guide_shapes(m["front"], m["left"], m["right"], -view.facing * request_deg)
    widths = [imaging.box_size(b)[0] for b in (imaging.bbox(x) for x in m.values()) if b is not None]
    k = framing_scale(ctx.front_h, widths + [guide_width(shapes)], ctx.cover.size)
    ref, _ = pose_reference(ctx.views[name], m[name], ctx.cover.size, k)
    guide, _ = turn_guide(m["front"], m["left"], m["right"], -view.facing * request_deg, ctx.cover.size, k)
    return view_prompt(view, request_deg, reference), [ctx.cover, ref, guide], k, reference


# Overall time limit for one stage 2h run. Each Model API call can wait TIMEOUT_S (240 s) and retry, and a
# run makes up to 16 generations and 52 SAM calls; a typical live run takes 7-12 minutes. The limit is checked
# before each generation, so the generation in flight finishes first.
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
    request = abs(view.yaw_deg)
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
                "so a cache recorded by another version can miss; make the views again, live: --stages 2h "
                "--live") from e
        mask, method = figure_mask(client, raw)
        # Raw px -> input-canvas px (the model may answer at another resolution) -> turnaround px.
        scale = ctx.cover.size[0] / raw.size[0] / framing
        img, placed = place_on_canvas(raw, mask, scale, ctx.size, ctx.bottom_row)
        face = measure_face(client, raw, mask) if mask.any() else {"offset": None, "error": "no figure"}
        checks, measured = view_checks(view, placed, ctx.model, ctx.profile_toe, float(mask.mean()), face,
                                       ctx.profile_face, profile_face_error=ctx.profile_face_error)
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
            "bbox_px": list(imaging.bbox(placed) or ()),
            "measured": measured,
            "checks": checks,
        })
        fit = (measured.get("fit") or {}).get(f"facing_{view.side}", {})
        key = (_RANK[status], measured.get("yaw_window_miss_deg", 99.0), fit.get("rms", 9.0))
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


def _status_checks(name: str, d: dict[str, Any]) -> list[C.Check]:
    """The gated `<view>.status_accepted` check for a view waiting for human review (its own gated checks
    passed, so nothing else closes the gate)."""
    if d["status"] == "needs review":
        return [C.check(f"{name}.status_accepted", 0.0, 1.0, op="==",
                        note="needs human review: a check could not decide")]
    return []


def _summary(per_view: dict[str, Any]) -> dict[str, list[str]]:
    return {f"{key}_views": [v for v, d in per_view.items() if d["status"] == status]
            for key, status in (("accepted", "accepted"), ("needs_review", "needs review"),
                                ("rejected", "rejected"))}


def _ordered(per_view: dict[str, Any]) -> dict[str, Any]:
    order = {v.name: i for i, v in enumerate(VIEWS)}
    return dict(sorted(per_view.items(), key=lambda kv: order.get(kv[0], len(order))))


STAGING_DIR = "heldout/.staging"


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


def _commit(run: RunFolder, st: _HeldoutStage, prev: dict[str, Any]) -> None:
    """Swap a finished run's files in: the earlier run's files (and any stray file at a view's paths) go,
    then the staged files move into place."""
    old = {heldout_rel(r) for r in prev.get("outputs", [])} - {None}
    for view in VIEWS:
        old |= {rel for status in _STATUS_DIR for rel in view_paths(view, status).values()}
    for rel in sorted(old):
        if (run.path / rel).is_file():
            (run.path / rel).unlink()
    for rel in st.outputs:
        if (run.path / STAGING_DIR / rel).is_file():
            run.file(rel)
            os.replace(run.path / STAGING_DIR / rel, run.path / rel)
    shutil.rmtree(run.path / STAGING_DIR, ignore_errors=True)
    _remove_empty_status_dirs(run)


def stage2h_heldout(run: RunFolder, client: ModelClient, *, attempts: int = ATTEMPTS,
                    time_limit_s: float | None = TIME_LIMIT_S) -> dict[str, Any]:
    """Generate, mask and check the four held-out views for a run whose stage 2 passed. Writes only
    `heldout/` and `stage-2h.json`; every earlier output is left as it is. The gate passes when every view
    is accepted; a view that needs human review or was rejected fails it.

    New files are written to STAGING_DIR and swapped in only when every view is made. Meanwhile
    `stage-2h.json` keeps the earlier record, marked as running; if the stage stops early (an error, a replay
    miss, the time limit) it is written with the error and the calls made, and the earlier files stay."""
    s2 = run.read_stage(2)
    if not s2 or not s2.get("ok"):
        raise FileNotFoundError("stage 2 has not passed in this run folder; run stage 2 first")
    chosen = list(VIEWS)
    prev = run.read_stage("2h") or {}  # type: ignore[arg-type]
    identity = stage2_identity(run)
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
    shutil.rmtree(run.path / STAGING_DIR, ignore_errors=True)  # a run that was killed
    per_view: dict[str, Any] = {}

    def assemble() -> None:
        pv = _ordered(per_view)
        st.checks = [c for c in st.checks if not c["name"].endswith(".status_accepted")]
        for name, d in pv.items():
            st.checks += _status_checks(name, d)
        st.details.update({"views": pv, **_summary(pv)})

    def stopped(error: str, status: str) -> dict[str, Any]:
        """The earlier record, its views, files, calls and cost untouched, with this run added to
        `stopped_runs` (its error, calls and cost) and its cost added to the total."""
        run_rec = st._record()
        base = dict(prev) if prev else {**run_rec, "ok": False, "outputs": [], "checks": [], "details": {},
                                        "calls": [], "cost_usd": 0.0, "run_cost_usd": 0.0}
        this = {"started": run_rec["started"], "seconds": run_rec["seconds"], "status": status,
                "error": error, "made": [v.name for v in chosen], "cost_usd": run_rec["cost_usd"],
                "calls": run_rec["calls"]}
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
        )
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
    assemble()
    st.details.update({
        "reference": "generated",
        "reference_note": ("the held-out views are generated images that can invent detail, so a score "
                           "against them measures consistency with the generated character, not accuracy"),
        "build_azimuths_deg": list(BUILD_AZIMUTHS_DEG),
        "azimuth_clearance_deg": AZIMUTH_CLEARANCE_DEG,
        "fit_error_deg": list(FIT_ERROR_DEG),
        "checks_version": checks_version(),
        "stage2_identity": identity,
        "profile_toe_offset_h": round(ctx.profile_toe, 4),
        "profile_face_offset": None if profile_face is None else round(profile_face, 4),
        "profile_face": _rounded(profile_face_found),
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
        _commit(run, st, prev)
    except BaseException as e:
        stopped(f"swapping the new files in failed: {type(e).__name__}: {e}"[:800], status="error")
        raise
    # This run's cost plus the runs that stopped since the run the earlier record describes.
    since = [r for r in earlier_stops if str(r.get("started", "")) > str(prev.get("started", ""))]
    stops_cost = sum(r.get("cost_usd", 0.0) for r in since)
    return st.finish(lambda rec: rec.update({"run_cost_usd": rec["cost_usd"], "stopped_runs": since,
                                             "cost_usd": round(rec["cost_usd"] + stops_cost, 6)}))


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
        if d.get("status") == "accepted" and img and sha and d.get("stage2_identity") == identity \
                and (run.path / img).is_file() and file_sha256(run.path / img) == sha:
            out[name] = img
    if not det.get("views") and isinstance(det.get("view"), str) and rec.get("ok"):
        img = f"heldout/{det['view']}.png"
        if img in rec.get("outputs", []) and run.exists(img):
            out[det["view"]] = img
    return out
