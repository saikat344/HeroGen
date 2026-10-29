"""Synthetic held-out 3/4 views: the stage-2 test humanoid (`herogen.blender.synth`) seen at any yaw.

The humanoid's parts are capsules, axis-aligned ellipsoids and axis-aligned rounded boxes. Turned about the
vertical axis and projected orthographically each keeps an exact closed-form silhouette, so the masks are
exact at every yaw:

- capsule: a stadium between the projected end points, same radius;
- ellipsoid with radii (rx, ry, rz): an ellipse with half-width sqrt((rx cos)^2 + (ry sin)^2), half-height rz;
- rounded box (a box shrunk by its rounding, plus a ball): a rounded rectangle with half-width
  (hx - r)|cos| + (hy - r)|sin| + r, half-height hz, rounding r.

Yaw follows `herogen/blender/views.py`: the camera sits at (sin yaw, -cos yaw, 0) looking at the figure, so
image right is u = X cos yaw + Y sin yaw. Yaw 0 is the front view, +90 the left view, -90 the right view;
+45 is the front-left 3/4 view (the character faces image left) and -45 the front-right one.

Run this file directly to print the measurement table the held-out thresholds were set from:
``python3 tests/test_herogen_pipeline_heldout_synth.py`` (``--test`` runs its tests instead).
"""

from __future__ import annotations

import math
import os
import re
import sys
import unittest
from unittest import mock

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.blender import synth  # noqa: E402  (read-only use of the stage-2 test humanoid)
from herogen.pipeline import heldout, imaging  # noqa: E402

SIZE = 512
PPU = 80.0  # pixels per stud: the 5.5-stud figure is 440 px tall, the same scale as synth.render_masks


def _turn(p, c: float, s: float) -> np.ndarray:
    """Image (u, v) of a world point for the camera at yaw with cos c, sin s."""
    x, y, z = (float(t) for t in p)
    return np.array([x * c + y * s, z])


def _prim(part, c: float, s: float):
    if isinstance(part, synth.Capsule):
        return ("capsule", _turn(part.a, c, s), _turn(part.b, c, s), part.r)
    if isinstance(part, synth.Ellipsoid):
        rx, ry, rz = part.radii
        return ("ellipse", _turn(part.c, c, s), np.array([math.hypot(rx * c, ry * s), rz]))
    hx, hy, hz = part.half
    r = part.round
    return ("rrect", _turn(part.c, c, s), np.array([(hx - r) * abs(c) + (hy - r) * abs(s) + r, hz]), r)


def turned_mask(fig: synth.Figure, yaw_deg: float, size: int = SIZE, ppu: float = PPU,
                only: list | None = None) -> np.ndarray:
    """Orthographic silhouette at ``yaw_deg``, centred horizontally on its box, vertically on Z 0..top
    (the frame of `synth.render_masks`, so every yaw shares the turnaround's rows). ``only`` draws just
    those parts, in the whole figure's frame."""
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    prims = [_prim(p, c, s) for p in fig.parts]
    lo = np.min([synth._bounds(p)[0] for p in prims], axis=0)
    hi = np.max([synth._bounds(p)[1] for p in prims], axis=0)
    u0, v0 = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    idx = (np.arange(size) + 0.5 - size / 2) / ppu
    uu, vv = np.meshgrid(u0 + idx, v0 - idx)
    m = np.zeros((size, size), dtype=bool)
    draw = prims if only is None else [_prim(p, c, s) for p in only]
    for p in draw:
        m |= synth._inside(p, uu, vv)
    return m


def turnaround(fig: synth.Figure, size: int = SIZE, ppu: float = PPU) -> dict[str, np.ndarray]:
    """Front, left, back and right masks of ``fig`` on one baseline, as stage 2 leaves them."""
    return {"front": turned_mask(fig, 0, size, ppu), "left": turned_mask(fig, 90, size, ppu),
            "back": turned_mask(fig, 180, size, ppu), "right": turned_mask(fig, -90, size, ppu)}


def _drop_to_floor(fig: synth.Figure) -> synth.Figure:
    """Shift every part down so the lowest point is on Z = 0 (a stride lowers the hips)."""
    lows = []
    for p in fig.parts:
        if isinstance(p, synth.Capsule):
            lows.append(min(p.a[2], p.b[2]) - p.r)
        elif isinstance(p, synth.Ellipsoid):
            lows.append(p.c[2] - p.radii[2])
        else:
            lows.append(p.c[2] - p.half[2])
    dz = min(lows)
    out = synth.Figure()
    for p in fig.parts:
        if isinstance(p, synth.Capsule):
            out.parts.append(synth.Capsule((p.a[0], p.a[1], p.a[2] - dz), (p.b[0], p.b[1], p.b[2] - dz), p.r))
        elif isinstance(p, synth.Ellipsoid):
            out.parts.append(synth.Ellipsoid((p.c[0], p.c[1], p.c[2] - dz), p.radii))
        else:
            out.parts.append(synth.RBox((p.c[0], p.c[1], p.c[2] - dz), p.half, p.round))
    return out


def face_parts(fig: synth.Figure) -> tuple[list, list]:
    """(face parts, head parts) of a figure from `figure`: what SAM 3.1 would find for `face` and `head`."""
    return [fig.parts[i] for i in fig.face_idx], [fig.parts[i] for i in fig.head_idx]


def figure(pose: str = "a", *, clothed: bool = True, arm_deg: float = 50.0, stride_deg: float = 25.0,
           leg_spread: float = 0.02, bulky: bool = False) -> synth.Figure:
    """The synth humanoid in a held-out test pose.

    pose: "a" (the turnaround's A-pose, arms 50 degrees from vertical), "down" (arms hanging, 8 degrees),
    "lowered" (arms at ``arm_deg``), "stride" (A-pose arms, legs swung ``stride_deg`` forward and back about
    the hips with the back knee bent, the hips dropped onto the floor). ``bulky`` adds deep shoulder armour
    (pauldrons wider and deeper than the arm). Every figure has a face: a flattened ellipsoid on the front
    of the head (`face_parts`).
    """
    arm = {"a": 50.0, "down": 8.0, "lowered": arm_deg, "stride": 50.0}[pose]
    fig = synth.humanoid(arm_angle_deg=arm, leg_spread=leg_spread, clothed=clothed)
    head_idx = [0] + ([17] if clothed else [])  # synth.humanoid's head, then its hair (first clothing part)
    fig.parts.append(synth.Ellipsoid((0.0, -0.30, 4.93), (0.28, 0.18, 0.30)))
    face_idx = [len(fig.parts) - 1]
    if bulky:
        fig.parts += [synth.Ellipsoid((0.78 * sx, 0.0, 4.18), (0.42, 0.46, 0.30)) for sx in (1.0, -1.0)]
    if pose == "stride":
        # synth.humanoid appends head, neck, chest, waist, pelvis, then per side (left, right) upper arm,
        # forearm, hand, thigh, shin, foot: indices 8-10 (left leg) and 14-16 (right leg).
        a = math.radians(stride_deg)
        for first, sx, swing, knee_bend in ((8, 1.0, -1.0, 0.0), (14, -1.0, 1.0, 20.0)):  # left leg forward
            hip = np.array([0.28 * sx, 0.0, 2.70])
            knee = hip + 1.25 * np.array([0.0, swing * math.sin(a), -math.cos(a)])
            knee += [(0.02 + leg_spread) * sx, 0, 0]
            b = a + math.radians(knee_bend)
            ankle = knee + 1.15 * np.array([0.0, swing * math.sin(b), -math.cos(b)]) + [leg_spread * sx, 0, 0]
            fig.parts[first : first + 3] = [
                synth.Capsule(tuple(hip), tuple(knee), 0.22), synth.Capsule(tuple(knee), tuple(ankle), 0.17),
                synth.RBox((ankle[0], ankle[1] - 0.18, ankle[2] - 0.17), (0.14, 0.32, 0.13), 0.06)]
        fig = _drop_to_floor(fig)
    fig.face_idx, fig.head_idx = face_idx, head_idx + face_idx
    return fig


def stick_figure(h: int = 200, pose: str = "a") -> synth.Figure:
    """The stick-and-block figure of `test_herogen_pipeline_synth.figure_mask` as a 3D figure, in px: its
    front mask and profiles are this figure's front and side silhouettes, so a fake turnaround sheet drawn
    with `figure_mask` and a fake held-out view drawn with this agree. pose: "a", "down" (arms hanging),
    "stride" (legs swung 25 degrees forward and back)."""
    f = synth.Figure()
    r = 0.07 * h
    f.parts.append(synth.Ellipsoid((0.0, 0.0, h - r), (r, r, r)))
    f.parts.append(synth.RBox((0.0, 0.0, h - 0.34 * h), (0.14 * h, 0.08 * h, 0.21 * h), 1.0))
    reach = 0.36 * h * math.sin(math.radians(45))
    for sx in (1.0, -1.0):
        top = (0.14 * h * sx, 0.0, h - 0.17 * h)
        end = (top[0] + sx * reach, 0.0, top[2] - reach) if pose != "down" else (top[0] + sx * 0.02 * h, 0.0,
                                                                                     top[2] - 0.36 * h)
        f.parts.append(synth.Capsule(top, end, 0.025 * h))
        swing = math.radians(25.0) * sx if pose == "stride" else 0.0  # left leg (+x) forward (-y)
        hip, ln = (0.08 * h * sx, 0.0, 0.5 * h), 0.5 * h
        ankle = (hip[0], -ln * math.sin(swing), hip[2] - ln * math.cos(swing))
        mid = tuple((a + b) / 2 for a, b in zip(hip, ankle, strict=True))
        if swing:
            f.parts.append(synth.Capsule(hip, ankle, 0.05 * h))
        else:
            f.parts.append(synth.RBox(mid, (0.05 * h, 0.05 * h, ln / 2), 1.0))
        f.parts.append(synth.RBox((ankle[0], ankle[1] - 0.035 * h, ankle[2] + 0.0175 * h),
                                  (0.05 * h, 0.095 * h, 0.0175 * h), 1.0))
    return _drop_to_floor(f)


REQUEST = re.compile(r"(\d+) degrees from the front, the toes and face toward the (left|right) edge")


def fake_heldout_image(prompt: str, yaw: float | None = None, pose: str = "a", frame=None, gain: float = 1.0):
    """A fake Muse Image answer to a held-out prompt: the stick figure turned ``gain`` times the requested
    yaw the way the prompt asks (or ``yaw`` degrees; negative turns it the wrong way), at the size and place
    of the figure in ``frame`` (the last image sent with the prompt), on a canvas ``frame``'s size. The face
    and head are not drawn."""
    from test_herogen_pipeline_synth import paint, place  # noqa: PLC0415

    deg, side = REQUEST.search(prompt).groups()
    turn = gain * float(deg) if yaw is None else yaw
    m = turned_mask(stick_figure(200, pose), turn if side == "left" else -turn, size=400, ppu=1.0)
    b = imaging.bbox(m)
    m = m[b[1] : b[3], b[0] : b[2]]
    fb = imaging.bbox(imaging.threshold_mask(frame))
    k = (fb[3] - fb[1]) / 200.0  # the reference figure's px per stick px (the stick stands 200 px)
    new = (max(1, round(m.shape[1] * k)), max(1, round(m.shape[0] * k)))
    m = imaging.from_mask_image(imaging.to_mask_image(m).resize(new, Image.BILINEAR))
    at = ((fb[0] + fb[2]) // 2 - m.shape[1] // 2, fb[3] - m.shape[0])
    return paint(place(m, frame.size, at), colour=(30, 80, 160), bg=(255, 255, 255))


FIGURES = {"plain": dict(clothed=False), "clothed": dict(clothed=True),
           "plain+armour": dict(clothed=False, bulky=True), "clothed+armour": dict(clothed=True, bulky=True)}
POSES = {"A-pose": ("a", {}), "arms 35": ("lowered", dict(arm_deg=35.0)), "arms down": ("down", {}),
         "stride 8": ("stride", dict(stride_deg=8.0)), "stride 12": ("stride", dict(stride_deg=12.0)),
         "stride 25": ("stride", dict(stride_deg=25.0))}
YAWS = (20, 30, 45, 60, 70)  # the pose and facing table


def face_offset_at(fig: synth.Figure, yaw: float) -> float:
    face, head = face_parts(fig)
    return heldout.face_offset(turned_mask(fig, yaw, only=face), turned_mask(fig, yaw, only=head))


def evaluate(view_name: str, pose: str, yaw: float, figure_kw: dict, pose_kw: dict | None = None,
             seen_fig: synth.Figure | None = None):
    """Checks of a synthetic held-out mask drawn at ``yaw`` against the A-pose turnaround of the same figure,
    as stage 2h measures them (SAM's face and head stand-ins are the figure's face and head parts).
    Returns (status, measurements, checks)."""
    view = heldout.VIEW_BY_NAME[view_name]
    t_fig = figure("a", **figure_kw)
    t = turnaround(t_fig)
    fig = seen_fig or figure(pose, **figure_kw, **(pose_kw or {}))
    seen = turned_mask(fig, yaw)
    model = heldout.turnaround_model(t["front"], t["left"], t["right"])
    profile_toe = float(np.mean([abs(heldout.toe_offset_h(t[v])) for v in ("left", "right")]))
    profile_face = float(np.mean([abs(face_offset_at(t_fig, y)) for y in (90, -90)]))
    face = {"offset": face_offset_at(fig, yaw), "face_found": True, "head_found": True}
    checks, measured = heldout.view_checks(view, seen, model, profile_toe, float(seen.mean()), face,
                                           profile_face)
    return heldout.view_status(checks), measured, checks


def _pose_facing_status(checks) -> str:
    """The status from the pose, facing, height and figure checks alone (no yaw window, no azimuths)."""
    kept = [c for c in checks if "window" not in c["name"] and "azimuth" not in c["name"]]
    return heldout.view_status(kept)


def table() -> list[tuple]:
    """The pose and facing table the thresholds were set from, yaws 20-70 each side: (view, figure, pose,
    yaw, status, status of the mirror image, RMS, found yaw, facing margin, toe turn, face turn). The
    statuses leave out the yaw-window and azimuth checks (`target_table` covers those)."""
    rows = []
    for side, sign in (("left", 1), ("right", -1)):
        for fname, fkw in FIGURES.items():
            for pname, (pose, pkw) in POSES.items():
                for yaw in YAWS:
                    view = f"front-{side}-{30 if yaw < 45 else 60}"
                    _, m, checks = evaluate(view, pose, sign * yaw, fkw, pkw)
                    wrong = _pose_facing_status(evaluate(view, pose, -sign * yaw, fkw, pkw)[2])
                    rows.append((view, fname, pname, yaw, _pose_facing_status(checks), wrong,
                                 m["fit"][f"facing_{side}"]["rms"], m["found_yaw_deg"],
                                 m["facing_fit_margin"], m["toe_turn"], m["face_turn"]))
    return rows


TARGET_OFFSETS = (-8, -5, -3, 0, 3, 5, 8)  # degrees from each view's target


def target_table() -> list[tuple]:
    """The yaw-window table: every view, every figure, A-pose and distorted poses drawn at its target and
    around it: (view, figure, pose, yaw drawn, found yaw, window miss, status)."""
    rows = []
    for view in heldout.VIEWS:
        for fname, fkw in FIGURES.items():
            for pname, (pose, pkw) in POSES.items():
                for off in (TARGET_OFFSETS if pname == "A-pose" else (-3, 0, 3)):
                    yaw = view.yaw_deg + (off if view.yaw_deg > 0 else -off)
                    status, m, _ = evaluate(view.name, pose, yaw, fkw, pkw)
                    rows.append((view.name, fname, pname, yaw, m["found_yaw_deg"], m["yaw_window_miss_deg"],
                                 status))
    return rows


DRIFTS = (-8, -5, -3, 0, 3, 5, 8)  # arm angle of the held-out figure minus the turnaround's, degrees


def drift_table() -> list[tuple]:
    """The drift study `FIT_ERROR_DEG` was set from: every figure, arms drifted by DRIFTS, drawn at
    and half a degree past both edges of every window and at the targets, facing left (the figures are
    mirror-symmetric): (target, drift, figure, yaw drawn, found yaw, found with the plain fit, status)."""
    rows = []
    for tgt in (30, 60):
        lo, hi = heldout.yaw_window(tgt)
        for drift in DRIFTS:
            for fname, fkw in FIGURES.items():
                for y in (lo - 0.5, lo, tgt, hi, hi + 0.5):
                    status, m, _ = evaluate(f"front-left-{tgt}", "lowered", y, fkw, dict(arm_deg=50.0 + drift))
                    rows.append((tgt, drift, fname, y, m["found_yaw_deg"],
                                 m["fit"]["facing_left"]["plain_yaw_deg"], status))
    return rows


def _failed(checks) -> list[str]:
    return [c["name"] for c in checks if not c["pass"]]


class TurnedMaskTest(unittest.TestCase):
    def test_front_and_profiles_match_the_stage2_renderer(self):
        fig = figure("a", clothed=True)
        ours = turnaround(fig)
        theirs, _ = synth.render_masks(fig, SIZE, PPU)
        for v in ("front", "left", "right", "back"):
            self.assertGreater(imaging.iou(ours[v], theirs[v]), 0.995, v)

    def test_mirror_yaws_are_mirror_images_for_a_symmetric_figure(self):
        fig = figure("a", clothed=False)
        a, b = turned_mask(fig, 45), np.fliplr(turned_mask(fig, -45))
        self.assertGreater(imaging.iou(*imaging.align_on_box_centres(a, b)), 0.98)

    def test_stick_figure_matches_the_fake_sheet(self):
        from test_herogen_pipeline_synth import figure_mask  # noqa: PLC0415

        fig = stick_figure(200)
        for view, yaw in (("front", 0), ("left", 90), ("right", -90)):
            m = turned_mask(fig, yaw, size=400, ppu=1.0)
            b = imaging.bbox(m)
            self.assertGreater(imaging.iou(*imaging.align_on_box_centres(m[b[1]:b[3], b[0]:b[2]],
                                                                         figure_mask(200, view))), 0.9, view)


class HeldoutChecksTest(unittest.TestCase):
    """Every check on exact synthetic turned masks: valid A-poses pass on every test figure at and around
    each view's target; distorted poses, the wrong facing, yaws off the window and building azimuths never
    pass."""

    def test_a_pose_accepted_in_every_window(self):
        # Drawn at and around each target, up to its window edge; the fit reads 0-2 degrees low, so the edge
        # nearer the diagonal is tested from inside (the drift tests cover the edges).
        drawn = {30: (26, 28, 30, 32, 33), 60: (59, 60, 62, 64, 65)}
        for fname, fkw in FIGURES.items():
            for view in heldout.VIEWS:
                for d in drawn[abs(int(view.yaw_deg))]:
                    y = math.copysign(d, view.yaw_deg)
                    status, m, checks = evaluate(view.name, "a", y, fkw)
                    self.assertEqual(status, "accepted", (fname, view.name, y, _failed(checks)))
                    self.assertLessEqual(abs(m["found_yaw_deg"] - y), 2, (fname, view.name, y))
                    self.assertEqual(m["facing_votes"], {"landmarks": 1, "toes": 1, "head": 1})

    def test_pose_and_facing_hold_at_every_yaw(self):
        # The pose and facing checks alone, 20-70 degrees: what the thresholds were set from.
        for fname, fkw in FIGURES.items():
            for side, sign in (("left", 1), ("right", -1)):
                for yaw in YAWS:
                    _, _, checks = evaluate(f"front-{side}-30", "a", sign * yaw, fkw)
                    self.assertEqual(_pose_facing_status(checks), "accepted", (fname, side, yaw))

    def test_distorted_poses_never_accepted(self):
        for fname, fkw in FIGURES.items():
            for pname in ("arms 35", "arms down", "stride 8", "stride 12", "stride 25"):
                pose, pkw = POSES[pname]
                for view in heldout.VIEWS:
                    status, _, checks = evaluate(view.name, pose, view.yaw_deg, fkw, pkw)
                    self.assertNotEqual(status, "accepted", (fname, pname, view.name))
                    if pname in ("arms 35", "arms down", "stride 25"):
                        self.assertEqual(status, "rejected", (fname, pname, view.name))
                        self.assertIn("pose_landmark_rms", _failed(checks))

    def test_small_stride_needs_review(self):
        status, _, checks = evaluate("front-left-60", "stride", 60, FIGURES["plain"], dict(stride_deg=8.0))
        self.assertEqual(status, "needs review")
        self.assertEqual(_failed(checks), ["pose_landmark_rms_clear"])

    def test_wrong_facing_rejected(self):
        for fname, fkw in FIGURES.items():
            for view in heldout.VIEWS:
                status, m, checks = evaluate(view.name, "a", -view.yaw_deg, fkw)
                self.assertEqual(status, "rejected", (fname, view.name))
                self.assertIn("facing_cues_against", _failed(checks))
                self.assertEqual(m["facing_votes"], {"landmarks": -1, "toes": -1, "head": -1})

    def test_yaw_just_outside_the_window_needs_review(self):
        for name, yaw in (("front-left-30", 22), ("front-right-30", -22), ("front-left-60", 68),
                          ("front-right-60", -68)):
            status, m, checks = evaluate(name, "a", yaw, FIGURES["plain"])
            self.assertEqual(status, "needs review", (name, yaw))
            self.assertEqual(_failed(checks), ["found_yaw_in_window"])
            self.assertTrue(0 < m["yaw_window_miss_deg"] <= heldout.YAW_WINDOW_REVIEW_DEG)

    def test_yaw_far_from_the_window_rejected(self):
        for name, yaw in (("front-left-30", 18), ("front-left-60", 75), ("front-right-60", -30)):
            status, m, checks = evaluate(name, "a", yaw, FIGURES["plain"])
            self.assertEqual(status, "rejected", (name, yaw))
            self.assertIn("found_yaw_window_miss_deg", _failed(checks))
        self.assertEqual(heldout.window_miss(heldout.VIEW_BY_NAME["front-right-60"], -52.0), 5.0)
        self.assertEqual(heldout.window_miss(heldout.VIEW_BY_NAME["front-left-30"], 34.0), 1.0)
        self.assertEqual(heldout.window_miss(heldout.VIEW_BY_NAME["front-right-60"], -60.0), 0.0)

    def test_building_azimuths_rejected(self):
        for name, yaw in (("front-left-30", 0), ("front-left-30", 5), ("front-left-30", 42),
                          ("front-right-60", -45), ("front-right-60", -50)):
            status, m, checks = evaluate(name, "a", yaw, FIGURES["clothed"])
            self.assertEqual(status, "rejected", (name, yaw))
            self.assertIn("found_yaw_clear_of_build_azimuths_deg", _failed(checks))
        for yaw in (85, 90):  # the profile camera: no turned fit either
            self.assertEqual(evaluate("front-left-60", "a", yaw, FIGURES["clothed"])[0], "rejected", yaw)
        self.assertEqual(heldout.azimuth_clearance(-30.0), 15.0)
        self.assertEqual(heldout.azimuth_clearance(-35.0), 10.0)
        self.assertEqual(heldout.azimuth_clearance(-81.0), 9.0)
        self.assertEqual(heldout.azimuth_clearance(178.0), 2.0)
        lo_err, hi_err = heldout.FIT_ERROR_DEG
        for view in heldout.VIEWS:  # a view found at a window edge by both fits still clears
            for y in view.window_deg:
                self.assertGreaterEqual(heldout.range_clearance(y - 2.0, y + 2.0),
                                        heldout.AZIMUTH_CLEARANCE_DEG, (view.name, y))
        self.assertEqual([v.window_deg for v in heldout.VIEWS], [(25.0, 33.0)] * 2 + [(57.0, 65.0)] * 2)
        # Found 34-35 is in the old 25-35 window but its true yaw can be within 10 degrees of the diagonal.
        _, m, checks = evaluate("front-left-30", "a", 35, FIGURES["plain"])
        clear = next(c for c in checks if c["name"] == "found_yaw_clear_of_build_azimuths_deg")
        self.assertEqual((clear["threshold"], clear["pass"]), (10.0, False))
        margin = max(-lo_err, hi_err)  # the larger bound, both sides
        self.assertEqual(m["true_yaw_range_deg"], [m["found_yaw_range_deg"][0] - margin,
                                                   m["found_yaw_range_deg"][1] + margin])
        # 60-degree views drawn just inside the old window edge read 55-56 and are rejected, not sent to review.
        for fname in ("plain", "clothed+armour"):
            for y in (56.0, 56.5, 57.0):
                status, _, checks = evaluate("front-left-60", "a", y, FIGURES[fname])
                self.assertEqual(status, "rejected", (fname, y))
                self.assertIn("found_yaw_clear_of_build_azimuths_deg", _failed(checks))
        self.assertEqual(heldout.range_clearance(30.0, 40.0), 5.0)
        self.assertEqual(heldout.range_clearance(-50.0, -40.0), 0.0)
        self.assertEqual(heldout.range_clearance(-34.0, -20.0), 11.0)

    def test_arm_drift_does_not_move_the_found_yaw(self):
        # Arms held 5 or 8 degrees higher or lower than the turnaround's, drawn at both edges of both
        # targets' windows, both sides: found minus drawn |yaw| stays inside FIT_ERROR_DEG (the
        # turnaround-arms fit was off by up to 12).
        for fname in ("plain", "clothed+armour"):
            for view in heldout.VIEWS:
                for edge in view.window_deg:
                    for drift in (-8, -5, 5, 8):
                        y = math.copysign(edge, view.yaw_deg)
                        _, m, _ = evaluate(view.name, "lowered", y, FIGURES[fname], dict(arm_deg=50.0 + drift))
                        err = abs(m["found_yaw_deg"]) - abs(y)
                        self.assertTrue(heldout.FIT_ERROR_DEG[0] <= err <= heldout.FIT_ERROR_DEG[1],
                                        (fname, view.name, y, drift, m["found_yaw_deg"]))

    def test_arm_drift_near_a_diagonal_never_accepted(self):
        # Drawn closer than AZIMUTH_CLEARANCE_DEG to the diagonal with drifted arms (raised arms used to read
        # 35 from 39, lowered ones 32 from 27): never accepted, whichever window it is judged by.
        for fname in ("plain", "clothed"):
            for name, ys in (("front-left-30", (35.5, 36, 39)), ("front-right-30", (-36, -39)),
                             ("front-left-60", (51, 54.5)), ("front-right-60", (-51, -54))):
                for drift in (-8, -5, 0, 5, 8):
                    for y in ys:
                        status, m, checks = evaluate(name, "lowered", y, FIGURES[fname],
                                                     dict(arm_deg=50.0 + drift))
                        self.assertNotEqual(status, "accepted", (fname, name, y, drift, m["found_yaw_deg"]))

    def test_yaw_range_spans_both_fits(self):
        # Arms as in the turnaround: the drift-robust and plain fits agree, the range is a degree at most.
        for fname in ("plain", "clothed+armour"):
            for name, y in (("front-left-30", 30), ("front-right-60", -60)):
                _, m, _ = evaluate(name, "a", y, FIGURES[fname])
                lo, hi = m["found_yaw_range_deg"]
                self.assertLessEqual(hi - lo, 1.0, (fname, name))
                self.assertLess(m["hand_z_morph"]["view"], 0.001)  # information only
        # Arms 5 degrees lower: the plain fit reads high, the range covers both readings.
        _, m, _ = evaluate("front-left-30", "lowered", 30, FIGURES["plain"], dict(arm_deg=45.0))
        fit = m["fit"]["facing_left"]
        self.assertEqual(m["found_yaw_range_deg"], sorted([fit["yaw_deg"], fit["plain_yaw_deg"]]))
        self.assertGreaterEqual(fit["plain_yaw_deg"] - fit["yaw_deg"], 2.0)
        margin = max(abs(e) for e in heldout.FIT_ERROR_DEG)
        self.assertEqual(m["true_yaw_range_deg"], [m["found_yaw_range_deg"][0] - margin,
                                                   m["found_yaw_range_deg"][1] + margin])

    def _judge_with_hands_off(self, name: str, y: float, dz: float, drift: float = 0.0):
        """Checks of a view drawn at ``y`` with the arms ``drift`` degrees off the turnaround's and both hands
        measured ``dz`` of the figure height too high (the turnaround measured correctly)."""
        real = heldout.landmarks
        t = turnaround(figure("a", clothed=False))
        model = heldout.turnaround_model(t["front"], t["left"], t["right"])
        fig = figure("lowered", clothed=False, arm_deg=50.0 + drift) if drift else figure("a", clothed=False)
        seen = turned_mask(fig, y)

        def shifted(mask):
            lm = real(mask)
            if lm is not None:
                lm.hands = [(x, z + dz) for x, z in lm.hands]
            return lm

        with mock.patch.object(heldout, "landmarks", shifted):
            checks, m = heldout.view_checks(heldout.VIEW_BY_NAME[name], seen, model, 0.3, 0.2, {"offset": None},
                                            None)
        return next(c for c in checks if c["name"] == "found_yaw_clear_of_build_azimuths_deg"), m

    def test_shared_hand_height_error_fails_the_clearance(self):
        # Both hands measured 0.01-0.02 of the height too high or too low, which the drift-robust fit reads as
        # arm drift: a view drawn within 10 degrees of the diagonal fails the clearance check (with the robust
        # fit alone it passed, at 36-37 degrees).
        for dz in (-0.02, -0.01, 0.01, 0.02):
            for name, y in (("front-left-30", 35.5), ("front-left-30", 36), ("front-left-30", 37),
                            ("front-left-60", 54), ("front-left-60", 55), ("front-left-60", 56.5)):
                clear, m = self._judge_with_hands_off(name, y, dz)
                self.assertFalse(clear["pass"], (dz, name, y, m["found_yaw_deg"], m["true_yaw_range_deg"]))

    def test_raised_arms_with_low_hands_miss_by_at_most_two_degrees(self):
        # The documented corner: arms raised 3-8 degrees and both hands measured 0.01-0.02 low push both fits
        # the same way. A view drawn up to 2 degrees too near the diagonal can pass the clearance (its true
        # clearance 8-10 degrees); one drawn nearer fails it.
        passed = []
        for drift in (3, 5, 8):
            for dz in (-0.02, -0.015, -0.01):
                for y in (35.5, 36, 37, 38, 39):
                    clear, m = self._judge_with_hands_off("front-left-30", y, dz, drift)
                    true_clearance = heldout.azimuth_clearance(y)
                    if clear["pass"]:
                        passed.append(y)
                        self.assertGreaterEqual(true_clearance, heldout.AZIMUTH_CLEARANCE_DEG - 2.0,
                                                (drift, dz, y, m["true_yaw_range_deg"]))
        self.assertTrue(passed)  # the corner exists, as documented
        self.assertLessEqual(max(passed), 37)

    def test_drift_compensation_follows_the_arm(self):
        # The front view's forearm runs outward and down; a hand seen higher moves outward along it.
        t = turnaround(figure("a", clothed=False))
        model = heldout.turnaround_model(t["front"], t["left"], t["right"])
        left, right = model.arm_slope
        self.assertLess(left, -heldout.ARM_SLOPE_MIN)
        self.assertGreater(right, heldout.ARM_SLOPE_MIN)
        z = [hz for _, hz in model.front.hands]
        x0 = model.hand_xs()
        raised = model.hand_xs((z[0] + 0.02, z[1] + 0.02))
        self.assertLess(raised[0], x0[0])
        self.assertGreater(raised[1], x0[1])
        # Raised far enough, the move stops at ARM_DRIFT_MAX_H; up to it, a hand moves along the tangent of its
        # arm's circle: the arms are 50 degrees from vertical, so cot(50) px outward per px up (within 10%,
        # the outer edge is measured from pixels).
        h = model.front.height
        # The cut is about 12 degrees of drift: what a 12-degree raise of the test humanoid's A-pose arm moves
        # its hand landmark, measured on the masks (0.065 of the height).
        raised = heldout.landmarks(turnaround(figure("lowered", clothed=False, arm_deg=62.0))["front"])
        moved = raised.hands[0][1] - model.front.hands[0][1]
        self.assertAlmostEqual(heldout.ARM_DRIFT_MAX_H, moved, delta=0.1 * moved)
        self.assertEqual(heldout.ARM_DRIFT_MAX_H, 0.06)
        cap = 0.06  # the cut, written out so a changed constant fails here
        far = model.hand_xs((z[0] + 1.0, z[1] + 1.0))
        self.assertEqual(far, model.hand_xs((z[0] + cap, z[1] + cap)))
        self.assertNotEqual(far, model.hand_xs((z[0] + 0.9 * cap, z[1] + 0.9 * cap)))
        expected = cap * h / math.tan(math.radians(50.0))
        self.assertAlmostEqual(x0[0] - far[0], expected, delta=0.1 * expected)
        self.assertAlmostEqual(far[1] - x0[1], expected, delta=0.1 * expected)
        # A near-hanging arm isn't moved.
        down = turnaround(figure("down", clothed=False))
        self.assertEqual(heldout.arm_slopes(down["front"], heldout.landmarks(down["front"]).hands), [0.0, 0.0])

    def test_pose_error_measured_against_the_turnaround_arms(self):
        # The drift-robust fit finds the yaw; the pose error there still counts the drifted arms, never less
        # than the plain fit's minimum.
        for drift in (-5, 5):
            _, m, _ = evaluate("front-left-30", "lowered", 30, FIGURES["plain"], dict(arm_deg=50.0 + drift))
            f = m["fit"]["facing_left"]
            self.assertGreaterEqual(f["rms"], f["plain_rms"])
            self.assertLess(f["drift_rms"], f["rms"])

    def test_undecided_facing_needs_review(self):
        # No toe or face cue, and a fit margin below the vote limit: nothing decides the facing.
        t = turnaround(figure("a"))
        model = heldout.turnaround_model(t["front"], t["left"], t["right"])
        seen = turned_mask(figure("a"), 30)
        with mock.patch.object(heldout, "FIT_FACING_MARGIN", 1.0):
            checks, m = heldout.view_checks(heldout.VIEW_BY_NAME["front-left-30"], seen, model, 0.0, 0.2,
                                            {"offset": None}, None)
        self.assertEqual(heldout.view_status(checks), "needs review")
        self.assertEqual(_failed(checks), ["facing_cues_for"])
        self.assertEqual(m["facing_votes"], {"landmarks": 0, "toes": 0, "head": 0})

    def test_unmeasurable_cues_do_not_vote(self):
        t = turnaround(figure("a"))
        model = heldout.turnaround_model(t["front"], t["left"], t["right"])
        seen = turned_mask(figure("a"), 30)
        checks, m = heldout.view_checks(heldout.VIEW_BY_NAME["front-left-30"], seen, model, 0.0, 0.2,
                                        {"offset": None, "face_found": False}, None)
        self.assertEqual(m["facing_votes"], {"landmarks": 1, "toes": 0, "head": 0})
        self.assertEqual(heldout.view_status(checks), "accepted")

    def test_no_figure_rejected(self):
        t = turnaround(figure("a"))
        model = heldout.turnaround_model(t["front"], t["left"], t["right"])
        checks, _ = heldout.view_checks(heldout.VIEWS[0], np.zeros((SIZE, SIZE), bool), model, 0.1, 0.0,
                                        {"offset": None}, None)
        self.assertEqual(heldout.view_status(checks), "rejected")
        self.assertIn("figure_measurable", _failed(checks))

    def test_short_figure_rejected(self):
        t = turnaround(figure("a"))
        model = heldout.turnaround_model(t["front"], t["left"], t["right"])
        seen = turned_mask(figure("a"), 30)
        b = imaging.bbox(seen)
        cut = seen.copy()
        cut[b[1] : b[1] + int(0.08 * (b[3] - b[1]))] = False  # the head cut off
        checks, _ = heldout.view_checks(heldout.VIEWS[0], cut, model, 0.1, 0.2, {"offset": None}, None)
        self.assertIn("height_agreement_front", _failed(checks))

    def test_face_offset_follows_the_head(self):
        fig = figure("a")
        self.assertAlmostEqual(face_offset_at(fig, 0), 0.0, places=3)
        self.assertLess(face_offset_at(fig, 45), -0.2)  # front-left: the face toward image left
        self.assertGreater(face_offset_at(fig, -45), 0.2)


class InputImagesTest(unittest.TestCase):
    """The pose reference and the turn guide."""

    def _grey(self, img) -> np.ndarray:
        return np.array(img.convert("L")) < 250

    def test_reference_and_guide_share_one_scale(self):
        # A wide A-pose (0.88 as wide as tall) on a 2:3 portrait canvas: held to the width, both drawings
        # the same height, and that height is the front figure's times the scale.
        t = turnaround(figure("a", clothed=False))
        size = (300, 450)
        fb = imaging.bbox(t["front"])
        fw, fh = imaging.box_size(fb)
        for yaw in (30.0, -60.0):
            shapes, h = heldout._guide_shapes(t["front"], t["left"], t["right"], yaw)
            k = heldout.framing_scale(h, [fw, heldout.guide_width(shapes)], size)
            self.assertAlmostEqual(k, heldout.POSE_REFERENCE_MAX_WIDTH * size[0] / fw)
            front_img = Image.fromarray((~t["front"]).astype(np.uint8) * 255).convert("RGB")
            ref, ref_h = heldout.pose_reference(front_img, t["front"], size, k)
            guide, guide_h = heldout.turn_guide(t["front"], t["left"], t["right"], yaw, size, k)
            self.assertEqual(ref_h, round(fh * k))
            self.assertLessEqual(abs(ref_h - guide_h), 1)
            for img in (ref, guide):
                b = imaging.bbox(self._grey(img))
                self.assertLessEqual(abs(imaging.box_size(b)[1] - ref_h), 0.02 * ref_h)  # the guide's strokes
                self.assertLessEqual(imaging.box_size(b)[0], heldout.POSE_REFERENCE_MAX_WIDTH * size[0] + 1)
            # Without a scale the guide picks the same rule on its own.
            self.assertEqual(heldout.turn_guide(t["front"], t["left"], t["right"], yaw, size)[1], guide_h)

    def test_generation_inputs_share_the_widest_drawings_scale(self):
        # The stage's own wiring (`_inputs`): one scale for the reference and the guide, set by whichever
        # drawing is widest on a narrow canvas: a deep backpack makes the profile widest at 30 degrees, a
        # shallow one leaves the guide widest.
        size = (200, 600)
        for depth, widest in ((2.0, "profile"), (0.6, "guide")):
            fig = figure("down", clothed=False)
            fig.parts.append(synth.RBox((0.0, 0.35 + depth / 2, 3.6), (0.3, depth / 2, 0.3), 0.1))
            t = turnaround(fig)
            masks = {k: t[k] for k in ("front", "left", "right")}
            views = {k: Image.fromarray(np.where(m, 60, 255).astype(np.uint8)).convert("RGB") for k, m in masks.items()}
            fb = imaging.bbox(t["front"])
            h = fb[3] - fb[1]
            ctx = heldout._Context(cover=Image.new("RGB", size, "white"), views=views, masks=masks, req_size="",
                                   size=SIZE, bottom_row=fb[3], front_h=h, model=None, profile_toe=0.0,
                                   profile_face=None)
            widths = {k: imaging.box_size(imaging.bbox(m))[0] for k, m in masks.items()}
            widths["guide"] = heldout.guide_width(heldout._guide_shapes(t["front"], t["left"], t["right"], 30.0)[0])
            if widest == "profile":
                self.assertGreater(widths["left"], max(widths["guide"], widths["front"]))
            else:
                self.assertGreater(widths["guide"], max(widths["left"], widths["front"]))
            _, images, k, _ = heldout._inputs(heldout.VIEW_BY_NAME["front-left-30"], ctx, 30.0)
            expected = heldout.POSE_REFERENCE_MAX_WIDTH * size[0] / max(widths.values())
            self.assertAlmostEqual(k, expected, places=9, msg=widest)
            ref_h = imaging.box_size(imaging.bbox(self._grey(images[1])))[1]
            guide_h = imaging.box_size(imaging.bbox(self._grey(images[2])))[1]
            self.assertLessEqual(abs(ref_h - h * k), 2, widest)
            self.assertLessEqual(abs(guide_h - h * k), 0.02 * h * k, widest)  # the guide's strokes

    def test_guide_ignores_a_sideways_offset_between_the_profiles(self):
        t = turnaround(figure("a", clothed=True))
        size = (300, 450)
        base = self._grey(heldout.turn_guide(t["front"], t["left"], t["right"], 30.0, size)[0])
        shifted = np.zeros_like(t["right"])
        shifted[:, 40:] = t["right"][:, :-40]  # the right profile drawn 40 px to the side
        moved = self._grey(heldout.turn_guide(t["front"], t["left"], shifted, 30.0, size)[0])
        self.assertGreater(imaging.iou(base, moved), 0.99)

    def test_guide_draws_legs_that_touch(self):
        # Legs pressed together (one run at the knees and soles): the guide still draws two legs and feet.
        t = turnaround(figure("a", clothed=False, leg_spread=-0.3))
        knees = heldout._runs(heldout._band(t["front"], 0.70, 0.71)[0].any(axis=0))
        self.assertEqual(len(knees), 1)
        guide = self._grey(heldout.turn_guide(t["front"], t["left"], t["right"], 30.0, (300, 450))[0])
        b = imaging.bbox(guide)
        h = b[3] - b[1]
        for frac in (0.75, 0.9, 0.98):
            self.assertTrue(guide[b[1] + int(frac * h) - 1].any(), frac)


class MeasureFaceTest(unittest.TestCase):
    class _Client:
        def __init__(self, parts):
            self.parts = parts

        def segment(self, _img, text):
            return self.parts.get(text, [])

    def _inst(self, m, conf):
        from herogen.pipeline.client import SamInstance  # noqa: PLC0415

        return SamInstance(mask=m, box=imaging.bbox(m), confidence=conf)

    def test_face_inside_head_on_the_figure(self):
        fig = figure("a")
        face, head = face_parts(fig)
        fm, hm = turned_mask(fig, 45, only=face), turned_mask(fig, 45, only=head)
        stray = np.zeros_like(fm)
        stray[:20, :20] = True  # a confident "face" off the figure (a belt buckle, a logo) is ignored
        client = self._Client({"face": [self._inst(stray, 0.99), self._inst(fm, 0.8)],
                               "head": [self._inst(hm, 0.9)]})
        out = heldout.measure_face(client, None, turned_mask(fig, 45))
        self.assertAlmostEqual(out["offset"], heldout.face_offset(fm, hm))

    def test_missing_face_or_error_is_not_measurable(self):
        fig = figure("a")
        out = heldout.measure_face(self._Client({}), None, turned_mask(fig, 45))
        self.assertIsNone(out["offset"])

        from herogen.pipeline.client import ApiError, ClientError, ReplayMiss  # noqa: PLC0415

        class Broken:
            def segment(self, *_a):
                raise ApiError("responses failed: HTTP 503")

        out = heldout.measure_face(Broken(), None, turned_mask(fig, 45))
        self.assertIsNone(out["offset"])
        self.assertTrue(out["api_error"])
        self.assertIn("HTTP 503", out["error"])

        class Missing:  # a partial replay cache: an error, not a missing cue
            def segment(self, *_a):
                raise ReplayMiss("no cached SAM response")

        class Bug:
            def segment(self, *_a):
                raise TypeError("not a measurement")

        class Corrupt:  # a cached mask of the wrong size: the cache is broken, not the API
            def segment(self, *_a):
                raise ClientError("cached mask x-mask-0.png has the wrong size")

        for client, error in ((Missing(), ReplayMiss), (Bug(), TypeError), (Corrupt(), ClientError)):
            with self.assertRaises(error):
                heldout.measure_face(client, None, turned_mask(fig, 45))



if __name__ == "__main__":
    if not sys.argv[1:]:
        print(f"{'view':<15} {'figure':<15} {'pose':<10} yaw  {'status':<13} {'mirrored':<9} rms     found  "
              "margin   toe   face")
        for r in table():
            print(f"{r[0]:<15} {r[1]:<15} {r[2]:<10} {r[3]:>3}  {r[4]:<13} {r[5]:<9} {r[6]:.4f}  "
                  f"{r[7]:+4.0f}  {r[8]:+.4f} {r[9]:+.2f} {r[10]:+.2f}")
        print()
        print(f"{'view':<15} {'figure':<15} {'pose':<10} drawn  found  miss  status")
        for r in target_table():
            print(f"{r[0]:<15} {r[1]:<15} {r[2]:<10} {r[3]:+5.0f}  {r[4]:+5.0f}  {r[5]:4.1f}  {r[6]}")
        print()
        rows = drift_table()
        print("drift study: found yaw minus drawn yaw, degrees (drift-robust fit | plain fit)")
        print("target  drift  min   max  | min   max   accepted/review/rejected")
        for tgt in (30, 60):
            for drift in DRIFTS:
                sel = [r for r in rows if r[0] == tgt and r[1] == drift]
                e = [r[4] - r[3] for r in sel]
                p = [r[5] - r[3] for r in sel]
                n = [sum(r[6] == s for r in sel) for s in ("accepted", "needs review", "rejected")]
                print(f"{tgt:>6}  {drift:+5d}  {min(e):+4.1f} {max(e):+5.1f} | {min(p):+5.1f} {max(p):+5.1f}   "
                      f"{n[0]}/{n[1]}/{n[2]}")
        e = [r[4] - r[3] for r in rows]
        print(f"found minus drawn: {min(e):+.1f} to {max(e):+.1f} degrees (FIT_ERROR_DEG {heldout.FIT_ERROR_DEG})")
    else:
        unittest.main(argv=[sys.argv[0]] + [a for a in sys.argv[1:] if a != "--test"])
