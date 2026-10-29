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
import sys
import unittest

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


def fake_heldout_image(prompt: str, yaw: float = 45.0, pose: str = "a", frame=None):
    """A fake Muse Image answer to a held-out prompt: the stick figure turned ``yaw`` degrees the way the
    prompt asks (a negative ``yaw`` turns it the wrong way), at the size and place the figure has in
    ``frame`` (the pose reference sent with the prompt), on a canvas ``frame``'s size. The face and head are
    not drawn."""
    from test_herogen_pipeline_synth import paint, place  # noqa: PLC0415

    view = next(v for v in heldout.VIEWS if v.prompt == prompt)
    m = turned_mask(stick_figure(200, pose), -view.facing * yaw, size=400, ppu=1.0)
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
YAWS = (20, 30, 45, 60, 70)


def face_offset_at(fig: synth.Figure, yaw: float) -> float:
    face, head = face_parts(fig)
    return heldout.face_offset(turned_mask(fig, yaw, only=face), turned_mask(fig, yaw, only=head))


def evaluate(view_name: str, pose: str, yaw: float, figure_kw: dict, pose_kw: dict | None = None):
    """Checks of a synthetic held-out mask drawn at ``yaw`` against the A-pose turnaround of the same figure,
    as stage 2h measures them (SAM's face and head stand-ins are the figure's face and head parts).
    Returns (status, measurements, checks)."""
    view = heldout.VIEW_BY_NAME[view_name]
    t_fig = figure("a", **figure_kw)
    t = turnaround(t_fig)
    fig = figure(pose, **figure_kw, **(pose_kw or {}))
    seen = turned_mask(fig, yaw)
    model = heldout.turnaround_model(t["front"], t["left"], t["right"])
    profile_toe = float(np.mean([abs(heldout.toe_offset_h(t[v])) for v in ("left", "right")]))
    profile_face = float(np.mean([abs(face_offset_at(t_fig, y)) for y in (90, -90)]))
    face = {"offset": face_offset_at(fig, yaw), "face_found": True, "head_found": True}
    checks, measured = heldout.view_checks(view, seen, model, profile_toe, float(seen.mean()), face,
                                           profile_face)
    return heldout.view_status(checks), measured, checks


def table() -> list[tuple]:
    """(view, figure, pose, yaw, status, status of the mirror image, RMS, found yaw, facing margin, toe turn,
    face turn) for both views, each drawn at its own side's yaw."""
    rows = []
    for view, sign in (("front-left-34", 1), ("front-right-34", -1)):
        side = "left" if sign > 0 else "right"
        for fname, fkw in FIGURES.items():
            for pname, (pose, pkw) in POSES.items():
                for yaw in YAWS:
                    status, m, _ = evaluate(view, pose, sign * yaw, fkw, pkw)
                    wrong, _, _ = evaluate(view, pose, -sign * yaw, fkw, pkw)
                    rows.append((view, fname, pname, yaw, status, wrong, m["fit"][f"facing_{side}"]["rms"],
                                 m["found_yaw_deg"], m["facing_fit_margin"], m["toe_turn"], m["face_turn"]))
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
    """Every gated check on exact synthetic 3/4 masks: valid A-poses pass at every yaw 20-70 on every test
    figure; distorted poses, the wrong facing and building azimuths never pass."""

    def test_a_pose_accepted_at_every_yaw(self):
        for fname, fkw in FIGURES.items():
            for yaw in YAWS:
                for view, y in (("front-left-34", yaw), ("front-right-34", -yaw)):
                    status, m, checks = evaluate(view, "a", y, fkw)
                    self.assertEqual(status, "accepted", (fname, view, y, _failed(checks)))
                    self.assertLessEqual(abs(abs(m["found_yaw_deg"]) - yaw), 4, (fname, view, y))
                    self.assertEqual(m["facing_votes"], {"landmarks": 1, "toes": 1, "head": 1})

    def test_distorted_poses_never_accepted(self):
        for fname, fkw in FIGURES.items():
            for pname in ("arms 35", "arms down", "stride 8", "stride 12", "stride 25"):
                pose, pkw = POSES[pname]
                for yaw in YAWS:
                    status, _, checks = evaluate("front-left-34", pose, yaw, fkw, pkw)
                    self.assertNotEqual(status, "accepted", (fname, pname, yaw))
                    if pname in ("arms 35", "arms down", "stride 25"):
                        self.assertEqual(status, "rejected", (fname, pname, yaw))
                        self.assertIn("pose_landmark_rms", _failed(checks))

    def test_small_stride_needs_review(self):
        status, _, checks = evaluate("front-left-34", "stride", 45, FIGURES["plain"], dict(stride_deg=12.0))
        self.assertEqual(status, "needs review")
        self.assertEqual(_failed(checks), ["pose_landmark_rms_clear"])

    def test_wrong_facing_rejected(self):
        for fname, fkw in FIGURES.items():
            for yaw in (20, 45, 70):
                for view, y in (("front-left-34", -yaw), ("front-right-34", yaw)):
                    status, m, checks = evaluate(view, "a", y, fkw)
                    self.assertEqual(status, "rejected", (fname, view, y))
                    self.assertIn("facing_cues_against", _failed(checks))
                    self.assertEqual(m["facing_votes"], {"landmarks": -1, "toes": -1, "head": -1})

    def test_building_azimuths_rejected(self):
        for yaw in (0, 5, -5):  # the front camera
            status, m, checks = evaluate("front-left-34", "a", yaw, FIGURES["clothed"])
            self.assertEqual(status, "rejected", yaw)
            self.assertIn("found_yaw_clear_of_build_azimuths_deg", _failed(checks))
        for yaw in (85, 90):  # the profile camera: no 3/4 fit either
            self.assertEqual(evaluate("front-left-34", "a", yaw, FIGURES["clothed"])[0], "rejected", yaw)
        self.assertEqual(heldout.azimuth_clearance(-45.0), 45.0)
        self.assertEqual(heldout.azimuth_clearance(-81.0), 9.0)
        self.assertEqual(heldout.azimuth_clearance(178.0), 2.0)

    def test_undecided_facing_needs_review(self):
        # Turned 12 degrees: the pose fits and clears the azimuths, but no facing cue reaches its threshold.
        status, m, checks = evaluate("front-left-34", "a", 12, FIGURES["plain"])
        self.assertEqual(status, "needs review")
        self.assertEqual(_failed(checks), ["facing_cues_for"])
        self.assertEqual(m["facing_votes"], {"landmarks": 0, "toes": 0, "head": 0})

    def test_unmeasurable_cues_do_not_vote(self):
        t = turnaround(figure("a"))
        model = heldout.turnaround_model(t["front"], t["left"], t["right"])
        seen = turned_mask(figure("a"), 45)
        checks, m = heldout.view_checks(heldout.VIEW_BY_NAME["front-left-34"], seen, model, 0.0, 0.2,
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
        seen = turned_mask(figure("a"), 45)
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
    else:
        unittest.main(argv=[sys.argv[0]] + [a for a in sys.argv[1:] if a != "--test"])
