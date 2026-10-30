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
from herogen.pipeline.client import ApiError, ReplayMiss, SamInstance  # noqa: E402

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


def _floor_shift(fig: synth.Figure) -> float:
    """How far `_drop_to_floor` moves ``fig`` down."""
    lows = []
    for p in fig.parts:
        if isinstance(p, synth.Capsule):
            lows.append(min(p.a[2], p.b[2]) - p.r)
        elif isinstance(p, synth.Ellipsoid):
            lows.append(p.c[2] - p.radii[2])
        else:
            lows.append(p.c[2] - p.half[2])
    return min(lows)


def _stick(h: int, pose: str) -> tuple[synth.Figure, dict[str, tuple[float, float, float]]]:
    """The stick figure before it is dropped onto the floor, and its body joints (Sapiens 2 names)."""
    f = synth.Figure()
    r = 0.07 * h
    f.parts.append(synth.Ellipsoid((0.0, 0.0, h - r), (r, r, r)))
    f.parts.append(synth.RBox((0.0, 0.0, h - 0.34 * h), (0.14 * h, 0.08 * h, 0.21 * h), 1.0))
    reach = 0.36 * h * math.sin(math.radians(45))
    j = {"nose": (0.0, -r, h - r), "neck": (0.0, 0.0, h - 2 * r),
         "left_eye": (0.4 * r, -0.9 * r, h - 0.7 * r), "right_eye": (-0.4 * r, -0.9 * r, h - 0.7 * r),
         "left_ear": (r, 0.0, h - r), "right_ear": (-r, 0.0, h - r)}
    for sx, side in ((1.0, "left"), (-1.0, "right")):
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
        foot = (ankle[0], ankle[1] - 0.035 * h, ankle[2] + 0.0175 * h)
        f.parts.append(synth.RBox(foot, (0.05 * h, 0.095 * h, 0.0175 * h), 1.0))
        j[f"{side}_shoulder"], j[f"{side}_wrist"] = top, end
        j[f"{side}_elbow"] = tuple((a + b) / 2 for a, b in zip(top, end, strict=True))
        j[f"{side}_hip"], j[f"{side}_knee"] = hip, mid
        j[f"{side}_ankle"] = (ankle[0], ankle[1], ankle[2] + 0.05 * h)
        j[f"{side}_big_toe"] = (foot[0] - 0.02 * h * sx, foot[1] - 0.08 * h, foot[2])
        j[f"{side}_small_toe"] = (foot[0] + 0.03 * h * sx, foot[1] - 0.07 * h, foot[2])
        j[f"{side}_heel"] = (foot[0], foot[1] + 0.085 * h, foot[2])
    return f, j


def stick_figure(h: int = 200, pose: str = "a") -> synth.Figure:
    """The stick-and-block figure of `test_herogen_pipeline_synth.figure_mask` as a 3D figure, in px: its
    front mask and profiles are this figure's front and side silhouettes, so a fake turnaround sheet drawn
    with `figure_mask` and a fake held-out view drawn with this agree. pose: "a", "down" (arms hanging),
    "stride" (legs swung 25 degrees forward and back)."""
    return _drop_to_floor(_stick(h, pose)[0])


def stick_joints(h: int = 200, pose: str = "a") -> dict[str, tuple[float, float, float]]:
    """`stick_figure`'s body joints, on the same floor."""
    f, j = _stick(h, pose)
    dz = _floor_shift(f)
    return {n: (x, y, z - dz) for n, (x, y, z) in j.items()}


def turned_joints(fig: synth.Figure, joints: dict, yaw_deg: float, size: int = SIZE, ppu: float = PPU
                  ) -> dict[str, tuple[float, float]]:
    """Image (col, row) of ``joints`` (pixel centres at integers) in `turned_mask`'s frame at ``yaw_deg``."""
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    prims = [_prim(p, c, s) for p in fig.parts]
    lo = np.min([synth._bounds(p)[0] for p in prims], axis=0)
    hi = np.max([synth._bounds(p)[1] for p in prims], axis=0)
    u0, v0 = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    out = {}
    for n, p in joints.items():
        u, v = _turn(p, c, s)
        out[n] = ((u - u0) * ppu + size / 2 - 0.5, (v0 - v) * ppu + size / 2 - 0.5)
    return out


def humanoid_joints(fig: synth.Figure) -> dict[str, tuple[float, float, float]]:
    """Body joints (Sapiens 2 names) of a `figure`, read from its parts so a stride's moved legs are followed:
    synth.humanoid's head, neck, then per side (left, right) upper arm, forearm, hand, thigh, shin, foot."""
    head, neck = fig.parts[0], fig.parts[1]
    (hx, hy, hz), (rx, ry, rz) = head.c, head.radii
    j = {"nose": (hx, hy - ry, hz), "neck": tuple(neck.a),
         "left_eye": (hx + 0.35 * rx, hy - 0.85 * ry, hz + 0.15 * rz),
         "right_eye": (hx - 0.35 * rx, hy - 0.85 * ry, hz + 0.15 * rz),
         "left_ear": (hx + rx, hy, hz), "right_ear": (hx - rx, hy, hz)}
    for side, first, sx in (("left", 5, 1.0), ("right", 11, -1.0)):
        upper, fore, _, thigh, shin, foot = fig.parts[first : first + 6]
        j[f"{side}_shoulder"], j[f"{side}_elbow"], j[f"{side}_wrist"] = upper.a, upper.b, fore.b
        j[f"{side}_hip"], j[f"{side}_knee"], j[f"{side}_ankle"] = thigh.a, thigh.b, shin.b
        (fx, fy, fz), (wx, wy, wz) = foot.c, foot.half
        j[f"{side}_big_toe"] = (fx - 0.5 * wx * sx, fy - wy, fz - wz)
        j[f"{side}_small_toe"] = (fx + 0.5 * wx * sx, fy - 0.8 * wy, fz - wz)
        j[f"{side}_heel"] = (fx, fy + wy, fz - wz)
    return {n: tuple(float(v) for v in p) for n, p in j.items()}


def keypoints_at(fig: synth.Figure, yaw_deg: float, score: float = 0.9) -> dict:
    """What `heldout.measure_keypoints` records for `turned_mask(fig, yaw_deg)`: the joints in that image's
    px, and its soles' row."""
    m = turned_mask(fig, yaw_deg)
    pts = turned_joints(fig, humanoid_joints(fig), yaw_deg)
    return {"joints": {n: [c, r, score] for n, (c, r) in pts.items()}, "bottom": float(imaging.bbox(m)[3])}


def body_of(fig: synth.Figure) -> "heldout.KP.BodyModel":
    """The turnaround joint model of ``fig`` (its front and profiles), as stage 2h builds it."""
    t = turnaround(fig)
    b = imaging.bbox(t["front"])
    return heldout.body_model({**{v: keypoints_at(fig, y) for v, y in (("front", 0), ("left", 90), ("right", -90))},
                               "front_height": float(b[3] - b[1])})


class FakePose:
    """A stand-in for `herogen.pipeline.sapiens.PoseClient` for images of the stick figure: the pose and yaw
    `fake_heldout_image` drew (else, for a turnaround view, those whose `turned_mask` best matches the figure in
    the box, height-normalised, among the turnaround's front and profiles), then the stick figure's joints
    seen there, mapped onto the box. ``fail``: raise `PoseError` instead (Sapiens down)."""

    POSES = ("a", "down", "stride")
    N = 48  # height of the normalised comparison grid, px
    _CANDS: list | None = None  # shared: [(pose, yaw, figure, box, normalised mask)], the masks stacked

    def __init__(self, fail: bool = False, mode: str = "live") -> None:
        from herogen.pipeline.sapiens import PoseLog  # noqa: PLC0415

        self.fail, self.mode, self.calls = fail, mode, []
        self.log = PoseLog()

    @classmethod
    def _norm(cls, m: np.ndarray) -> np.ndarray:
        b = imaging.bbox(m)
        crop = m[b[1] : b[3], b[0] : b[2]]
        k = cls.N / crop.shape[0]
        w = max(1, min(cls.N * 2, round(crop.shape[1] * k)))
        img = imaging.to_mask_image(crop).resize((w, cls.N), Image.BILINEAR)
        out = np.zeros((cls.N, cls.N * 2), dtype=bool)
        x0 = cls.N - w // 2
        drawn = imaging.from_mask_image(img)
        out[:, max(0, x0) : x0 + w] = drawn[:, max(0, -x0) : max(0, -x0) + min(w, 2 * cls.N)]
        return out

    @classmethod
    def _candidates(cls) -> tuple[list, np.ndarray]:
        if cls._CANDS is None:
            cands = []
            for pose in cls.POSES:
                fig = stick_figure(200, pose)
                for yaw in (0, 90, -90):  # the turnaround's front and profiles
                    m = turned_mask(fig, yaw, size=200, ppu=0.5)
                    cands.append((pose, float(yaw), fig, imaging.bbox(m)))
                    cands[-1] += (cls._norm(m),)
            cls._CANDS = [cands, np.stack([c[4] for c in cands])]
        return cls._CANDS[0], cls._CANDS[1]

    def keypoints(self, image: Image.Image, box: list[float]) -> dict[str, tuple[float, float, float]]:
        from herogen.pipeline.sapiens import PoseError  # noqa: PLC0415

        self.calls.append(list(box))
        self.log.calls.append({"model": "fake-pose", "cached": False, "cost_usd": 0.0})
        if self.fail:
            raise PoseError("fake Sapiens is down")
        mask = np.asarray(image.convert("L")) < 200  # the fakes paint the figure on white
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        sub = np.zeros_like(mask)
        sub[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        drawn = DRAWN.get(_pixels(image))
        if drawn is not None:  # a fake held-out view: its pose and turn are known exactly
            pose, yaw = drawn
            fig = stick_figure(200, pose)
            cb = imaging.bbox(turned_mask(fig, yaw, size=200, ppu=0.5))
        else:  # a turnaround view: the best-matching pose and yaw
            cands, stack = self._candidates()
            pose, yaw, fig, cb, _ = cands[int(np.argmax((stack == self._norm(sub)).sum(axis=(1, 2))))]
        self.found = (pose, yaw)
        k = (y1 - y0) / (cb[3] - cb[1])
        pts = turned_joints(fig, stick_joints(200, pose), yaw, size=200, ppu=0.5)
        return {n: (x0 + (c + 0.5 - cb[0]) * k - 0.5, y0 + (r + 0.5 - cb[1]) * k - 0.5, 0.9)
                for n, (c, r) in pts.items()}

    def close(self) -> None:
        pass


def use_fake_pose(test: unittest.TestCase, **kw) -> list[FakePose]:
    """Make every stage 2h in ``test`` read keypoints from a `FakePose` (no Modal); returns the list each new
    one is appended to."""
    made: list[FakePose] = []

    def factory(_cache_dir, mode: str = "replay") -> FakePose:
        made.append(FakePose(mode=mode, **kw))
        return made[-1]

    p = mock.patch.object(heldout, "PoseClient", factory)
    p.start()
    test.addCleanup(p.stop)
    return made


REQUEST = re.compile(r"(\d+) degrees from the front, the toes and face toward the (left|right) edge")


def fake_heldout_image(prompt: str, yaw: float | None = None, pose: str = "a", frame=None, gain: float = 1.0):
    """A fake Muse Image answer to a held-out prompt: the stick figure turned ``gain`` times the requested
    yaw the way the prompt asks (or ``yaw`` degrees; negative turns it the wrong way), at the size and place
    of the figure in ``frame`` (the last image sent with the prompt), on a canvas ``frame``'s size. The face
    and head are not drawn."""
    from test_herogen_pipeline_synth import paint, place  # noqa: PLC0415

    deg, side = REQUEST.search(prompt).groups()
    turn = gain * float(deg) if yaw is None else yaw
    signed = turn if side == "left" else -turn
    m = turned_mask(stick_figure(200, pose), signed, size=400, ppu=1.0)
    b = imaging.bbox(m)
    m = m[b[1] : b[3], b[0] : b[2]]
    fb = imaging.bbox(imaging.threshold_mask(frame))
    k = (fb[3] - fb[1]) / 200.0  # the reference figure's px per stick px (the stick stands 200 px)
    new = (max(1, round(m.shape[1] * k)), max(1, round(m.shape[0] * k)))
    m = imaging.from_mask_image(imaging.to_mask_image(m).resize(new, Image.BILINEAR))
    at = ((fb[0] + fb[2]) // 2 - m.shape[1] // 2, fb[3] - m.shape[0])
    img = paint(place(m, frame.size, at), colour=(30, 80, 160), bg=(255, 255, 255))
    DRAWN[_pixels(img)] = (pose, signed)
    return img


DRAWN: dict[str, tuple[str, float]] = {}  # what `fake_heldout_image` drew, by pixels: (pose, yaw)


def _pixels(img: Image.Image) -> str:
    import hashlib  # noqa: PLC0415

    a = np.asarray(img.convert("RGB"))
    return hashlib.sha256(repr(a.shape).encode() + a.tobytes()).hexdigest()


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
                                           profile_face, body=body_of(t_fig), keypoints=keypoints_at(fig, yaw))
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
                                 m["fit"][f"facing_{side}"]["rms"], _num(m["found_yaw_deg"]),
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
                    rows.append((view.name, fname, pname, yaw, _num(m["found_yaw_deg"]),
                                 _num(m["yaw_window_miss_deg"]), status))
    return rows


DRIFTS = (-8, -5, -3, 0, 3, 5, 8)  # arm angle of the held-out figure minus the turnaround's, degrees


def drift_table() -> list[tuple]:
    """The drift study of the silhouette fit: every figure, arms drifted by DRIFTS, drawn at and half a degree
    past both edges of every window and at the targets, facing left (the figures are mirror-symmetric): (target,
    drift, figure, yaw drawn, found yaw (body keypoints), found with the silhouette's plain fit, status)."""
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


def _num(v) -> float:
    return float("nan") if v is None else float(v)


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
        # Drawn across each window; exact keypoints are found to the fit's half-degree grid.
        drawn = {30: (25, 25.5, 26), 60: (57.5, 59, 61, 63, 65)}
        for fname, fkw in FIGURES.items():
            for view in heldout.VIEWS:
                for d in drawn[abs(int(view.yaw_deg))]:
                    y = math.copysign(d, view.yaw_deg)
                    status, m, checks = evaluate(view.name, "a", y, fkw)
                    self.assertEqual(status, "accepted", (fname, view.name, y, _failed(checks)))
                    self.assertLessEqual(abs(m["found_yaw_deg"] - y), 0.5, (fname, view.name, y))
                    self.assertEqual(m["facing_votes"], {"keypoints": 1, "head": 1})

    def test_pose_and_facing_hold_at_every_yaw(self):
        # The pose and facing checks alone, 20-70 degrees: what the thresholds were set from.
        for fname, fkw in FIGURES.items():
            for side, sign in (("left", 1), ("right", -1)):
                for yaw in YAWS:
                    _, _, checks = evaluate(f"front-{side}-30", "a", sign * yaw, fkw)
                    self.assertEqual(_pose_facing_status(checks), "accepted", (fname, side, yaw))

    def test_distorted_poses_never_accepted(self):
        # Arms hanging (82 degrees below horizontal) or raised nearly level (10), and strides.
        cases = {"arms down": ("down", {}), "arms raised": ("lowered", dict(arm_deg=80.0)),
                 **{p: POSES[p] for p in ("stride 8", "stride 12", "stride 25")}}
        for fname, fkw in FIGURES.items():
            for pname, (pose, pkw) in cases.items():
                for view in heldout.VIEWS:
                    y = math.copysign(view.aim_deg, view.yaw_deg)
                    status, m, checks = evaluate(view.name, pose, y, fkw, pkw)
                    self.assertEqual(status, "rejected", (fname, pname, view.name))
                    if pname.startswith("arms"):
                        self.assertIn("arm_angle_near_off_band_deg", _failed(checks), (fname, pname, view.name))

    def test_arms_a_little_off_the_a_pose_pass(self):
        # The turnaround's arms are 40 degrees below horizontal; 55 (arms 35 from vertical) is still an A-pose.
        for fname, fkw in FIGURES.items():
            for view in heldout.VIEWS:
                y = math.copysign(view.aim_deg, view.yaw_deg)
                status, m, checks = evaluate(view.name, "lowered", y, fkw, dict(arm_deg=35.0))
                self.assertEqual(status, "accepted", (fname, view.name, _failed(checks)))
                self.assertAlmostEqual(m["arm_angles_deg"]["near"], 55.0, delta=0.5)

    def test_small_stride_never_accepted(self):
        # An 8-degree stride: the legs no longer match the turnaround's, so the keypoint fit can't place the
        # turn either (it runs to the profile).
        status, m, checks = evaluate("front-left-60", "stride", 61, FIGURES["plain"], dict(stride_deg=8.0))
        self.assertEqual(status, "rejected")
        self.assertIn("found_yaw_window_miss_deg", _failed(checks))
        self.assertGreater(m["keypoint_fit"]["facing_left"]["rms"], 0.03)

    def test_wrong_facing_rejected(self):
        for fname, fkw in FIGURES.items():
            for view in heldout.VIEWS:
                status, m, checks = evaluate(view.name, "a", -view.yaw_deg, fkw)
                self.assertEqual(status, "rejected", (fname, view.name))
                self.assertIn("facing_cues_against", _failed(checks))
                self.assertEqual(m["facing_votes"], {"keypoints": -1, "head": -1})

    def test_yaw_just_outside_the_window_needs_review(self):
        for name, yaw in (("front-left-30", 22), ("front-right-30", -22), ("front-left-60", 67),
                          ("front-right-60", -67)):
            status, m, checks = evaluate(name, "a", yaw, FIGURES["plain"])
            self.assertEqual(status, "needs review", (name, yaw))
            self.assertEqual(_failed(checks), ["found_yaw_in_window"])
            self.assertTrue(0 < m["yaw_window_miss_deg"] <= heldout.YAW_WINDOW_REVIEW_DEG)

    def test_yaw_far_from_the_window_rejected(self):
        for name, yaw in (("front-left-30", 18), ("front-left-60", 75), ("front-right-60", -30)):
            status, m, checks = evaluate(name, "a", yaw, FIGURES["plain"])
            self.assertEqual(status, "rejected", (name, yaw))
            self.assertIn("found_yaw_window_miss_deg", _failed(checks))
        self.assertAlmostEqual(heldout.window_miss(heldout.VIEW_BY_NAME["front-right-60"], -52.0), 5.0)
        self.assertAlmostEqual(heldout.window_miss(heldout.VIEW_BY_NAME["front-left-30"], 34.0), 7.7)
        self.assertEqual(heldout.window_miss(heldout.VIEW_BY_NAME["front-right-60"], -60.0), 0.0)

    def test_windows_keep_every_allowed_true_yaw_clear(self):
        # The windows follow from the measured fit error: the 30-degree one ends below its target.
        self.assertEqual(heldout.FIT_ERROR_DEG, (-8.7, 2.0))
        self.assertEqual([tuple(round(e, 6) for e in v.window_deg) for v in heldout.VIEWS],
                         [(25.0, 26.3)] * 2 + [(57.0, 65.0)] * 2)
        self.assertEqual([round(v.aim_deg, 6) for v in heldout.VIEWS], [25.65, 25.65, 61.0, 61.0])
        for view in heldout.VIEWS:
            for edge in view.window_deg:
                rng = heldout.yaw_range(math.copysign(edge, view.yaw_deg))
                self.assertGreaterEqual(heldout.range_clearance(*rng["true_deg"]) + 1e-9,
                                        heldout.AZIMUTH_CLEARANCE_DEG, (view.name, edge))

    def test_true_yaw_range_from_the_measured_error(self):
        lo_err, hi_err = heldout.FIT_ERROR_DEG
        for found in (26.0, -26.0, 61.0, -61.0):
            rng = heldout.yaw_range(found)
            lo, hi = sorted(abs(v) for v in rng["true_deg"])
            self.assertAlmostEqual(lo, abs(found) - hi_err)
            self.assertAlmostEqual(hi, abs(found) - lo_err)
            self.assertEqual(rng["true_deg"][0] > 0, found > 0)
        _, m, _ = evaluate("front-left-30", "a", 26, FIGURES["plain"])
        self.assertEqual(m["true_yaw_range_deg"], heldout.yaw_range(m["found_yaw_deg"])["true_deg"])

    def test_building_azimuths_rejected(self):
        # Found at or near a building azimuth, or where the measured error lets the true yaw come within
        # AZIMUTH_CLEARANCE_DEG of one: the target 30 itself (its true yaw can be 38.7), and 55-56.5 at 60.
        for name, yaw in (("front-left-30", 0), ("front-left-30", 5), ("front-left-30", 30),
                          ("front-left-30", 42), ("front-right-60", -45), ("front-right-60", -50),
                          ("front-left-60", 55), ("front-left-60", 56.5), ("front-left-60", 85)):
            status, m, checks = evaluate(name, "a", yaw, FIGURES["clothed"])
            self.assertEqual(status, "rejected", (name, yaw))
            self.assertIn("found_yaw_clear_of_build_azimuths_deg", _failed(checks))
        self.assertEqual(heldout.azimuth_clearance(-30.0), 15.0)
        self.assertEqual(heldout.azimuth_clearance(-35.0), 10.0)
        self.assertEqual(heldout.azimuth_clearance(-81.0), 9.0)
        self.assertEqual(heldout.azimuth_clearance(178.0), 2.0)
        self.assertEqual(heldout.range_clearance(30.0, 40.0), 5.0)
        self.assertEqual(heldout.range_clearance(-50.0, -40.0), 0.0)
        self.assertEqual(heldout.range_clearance(-34.0, -20.0), 11.0)

    def test_without_keypoints_the_turn_needs_review(self):
        t_fig = figure("a")
        t = turnaround(t_fig)
        model = heldout.turnaround_model(t["front"], t["left"], t["right"])
        seen = turned_mask(t_fig, 26)
        for body, kps, why in ((None, keypoints_at(t_fig, 26), "no turnaround keypoints"),
                               (body_of(t_fig), {"joints": None, "error": "PoseError: down", "api_error": True},
                                "PoseError: down"),
                               (body_of(t_fig), {"joints": {"nose": [1.0, 1.0, 0.9]}, "bottom": 10.0},
                                "too few body keypoints found")):
            checks, m = heldout.view_checks(heldout.VIEW_BY_NAME["front-left-30"], seen, model, 0.3, 0.2,
                                            {"offset": None}, None, body=body, keypoints=kps)
            self.assertEqual(heldout.view_status(checks), "needs review")
            # Without keypoints neither the turn, the arms nor (with no face cue either) the facing is decided.
            self.assertEqual(_failed(checks), ["turn_measured", "arm_angle_measured", "facing_cues_for"])
            self.assertIn(why, next(c for c in checks if c["name"] == "turn_measured")["note"])
            self.assertIsNone(m["found_yaw_deg"])
        self.assertEqual(heldout.next_request(heldout.VIEW_BY_NAME["front-left-30"], 25.65, None), 25.65)

    def test_arm_drift_does_not_move_the_found_yaw(self):
        # Arms held 5 or 8 degrees higher or lower than the turnaround's: the arms are not in the turn fit.
        for fname in ("plain", "clothed+armour"):
            for view in heldout.VIEWS:
                for drift in (-8, -5, 5, 8):
                    y = math.copysign(view.aim_deg, view.yaw_deg)
                    _, m, _ = evaluate(view.name, "lowered", y, FIGURES[fname], dict(arm_deg=50.0 + drift))
                    self.assertLessEqual(abs(m["found_yaw_deg"] - y), 0.5, (fname, view.name, drift))

    def test_arm_drift_near_a_diagonal_never_accepted(self):
        for fname in ("plain", "clothed"):
            for name, ys in (("front-left-30", (35.5, 36, 39)), ("front-right-30", (-36, -39)),
                             ("front-left-60", (51, 54.5)), ("front-right-60", (-51, -54))):
                for drift in (-8, -5, 0, 5, 8):
                    for y in ys:
                        status, m, checks = evaluate(name, "lowered", y, FIGURES[fname],
                                                     dict(arm_deg=50.0 + drift))
                        self.assertNotEqual(status, "accepted", (fname, name, y, drift, m["found_yaw_deg"]))

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
        # No face cue, and a keypoint margin below the vote limit: nothing decides the facing.
        t = turnaround(figure("a"))
        model = heldout.turnaround_model(t["front"], t["left"], t["right"])
        seen = turned_mask(figure("a"), 26)
        with mock.patch.object(heldout, "KP_FACING_MARGIN", 1.0):
            checks, m = heldout.view_checks(heldout.VIEW_BY_NAME["front-left-30"], seen, model, 0.0, 0.2,
                                            {"offset": None}, None, body=body_of(figure("a")),
                                            keypoints=keypoints_at(figure("a"), 26))
        self.assertEqual(heldout.view_status(checks), "needs review")
        self.assertEqual(_failed(checks), ["facing_cues_for"])
        self.assertEqual(m["facing_votes"], {"keypoints": 0, "head": 0})

    def test_unmeasurable_cues_do_not_vote(self):
        t = turnaround(figure("a"))
        model = heldout.turnaround_model(t["front"], t["left"], t["right"])
        seen = turned_mask(figure("a"), 26)
        checks, m = heldout.view_checks(heldout.VIEW_BY_NAME["front-left-30"], seen, model, 0.0, 0.2,
                                        {"offset": None, "face_found": False}, None, body=body_of(figure("a")),
                                        keypoints=keypoints_at(figure("a"), 26))
        self.assertEqual(m["facing_votes"], {"keypoints": 1, "head": 0})
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


    def test_central_run_of_a_parted_row(self):
        row = np.zeros(100, dtype=bool)
        for a, b in ((5, 10), (30, 45), (55, 70), (90, 95)):  # hand, body, body, hand
            row[a:b] = True
        self.assertEqual(heldout.central_run(row, 50.0), (30, 70))
        self.assertEqual(heldout.central_run(row, 40.0), (30, 45))  # the run holding the centre
        one_side = row.copy()
        one_side[50:] = False
        self.assertEqual(heldout.central_run(one_side, 50.0), (30, 45))
        self.assertEqual(heldout.central_run(np.zeros(100, dtype=bool), 50.0), (49.0, 51.0))


def _capsule(mask: np.ndarray, p: tuple[float, float], q: tuple[float, float], r: float) -> np.ndarray:
    """``mask`` with a 2D capsule (x, y px) from ``p`` to ``q`` of radius ``r`` drawn in."""
    yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    p_, d = np.array(p, float), np.array(q, float) - np.array(p, float)
    t = np.clip(((xx - p_[0]) * d[0] + (yy - p_[1]) * d[1]) / (d @ d), 0, 1)
    return mask | ((xx - p_[0] - t * d[0]) ** 2 + (yy - p_[1] - t * d[1]) ** 2 <= r * r)


class GuideLegsTest(unittest.TestCase):
    """The turn guide's legs stand on the feet SAM finds: tails, staffs, swords, hands, capes and shields are
    not legs. Without two feet, the outline rule (main's: the outermost runs).

    The test humanoid (legs a little apart) stands 440 px tall from row 36: the hip row (0.52) is row 264,
    the knee row (0.71) row 348, the sole row (0.98) row 467; its legs' knee runs are x 207-243 and 269-305,
    its centre x 256."""

    ROWS = heldout.GuideLegs.ROWS

    def _front(self, parted: bool = True, arm: float = 50.0, **kw) -> np.ndarray:
        fig = figure("lowered", clothed=False, arm_deg=arm, leg_spread=kw.get("leg_spread", 0.1))
        if parted:  # a short pelvis: the legs part above the hip row
            fig.parts[4] = synth.RBox((0.0, 0.0, 2.95), (0.56, 0.30, 0.15), 0.12)
        return turnaround(fig)["front"]

    def _feet(self, plain: np.ndarray) -> list[np.ndarray]:
        """What a foot detector finds: each sole-row run's column span over the lowest 8% of the figure."""
        b = imaging.bbox(plain)
        out = []
        for a, c in self._runs(plain, "sole"):
            f = np.zeros_like(plain)
            f[b[3] - int(0.08 * (b[3] - b[1])) : b[3], a:c] = True
            out.append(f & plain)
        return out

    def _runs(self, front: np.ndarray, row: str) -> list[tuple[int, int]]:
        return heldout._runs(front[heldout._row_at(front, self.ROWS[row])])

    def _legs(self, front: np.ndarray, feet: list[np.ndarray]) -> heldout.GuideLegs:
        return heldout.guide_legs(front, feet, "foot")

    def _same_legs(self, plain: np.ndarray, front: np.ndarray) -> heldout.GuideLegs:
        """The legs of ``front`` (``plain`` with something added that touches no leg) on the feet of
        ``plain`` equal ``plain``'s: its knee and sole runs and, for the hips, the legs' span at the hip
        row."""
        feet = self._feet(plain)
        want, got = self._legs(plain, feet), self._legs(front, feet)
        self.assertEqual(got.method, "sam-3.1:foot")
        self.assertEqual((got.hips, got.knees, got.soles), (want.hips, want.knees, want.soles))
        self.assertEqual(want.knees, tuple(self._runs(plain, "knee")))
        self.assertEqual(want.soles, tuple(self._runs(plain, "sole")))
        return got

    def _leg_lines(self, front: np.ndarray, legs: heldout.GuideLegs | None):
        t = {"left": np.fliplr(front), "right": front}  # any profile: only the front's x is checked at yaw 0
        shapes, h = heldout._guide_shapes(front, t["left"], t["right"], 0.0, legs)
        lw = 0.035 * h
        return [pts for kind, pts, w in shapes if kind == "line" and w == 1.3 * lw], shapes

    def test_parted_hip_row_with_hands_in_it(self):
        # RC7: the legs part above the hip row, so no run there holds the centre; with the arms at 40 degrees
        # the hands reach into that row (the first run is a hand). Feet or no feet, the hips span the legs,
        # each leg is straight, and the torso's lowest edge is the hips.
        for arm in (40.0, 50.0):
            front = self._front(arm=arm)
            fb = imaging.bbox(front)
            cx = (fb[0] + fb[2]) / 2
            hip_runs = self._runs(front, "hip")
            self.assertFalse([r for r in hip_runs if r[0] <= cx <= r[1]], arm)
            if arm == 40.0:
                self.assertEqual(len(hip_runs), 4)  # hand, leg, leg, hand
            legs_at_hips = [r for r in hip_runs if abs((r[0] + r[1]) / 2 - cx) < 60]
            for legs in (self._legs(front, self._feet(front)), heldout.guide_legs(front, [], None)):
                self.assertEqual(legs.hips, (legs_at_hips[0][0], legs_at_hips[1][1]), (arm, legs.method))
                lines, shapes = self._leg_lines(front, legs)
                for pts, (ka, kb), sign in zip(lines, self._runs(front, "knee"), (-1, 1), strict=True):
                    (hip_u, _), (knee_u, _) = pts[0], pts[1]
                    self.assertGreater(sign * hip_u, 0, arm)
                    self.assertAlmostEqual(knee_u, (ka + kb) / 2 - cx)
                    # Neither crossed nor bent: the hip is above its knee, give or take a tenth of the height.
                    self.assertLess(abs(hip_u - knee_u), 0.1 * (fb[3] - fb[1]), arm)
                torso = next(pts for kind, pts, _ in shapes if kind == "polygon")
                at_hips = sorted(u for u, v in torso if v == self.ROWS["hip"])
                self.assertEqual(at_hips, [legs.hips[0] - cx, legs.hips[1] - cx])

    def test_pelvis_covering_the_hip_row(self):
        front = self._front(parted=False)
        fb = imaging.bbox(front)
        cx = (fb[0] + fb[2]) / 2
        hit = [r for r in self._runs(front, "hip") if r[0] <= cx <= r[1]]
        for legs in (self._legs(front, self._feet(front)), heldout.guide_legs(front, [], None)):
            self.assertEqual(legs.hips, hit[0], legs.method)

    def test_props_beside_the_legs(self):
        for parted in (False, True):
            plain = self._front(parted=parted)
            cases = {
                "vertical staff": _capsule(plain, (92.0, 60.0), (92.0, 468.0), 6.0),
                "sword to the knee row": _capsule(plain, (100.0, 230.0), (115.0, 380.0), 5.0),
                "hand at knee height": _capsule(plain, (178.0, 340.0), (178.0, 356.0), 14.0),
                "tail to the knee row": _capsule(plain, (300.0, 290.0), (345.0, 380.0), 10.0),
                "shield beside a leg": plain | _box(plain, 120, 250, 195, 420),
            }
            for name, front in cases.items():
                self.assertGreater(len(self._runs(front, "knee")), 2, name)
                with self.subTest(name, parted=parted):
                    self._same_legs(plain, front)

    def test_props_between_the_legs(self):
        # Between the legs, nearer the centre than either: a tail to the floor or to the knee row, a staff
        # planted between the feet, a prop off the centre.
        plain = self._front()
        cases = {
            "tail to the floor": _capsule(plain, (256.0, 275.0), (256.0, 468.0), 6.0),
            "tail to the knee row": _capsule(plain, (256.0, 275.0), (256.0, 370.0), 6.0),
            "staff between the feet": _capsule(plain, (256.0, 120.0), (258.0, 470.0), 5.0),
            "prop off the centre": _capsule(plain, (262.0, 275.0), (262.0, 470.0), 4.0),
        }
        for name, front in cases.items():
            self.assertEqual(len(self._runs(front, "knee")), 3, name)
            with self.subTest(name):
                self._same_legs(plain, front)

    def test_thin_legs_with_a_tail_between(self):
        fig = figure("a", clothed=False, leg_spread=0.25)
        for i in (8, 9, 14, 15):  # thighs and shins a third as thick
            fig.parts[i] = synth.Capsule(fig.parts[i].a, fig.parts[i].b, fig.parts[i].r / 3)
        plain = turnaround(fig)["front"]
        knees = self._runs(plain, "knee")
        self.assertEqual(len(knees), 2)
        front = _capsule(plain, (256.0, 300.0), (256.0, 464.0), 10.0)  # thicker than either leg
        self.assertEqual(len(self._runs(front, "knee")), 3)
        self._same_legs(plain, front)

    def test_one_sided_cape_backpack_or_armour(self):
        # Something that widens the torso on one side only (a cape, a backpack, a pauldron) moves the
        # figure's centre and the torso's middle, but not the feet.
        plain = self._front()
        for name, front in {"cape to 0.47": plain | _box(plain, 296, 90, 420, 243),
                            "torso wider image-right": plain | _box(plain, 296, 110, 360, 250)}.items():
            with self.subTest(name):
                self._same_legs(plain, front)

    def test_shield_covering_the_centre(self):
        # The legs are hidden behind a shield from the hips to below the knees: the soles still stand on
        # the feet, the knees split the shield's run and the hips are its run.
        plain = self._front()
        front = plain | _box(plain, 180, 240, 332, 400)
        legs = self._legs(front, self._feet(plain))
        self.assertEqual(legs.soles, tuple(self._runs(plain, "sole")))
        (shield,) = self._runs(front, "knee")
        self.assertEqual(legs.knees, ((shield[0], sum(shield) / 2), (sum(shield) / 2, shield[1])))
        self.assertEqual(legs.hips, self._runs(front, "hip")[0] if len(self._runs(front, "hip")) == 1
                         else next(r for r in self._runs(front, "hip") if r[0] <= 256 < r[1]))

    def test_wider_shield_touching_a_foot_at_the_sole_row(self):
        plain = np.zeros((480, 400), dtype=bool)
        plain[40:100, 185:215] = True
        plain[100:240, 145:255] = True
        plain[240:440, 160:180] = True
        plain[240:440, 220:240] = True
        front = plain | _box(plain, 243, 250, 303, 440)  # shield three times wider than a leg
        front[425:440, 240:243] = True  # shield touches the foot in the bottom 15 rows
        for mirror in (False, True):
            with self.subTest(mirror=mirror):
                body, shield = (np.fliplr(m) if mirror else m for m in (plain, front))
                feet = self._feet(body)
                self.assertEqual(len(self._runs(shield, "sole")), 2)  # one foot has merged with the shield
                self.assertEqual(len(self._runs(shield, "knee")), 3)
                legs = self._legs(shield, feet)
                self.assertEqual(legs.knees, tuple(self._runs(body, "knee")))
                self.assertEqual(legs.soles, tuple(self._runs(body, "sole")))
                self.assertEqual(legs.hips, (160, 240))
                lines, _ = self._leg_lines(shield, legs)
                fb = imaging.bbox(shield)
                cx = (fb[0] + fb[2]) / 2
                for line, (a, b) in zip(lines, legs.soles, strict=True):
                    self.assertEqual(line[1][0] + cx, (a + b) / 2)  # knee
                    self.assertEqual(line[2][0] + cx, (a + b) / 2)  # ankle

    def test_arm_thicker_than_the_thigh_it_touches(self):
        # An arm 40 px thick (the thigh is 36) that touches the image-left thigh between the hip and knee
        # rows and is apart from it at the hip row: the hips stay on the legs.
        plain = self._front()
        front = _capsule(plain, (170.0, 150.0), (200.0, 320.0), 20.0)
        self.assertEqual(len(self._runs(front, "hip")), 3)  # arm, thigh, thigh
        self._same_legs(plain, front)

    def test_fragment_at_the_sole_row(self):
        plain = self._front(parted=False)
        front = plain.copy()
        front[455:476, 255:257] = True  # a 2 px sliver between the feet
        self._same_legs(plain, front)

    def test_feet_together(self):
        # Legs pressed together (one run at the knee row) and feet touching (one run at the sole row): the
        # soles stay within the detected feet, the knee run is split at its middle.
        plain = turnaround(figure("a", clothed=False, leg_spread=-0.3))["front"]
        feet = self._feet(plain)
        (fl, fr) = self._runs(plain, "sole")
        b = imaging.bbox(plain)
        plain[b[3] - 20 : b[3], fl[1] : fr[0]] = True  # the feet touch
        self.assertEqual(len(self._runs(plain, "sole")), 1)
        legs = self._legs(plain, feet)
        self.assertEqual(legs.soles, (fl, fr))
        (knee,) = self._runs(plain, "knee")
        self.assertEqual(legs.knees, ((knee[0], sum(knee) / 2), (sum(knee) / 2, knee[1])))
        front = _capsule(plain, (150.0, 60.0), (150.0, 468.0), 6.0)  # and a staff beside them
        got = self._legs(front, feet)
        self.assertEqual((got.knees, got.soles), (legs.knees, legs.soles))

    def test_without_feet_the_outline_rule(self):
        # No two feet (a long robe; SAM found one or none): main's rule, the outermost runs, which a staff
        # beside the legs fools. The record says which rule drew the guide.
        plain = self._front(parted=False)
        front = _capsule(plain, (92.0, 60.0), (92.0, 468.0), 6.0)
        for feet in ([], self._feet(plain)[:1]):
            legs = heldout.guide_legs(front, feet, "foot" if feet else None)
            self.assertEqual(legs.method, "outline")
            self.assertEqual(legs.knees[0], self._runs(front, "knee")[0])  # the staff
            self.assertEqual(legs.hips, heldout.legs_from_outline(plain).hips)
        # A "foot" with no pixel on the sole row can't anchor a leg.
        floating = self._feet(plain)
        floating[0] = np.roll(floating[0], -60, axis=0)
        self.assertEqual(heldout.guide_legs(plain, floating, "foot").method, "outline")


def _box(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = True
    return out


class FindFeetTest(unittest.TestCase):
    """`find_feet`: which SAM instances count as feet, and the prompts tried in turn."""

    class _Client:
        def __init__(self, answers):
            self.answers, self.asked = answers, []

        def segment(self, _img, text):
            self.asked.append(text)
            return self.answers.get(text, [])

    def _inst(self, m, conf=0.9):
        return SamInstance(mask=m, box=imaging.bbox(m), confidence=conf)

    def setUp(self):
        self.front = turnaround(figure("a", clothed=False, leg_spread=0.1))["front"]
        b = imaging.bbox(self.front)
        self.feet = []
        for a, c in heldout._runs(self.front[heldout._row_at(self.front, 0.98)]):
            f = np.zeros_like(self.front)
            f[b[3] - 30 : b[3], a:c] = True
            self.feet.append(f & self.front)

    def test_first_prompt_with_two_feet(self):
        left, right = self.feet
        off_figure = np.zeros_like(self.front)
        off_figure[5:30, 5:30] = True
        high = np.roll(left, -120, axis=0) & self.front  # on the figure, but not on the sole row
        shoes = [self._inst(right, 0.7), self._inst(right, 0.8), self._inst(left, 0.6)]
        client = self._Client({"foot": [self._inst(right), self._inst(off_figure), self._inst(high)],
                               "shoe": shoes})
        feet, found = heldout.find_feet(client, None, self.front)
        self.assertEqual(client.asked, ["foot", "shoe"])  # "foot" found one foot; "boot" not needed
        self.assertEqual(found["prompt"], "shoe")
        self.assertEqual(len(found["found"]["shoe"]), 2)  # the duplicate right foot kept once
        self.assertTrue(np.array_equal(feet[0], left) and np.array_equal(feet[1], right))  # image-left first

    def test_no_feet(self):
        client = self._Client({})
        self.assertEqual(heldout.find_feet(client, None, self.front)[0], [])
        self.assertEqual(client.asked, ["foot", "shoe", "boot"])

    def test_guide_fallback_records_missing_or_unusable_feet(self):
        for feet in ([], [np.zeros_like(self.front), self.feet[1]]):
            with self.subTest(count=len(feet)), mock.patch.object(
                    heldout, "find_feet", return_value=(feet, {"prompt": None, "found": {}})):
                legs, found = heldout.find_guide_legs(None, None, self.front)
                self.assertEqual(legs, heldout.legs_from_outline(self.front))
                self.assertEqual(found["fallback_reason"], "two usable feet were not found on the sole row")

    def test_model_api_failure_stops(self):
        for error in (ApiError("HTTP 500"), ReplayMiss("no cached foot")):
            with self.subTest(error=type(error).__name__):
                client = self._Client({})
                with mock.patch.object(client, "segment", side_effect=error) as segment, \
                        self.assertRaises(type(error)):
                    heldout.find_guide_legs(client, None, self.front)
                self.assertEqual(segment.call_count, 1)


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
        print(f"found minus drawn: {min(e):+.1f} to {max(e):+.1f} degrees on exact synthetic keypoints "
              f"(FIT_ERROR_DEG {heldout.FIT_ERROR_DEG} is measured on renders)")
    else:
        unittest.main(argv=[sys.argv[0]] + [a for a in sys.argv[1:] if a != "--test"])
