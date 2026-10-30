"""Body keypoints and the measurements stage 2h judges a held-out view by: its turn, its pose and its arms.

The keypoints are Sapiens 2's (Meta; `herogen/pipeline/sapiens.py` runs it): 308 named points per image, of
which only the body joints below are used. Everything here is plain geometry on those points, in the turnaround's
pixels, so it is tested on synthetic keypoints.

Coordinates. A view's keypoint (x, y) in image px becomes (u, z) in turnaround px: u = x * scale across, z =
(bottom - y) * scale up from the soles, `bottom` the figure mask's lowest row edge and `scale` the view's pixels
per turnaround pixel (1 for the turnaround's own views; for a generated view the framing scale stage 2h already
uses, never the figure's own height). Heights and errors are reported as fractions of the front figure's height.

3D model (`body_model`): the camera of docs/ARCHITECTURE.md turned by a yaw sees a point (x, y, z) at u = x cos +
y sin. Each joint's x and z come from the front view; its depth y from the two profiles (the left view's u is +y,
the right view's u is -y). In a profile the network often swaps left and right or puts the hidden joint on top of
the visible one, so a left/right pair gets one depth, the mean of the pair in each profile: a standing A-pose is
left-right symmetric in depth. Each profile's depths are centred on the joints both profiles found, and the two
are averaged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

Keypoints = dict[str, tuple[float, float, float]]  # name -> (x px, y px, score)

SIDES = ("left", "right")
# The joints whose positions turn rigidly with the body in an A-pose: the head, the torso and the legs. The arms
# are left out of the turn: a generated view often holds them a few degrees off the turnaround's (see `fit_yaw`).
HEAD = ("nose", "neck", "left_eye", "right_eye", "left_ear", "right_ear")
TORSO_LEGS = tuple(f"{s}_{j}" for j in ("shoulder", "hip", "knee", "ankle", "big_toe", "small_toe", "heel")
                   for s in SIDES)
TURN_JOINTS = HEAD + TORSO_LEGS
ARM_JOINTS = tuple(f"{s}_{j}" for j in ("elbow", "wrist") for s in SIDES)
POSE_JOINTS = TURN_JOINTS + ARM_JOINTS
CENTRE_JOINTS = tuple(f"{s}_{j}" for j in ("shoulder", "hip") for s in SIDES)


def pair_name(name: str) -> str | None:
    """The same joint on the other side ("left_knee" -> "right_knee"), or None for a midline joint."""
    for a, b in (("left_", "right_"), ("right_", "left_")):
        if name.startswith(a):
            return b + name[len(a):]
    return None


def to_uz(kps: Keypoints, bottom: float, scale: float = 1.0, min_score: float = 0.0
          ) -> dict[str, tuple[float, float, float]]:
    """(u, z, score) per joint in turnaround px: u across, z up from the soles; joints scoring under
    ``min_score`` are dropped."""
    return {n: (x * scale, (bottom - y) * scale, s) for n, (x, y, s) in kps.items() if s >= min_score}


@dataclass
class BodyModel:
    """Joints in 3D (turnaround px): x across and z up from the front view, y depth (+ toward the back) from the
    profiles. ``height``: the front figure's height."""

    x: dict[str, float]
    z: dict[str, float]
    y: dict[str, float]
    w: dict[str, float]  # the front view's score per joint
    height: float
    arm_length: dict[str, float] = field(default_factory=dict)  # shoulder to wrist in the front view, per side
    arm_drop_deg: dict[str, float] = field(default_factory=dict)  # the front view's arm angle below horizontal

    def predict(self, name: str, yaw_deg: float) -> float:
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        return self.x[name] * c + self.y[name] * s


def _profile_depth(uz: dict[str, tuple[float, float, float]], sign: float) -> dict[str, float]:
    """One profile's depth per joint (pairs merged to their mean), centred on CENTRE_JOINTS when found."""
    d: dict[str, float] = {}
    for n, (u, _, _) in uz.items():
        p = pair_name(n)
        if p and p in uz:
            d[n] = sign * (u + uz[p][0]) / 2
        else:
            d[n] = sign * u
    return d


def body_model(front: Keypoints, left: Keypoints | None, right: Keypoints | None, *, front_bottom: float,
               front_height: float, left_bottom: float | None = None, right_bottom: float | None = None,
               min_score: float = 0.3) -> BodyModel | None:
    """The 3D joint model of a turnaround from its front view and profiles (all at the turnaround's scale). None
    when the front view has fewer than four turn joints or no profile gives any depth."""
    f = to_uz(front, front_bottom, 1.0, min_score)
    if sum(n in f for n in TURN_JOINTS) < 4:
        return None
    depths = []
    for kps, bottom, sign in ((left, left_bottom, 1.0), (right, right_bottom, -1.0)):
        if kps is None:
            continue
        uz = to_uz(kps, bottom if bottom is not None else front_bottom, 1.0, min_score)
        if uz:
            depths.append(_profile_depth(uz, sign))
    if not depths:
        return None
    common = [n for n in CENTRE_JOINTS if all(n in d for d in depths)]
    if not common:
        common = sorted(set.intersection(*(set(d) for d in depths)))
    if not common:
        depths = depths[:1]
        common = list(depths[0])
    centred = [{n: v - float(np.mean([d[c] for c in common])) for n, v in d.items()} for d in depths]
    y = {}
    for n in f:
        vals = [d[n] for d in centred if n in d]
        if vals:
            y[n] = float(np.mean(vals))
    names = [n for n in f if n in y]
    m = BodyModel(x={n: f[n][0] for n in names}, z={n: f[n][1] for n in names}, y={n: y[n] for n in names},
                  w={n: f[n][2] for n in names}, height=float(front_height))
    for s in SIDES:
        sh, wr = f.get(f"{s}_shoulder"), f.get(f"{s}_wrist")
        if sh and wr:
            m.arm_length[s] = math.hypot(wr[0] - sh[0], wr[1] - sh[1])
            m.arm_drop_deg[s] = math.degrees(math.atan2(sh[1] - wr[1], abs(wr[0] - sh[0])))
    return m


YAW_GRID = np.arange(-89.0, 89.01, 0.5)


def _fit_at(model: BodyModel, seen: dict[str, tuple[float, float, float]], names: list[str], yaw: float
            ) -> np.ndarray:
    """Horizontal residuals (seen - predicted, after the best weighted shift) of ``names`` at ``yaw``, in
    turnaround px."""
    pred = np.array([model.predict(n, yaw) for n in names])
    obs = np.array([seen[n][0] for n in names])
    w = np.array([min(seen[n][2], model.w[n]) for n in names])
    r = obs - pred
    return r - float(np.sum(w * r) / np.sum(w))


def fit_yaw(model: BodyModel, seen: dict[str, tuple[float, float, float]], joints: tuple[str, ...] = TURN_JOINTS
            ) -> dict | None:
    """The yaw at which the model's turn joints best match a view's (weighted least squares over YAW_GRID, with a
    free sideways shift), separately for a figure facing image left (yaw > 0) and image right (yaw < 0).

    Returns {"yaw_deg", "rms"} for the best overall, and per side under "facing_left" / "facing_right", the rms
    as a fraction of the front figure's height, with "joints" (how many were used); None with fewer than four
    joints seen in both."""
    names = [n for n in joints if n in seen and n in model.x]
    if len(names) < 4:
        return None
    w = np.array([min(seen[n][2], model.w[n]) for n in names])
    rms = []
    for yaw in YAW_GRID:
        r = _fit_at(model, seen, names, float(yaw))
        rms.append(math.sqrt(float(np.sum(w * r * r) / np.sum(w))) / model.height)
    rms = np.array(rms)
    out: dict = {"joints": len(names)}
    for side, sel in (("left", YAW_GRID > 0), ("right", YAW_GRID < 0)):
        i = int(np.argmin(np.where(sel, rms, np.inf)))
        out[f"facing_{side}"] = {"yaw_deg": float(YAW_GRID[i]), "rms": float(rms[i])}
    i = int(np.argmin(rms))
    out.update({"yaw_deg": float(YAW_GRID[i]), "rms": float(rms[i])})
    return out


def pose_residuals(model: BodyModel, seen: dict[str, tuple[float, float, float]], yaw_deg: float,
                   joints: tuple[str, ...] = POSE_JOINTS) -> dict[str, tuple[float, float]]:
    """Each joint's (across, up) error at ``yaw_deg`` as fractions of the front figure's height: across after the
    best sideways shift of the turn joints, up against the front view's height above the soles."""
    names = [n for n in joints if n in seen and n in model.x]
    base = [n for n in TURN_JOINTS if n in names]
    if not base:
        return {}
    w = np.array([min(seen[n][2], model.w[n]) for n in base])
    r = np.array([seen[n][0] - model.predict(n, yaw_deg) for n in base])
    shift = float(np.sum(w * r) / np.sum(w))
    return {n: ((seen[n][0] - model.predict(n, yaw_deg) - shift) / model.height,
                (seen[n][1] - model.z[n]) / model.height) for n in names}


def pose_rms(res: dict[str, tuple[float, float]]) -> float | None:
    if not res:
        return None
    return math.sqrt(float(np.mean([a * a + b * b for a, b in res.values()])))


def arm_drop_deg(model: BodyModel, seen: dict[str, tuple[float, float, float]], side: str) -> float | None:
    """A view's ``side`` arm angle below horizontal, from how far its wrist hangs below its shoulder over the
    arm's length in the front view: asin(drop / length). Independent of the yaw for a straight arm (an orthographic
    camera with no elevation keeps heights). None when either joint or the front arm is missing."""
    sh, wr = seen.get(f"{side}_shoulder"), seen.get(f"{side}_wrist")
    length = model.arm_length.get(side)
    if not sh or not wr or not length:
        return None
    return math.degrees(math.asin(max(-1.0, min(1.0, (sh[1] - wr[1]) / length))))
