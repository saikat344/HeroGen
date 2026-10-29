"""Pose test: rotate sections about their pivots and measure how far the joints open.

Gap measure, per joint and pose: the rim of a cut (the loop both sections shared at rest) is carried along by
the parent and by the child. A rim point carried by one section that ends up outside the other section's
closed surface marks an exposed cut; its distance to that surface is the gap there. The joint's gap is the
largest such distance over both sides, in studs. At rest every rim point lies on both surfaces, so the gap is
0; a covered joint stays near 0 in any pose.

Poses follow ZBurst's intake deformation set (arms up, deep crouch, aim/arms forward, run, twist, head turn).
"""
from __future__ import annotations

import numpy as np

from . import PARENT, PIVOT_JOINT, SECTIONS
from .carve import point_mesh_distance, ray_hits, winding_inside
from .meshindex import MeshIndex

FWD = np.array([0.0, -1.0, 0.0])
BACK = np.array([0.0, 1.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])


def rot(axis, deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    t = np.radians(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def swing(bone_dir, target, deg: float) -> np.ndarray:
    """Rotate a bone direction towards `target` by `deg`."""
    ax = np.cross(bone_dir, target)
    if np.linalg.norm(ax) < 1e-9:
        ax = np.cross(bone_dir, [1.0, 0, 0])
    return rot(ax, deg)


def pose_library(bone_dirs: dict) -> dict[str, dict[str, np.ndarray]]:
    """Local rotations per section (in the parent's frame) for each test pose."""
    d = bone_dirs
    out_l, out_r = np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])
    poses = {"rest": {}}
    for name, s1, s2 in (("walk_a", "Left", "Right"), ("walk_b", "Right", "Left")):
        poses[name] = {
            s1 + "UpperLeg": swing(d[s1 + "UpperLeg"], FWD, 30), s1 + "LowerLeg": swing(d[s1 + "LowerLeg"], BACK, 15),
            s2 + "UpperLeg": swing(d[s2 + "UpperLeg"], BACK, 25), s2 + "LowerLeg": swing(d[s2 + "LowerLeg"], BACK, 40),
            s2 + "Foot": rot([1, 0, 0], -15),
            s1 + "UpperArm": swing(d[s1 + "UpperArm"], BACK, 25), s2 + "UpperArm": swing(d[s2 + "UpperArm"], FWD, 25),
            s1 + "LowerArm": swing(d[s1 + "LowerArm"], FWD, 20), s2 + "LowerArm": swing(d[s2 + "LowerArm"], FWD, 30),
        }
    poses["arms_forward"] = {
        "LeftUpperArm": swing(d["LeftUpperArm"], FWD, 80), "RightUpperArm": swing(d["RightUpperArm"], FWD, 80),
        "LeftLowerArm": swing(d["LeftLowerArm"], FWD, 20), "RightLowerArm": swing(d["RightLowerArm"], FWD, 20),
        "LeftHand": rot(d["LeftLowerArm"], 30), "RightHand": rot(d["RightLowerArm"], -30),
    }
    poses["arms_up"] = {
        "LeftUpperArm": swing(d["LeftUpperArm"], out_l + UP, 75), "RightUpperArm": swing(d["RightUpperArm"], out_r + UP, 75),
        "LeftLowerArm": swing(d["LeftLowerArm"], UP, 30), "RightLowerArm": swing(d["RightLowerArm"], UP, 30),
    }
    poses["crouch"] = {
        "LeftUpperLeg": swing(d["LeftUpperLeg"], FWD, 75), "RightUpperLeg": swing(d["RightUpperLeg"], FWD, 75),
        "LeftLowerLeg": swing(d["LeftLowerLeg"], BACK, 100), "RightLowerLeg": swing(d["RightLowerLeg"], BACK, 100),
        "LeftFoot": rot([1, 0, 0], 25), "RightFoot": rot([1, 0, 0], 25),
        "UpperTorso": swing(UP, FWD, 25), "Head": swing(UP, BACK, 20),
        "LeftUpperArm": swing(d["LeftUpperArm"], FWD, 40), "RightUpperArm": swing(d["RightUpperArm"], FWD, 40),
    }
    poses["twist_turn"] = {
        "UpperTorso": rot(UP, 30), "Head": rot(UP, 45),
        "LeftUpperArm": swing(d["LeftUpperArm"], FWD, 30), "RightUpperArm": swing(d["RightUpperArm"], BACK, 30),
    }
    return poses


def world_matrices(local: dict[str, np.ndarray], pivots: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """4x4 world transforms (relative to rest) for every section, composed down the hierarchy."""
    out: dict[str, np.ndarray] = {}

    def get(name: str) -> np.ndarray:
        if name in out:
            return out[name]
        r = local.get(name, np.eye(3))
        p = np.asarray(pivots[name], dtype=np.float64)
        m = np.eye(4)
        m[:3, :3] = r
        m[:3, 3] = p - r @ p
        parent = PARENT[name]
        out[name] = (get(parent) @ m) if parent else m
        return out[name]

    for n in SECTIONS:
        get(n)
    return out


def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ m[:3, :3].T + m[:3, 3]


def _directions() -> np.ndarray:
    d = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)],
                 dtype=np.float64)
    return d / np.linalg.norm(d, axis=1)[:, None]


VIEW_DIRS = _directions()     # 26 directions a gap is looked for from (cube faces, edges and corners)
VIS_RADIUS = 1.0              # studs: posed geometry within this distance of the joint can hide a gap
VIS_MIN_GAP = 0.01            # studs: smaller exposures are not ray-tested


class _Brute:
    """The brute-force queries (every point against every triangle), kept to check `MeshIndex` against."""

    def __init__(self, verts, faces):
        self.verts, self.faces = verts, faces
        self.tri = verts[faces]

    def inside(self, pts):
        return winding_inside(pts, self.verts, self.faces)

    def distance(self, pts):
        return point_mesh_distance(pts, self.verts, self.faces)

    def ray_hits(self, o, d):
        return ray_hits(o, d, self.tri)


def joint_gaps(sections: dict, poses: dict, visible: bool = False, fast: bool = True) -> dict:
    """Largest gap (studs) per joint per pose, using the rim shared by each child and its parent.

    With `visible`, a rim point outside the other section only counts if it can be seen: a ray from it in at
    least one of 26 directions leaves the posed body (every section, shoulder covers included, within
    VIS_RADIUS of the joint) without hitting anything. A gap under a cover, or inside a crease the body itself
    closes, is therefore not counted; a gap a cover only hides at rest is.

    `fast` (the default) answers the inside, distance and ray queries through `MeshIndex` grids, one per section
    for its rest frame and one per posed joint neighbourhood; `fast=False` runs the brute-force queries.
    """
    query = MeshIndex if fast else _Brute
    index: dict[str, object] = {}

    def rest_index(name: str):
        if name not in index:
            index[name] = query(sections[name].verts, sections[name].faces)
        return index[name]

    pivots = {n: s.pivot for n, s in sections.items()}
    rims = {}
    for name, s in sections.items():
        parent = PARENT[name]
        if parent is None:
            continue
        pts = [r["points"] for r in s.rims if r["neighbour"] == parent]
        if pts and len(sections[parent].faces) and len(s.faces):
            rims[name] = np.concatenate(pts)
    result: dict[str, dict[str, float]] = {}
    for pname, local in poses.items():
        mats = world_matrices(local, pivots)
        posed = None
        if visible:
            posed = {n: _apply(mats[n], s.verts) for n, s in sections.items() if len(s.faces)}
        row = {}
        for child, rim in rims.items():
            parent = PARENT[child]
            mp, mc = mats[parent], mats[child]
            worst = 0.0
            near = None
            for carrier, other, mo in ((mp, child, mc), (mc, parent, mp)):
                # rim carried by one section, tested against the other's rest-frame surface
                pts = _apply(np.linalg.inv(mo) @ carrier, rim)
                so = rest_index(other)
                out = ~so.inside(pts)
                if not out.any():
                    continue
                dist = so.distance(pts[out])
                if visible:
                    keep = dist > VIS_MIN_GAP
                    if not keep.any():
                        worst = max(worst, float(dist.max()))
                        continue
                    world = _apply(carrier, rim[out][keep])
                    if near is None:
                        # the posed body near the joint: every triangle whose centre is within VIS_RADIUS
                        centre = mats[child][:3, :3] @ pivots[child] + mats[child][:3, 3]
                        vs, fs, base = [], [], 0
                        for n, pv in posed.items():
                            f = sections[n].faces
                            f = f[np.linalg.norm(pv[f].mean(axis=1) - centre, axis=1) < VIS_RADIUS]
                            vs.append(pv)
                            fs.append(f + base)
                            base += len(pv)
                        near = query(np.concatenate(vs), np.concatenate(fs))
                    k, nd = len(world), len(VIEW_DIRS)
                    o = np.repeat(world, nd, axis=0)
                    d = np.tile(VIEW_DIRS, (k, 1))
                    blocked = near.ray_hits(o + d * 2e-3, d).reshape(k, nd)
                    seen = ~blocked.all(axis=1)
                    small = dist[~keep]
                    cand = list(dist[keep][seen]) + list(small)
                    if cand:
                        worst = max(worst, float(max(cand)))
                else:
                    worst = max(worst, float(dist.max()))
            row[PIVOT_JOINT[child]] = round(worst, 4)
        result[pname] = row
    return result


def bone_directions(sk) -> dict[str, np.ndarray]:
    j, e = sk.joints, sk.ends

    def u(a, b):
        v = np.asarray(b) - np.asarray(a)
        return v / max(np.linalg.norm(v), 1e-12)

    d = {}
    for s in ("Left", "Right"):
        d[s + "UpperArm"] = u(j[s + "Shoulder"], j[s + "Elbow"])
        d[s + "LowerArm"] = u(j[s + "Elbow"], j[s + "Wrist"])
        d[s + "UpperLeg"] = u(j[s + "Hip"], j[s + "Knee"])
        d[s + "LowerLeg"] = u(j[s + "Knee"], j[s + "Ankle"])
        d[s + "Foot"] = u(j[s + "Ankle"], e[s + "Toe"])
    return d
