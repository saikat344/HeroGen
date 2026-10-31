"""Split a body mesh into the 15 R15 sections with planar cuts at the joints, then close every cut.

Labelling (per vertex):
1. Chain by the nearest bone, distance divided by the bone's radius (so a thin arm does not claim the side of a
   wide torso): torso, left/right arm, left/right leg.
2. Arm vertices proximal to the shoulder plane (normal along the upper arm) go to the torso. Leg sections are
   everything past the hip plane (normal along the thigh, just below the crotch) on that side of the body axis.
3. Inside a chain, planes through the joints split the segments: neck and waist (torso), elbow and wrist
   (arms, normals bisecting the adjacent bones), knee (bisector) and ankle (along the shin).

Cutting: every edge whose ends carry different labels is cut where the boundary function between those two
labels is zero (the joint plane when the two are adjacent, so cuts are exactly planar), and straddling
triangles are split. Both sides get the same cut vertices, so the rest pose is crack-free.

Closing: each cut opening becomes a spherical cap centred on the joint pivot and bulging into the neighbour.
Rotating a section about its pivot keeps its cap on the same sphere, so a round joint stays covered in any pose
(measured in pose.py); `cap="flat"` gives plain fans for comparison. Shoulders are cut one arm radius out from
the joint, where the arm is rounder; every pivot is the centre (area centroid) of its cut, as in R15. Any cut can
be moved along its normal (`cut_offsets`; stage 4 chooses them, see `cuts`).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from . import PARENT, PIVOT_JOINT, SECTIONS
from .carve import winding_inside
from .joints import Skeleton

SID = {n: i for i, n in enumerate(SECTIONS)}
SHOULDER_CUT_OFFSET = 1.0     # in arm radii, along the upper arm (see build_rig)
# A cut vertex stays at least this far (studs) from both ends of the edge it splits: 18 times the parts' weld
# tolerance (1e-6 of the height), a few hundredths of a pixel at the views' scale.
CUT_MIN_STUDS = 1e-4
HIP_CUT_OFFSET = 0.75         # in thigh radii, along the thigh (see build_rig)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


def seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    t = np.clip(((p - a) @ d) / max(float(d @ d), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * d), axis=1)


@dataclass
class Rig:
    bones: dict          # section -> (a, b, radius)
    planes: dict         # joint -> (point, normal pointing to the child side)
    chain: dict          # section -> chain name
    x0: float
    crotch_z: float


def build_rig(sk: Skeleton, cut_offsets: dict | None = None) -> Rig:
    j, e, r = sk.joints, sk.ends, sk.radii
    bones = {
        "Head": (j["Neck"], j["HeadTop"], r["head"]),
        "UpperTorso": (j["Waist"], j["Neck"], r["torso"]),
        "LowerTorso": (e["HipCentre"], j["Waist"], r["torso"]),
    }
    chain = {"Head": "torso", "UpperTorso": "torso", "LowerTorso": "torso"}
    planes = {
        "Neck": (j["Neck"], _unit(j["HeadTop"] - j["Neck"])),
        "Waist": (j["Waist"], _unit(j["Neck"] - j["Waist"])),
    }
    x0 = float(j["Neck"][0])
    cz = float(r["crotch_z"])
    for side, sgn in (("Left", 1.0), ("Right", -1.0)):
        sh, el, wr, tip = j[side + "Shoulder"], j[side + "Elbow"], j[side + "Wrist"], e[side + "HandTip"]
        ra = r[side + "Arm"]
        bones[side + "UpperArm"] = (sh, el, ra)
        bones[side + "LowerArm"] = (el, wr, ra * 0.9)
        bones[side + "Hand"] = (wr, tip, ra * 0.9)
        d1, d2, d3 = _unit(el - sh), _unit(wr - el), _unit(tip - wr)
        # The shoulder cut sits one arm radius out along the arm, where its cross-section is round; at the
        # joint itself the cut follows the irregular arm/chest junction and no cap covers it in poses. Worst
        # shoulder gap on the synthetic body (high / low LOD candidates): at the joint 0.22 / 0.25 studs, half
        # a radius out 0.10 / 0.13-0.17, one radius out 0.08 / 0.09-0.12. The pivot (the cut's centre, as in
        # R15) then sits about 0.8 cut radii from the estimated anatomical joint.
        planes[side + "Shoulder"] = (sh + SHOULDER_CUT_OFFSET * ra * d1, d1)
        planes[side + "Elbow"] = (el, _unit(d1 + d2))
        planes[side + "Wrist"] = (wr, _unit(d2 + d3))
        hip, kn, an, toe = j[side + "Hip"], j[side + "Knee"], j[side + "Ankle"], e[side + "Toe"]
        rl = r[side + "Leg"]
        bones[side + "UpperLeg"] = (hip, kn, rl)
        bones[side + "LowerLeg"] = (kn, an, r["shin"])
        bones[side + "Foot"] = (an, toe, r["shin"])
        # The hip cut is perpendicular to the thigh, three quarters of a thigh radius below the hip joint (a
        # quarter radius below the crotch), where each thigh is a separate round section; the LowerTorso keeps
        # the tops of the thighs, as in R15. Measured worst hip gap (synthetic / stage-2 stand-in, high LOD):
        # an oblique cut through the hip joint and the crotch 0.10 / 0.19 studs, this cut 0.07 / 0.13.
        dl = _unit(kn - hip)
        planes[side + "Hip"] = (hip + HIP_CUT_OFFSET * rl * dl, dl)
        planes[side + "Knee"] = (kn, _unit(_unit(kn - hip) + _unit(an - kn)))
        planes[side + "Ankle"] = (an, _unit(an - kn))
        for s in ("UpperArm", "LowerArm", "Hand"):
            chain[side + s] = side + "Arm"
        for s in ("UpperLeg", "LowerLeg", "Foot"):
            chain[side + s] = side + "Leg"
    for jn, off in (cut_offsets or {}).items():     # a cut moved along its own normal (see stage 4)
        p, n = planes[jn]
        planes[jn] = (p + float(off) * n, n)
    return Rig(bones, planes, chain, x0, cz)


def joint_between(a: str, b: str) -> str | None:
    """The joint shared by two sections if one is the other's parent."""
    if PARENT.get(b) == a:
        return PIVOT_JOINT[b]
    if PARENT.get(a) == b:
        return PIVOT_JOINT[a]
    return None


LEGS = ("UpperLeg", "LowerLeg", "Foot")


def compatible(a: int, b: int) -> bool:
    """Two labels whose shared boundary is a known plane (parent/child, or left/right leg)."""
    sa, sb = SECTIONS[a], SECTIONS[b]
    if joint_between(sa, sb):
        return True
    return sa.endswith(LEGS) and sb.endswith(LEGS) and sa[:4] != sb[:4]


class Labeller:
    def __init__(self, rig: Rig):
        self.rig = rig

    def nd(self, pts: np.ndarray) -> dict:
        return {s: seg_dist(pts, a, b) / rad for s, (a, b, rad) in self.rig.bones.items()}

    def pd(self, pts: np.ndarray) -> dict:
        return {k: (pts - p) @ n for k, (p, n) in self.rig.planes.items()}

    def labels(self, pts: np.ndarray) -> np.ndarray:
        rig = self.rig
        pd = self.pd(pts)
        ndd = self.nd(pts)
        names = list(rig.bones)
        nd = np.stack([ndd[s] for s in names], axis=1)
        nearest = np.array([rig.chain[s] for s in names])[np.argmin(nd, axis=1)]
        lab = np.full(len(pts), -1, dtype=np.int64)
        x = pts[:, 0]
        for side in ("Left", "Right"):
            # past the shoulder plane, anything hugging the upper arm (within 1.4 arm radii of its bone) is
            # arm even where the torso bone is nearer, so the shoulder cut stays on the plane at the armpit
            near_arm = (nearest == side + "Arm") | (ndd[side + "UpperArm"] < 1.4)
            arm = near_arm & (pd[side + "Shoulder"] > 0)
            lab[arm & (pd[side + "Elbow"] <= 0)] = SID[side + "UpperArm"]
            lab[arm & (pd[side + "Elbow"] > 0) & (pd[side + "Wrist"] <= 0)] = SID[side + "LowerArm"]
            lab[arm & (pd[side + "Wrist"] > 0)] = SID[side + "Hand"]
        free = lab < 0
        for side, sgn in (("Left", 1), ("Right", -1)):
            leg = free & ((x - rig.x0) * sgn >= 0) & (pd[side + "Hip"] > 0)
            lab[leg & (pd[side + "Knee"] <= 0)] = SID[side + "UpperLeg"]
            lab[leg & (pd[side + "Knee"] > 0) & (pd[side + "Ankle"] <= 0)] = SID[side + "LowerLeg"]
            lab[leg & (pd[side + "Ankle"] > 0)] = SID[side + "Foot"]
        free = lab < 0
        lab[free & (pd["Neck"] > 0)] = SID["Head"]
        lab[free & (pd["Neck"] <= 0) & (pd["Waist"] > 0)] = SID["UpperTorso"]
        lab[free & (pd["Waist"] <= 0)] = SID["LowerTorso"]
        return lab

    def boundary(self, a: int, b: int, pts: np.ndarray) -> np.ndarray:
        """Signed boundary function between sections a and b at points pts (> 0 on a's side)."""
        sa, sb = SECTIONS[a], SECTIONS[b]
        jn = joint_between(sa, sb)
        if jn is not None:
            p, n = self.rig.planes[jn]
            g = (pts - p) @ n
            return g if PARENT.get(sa) == sb else -g
        if sa.endswith(LEGS) and sb.endswith(LEGS) and sa[:4] != sb[:4]:
            g = pts[:, 0] - self.rig.x0
            return g if sa.startswith("Left") else -g
        nd = self.nd(pts)
        return nd[sb] - nd[sa]


def refine(verts: np.ndarray, faces: np.ndarray, lb: Labeller, max_iter: int = 6):
    """Split edges whose end labels are not neighbours, where a third section lies between them.

    On a coarse LOD one triangle can reach from the UpperTorso into a leg past the thin LowerTorso band;
    halving those edges (and conformingly their triangles) lets every cut be a plane cut.
    """
    added = 0
    for _ in range(max_iter):
        lab = lb.labels(verts)
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        e.sort(axis=1)
        e = np.unique(e, axis=0)
        la, lb_ = lab[e[:, 0]], lab[e[:, 1]]
        diff = la != lb_
        cand = [(x, a, b) for x, a, b in zip(e[diff].tolist(), la[diff], lb_[diff]) if not compatible(a, b)]
        if not cand:
            break
        # Split only edges that pass through a third section (a thin band between them). Two sections that
        # really touch (a hand resting on a thigh) stay as they are and are cut by the nearest-bone boundary.
        ce = np.array([x for x, _, _ in cand])
        ts = np.array([0.2, 0.4, 0.6, 0.8])
        samples = verts[ce[:, 0]][:, None] + ts[None, :, None] * (verts[ce[:, 1]] - verts[ce[:, 0]])[:, None]
        sl = lb.labels(samples.reshape(-1, 3)).reshape(len(ce), len(ts))
        ends = np.array([[a, b] for _, a, b in cand])
        third = ((sl != ends[:, :1]) & (sl != ends[:, 1:])).any(axis=1)
        bad = [tuple(x) for x in ce[third].tolist()]
        if not bad or added > len(verts):
            break
        mid = {}
        new_pts = []
        for a, b in bad:
            mid[(a, b)] = len(verts) + len(new_pts)
            new_pts.append((verts[a] + verts[b]) / 2)
        verts = np.concatenate([verts, np.array(new_pts)])
        added += len(new_pts)
        out = []
        for f in faces.tolist():
            ms = [mid.get((min(f[k], f[(k + 1) % 3]), max(f[k], f[(k + 1) % 3]))) for k in range(3)]
            nm = sum(m is not None for m in ms)
            if nm == 0:
                out.append(f)
                continue
            # rotate so that edge 0 (a-b) is marked and, with two marks, edge 2 (c-a) is the unmarked one
            for r in range(3):
                if ms[0] is not None and (nm != 2 or ms[2] is None):
                    break
                f = f[1:] + f[:1]
                ms = ms[1:] + ms[:1]
            a, b, c = f
            mab, mbc, mca = ms
            if nm == 1:
                out += [[a, mab, c], [mab, b, c]]
            elif nm == 2:
                out += [[mab, b, mbc], [a, mab, mbc], [a, mbc, c]]
            else:
                out += [[a, mab, mca], [mab, b, mbc], [mca, mbc, c], [mab, mbc, mca]]
        faces = np.array(out, dtype=np.int64)
    return verts, faces, added


def split_mesh(verts: np.ndarray, faces: np.ndarray, sk: Skeleton, cut_offsets: dict | None = None):
    rig = build_rig(sk, cut_offsets)
    lb = Labeller(rig)
    verts, faces, refined = refine(verts, faces, lb)
    lab = lb.labels(verts)
    cut_pos: list[np.ndarray] = []
    cut_key: dict[tuple, int] = {}
    cut_pair: list[tuple[int, int]] = []
    n0 = len(verts)

    def cut(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key in cut_key:
            return cut_key[key]
        la, lb_ = lab[key[0]], lab[key[1]]
        g = lb.boundary(la, lb_, verts[list(key)])
        t = g[0] / (g[0] - g[1]) if g[0] * g[1] < 0 else 0.5
        # a vertex that lies almost on the plane would put the new cut vertex on top of it: closer than the weld
        # tolerance, the sliver between them collapses when the part is welded and leaves boundary edges
        length = float(np.linalg.norm(verts[key[1]] - verts[key[0]]))
        lo = CUT_MIN_STUDS / length if length > 2 * CUT_MIN_STUDS else 0.5
        t = float(np.clip(t, lo, 1.0 - lo))
        p = verts[key[0]] + t * (verts[key[1]] - verts[key[0]])
        cut_key[key] = n0 + len(cut_pos)
        cut_pos.append(p)
        cut_pair.append((int(la), int(lb_)))
        return cut_key[key]

    fl = lab[faces]
    same = (fl[:, 0] == fl[:, 1]) & (fl[:, 1] == fl[:, 2])
    sec_faces: dict[int, list] = defaultdict(list)
    for s in np.unique(fl[same, 0]):
        sec_faces[int(s)].append(faces[same & (fl[:, 0] == s)])
    extra: dict[int, list] = defaultdict(list)
    centre_pos: list[np.ndarray] = []
    for f, l in zip(faces[~same], fl[~same]):
        a, b, c = (int(x) for x in f)
        la, lb2, lc = (int(x) for x in l)
        if la != lb2 and lb2 != lc and la != lc:
            pab, pbc, pca = cut(a, b), cut(b, c), cut(c, a)
            centre_pos.append((verts[a] + verts[b] + verts[c]) / 3)
            mi = -len(centre_pos)          # centre vertices are numbered after the cut vertices below
            extra[la] += [(a, pab, mi), (a, mi, pca)]
            extra[lb2] += [(b, pbc, mi), (b, mi, pab)]
            extra[lc] += [(c, pca, mi), (c, mi, pbc)]
            continue
        # rotate so the odd vertex is last: (a, b) share a label, c differs
        if la == lb2:
            pass
        elif lb2 == lc:
            a, b, c, la, lb2, lc = b, c, a, lb2, lc, la
        else:
            a, b, c, la, lb2, lc = c, a, b, lc, la, lb2
        pbc, pca = cut(b, c), cut(c, a)
        extra[la] += [(a, b, pbc), (a, pbc, pca)]
        extra[lc] += [(pca, pbc, c)]
    all_pos = np.concatenate([verts, np.array(cut_pos).reshape(-1, 3), np.array(centre_pos).reshape(-1, 3)])
    n_cut = len(cut_pos)

    def fix(t):
        return [x if x >= 0 else n0 + n_cut + (-x - 1) for x in t]

    pair_of = {n0 + i: p for i, p in enumerate(cut_pair)}
    sections = {}
    for s in range(len(SECTIONS)):
        parts = sec_faces.get(s, [])
        ex = np.array([fix(t) for t in extra.get(s, [])], dtype=np.int64).reshape(-1, 3)
        fs = np.concatenate(parts + [ex]) if parts or len(ex) else np.zeros((0, 3), np.int64)
        sections[SECTIONS[s]] = fs
    return rig, lb, all_pos, sections, pair_of, refined


def boundary_loops(faces: np.ndarray) -> list[list[int]]:
    """Directed boundary loops (edges a->b of faces whose twin b->a is missing)."""
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    present = set(map(tuple, directed.tolist()))
    nxt: dict[int, list[int]] = defaultdict(list)
    for a, b in directed.tolist():
        if (b, a) not in present:
            nxt[a].append(b)
    loops, used = [], set()
    for start in list(nxt):
        for b0 in nxt[start]:
            if (start, b0) in used:
                continue
            loop = [start]
            used.add((start, b0))
            cur = b0
            guard = 0
            while cur != start and guard < 100000:
                loop.append(cur)
                cands = [c for c in nxt.get(cur, []) if (cur, c) not in used]
                if not cands:
                    break
                used.add((cur, cands[0]))
                cur = cands[0]
                guard += 1
            if cur == start and len(loop) >= 3:
                loops.extend(_simple_loops(loop))
    return loops


def _simple_loops(loop: list[int]) -> list[list[int]]:
    """Split a traced loop that revisits a vertex (two holes touching at a point) into simple loops."""
    out, stack, pos = [], [], {}
    for v in loop:
        if v in pos:
            i = pos[v]
            cyc = stack[i:]
            for u in cyc:
                pos.pop(u, None)
            del stack[i:]
            if len(cyc) >= 3:
                out.append(cyc)
        pos[v] = len(stack)
        stack.append(v)
    if len(stack) >= 3:
        out.append(stack)
    return out


def _slerp(u: np.ndarray, n: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation from unit vectors u [k, 3] towards the unit vector n by fraction t."""
    dot = np.clip(u @ n, -1.0, 1.0)
    om = np.arccos(dot)[:, None]
    so = np.sin(om)
    lin = (1 - t) * u + t * n[None]
    lin /= np.maximum(np.linalg.norm(lin, axis=1), 1e-12)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        sl = (np.sin((1 - t) * om) * u + np.sin(t * om) * n[None]) / so
    return np.where(so > 1e-6, sl, lin)


def cap_geometry(rim: np.ndarray, cp: dict):
    """Ring points and apex of a spherical cap over a rim loop.

    Each of `ring_points` rim samples (even by arc length) swings along its great circle about the pivot
    towards the cap axis, at its own distance from the pivot: rotating the section about the pivot keeps the
    cap on the same sphere, which is what keeps a joint covered in poses. Heights above the rim plane are
    limited to `depth`, then scaled per sample by `squash` and at the apex by `squash_apex`, which keeps the
    cap inside a neighbour that is thin in places.
    """
    c, nd, depth, k = cp["centre"], _unit(cp["dir"]), cp["depth"], cp["rings"]
    m = len(rim)
    q = int(min(m, max(6, cp.get("ring_points", m))))
    seg = np.linalg.norm(np.roll(rim, -1, axis=0) - rim, axis=1)
    t_rim = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-12)   # t of rim vertex i, i=0..m
    sample = np.unique(np.searchsorted(t_rim[:m], np.arange(q) / q, side="left").clip(0, m - 1))
    if len(sample) < 3:      # a tiny opening (three or four rim vertices, unevenly spaced): ring on every one
        sample = np.arange(m)
    t_ring = np.concatenate([t_rim[sample], [1.0]])
    w = rim[sample] - c
    rad = np.linalg.norm(w, axis=1)
    u = w / np.maximum(rad, 1e-9)[:, None]
    rim_c = rim.mean(axis=0)
    ring_pts = [c + _slerp(u, nd, ring / (k + 1)) * rad[:, None] for ring in range(1, k + 1)]
    apex = c + nd * float(rad.mean())
    top = float((apex - rim_c) @ nd)
    base = min(1.0, depth / top) if top > 0 and depth > 0 else 1.0
    col = base * np.broadcast_to(np.asarray(cp.get("squash", 1.0), dtype=np.float64), (len(sample),))
    ring_pts = [x - ((x - rim_c) @ nd * (1 - col))[:, None] * nd[None] for x in ring_pts]
    a_sq = base * float(cp.get("squash_apex", 1.0))
    apex = apex - nd * (top * (1 - a_sq))
    return sample, t_rim, t_ring, ring_pts, apex


def close_section(verts: np.ndarray, faces: np.ndarray, loops: list[list[int]], caps: list[dict]):
    """Append cap geometry for each loop. `caps[i]` = {"centre", "dir", "depth", "rings", "ring_points",
    "squash"}, or None for a flat fan.

    A dome's first ring has `ring_points` vertices sampled evenly along the rim (by arc length) and is zipped
    to the full-resolution rim, so a dome costs about rim + ring_points * (2 * rings) triangles instead of
    rim * (2 * rings + 1).
    """
    v = [verts]
    f = [faces]
    n = len(verts)
    for loop, cp in zip(loops, caps):
        ids = np.array(loop)
        rim = verts[ids]
        if cp is None or cp["rings"] < 1:
            c = rim.mean(axis=0)
            v.append(c[None])
            f.append(np.stack([np.roll(ids, -1), ids, np.full(len(ids), n)], axis=1))
            n += 1
            continue
        m = len(ids)
        sample, t_rim, t_ring, ring_pts, apex = cap_geometry(rim, cp)
        q = len(sample)
        rings_idx = []
        for pts in ring_pts:
            v.append(pts)
            rings_idx.append(np.arange(n, n + q))
            n += q
        # zip the rim (m vertices) to the first ring (q vertices): rim edge a->b with ring point r gives
        # (b, a, r); ring edge r->s with rim point b gives (b, r, s)
        r0 = rings_idx[0]
        i_r, j_r = 0, 0
        tris = []
        while i_r < m or j_r < q:
            adv_rim = j_r >= q or (i_r < m and t_rim[i_r + 1] <= t_ring[j_r + 1])
            if adv_rim:
                tris.append((ids[(i_r + 1) % m], ids[i_r % m], r0[j_r % q]))
                i_r += 1
            else:
                tris.append((ids[i_r % m], r0[j_r % q], r0[(j_r + 1) % q]))
                j_r += 1
        f.append(np.array(tris, dtype=np.int64))
        for ra, rb in zip(rings_idx[:-1], rings_idx[1:]):
            ra2, rb2 = np.roll(ra, -1), np.roll(rb, -1)
            f.append(np.stack([ra2, ra, rb], axis=1))
            f.append(np.stack([ra2, rb, rb2], axis=1))
        last = rings_idx[-1]
        v.append(apex[None])
        f.append(np.stack([np.roll(last, -1), last, np.full(q, n)], axis=1))
        n += 1
    return np.concatenate(v), np.concatenate(f)


@dataclass
class Section:
    name: str
    verts: np.ndarray
    faces: np.ndarray
    pivot: np.ndarray
    rims: list            # [{"neighbour", "joint", "points": [k,3], "edges": [m,2]}], edges index points
    cap_triangles: int
    cap_points_outside: int = 0   # cap samples still outside the neighbour after squashing (poke-through)
    cover_rims: list | None = None   # historical cap-owner samples for silhouette cover sizing only


def _neighbour_of(g: int, name: str, pair_of: dict) -> str | None:
    if g not in pair_of:
        return None
    a, b = pair_of[g]
    return SECTIONS[b] if SECTIONS[a] == name else SECTIONS[a]


def boundary_edges(faces: np.ndarray) -> set[tuple[int, int]]:
    """Undirected edges belonging to exactly one triangle, before any caps are added."""
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return set(map(tuple, unique[counts == 1].tolist()))


def rim_loop_vertices(rim: dict) -> np.ndarray:
    """Vertices on nondegenerate closed loops of shared edges, excluding dangling contacts.

    Biconnected blocks separate loops that touch at a vertex from bridges and open chains. A block with
    at least three noncollinear vertices encloses an opening; every edge in that block belongs to a cycle.
    This also admits several loops joined at junctions, without inventing an edge between chain endpoints.
    """
    points = np.asarray(rim["points"])
    edges = np.asarray(rim.get("edges", []), dtype=np.int64)
    n = len(points)
    empty = np.empty(0, dtype=np.int64)
    if n < 3 or edges.ndim != 2 or edges.shape[1] != 2 or len(edges) < 3:
        return empty
    if np.any(edges < 0) or np.any(edges >= n) or not np.isfinite(points).all():
        return empty
    if np.any(edges[:, 0] == edges[:, 1]) or len(np.unique(np.sort(edges, axis=1), axis=0)) != len(edges):
        return empty
    adjacent = defaultdict(list)
    for a, b in edges.tolist():
        adjacent[a].append(b)
        adjacent[b].append(a)
    discovered, low, parent = {}, {}, {}
    selected = set()
    # Iterative depth-first search avoids a recursion limit on densely sampled openings.
    for root in sorted(adjacent):
        if root in discovered:
            continue
        discovered[root] = low[root] = len(discovered)
        pending, path = [(root, iter(adjacent[root]))], []
        while pending:
            v, neighbours = pending[-1]
            w = next(neighbours, None)
            if w is not None:
                if w not in discovered:
                    parent[w] = v
                    path.append((v, w))
                    discovered[w] = low[w] = len(discovered)
                    pending.append((w, iter(adjacent[w])))
                elif w != parent.get(v) and discovered[w] < discovered[v]:
                    path.append((v, w))
                    low[v] = min(low[v], discovered[w])
                continue
            pending.pop()
            if v not in parent:
                continue
            p = parent[v]
            low[p] = min(low[p], low[v])
            if low[v] >= discovered[p]:
                block = set()
                while path:
                    edge = path.pop()
                    block.update(edge)
                    if edge == (p, v):
                        break
                ids = sorted(block)
                # Rank is a degeneracy check, not a size gate. Collinear cycles enclose no opening.
                if len(ids) >= 3 and np.linalg.matrix_rank(points[ids] - points[ids[0]]) >= 2:
                    selected.update(ids)
    return np.asarray(sorted(selected), dtype=np.int64)


def is_closed_rim(rim: dict) -> bool:
    """Whether a contact contains at least one closed loop with enclosed area."""
    return bool(len(rim_loop_vertices(rim)))


def _shared_rims(name: str, boundaries: dict, pos: np.ndarray) -> list:
    """Rings present in both sections' uncapped boundaries, including junction vertices.

    Open chains are retained as diagnostic contacts; only vertices on closed loops measure gaps.
    Cap chords are deliberately absent: a cap cannot invent a shared boundary on the original body.
    """
    rims = []
    for nb in SECTIONS:
        if nb == name:
            continue
        shared = boundaries[name] & boundaries[nb]
        adjacent = defaultdict(set)
        for a, b in shared:
            adjacent[a].add(b)
            adjacent[b].add(a)
        remaining = set(adjacent)
        while remaining:
            component, pending = set(), [min(remaining)]
            while pending:
                v = pending.pop()
                if v not in component:
                    component.add(v)
                    pending.extend(adjacent[v] - component)
            remaining -= component
            ids = sorted(component)
            remap = {v: i for i, v in enumerate(ids)}
            edges = [(remap[a], remap[b]) for a, b in sorted(shared) if a in component]
            rims.append({"neighbour": nb, "joint": joint_between(name, nb), "points": pos[ids],
                         "edges": np.asarray(edges, dtype=np.int64)})
    return rims


def _split_two_neighbours(loop: list[int], nbs: list, pos: np.ndarray, new_pos: list, base: int):
    """A loop facing two neighbours (both legs under the LowerTorso) becomes two loops joined by a chord.

    The two sub-loops run along the chord in opposite directions, so their caps share it and the section stays
    closed. Returns [(sub_loop, neighbour)] or None when the loop is not two clean runs.
    """
    n = len(loop)
    lab = list(nbs)
    runs = []
    for i in range(n):
        if lab[i] is not None and lab[i] != lab[i - 1]:
            # a run starts at i if the previous labelled vertex has another label
            k = i - 1
            while lab[k % n] is None and (k % n) != i:
                k -= 1
            if lab[k % n] != lab[i]:
                runs.append(i)
    if len(runs) != 2 or lab[runs[0]] == lab[runs[1]]:
        return None
    r0, r1 = runs
    seq = loop[r0:] + loop[:r0]
    labs = lab[r0:] + lab[:r0]
    b_start = (r1 - r0) % n
    a_lab, b_lab = labs[0], labs[b_start]
    a_end = max(i for i in range(b_start) if labs[i] == a_lab)
    b_end = max(i for i in range(b_start, n) if labs[i] == b_lab)
    gap1 = list(range(a_end + 1, b_start))
    gap2 = list(range(b_end + 1, n))
    s1 = gap1[len(gap1) // 2] if gap1 else b_start
    s2 = gap2[len(gap2) // 2] if gap2 else 0
    path_a = seq[s2:] + seq[:s1 + 1] if s2 else seq[:s1 + 1]
    path_b = seq[s1:s2 + 1] if s2 else seq[s1:] + seq[:1]
    p1, p2 = pos[seq[s1]], pos[seq[s2]]
    edge = float(np.mean(np.linalg.norm(pos[seq] - pos[np.roll(seq, 1)], axis=1)))
    m = max(1, int(round(np.linalg.norm(p2 - p1) / max(edge, 1e-6))) - 1)
    chord = []
    for k in range(1, m + 1):
        new_pos.append(p1 + (p2 - p1) * k / (m + 1))
        chord.append(base + len(new_pos) - 1)
    # A: s2 .. A .. s1, then the chord s1 -> s2; B: s1 .. B .. s2, then the chord s2 -> s1
    return [(path_a + chord, a_lab), (path_b + chord[::-1], b_lab)]


def _plane_coords(loop: np.ndarray, nrm: np.ndarray):
    o = loop.mean(axis=0)
    u = np.cross(nrm, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(nrm, [0.0, 1.0, 0.0])
    u = u / np.linalg.norm(u)
    v = np.cross(nrm, u)
    return o, u, v, np.stack([(loop - o) @ u, (loop - o) @ v], axis=1)


def loop_area(loop: np.ndarray, nrm: np.ndarray) -> float:
    """Area enclosed by an ordered loop of points, projected onto the plane with normal `nrm`."""
    if len(loop) < 3:
        return 0.0
    _, _, _, q = _plane_coords(loop, nrm)
    q2 = np.roll(q, -1, axis=0)
    return float(abs((q[:, 0] * q2[:, 1] - q2[:, 0] * q[:, 1]).sum()) / 2)


def cut_centre(loop: np.ndarray, nrm: np.ndarray, method: str = "area") -> np.ndarray:
    """Centre of a cut opening (an ordered loop of points on a plane with normal `nrm`): "area" (the pivot), the
    centroid of the enclosed polygon, which depends only on the cut's shape; "arc", the arc-length-weighted
    centroid of the loop; "vertex", the mean of its vertices, which moves with how densely the mesh happens to be
    sampled along the loop (on the pencil-sketch hybrid's waist the vertex mean wandered 0.06 studs between a
    16k-triangle decimation and the full 159k mesh while the area centroid stayed within 0.001)."""
    o = loop.mean(axis=0)
    if method == "vertex" or len(loop) < 3:
        return o
    u = np.cross(nrm, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(nrm, [0.0, 1.0, 0.0])
    u = u / np.linalg.norm(u)
    v = np.cross(nrm, u)
    q = np.stack([(loop - o) @ u, (loop - o) @ v], axis=1)
    q2 = np.roll(q, -1, axis=0)
    if method == "arc":
        seg = np.linalg.norm(q2 - q, axis=1)
        c = ((q + q2) / 2 * seg[:, None]).sum(axis=0) / max(seg.sum(), 1e-12)
    else:
        cr = q[:, 0] * q2[:, 1] - q2[:, 0] * q[:, 1]
        area = cr.sum() / 2
        if abs(area) < 1e-12:
            return o
        c = ((q + q2) * cr[:, None]).sum(axis=0) / (6 * area)
    return o + c[0] * u + c[1] * v


def build_sections(verts, faces, sk: Skeleton, cap: str = "dome", rings: int = 2,
                   ring_points: int = 16, cut_offsets: dict | None = None) -> tuple[dict, Rig]:
    """Split, find each cut loop's neighbour and joint, set pivots and close the openings."""
    rig, lb, pos, sec_faces, pair_of, refined = split_mesh(verts, faces, sk, cut_offsets)
    new_pos: list[np.ndarray] = []
    base = len(pos)
    loop_info: dict[str, list] = {}
    for name, fs in sec_faces.items():
        if len(fs) == 0:
            continue
        infos = []
        for loop in boundary_loops(fs):
            nbs = [_neighbour_of(g, name, pair_of) for g in loop]
            neighbours = {x for x in nbs if x is not None and
                          (sk.source == "sam3d-body" or joint_between(name, x))}
            parts = None
            if len(neighbours) == 2:
                parts = _split_two_neighbours(loop, nbs, pos, new_pos, base)
            if parts is None:
                cnt = Counter(x for x in nbs if x is not None)
                parts = [(loop, cnt.most_common(1)[0][0] if cnt else None)]
            for sub, nb in parts:
                jn = joint_between(name, nb) if nb else None
                shared = [g for g in sub if _neighbour_of(g, name, pair_of) == nb]
                infos.append({"loop": sub, "neighbour": nb, "joint": jn, "shared": shared})
        loop_info[name] = infos
    if new_pos:
        pos = np.concatenate([pos, np.array(new_pos)])
    # pivot per joint: area centroid of the child's largest opening at that joint, on the cut plane
    joint_centre: dict[str, tuple] = {}
    for name, infos in loop_info.items():
        for info in infos:
            jn = info["joint"]
            if jn is None or jn != PIVOT_JOINT[name] or not info["shared"]:
                continue
            pts = pos[info["loop"]]
            p, nrm = rig.planes[jn]
            area = loop_area(pts, nrm)         # the largest opening by area, not by vertex count
            if jn not in joint_centre or area > joint_centre[jn][1]:
                c = cut_centre(pts, nrm)
                c = c - ((c - p) @ nrm) * nrm      # onto the cut plane (R15: the pivot is the cut's centre)
                joint_centre[jn] = (c, area)
    flat_cache: dict[str, tuple] = {}

    def closed_flat(name: str):
        """The section closed with flat fans: the volume a neighbour's cap must stay inside."""
        if name not in flat_cache:
            fs = sec_faces[name]
            extra_v, extra_f = [], []
            for info in loop_info.get(name, []):
                ids = np.array(info["loop"])
                k = len(pos) + len(extra_v)
                extra_v.append(pos[ids].mean(axis=0))
                extra_f.append(np.stack([np.roll(ids, -1), ids, np.full(len(ids), k)], axis=1))
            v = np.concatenate([pos, np.array(extra_v).reshape(-1, 3)])
            f = np.concatenate([fs] + extra_f) if extra_f else fs
            used = np.unique(f)
            remap = np.full(len(v), -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
            flat_cache[name] = (v[used], remap[f])
        return flat_cache[name]

    boundaries = {name: boundary_edges(fs) for name, fs in sec_faces.items()}
    out = {}
    for name in SECTIONS:
        fs = sec_faces.get(name)
        jn_pivot = PIVOT_JOINT[name]
        pivot = joint_centre[jn_pivot][0] if jn_pivot in joint_centre else np.asarray(sk.joints[jn_pivot], float)
        if fs is None or len(fs) == 0:
            out[name] = Section(name, np.zeros((0, 3)), np.zeros((0, 3), np.int64), pivot, [], 0)
            continue
        infos = loop_info[name]
        used = np.unique(np.concatenate([fs.ravel()] + [np.array(i["loop"]) for i in infos]))
        remap = np.full(len(pos), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        lv, lf = pos[used], remap[fs]
        loops, caps = [], []
        rims = _shared_rims(name, boundaries, pos)
        sec_centroid = pos[np.unique(fs)].mean(axis=0)
        for info in infos:
            loops.append([int(remap[g]) for g in info["loop"]])
            jn = info["joint"]
            if cap == "flat" or jn is None:
                caps.append(None)
                continue
            p, nrm = rig.planes[jn]
            c = joint_centre[jn][0] if jn in joint_centre else np.asarray(p, float)
            d = nrm if (c - sec_centroid) @ nrm > 0 else -nrm
            rad = float(np.linalg.norm(pos[info["loop"]] - c, axis=1).mean())
            # keep the dome inside the neighbour: no deeper than 70% of the neighbour's reach along d
            reach = rad
            nb_faces = sec_faces.get(info["neighbour"])
            if nb_faces is not None and len(nb_faces):
                reach = float(((pos[np.unique(nb_faces)] - c) @ d).max())
            cp = {"centre": c, "dir": d, "depth": max(0.2 * rad, min(rad, 0.7 * reach)),
                  "rings": rings, "ring_points": ring_points}
            # flatten the cap until it is hidden inside the neighbour at rest
            nb = info["neighbour"]
            if nb in sec_faces and len(sec_faces[nb]):
                nv, nf = closed_flat(nb)
                rim_pts = pos[info["loop"]]
                n_s = len(cap_geometry(rim_pts, cp)[0])
                col = np.ones(n_s)
                cp["squash"], cp["squash_apex"] = col, 1.0
                for level in (0.75, 0.5, 0.3, 0.15, None):
                    _, _, _, ring_pts, apex = cap_geometry(rim_pts, cp)
                    inside = winding_inside(np.concatenate(ring_pts), nv, nf).reshape(len(ring_pts), n_s)
                    out_r = ~inside.all(axis=0)
                    out_a = not winding_inside(apex[None], nv, nf)[0]
                    if level is None or (not out_r.any() and not out_a):
                        break
                    col = np.where(out_r, np.minimum(col, level), col)
                    cp["squash"] = col
                    if out_a:
                        cp["squash_apex"] = min(cp["squash_apex"], level)
                cp["outside"] = int(out_r.sum() + out_a)
            caps.append(cp)
        n_before = len(lf)
        cv, cf = close_section(lv, lf, loops, caps)
        poke = int(sum(cp.get("outside", 0) for cp in caps if cp is not None))
        cover_rims = ([{"neighbour": i["neighbour"], "points": pos[i["shared"]]} for i in infos]
                      if sk.source == "silhouette" else None)
        out[name] = Section(name, cv, cf, pivot, rims, int(len(cf) - n_before), poke, cover_rims)
    return out, rig
