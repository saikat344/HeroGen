"""Shoulder covers: overlapping pad pieces that hide the shoulder joint, as rigid-section characters do.

A cover is the carve's own surface around the shoulder pivot (every body face within `radius` cut radii of it,
largest connected patch), pushed out along the vertex normals by up to `outset` and closed into a thin shell
whose inner face sits `inset` below the surface. Each vertex is pushed out only as far as every observed
view's mask allows, so the cover thickens where it faces a camera and the rest silhouette stays the carve's. The cover is merged into its owner section, so a GLB still has exactly the 15 R15 parts.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import split


COVER_RADIUS = 1.8     # in shoulder cut radii (see stage4 for the measurements behind the defaults)
COVER_OUTSET = 0.03    # studs outside the carved surface
COVER_INSET = 0.02     # studs inside it (the shell's inner face)
# The least outset of any outer vertex. A vertex the masks hold at zero outset would sit exactly on the body
# surface it was taken from, and decimations of one mesh share vertex positions, so it could land on a vertex of
# its owner section; welding then joins the two shells and the part is no longer manifold (seen on the pencil
# sketch's carve under roblox-avatar: UpperTorso with 3 non-manifold edges). 0.001 studs is a fifth of a pixel
# at the views' pixel scale and far above the 1e-6-height weld tolerance.
COVER_MIN_OUTSET = 0.001
MAX_PINCH_PASSES = 64  # each pass drops at least one fan; a patch still pinched after this many is an error


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    n = np.zeros_like(verts)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]


def shell(verts: np.ndarray, faces: np.ndarray, outset: float, inset: float, inside=None):
    """Close a surface patch into a shell: outer copy at +outset, reversed inner copy at -inset, walls.

    With `inside` (points -> bool, True where every observed view's mask covers the point), each outer vertex
    takes the largest outset (1, 0.75, 0.5, 0.25 or 0 of it) that stays inside the masks, so the cover thickens
    only where it faces a camera and never widens a silhouette; no vertex goes out less than `COVER_MIN_OUTSET`,
    so the shell never shares a point with the surface it covers.
    """
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    v, f = verts[used], remap[faces]
    nrm = vertex_normals(v, f)
    n = len(v)
    scale = np.ones(n)
    if inside is not None:
        scale = np.zeros(n)
        for frac in (0.25, 0.5, 0.75, 1.0):
            ok = inside(v + frac * outset * nrm)
            scale = np.where(ok & (scale == frac - 0.25), frac, scale)
    outer, inner = v + np.maximum(outset * scale, COVER_MIN_OUTSET)[:, None] * nrm, v - inset * nrm
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    present = set(map(tuple, directed.tolist()))
    walls = []
    for a, b in directed.tolist():
        if (b, a) not in present:          # boundary edge a->b of the outer patch
            walls += [(b, a, a + n), (b, a + n, b + n)]
    fs = np.concatenate([f, f[:, ::-1] + n, np.array(walls, dtype=np.int64).reshape(-1, 3)])
    return np.concatenate([outer, inner]), fs


def build_cover(body_v: np.ndarray, body_f: np.ndarray, centre: np.ndarray, radius: float,
                outset: float = COVER_OUTSET, inset: float = COVER_INSET, inside=None):
    """The cover around one shoulder, or None if the patch is empty."""
    d = np.linalg.norm(body_v - centre, axis=1)
    sel = (d[body_f] < radius).all(axis=1)
    if not sel.any():
        return None
    patch = manifold_patch(body_f[sel])
    if len(patch) == 0:
        return None
    return shell(body_v, patch, outset, inset, inside)


def _largest(patch: np.ndarray) -> np.ndarray:
    """The largest piece of a face set whose faces are joined through shared edges (a shared vertex alone does
    not join them)."""
    parent = np.arange(len(patch))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[tuple, int] = {}
    for i, t in enumerate(patch.tolist()):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e = (a, b) if a < b else (b, a)
            if e in first:
                ra, rb = root(first[e]), root(i)
                if ra != rb:
                    parent[ra] = rb
            else:
                first[e] = i
    roots = np.array([root(i) for i in range(len(patch))])
    ids, counts = np.unique(roots, return_counts=True)
    return patch[roots == ids[np.argmax(counts)]]


def manifold_patch(patch: np.ndarray) -> np.ndarray:
    """The largest edge-connected piece of a face set whose boundary never touches itself.

    Faces within a radius can meet the rest of the patch at a single vertex (a pinch: two fans around one vertex
    that share no edge). Closing such a patch into a shell puts four wall triangles on that vertex's wall edge,
    so the cover is not manifold. Pieces joined only at a vertex are separated (the largest is kept), and at a
    pinch inside one piece the fans other than the largest are dropped, until none is left; a patch still pinched
    after `MAX_PINCH_PASSES` passes is an error, never returned.
    """
    if len(patch) == 0:
        return patch
    patch = _largest(patch)
    for npass in range(MAX_PINCH_PASSES + 1):
        directed = np.concatenate([patch[:, [0, 1]], patch[:, [1, 2]], patch[:, [2, 0]]])
        present = set(map(tuple, directed.tolist()))
        starts: dict[int, int] = {}
        for a, b in directed.tolist():
            if (b, a) not in present:
                starts[a] = starts.get(a, 0) + 1
        pinched = [v for v, k in starts.items() if k > 1]
        if not pinched:
            return patch
        if npass == MAX_PINCH_PASSES:
            break
        drop = np.zeros(len(patch), dtype=bool)
        for v in pinched:
            around = np.nonzero((patch == v).any(axis=1))[0]
            # fans: faces around v joined through an edge that contains v
            fan = {int(i): int(i) for i in around}

            def root(i):
                while fan[i] != i:
                    i = fan[i]
                return i

            for x in around:
                for y in around:
                    if x < y and len((set(patch[x]) & set(patch[y])) - {v}) == 1:
                        fan[root(int(x))] = root(int(y))
            groups: dict[int, list] = {}
            for i in around:
                groups.setdefault(root(int(i)), []).append(int(i))
            keep = max(groups.values(), key=len)
            for g in groups.values():
                if g is not keep:
                    drop[g] = True
        patch = patch[~drop]
        if len(patch) == 0:
            return patch
        patch = _largest(patch)
    raise ValueError(f"cover patch still pinched after {MAX_PINCH_PASSES} passes")


def add_covers(sections: dict, body_v: np.ndarray, body_f: np.ndarray, owner: str = "UpperArm",
               radius: float = COVER_RADIUS, outset: float = COVER_OUTSET, inside=None) -> tuple[dict, dict]:
    """Return (sections with covers merged into their owners, cover info per side).

    `owner` is "UpperArm" (the pad turns with the arm) or "UpperTorso" (the pad stays on the body).
    """
    out = dict(sections)
    info = {}
    for side in ("Left", "Right"):
        arm = sections[side + "UpperArm"]
        if arm.cover_rims is not None:
            # Silhouette covers keep their original cap-owner sizing. These samples never measure a joint.
            rim = [r["points"] for r in arm.cover_rims if r["neighbour"] == "UpperTorso" and len(r["points"])]
        else:
            rim = [r["points"] for r in arm.rims if r["neighbour"] == "UpperTorso" and split.is_closed_rim(r)]
        if not rim:
            continue
        rim_r = float(np.linalg.norm(np.concatenate(rim) - arm.pivot, axis=1).mean())
        cov = build_cover(body_v, body_f, arm.pivot, radius * rim_r, outset, inside=inside)
        if cov is None:
            continue
        cv, cf = cov
        name = side + "UpperArm" if owner == "UpperArm" else "UpperTorso"
        s = out[name]
        out[name] = replace(s, verts=np.concatenate([s.verts, cv]), faces=np.concatenate([s.faces, cf + len(s.verts)]),
                            cap_triangles=s.cap_triangles)
        info[side + "ShoulderCover"] = {"owner": name, "triangles": int(len(cf)),
                                        "radius_studs": round(radius * rim_r, 4), "outset_studs": outset}
    return out, info
