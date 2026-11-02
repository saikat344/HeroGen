"""Reconcile rig landmarks with the geometry that will actually be sectioned and exported."""
from __future__ import annotations

import numpy as np

from . import PARENT
from .meshindex import MeshIndex


def sleeve_support(verts, faces, a, b):
    """Locate a closed sleeve cross section near a bone's midpoint, including hollow cuffs.

    A winding test at the bone centre rejects a hollow sleeve even when its enclosing surface exists.
    Intersect the source triangles with the transverse plane instead. Only a closed local contour fits:
    both its centre's displacement and its radius must fit within half this segment's length. A remote
    torso, robe hem or prop cannot supply an arbitrarily distant substitute for an absent arm.
    """
    direction = b - a
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        return None
    direction /= length
    midpoint = (a + b) / 2
    distance = (verts - midpoint) @ direction
    signed = distance[faces]
    crossing = (signed.min(1) < 0) & (signed.max(1) > 0)
    triangles, signed = verts[faces[crossing]], signed[crossing]
    points, ids, adjacency = [], {}, {}
    # Quantisation only joins the two evaluations of an edge shared by source triangles.
    tolerance = max(float(np.ptp(verts, axis=0).max()) * 1e-8, 1e-12)
    for triangle, distances in zip(triangles, signed):
        pair = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if distances[i] * distances[j] >= 0:
                continue
            point = triangle[i] + distances[i] / (distances[i] - distances[j]) * (triangle[j] - triangle[i])
            key = tuple(np.rint(point / tolerance).astype(np.int64))
            if key not in ids:
                ids[key] = len(points)
                points.append(point)
            pair.append(ids[key])
        if len(pair) == 2:
            i, j = pair
            adjacency.setdefault(i, set()).add(j)
            adjacency.setdefault(j, set()).add(i)
    points = np.asarray(points)
    remaining, candidates = set(adjacency), []
    while remaining:
        component = {remaining.pop()}
        pending = list(component)
        while pending:
            i = pending.pop()
            for j in adjacency[i] - component:
                component.add(j)
                remaining.discard(j)
                pending.append(j)
        if len(component) < 3 or any(len(adjacency[i]) != 2 for i in component):
            continue
        edges = [(i, j) for i in component for j in adjacency[i] if i < j]
        start, end = points[np.asarray(edges).T]
        weights = np.linalg.norm(end - start, axis=1)
        if weights.sum() <= tolerance:
            continue
        centre = np.average((start + end) / 2, axis=0, weights=weights)
        offset = float(np.linalg.norm(centre - midpoint))
        radius = float(np.linalg.norm(points[list(component)] - centre, axis=1).max())
        if offset <= length / 2 and tolerance < radius <= length / 2:
            candidates.append((offset, centre, radius))
    if not candidates:
        return None
    offset, centre, radius = min(candidates, key=lambda item: item[0])
    return {"centre": centre.tolist(), "radius_studs": radius, "offset_studs": offset}


def _mask_radius(front, point, direction):
    """Half width of the occupied front-mask interval across a bone, nearest its centre."""
    d = np.asarray(direction, float)[[0, 2]]
    d /= max(float(np.linalg.norm(d)), 1e-12)
    perpendicular = np.array([-d[1], d[0]])
    steps = np.arange(-front.size, front.size + 1) / front.pps
    xz = np.asarray(point)[[0, 2]] + steps[:, None] * perpendicular
    row, col = front.uv_to_pixel(xz[:, 0] * (-1 if front.name == "back" else 1), xz[:, 1])
    valid = (row >= 0) & (row < front.size) & (col >= 0) & (col < front.size)
    occupied = np.zeros(len(steps), bool)
    occupied[valid] = front.mask[row[valid], col[valid]]
    edges = np.diff(np.r_[False, occupied, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        raise ValueError("cannot measure the rig bone's width in the front mask")
    k = int(np.argmin(np.maximum(steps[starts], -steps[stops - 1])))
    return float((stops[k] - starts[k]) / (2 * front.pps))


def _mesh_radius(index, point, direction):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    u = np.cross(d, [0, 1, 0])
    if np.linalg.norm(u) < 1e-12:
        u = np.cross(d, [1, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    theta = np.arange(16) * (2 * np.pi / 16)
    directions = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    _, _, distances = index.ray_first_hit(np.tile(point, (16, 1)), directions)
    diameters = distances[:8] + distances[8:]
    finite = diameters[np.isfinite(diameters)]
    if not len(finite):
        raise ValueError("cannot measure the arm cross section at the recorded forearm")
    return float(np.median(finite) / 2)


def reconcile_skeleton(sk, record, verts, faces, front):
    """Repair contradictory hybrid landmarks using the cached anatomical joints and source geometry.

    Silhouette limb roots remain preferred. A pair of hips must straddle the pelvis, and an A-pose shoulder
    must be above its elbow. Props can invalidate those relationships while the anatomical record remains
    usable. Distal hand endpoints must continue outwards from the wrists, rather than follow a staff.
    """
    changes = {}
    anatomical = (record or {}).get("sources", {}).get("sam3d-body", {})
    j = sk.joints
    root_x = j["Root"][0]
    hybrid = (record or {}).get("source") == "sam3d-body"
    hip_bad = hybrid and not j["RightHip"][0] < root_x < j["LeftHip"][0]
    replacements = [s + "Hip" for s in ("Left", "Right")] if hip_bad else []
    replacements += [s + "Shoulder" for s in ("Left", "Right")
                     if hybrid and j[s + "Shoulder"][2] <= j[s + "Elbow"][2]]
    for name in replacements:
        if name not in anatomical:
            raise ValueError(f"contradictory {name} landmark without a recorded anatomical alternative")
        old = j[name].copy()
        j[name] = np.asarray(anatomical[name], float)
        changes[name] = {"source": "recorded anatomical joint", "distance_studs": float(np.linalg.norm(j[name] - old))}
        sk.evidence[name] = "recorded anatomical joint: silhouette limb root contradicts the limb topology"
    if hip_bad:
        if not j["RightHip"][0] < root_x < j["LeftHip"][0]:
            raise ValueError("recorded anatomical hips do not straddle the pelvis")
        sk.ends["HipCentre"] = (j["LeftHip"] + j["RightHip"]) / 2
        sk.radii["torso"] = _mask_radius(front, j["Waist"], [0, 0, 1])
        for side in ("Left", "Right"):
            # Each leg's medial boundary is the body axis, even where a robe joins the legs in the mask.
            hip, knee = j[side + "Hip"], j[side + "Knee"]
            sk.radii[side + "Leg"] = min(abs(hip[0] - root_x), _mask_radius(front, (hip + knee) / 2, knee - hip))
        sk.radii["shin"] = min(_mask_radius(front, j[s + "Ankle"], [0, 0, 1]) for s in ("Left", "Right"))
        sk.radii["head"] = _mask_radius(front, (j["Neck"] + j["HeadTop"]) / 2, [0, 0, 1])
        sk.radii["neck"] = _mask_radius(front, j["Neck"], [0, 0, 1])
    for side in ("Left", "Right"):
        elbow, wrist = j[side + "Elbow"], j[side + "Wrist"]
        tip = sk.ends[side + "HandTip"]
        outward = wrist[0] - root_x
        if (tip[0] - wrist[0]) * outward < 0:
            direction = wrist - elbow
            length = float(np.linalg.norm(direction))
            direction /= max(length, 1e-12)
            radius = _mesh_radius(MeshIndex(verts, faces), (elbow + wrist) / 2, direction)
            delta = verts - wrist
            along = delta @ direction
            across = np.linalg.norm(delta - along[:, None] * direction, axis=1)
            # The distal bone follows the forearm through the nearby hand surface, excluding remote props.
            candidates = (along > 0) & (along <= length) & (across <= radius)
            if not candidates.any():
                raise ValueError(f"no source hand surface beyond {side}Wrist")
            new = wrist + direction * float(along[candidates].max())
            sk.ends[side + "HandTip"] = new
            sk.radii[side + "Arm"] = radius
            changes[side + "HandTip"] = {"source": "source hand along the forearm", "distance_studs": float(np.linalg.norm(new - tip))}
    return changes


def fit_pivots(sections):
    """Keep existing pivots within both part envelopes; otherwise use the nearest shared cut edge.

    A cut centroid projected onto an infinite joint plane need not remain in the finite opening. A missing
    child loop can leave the anatomical estimate in place. When no shared edge exists, retain the pivot
    assigned by build_sections (the cut centroid or anatomical estimate) and record that fallback.
    Root has no neighbouring mesh, so its fallback is the nearest point on its own surface.
    """
    changes = {}
    for name, section in sections.items():
        if not len(section.faces):
            continue
        parent = sections.get(PARENT[name])
        edges = _shared_edges(section, parent) if parent is not None else None
        if edges is not None and not len(edges):
            changes[name] = {"source": "section pivot", "distance_studs": 0.0,
                             "pivot_fallback": f"no shared cut edge with {PARENT[name]}; retained section pivot"}
            continue
        neighbours = [section] + ([parent] if parent is not None else [])
        pivot = section.pivot
        if all(np.all(pivot >= s.verts.min(0) - 1e-4) and np.all(pivot <= s.verts.max(0) + 1e-4)
               for s in neighbours):
            continue
        if parent is None:
            index = MeshIndex(section.verts, section.faces)
            face, bary, _ = index.closest(pivot[None])
            new = bary[0] @ section.verts[section.faces[face[0]]]
        else:
            a, b = section.verts[edges[:, 0]], section.verts[edges[:, 1]]
            d = b - a
            t = np.clip(((pivot - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-20), 0, 1)
            candidates = a + t[:, None] * d
            new = candidates[np.argmin(np.linalg.norm(candidates - pivot, axis=1))]
        changes[name] = {"source": "shared cut edge" if parent else "root section surface",
                         "distance_studs": round(float(np.linalg.norm(new - pivot)), 6)}
        section.pivot = new.copy()
    return changes


def _shared_edges(section, parent):
    """The section's mesh edges that also appear in its parent, matched on rounded vertex positions."""
    rounded_parent = np.round(parent.verts, 7)
    parent_edges = np.concatenate([parent.faces[:, [0, 1]], parent.faces[:, [1, 2]], parent.faces[:, [2, 0]]])
    common = {tuple(sorted((tuple(rounded_parent[a]), tuple(rounded_parent[b])))) for a, b in parent_edges}
    rounded = np.round(section.verts, 7)
    edges = np.concatenate([section.faces[:, [0, 1]], section.faces[:, [1, 2]], section.faces[:, [2, 0]]])
    shared = np.array([tuple(sorted((tuple(rounded[a]), tuple(rounded[b])))) in common for a, b in edges], bool)
    return edges[shared]
