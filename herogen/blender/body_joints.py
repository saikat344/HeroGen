"""Joints from SAM 3D Body (`--joints sam3d-body`), mapped into the contract frame and cross-checked.

SAM 3D Body sees `views/front.png` only. Its 2D keypoints give X and Z through the front view's own mapping
(`views.View`: u = 0 at the mask's bounding-box centre, v = 0 at the feet, one pixel scale), exactly as the
front mask does:

    X = (u - cmid) / pps,    Z = (vref - v) / pps

Named MHR70 keypoints give the limbs (shoulders, elbows, wrists, hips, knees, ankles). The MHR skeleton's
unnamed joints give the trunk: skeleton joint 113 (the head's root, below the skull) for Neck, 34 (first
spine joint above the pelvis) for Waist and 1 (pelvis) for Root, projected with the model's own camera
(uv = f * xy / z + c). HeadTop is the top of the projected body mesh.

Depth (Y) comes from one of two methods (`Y_METHODS`):
- `mesh`: the mid-depth of the stage-3 mesh along the front view's ray through (X, Z): halfway between the
  nearest and farthest surface (for an A-posed limb, the limb's centre).
- `mhr3d`: the MHR 3D keypoints levelled (neck over the ankles, hips along X), scaled to the body height, and
  placed so the pelvis sits at the mesh's mid-depth there.
The silhouette joint finder runs too, as fallback and cross-check; joints more than `FLAG_STUDS` apart are
flagged in `03-joints.json`.

Limb roots (`LIMB_ROOTS`: shoulders and hips) are cut where the outline separates the limb from the trunk: with
`limb_roots="silhouette"` (the default) their X/Z come from the silhouette finder and their Y from the same mesh
mid-depth; SAM 3D Body's own (anatomical) shoulders and hips are kept in the record. Measured on the pencil-sketch
figure (hybrid mesh from the pinned runners' results, roblox-ingame, gated poses, high / low LOD worst visible
gap, studs): every joint from SAM 3D Body 0.358 / 0.379 (shoulders in arms_up; moving the shoulder cut out to 2.5
arm radii 0.212 / 0.226), silhouette shoulders and hips with SAM 3D Body for the rest 0.284 / 0.282 (the left hip
in walk_a), all joints from the silhouette finder 0.373 / 0.408. The model's shoulder sits inside the costume's
shoulder pads, where no cut stays closed when the arm swings. On a source mesh recorded before the runners were
pinned the same three measured 0.411, 0.149 and 0.427 (high LOD): the gap follows the source mesh's run-to-run
variation at least as much as the choice of joints.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import raster
from .record import write_atomic
from .views import View

LIMBS = {
    "LeftShoulder": "left-shoulder", "RightShoulder": "right-shoulder",
    "LeftElbow": "left-elbow", "RightElbow": "right-elbow",
    "LeftWrist": "left-wrist", "RightWrist": "right-wrist",
    "LeftHip": "left-hip", "RightHip": "right-hip",
    "LeftKnee": "left-knee", "RightKnee": "right-knee",
    "LeftAnkle": "left-ankle", "RightAnkle": "right-ankle",
}
TRUNK = {"Neck": 113, "Waist": 34, "Root": 1}
JOINT_NAMES = ("HeadTop", "Neck", "Waist", "Root") + tuple(LIMBS)
Y_METHODS = ("mesh", "mhr3d")
LIMB_ROOTS = ("LeftShoulder", "RightShoulder", "LeftHip", "RightHip")
FLAG_STUDS = 0.5
HEADTOP_DEPTH_DROP = 0.15     # studs below the head top where HeadTop's depth is read


def image_points(body: dict) -> dict[str, np.ndarray]:
    """Joint name -> image (u, v) in pixels, from a SAM 3D Body `joints.json`."""
    kp = body["mhr70"]
    out = {name: np.asarray(kp[k]["uv"], float) for name, k in LIMBS.items()}
    f = float(body["focal_length"])
    c = np.asarray(body["principal_point"], float)
    sk = np.asarray(body["mhr_skeleton_joint_coords"], float)
    for name, j in TRUNK.items():
        p = sk[j]
        out[name] = f * p[:2] / p[2] + c
    out["HeadTop"] = np.asarray(body["mesh_top_uv"], float)
    return out


def front_xz(front: View, uv: np.ndarray) -> tuple[float, float]:
    return float((uv[0] - front.cmid) / front.pps), float((front.vref - uv[1]) / front.pps)


def depth_maps(front: View, verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest and farthest surface Y at every front-view pixel (inf where the mesh is absent)."""
    near = raster.render_view(front, verts, faces)
    flipped = verts.copy()
    flipped[:, 1] = -flipped[:, 1]
    far = raster.render_view(front, flipped, faces)
    return near.depth, np.where(np.isfinite(far.depth), -far.depth, np.inf)


def mid_depth(front: View, near: np.ndarray, far: np.ndarray, x: float, z: float) -> tuple[float, float]:
    """Mid-depth of the mesh on the ray through (x, z); the nearest covered pixel if the ray misses.
    Returns (y, distance in studs to the pixel used)."""
    row, col = front.uv_to_pixel(np.array([x]), np.array([z]))
    r, c = int(row[0]), int(col[0])
    ok = np.isfinite(near) & np.isfinite(far)
    if not ok.any():
        raise ValueError("the stage 3 mesh covers no pixel of the front view; no joint depth can be read")
    if not (0 <= r < ok.shape[0] and 0 <= c < ok.shape[1] and ok[r, c]):
        rr, cc = np.nonzero(ok)
        k = int(np.argmin((rr - r) ** 2 + (cc - c) ** 2))
        dist = math.hypot(rr[k] - r, cc[k] - c) / front.pps
        r, c = int(rr[k]), int(cc[k])
    else:
        dist = 0.0
    return float((near[r, c] + far[r, c]) / 2), dist


def _to_zup(p: np.ndarray) -> np.ndarray:
    """OpenCV camera (x right, y down, z forward) -> contract axes (X right, Y away from the viewer, Z up)."""
    p = np.atleast_2d(p)
    return np.stack([p[:, 0], p[:, 2], -p[:, 1]], axis=1)


def mhr_levelled(body: dict) -> dict[str, np.ndarray]:
    """MHR 3D joints levelled (neck over the ankle midpoint, hips along X), in metres; the caller scales them."""
    kp = {k: _to_zup(np.asarray(v["xyz"], float))[0] for k, v in body["mhr70"].items()}
    sk = _to_zup(np.asarray(body["mhr_skeleton_joint_coords"], float))
    up = kp["neck"] - 0.5 * (kp["left-ankle"] + kp["right-ankle"])
    pitch = math.atan2(up[1], up[2])
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    up1 = rx @ up
    roll = -math.atan2(up1[0], up1[2])
    ry = np.array([[math.cos(roll), 0, math.sin(roll)], [0, 1, 0], [-math.sin(roll), 0, math.cos(roll)]])
    r = ry @ rx
    hips = r @ (kp["left-hip"] - kp["right-hip"])
    yaw = -math.atan2(hips[1], hips[0])
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    r = rz @ r
    pts = {name: r @ kp[k] for name, k in LIMBS.items()}
    for name, j in TRUNK.items():
        pts[name] = r @ sk[j]
    return {"points": pts, "rotation_deg": [math.degrees(pitch), math.degrees(roll), math.degrees(yaw)]}


def from_body(body: dict, front: View, verts: np.ndarray, faces: np.ndarray, height: float,
              y_method: str = "mesh") -> tuple[dict, dict]:
    """Contract-frame joints and their evidence from a SAM 3D Body result."""
    if y_method not in Y_METHODS:
        raise ValueError(f"y_method must be one of {Y_METHODS}")
    uv = image_points(body)
    near, far = depth_maps(front, verts, faces)
    joints, evidence = {}, {}
    for name in JOINT_NAMES:
        x, z = front_xz(front, uv[name])
        # the top of the head is a sliver on the ray; take the head's depth a little below it
        y, miss = mid_depth(front, near, far, x, z - (HEADTOP_DEPTH_DROP if name == "HeadTop" else 0.0))
        joints[name] = np.array([x, y, z])
        src = (f"MHR70 '{LIMBS[name]}' 2D keypoint" if name in LIMBS else
               f"MHR skeleton joint {TRUNK[name]} projected" if name in TRUNK else "top of the projected MHR mesh")
        evidence[name] = src + "; X/Z through the front view mapping; Y mesh mid-depth" + (
            f" at the nearest covered pixel ({miss:.3f} studs away)" if miss else "")
    if y_method == "mhr3d":
        lev = mhr_levelled(body)
        pts = lev["points"]
        # scale: the image span from ankles to the head root, against the levelled 3D span
        img_span = joints["Neck"][2] - 0.5 * (joints["LeftAnkle"][2] + joints["RightAnkle"][2])
        a3 = 0.5 * (pts["LeftAnkle"] + pts["RightAnkle"])
        s = img_span / max(pts["Neck"][2] - a3[2], 1e-6)
        y_root = joints["Root"][1]
        for name in JOINT_NAMES:
            if name == "HeadTop":
                continue
            joints[name][1] = y_root + (pts[name][1] - pts["Root"][1]) * s
            evidence[name] = evidence[name].replace("Y mesh mid-depth", "Y from the levelled MHR 3D joints")
        joints["HeadTop"][1] = joints["Neck"][1]
        evidence["HeadTop"] = evidence["HeadTop"].replace("Y mesh mid-depth", "Y of the Neck")
    return joints, evidence


def limb_roots_from_silhouette(joints: dict, evidence: dict, silhouette: dict, front: View, verts: np.ndarray,
                               faces: np.ndarray) -> list[str]:
    """Replace the limb roots' X/Z with the silhouette finder's (Y: mesh mid-depth there). Returns the names."""
    near, far = depth_maps(front, verts, faces)
    for name in LIMB_ROOTS:
        x, _, z = (float(c) for c in silhouette[name])
        y, _ = mid_depth(front, near, far, x, z)
        joints[name] = np.array([x, y, z])
        evidence[name] = ("limb root cut where the outline separates the limb (silhouette finder X/Z); Y mesh "
                          "mid-depth; SAM 3D Body's anatomical joint kept under sources")
    return list(LIMB_ROOTS)


def cross_check(a: dict, b: dict) -> dict[str, float]:
    return {k: round(float(np.linalg.norm(np.asarray(a[k]) - np.asarray(b[k]))), 4) for k in a if k in b}


def write_joints(path: Path, source: str, joints: dict, evidence: dict, extra: dict | None = None) -> dict:
    rec = {"source": source, "units": "studs",
           "frame": "contract: +Z up, feet at z=0, faces -Y, x/y centred on the figure's bounding box",
           "joints": {k: [round(float(c), 4) for c in v] for k, v in joints.items()},
           "evidence": evidence}
    if extra:
        rec.update(extra)
    write_atomic(path, json.dumps(rec, indent=2))      # replaced in one step: never a partial file
    return rec


def load_joints(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def apply_to_skeleton(sk, rec: dict):
    """The silhouette skeleton with its joints replaced by a `03-joints.json` record's (radii, hand tips and the
    crotch height stay from the silhouette; the hip centre and toe X follow the new joints)."""
    joints = {k: np.asarray(v, float) for k, v in rec["joints"].items()}
    missing = [k for k in sk.joints if k not in joints]
    if missing:
        raise KeyError(f"03-joints.json lacks {missing}")
    sk.joints = joints
    sk.ends["HipCentre"] = (joints["LeftHip"] + joints["RightHip"]) / 2
    for side in ("Left", "Right"):
        toe = np.asarray(sk.ends[side + "Toe"], float).copy()
        toe[0] = joints[side + "Ankle"][0]
        toe[2] = min(toe[2], 0.4 * joints[side + "Ankle"][2])
        sk.ends[side + "Toe"] = toe
    sk.evidence = dict(rec.get("evidence", {}))
    sk.notes.append(f"joints from 03-joints.json (source {rec.get('source')})")
    return sk


def fit_skeleton(sk, rec: dict, front: View, verts: np.ndarray, faces: np.ndarray) -> dict:
    """Fit hybrid limb radii to the mesh being cut and reject incompatible silhouette root overrides.

    The silhouette finder supplies a complete skeleton for its own pose and proportions. Its radii must not
    weight a different Body skeleton's bone distances. Measure the limb depth at its Body elbow/knee instead.
    A shoulder override must leave the shoulder cut between the Body shoulder and elbow; a hip must stay below
    the waist and above its knee. Invalid overrides use the original Body joint, retained in stage 3's record.
    A coherent skeleton with every shared rim keeps its existing radii. Otherwise the measured radii repair
    the nearest-bone weights. No pose score or acceptance threshold chooses this fit. Carve skeletons do not
    use it.
    """
    from . import cuts, split

    sections, _ = split.build_sections(verts, faces, sk, cap="flat")
    missing = cuts.unmeasurable_sections(sections)
    near, far = depth_maps(front, verts, faces)
    model = rec.get("sources", {}).get("sam3d-body", rec["joints"])
    replaced, radii = {}, {}
    for side in ("Left", "Right"):
        for joint, kind in (("Elbow", "Arm"), ("Knee", "Leg")):
            q = sk.joints[side + joint]
            row, col = front.uv_to_pixel(q[:1], q[2:])
            r, c = int(row[0]), int(col[0])
            ok = np.isfinite(near) & np.isfinite(far)
            if not (0 <= r < ok.shape[0] and 0 <= c < ok.shape[1] and ok[r, c]):
                rr, cc = np.nonzero(ok)
                if not len(rr):
                    raise cuts.UnmeasurableJoints({side + joint: "reference mesh covers no front pixel"})
                i = int(np.argmin((rr - r) ** 2 + (cc - c) ** 2))
                r, c = int(rr[i]), int(cc[i])
            radius = float((far[r, c] - near[r, c]) / 2)
            if radius <= 0:
                raise cuts.UnmeasurableJoints({side + joint: "reference mesh has no depth at the joint"})
            radii[side + kind] = radius
        shoulder = side + "Shoulder"
        elbow = sk.joints[side + "Elbow"]
        original = np.asarray(model[shoulder], float)
        direction = elbow - original
        length = float(np.linalg.norm(direction))
        direction = direction / max(length, 1e-12)
        along = float((sk.joints[shoulder] - original) @ direction)
        if not 0 <= along < length - split.SHOULDER_CUT_OFFSET * radii[side + "Arm"]:
            replaced[shoulder] = sk.joints[shoulder].tolist()
            sk.joints[shoulder] = original.copy()
        hip = side + "Hip"
        up = sk.joints["Neck"] - sk.joints["Waist"]
        if ((sk.joints[hip] - sk.joints["Waist"]) @ up >= 0 or
                (sk.joints[hip] - sk.joints[side + "Knee"]) @ up <= 0):
            replaced[hip] = sk.joints[hip].tolist()
            sk.joints[hip] = np.asarray(model[hip], float).copy()
    # A coherent, measurable skeleton keeps its original proportions. Fitting only repairs invalid input;
    # it must not move a valid hip cut simply because another estimate of the radius is available.
    if replaced or missing:
        sk.radii.update(radii)
    sk.ends["HipCentre"] = (sk.joints["LeftHip"] + sk.joints["RightHip"]) / 2
    for name in replaced:
        sk.evidence[name] = "recorded SAM 3D Body joint; silhouette override would remove or reverse a section"
    return {"method": fit_skeleton.__doc__.strip(), "radii": {k: sk.radii[k] for k in radii},
            "measured_radii": radii, "replaced_roots": replaced, "unmeasurable_before": missing}
