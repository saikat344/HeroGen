"""Stage 3: base mesh by silhouette carving, checked by silhouette IoU against the masks."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import HEIGHT_STUDS, VIEWS, bl, carve, joints
from .record import StageRecord, check
from .views import load_mask, load_views

# Gate on the worst view's silhouette IoU. Measured on the synthetic humanoid (512 px masks, 440 px figure),
# worst view: 0.93 at 96 voxels tall, 0.955 at 128, 0.971 at 192 (the default), 0.972 at 256. Shifting a view
# is absorbed by the bounding-box alignment; one side view drawn 5% too large drops that view to 0.87. On the
# stage-2 stand-in (generated views that disagree a little) the observed views score 0.91-0.98. 0.90 passes
# honest disagreement and catches a view that is out of scale with the others.
IOU_GATE = 0.90
DEFAULT_NZ = 192


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def work_dir(run: Path) -> Path:
    return run / "blender-work"


def run_stage3(run: Path, height: float = HEIGHT_STUDS, nz: int = DEFAULT_NZ, timeout: float = 600.0) -> dict:
    rec = StageRecord(run, 3, "base mesh")
    vs = load_views(run, height)
    for n in vs.notes:
        rec.note(n)
    if vs.missing:
        rec.note("missing views skipped: " + ", ".join(vs.missing))
    try:
        skeleton = joints.estimate(vs, height)
    except Exception as exc:          # the prior is optional; the carve itself does not need joints
        skeleton = None
        rec.note(f"joint estimate failed ({exc}); arm depth prior skipped")
    verts, faces, info = carve.build_mesh(vs, height, nz, skeleton=skeleton)
    if info["detached_side_prior_voxels"]:
        rec.note(f"detached side-view parts (scarf tail, strap) removed from body columns: "
                 f"{info['detached_side_prior_voxels']} voxels removed")
    if info["arm_depth_prior_columns"]:
        rec.note(f"arm depth prior applied to {info['arm_depth_prior_columns']} voxel columns "
                 "(arms hang over the torso in the side views and no view constrains their depth)")
    work = work_dir(run)
    work.mkdir(parents=True, exist_ok=True)
    mesh_npz = work / "03-mesh.npz"
    np.savez(mesh_npz, verts=verts, faces=faces)
    renders = run / "03-renders"
    renders.mkdir(exist_ok=True)
    cams = [vs.views[v].camera() for v in VIEWS if v in vs.views]
    job = {"tasks": [{"task": "silhouettes", "mesh": str(mesh_npz.resolve()),
                      "glb": str((run / "03-mesh.glb").resolve()), "cameras": cams,
                      "out_dir": str(renders.resolve())}]}
    res = bl.run_job(job, work, "stage3", timeout=timeout)
    rec.data["tools"] = [f"blender {res['blender']}"]
    ious = {}
    for v in VIEWS:
        if v not in vs.views:
            continue
        rendered = load_mask(renders / f"{v}.png")
        ious[v] = round(iou(rendered, vs.views[v].mask), 4)
        # A view that stage 2 derived from the others (meta mask_method "visual_hull") is not an observation:
        # its IoU is reported, not gated (rounding unconstrained arms is expected to shrink it).
        method = str((vs.meta.get("mask_method") or {}).get(v, ""))
        derived = "visual_hull" in method
        rec.add(check(f"silhouette_iou_{v}", ious[v], IOU_GATE, gate=not derived,
                      **({"note": f"derived view ({method}); not gated"} if derived else {})))
    # front/back and left/right are mirror silhouettes, so only three directions are independent; the carve
    # needs at least the front-or-back and left-or-right pair.
    indep = sum(any(v in vs.views for v in grp) for grp in (("front", "back"), ("left", "right"), ("top",)))
    rec.add(check("independent_views", indep, 2, views=sorted(vs.views)))
    tri = int(len(faces))
    rec.add(check("triangles", tri, None, gate=False))
    e_all = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e_all, axis=0, return_counts=True)
    rec.add(check("open_edges", int((counts == 1).sum()), None, passed=bool((counts == 1).sum() == 0)))
    rec.data["outputs"] = ["03-mesh.glb"] + [f"03-renders/{v}.png" for v in VIEWS if v in vs.views]
    rec.data["mesh"] = {**info, "triangles": tri, "vertices": int(len(verts)), "height_studs": height,
                        "pixels_per_stud": round(vs.pps, 4), "figure_height_px": vs.height_px,
                        "views_missing": vs.missing, "iou": ious,
                        "frame": "studs, +Z up, feet at z=0, faces -Y, x/y centred on the figure's bounding box",
                        "method": "visual hull: voxel centre kept if inside every mask; arms rounded where no "
                                  "view constrains their depth; gaussian blur; surface nets; Taubin smoothing"}
    return rec.finish()
