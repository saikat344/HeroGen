"""Stage 3: hybrid body and joints, recorded carve fallback, and multiview visual acceptance."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import HEIGHT_STUDS, VIEWS, bl, carve, joints
from .record import StageRecord, check
from .views import load_mask, load_views

# Historical observed-view threshold retained only for silhouette diagnostics.
IOU_GATE = 0.90
DEFAULT_NZ = 192


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def carve_view_ious(vs, height: float = HEIGHT_STUDS, nz: int = DEFAULT_NZ) -> dict[str, float]:
    """Each view's silhouette IoU against the carve stage 3 builds from ``vs`` (same joints, arm prior, carve,
    smoothing and resolution), with the mesh rasterised in numpy (`raster`) instead of rendered by Blender.
    On the same mesh the two renders agree: six stage-3 runs measured with Blender were reproduced to 1e-4 in
    every view. Stage 2 uses this to keep only sheets whose carve passes this stage's IoU gate."""
    from . import raster

    try:
        skeleton = joints.estimate(vs, height)
    except Exception:  # noqa: BLE001  (as in run_stage3: the prior is optional, the carve does not need it)
        skeleton = None
    verts, faces, _ = carve.build_mesh(vs, height, nz, skeleton=skeleton)
    return {name: round(iou(raster.render_view(v, verts, faces).mask, v.mask), 4) for name, v in vs.views.items()}


def work_dir(run: Path) -> Path:
    return run / "blender-work"


MESH_SOURCES = ("carve", "sam3d-objects")
JOINT_SOURCES = ("silhouette", "sam3d-body")


def _remote(run: Path, remote):
    from .remote.client import RemoteClient

    if remote is None:
        return RemoteClient(run.parent / "_cache-remote", mode="replay")
    return remote


def _replace(path: Path, write) -> None:
    """Write `path` through a temporary file in the same folder, so a failure never leaves a partial file."""
    tmp = path.with_name(path.name + ".part")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def error_text(exc: BaseException, run: Path) -> str:
    """An error as a record note or message: one line, its type (unless it is a remote error, which says what
    failed) and its text, with no local paths."""
    from .remote.client import RemoteError

    text = str(exc) if isinstance(exc, RemoteError) else f"{type(exc).__name__}: {exc}"
    for p, name in ((str(run.resolve()), run.name), (str(run), run.name), (str(Path.home()), "~")):
        if len(p) > 1:
            text = text.replace(p, name)
    return " ".join(text.split())


def _remote_source(name: str, run: Path, fn, *args, **kwargs):
    """Run one remote source. Any failure, of the call itself, of its cached output (a corrupt GLB, a missing
    file) or of mapping its result (an empty mask, a degenerate mesh), becomes a `RemoteError` naming the
    source, so the caller falls back or stops cleanly."""
    from .remote.client import RemoteError

    try:
        return fn(*args, **kwargs)
    except RemoteError:
        raise
    except Exception as exc:  # noqa: BLE001 - every failure of an optional source is reported, never a crash
        raise RemoteError(f"{name}: {error_text(exc, run)}") from exc


def _account(rec: StageRecord, model: str, res) -> None:
    """A remote call in the record (the model and any new spend), as soon as the call returns, so a later
    failure of the source still shows what was called and spent."""
    rec.data["models"].append(model)
    est = res.entry.get("cost_usd_estimate")
    if not res.cached and est is None:
        rec.note(f"{model}: the cost of this call is unknown (no list price for GPU {res.entry.get('gpu')})")
    rec.data["cost_usd"] = round(rec.data["cost_usd"] + (0.0 if res.cached or est is None else float(est)), 4)
    if model == "sam-3d-objects":
        rec.data["cost_basis"] = res.entry.get("cost_basis", rec.data["cost_basis"]) + (
            "; replayed from the cache (no new spend)" if res.cached else "")


def _derived(vs, view: str) -> bool:
    """True when stage 2 derived `view` from the other masks (meta mask_method "visual_hull"): not an
    observation."""
    return "visual_hull" in str((vs.meta.get("mask_method") or {}).get(view, ""))


def _objects_mesh(run: Path, vs, height: float, remote, rec: StageRecord):
    """SAM 3D Objects on the front view and its mask, the mesh put in the contract frame. Returns the mesh, its
    record and the remote result; files are written only once everything succeeded (`03-source.glb` by the
    caller when the source is used)."""
    from PIL import Image

    from . import sam3d_mesh

    res = remote.objects(run / "views" / "front.png", run / "masks" / "front.png")
    _account(rec, "sam-3d-objects", res)
    src = sam3d_mesh.load_source(res.path("mesh.glb"))
    with Image.open(run / "views" / "front.png") as im:
        img = np.asarray(im.convert("RGB"), dtype=np.float64)
    sides, excluded = sam3d_mesh.pose_views(vs)
    src_verts, info = sam3d_mesh.orient(src, vs.views["front"], img, height, sides=sides,
                                        sides_excluded=excluded)
    verts, faces, info["remesh"] = sam3d_mesh.remesh(src_verts, src.faces, height)
    depth_views = {v: vs.views[v] for v in sam3d_mesh.DEPTH_VIEWS if v in vs.views and not _derived(vs, v)}
    dy, info["depth_register"] = sam3d_mesh.register_depth(verts, faces, depth_views)
    shift = np.array([0.0, dy, 0.0])
    verts, src_verts = verts + shift, src_verts + shift      # the texture fallback's source moves with the mesh
    work = work_dir(run)
    work.mkdir(parents=True, exist_ok=True)

    def save_npz(tmp: Path) -> None:
        with open(tmp, "wb") as fh:
            np.savez(fh, verts=src_verts, faces=src.faces, corner_uv=src.corner_uv)

    _replace(work / "03-source-uv.npz", save_npz)
    if src.texture is not None:
        _replace(work / "03-source-texture.png",
                 lambda tmp: Image.fromarray(src.texture.astype(np.uint8)).save(tmp, format="PNG"))
    run_log = res.entry.get("run", {})
    job = (run_log.get("runs") or {}).get("job", {})
    info["remote"] = {"cached": res.cached, "key": res.dir.name, "gpu": run_log.get("gpu"),
                      "inference_seconds": job.get("seconds"), "load_seconds": run_log.get("load_seconds"),
                      "wall_seconds": res.entry.get("wall_seconds"),
                      "vram_peak_mib": run_log.get("vram_peak_mib_nvidia_smi"),
                      "source_triangles": job.get("glb_faces"), "texture_size": job.get("texture_size"),
                      "renderer": run_log.get("renderer"), "bake_backend": run_log.get("bake_backend"),
                      "cost_usd_estimate": res.entry.get("cost_usd_estimate")}
    return verts, faces, info, res


BODY_BOX_SOURCE = "front mask bounding box (stage 2)"


def body_box(vs) -> list:
    """The front mask's bounding box [x0, y0, x1, y1] (pixel edges): SAM 3D Body's person box. Stage 2's mask
    already isolates the figure, so by default no person detector runs."""
    from .views import bbox as mask_bbox

    r0, r1, c0, c1 = mask_bbox(vs.views["front"].mask)
    return [c0, r0, c1 + 1, r1 + 1]


def _body_joints(run: Path, vs, height: float, verts, faces, remote, rec: StageRecord, skeleton,
                 y_method: str = "mesh", limb_roots: str = "silhouette", body_detector: str | None = None) -> dict:
    from . import body_joints as bj

    box = body_box(vs)
    res = remote.body(run / "views" / "front.png", box, BODY_BOX_SOURCE, detector=body_detector)
    _account(rec, "sam-3d-body", res)
    body = json.loads(res.path("joints.json").read_text())
    pts, evidence = bj.from_body(body, vs.views["front"], verts, faces, height, y_method=y_method)
    model_pts = {k: v.copy() for k, v in pts.items()}
    joint_sources = {k: "sam3d-body" for k in pts}
    if limb_roots == "silhouette" and skeleton is not None:
        for k in bj.limb_roots_from_silhouette(pts, evidence, skeleton.as_json(), vs.views["front"], verts, faces):
            joint_sources[k] = "silhouette (X/Z), mesh (Y)"
    alt = bj.from_body(body, vs.views["front"], verts, faces, height,
                       y_method=[m for m in bj.Y_METHODS if m != y_method][0])[0]
    run_log = res.entry.get("run", {})
    job = (run_log.get("runs") or {}).get("job", {})
    extra = {"y_method": y_method, "limb_roots": limb_roots, "joint_sources": joint_sources,
             "joint_method": bj.__doc__.strip(),
             "remote": {"cached": res.cached, "key": res.dir.name, "gpu": run_log.get("gpu_name") or run_log.get("gpu"),
                        "device": run_log.get("device"), "inference_seconds": job.get("seconds"),
                        "person_box": box, "box_source": job.get("box") or BODY_BOX_SOURCE,
                        "detector": body_detector,
                        "code": {k: res.entry.get("request", {}).get(k) for k in ("rev", "moge", "dinov3")},
                        "load_seconds": run_log.get("load_seconds"), "wall_seconds": res.entry.get("wall_seconds"),
                        "vram_peak_mib": run_log.get("vram_peak_mib_nvidia_smi"),
                        "torch_max_allocated_mib": job.get("torch_max_allocated_mib"),
                        "cost_usd_estimate": res.entry.get("cost_usd_estimate")},
             "sources": {"sam3d-body": {k: [round(float(c), 4) for c in v] for k, v in model_pts.items()},
                         "sam3d-body-y-" + [m for m in bj.Y_METHODS if m != y_method][0]:
                             {k: [round(float(c), 4) for c in v] for k, v in alt.items()}}}
    if skeleton is not None:
        sil = skeleton.as_json()
        dist = bj.cross_check(model_pts, sil)
        extra["sources"]["silhouette"] = sil
        extra["silhouette_evidence"] = skeleton.evidence
        extra["distance_to_silhouette"] = dist
        extra["flag_threshold_studs"] = bj.FLAG_STUDS
        extra["flagged"] = sorted(k for k, d in dist.items() if d > bj.FLAG_STUDS)
    return bj.write_joints(run / "03-joints.json", "sam3d-body", pts, evidence, extra)


def run_stage3(run: Path, height: float = HEIGHT_STUDS, nz: int = DEFAULT_NZ, timeout: float = 600.0,
               mesh_source: str = "sam3d-objects", joints_source: str = "sam3d-body", remote=None,
               fallback: bool = True, y_method: str = "mesh", limb_roots: str = "silhouette",
               body_detector: str | None = None, quality_client=None) -> dict:
    """Build the hybrid by default; a failed hybrid build retries the local carve once, with a reason.

    Quality rejection is a completed build and never triggers another constructor. Spend from a failed
    attempt remains in the same record. ``fallback=False`` preserves the original build error.
    """
    if mesh_source not in MESH_SOURCES or joints_source not in JOINT_SOURCES:
        raise ValueError(f"mesh_source in {MESH_SOURCES}, joints_source in {JOINT_SOURCES}")
    (run / "stage-3.json").unlink(missing_ok=True)
    rec = StageRecord(run, 3, "base mesh")
    rec.data.update(mesh_source_requested=mesh_source, joints_source_requested=joints_source, fallback=None)
    kwargs = dict(height=height, nz=nz, timeout=timeout, remote=remote, y_method=y_method,
                  limb_roots=limb_roots, body_detector=body_detector, rec=rec, quality_client=quality_client)
    from .remote.client import RemoteError

    try:
        return _run_stage3(run, mesh_source=mesh_source, joints_source=joints_source,
                           fallback=fallback and mesh_source == "carve", **kwargs)
    except (RemoteError, bl.BlenderError, ModuleNotFoundError, ValueError, OSError) as exc:
        if "body_acceptance" in rec.data:
            from .body_acceptance import BodyQualityError

            # Storage errors after acceptance starts must not masquerade as construction failures.
            raise BodyQualityError(f"Body acceptance stopped: {error_text(exc, run)}") from exc
        if not fallback or mesh_source != "sam3d-objects":
            raise
        reason = error_text(exc, run)
        rec.note(f"hybrid build failed ({reason}); fell back to carve and silhouette joints")
        rec.data["fallback"] = {"from": "sam3d-objects", "to": "carve", "reason": reason}
        rec.data["checks"] = []
        return _run_stage3(run, mesh_source="carve", joints_source="silhouette", fallback=False, **kwargs)


def _run_stage3(run: Path, height: float = HEIGHT_STUDS, nz: int = DEFAULT_NZ, timeout: float = 600.0,
               mesh_source: str = "carve", joints_source: str = "silhouette", remote=None,
               fallback: bool = True, y_method: str = "mesh", limb_roots: str = "silhouette",
               body_detector: str | None = None, rec: StageRecord | None = None, quality_client=None) -> dict:
    """Stage 3. `mesh_source` "carve" (default) or "sam3d-objects"; `joints_source` "silhouette" (default) or
    "sam3d-body". A remote source that fails falls back to the local one (recorded) unless `fallback` is off.
    `body_detector` "vitdet" lets SAM 3D Body find the person itself (default None: the front mask's box)."""
    from . import body_joints as bj
    from .remote.client import RemoteError

    if mesh_source not in MESH_SOURCES or joints_source not in JOINT_SOURCES:
        raise ValueError(f"mesh_source in {MESH_SOURCES}, joints_source in {JOINT_SOURCES}")
    if mesh_source == "sam3d-objects":
        import importlib.util

        if importlib.util.find_spec("scipy") is None:
            raise ModuleNotFoundError("mesh source sam3d-objects needs scipy (pip install \".[remote]\")")
    # An earlier record must not outlive a run that stops: stage 4 reads only a record this run finished.
    (run / "stage-3.json").unlink(missing_ok=True)
    rec = rec if rec is not None else StageRecord(run, 3, "base mesh")
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
    for stale in ("03-source.glb", "03-joints.json"):
        if (run / stale).exists():
            (run / stale).unlink()
    work = work_dir(run)
    for stale in ("03-source-uv.npz", "03-source-texture.png"):
        (work / stale).unlink(missing_ok=True)
    used_source, source_info, objects_res = mesh_source, None, None
    if mesh_source == "sam3d-objects":
        try:
            verts, faces, source_info, objects_res = _remote_source("sam3d-objects", run, _objects_mesh, run, vs,
                                                                    height, _remote(run, remote), rec)
        except RemoteError as exc:
            if not fallback:
                raise
            rec.note(f"sam3d-objects failed ({error_text(exc, run)[:200]}); fell back to the carve")
            used_source = "carve"
    if used_source == "carve":
        for stale in ("03-source-uv.npz", "03-source-texture.png"):     # a source that failed part-way
            (work / stale).unlink(missing_ok=True)
        verts, faces, info = carve.build_mesh(vs, height, nz, skeleton=skeleton)
        if info["detached_side_prior_voxels"]:
            rec.note(f"detached side-view parts (scarf tail, strap) removed from body columns: "
                     f"{info['detached_side_prior_voxels']} voxels removed")
        if info["arm_depth_prior_columns"]:
            rec.note(f"arm depth prior applied to {info['arm_depth_prior_columns']} voxel columns "
                     "(arms hang over the torso in the side views and no view constrains their depth)")
        method = ("visual hull: voxel centre kept if inside every mask; arms rounded where no view constrains "
                  "their depth; gaussian blur; surface nets; Taubin smoothing")
        shape_views = sorted(v for v in vs.views if not _derived(vs, v))
        pose_views = []
    else:
        from . import sam3d_mesh

        info = {"source": source_info}
        method = sam3d_mesh.__doc__.strip()
        shape_views = ["front"]
        pose = source_info.get("pose_fit") or {}
        pose_views = pose.get("views", []) if pose.get("status") == "fitted" else []
    work.mkdir(parents=True, exist_ok=True)
    mesh_npz = work / "03-mesh.npz"

    def save_mesh(tmp: Path) -> None:
        with open(tmp, "wb") as fh:
            np.savez(fh, verts=verts, faces=faces)

    _replace(mesh_npz, save_mesh)
    if objects_res is not None and used_source == "sam3d-objects":
        import shutil

        _replace(run / "03-source.glb", lambda tmp: shutil.copyfile(objects_res.path("mesh.glb"), tmp))
    joints_used = "silhouette"
    if joints_source == "sam3d-body":
        try:
            jrec = _remote_source("sam3d-body", run, _body_joints, run, vs, height, verts, faces,
                                  _remote(run, remote), rec, skeleton, y_method=y_method, limb_roots=limb_roots,
                                  body_detector=body_detector)
            joints_used = "sam3d-body"
            if jrec.get("flagged"):
                rec.note(f"sam3d-body joints more than {bj.FLAG_STUDS} studs from the silhouette finder: "
                         + ", ".join(jrec["flagged"]))
        except RemoteError as exc:
            if not fallback:
                raise
            (run / "03-joints.json").unlink(missing_ok=True)
            rec.note(f"sam3d-body failed ({error_text(exc, run)[:200]}); fell back to the silhouette joints")
    if joints_used == "silhouette" and skeleton is not None:
        bj.write_joints(run / "03-joints.json", "silhouette", skeleton.joints, skeleton.evidence)
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
        # All observed IoUs remain diagnostics, with construction/registration provenance.
        method_v = str((vs.meta.get("mask_method") or {}).get(v, ""))
        derived = _derived(vs, v)
        unseen = v not in shape_views and not derived
        posed = "; its pitch was registered on this view" if v in pose_views else ""
        note = ({"note": f"derived view ({method_v}); not gated"} if derived else
                {"note": f"shape not built from this view (mesh source {used_source} saw only "
                         f"{', '.join(shape_views)}){posed}"} if unseen else {})
        rec.add(check(f"silhouette_iou_{v}", ious[v], IOU_GATE, gate=False, **note))
    if used_source == "carve":
        # front/back and left/right are mirror silhouettes, so only three directions are independent; the carve
        # needs at least the front-or-back and left-or-right pair.
        indep = sum(any(v in vs.views for v in grp) for grp in (("front", "back"), ("left", "right"), ("top",)))
        rec.add(check("independent_views", indep, 2, views=sorted(vs.views)))
    else:
        rec.add(check("front_view_present", int("front" in vs.views), 1))
    tri = int(len(faces))
    rec.add(check("triangles", tri, None, gate=False))
    e_all = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e_all, axis=0, return_counts=True)
    rec.add(check("open_edges", int((counts == 1).sum()), None, passed=bool((counts == 1).sum() == 0)))
    rec.add(check("non_manifold_edges", int((counts > 2).sum()), None, gate=False,
                  note="edges shared by more than two triangles"))
    rec.data["outputs"] = (["03-mesh.glb"] + (["03-source.glb"] if used_source == "sam3d-objects" else [])
                           + (["03-joints.json"] if (run / "03-joints.json").exists() else [])
                           + [f"03-renders/{v}.png" for v in VIEWS if v in vs.views])
    rec.data["mesh_source"] = used_source
    rec.data.setdefault("mesh_source_requested", mesh_source)
    rec.data["joints_source"] = joints_used
    rec.data["shape_views"] = shape_views
    rec.data["pose_views"] = pose_views
    rec.data["mesh"] = {**info, "triangles": tri, "vertices": int(len(verts)), "height_studs": height,
                        "pixels_per_stud": round(vs.pps, 4), "figure_height_px": vs.height_px,
                        "views_missing": vs.missing, "iou": ious,
                        "frame": "studs, +Z up, feet at z=0, faces -Y, " + (
                            "x/y centred on the figure's bounding box" if used_source == "carve" else
                            "x centred on the mesh's bounding box (its front outline conformed to the front mask), "
                            "y registered on the observed depth views"),
                        "method": method}
    from .body_acceptance import assess

    if all(c["pass"] for c in rec.data["checks"] if c.get("gate", True)):
        assess(run, verts, faces, rec, client=quality_client)
    else:
        rec.note("visual acceptance skipped because the mesh failed structural checks")
    rec.data["rejection_reasons"] = [
        f"{c['name']}: {c.get('reason', 'required structural check failed')}"
        for c in rec.data["checks"] if c.get("gate", True) and not c["pass"]]
    return rec.finish()
