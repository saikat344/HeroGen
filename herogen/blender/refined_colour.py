"""Validate and place stage 3f's colour source in the sectioned body's frame.

Only the raw source and its stage-3 NPZ share a vertex correspondence. The refined GLB may split and reorder
vertices at UV seams, so its frame is checked against source triangles, never by matching vertex indices.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import sam3d_mesh, texture
from .meshindex import MeshIndex

# Float32 rotation/export round trips over source spans 0.2, 2 and 20 measured at most 5.34e-8 of body height.
# One part per million allows rounding through Blender while rejecting a 1e-4-height displaced control.
FRAME_TOL_HEIGHT = 1e-6
SURFACE_SAMPLE_FACES = 4096


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(run: Path, work: Path, height: float) -> tuple[texture.SourceColour | None, dict]:
    """A validated primary colour source, or a recorded reason to retain the view bake.

    Fit an affine transform to the exact raw/source-NPZ correspondence. Check its maximum residual and the
    exact distance of deterministic refined surface samples to the NPZ's triangle surface, in studs. The
    sampling covers up to 4096 area-stratified faces, four points per face, plus the six vertex extrema.
    """
    info = {"used": False, "reason": None, "refined_glb_sha256": None, "stage3f_source_sha256": None,
            "current_source_sha256": None, "frame_fit": None, "surface_distance": None,
            "threshold_studs": FRAME_TOL_HEIGHT * height, "threshold_height_fraction": FRAME_TOL_HEIGHT}
    raw_path, refined_path = run / "03-source.glb", run / "03-refined.glb"
    uv_path, record_path = work / "03-source-uv.npz", run / "stage-3f.json"

    def reject(reason):
        info["reason"] = reason
        return None, info

    try:
        if refined_path.is_file():
            info["refined_glb_sha256"] = _sha256(refined_path)
        if not record_path.is_file():
            return reject("no stage-3f.json")
        record = json.loads(record_path.read_text())
        if not isinstance(record, dict):
            return reject("stage-3f.json is not an object")
        info["stage3f_source_sha256"] = record.get("source_sha256")
        if record.get("status") != "complete":
            return reject("stage 3f status is not complete")
        if not raw_path.is_file():
            return reject("no 03-source.glb")
        info["current_source_sha256"] = _sha256(raw_path)
        if info["stage3f_source_sha256"] != info["current_source_sha256"]:
            return reject("stage 3f source_sha256 does not match current 03-source.glb")
        if not refined_path.is_file():
            return reject("no 03-refined.glb")
        if not uv_path.is_file():
            return reject("no 03-source-uv.npz")
        # Do not retain the raw GLB's decoded image: only its vertex order is needed for the fit.
        raw = sam3d_mesh.load_source(raw_path).verts
        with np.load(uv_path) as data:
            target, faces = data["verts"], data["faces"]
        if raw.shape != target.shape or len(raw) < 4 or not np.isfinite([raw, target]).all():
            return reject("raw/source-NPZ vertex correspondence is invalid")
        origin = raw.mean(axis=0)
        design = np.column_stack((raw - origin, np.ones(len(raw))))
        affine, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank != 4 or np.linalg.matrix_rank(affine[:3]) != 3:
            return reject("raw/source-NPZ frame fit is singular")
        residual = np.linalg.norm(design @ affine - target, axis=1)
        info["frame_fit"] = {"method": "affine", "vertices": len(raw),
                             "rms": float(np.sqrt(np.mean(residual ** 2))), "max": float(residual.max()),
                             "linear": affine[:3].tolist(), "translation": (affine[3] - origin @ affine[:3]).tolist()}
        if residual.max() > info["threshold_studs"]:
            return reject("raw/source-NPZ frame-fit residual exceeds threshold")
        refined = sam3d_mesh.load_source(refined_path)
        if refined.texture is None:
            return reject("03-refined.glb has no base-colour texture")
        if not np.isfinite(refined.corner_uv).all():
            return reject("03-refined.glb has non-finite corner UVs")
        placed = (refined.verts - origin) @ affine[:3] + affine[3]
        if not np.isfinite(placed).all():
            return reject("03-refined.glb has non-finite positions")
        tri = placed[refined.faces]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        if not np.isfinite(area).all() or area.sum() <= 0:
            return reject("03-refined.glb has no finite surface area")
        count = min(len(tri), SURFACE_SAMPLE_FACES)
        if len(tri) <= count:
            selected = np.arange(len(tri))
        else:
            selected = np.unique(np.searchsorted(np.cumsum(area),
                                                (np.arange(count) + 0.5) * area.sum() / count))
        bary = np.array([[1/3, 1/3, 1/3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
        samples = np.einsum("bk,mkj->mbj", bary, tri[selected]).reshape(-1, 3)
        extrema = np.r_[placed.argmin(axis=0), placed.argmax(axis=0)]
        samples = np.concatenate((samples, placed[extrema]))
        index = MeshIndex(target, faces)
        distances = np.concatenate([index.distance(samples[i:i + 1024]) for i in range(0, len(samples), 1024)])
        if not np.isfinite(distances).all():
            return reject("refined/source-NPZ surface distance is not finite")
        info["surface_distance"] = {"samples": len(samples), "mean": float(distances.mean()),
                                    "p95": float(np.percentile(distances, 95)), "max": float(distances.max())}
        if distances.max() > info["threshold_studs"]:
            return reject("refined/source-NPZ surface distance exceeds threshold")
        primary = texture.SourceColour(placed, refined.faces, refined.corner_uv, refined.texture)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        return reject(f"refined colour source could not be read ({type(exc).__name__})")
    info["used"] = True
    return primary, info
