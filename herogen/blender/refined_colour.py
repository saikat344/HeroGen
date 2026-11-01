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

# Float32 rotation/export round trips over source spans 0.2, 2 and 20 measured below 1e-7 of bbox diagonal.
# One part per million retains the rounding allowance and rejects a 1e-4-size displaced control.
FRAME_TOL_SIZE = 1e-6
SURFACE_SAMPLE_FACES = 4096


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _surface_samples(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Four barycentric points per face, area-stratified to at most 4096 faces."""
    tri = verts[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError("mesh has no finite surface area")
    count = min(len(tri), SURFACE_SAMPLE_FACES)
    selected = (np.arange(len(tri)) if len(tri) <= count else
                np.unique(np.searchsorted(np.cumsum(area), (np.arange(count) + .5) * area.sum() / count)))
    bary = np.array([[1/3, 1/3, 1/3], [.6, .2, .2], [.2, .6, .2], [.2, .2, .6]])
    return np.einsum("bk,mkj->mbj", bary, tri[selected]).reshape(-1, 3)


def _closest(index: MeshIndex, points: np.ndarray):
    batches = [index.closest(points[i:i + 256]) for i in range(0, len(points), 256)]
    return tuple(np.concatenate([b[k] for b in batches]) for k in range(3))


def _stats(distances: np.ndarray, size: float) -> dict:
    return {"samples": len(distances), "mean": float(distances.mean()),
            "p95": float(np.percentile(distances, 95)), "max": float(distances.max()),
            "size_fraction": {"mean": float(distances.mean() / size),
                              "p95": float(np.percentile(distances, 95) / size),
                              "max": float(distances.max() / size)}}


def load(run: Path, work: Path, height: float) -> tuple[texture.SourceColour | None, dict]:
    """A validated primary colour source, or a recorded reason to retain the view bake.

    Each raw source triangle maps linearly to the corresponding stage-3 NPZ triangle. Refined vertices and
    surface samples locate their nearest raw triangle and barycentrics independently of the refined topology.
    Geometry is checked in raw units against one millionth of the raw bbox diagonal; placement is recorded
    in studs. Height is retained in the call signature for stage-4 callers; it does not set the raw tolerance.
    """
    info = {"used": False, "reason": None, "refined_glb_sha256": None, "stage3f_source_sha256": None,
            "current_source_sha256": None, "source_correspondence": None}
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
        raw_source = sam3d_mesh.load_source(raw_path)
        raw, raw_faces = raw_source.verts, raw_source.faces
        del raw_source
        with np.load(uv_path) as data:
            target, faces = data["verts"], data["faces"]
        if raw.shape != target.shape or not len(raw) or not np.isfinite([raw, target]).all():
            return reject("raw/source-NPZ vertex correspondence is invalid")
        if not np.issubdtype(faces.dtype, np.integer) or not np.array_equal(raw_faces, faces):
            return reject("raw/source-NPZ faces do not match in order")
        size = float(np.linalg.norm(np.ptp(raw, axis=0)))
        if size <= 0 or not np.isfinite(size):
            return reject("raw source size is invalid")
        correspondence = info["source_correspondence"] = {
            "method": "piecewise-linear source correspondence", "vertices": len(raw), "faces": len(faces),
            "faces_match": True, "raw_size": size, "size_measure": "bbox diagonal",
            "threshold_raw": FRAME_TOL_SIZE * size, "threshold_size_fraction": FRAME_TOL_SIZE,
            "geometry_distance_raw": None, "placement_distance_studs": None}
        refined = sam3d_mesh.load_source(refined_path)
        if refined.texture is None:
            return reject("03-refined.glb has no base-colour texture")
        if not np.isfinite(refined.corner_uv).all():
            return reject("03-refined.glb has non-finite corner UVs")
        if not np.isfinite(refined.verts).all():
            return reject("03-refined.glb has non-finite positions")
        samples = _surface_samples(refined.verts, refined.faces)
        points = np.concatenate((refined.verts, samples))
        triangle, bary, distances = _closest(MeshIndex(raw, raw_faces), points)
        if not np.isfinite(distances).all():
            return reject("refined/raw source surface distance is not finite")
        correspondence["geometry_distance_raw"] = _stats(distances, size)
        correspondence["geometry_distance_raw"].update(vertices=len(refined.verts), surface_samples=len(samples))
        if distances.max() > correspondence["threshold_raw"]:
            return reject("refined/raw source surface distance exceeds threshold")
        target_tri = target[faces]
        area = np.linalg.norm(np.cross(target_tri[:, 1] - target_tri[:, 0],
                                       target_tri[:, 2] - target_tri[:, 0]), axis=1)
        # Relative roundoff bound on area, not a geometric deformation gate.
        target_size = float(np.linalg.norm(np.ptp(target, axis=0)))
        if not np.isfinite(area).all() or not np.isfinite(target_size):
            return reject("source-NPZ surface size or area is not finite")
        correspondence["placed_size_studs"] = target_size
        if (area[triangle] <= np.finfo(float).eps * target_size ** 2).any():
            return reject("degenerate source-NPZ triangle at refined points")
        mapped = np.einsum("ni,nij->nj", bary, target_tri[triangle])
        placed = mapped[:len(refined.verts)]
        index = MeshIndex(target, faces)
        placement = np.concatenate([index.distance(mapped[i:i + 256]) for i in range(0, len(mapped), 256)])
        if not np.isfinite(placement).all():
            return reject("placed/source-NPZ surface distance is not finite")
        correspondence["placement_distance_studs"] = _stats(placement, target_size)
        # Also expose interpolation across new refined edges spanning multiple source triangles.
        placed_samples = _surface_samples(placed, refined.faces)
        placed_distances = np.concatenate([index.distance(placed_samples[i:i + 256])
                                           for i in range(0, len(placed_samples), 256)])
        if not np.isfinite(placed_distances).all():
            return reject("placed refined mesh/source-NPZ surface distance is not finite")
        correspondence["placed_mesh_distance_studs"] = _stats(placed_distances, target_size)
        primary = texture.SourceColour(placed, refined.faces, refined.corner_uv, refined.texture)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        return reject(f"refined colour source could not be read ({type(exc).__name__})")
    info["used"] = True
    return primary, info
