"""Exact orthographic cameras around the contract frame, and the frame itself.

World frame: +Z up, feet at Z = 0, the figure centred on its X/Y bounding box, facing -Y, its left at +X.
A camera at azimuth `a` (degrees, elevation 0) sits on the direction (sin a, -cos a, 0) and looks back at the
vertical axis, so azimuth 0 is the front view, 90 the left, 180 the back and 270 the right, with the same
image axes as `herogen/blender/views.py`:

    image right = (cos a, sin a, 0),   image up = +Z,   looking direction = (-sin a, cos a, 0)

Every camera shares one pixel scale and puts the world point (0, 0, height / 2) at the image centre. In
pixel-edge coordinates (column to the right, row down, pixel (r, c) covers [c, c+1) x [r, r+1)):

    col = (p . right) * ppu + size / 2,     row = size / 2 - (p . up - height / 2) * ppu

so the feet (Z = 0) sit on row size / 2 + height * ppu / 2 in every view.
"""
from __future__ import annotations

import math

import numpy as np

from . import BUILDING_AZIMUTHS, ELEVATION_DEG, EVALUATION_AZIMUTHS, HEIGHT_STUDS, view_name

# The figure fills this fraction of the image height (the synthetic runs use 440 px of 512), and at most this
# fraction of the width in any view; the scale is the larger one that satisfies both.
HEIGHT_FILL = 440.0 / 512.0
WIDTH_FILL = 0.94
CAMERA_DISTANCE = 50.0  # studs from the image-centre point; the clip range is 0.01 to 500
MIN_EVAL_SEPARATION_DEG = 10.0


def axes(azimuth_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, up, forward) unit vectors of the camera at this azimuth."""
    a = math.radians(azimuth_deg)
    right = np.array([math.cos(a), math.sin(a), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    forward = np.array([-math.sin(a), math.cos(a), 0.0])
    return right, up, forward


def angular_gap(a: float, b: float) -> float:
    """Smallest angle between two azimuths, in degrees."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def camera(azimuth: int, role: str, ppu: float, size: int, height: float = HEIGHT_STUDS) -> dict:
    """One camera, in the dict form `herogen/blender/blender_job.add_camera` reads, with role and mapping."""
    right, up, forward = axes(azimuth)
    centre = np.array([0.0, 0.0, height / 2.0])
    return {
        "view": view_name(azimuth),
        "azimuth_deg": int(azimuth),
        "elevation_deg": ELEVATION_DEG,
        "role": role,
        "projection": "orthographic",
        "right": right.round(12).tolist(), "up": up.tolist(), "forward": forward.round(12).tolist(),
        "location": (centre - forward * CAMERA_DISTANCE).round(12).tolist(),
        "image_centre_world": centre.tolist(),
        "ortho_scale": size / ppu,
        "resolution": int(size),
        "pixels_per_unit": ppu,
    }


def all_cameras(ppu: float, size: int, height: float = HEIGHT_STUDS) -> list[dict]:
    """Building views (the four contract views, then the four diagonals), then the evaluation views."""
    cams = [camera(a, "building", ppu, size, height) for a in BUILDING_AZIMUTHS]
    cams += [camera(a, "evaluation", ppu, size, height) for a in EVALUATION_AZIMUTHS]
    return cams


def project(points: np.ndarray, cam: dict) -> np.ndarray:
    """World points [N, 3] to continuous (col, row) pixel-edge coordinates [N, 2]."""
    p = np.asarray(points, dtype=np.float64)
    size, ppu = cam["resolution"], cam["pixels_per_unit"]
    centre = np.asarray(cam["image_centre_world"])
    u = (p - centre) @ np.asarray(cam["right"])
    v = (p - centre) @ np.asarray(cam["up"])
    return np.stack([u * ppu + size / 2.0, size / 2.0 - v * ppu], axis=1)


def pixel_rays(cam: dict, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ray origins [N, 3] through the centres of pixels (rows, cols), and the shared direction [3].

    Origins lie on the plane `CAMERA_DISTANCE` in front of the image-centre point; a hit's depth is measured
    from the plane through the image-centre point (see `depth_of`)."""
    size, ppu = cam["resolution"], cam["pixels_per_unit"]
    u = (np.asarray(cols, dtype=np.float64) + 0.5 - size / 2.0) / ppu
    v = (size / 2.0 - (np.asarray(rows, dtype=np.float64) + 0.5)) / ppu
    right, up, fwd = (np.asarray(cam[k], dtype=np.float64) for k in ("right", "up", "forward"))
    centre = np.asarray(cam["image_centre_world"], dtype=np.float64)
    origins = centre + u[:, None] * right + v[:, None] * up - fwd * CAMERA_DISTANCE
    return origins, fwd


def depth_of(points: np.ndarray, cam: dict) -> np.ndarray:
    """Signed distance along the looking direction from the plane through the image-centre point (studs);
    negative is nearer the camera."""
    centre = np.asarray(cam["image_centre_world"])
    return (np.asarray(points, dtype=np.float64) - centre) @ np.asarray(cam["forward"])


def choose_ppu(verts: np.ndarray, size: int, height: float = HEIGHT_STUDS,
               azimuths: tuple[int, ...] = BUILDING_AZIMUTHS + EVALUATION_AZIMUTHS) -> float:
    """One pixel scale for every view: the figure is `HEIGHT_FILL` of the image tall unless its widest
    view would pass `WIDTH_FILL` of the width, in which case the scale shrinks so that view fits."""
    v = np.asarray(verts, dtype=np.float64)
    half = max(float(np.abs(v @ axes(a)[0]).max()) for a in azimuths)
    ppu = HEIGHT_FILL * size / height
    if half > 0:
        ppu = min(ppu, WIDTH_FILL * size / (2.0 * half))
    return round(ppu, 4)


def contract_matrix(bounds_min, bounds_max, height: float = HEIGHT_STUDS) -> np.ndarray:
    """4x4 uniform scale + translation taking a figure with these world bounds into the contract frame: X/Y
    bounding-box centre on the Z axis, lowest point at Z = 0, total height `height`."""
    lo, hi = np.asarray(bounds_min, dtype=np.float64), np.asarray(bounds_max, dtype=np.float64)
    extent = hi[2] - lo[2]
    if not extent > 0:
        raise ValueError("the figure has no height")
    s = height / extent
    m = np.eye(4)
    m[:3, :3] *= s
    m[:3, 3] = -s * np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]])
    return m


def yaw_matrix(degrees: float) -> np.ndarray:
    """4x4 rotation about +Z (counter-clockwise seen from above)."""
    a = math.radians(degrees)
    m = np.eye(4)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = math.cos(a), -math.sin(a), math.sin(a), math.cos(a)
    return m
