"""Orthographic cameras shared by the Blender renders and the back-projection.

World coordinates are Blender's after glTF import (Z up, the body faces -Y). A camera at `angle` degrees looks
along forward = (sin a, cos a, 0) with right = (cos a, -sin a, 0) and up = +Z; `ortho` spans the larger image side
(Blender's automatic sensor fit). Pixel coordinates are edge coordinates: pixel (0, 0) spans [0, 1) x [0, 1).
"""
import math
import os

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

POS_LO, POS_EXT = -2.0, 4.0


def camera(name, centre, angle, ortho, res):
    return {"name": name, "centre": [float(v) for v in centre], "angle": float(angle), "ortho": float(ortho),
            "res": [int(res[0]), int(res[1])]}


def axes(cam):
    a = math.radians(cam["angle"])
    right = np.array([math.cos(a), -math.sin(a), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    fwd = np.array([math.sin(a), math.cos(a), 0.0])
    return right, up, fwd


def project(cam, p):
    """World points [..., 3] -> pixel edge coordinates (x, y) and depth along the view direction."""
    right, up, fwd = axes(cam)
    w, h = cam["res"]
    k = max(w, h) / cam["ortho"]
    d = p - np.asarray(cam["centre"])
    return w / 2 + (d @ right) * k, h / 2 - (d @ up) * k, d @ fwd


def px_per_unit(cam):
    return max(cam["res"]) / cam["ortho"]


def read_exr(path):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    img = img[..., [2, 1, 0, 3]] if img.shape[-1] == 4 else img[..., ::-1]
    return img.astype(np.float32)


def decode_pos(exr):
    """Position pass/map -> world positions and a validity mask (alpha, or non-zero)."""
    rgb = exr[..., :3]
    valid = exr[..., 3] > 0.5 if exr.shape[-1] == 4 else np.abs(rgb).sum(-1) > 1e-6
    return rgb * POS_EXT + POS_LO, valid
