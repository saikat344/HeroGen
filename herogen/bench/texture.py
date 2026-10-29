"""Reading a texture's alpha the way Cycles does (pure Python + numpy, so it also runs inside Blender).

The masks clip at the same texels as the renders only if they sample the same values: texel centres at
half-integers, bilinear weights for "Linear" and the containing texel for "Closest", and the image node's
extension mode at the borders ("CLIP" reads as transparent outside the image).
"""
from __future__ import annotations

import math

import numpy as np

INTERPOLATIONS = ("Linear", "Closest")
EXTENSIONS = ("REPEAT", "EXTEND", "CLIP", "MIRROR")


def wrap(i: int, n: int, extension: str) -> tuple[int, bool]:
    """Texel index `i` folded into 0..n-1 by the extension mode, and whether it lies inside the image."""
    if extension == "REPEAT":
        return i % n, True
    if extension == "MIRROR":
        p = i % (2 * n)
        return (p if p < n else 2 * n - 1 - p), True
    if extension == "EXTEND":
        return min(max(i, 0), n - 1), True
    if extension == "CLIP":
        return i, 0 <= i < n
    raise ValueError(f"unsupported extension {extension!r} (use {', '.join(EXTENSIONS)})")


def sample(values: np.ndarray, u: float, v: float, interpolation: str = "Linear",
           extension: str = "REPEAT") -> float:
    """Value of a [rows, cols] texture at UV (u, v); row 0 is v = 0 (Blender's pixel order)."""
    h, w = values.shape
    if interpolation == "Closest":
        (ix, okx), (iy, oky) = wrap(math.floor(u * w), w, extension), wrap(math.floor(v * h), h, extension)
        return float(values[iy, ix]) if okx and oky else 0.0
    if interpolation != "Linear":
        raise ValueError(f"unsupported interpolation {interpolation!r} (use {', '.join(INTERPOLATIONS)})")
    fx, fy = u * w - 0.5, v * h - 0.5
    x0, y0 = math.floor(fx), math.floor(fy)
    tx, ty = fx - x0, fy - y0
    total = 0.0
    for dx, wx in ((0, 1.0 - tx), (1, tx)):
        ix, okx = wrap(x0 + dx, w, extension)
        for dy, wy in ((0, 1.0 - ty), (1, ty)):
            iy, oky = wrap(y0 + dy, h, extension)
            if okx and oky:
                total += wx * wy * float(values[iy, ix])
    return total


def barycentric_uv(a, b, c, point, uv_a, uv_b, uv_c) -> np.ndarray:
    """UV at `point` on triangle (a, b, c) with corner UVs, by barycentric weights; the centroid's UV for a
    degenerate triangle."""
    a, b, c, p = (np.asarray(x, dtype=np.float64) for x in (a, b, c, point))
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    if den == 0.0:
        w1 = w2 = 1.0 / 3.0
    else:
        w1 = (d11 * d20 - d01 * d21) / den
        w2 = (d00 * d21 - d01 * d20) / den
    return (1.0 - w1 - w2) * np.asarray(uv_a) + w1 * np.asarray(uv_b) + w2 * np.asarray(uv_c)
