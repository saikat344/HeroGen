"""Orthographic views of the contract: mask loading and the pixel <-> world mapping.

World frame (Blender): +Z up, feet at Z = 0, the character faces -Y, its left is +X.

Each view maps world points to image coordinates (u to the image right, v to the image up):

    front: u = X,  v = Z        back:  u = -X, v = Z
    left:  u = Y,  v = Z        right: u = -Y, v = Z
    top:   u = X,  v = Y   (front of the character at the bottom of the image)

The contract centres the figure in every view and uses one pixel scale. Instead of trusting the exact padding,
each mask is aligned by its own bounding box: u = 0 at the box centre, and v = 0 at the feet (bottom edge) for
the four side views or at the box centre for the top view. Every axis is then measured the same way in each
view that sees it (the X extent in front and top is the same projection), so the views agree to within a
pixel whatever the padding. The resulting frame is centred on the figure's X/Y bounding box, feet at Z = 0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import VIEWS

# Rows of each view's 2x3 world -> (u, v) projection.
PROJ = {
    "front": np.array([[1.0, 0, 0], [0, 0, 1.0]]),
    "back": np.array([[-1.0, 0, 0], [0, 0, 1.0]]),
    "left": np.array([[0, 1.0, 0], [0, 0, 1.0]]),
    "right": np.array([[0, -1.0, 0], [0, 0, 1.0]]),
    "top": np.array([[1.0, 0, 0], [0, 1.0, 0]]),
}
# Camera looking direction in the world for each view.
FORWARD = {
    "front": np.array([0, 1.0, 0]), "back": np.array([0, -1.0, 0]),
    "left": np.array([-1.0, 0, 0]), "right": np.array([1.0, 0, 0]),
    "top": np.array([0, 0, -1.0]),
}


def load_mask(path: Path) -> np.ndarray:
    """A mask PNG (white figure on black, any PIL mode) as a bool array, True on the figure."""
    from PIL import Image

    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA"):
            a = np.asarray(im.convert("RGBA"), dtype=np.uint8)
            lum = a[..., :3].mean(axis=2) * (a[..., 3] / 255.0)
        else:
            lum = np.asarray(im.convert("L"), dtype=np.float64)
    return lum > 127.5


def save_mask(path: Path, mask: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


@dataclass
class View:
    name: str
    mask: np.ndarray            # bool [rows, cols]
    pps: float                  # pixels per stud (after scaling to the contract height)
    cmid: float                 # bbox centre column (pixel-edge coordinates)
    vref: float                 # row edge where v = 0 (feet for side views, bbox centre for top)

    @property
    def size(self) -> int:
        return int(self.mask.shape[0])

    def uv_to_pixel(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Integer (row, col) of the pixel containing each (u, v) point."""
        col = np.floor(u * self.pps + self.cmid).astype(np.int64)
        row = np.floor(self.vref - v * self.pps).astype(np.int64)
        return row, col

    def pixel_uv(self) -> tuple[np.ndarray, np.ndarray]:
        """(u, v) of every pixel centre, as two [rows, cols] arrays."""
        n_r, n_c = self.mask.shape
        cols = (np.arange(n_c) + 0.5 - self.cmid) / self.pps
        rows = (self.vref - (np.arange(n_r) + 0.5)) / self.pps
        return np.meshgrid(cols, rows)

    def inside(self, pts: np.ndarray) -> np.ndarray:
        """True where world points [N, 3] project onto the figure."""
        uv = pts @ PROJ[self.name].T
        row, col = self.uv_to_pixel(uv[:, 0], uv[:, 1])
        ok = (row >= 0) & (row < self.mask.shape[0]) & (col >= 0) & (col < self.mask.shape[1])
        out = np.zeros(len(pts), dtype=bool)
        out[ok] = self.mask[row[ok], col[ok]]
        return out

    def camera(self) -> dict:
        """Orthographic camera reproducing this view's pixel mapping, for the Blender renderer."""
        size = self.size
        u_c = (size / 2 - self.cmid) / self.pps
        v_c = (self.vref - size / 2) / self.pps
        p = PROJ[self.name]
        centre = p[0] * u_c + p[1] * v_c            # world point at the image centre (depth 0)
        fwd = FORWARD[self.name]
        return {
            "view": self.name,
            "right": p[0].tolist(), "up": p[1].tolist(), "forward": fwd.tolist(),
            "location": (centre - fwd * 50.0).tolist(),
            "ortho_scale": size / self.pps, "resolution": size,
        }


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(rmin, rmax, cmin, cmax) of the figure, inclusive, or None for an empty mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0:
        return None
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labels (scipy when available, else a plain flood fill)."""
    try:
        from scipy import ndimage

        lab, n = ndimage.label(mask)
        return lab, int(n)
    except ImportError:
        pass
    labels = np.zeros(mask.shape, dtype=np.int32)
    n_r, n_c = mask.shape
    cur = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if labels[y0, x0]:
            continue
        cur += 1
        stack = [(y0, x0)]
        labels[y0, x0] = cur
        while stack:
            y, x = stack.pop()
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < n_r and 0 <= xx < n_c and mask[yy, xx] and not labels[yy, xx]:
                    labels[yy, xx] = cur
                    stack.append((yy, xx))
    return labels, cur


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 4-connected figure component (drops specks a segmenter leaves behind)."""
    lab, n = label(mask)
    if n <= 1:
        return mask.copy()
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == int(np.argmax(counts))


@dataclass
class ViewSet:
    views: dict[str, View]
    missing: list[str]
    meta: dict
    pps: float                  # pixels per stud
    height_px: int              # figure height in pixels (front, else the first side view)
    notes: list[str]


def load_views(run: Path, height_studs: float) -> ViewSet:
    """Read `masks/{view}.png` and `views/meta.json`; missing views are skipped and recorded."""
    meta_path = run / "views" / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    notes: list[str] = []
    masks: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for name in VIEWS:
        p = run / "masks" / f"{name}.png"
        if not p.exists():
            missing.append(name)
            continue
        m = load_mask(p)
        if m.shape[0] != m.shape[1]:
            notes.append(f"{name} mask is not square {m.shape}")
        if not m.any():
            notes.append(f"{name} mask is empty; skipped")
            missing.append(name)
            continue
        kept = largest_component(m)
        if kept.sum() < m.sum():
            notes.append(f"{name}: dropped {int(m.sum() - kept.sum())} px outside the largest component")
        masks[name] = kept
    side = [v for v in ("front", "back", "left", "right") if v in masks]
    if not side:
        raise ValueError("stage 3 needs at least one of the front/back/left/right masks")
    heights = {v: bbox(masks[v])[1] - bbox(masks[v])[0] + 1 for v in side}
    ref = "front" if "front" in heights else side[0]
    height_px = heights[ref]
    pps = height_px / height_studs
    spread = max(heights.values()) - min(heights.values())
    if spread > 2:
        notes.append(f"figure heights differ across side views by {spread} px: {heights}")
    if meta.get("pixels_per_unit") and meta.get("height_units"):
        expect = float(meta["pixels_per_unit"]) * float(meta["height_units"])
        if abs(expect - height_px) > 0.02 * height_px:
            notes.append(f"meta height {expect:.1f} px disagrees with the {ref} mask height {height_px} px")
    views: dict[str, View] = {}
    for name, m in masks.items():
        r0, r1, c0, c1 = bbox(m)
        cmid = (c0 + c1 + 1) / 2.0
        vref = (r0 + r1 + 1) / 2.0 if name == "top" else float(r1 + 1)
        views[name] = View(name, m, pps, cmid, vref)
    return ViewSet(views, missing, meta, pps, height_px, notes)
