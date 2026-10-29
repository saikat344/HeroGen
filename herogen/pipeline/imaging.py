"""Mask and image geometry for the pipeline: threshold masks, bounding boxes, column splits, square padding."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

Box = tuple[int, int, int, int]  # left, top, right, bottom (right/bottom exclusive)


def to_mask_image(mask: np.ndarray) -> Image.Image:
    """White figure on black, 8-bit."""
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


def from_mask_image(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L")) > 127


def background_colour(rgb: np.ndarray, border: int = 4) -> np.ndarray:
    """Median colour of the image border, taken as the background."""
    edges = np.concatenate(
        [
            rgb[:border].reshape(-1, 3),
            rgb[-border:].reshape(-1, 3),
            rgb[:, :border].reshape(-1, 3),
            rgb[:, -border:].reshape(-1, 3),
        ]
    )
    return np.median(edges, axis=0)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions that the border can't reach (closed outlines become solid)."""
    h, w = mask.shape
    canvas = Image.new("L", (w + 2, h + 2), 0)
    canvas.paste(to_mask_image(mask), (1, 1))
    ImageDraw.floodfill(canvas, (0, 0), 128)
    reach = np.array(canvas)[1:-1, 1:-1]
    return reach != 128


def _shift_or(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


def dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
    padded = np.pad(mask, r)
    return _shift_or(padded, r)[r:-r, r:-r] if r > 0 else mask.copy()


def erode(mask: np.ndarray, r: int = 1) -> np.ndarray:
    return ~dilate(~mask, r) if r > 0 else mask.copy()


def threshold_mask(img: Image.Image, tol: float = 40.0, close_px: int = 2) -> np.ndarray:
    """Background-threshold silhouette: pixels far from the border colour, outlines closed and filled.

    ``close_px`` dilates then erodes so gaps in pencil lines don't leak the flood fill; a 1 px opening
    removes isolated specks.
    """
    rgb = np.array(img.convert("RGB")).astype(np.int16)
    bg = background_colour(rgb)
    dist = np.abs(rgb - bg).max(axis=2)
    fg = dist > tol
    if close_px > 0:
        fg = erode(fill_holes(dilate(fg, close_px)), close_px)
    fg = dilate(erode(fg, 1), 1)
    return fill_holes(fg)


def bbox(mask: np.ndarray) -> Box | None:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def box_size(b: Box) -> tuple[int, int]:
    return b[2] - b[0], b[3] - b[1]


def column_runs(mask: np.ndarray, min_pixels: int = 1) -> list[tuple[int, int]]:
    """Runs of consecutive columns holding at least ``min_pixels`` figure pixels: [(start, end_exclusive)]."""
    occupied = mask.sum(axis=0) >= min_pixels
    runs: list[tuple[int, int]] = []
    start = None
    for x, on in enumerate(occupied):
        if on and start is None:
            start = x
        elif not on and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, len(occupied)))
    return runs


def split_columns(mask: np.ndarray, expected: int | None = None, min_frac: float = 0.015) -> list[tuple[int, int]]:
    """Column extents of the figures in a one-row sheet, left to right.

    Runs narrower than ``min_frac`` of the image width or holding under 0.2% of the figure pixels are noise
    and dropped. If more runs remain than ``expected``, the narrowest gaps are merged first (a figure can
    break into pieces where an arm leaves a gap); fewer runs than expected are returned as they are.
    """
    h, w = mask.shape
    total = max(1, int(mask.sum()))
    runs = [
        r for r in column_runs(mask)
        if (r[1] - r[0]) >= max(2, int(min_frac * w)) and mask[:, r[0]:r[1]].sum() >= 0.002 * total
    ]
    if expected is not None:
        while len(runs) > expected:
            gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
            i = int(np.argmin(gaps))
            runs[i : i + 2] = [(runs[i][0], runs[i + 1][1])]
    return runs


def paste_centred(img: Image.Image, size: int, fill, centre_x: float, centre_y: float) -> Image.Image:
    """Place ``img`` on a ``size`` x ``size`` canvas so its point (centre_x, centre_y) lands at the centre.

    Pure translation: no rescaling, so the pixel scale is unchanged.
    """
    canvas = Image.new(img.mode, (size, size), fill)
    ox = int(round(size / 2 - centre_x))
    oy = int(round(size / 2 - centre_y))
    canvas.paste(img, (ox, oy))
    return canvas


def normalise_silhouette(mask: np.ndarray, size: tuple[int, int] = (128, 256)) -> np.ndarray:
    """Crop to the bounding box and resize to a fixed grid, removing scale and position (and aspect)."""
    b = bbox(mask)
    if b is None:
        return np.zeros((size[1], size[0]), dtype=bool)
    crop = to_mask_image(mask[b[1] : b[3], b[0] : b[2]])
    return np.array(crop.resize(size, Image.BILINEAR)) > 127


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def align_on_box_centres(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate two same-scale masks onto one canvas so their bounding-box centres coincide."""
    ba, bb = bbox(a), bbox(b)
    if ba is None or bb is None:
        return a, b
    ca = a[ba[1] : ba[3], ba[0] : ba[2]]
    cb = b[bb[1] : bb[3], bb[0] : bb[2]]
    h = max(ca.shape[0], cb.shape[0])
    w = max(ca.shape[1], cb.shape[1])
    out = []
    for c in (ca, cb):
        canvas = np.zeros((h, w), dtype=bool)
        oy = (h - c.shape[0]) // 2
        ox = (w - c.shape[1]) // 2
        canvas[oy : oy + c.shape[0], ox : ox + c.shape[1]] = c
        out.append(canvas)
    return out[0], out[1]


def facing_offset(mask: np.ndarray, sole_frac: float = 0.03, shin: tuple[float, float] = (0.10, 0.15)) -> float:
    """Which way a side-view figure faces, from its feet: + means image right, - image left, 0 unknown.

    Toes stick out forward of the shin much further than heels stick out behind, so compare how far the
    sole (bottom ``sole_frac`` of the figure) extends past the shin band on each side. Normalised by the
    figure width.
    """
    b = bbox(mask)
    if b is None:
        return 0.0
    w, h = box_size(b)
    crop = mask[b[1] : b[3], b[0] : b[2]]
    sole = np.flatnonzero(crop[int(h * (1 - sole_frac)) :].any(axis=0))
    leg = np.flatnonzero(crop[int(h * (1 - shin[1])) : int(h * (1 - shin[0]))].any(axis=0))
    if sole.size == 0 or leg.size == 0:
        return 0.0
    forward_right = int(sole.max()) - int(leg.max())
    forward_left = int(leg.min()) - int(sole.min())
    return float((forward_right - forward_left) / max(1, w))


def top_from_hull(front: np.ndarray, back: np.ndarray, right: np.ndarray, left: np.ndarray) -> np.ndarray:
    """Top-view silhouette implied by the four side views (the plan projection of their visual hull).

    All masks are square, same size, same pixel scale, rows aligned (one baseline). Per the contract
    cameras: front and the mirrored back give the X-Z silhouette (image right = +X); right and the mirrored
    left give the Y-Z silhouette with image right = the character's front. The top view has +X to the right
    and the character's front at the bottom, so a column in the side view becomes a row in the top view.
    """
    xz = front & np.fliplr(back)
    yz = right & np.fliplr(left)
    n = front.shape[0]
    top = np.zeros((n, n), dtype=bool)
    rows = np.flatnonzero(xz.any(axis=1) & yz.any(axis=1))
    for z in rows:
        top |= np.outer(yz[z], xz[z])
    return top
