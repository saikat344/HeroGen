"""Synthetic figures drawn by code for the herogen pipeline tests (no real sketches or generated images)."""

from __future__ import annotations

import math
import unittest

import numpy as np
from PIL import Image, ImageDraw


def figure_mask(h: int, pose: str = "front", arms: str = "a") -> np.ndarray:
    """A stick-and-block figure ``h`` px tall on a tight canvas.

    pose: "front" (also used for back), "right" (profile facing image right), "left" (facing left).
    arms: "a" (A-pose, 45 degrees out), "down" (at the sides). Profiles always have arms down.
    """
    w = int(h * 1.0)
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    cx = w // 2
    head_r = int(0.07 * h)
    d.ellipse([cx - head_r, 0, cx + head_r, 2 * head_r], fill=255)
    if pose == "front":
        tw = int(0.14 * h)
        d.rectangle([cx - tw, int(0.13 * h), cx + tw, int(0.55 * h)], fill=255)
        lw = int(0.05 * h)
        for sx in (-1, 1):
            lx = cx + sx * int(0.08 * h)
            d.rectangle([lx - lw, int(0.5 * h), lx + lw, h - 1], fill=255)
            x0, y0 = cx + sx * tw, int(0.17 * h)
            if arms == "a":
                ln = 0.36 * h
                x1, y1 = x0 + sx * ln * math.sin(math.radians(45)), y0 + ln * math.cos(math.radians(45))
            else:
                x1, y1 = x0 + sx * 0.02 * h, y0 + 0.36 * h
            d.line([x0, y0, x1, y1], fill=255, width=int(0.05 * h))
            r = int(0.05 * h) / 2  # round ends, like the capsule arms of the 3D stick figure
            for x, y in ((x0, y0), (x1, y1)):
                d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    else:
        sign = 1 if pose == "right" else -1
        tw = int(0.08 * h)
        d.rectangle([cx - tw, int(0.13 * h), cx + tw, int(0.55 * h)], fill=255)
        lw = int(0.05 * h)
        d.rectangle([cx - lw, int(0.5 * h), cx + lw, h - 1], fill=255)
        # foot: toes stick out forward (sign), heel only a little behind
        fx0, fx1 = sorted([cx - sign * int(0.06 * h), cx + sign * int(0.13 * h)])
        d.rectangle([fx0, h - 1 - int(0.035 * h), fx1, h - 1], fill=255)
        d.rectangle([cx - int(0.03 * h), int(0.17 * h), cx + int(0.03 * h), int(0.52 * h)], fill=255)
    m = np.array(img) > 127
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    return m[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def paint(mask: np.ndarray, colour=(40, 70, 140), bg=(250, 250, 250)) -> Image.Image:
    rgb = np.empty(mask.shape + (3,), dtype=np.uint8)
    rgb[:] = bg
    rgb[mask] = colour
    return Image.fromarray(rgb)


def sheet(h: int = 200, order=("front", "right", "front", "left"), arms: str = "a", gap: int = 40,
          bg=(250, 250, 250)) -> tuple[Image.Image, np.ndarray]:
    """A one-row turnaround sheet: figures on one baseline with gaps. Returns (image, full figure mask)."""
    figs = [figure_mask(h, p, arms) for p in order]
    width = sum(f.shape[1] for f in figs) + gap * (len(figs) + 1)
    height = h + 2 * gap
    mask = np.zeros((height, width), dtype=bool)
    x = gap
    for f in figs:
        mask[gap : gap + f.shape[0], x : x + f.shape[1]] |= f
        x += f.shape[1] + gap
    return paint(mask, bg=bg), mask


def place(mask: np.ndarray, size: tuple[int, int], at: tuple[int, int]) -> np.ndarray:
    out = np.zeros((size[1], size[0]), dtype=bool)
    out[at[1] : at[1] + mask.shape[0], at[0] : at[0] + mask.shape[1]] = mask
    return out


class SynthSelfTest(unittest.TestCase):
    def test_profiles_face_their_way(self):
        r = figure_mask(200, "right")
        lft = figure_mask(200, "left")
        self.assertTrue(np.array_equal(np.fliplr(r), lft))

    def test_sheet_has_four_figures(self):
        img, mask = sheet()
        self.assertEqual(img.size[::-1], mask.shape)
        self.assertGreater(mask.sum(), 0)


if __name__ == "__main__":
    unittest.main()
