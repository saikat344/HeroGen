"""Input checks: is the sketch usable at all? Run in stages 0 and 1 before any image is generated.

Each check is a measured check (`checks.check`) with a ``verdict``: "pass", "warn" or "reject". A rejected
check is gated and stops the run with the stage's status "input_rejected"; a warning is recorded with a
``message`` for the user and the run goes on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from herogen.pipeline import checks as C
from herogen.pipeline.client import SamInstance

Check = dict[str, Any]

VERDICTS = ("pass", "warn", "reject")

# Stage 0. Muse Image works from the sketch's pixels; a side under 256 px leaves too little to keep.
MIN_SIDE_PX = 256
# Stage 0. Line contrast (`line_contrast`), in grey levels. On the tuning sketches s001-s045 the two faint-pencil
# sketches measure 11 and 12 and the 43 others 85-253. Below 32 (about the geometric mean of 12 and 85, a factor of
# 2.7 from each) the lines are too faint: SAM 3.1 found no `character` in either faint sketch, and Muse Image
# invented the detail it couldn't see.
MIN_LINE_CONTRAST = 32
# Stage 1. A SAM 3.1 instance of the figure prompt at least this share of the largest one's area is another
# character; a smaller one is an inset or a detail drawing. On the tuning sketches the two multiple-character
# sketches have a second instance at 0.66 and 0.77 of the largest, and every other sketch has a single instance,
# so the tuning set only bounds the cut to (0, 0.66]: it holds no small extra figures to place it against. 0.25 is
# the diagnosis's proposal, measured there and kept unchanged; the unseen concept sheets s046-s058 are the test.
OTHER_CHARACTER_MIN_FRAC = 0.25


def input_check(name: str, value: float, threshold: float, op: str, on_fail: str, message: str,
                note: str = "") -> Check:
    """A check whose failure has the verdict ``on_fail`` ("reject" or "warn"). Only a check that rejects is
    gated. ``message`` tells the user what is wrong and what to do; it is kept only when the check fails."""
    if on_fail not in ("reject", "warn"):
        raise ValueError(f"on_fail must be 'reject' or 'warn', not {on_fail!r}")
    out = C.check(name, value, threshold, op=op, gate=on_fail == "reject", note=note)
    out["verdict"] = "pass" if out["pass"] else on_fail
    if not out["pass"]:
        out["message"] = message
    return out


def rejections(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.get("verdict") == "reject"]


def warnings(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.get("verdict") == "warn"]


def rejection_error(checks: list[Check]) -> str:
    """The error of a rejected input: every rejecting check's message, in order."""
    return "; ".join(c["message"] for c in rejections(checks))


def size_check(width: int, height: int) -> Check:
    side = min(width, height)
    return input_check(
        "min_side_px", side, MIN_SIDE_PX, ">=", "reject",
        f"the sketch is too small: its shorter side is {side} px and HeroGen needs at least {MIN_SIDE_PX} px. "
        "Scan or export it at a higher resolution",
    )


def line_contrast(img: Image.Image) -> float:
    """How much darker the darkest lines are than the paper, in grey levels (0-255): the median luminance (the
    paper, which covers most of a sketch) minus the 1st percentile (the darkest lines). Lines must cover at least
    1% of the image to be measured (on the tuning sketches they cover 5.4% or more), and light lines on dark paper
    measure as faint."""
    grey = np.asarray(img.convert("L"), dtype=np.float32)
    return float(np.median(grey) - np.percentile(grey, 1))


def contrast_check(img: Image.Image) -> Check:
    v = line_contrast(img)
    return input_check(
        "sketch_line_contrast", v, MIN_LINE_CONTRAST, ">=", "reject",
        f"the lines are too faint: the darkest lines are only {v:.0f} grey levels darker than the paper and HeroGen "
        f"needs at least {MIN_LINE_CONTRAST}. Darken the lines, or scan or photograph the sketch with more contrast",
    )


def by_area(instances: list[SamInstance]) -> list[SamInstance]:
    """Non-empty instances, largest first."""
    return sorted((i for i in instances if i.mask.any()), key=lambda i: -int(i.mask.sum()))


def character_checks(instances: list[SamInstance], no_character: str) -> tuple[list[Check], dict[str, Any]]:
    """How many characters SAM 3.1 found in the sketch (``instances``, the figure prompt's): exactly one at least
    `OTHER_CHARACTER_MIN_FRAC` of the largest one's area, else reject (``no_character`` is the message for none);
    smaller extra figures warn, since only the largest figure is built."""
    figs = by_area(instances)
    fracs = [round(int(i.mask.sum()) / int(figs[0].mask.sum()), 4) for i in figs] if figs else []
    big = [f for f in fracs if f >= OTHER_CHARACTER_MIN_FRAC]
    small = [f for f in fracs if f < OTHER_CHARACTER_MIN_FRAC]
    if not big:
        msg = no_character
    else:
        msg = (f"the sketch shows {len(big)} characters: SAM 3.1 found {len(big) - 1} more at least "
               f"{OTHER_CHARACTER_MIN_FRAC:.0%} the size of the largest (the next is {big[1]:.0%} of it). "
               "Draw one character per sketch") if len(big) > 1 else ""
    checks = [
        input_check("sketch_characters_found", len(big), 1, "==", "reject", msg,
                    note=f"figures at least {OTHER_CHARACTER_MIN_FRAC} of the largest one's area"),
        input_check(
            "sketch_small_extra_figures", len(small), 0, "<=", "warn",
            f"besides the main character the sketch has {len(small)} smaller figure{'s' if len(small) > 1 else ''} "
            f"(each under {OTHER_CHARACTER_MIN_FRAC:.0%} of its size, such as insets or detail drawings); HeroGen "
            "builds only the largest figure",
        ),
    ]
    return checks, {"figure_area_fracs": fracs}
