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
from herogen.pipeline import imaging
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
# Stage 1. Body parts are found with these SAM 3.1 prompts on the sketch, and belong to the main figure when at
# least ON_FIGURE_MIN of their area lies on its mask (on the tuning sketches a part on the figure measures
# 0.93-1.0 and one on another character 0.0).
FOOT_TEXT, LEG_TEXT, FACE_TEXT, HEAD_TEXT = "foot", "leg", "face", "head"
ON_FIGURE_MIN = 0.5
# Facing: the SAM 3.1 `face` centroid's offset from the `head` centroid, over the head's box width (`face_offset`,
# the held-out views' face cue), on the main figure. On the tuning sketches the two side-view sketches measure
# -0.30 (s040) and +0.43 (s045); every other sketch where it could be measured is within 0.157 (s033),
# three-quarter sketches included. Over 0.23, the middle of that gap, the sketch is likely a side view. Two side
# views are too few to reject on, so it warns.
FACE_TURN_MAX = 0.23
# Feet: no `foot` on the main figure means the feet can't be seen. On the tuning sketches every sketch whose
# figure SAM found has at least one foot on it except five: the two cropped at the knees (s008, s041), where
# `leg` finds two legs on the figure, and the two robes that hide the feet (s016, s031) and the side view s045,
# where it finds none. So legs without feet reject the sketch as cropped, and neither feet nor legs warn.


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


def parts_on(instances: list[SamInstance], figure: np.ndarray) -> int:
    """How many of ``instances`` (a body part's) lie mostly on the ``figure`` mask."""
    return sum(1 for i in instances if i.mask.any() and (i.mask & figure).sum() >= ON_FIGURE_MIN * i.mask.sum())


def feet_check(feet: int, legs: int | None) -> Check:
    """Feet found on the main figure, at least 1. ``legs`` (on the figure, asked only when there are no feet)
    decides what a failure means: legs without feet are a cropped figure (reject), no legs either is a lower
    body hidden, by a long robe say (warn)."""
    cropped = bool(legs)
    msg = ("the figure looks cropped: SAM 3.1 found its legs but no feet. HeroGen needs the whole body from the top "
           "of the head to the feet; draw or scan the full figure" if cropped else
           "no feet or legs found on the figure (a long robe or dress may hide them); HeroGen will have to guess "
           "the feet and legs")
    return input_check("sketch_feet_on_figure", feet, 1, ">=", "reject" if cropped else "warn", msg,
                       note="without feet: reject when legs are found on the figure (cropped), else warn")


def best_instance(instances: list[SamInstance], within: np.ndarray | None = None) -> SamInstance | None:
    """The most confident instance (largest when no confidence), keeping only those mostly on ``within``."""
    keep = [i for i in instances if i.mask.any()]
    if within is not None:
        keep = [i for i in keep if (i.mask & within).sum() >= 0.5 * i.mask.sum()]
    if not keep:
        return None
    return max(keep, key=lambda i: (i.confidence if i.confidence is not None else 0.0, int(i.mask.sum())))


def face_offset(face: np.ndarray, head: np.ndarray) -> float:
    """Which way the head is turned: (face centroid x - head centroid x) / head box width; + is image
    right."""
    hb = imaging.bbox(head)
    if hb is None or not face.any():
        return 0.0
    fx = float(np.nonzero(face)[1].mean())
    hx = float(np.nonzero(head)[1].mean())
    return (fx - hx) / max(1, hb[2] - hb[0])


def facing_check(face: SamInstance | None, head: SamInstance | None) -> tuple[Check, dict[str, Any]]:
    """Is the sketch a side view? |`face_offset`| of the face (at least half inside the head) within the head, at
    most `FACE_TURN_MAX`, else warn. With no face or head found on the figure (a full helmet, a back view) it is
    not measured: the check's value is null and its verdict "not_measured", which neither warns nor stops."""
    name, threshold, op = "sketch_face_offset_in_head", FACE_TURN_MAX, "<="
    reason = ("no head found on the figure" if head is None else "no face found on the figure" if face is None else
              "the face is not inside the head" if (face.mask & head.mask).sum() < 0.5 * face.mask.sum() else "")
    if reason:
        return ({"name": name, "value": None, "threshold": threshold, "op": op, "pass": True, "gate": False,
                 "verdict": "not_measured", "note": f"not measured: {reason}"}, {"face_offset": None})
    off = face_offset(face.mask, head.mask)
    side = "right" if off > 0 else "left"
    check = input_check(
        name, abs(off), threshold, op, "warn",
        f"the face is turned to the image {side} ({abs(off):.2f} of the head's width off its centre; over "
        f"{threshold} reads as a side view). HeroGen builds from a front view, so it may have to invent the front; "
        "a front-facing sketch works best",
    )
    return check, {"face_offset": round(off, 4)}
