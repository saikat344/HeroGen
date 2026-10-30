"""Measured checks. Each returns the contract's check dict: name, value, threshold, pass, gate."""

from __future__ import annotations

from typing import Any

import numpy as np

from herogen.pipeline import imaging

Check = dict[str, Any]


def check(name: str, value: float, threshold: float, op: str = ">=", gate: bool = True, note: str = "") -> Check:
    if op == ">=":
        ok = value >= threshold
    elif op == "<=":
        ok = value <= threshold
    elif op == "==":
        ok = value == threshold
    elif op == ">":
        ok = value > threshold
    elif op == "<":
        ok = value < threshold
    else:
        raise ValueError(f"unknown op {op!r}")
    out: Check = {
        "name": name,
        "value": round(float(value), 4),
        "threshold": threshold,
        "op": op,
        "pass": bool(ok),
        "gate": gate,
    }
    if note:
        out["note"] = note
    return out


def gate_ok(checks: list[Check]) -> bool:
    """A gate fails when any check with gate true (the default) fails."""
    return all(c["pass"] for c in checks if c.get("gate", True))


def agreement(values: list[float]) -> float:
    """min / max of positive values; 1.0 means identical, 0.0 when any value is zero or missing."""
    if not values or min(values) <= 0:
        return 0.0
    return float(min(values) / max(values))


def aspect(mask: np.ndarray) -> float:
    """Bounding-box height / width (0 for an empty mask)."""
    b = imaging.bbox(mask)
    if b is None:
        return 0.0
    w, h = imaging.box_size(b)
    return h / w


def silhouette_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two silhouettes after cropping each to its box and resizing to one grid (scale-free)."""
    return imaging.iou(imaging.normalise_silhouette(a), imaging.normalise_silhouette(b))


def mirror_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of ``a`` against ``b`` flipped left-right, same pixel scale, aligned on box centres."""
    fa, fb = imaging.align_on_box_centres(a, np.fliplr(b))
    return imaging.iou(fa, fb)


# Stage 1 thresholds. The cover is an edit of the sketch that must keep pose, proportions and costume.
# A 0.85 box-aspect agreement allows cleaned-up lines, a slightly wider stance or a helmet crest without
# letting a different pose through (arms out vs. down moves the aspect by 30% or more). A 0.65 IoU of the
# scale-normalised silhouettes tolerates line cleanup and small limb drift; a changed pose or a cropped
# figure measures well below 0.5.
COVER_ASPECT_MIN = 0.85
COVER_IOU_MIN = 0.65


def cover_checks(sketch_mask: np.ndarray, cover_mask: np.ndarray) -> tuple[list[Check], dict[str, Any]]:
    a_s, a_c = aspect(sketch_mask), aspect(cover_mask)
    checks = [
        check("sketch_mask_area_frac", float(sketch_mask.mean()), 0.01, gate=True),
        check("cover_mask_area_frac", float(cover_mask.mean()), 0.01, gate=True),
        check("aspect_agreement", agreement([a_s, a_c]), COVER_ASPECT_MIN),
        check("silhouette_iou_normalised", silhouette_iou(sketch_mask, cover_mask), COVER_IOU_MIN),
    ]
    return checks, {"sketch_aspect_h_over_w": round(a_s, 4), "cover_aspect_h_over_w": round(a_c, 4)}


# Stage 2 thresholds. Orthographic views of one figure at one scale have equal heights and the front and
# back silhouettes are mirror images, so the limits only absorb drawing noise of a generated sheet: 0.95
# for heights (about 25 px on a 500 px figure), 0.90 for front/back width (A-pose arm angles drift), 0.70
# IoU for the left/right mirror (side views of a generated character differ in small details like a
# shoulder plate; a wrong-pose view drops far below). Top view against front width and side width: 0.80,
# since a drawn top view's extent depends on how far the model spreads the arms and feet.
HEIGHT_AGREE_MIN = 0.95
FRONT_BACK_WIDTH_MIN = 0.90
MIRROR_IOU_MIN = 0.70
TOP_EXTENT_MIN = 0.80
# A-pose: the widest span across the hands band (35-60% of the height from the top) over the widest span
# across the shoulders (15-25%). Measured on generated sheets: arms hanging at the sides 1.21-1.25, arms
# about 25 degrees out 1.60. Recorded, not gated: fixed height bands land on hat brims, big chibi heads,
# shoulder armour and T-posed arms. It failed 9 of 21 true turnarounds of real game characters (bench-gt:
# T-pose, chibi, witch hat), 30 of 105 judged-usable generated sheets, and, once stage 2 carves each sheet as
# stage 3 does (`stages.carve_check`), it rejected 1 sheet that carves and was judged usable while stopping no
# sheet that was judged unusable (155 labelled attempts, tuning and held-out sketches alike).
A_POSE_SPREAD_MIN = 1.5
# Profile depth: a standing figure seen from the side with its arms down or out to the sides measures
# 0.20-0.28 of its height (measured on four generated sheets); an arm reaching forward measured 0.40.
# Recorded, not gated: the ratio measures body shape, not pose. Round bellies, robes, hat brims, chibi heads,
# wings, tails and backpacks are all deeper than 0.33. It failed 9 of 21 true turnarounds of real game
# characters (bench-gt), 67 of 105 judged-usable generated sheets, and, once stage 2 carves each sheet as
# stage 3 does (`stages.carve_check`), it rejected 16 sheets that carve and were judged usable (12 tuning, 4
# held-out) while stopping no sheet judged unusable (155 labelled attempts).
SIDE_DEPTH_MAX = 0.33
SIDE_VIEWS = ("front", "right", "back", "left")
# SAM coverage: a figure's SAM 3.1 mask px over the background-threshold px in the same region. Under half
# means SAM missed much of the figure (or the threshold picked up more than the figure), so its masks can't be
# measured. 0.5 is the rule that earlier versions used to switch a view to the threshold mask; it now fails
# the gate instead. On a replay of the 45-sketch benchmark the lowest values were 0.83 (87 split sheets) and
# 0.76 (413 held-out generations), so none failed it.
SAM_COVERAGE_MIN = 0.5


def band_span(mask: np.ndarray, lo: float, hi: float) -> int:
    """Widest left-to-right extent of any row between fractions ``lo`` and ``hi`` of the figure height."""
    b = imaging.bbox(mask)
    if b is None:
        return 0
    h = b[3] - b[1]
    rows = mask[b[1] + int(lo * h) : b[1] + max(int(lo * h) + 1, int(hi * h))]
    best = 0
    for r in rows:
        xs = np.flatnonzero(r)
        if xs.size:
            best = max(best, int(xs[-1] - xs[0] + 1))
    return best


def a_pose_spread(front: np.ndarray) -> float:
    shoulders = band_span(front, 0.15, 0.25)
    return band_span(front, 0.35, 0.60) / shoulders if shoulders else 0.0


def side_view_checks(masks: dict[str, np.ndarray], *, figures_found: int, figures_expected: int = 4) -> list[Check]:
    """Gate checks on the four same-scale side views (front, right, back, left)."""
    out: list[Check] = [check("figures_found", figures_found, figures_expected, op="==")]
    if figures_found != figures_expected or any(v not in masks for v in SIDE_VIEWS):
        return out
    size = {v: imaging.box_size(imaging.bbox(masks[v]) or (0, 0, 0, 0)) for v in SIDE_VIEWS}
    out.append(check("height_agreement_4_views", agreement([size[v][1] for v in SIDE_VIEWS]), HEIGHT_AGREE_MIN))
    out.append(
        check("front_back_width_agreement", agreement([size["front"][0], size["back"][0]]), FRONT_BACK_WIDTH_MIN)
    )
    out.append(check("left_right_mirror_iou", mirror_iou(masks["left"], masks["right"]), MIRROR_IOU_MIN))
    out.append(check("a_pose_hand_span_over_shoulders", a_pose_spread(masks["front"]), A_POSE_SPREAD_MIN, gate=False,
                     note="recorded, not gated: fixed height bands misread hats, big heads and T-poses"))
    for v in ("right", "left"):
        w, h = size[v]
        out.append(check(f"{v}_profile_width_over_height", w / h if h else 1.0, SIDE_DEPTH_MAX, op="<=", gate=False,
                         note="recorded, not gated: measures body shape (belly, robe, hat, wings), not pose"))
    return out


def shortfall(c: Check) -> float:
    """How far a failed check's value is on the wrong side of its threshold, as a fraction of the threshold
    (0 for a check that passed)."""
    if c["pass"]:
        return 0.0
    t = float(c["threshold"])
    return abs(float(c["value"]) - t) / (abs(t) or 1.0)


def attempt_rank(checks: list[Check], expected: tuple[str, ...]) -> tuple[int, float]:
    """How bad one attempt's gated checks are, lower is better: (failed gates, their summed `shortfall`). A gate
    in ``expected`` that was never measured (a sheet that could not be split has only `figures_found`) counts as
    failed with a shortfall of 1, so an attempt is never ranked well for what it skipped."""
    gated = [c for c in checks if c.get("gate", True)]
    failed = [c for c in gated if not c["pass"]]
    missing = [n for n in expected if n not in {c["name"] for c in gated}]
    return len(failed) + len(missing), round(sum(shortfall(c) for c in failed) + len(missing), 6)


def top_view_checks(masks: dict[str, np.ndarray], top: np.ndarray, *, prefix: str = "top", gate: bool = True,
                    note: str = "") -> list[Check]:
    """Top-view extents at the shared scale: X extent vs. front width, Y extent (depth) vs. side width."""
    size = {v: imaging.box_size(imaging.bbox(masks[v]) or (0, 0, 0, 0)) for v in SIDE_VIEWS}
    tw, td = imaging.box_size(imaging.bbox(top) or (0, 0, 0, 0))
    side_w = (size["left"][0] + size["right"][0]) / 2
    return [
        check(
            f"{prefix}_width_vs_front_width", agreement([tw, size["front"][0]]), TOP_EXTENT_MIN, gate=gate, note=note
        ),
        check(f"{prefix}_depth_vs_side_width", agreement([td, side_w]), TOP_EXTENT_MIN, gate=gate, note=note),
    ]


def bbox_sizes(masks: dict[str, np.ndarray]) -> dict[str, list[int]]:
    return {v: list(imaging.box_size(imaging.bbox(m) or (0, 0, 0, 0))) for v, m in masks.items()}
