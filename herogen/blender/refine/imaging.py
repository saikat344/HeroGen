"""Registration, measured repaint gates, colour matching and visible texel projection."""
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from herogen.pipeline.facesource import ncc, refine_affine, sample
from . import character as C
from .cameras import axes, project, px_per_unit

WORK = 512
BG_TOL = 0.07          # an edit pixel is figure when any channel differs from the grey by more than this
# Outline gate, per 1024 render pixels: IoU of the figures, median and 90th-percentile distance between their outlines
# (both directions). Measured values from s001's head views; not tuned on a held-out set.
GATE = {"iou_min": 0.90, "median_px": 6.0, "p90_px": 16.0}
EXTERIOR_DILATE = 16   # known exterior background margin outside the render figure, in pixels
CUT_DILATE = 16        # frame-cut exclusion around border-touching silhouettes when sampling chamfer
# Bounded close-up outline allowance, per 1024 render pixels: GATE was measured on s001's smooth head views, so
# close-up cameras use a capped allowance rather than multiplying GATE by the camera zoom ratio.
CLOSEUP_GATE = {"iou_min": 0.90, "median_px": 10.0, "p90_px": 24.0}
# Part gate: SAM 3.1 masks of these parts on the render and on the registered repaint. Where the repaint shows a part
# (more than PART_MIN_AREA of the figure), at most PART_EXTRA_MAX of it may lie outside the render's mask of that
# part, or the repaint is rejected (a face painted on the back of the head, a profile turned toward the camera).
# Containment, not IoU: on the soft render SAM 3.1's face includes a beard that it leaves out on a sharp repaint
# (s015: IoU 0.32-0.47 on repaints that kept the face where it was). Only `face`: SAM 3.1 does not see SAM's
# smooth helmet-like hair as hair in the render (s001: hair IoU 0.39-0.41 on a repaint that kept the hair outline).
PARTS = ("face",)
PART_EXTRA_MAX = 0.25
PART_MIN_AREA = 0.005
FEATHER_PX = 14        # in render pixels, from the outline and depth jumps
CLOSEUP_FEATHER = True
CLOSEUP_FEATHER_PX = 96  # raised-cosine ramp on frame-cut borders of head/hair close-ups, per 1024 render pixels
COLOUR_SIGMA = 0.04    # residual low-frequency colour match, as a fraction of the render's larger side
EXPOSURE_SOLVE = False
EXPOSURE_MODE = "scalar"       # "scalar" (log-gain least squares) or "affine" (scalar gain + bounded offset)
EXPOSURE_SIGMA_PX = 12.0       # surface smoothing in body-camera pixels per 1024 for 3D overlap comparisons
EXPOSURE_GAIN_CLIP = (0.80, 1.25)
EXPOSURE_OFFSET_CLIP = 0.06
EXPOSURE_REG = 0.03
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
Q0_FACING = 0.30       # SAM's colour counts as a view with cos^2 = 0.30 at body resolution
JUMP = 0.004          # a depth step above this fraction of the body height between neighbour pixels is an edge
VIS_WINDOW = 3         # a texel is visible only if no surface in this pixel window lies in front of it
RAMP = 1.0             # a view takes a texel fully once its quality is (1 + RAMP) x the best so far
ATTEMPTS = 3           # repaint samples per view before the view is skipped
# Hair mode: the outline may move; the repaint is used only inside the render's figure
# eroded by INTERIOR_ERODE px (per 1024) and where its content agrees with the render (`interior_agreement`).
AGREE_SIGMA = 0.008
AGREE_TOL = (0.08, 0.16)
INTERIOR_ERODE = 6
# Hair-mode gate, from stored rf2r head repaints: good views agree on 0.94-1.0 of the interior (sharpened faces
# 0.76-0.89, which hair mode never writes); s015's +90 repaints that moved hair onto the scalp agree on 0.75-0.78;
# zoomed s001 profiles have figure IoU 0.63-0.71.
HAIR_GATE = {"iou_min": 0.85, "scale_max_change": 0.10, "agree_min": 0.80}
HAIR_ATTEMPTS = 2
HAIR_BONUS = 2.0       # hair views take texels from earlier, softer passes of the same camera
FACE_KEEP_PX = 12      # hair views never write the render's SAM 3.1 face mask grown by this (per 1024)
# Every pass preserves pixels where the render and repaint disagree about hair. Hair-mode writes inside
# the render head additionally require hair in both masks; colour agreement alone cannot protect a bald scalp.


def grey(a):
    return a[..., :3] @ np.array([0.299, 0.587, 0.114])


def border_touch(m):
    """Mask pixels that lie on any of the four image borders."""
    frame = np.zeros(m.shape, bool)
    frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = True
    return np.asarray(m, bool) & frame


def _background_colour(rgb, render_figure):
    """Measure a uniform neutral backdrop only in the render's known exterior.

    The render selects colour samples, never the edit silhouette. Coloured or nonuniform
    exterior content cannot calibrate the background and keeps the configured grey.
    """
    exterior = ~ndimage.binary_dilation(render_figure, iterations=EXTERIOR_DILATE)
    if exterior.any():
        samples = rgb[exterior]
        candidate = np.median(samples, axis=0)
        uniform = (np.abs(samples - candidate) <= BG_TOL).all(-1).mean()
        # Keep the established tolerance band when it already recognises the backdrop.
        # Recentring it for tiny drift can erase near-grey surface instead of fixing a flood.
        if (np.abs(candidate - C.BG) > BG_TOL).any() and np.ptp(candidate) <= BG_TOL / 2 and uniform >= 0.90:
            return candidate
    return C.BG


def figure_mask(rgb, *, render_figure=None):
    bg = C.BG if render_figure is None else _background_colour(rgb, render_figure)
    m = (np.abs(rgb - bg) > BG_TOL).any(-1)
    if render_figure is None:
        m = ndimage.binary_opening(m, iterations=2)
        m = ndimage.binary_closing(m, iterations=3)
        m = ndimage.binary_fill_holes(m)
    else:
        render_figure = np.asarray(render_figure, bool)
        pad = 6
        closed = ndimage.binary_closing(np.pad(m, pad, mode="edge"), iterations=3)[pad:-pad, pad:-pad]
        exterior = ~ndimage.binary_dilation(render_figure, iterations=EXTERIOR_DILATE)
        if not exterior.any():
            exterior = ~render_figure
        covered = border_touch(render_figure)
        walkable = ~closed & ~covered
        seed = walkable & exterior
        if seed.any():
            bg = ndimage.binary_propagation(seed, mask=walkable)
            bg |= ndimage.binary_dilation(bg, iterations=1) & covered & ~closed
            m = ~bg
            if m.any():
                m = ndimage.binary_opening(np.pad(m, pad, mode="edge"), iterations=2)[pad:-pad, pad:-pad]
        elif exterior.any():
            m = np.ones_like(closed)
        else:
            m = (ndimage.binary_opening(np.pad(closed, pad, mode="edge"), iterations=2)[pad:-pad, pad:-pad]
                 if closed.any() else closed)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.flatnonzero(sizes > 0.02 * sizes.max()))
    return m


def closeup_figure_mask(rgb, render_figure):
    """Separate grey background from grey surface cut by the frame.

    Only grey components touching known render background can be background. A grey metal/fabric patch
    touching a foreground frame edge remains surface. The prior seeds components, not the silhouette:
    new non-grey geometry and connected grey gaps can still move the contour and fail the same gate.
    """
    near_bg = (np.abs(rgb - C.BG) <= BG_TOL).all(-1)
    if near_bg.all():
        return np.zeros_like(render_figure)
    labels, _ = ndimage.label(near_bg)
    seeds = np.unique(labels[near_bg & ~render_figure])
    background = np.isin(labels, seeds) & near_bg
    # Extend the cut surface/background through the frame before morphology, rather than introducing
    # a false empty frame or closing real background gaps with a false foreground frame.
    figure = np.pad(~background, 10, mode="edge")
    figure = ndimage.binary_opening(figure, iterations=2)
    figure = ndimage.binary_closing(figure, iterations=3)
    return figure[10:-10, 10:-10]


def boundary(m):
    return m & ~ndimage.binary_erosion(m)


def chamfer(a, b, *, unclipped=True):
    """Outline distances (median, p90) in pixels, both directions; the image border is not an outline."""
    if not unclipped:
        def bd(m):
            e = m & ~ndimage.binary_erosion(m, border_value=1)
            e[:4], e[-4:], e[:, :4], e[:, -4:] = False, False, False, False
            return e
        ba, bb = bd(a), bd(b)
        if not ba.any() or not bb.any():
            return 1e9, 1e9
        da = ndimage.distance_transform_edt(~ba)
        db = ndimage.distance_transform_edt(~bb)
        d = np.concatenate([db[ba], da[bb]])
        return float(np.median(d)), float(np.quantile(d, 0.9))
    ba = a & ~ndimage.binary_erosion(a, border_value=0)
    bb = b & ~ndimage.binary_erosion(b, border_value=0)
    if not ba.any() or not bb.any():
        return 1e9, 1e9
    da = ndimage.distance_transform_edt(~ba)
    db = ndimage.distance_transform_edt(~bb)
    touch = border_touch(a | b)
    cut = ndimage.binary_dilation(touch, iterations=CUT_DILATE) if touch.any() else touch
    sa, sb = ba & ~cut, bb & ~cut
    if not sa.any() or not sb.any():
        return 1e9, 1e9
    d = np.concatenate([db[sa], da[sb]])
    return float(np.median(d)), float(np.quantile(d, 0.9))


def is_closeup(cam, body_cam):
    """True when `cam` frames the mesh at finer world scale than `body_cam`."""
    return px_per_unit(cam) > px_per_unit(body_cam) * 1.05


def outline_gate(a, b, *, unclipped=True, closeup=False):
    """(passed, measures) for the figure outline gate, in render pixels per 1024."""
    overlap = float((a & b).sum() / max(1, (a | b).sum()))
    med, p90 = (v * 1024 / max(a.shape) for v in chamfer(a, b, unclipped=unclipped))
    limits = CLOSEUP_GATE if closeup else GATE
    measures = {
        "iou": overlap,
        "median_px": med,
        "p90_px": p90,
        "median_limit": limits["median_px"],
        "p90_limit": limits["p90_px"],
    }
    passed = (
        overlap >= limits["iou_min"]
        and med <= limits["median_px"]
        and p90 <= limits["p90_px"]
    )
    return passed, measures


def frame_feather(shape, width=96):
    """Raised cosine from zero at every frame edge to one `width` pixels inside."""
    if width <= 0:
        raise ValueError("frame feather width must be positive")
    h, w = shape
    y, x = np.ogrid[:h, :w]
    distance = np.minimum(np.minimum(y, h - 1 - y), np.minimum(x, w - 1 - x))
    return (0.5 - 0.5 * np.cos(np.pi * np.clip(distance / width, 0, 1))).astype(np.float32)


def closeup_border_px(shape, width=CLOSEUP_FEATHER_PX):
    """Scale `width` (measured per 1024 render pixels) to `shape`."""
    if not math.isfinite(width) or width <= 0:
        raise ValueError("close-up feather width must be positive")
    return max(1, round(width * max(shape) / 1024))


def closeup_feather(figure, width=CLOSEUP_FEATHER_PX):
    """Raised cosine from zero on frame-cut silhouette borders to one `closeup_border_px` inside."""
    w = closeup_border_px(figure.shape, width)
    touch = border_touch(figure)
    if not touch.any():
        return np.ones(figure.shape, np.float32)
    dist = ndimage.distance_transform_edt(~touch)
    return (0.5 - 0.5 * np.cos(np.pi * np.clip(dist / w, 0.0, 1.0))).astype(np.float32)


def closeup_outline(a, b, zoom):
    """Keep the measured body limits in world units; ignore frame-cut contours and border IoU.

    A close-up filled entirely by the surface has no silhouette to move. This is accepted only when both
    images have no interior contours and their interior figure overlap passes the unchanged IoU limit.
    """
    if not math.isfinite(zoom) or zoom <= 0:
        raise ValueError("close-up zoom must be positive")
    size = max(a.shape)
    border = max(4, round(8 * zoom * size / 1024))
    inner = frame_feather(a.shape, border) >= 1
    overlap = float((a & b & inner).sum() / max(1, ((a | b) & inner).sum()))
    ba = (a & ~ndimage.binary_erosion(a, border_value=1)) & inner
    bb = (b & ~ndimage.binary_erosion(b, border_value=1)) & inner
    if ba.any() and bb.any():
        d = np.concatenate([ndimage.distance_transform_edt(~bb)[ba], ndimage.distance_transform_edt(~ba)[bb]])
        med, p90 = np.quantile(d, [.5, .9]) * 1024 / size
    elif not ba.any() and not bb.any() and (a & b & inner).any():
        med, p90 = 0.0, 0.0
    else:
        med, p90 = 1e9, 1e9
    measures = {"iou": overlap, "median_px": float(med), "p90_px": float(p90), "zoom": zoom,
                "border_excluded_px": border, "median_limit_px": GATE["median_px"] * zoom,
                "p90_limit_px": GATE["p90_px"] * zoom}
    passed = (overlap >= GATE["iou_min"] and med <= measures["median_limit_px"]
              and p90 <= measures["p90_limit_px"])
    return passed, measures


def warp_grid(img, mapx, mapy):
    return cv2.remap(img.astype(np.float32), mapx.astype(np.float32), mapy.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _refine_similarity(target, source, weight, initial=None):
    """Fit uniform scale, rotation and translation, with photometric gain/bias.

    A repaint of the same orthographic camera can change framing, but cannot shear or
    scale just one axis. An unrestricted affine fit can squeeze a narrow side view
    into background to explain changed paint colours instead of registering it.
    """
    # scale*cos(angle), scale*sin(angle), tx, ty
    p = (np.array([1.0, 0, 0, 0]) if initial is None else
         np.array([initial[0, 0], initial[1, 0], initial[0, 2], initial[1, 2]]))
    for factor in (4, 2, 1):
        size = target.shape[0] // factor
        t = cv2.resize(target, (size, size), interpolation=cv2.INTER_AREA)
        s = cv2.resize(source, (size, size), interpolation=cv2.INTER_AREA)
        w = cv2.resize(weight, (size, size), interpolation=cv2.INTER_AREA) > 0.5
        if w.sum() < 64:
            return None
        yy, xx = np.indices((size, size))
        x, y, tv = xx[w] + 0.5, yy[w] + 0.5, t[w]
        gy, gx = np.gradient(s)
        p[2:] /= factor
        for _ in range(30):
            u, v = p[0] * x - p[1] * y + p[2], p[1] * x + p[0] * y + p[3]
            iv = sample(s, u, v)
            ix, iy = sample(gx, u, v), sample(gy, u, v)
            var = float(iv.var())
            gain = float(((iv - iv.mean()) * (tv - tv.mean())).mean() / var) if var > 1e-9 else 1.0
            bias = float(tv.mean() - gain * iv.mean())
            residual = tv - gain * iv - bias
            jac = np.stack([gain * (ix * x + iy * y), gain * (-ix * y + iy * x),
                            gain * ix, gain * iy, iv, np.ones_like(iv)], axis=1)
            hessian = jac.T @ jac
            try:
                delta = np.linalg.solve(hessian + 1e-6 * np.eye(6) * np.trace(hessian) / 6, jac.T @ residual)
            except np.linalg.LinAlgError:
                return None
            p += delta[:4]
            if not np.isfinite(p).all() or p[0] ** 2 + p[1] ** 2 < 1e-6:
                return None
            if np.abs(delta[:4]).max() < 1e-4 * max(1.0, size / 64):
                break
        p[2:] *= factor
    return np.array([[p[0], -p[1], p[2]], [p[1], p[0], p[3]]])


# Work-space fit limits measured on 973 saved repaint attempts. Singular values
# measure scale along every direction; determinant also catches two-axis shrinkage.
# These select a registration method before any outline or hair gate is evaluated.
AFFINE_SCALE = (0.85, 1.25)
AFFINE_DETERMINANT_MIN = 0.80
AFFINE_SHEAR_MAX = 0.15
AFFINE_ROTATION_MAX = 5.0   # degrees, from the polar rotation of the linear map
AFFINE_CENTROID_MAX = 0.125  # displacement as a fraction of the longer image side


def _affine_degeneracy(matrix, figure):
    """Diagnose only the primary work-space transform, independent of edit masks/gates."""
    if matrix is None:
        return {"reasons": ["fit_failed"]}
    if not np.isfinite(matrix).all():
        return {"reasons": ["nonfinite"]}
    linear = matrix[:, :2]
    scales = np.linalg.svd(linear, compute_uv=False)
    determinant = float(np.linalg.det(linear))
    axes_scale = np.linalg.norm(linear, axis=0)
    shear = float(abs(np.dot(linear[:, 0], linear[:, 1])) / max(1e-20, np.prod(axes_scale)))
    rotation = float(np.degrees(np.arctan2(linear[1, 0] - linear[0, 1], linear[0, 0] + linear[1, 1])))
    reasons = []
    if abs(rotation) > AFFINE_ROTATION_MAX:
        reasons.append("rotation")
    if scales.min() < AFFINE_SCALE[0] or scales.max() > AFFINE_SCALE[1]:
        reasons.append("scale")
    if determinant < AFFINE_DETERMINANT_MIN:
        reasons.append("determinant")
    if shear > AFFINE_SHEAR_MAX:
        reasons.append("shear")
    y, x = np.nonzero(figure)
    displacement = None
    if len(x):
        centroid = np.array([x.mean() + 0.5, y.mean() + 0.5])
        displacement = float(np.linalg.norm(linear @ centroid + matrix[:, 2] - centroid) / max(figure.shape))
        if displacement > AFFINE_CENTROID_MAX:
            reasons.append("centroid")
    else:
        reasons.append("empty_figure")
    return {"reasons": reasons, "singular_scales": scales.tolist(), "determinant": determinant,
            "shear": shear, "rotation_degrees": rotation, "centroid_displacement": displacement}


def _register_similarity(render_rgb, alpha, edit_rgb, wh, ww):
    """Constrained rescue for a degenerate affine; never selected by a gate result."""
    pad = max(wh, ww)
    def work_image(img, *, edges=False):
        small = cv2.resize(grey(img), (ww, wh), interpolation=cv2.INTER_AREA)
        smooth = cv2.GaussianBlur(small, (0, 0), 1.5)
        if edges:
            gy, gx = np.gradient(smooth)
            smooth = np.hypot(gx, gy)
        out = np.zeros((pad, pad), np.float64)
        out[:wh, :ww] = smooth
        return out

    # Edge magnitude retains geometry when paint reverses luminance contrast.
    tgt, src = work_image(render_rgb, edges=True), work_image(edit_rgb, edges=True)
    # Include real exterior pixels in the fit; exclude the square canvas padding.
    wt = np.zeros((pad, pad), np.float64)
    wt[:wh, :ww] = 1
    M0 = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    # Edge matching has local optima when a repaint changes scale. Use a brightness
    # fit as a second start, then select by edge correlation, never by a gate result.
    brightness = _refine_similarity(work_image(render_rgb), work_image(edit_rgb), wt)
    candidates = [M0, brightness, _refine_similarity(tgt, src, wt)]
    if brightness is not None:
        candidates.append(_refine_similarity(tgt, src, wt, initial=brightness))
    wy, wx = np.indices((pad, pad), dtype=np.float64)
    wx, wy = wx + 0.5, wy + 0.5

    work_figure = cv2.resize((alpha > 0.5).astype(np.uint8), (ww, wh), interpolation=cv2.INTER_NEAREST) > 0
    contour = work_figure & ~ndimage.binary_erosion(work_figure, border_value=1)
    band = np.zeros((pad, pad), bool)
    band[:wh, :ww] = ndimage.binary_dilation(contour, iterations=5)
    # The contour band includes three blur sigmas on either side. Repaint detail
    # inside the figure must not choose a different framing from matching outlines.
    score_mask = band if band.sum() >= 64 else wt > 0

    def edge_score(matrix):
        warped = sample(src, matrix[0, 0] * wx + matrix[0, 1] * wy + matrix[0, 2],
                        matrix[1, 0] * wx + matrix[1, 1] * wy + matrix[1, 2])
        return ncc(tgt, warped, score_mask)

    M = max((matrix for matrix in candidates if matrix is not None), key=edge_score)
    return M


def register(render_rgb, alpha, edit_rgb):
    """Map render pixel edge coordinates to edit coordinates, retaining a valid affine.

    Only a degenerate primary fit invokes constrained registration. The decision uses
    transform geometry and the render silhouette, before measuring the repaint outline.
    """
    h, w = alpha.shape
    eh, ew = edit_rgb.shape[:2]
    s = WORK / max(h, w)
    wh, ww = round(h * s), round(w * s)
    pad = max(wh, ww)

    def sq(img):
        out = np.full((pad, pad), float(np.median(img)), np.float64)
        out[:img.shape[0], :img.shape[1]] = img
        return out

    tgt = sq(cv2.GaussianBlur(cv2.resize(grey(render_rgb), (ww, wh), interpolation=cv2.INTER_AREA), (0, 0), 1.5))
    src = sq(cv2.GaussianBlur(cv2.resize(grey(edit_rgb), (ww, wh), interpolation=cv2.INTER_AREA), (0, 0), 1.5))
    wt = sq(cv2.resize(ndimage.binary_dilation(alpha > 0.5, iterations=12).astype(np.float64), (ww, wh)) > 0.5)
    primary = refine_affine(tgt, src, wt, np.array([[1.0, 0, 0], [0, 1.0, 0]]))
    figure = cv2.resize((alpha > 0.5).astype(np.uint8), (ww, wh), interpolation=cv2.INTER_NEAREST) > 0
    diagnostics = _affine_degeneracy(primary, figure)
    fallback = bool(diagnostics["reasons"])
    M = _register_similarity(render_rgb, alpha, edit_rgb, wh, ww) if fallback else primary
    # work coords are a uniform scaling of render and edit coords
    sx_r, sx_e = ww / w, ww / ew
    A = np.diag([1 / sx_e, 1 / sx_e, 1.0]) @ np.vstack([M, [0, 0, 1]]) @ np.diag([sx_r, sx_r, 1.0])
    xs, ys = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    mapx = A[0, 0] * xs + A[0, 1] * ys + A[0, 2]
    mapy = A[1, 0] * xs + A[1, 1] * ys + A[1, 2]
    info = {"affine_render_to_edit": np.round(A[:2], 5).tolist(),
            "scale": round(math.sqrt(abs(np.linalg.det(A[:2, :2]))) * w / ew, 4),
            "method": "similarity_fallback" if fallback else "affine",
            "affine_diagnostics": diagnostics}
    warped = warp_grid(edit_rgb, mapx - 0.5, mapy - 0.5)
    return mapx, mapy, warped, info


def part_masks(client, image, shape, mapx=None, mapy=None):
    """SAM 3.1 masks (union of instances) per part, in render pixels (warped by the map when given)."""
    out = {}
    for part in PARTS:
        m = np.zeros(image.size[::-1], bool)
        for inst in client.segment(image, part):
            m |= inst.mask
        if mapx is not None:
            m = cv2.remap(m.astype(np.uint8), (mapx - 0.5).astype(np.float32), (mapy - 0.5).astype(np.float32),
                          cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        out[part] = m
    return out


def part_gate(parts_r, parts_e, fig):
    """(ok, record, disagreement mask): per part, the share of the repaint's part outside the render's, and IoU."""
    area = max(1, int(fig.sum()))
    rec, ok, disagree = {}, True, np.zeros_like(fig)
    for part in PARTS:
        a, b = parts_r[part] & fig, parts_e[part] & fig
        if b.sum() < PART_MIN_AREA * area:
            rec[part] = None
            continue
        extra = float((b & ~a).sum() / b.sum())
        rec[part] = {"outside": round(extra, 3), "iou": round(float((a & b).sum() / max(1, (a | b).sum())), 3)}
        ok &= extra <= PART_EXTRA_MAX
        disagree |= a ^ b
    return ok, rec, disagree


def hair_allowed(client, inp, edit, mapx, mapy, shape, *, hair_only=True):
    """Every pass preserves hair disagreements; hair passes also protect non-hair head pixels."""
    head = np.zeros(shape, bool)
    if hair_only:
        for inst in client.segment(inp, "head"):
            head |= inst.mask
    render_hair = np.zeros(shape, bool)
    for inst in client.segment(inp, "hair"):
        render_hair |= inst.mask
    hair = part_masks_text(client, edit, "hair", mapx, mapy)
    both = hair & render_hair
    allowed = ~(hair ^ render_hair)
    return allowed & ~(head & ~both) if hair_only else allowed


def part_masks_text(client, image, text, mapx, mapy):
    m = np.zeros(image.size[::-1], bool)
    for inst in client.segment(image, text):
        m |= inst.mask
    return cv2.remap(m.astype(np.uint8), (mapx - 0.5).astype(np.float32), (mapy - 0.5).astype(np.float32),
                     cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0


def hair_weight(client, inp, edit, render_rgb, warped, alpha, mapx, mapy, face_r, *, exterior_mask=True):
    """The weight map of an accepted hair-mode repaint (module notes)."""
    fig_r = alpha > 0.5
    fig_e = figure_mask(warped, render_figure=fig_r if exterior_mask else None)
    _, agree, _ = interior_agreement(render_rgb, warped, fig_r, fig_e)
    w = cv2.GaussianBlur(agree.astype(np.float32), (0, 0), 2.0)
    if face_r is not None and face_r.any():
        w *= ~ndimage.binary_dilation(face_r, iterations=max(1, round(FACE_KEEP_PX * max(fig_r.shape) / 1024)))
    return w * (agree > 0) * hair_allowed(client, inp, edit, mapx, mapy, fig_r.shape)


def masked_blur(img, m, sigma):
    m = m.astype(np.float32)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    if img.ndim == 3:
        num = np.stack([cv2.GaussianBlur(img[..., c] * m, (0, 0), sigma) for c in range(img.shape[2])], -1)
        return num / np.maximum(den, 1e-4)[..., None]
    return cv2.GaussianBlur(img * m, (0, 0), sigma) / np.maximum(den, 1e-4)


def _features(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.stack([np.ones_like(r), r, g, b, r * r, g * g, b * b, r * g, r * b, g * b], -1)


def colour_match(render_rgb, warped, both, mapped_only=False):
    """Map the repaint's palette onto the render's (SAM's colours), keeping the repaint's detail: a ridge-
    regularised quadratic colour change (repaint -> render) fitted on both images blurred to the render's own
    softness inside the shared figure, with the worst 15% of pixels dropped and refitted; then the remaining
    low-frequency difference (COLOUR_SIGMA) is added back (skipped with `mapped_only`)."""
    sig = 0.004 * max(both.shape)
    m = ndimage.binary_erosion(both, iterations=3)
    src = cv2.GaussianBlur(warped, (0, 0), sig)[m]
    dst = cv2.GaussianBlur(render_rgb, (0, 0), sig)[m]
    if len(src) > 200000:
        pick = np.random.default_rng(0).choice(len(src), 200000, replace=False)
        src, dst = src[pick], dst[pick]
    if len(src) < 10:
        return warped.copy()
    F = _features(src)
    lam = 1e-2 * len(F) * np.eye(F.shape[1])
    lam[0, 0] = 0
    keep = np.ones(len(F), bool)
    for _ in range(2):
        W = np.linalg.solve(F[keep].T @ F[keep] + lam, F[keep].T @ (dst - src)[keep])
        res = np.linalg.norm(dst - src - F @ W, axis=1)
        keep = res <= np.quantile(res, 0.85)
    mapped = np.clip(warped + _features(warped) @ W, 0, 1).astype(np.float32)
    if mapped_only:
        return mapped
    corr = masked_blur((render_rgb - mapped).astype(np.float32), both, COLOUR_SIGMA * max(both.shape))
    return np.clip(mapped + corr, 0, 1)


def interior_agreement(render_rgb, warped, fig_r, fig_e):
    """Hair mode: where the repaint's content agrees with the render's inside the figure. The repaint is mapped to
    the render's palette (without the low-frequency residual, which would hide disagreement), both are blurred by
    AGREE_SIGMA, and a pixel agrees fully when its largest channel difference is under AGREE_TOL[0], not at all over
    AGREE_TOL[1]. Returns (share of the eroded figure that agrees, soft agreement map, eroded figure)."""
    size = max(fig_r.shape)
    both = fig_r & fig_e
    mapped = colour_match(render_rgb, warped, both, mapped_only=True)
    sig = AGREE_SIGMA * size
    d = np.abs(cv2.GaussianBlur(mapped, (0, 0), sig) - cv2.GaussianBlur(render_rgb, (0, 0), sig)).max(-1)
    lo, hi = AGREE_TOL
    agree = np.clip((hi - d) / (hi - lo), 0, 1)
    e = max(1, round(INTERIOR_ERODE * size / 1024))
    interior = ndimage.binary_erosion(fig_r, iterations=e) & ndimage.binary_erosion(fig_e, iterations=max(1, e // 2))
    frac = float((agree[interior] > 0.5).mean()) if interior.any() else 0.0
    return frac, agree * interior, interior


def feather_map(alpha, depth, valid, cam_k, height):
    fig = alpha > 0.5
    edges = boundary(fig)
    d = np.where(valid, depth, np.nan)
    jump = np.zeros_like(fig)
    thr = JUMP * height
    for ax in (0, 1):
        dd = np.abs(np.diff(d, axis=ax))
        j = np.nan_to_num(dd, nan=0.0) > thr
        if ax == 0:
            jump[1:] |= j
            jump[:-1] |= j
        else:
            jump[:, 1:] |= j
            jump[:, :-1] |= j
    dist = ndimage.distance_transform_edt(~(edges | jump))
    f = np.clip((dist - 1) / FEATHER_PX, 0, 1)
    return (f * f * (3 - 2 * f)) * fig


def apply_exposure(rgb, figure, gain, offset=0.0):
    """Apply `(gain, offset)` inside `figure`, leaving exterior background pixels untouched."""
    if gain == 1.0 and offset == 0.0:
        return rgb
    fig = np.asarray(figure, bool)
    adjusted = np.clip(rgb * np.float32(gain) + np.float32(offset), 0.0, 1.0).astype(np.float32)
    return np.where(fig[..., None], adjusted, rgb.astype(np.float32))


def solve_exposure(views, height, *, mode=None, sigma_px=None, gain_clip=None, offset_clip=None, reg=None):
    """Solve per-camera exposure over shared 3D surface samples, anchored to accepted non-hair views."""
    mode = EXPOSURE_MODE if mode is None else mode
    if mode not in ("scalar", "affine"):
        raise ValueError(f"unknown exposure mode {mode!r} (use scalar or affine)")
    sigma_px = EXPOSURE_SIGMA_PX if sigma_px is None else sigma_px
    gain_clip = EXPOSURE_GAIN_CLIP if gain_clip is None else gain_clip
    offset_clip = EXPOSURE_OFFSET_CLIP if offset_clip is None else offset_clip
    reg = EXPOSURE_REG if reg is None else reg

    n_views = len(views)
    if n_views < 2:
        return {v["key"]: (1.0, 0.0) for v in views}

    w_luma = np.asarray(LUMA_WEIGHTS, dtype=np.float32)
    ppus = [px_per_unit(v["cam"]) for v in views]
    min_ppu = min(ppus)
    ref_idx = int(np.argmin(ppus))
    res_scale = min(views[ref_idx]["valid"].shape) / 1024.0

    lumas, masks, mean_luma, area_w = [], [], [], []
    min_mask_px = max(8, round(64 * res_scale * res_scale))
    for idx, v in enumerate(views):
        erode_px = max(1, round(3.0 * res_scale * ppus[idx] / min_ppu))
        raw_mask = np.asarray(v["valid"], bool) & (np.asarray(v["weight"]) > 0.5)
        mask = ndimage.binary_erosion(raw_mask, iterations=erode_px) if raw_mask.any() else raw_mask
        if int(mask.sum()) < min_mask_px:
            mask = raw_mask
        luma = (v["refined"][..., :3] @ w_luma).astype(np.float32)
        sig = max(0.5, sigma_px * res_scale * (ppus[idx] / min_ppu))
        lumas.append(masked_blur(luma, mask, sig))
        masks.append(mask)
        if mask.any():
            mean_luma.append(float(luma[mask].mean()))
            area_w.append(float(mask.sum()) / ((ppus[idx] / 1000.0) ** 2))
        else:
            mean_luma.append(0.5)
            area_w.append(1.0)

    overlaps = []
    adj = [set() for _ in range(n_views)]
    for i in range(n_views):
        for j in range(i + 1, n_views):
            vi, vj = views[i], views[j]
            delta = abs((vi["cam"]["angle"] - vj["cam"]["angle"] + 180) % 360 - 180)
            hair_i, hair_j = bool(vi.get("hair", False)), bool(vj.get("hair", False))
            close_i, close_j = bool(vi.get("closeup", False)), bool(vj.get("closeup", False))
            if delta > 90:
                continue
            if hair_i != hair_j and delta > 45:
                continue
            if (close_i != close_j) and not (hair_i or hair_j) and delta > 45:
                continue

            stride = max(1, min(4, min(max(masks[i].shape), max(masks[j].shape)) // 128))
            min_dir = max(8, round(32 * (res_scale / (stride / 4.0)) ** 2))
            li_parts, lj_parts = [], []
            for s_idx, d_idx, flip in ((i, j, False), (j, i, True)):
                vs, vd = views[s_idx], views[d_idx]
                ms, md = masks[s_idx], masks[d_idx]
                ys, xs = np.where(ms[::stride, ::stride])
                ys *= stride
                xs *= stride
                if len(ys) == 0:
                    continue
                pts = vs["pos"][ys, xs]
                px, py, _ = project(vd["cam"], pts)
                xi, yi = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
                hd, wd = md.shape
                ins = (xi >= 0) & (xi < wd) & (yi >= 0) & (yi < hd)
                xic, yic = np.clip(xi, 0, wd - 1), np.clip(yi, 0, hd - 1)
                tol = 2.0 / min(ppus[s_idx], ppus[d_idx]) + 0.001 * height
                keep = ins & md[yic, xic] & (np.linalg.norm(pts - vd["pos"][yic, xic], axis=1) <= tol)
                if int(keep.sum()) < min_dir:
                    continue
                ls = lumas[s_idx][ys[keep], xs[keep]]
                ld = lumas[d_idx][yic[keep], xic[keep]]
                if not flip:
                    li_parts.append(ls)
                    lj_parts.append(ld)
                else:
                    li_parts.append(ld)
                    lj_parts.append(ls)

            if not li_parts:
                continue
            li = np.concatenate(li_parts).astype(np.float64)
            lj = np.concatenate(lj_parts).astype(np.float64)
            if len(li) < 2 * min_dir:
                continue
            scale_factor = ((min_ppu / ppus[i]) ** 2 + (min_ppu / ppus[j]) ** 2) * 0.5
            area_1024 = len(li) * (stride ** 2) * scale_factor / max(res_scale ** 2, 1e-6)
            eff_w = (area_1024 / 16.0) * math.cos(math.radians(delta * 0.6))
            overlaps.append((i, j, li, lj, eff_w))
            adj[i].add(j)
            adj[j].add(i)

    if not overlaps:
        return {v["key"]: (1.0, 0.0) for v in views}

    u = np.zeros(n_views, np.float64)
    for _ in range(6):
        H = np.eye(n_views, dtype=np.float64) * reg
        rhs = np.zeros(n_views, np.float64)
        for i, j, li, lj, eff_w in overlaps:
            hair_i, hair_j = bool(views[i].get("hair", False)), bool(views[j].get("hair", False))
            pair_w = 0.20 if (hair_i != hair_j) else (0.50 if hair_i else 1.0)
            gi, gj = math.exp(u[i]), math.exp(u[j])
            res = gi * li - gj * lj
            med = float(np.median(res))
            mad = float(np.median(np.abs(res - med))) + 1e-4
            inlier = (np.abs(res - med) <= max(0.08, 2.5 * mad)) & (np.abs(li - lj) <= 0.22)
            inlier_frac = float(inlier.mean())
            if inlier_frac < 0.35:
                inlier = np.abs(res - med) <= np.quantile(np.abs(res - med), 0.50)
                pair_w *= 0.15
            si2 = float((li[inlier] ** 2).mean())
            sj2 = float((lj[inlier] ** 2).mean())
            if si2 < 1e-3 or sj2 < 1e-3:
                continue
            target = 0.5 * math.log(sj2 / si2)
            rms_scale = math.sqrt(math.sqrt(si2 * sj2))
            conf = pair_w * (rms_scale ** 2) * min(2.0, eff_w * inlier_frac / 2500.0)
            pair_res = rms_scale * ((u[i] - u[j]) - target)
            if abs(pair_res) > 0.06:
                conf *= 0.06 / abs(pair_res)
            H[i, i] += conf
            H[j, j] += conf
            H[i, j] -= conf
            H[j, i] -= conf
            rhs[i] += conf * target
            rhs[j] -= conf * target
        u = np.linalg.solve(H, rhs)

    gains = np.exp(u)
    seen = [False] * n_views
    components = []
    for start in range(n_views):
        if seen[start]:
            continue
        comp, stack = [], [start]
        seen[start] = True
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        components.append(comp)

    for comp in components:
        if len(comp) == 1:
            gains[comp[0]] = 1.0
            continue
        anchor_idx = [i for i in comp if not views[i].get("hair", False)] or comp
        num = sum(area_w[i] * mean_luma[i] for i in anchor_idx)
        den = sum(area_w[i] * gains[i] * mean_luma[i] for i in anchor_idx)
        if den > 1e-8:
            for i in comp:
                gains[i] *= num / den
    gains = np.clip(gains, gain_clip[0], gain_clip[1])

    if mode == "scalar":
        return {views[i]["key"]: (float(gains[i]), 0.0) for i in range(n_views)}

    H_b = np.eye(n_views, dtype=np.float64) * 0.25
    rhs_b = np.zeros(n_views, np.float64)
    for i, j, li, lj, eff_w in overlaps:
        hair_i, hair_j = bool(views[i].get("hair", False)), bool(views[j].get("hair", False))
        pair_w = 0.20 if (hair_i != hair_j) else 1.0
        inlier = np.abs(li - lj) <= 0.22
        if int(inlier.sum()) < 16:
            continue
        diff = float((gains[i] * li[inlier] - gains[j] * lj[inlier]).mean())
        conf = pair_w * min(2.0, eff_w * float(inlier.mean()) / 2500.0)
        H_b[i, i] += conf
        H_b[j, j] += conf
        H_b[i, j] -= conf
        H_b[j, i] -= conf
        rhs_b[i] -= conf * diff
        rhs_b[j] += conf * diff
    b = np.linalg.solve(H_b, rhs_b)
    for comp in components:
        if len(comp) == 1:
            b[comp[0]] = 0.0
            continue
        anchor_idx = [i for i in comp if not views[i].get("hair", False)] or comp
        tot_a = sum(area_w[i] for i in anchor_idx)
        shift = sum(area_w[i] * b[i] for i in anchor_idx) / max(tot_a, 1e-8)
        for i in comp:
            b[i] -= shift
    b = np.clip(b, -offset_clip, offset_clip)
    return {views[i]["key"]: (float(gains[i]), float(b[i])) for i in range(n_views)}


class Texture:
    def __init__(self, prep, info):
        self.prep = Path(prep)
        P, N, cov = C.texel_maps(self.prep)
        base = np.asarray(Image.open(self.prep / "base.png").convert("RGB"), dtype=np.float32) / 255
        self._init_arrays(P, N, cov, base, info["height"])

    @classmethod
    def from_arrays(cls, pos, nrm, cov, base_rgb, height):
        self = cls.__new__(cls)
        self.prep = None
        self._init_arrays(pos, nrm, cov, np.asarray(base_rgb, dtype=np.float32), height)
        return self

    def _init_arrays(self, pos, nrm, cov, base, height):
        cov = np.asarray(cov, bool)
        if not cov.any():
            raise ValueError("the refined atlas has no covered texels")
        self.cov = cov
        self.idx = np.flatnonzero(cov.ravel())
        self.P = pos.reshape(-1, 3)[self.idx].astype(np.float64)
        self.N = nrm.reshape(-1, 3)[self.idx].astype(np.float64)
        self.base = base.astype(np.float32)
        r0, c0 = np.divmod(self.idx, self.base.shape[1])
        self.base_cov = self.base[r0, c0].copy()
        k_body = max(C.BODY_RES) / (height * C.BODY_FRAME)
        self.q0 = float(Q0_FACING * k_body)
        _, near = ndimage.distance_transform_edt(~cov, return_indices=True)
        # every texel no triangle covers takes its nearest covered texel's colour: island margins, and the spots of
        # sliver triangles too thin to cover a texel centre (left black by the bake, visible as dark specks)
        self.margin = ~cov
        self.margin_src = (near[0][self.margin], near[1][self.margin])
        self.height = height
        self.reset()

    def reset(self):
        self.T = self.base.copy()
        self.T[self.margin] = self.T[self.margin_src]
        self.U = np.ones(len(self.idx), np.float64)
        self.R = np.zeros((len(self.idx), 3), np.float32)
        self.Q_ref = np.zeros(len(self.idx), np.float64)
        self.Q = np.full(len(self.idx), self.q0, np.float64)

    def save(self, path):
        t = self.T.copy()
        t[self.margin] = t[self.margin_src]
        Image.fromarray((np.clip(t, 0, 1) * 255).round().astype(np.uint8)).save(path)
        return path

    def render_view(self, cam, depth_img, valid_img):
        """Project current covered atlas colours onto `cam`'s pixel grid, returning `(rgb, mask)`."""
        h, w = valid_img.shape
        x, y, d = project(cam, self.P)
        xi, yi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        ins = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        xic, yic = np.clip(xi, 0, w - 1), np.clip(yi, 0, h - 1)
        k = px_per_unit(cam)
        eps = 3.0 / k + 0.001 * self.height
        front = ndimage.minimum_filter(np.where(valid_img, depth_img, np.inf), size=VIS_WINDOW)
        _, _, fwd = axes(cam)
        vis = ins & valid_img[yic, xic] & (d <= front[yic, xic] + eps) & ((self.N @ fwd) < 0)
        r_uv, c_uv = np.divmod(self.idx[vis], self.T.shape[1])
        acc = np.zeros((h, w, 3), np.float32)
        cnt = np.zeros((h, w), np.float32)
        np.add.at(acc, (yic[vis], xic[vis]), self.T[r_uv, c_uv])
        np.add.at(cnt, (yic[vis], xic[vis]), 1.0)
        has = cnt > 0
        out = np.full((h, w, 3), C.BG, np.float32)
        out[has] = acc[has] / cnt[has, None]
        return out, has

    def apply(self, cam, refined, feather, depth_img, valid_img, weight_mask, bonus=1.0):
        """Project `refined` (float RGB on the render grid) onto the texels this camera sees."""
        x, y, d = project(cam, self.P)
        h, w = feather.shape
        xi = np.floor(x).astype(np.int64)
        yi = np.floor(y).astype(np.int64)
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        xi, yi = np.clip(xi, 0, w - 1), np.clip(yi, 0, h - 1)
        k = px_per_unit(cam)
        eps = 3.0 / k + 0.001 * self.height
        front = ndimage.minimum_filter(np.where(valid_img, depth_img, np.inf), size=VIS_WINDOW)
        vis = inside & valid_img[yi, xi] & (d <= front[yi, xi] + eps) & (weight_mask[yi, xi] > 0)
        _, _, fwd = axes(cam)
        cos = np.clip(-(self.N @ fwd), 0, 1)
        q = cos ** 2 * k * bonus
        if getattr(self, "U", None) is None:
            t = np.clip((q / self.Q - 1) / RAMP, 0, 1)
            a = vis * (t * t * (3 - 2 * t))
            cand = np.flatnonzero(a > 1e-3)
            xs, ys = x[cand], y[cand]
            a_c = a[cand] * sample(feather.astype(np.float32), xs, ys) * sample(weight_mask.astype(np.float32), xs, ys)
            keep = a_c > 1e-3
            cand, xs, ys, a_c = cand[keep], xs[keep], ys[keep], a_c[keep]
            if not len(cand):
                return 0
            cols = sample(refined.astype(np.float32), xs, ys)
            ii = self.idx[cand]
            r, c = np.divmod(ii, self.T.shape[1])
            av = a_c[:, None]
            self.T[r, c] = self.T[r, c] * (1 - av) + cols * av
            self.Q[cand] = self.Q[cand] + a_c * (q[cand] - self.Q[cand])
            return int(len(cand))

        t0 = np.clip((q / self.q0 - 1) / RAMP, 0, 1)
        s0 = vis * (t0 * t0 * (3 - 2 * t0))
        has_ref = self.Q_ref > 0
        t_ref = np.where(has_ref, np.clip((q / np.maximum(self.Q_ref, 1e-12) - 1) / RAMP, 0, 1), 0.0)
        s_ref = vis * (t_ref * t_ref * (3 - 2 * t_ref))
        eff = self.U * s0 + (1.0 - self.U) * s_ref
        cand = np.flatnonzero(eff > 1e-3)
        if not len(cand):
            return 0
        xs, ys = x[cand], y[cand]
        w_sp = (sample(feather.astype(np.float32), xs, ys)
                * sample(weight_mask.astype(np.float32), xs, ys)).astype(np.float64)
        keep = (eff[cand] * w_sp) > 1e-3
        cand, xs, ys, w_sp = cand[keep], xs[keep], ys[keep], w_sp[keep]
        if not len(cand):
            return 0
        cols = sample(refined.astype(np.float32), xs, ys).astype(np.float32)
        u_old = self.U[cand]
        m_old = 1.0 - u_old
        du = u_old * (s0[cand] * w_sp)
        u_new = u_old - du
        m_new = m_old + du
        a_ref = s_ref[cand] * w_sp
        m_keep = m_old * (1.0 - a_ref)
        m_add = m_old * a_ref + du
        denom = np.maximum(m_new, 1e-12)
        r_new = ((m_keep[:, None] * self.R[cand] + m_add[:, None] * cols) / denom[:, None]).astype(np.float32)
        q_new = (m_keep * self.Q_ref[cand] + m_add * q[cand]) / denom
        self.U[cand] = u_new
        self.R[cand] = r_new
        self.Q_ref[cand] = q_new
        self.Q[cand] = u_new * self.q0 + m_new * q_new
        ii = self.idx[cand]
        r, c = np.divmod(ii, self.T.shape[1])
        self.T[r, c] = (u_new[:, None] * self.base_cov[cand] + m_new[:, None] * r_new).astype(np.float32)
        return int(len(cand))


def load_accepted_views(work, view_records, height, *, body_cam=None):
    """Load saved per-view render masks, 3D positions and accepted repaints from `work`."""
    from .cameras import decode_pos, read_exr

    work = Path(work)
    views = []
    for key, rec in view_records.items():
        if not rec.get("accepted"):
            continue
        cam = rec["camera"]
        cname = cam["name"]
        folder = work / key
        with Image.open(folder / f"{cname}.png") as rgba:
            alpha = np.asarray(rgba.convert("RGBA"))[..., 3].astype(np.float32) / 255.0
        pos, valid = decode_pos(read_exr(folder / f"{cname}.pos.exr"))
        _, _, depth = project(cam, pos)
        with Image.open(folder / "refined.png") as img:
            refined = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(folder / "weight.png") as img:
            weight = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        hair = key.startswith("hair-")
        closeup = bool(body_cam is not None and is_closeup(cam, body_cam)) or ("frame_border_px" in cam)
        views.append({
            "key": key,
            "cam": cam,
            "hair": hair,
            "closeup": closeup,
            "alpha": alpha,
            "pos": pos.astype(np.float32),
            "valid": valid,
            "depth": depth.astype(np.float32),
            "refined": refined,
            "weight": weight,
            "base_feather": feather_map(alpha, depth, valid, px_per_unit(cam), height).astype(np.float32),
        })
    return views


def rebake_views(tex, work=None, view_records=None, *, body_cam=None, views=None, solve=None,
                 feather_closeups=None, feather_px=None, mode=None):
    """Solve cross-view exposure on accepted views and re-project them coarse-to-fine onto `tex`."""
    if not isinstance(getattr(tex, "U", None), np.ndarray):
        return {}
    solve = EXPOSURE_SOLVE if solve is None else solve
    feather_closeups = CLOSEUP_FEATHER if feather_closeups is None else feather_closeups
    feather_px = CLOSEUP_FEATHER_PX if feather_px is None else feather_px
    mode = EXPOSURE_MODE if mode is None else mode
    if views is None:
        if work is None or view_records is None:
            return {}
        views = load_accepted_views(work, view_records, tex.height, body_cam=body_cam)
    if not views:
        return {}
    gains = (solve_exposure(views, tex.height, mode=mode) if solve
             else {v["key"]: (1.0, 0.0) for v in views})
    tex.reset()
    ordered = sorted(
        views,
        key=lambda v: (
            2 if "frame_border_px" in v["cam"] else (1 if v["hair"] else 0),
            px_per_unit(v["cam"]) * (HAIR_BONUS if v["hair"] else 1.0),
        ),
    )
    for v in ordered:
        g, b = gains[v["key"]]
        fig = v["alpha"] > 0.5
        rgb = apply_exposure(v["refined"], fig, g, b)
        feather = (v["base_feather"].copy() if "base_feather" in v
                   else feather_map(v["alpha"], v["depth"], v["valid"], px_per_unit(v["cam"]), tex.height).astype(np.float32))
        if "frame_border_px" in v["cam"]:
            feather *= frame_feather(v["alpha"].shape, v["cam"]["frame_border_px"])
        elif feather_closeups and (v["hair"] or v["closeup"]):
            feather *= closeup_feather(fig, feather_px)
        weight = v["weight"].copy()
        if v["hair"]:
            cur_rgb, has_cur = tex.render_view(v["cam"], v["depth"], v["valid"])
            sig = AGREE_SIGMA * max(weight.shape)
            m_valid = has_cur & fig
            if m_valid.any():
                d_lf = np.abs(masked_blur(rgb, m_valid, sig) - masked_blur(cur_rgb, m_valid, sig)).max(-1)
                lo, hi = AGREE_TOL
                weight *= np.where(has_cur, np.clip((hi - d_lf) / (hi - lo), 0.0, 1.0), 1.0).astype(np.float32)
        tex.apply(v["cam"], rgb, feather, v["depth"], v["valid"], weight, HAIR_BONUS if v["hair"] else 1.0)
    if mode == "affine":
        return {k: {"gain": round(g, 5), "offset": round(b, 5)} for k, (g, b) in gains.items()}
    return {k: round(g, 5) for k, (g, _) in gains.items()}
