"""Registration, measured repaint gates, colour matching and visible texel projection."""
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from herogen.pipeline.facesource import refine_affine, sample
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
COLOUR_SIGMA = 0.04    # residual low-frequency colour match, as a fraction of the render's larger side
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


def figure_mask(rgb, *, render_figure=None):
    m = (np.abs(rgb - C.BG) > BG_TOL).any(-1)
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


def register(render_rgb, alpha, edit_rgb):
    """Map from render pixel edge coordinates to edit pixel edge coordinates: returns (mapx, mapy) over the
    render grid (pixel centres), the warped edit at render resolution, and measurements."""
    h, w = alpha.shape
    eh, ew = edit_rgb.shape[:2]
    s = WORK / max(h, w)
    wh, ww = round(h * s), round(w * s)
    pad = max(wh, ww)

    def sq(img):  # top-left into a square work canvas (refine_affine works on squares)
        out = np.full((pad, pad), float(np.median(img)), np.float64)
        out[: img.shape[0], : img.shape[1]] = img
        return out

    tgt = sq(cv2.GaussianBlur(cv2.resize(grey(render_rgb), (ww, wh), interpolation=cv2.INTER_AREA), (0, 0), 1.5))
    src = sq(cv2.GaussianBlur(cv2.resize(grey(edit_rgb), (ww, wh), interpolation=cv2.INTER_AREA), (0, 0), 1.5))
    wt = sq(cv2.resize(ndimage.binary_dilation(alpha > 0.5, iterations=12).astype(np.float64), (ww, wh)) > 0.5)
    M0 = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    M = refine_affine(tgt, src, wt, M0)
    if M is None:
        M = M0
    # work coords are a uniform scaling of render and edit coords
    sx_r, sx_e = ww / w, ww / ew
    A = np.diag([1 / sx_e, 1 / sx_e, 1.0]) @ np.vstack([M, [0, 0, 1]]) @ np.diag([sx_r, sx_r, 1.0])
    xs, ys = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    mapx = A[0, 0] * xs + A[0, 1] * ys + A[0, 2]
    mapy = A[1, 0] * xs + A[1, 1] * ys + A[1, 2]
    info = {"affine_render_to_edit": np.round(A[:2], 5).tolist(),
            "scale": round(math.sqrt(abs(np.linalg.det(A[:2, :2]))) * w / ew, 4)}
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


class Texture:
    def __init__(self, prep, info):
        self.prep = Path(prep)
        P, N, cov = C.texel_maps(self.prep)
        if not cov.any():
            raise ValueError("the refined atlas has no covered texels")
        self.cov = cov
        self.idx = np.flatnonzero(cov.ravel())
        self.P = P.reshape(-1, 3)[self.idx].astype(np.float64)
        self.N = N.reshape(-1, 3)[self.idx].astype(np.float64)
        base = np.asarray(Image.open(self.prep / "base.png").convert("RGB"), dtype=np.float32) / 255
        self.T = base
        k_body = max(C.BODY_RES) / (info["height"] * C.BODY_FRAME)
        self.Q = np.full(len(self.idx), Q0_FACING * k_body, np.float64)
        dist, near = ndimage.distance_transform_edt(~cov, return_indices=True)
        # every texel no triangle covers takes its nearest covered texel's colour: island margins, and the spots of
        # sliver triangles too thin to cover a texel centre (left black by the bake, visible as dark specks)
        self.margin = ~cov
        self.margin_src = (near[0][self.margin], near[1][self.margin])
        self.T[self.margin] = self.T[self.margin_src]
        self.height = info["height"]

    def save(self, path):
        t = self.T.copy()
        t[self.margin] = t[self.margin_src]
        Image.fromarray((np.clip(t, 0, 1) * 255).round().astype(np.uint8)).save(path)
        return path

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
        sel = cand
        a = np.zeros_like(a)
        a[sel] = a_c
        ii = self.idx[sel]
        r, c = np.divmod(ii, self.T.shape[1])
        av = a[sel][:, None]
        self.T[r, c] = self.T[r, c] * (1 - av) + cols * av
        self.Q[sel] = self.Q[sel] + a[sel] * (q[sel] - self.Q[sel])
        return int(len(sel))
