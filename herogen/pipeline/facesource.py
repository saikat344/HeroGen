"""Stage 2f, the face source: sharp close-ups of the head that stage 4 bakes into the body's own Head texture.

A turnaround view gives the face only a few dozen pixels. For each of the front, left and right views this stage
cuts a square around the head, asks Muse Image for a faithful upscale of that crop (`PROMPTS`: the same framing
and angle, every shape and colour where it is, flat areas kept flat, no shading, texture or detail that is not
in the input), registers the close-up back onto the view and records how well it agrees. Inside the close-up's
footprint, stage 4 then adds the close-up's detail above the view's own resolution to the Head's texture; the
view's colours stay (`herogen.blender.texture`).

Crops. The front crop (`front_crop_box`) is a square anchored on the front view's SAM 3.1 face box (stage 2's
part mask `masks/front-face.png`): its top is the head box's top (`masks/front-head.png`), but no more than 1.2
face heights above the face (0.6 without a head mask); its bottom is 0.45 face heights below the chin; its side is
the largest of twice the face size, the top-to-bottom height and 1.1 times the head width (the head width capped
at three face sizes, so a hat brim or hood is kept but a cape is not); `CROP_TOP_PAD` of the side is added above
the top. Without a face mask (a full helmet) the square is the head box grown by `CROP_MARGIN` on each side, or,
for a head box much taller than wide (a spire, long hair), a square of 1.3 head widths sitting on the box's
bottom. The profile crops (`profile_crop_box`) reuse the front crop's rows, since the turnaround views share one
pixel scale and the feet baseline, shifted by the difference between the two figure masks' bottom rows, and are
centred on the side figure within the top 80% of those rows; the side is the front crop's, or 1.15 times the
side figure's width there if that is wider. Each crop is resampled to `INPUT_SIDE` px (Lanczos; outside the view
the border colour) and sent with `EDIT_SIZE`.

Registration (`register`). The close-up maps back to the view by an affine transform A (close-up pixel edge
coordinates to view pixel edge coordinates). It starts from the crop box (a uniform scale and a shift), then a
coarse-to-fine Gauss-Newton fit of a full affine with gain and bias (`refine_affine`) aligns the close-up,
blurred to the view's own resolution, with the view crop at `WORK` px, weighted to the head region. The fitted
transform is kept only when it agrees better than the crop box (normalised cross-correlation of grey levels in
the head region at `WORK` px) and changes the scale by less than a factor `MAX_RESCALE`. Agreement is then
measured in the view's own pixels (`ncc_view`): the close-up, prefiltered to the view's resolution and warped by
A, against the view, inside the check region (the front view's face mask, else its head mask; for a profile, the
SAM 3.1 head mask of that view, else the figure inside the crop).

Gates, per view (`NCC_MIN`, `REFRAME_MAX`): a close-up is used only when its NCC in the view reaches the view's
threshold and its scale stays within `REFRAME_MAX` of the crop box's (Muse kept the framing). The thresholds come
from the tuning characters only (see the constants).

Footprint. Where stage 4 may use a close-up: the view's SAM 3.1 head mask (the front's from stage 2; a profile's
from this stage), grown by 2 px, inside the crop box; without a head mask, the figure inside the crop box. Saved as
`masks/closeup-<view>.png`; stage 4 feathers its edge.

Outputs: `views/closeup-<view>.png` (the close-up as Muse Image returned it), `views/closeup-<view>-input.png`
(the crop sent), `masks/closeup-<view>.png` (the footprint) and `views/closeup.json`: per view the recipe, the
crop box, A, the measurements, `used` and, when a close-up is not used, why; plus the SHA-256 of the view image
it was made from, so stage 4 ignores a close-up of an older turnaround. A view that fails keeps no image; the
record is written in every case, also when no view is used (then with each view's reason, or the stage's
`fallback`).

This stage never fails a hero over its own failure. Any error (a Model API failure after the client's retries, a
missing input, a bug) is recorded with its reason (`fallback`), no close-up image is written, and stage 4 bakes the
plain views as before; the fallback is loud: the CLI prints it to stderr, and stage 4's record notes it. A replay
miss (no cached answer when replaying) is not a model failure but a broken cache or run: the stage's record is
replaced by a failed one (`ok` false, `status` "error", the error, no outputs, no view used) and the miss is raised,
as in every other stage.
Its record is `stage-2f.json` (the same record, per view under `details.closeups`); no check is gated. No file
another stage reads is changed (`views/<view>.png`, `masks/<view>.png` and `views/meta.json` are untouched), so the
held-out views stay valid.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from herogen.pipeline import checks as C
from herogen.pipeline import imaging
from herogen.pipeline.client import SAM_MODEL, ModelClient, ReplayMiss
from herogen.pipeline.runfolder import RunFolder

STAGE = "2f"
STAGE_NAME = "face source"
VIEWS = ("front", "left", "right")
RECORD = "views/closeup.json"
HEAD_TEXT = "head"

# Frozen prompts (a change is a new recipe with a new name, never an edit in place).
PROMPTS = {
    "FC4": (
        "Upscale this image to a high resolution. The result must look exactly like the input, only sharper and "
        "cleaner: the same framing, head size, position and straight-on front angle, every shape, line and "
        "colour in the same place. Keep flat colour areas flat and soft shading soft. Do not add shading, "
        "texture, hair strands, wrinkles, pores, highlights or any detail that is not already in the input. "
        "Do not change the design or the art style. Keep the plain background."
    ),
    "PC4": (
        "Upscale this image to a high resolution. The result must look exactly like the input, only sharper and "
        "cleaner: the same framing, head size, position and strict side-profile angle, every shape, line and "
        "colour in the same place. Keep flat colour areas flat and soft shading soft. Do not add shading, "
        "texture, hair strands, wrinkles, pores, highlights or any detail that is not already in the input. "
        "Do not change the design or the art style. Keep the plain background."
    ),
}
RECIPE = {"front": "FC4", "left": "PC4", "right": "PC4"}

CROP_MARGIN = 0.18
CROP_TOP_PAD = 0.08
INPUT_SIDE = 1024
EDIT_SIZE = "1024x1024"
WORK = 512                 # registration resolution of the crop (px)
MAX_RESCALE = 1.6          # a fitted transform may change the crop box's scale by less than this factor
FOOTPRINT_GROW_PX = 2
FALLBACK_CHARS = 600       # a fallback's reason is kept to this many characters

# Gates, set on the tuning characters only (ground-truth C01-C09, generated s002-s028, s044, s045: 35 front and
# 69 profile close-ups made with these prompts and crops), NCC in the view (`ncc_view`): front min 0.968, median
# 0.997; profiles min 0.987, median 0.997; every one kept the crop's framing (scale ratio 0.97-1.03). For contrast,
# the same characters' "redraw sharper" close-ups, which add shading and detail the view does not have, score
# 0.04-0.97 (median about 0.91). Each threshold sits just under the worst tuning close-up of its kind.
NCC_MIN = {"front": 0.95, "left": 0.97, "right": 0.97}
REFRAME_MAX = 1.25


# ---- crops ----------------------------------------------------------------------------------------------


Box = tuple[float, float, float, float]


def front_crop_box(face_box: Box | None, head_box: Box | None) -> Box | None:
    """The front crop (module notes) from the front view's face and head boxes (l, t, r, b), or None."""
    if face_box is not None:
        fl, ft, fr, fbm = face_box
        fw, fh = fr - fl, fbm - ft
        top = ft - 0.6 * fh
        hw = 0.0
        if head_box is not None:
            top = max(min(top, head_box[1]), ft - 1.2 * fh)
            hw = min(head_box[2] - head_box[0], 3 * max(fw, fh))
        bottom = fbm + 0.45 * fh
        side = max(bottom - top, 2.0 * max(fw, fh), 1.1 * hw)
        cx = (fl + fr) / 2
        y0 = top - CROP_TOP_PAD * side
        return (cx - side / 2, y0, cx + side / 2, y0 + side)
    if head_box is not None:
        l, t, r, b = head_box
        w = r - l
        if b - t > 1.6 * w:
            side = 1.3 * w
            cx = (l + r) / 2
            y1 = b + 0.1 * w
            return (cx - side / 2, y1 - side, cx + side / 2, y1)
        cx, cy = (l + r) / 2, (t + b) / 2
        side = max(r - l, b - t) * (1 + 2 * CROP_MARGIN)
        return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    return None


def profile_crop_box(front_box: Box, front_figure: np.ndarray, side_figure: np.ndarray) -> Box | None:
    """The profile crop (module notes) from the front crop and the two figure masks, or None."""
    fb, sb = imaging.bbox(front_figure), imaging.bbox(side_figure)
    if fb is None or sb is None:
        return None
    l, t, r, _ = front_box
    side = r - l
    t2 = t + (sb[3] - fb[3])
    r0 = int(max(0, math.floor(t2)))
    r1 = int(min(side_figure.shape[0], math.ceil(t2 + 0.8 * side)))
    if r1 <= r0:
        return None
    cols = np.flatnonzero(side_figure[r0:r1].any(axis=0))
    if cols.size == 0:
        return None
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    side = max(side, 1.15 * (c1 - c0))
    cx = (c0 + c1) / 2
    return (cx - side / 2, t2, cx + side / 2, t2 + side)


def crop(img: Image.Image, box: Box, size: int) -> Image.Image:
    """The float box (it may leave the image) resampled to size x size (Lanczos); outside the image, the
    border's median colour."""
    a = np.asarray(img.convert("RGB"))
    fill = tuple(int(v) for v in np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), axis=0))
    l, t, r, b = box
    pad = int(math.ceil(max(0.0, -l, -t, r - img.width, b - img.height))) + 2
    canvas = Image.new("RGB", (img.width + 2 * pad, img.height + 2 * pad), fill)
    canvas.paste(img.convert("RGB"), (pad, pad))
    return canvas.resize((size, size), Image.LANCZOS, box=(l + pad, t + pad, r + pad, b + pad))


def box_mask(shape: tuple[int, int], box: Box) -> np.ndarray:
    """Pixels whose centre lies inside the float box."""
    rows = np.arange(shape[0]) + 0.5
    cols = np.arange(shape[1]) + 0.5
    return ((rows >= box[1]) & (rows < box[3]))[:, None] & ((cols >= box[0]) & (cols < box[2]))[None, :]


# ---- image helpers --------------------------------------------------------------------------------------


GREY = np.array([0.299, 0.587, 0.114])


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (edge-padded) of a [H, W] or [H, W, C] float image; sigma in pixels."""
    if sigma < 0.3:
        return img.astype(np.float64, copy=True)
    r = max(1, int(math.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = img.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        n = out.shape[axis]
        out = sum(w * np.take(p, np.arange(i, i + n), axis=axis) for i, w in enumerate(k))
    return out


def sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample at pixel-edge coordinates (x right, y down; pixel (0, 0) spans [0, 1)), clamped."""
    h, w = img.shape[:2]
    xc = np.clip(x - 0.5, 0, w - 1.000001)
    yc = np.clip(y - 0.5, 0, h - 1.000001)
    x0, y0 = np.floor(xc).astype(np.int64), np.floor(yc).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = xc - x0, yc - y0
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    return ((img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy)
            + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy)


def _grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre edge coordinates (x, y) of every pixel."""
    return np.meshgrid(np.arange(shape[1]) + 0.5, np.arange(shape[0]) + 0.5)


def warp(src: np.ndarray, M: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """``src`` sampled at M (2x3, destination edge coordinates -> source edge coordinates) over ``shape``."""
    x, y = _grid(shape)
    return sample(src, M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2])


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Normalised cross-correlation of two grey images inside ``mask`` (0 when either is flat there)."""
    if mask.sum() < 16:
        return 0.0
    x, y = a[mask] - a[mask].mean(), b[mask] - b[mask].mean()
    den = math.sqrt(float((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / den) if den > 1e-9 else 0.0


def _h(M: np.ndarray) -> np.ndarray:
    return np.vstack([M, [0.0, 0.0, 1.0]])


def refine_affine(target: np.ndarray, source: np.ndarray, weight: np.ndarray, M0: np.ndarray,
                  scales: tuple[int, ...] = (4, 2, 1), iterations: int = 30) -> np.ndarray | None:
    """Affine M (target edge coordinates -> source edge coordinates) that makes ``source`` sampled at M agree
    with ``target`` where ``weight`` > 0: Gauss-Newton on the six affine parameters plus a gain and a bias,
    coarse to fine (``scales``: the downsampling factor of each pass). Grey images of one size. None when the
    fit breaks down (too few weighted pixels, a singular system, a degenerate transform)."""
    M = np.asarray(M0, dtype=np.float64).copy()
    for f in scales:
        if f > 1:
            n = target.shape[0] // f
            t = _downsample(target, f)
            s = _downsample(source, f)
            w = _downsample(weight.astype(np.float64), f) > 0.5
        else:
            n, t, s, w = target.shape[0], target, source, weight > 0
        if w.sum() < 64:
            return None
        S, Si = np.diag([f, f, 1.0]), np.diag([1.0 / f, 1.0 / f, 1.0])
        Mf = (Si @ _h(M) @ S)[:2]
        x, y = _grid((n, n))
        xs, ys, tv = x[w], y[w], t[w]
        gy, gx = np.gradient(s)
        for _ in range(iterations):
            u = Mf[0, 0] * xs + Mf[0, 1] * ys + Mf[0, 2]
            v = Mf[1, 0] * xs + Mf[1, 1] * ys + Mf[1, 2]
            iv = sample(s, u, v)
            ix, iy = sample(gx, u, v), sample(gy, u, v)
            # residual of target ~ gain * source(M x) + bias; parameters: dM (6), gain, bias
            g, b = _gain_bias(iv, tv)
            r = tv - (g * iv + b)
            J = np.stack([g * ix * xs, g * ix * ys, g * ix, g * iy * xs, g * iy * ys, g * iy, iv,
                          np.ones_like(iv)], axis=1)
            H = J.T @ J
            try:
                d = np.linalg.solve(H + 1e-6 * np.eye(8) * np.trace(H) / 8, J.T @ r)
            except np.linalg.LinAlgError:
                return None
            Mf = Mf + d[:6].reshape(2, 3)
            if not np.all(np.isfinite(Mf)) or abs(np.linalg.det(Mf[:, :2])) < 1e-6:
                return None
            if np.abs(d[:6]).max() < 1e-4 * max(1.0, n / 64):
                break
        M = (S @ _h(Mf) @ Si)[:2]
    return M


def _gain_bias(iv: np.ndarray, tv: np.ndarray) -> tuple[float, float]:
    var = float(iv.var())
    g = float(((iv - iv.mean()) * (tv - tv.mean())).mean() / var) if var > 1e-9 else 1.0
    return g, float(tv.mean() - g * iv.mean())


def _downsample(img: np.ndarray, f: int) -> np.ndarray:
    n = img.shape[0] // f
    return blur(img[: n * f, : n * f], 0.5 * f).reshape(n, f, n, f).mean(axis=(1, 3))


# ---- registration ---------------------------------------------------------------------------------------


def register(closeup: np.ndarray, view: np.ndarray, box: Box, region: np.ndarray, check: np.ndarray
             ) -> dict[str, Any]:
    """Register a close-up (float RGB, square) to its view (float RGB) from the crop box it was made from.

    ``region`` weights the fit (the head), ``check`` is where agreement is measured; both are view masks.
    Returns A (close-up edge coordinates -> view edge coordinates, 2x3 list), which start it came from and
    the measurements (module notes)."""
    S = closeup.shape[0]
    l, t, r, _ = box
    side = r - l
    s0 = side / S
    A_box = np.array([[s0, 0.0, l], [0.0, s0, t]])
    k = WORK / side
    V = np.array([[k, 0.0, -k * l], [0.0, k, -k * t]])          # view edge coords -> work edge coords
    Vi = np.linalg.inv(_h(V))[:2]
    g_view = view @ GREY
    g_cu = blur(closeup @ GREY, 0.42 / max(s0, 1e-6))            # the close-up at the view's own resolution
    target = warp(g_view, Vi, (WORK, WORK))
    weight = warp(region.astype(np.float64), Vi, (WORK, WORK)) > 0.5

    def to_work(A: np.ndarray) -> np.ndarray:
        """Work coords -> close-up coords for a close-up -> view transform A."""
        return (np.linalg.inv(_h(A)) @ np.linalg.inv(_h(V)))[:2]

    src_scale = WORK / S                                          # close-up resampled to WORK px for speed
    src = blur(g_cu, 0.5 / src_scale) if src_scale < 1 else g_cu
    src = warp(src, np.array([[1 / src_scale, 0, 0], [0, 1 / src_scale, 0]]), (WORK, WORK))
    Ssrc = np.diag([src_scale, src_scale, 1.0])

    def score(A: np.ndarray) -> float:
        M = (Ssrc @ _h(to_work(A)))[:2]
        return ncc(target, warp(src, M, (WORK, WORK)), weight)

    out: dict[str, Any] = {"closeup_size": S, "crop_box": [round(float(v), 3) for v in box]}
    cands = {"crop_box": A_box}
    M0 = (Ssrc @ _h(to_work(A_box)))[:2]
    M = refine_affine(target, src, weight, M0)
    if M is not None:
        Mc = (np.diag([1 / src_scale, 1 / src_scale, 1.0]) @ _h(M))[:2]      # work -> close-up coords
        A_fit = np.linalg.inv(_h(Mc) @ _h(V))[:2]                     # close-up -> view
        scale = math.sqrt(abs(np.linalg.det(A_fit[:, :2])))
        if abs(math.log(scale / s0)) < math.log(MAX_RESCALE):
            cands["fitted"] = A_fit
    scores = {nm: score(A) for nm, A in cands.items()}
    pick = max(scores, key=scores.get)
    A = cands[pick]
    L = A[:, :2]
    sv = np.linalg.svd(L, compute_uv=False)
    scale = math.sqrt(abs(np.linalg.det(L)))
    out.update({
        "A": np.round(A, 6).tolist(),
        "start": "crop_box",
        "used_fit": pick == "fitted",
        "ncc_work": {nm: round(v, 4) for nm, v in scores.items()},
        "scale_ratio": round(scale / s0, 4),
        "anisotropy": round(float(sv[0] / sv[1] - 1), 4),
        "rotation_deg": round(math.degrees(math.atan2(L[1, 0], L[0, 0])), 3),
        "ncc_view": round(ncc_view(closeup, view, A, check), 4),
    })
    return out


def closeup_in_view(closeup: np.ndarray, A: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The close-up prefiltered to the view's resolution and warped into it by A, and where it lands."""
    scale = math.sqrt(abs(np.linalg.det(np.asarray(A)[:, :2])))
    src = blur(closeup, 0.42 / max(scale, 1e-6))
    Ai = np.linalg.inv(_h(np.asarray(A)))[:2]
    x, y = _grid(shape)
    u, v = Ai[0, 0] * x + Ai[0, 1] * y + Ai[0, 2], Ai[1, 0] * x + Ai[1, 1] * y + Ai[1, 2]
    S = closeup.shape[0]
    inside = (u >= 0) & (u < S) & (v >= 0) & (v < S)
    return sample(src, u, v), inside


def ncc_view(closeup: np.ndarray, view: np.ndarray, A: np.ndarray, check: np.ndarray) -> float:
    """Grey-level NCC between the close-up warped into the view and the view, inside ``check`` and the
    close-up's frame (computed on the check region's bounding box only)."""
    b = imaging.bbox(check)
    if b is None:
        return 0.0
    l, t, r, bt = b
    A_loc = np.asarray(A, dtype=np.float64).copy()
    A_loc[:, 2] -= [l, t]
    warped, inside = closeup_in_view(closeup, A_loc, (bt - t, r - l))
    m = check[t:bt, l:r] & inside
    return ncc(view[t:bt, l:r] @ GREY, warped @ GREY, m)


# ---- the stage ------------------------------------------------------------------------------------------


def _load_mask(run: RunFolder, rel: str, shape: tuple[int, int]) -> np.ndarray | None:
    if not run.exists(rel):
        return None
    with Image.open(run.path / rel) as im:
        m = np.asarray(im.convert("L")) > 127
    return m if m.shape == shape and m.any() else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clear(run: RunFolder) -> None:
    """Remove every output of an earlier run of this stage, so a fallback never leaves an old close-up."""
    for v in VIEWS:
        for rel in (f"views/closeup-{v}.png", f"views/closeup-{v}-input.png", f"masks/closeup-{v}.png"):
            if run.exists(rel):
                (run.path / rel).unlink()
    if run.exists(RECORD):
        (run.path / RECORD).unlink()


def _profile_head(client: ModelClient, image: Image.Image, figure: np.ndarray, box: Box) -> tuple[np.ndarray | None,
                                                                                                    dict]:
    """SAM 3.1 `head` on a profile view: the most confident instance lying mostly inside the crop and on the
    figure, clipped to the figure."""
    from herogen.pipeline.heldout import _best

    insts = client.segment(image, HEAD_TEXT)
    near = imaging.dilate(figure, 4) & box_mask(figure.shape, box)
    best = _best(insts, near)
    rec = {"method": f"{SAM_MODEL}:{HEAD_TEXT}", "instances": len(insts)}
    if best is None:
        return None, {**rec, "found": False}
    m = best.mask & figure
    return (m if m.any() else None), {**rec, "found": bool(m.any()), "confidence": best.confidence}


def make_face_source(run: RunFolder, client: ModelClient) -> dict[str, Any]:
    """Make, register and gate the close-ups; write the outputs and `views/closeup.json`. Returns the record.
    Raises on a missing input or a model failure (the stage records it and falls back)."""
    imgs: dict[str, Image.Image] = {}
    figs: dict[str, np.ndarray] = {}
    for v in VIEWS:
        for rel in (f"views/{v}.png", f"masks/{v}.png"):
            if not run.exists(rel):
                raise FileNotFoundError(f"{rel} is missing; run stage 2 first")
        with Image.open(run.path / f"views/{v}.png") as im:
            imgs[v] = im.convert("RGB")
        with Image.open(run.path / f"masks/{v}.png") as im:
            figs[v] = np.asarray(im.convert("L")) > 127
    shape = figs["front"].shape
    face = _load_mask(run, "masks/front-face.png", shape)
    head = _load_mask(run, "masks/front-head.png", shape)
    fbox = front_crop_box(imaging.bbox(face) if face is not None else None,
                          imaging.bbox(head) if head is not None else None)
    record: dict[str, Any] = {"stage": STAGE, "views": {}, "input_side": INPUT_SIDE, "edit_size": EDIT_SIZE,
                              "gates": {"ncc_min": NCC_MIN, "reframe_max": REFRAME_MAX}}
    if fbox is None:
        record["fallback"] = "no face or head mask in the front view (stage 2 part masks), so no head to crop"
        return record
    for v in VIEWS:
        rec: dict[str, Any] = {"recipe": RECIPE[v], "used": False,
                               "view_sha256": _sha256(run.path / f"views/{v}.png")}
        record["views"][v] = rec
        box = fbox if v == "front" else profile_crop_box(fbox, figs["front"], figs[v])
        if box is None:
            rec["reason"] = "no figure in the crop rows"
            continue
        rec["crop_box"] = [round(float(x), 3) for x in box]
        inner = box_mask(figs[v].shape, box)
        if v == "front":
            vhead, check = head, (face if face is not None else head)
            rec["head_mask"] = "masks/front-head.png" if head is not None else None
        else:
            vhead, rec["head_sam"] = _profile_head(client, imgs[v], figs[v], box)
            check = vhead
        if vhead is not None:
            footprint = imaging.dilate(vhead, FOOTPRINT_GROW_PX) & inner
        else:
            footprint = figs[v] & inner
        if check is None:
            check = figs[v] & inner
        region = imaging.dilate(vhead if vhead is not None else figs[v] & inner, 4) & inner
        inp = crop(imgs[v], box, INPUT_SIDE)
        out = client.edit_image(PROMPTS[RECIPE[v]], [inp], size=EDIT_SIZE)
        cu = np.asarray(out.convert("RGB"), dtype=np.float64)
        if cu.shape[0] != cu.shape[1]:
            rec["reason"] = f"close-up is not square ({cu.shape[1]}x{cu.shape[0]})"
            continue
        reg = register(cu, np.asarray(imgs[v], dtype=np.float64), box, region, check & inner)
        rec.update(reg)
        passed = reg["ncc_view"] >= NCC_MIN[v] and abs(math.log(reg["scale_ratio"])) < math.log(REFRAME_MAX)
        if not passed:
            why = []
            if reg["ncc_view"] < NCC_MIN[v]:
                why.append(f"NCC in the view {reg['ncc_view']:.3f} < {NCC_MIN[v]}")
            if abs(math.log(reg["scale_ratio"])) >= math.log(REFRAME_MAX):
                why.append(f"reframed: scale {reg['scale_ratio']:.3f} of the crop box's")
            rec["reason"] = "; ".join(why)
            continue
        out.convert("RGB").save(run.file(f"views/closeup-{v}.png"), format="PNG")
        inp.save(run.file(f"views/closeup-{v}-input.png"), format="PNG")
        imaging.to_mask_image(footprint).save(run.file(f"masks/closeup-{v}.png"))
        rec.update({"used": True, "file": f"views/closeup-{v}.png", "input": f"views/closeup-{v}-input.png",
                    "footprint": f"masks/closeup-{v}.png", "footprint_pixels": int(footprint.sum())})
    return record


def stage2f_face_source(run: RunFolder, client: ModelClient) -> dict[str, Any]:
    """Run stage 2f (module notes). Never raises for a failure of its own: the record says what happened."""
    from herogen.pipeline.stages import _Stage

    st = _Stage(run, client, STAGE, STAGE_NAME)  # type: ignore[arg-type]
    _clear(run)
    try:
        record = make_face_source(run, client)
    except ReplayMiss as e:
        # a replay with no cached answer is a broken cache or run, not a model failure: it stops the run, as in
        # every other stage, and the record of an earlier run is replaced by a failed one (no outputs, nothing used)
        _clear(run)
        st.details = {"stage": STAGE, "closeups": {}, "used_views": []}
        st.fail(f"{type(e).__name__}: {e}"[:FALLBACK_CHARS])
        raise
    except Exception as e:  # noqa: BLE001  (any other failure falls back to the plain views; the reason is kept)
        _clear(run)
        record = {"stage": STAGE, "views": {}, "fallback": f"{type(e).__name__}: {e}"[:FALLBACK_CHARS]}
    used = sorted(v for v, r in record["views"].items() if r.get("used"))
    record["used_views"] = used
    # The record is written even when no close-up is used, so stage 4 can say why for each view.
    run.write_json(RECORD, record)
    st.outputs += [RECORD] + [x for v in used for x in (record["views"][v]["file"], record["views"][v]["input"],
                                                      record["views"][v]["footprint"])]
    # per view under "closeups": a stage record's "views" means held-out views (stage 2h) to other readers
    st.details = {**{k: v for k, v in record.items() if k != "views"}, "closeups": record["views"]}
    for v, r in record["views"].items():
        if "ncc_view" in r:
            st.checks.append(C.check(f"closeup_ncc_{v}", r["ncc_view"], NCC_MIN[v], gate=False))
    if not used:
        st.notes.append("No close-up is used: stage 4 bakes the plain views. "
                        + (record.get("fallback") or "; ".join(f"{v}: {r.get('reason', '?')}"
                                                              for v, r in record["views"].items())))
    return st.finish()
