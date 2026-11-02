"""A bounded, global Lab grade from the source front render to the character artwork."""
import cv2
import numpy as np

MIX = 0.85
AB_SCALE = (0.6, 1.6)
ERODE = 5
QUANTILES = np.linspace(0, 1, 257)


def _rgb(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("palette grade needs finite RGB images")
    if rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("palette grade RGB must be in [0, 1]")
    return rgb


def brightness(rgb, mask):
    """Encoded RGB luma percentiles, on the same [0, 1] scale as the input."""
    values = rgb[mask] @ np.array([0.299, 0.587, 0.114])
    return dict(zip(("p5", "p50", "p95", "p99"), np.quantile(values, [.05, .5, .95, .99]).tolist()))


def fit(source, source_mask, art, art_mask):
    """Fit the prototype's 85% L quantile curve and bounded a/b mean/spread match.

    Artwork edges are eroded by 5x5 pixels to exclude background contamination. Output L is capped at the
    mixed p99; repeated source quantiles share their mean destination, keeping flat palettes deterministic.
    """
    source, art = _rgb(source), _rgb(art)
    source_mask, art_mask = np.asarray(source_mask, bool), np.asarray(art_mask, bool)
    if source_mask.shape != source.shape[:2] or art_mask.shape != art.shape[:2]:
        raise ValueError("palette grade mask dimensions differ from their image")
    art_mask = cv2.erode(art_mask.astype(np.uint8), np.ones((ERODE, ERODE), np.uint8)) > 0
    if not source_mask.any() or not art_mask.any():
        raise ValueError("palette grade has no figure pixels after artwork erosion")
    src = cv2.cvtColor(source, cv2.COLOR_RGB2Lab)[source_mask]
    dst = cv2.cvtColor(art, cv2.COLOR_RGB2Lab)[art_mask]
    src_l, art_l = np.quantile(src[:, 0], QUANTILES), np.quantile(dst[:, 0], QUANTILES)
    dst_l = MIX * art_l + (1 - MIX) * src_l
    cap = float(MIX * np.quantile(dst[:, 0], .99) + (1 - MIX) * np.quantile(src[:, 0], .99))
    unique, inverse, counts = np.unique(src_l, return_inverse=True, return_counts=True)
    dst_l = np.bincount(inverse, weights=dst_l) / counts
    scale = np.clip(dst[:, 1:].std(0) / np.maximum(src[:, 1:].std(0), 1e-3), *AB_SCALE)
    shift = dst[:, 1:].mean(0) - src[:, 1:].mean(0) * scale
    grade = {"status": "applied", "mix": MIX, "artwork_erode_px": ERODE,
             "source_pixels": int(source_mask.sum()), "artwork_pixels": int(art_mask.sum()),
             "L_src": unique.tolist(), "L_dst": np.minimum(dst_l, cap).tolist(), "L_cap_p99": cap,
             "ab_scale": scale.tolist(), "ab_shift": shift.tolist(),
             "brightness_before": brightness(source, source_mask), "brightness_artwork": brightness(art, art_mask)}
    grade["brightness_after"] = brightness(apply(source, grade), source_mask)
    return grade


def apply(rgb, grade):
    """Apply the same grade to an atlas or a colour target, with bounded temporary memory."""
    rgb = _rgb(rgb)
    out = np.empty_like(rgb)
    for row in range(0, len(rgb), 256):
        lab = cv2.cvtColor(rgb[row:row + 256], cv2.COLOR_RGB2Lab)
        lab[..., 0] = np.minimum(np.interp(lab[..., 0], grade["L_src"], grade["L_dst"]), grade["L_cap_p99"])
        lab[..., 1:] = lab[..., 1:] * np.asarray(grade["ab_scale"]) + np.asarray(grade["ab_shift"])
        out[row:row + 256] = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
    return out
