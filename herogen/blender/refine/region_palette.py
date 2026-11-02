"""Opt-in, saved-mask palette calibration of an unlit atlas against artwork.

Regions start from semantic masks; connected skin colour completes the face within the head.
Unlabelled surface texels retain their values;
atlas gutters are re-extended at export.
Clothing uses independently matched colour clusters; skin and hair use semantic regions.
Each displacement cap is the RMS distance between matched Lab marginal quantiles, measured anew
for each export. This bounds the correction; it does not assert pixel correspondence or fidelity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REGIONS = ("skin", "hair", "rest")
QUANTILES = np.linspace(0, 1, 257)


def _rgb(value):
    rgb = np.asarray(value, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("palette RGB must be a finite H x W x 3 array")
    if rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("palette RGB must be in [0, 1]")
    return rgb


def _masks(masks, shape):
    unknown = set(masks) - set(REGIONS)
    if unknown:
        raise ValueError("unknown palette region: " + ", ".join(sorted(unknown)))
    occupied = np.zeros(shape, bool)
    result = {}
    for name, value in masks.items():
        mask = np.asarray(value, bool)
        if mask.shape != shape:
            raise ValueError(f"{name} palette mask has the wrong dimensions")
        if (occupied & mask).any():
            raise ValueError("palette region masks must not overlap")
        occupied |= mask
        result[name] = mask
    return result


def _curves(a, b):
    qa, qb = np.quantile(a, QUANTILES, axis=0), np.quantile(b, QUANTILES, axis=0)
    curves = []
    for c in range(3):
        unique, inverse, count = np.unique(qa[:, c], return_inverse=True, return_counts=True)
        curves.append({"source": unique.tolist(),
                       "target": (np.bincount(inverse, weights=qb[:, c]) / count).tolist()})
    return curves, float(np.sqrt(np.mean(np.sum((qb - qa) ** 2, axis=1))))


def _clusters(lab):
    """Deterministic farthest-first Lab Lloyd clusters, with reduced lighting influence.

    Eight centres bound work; a 6 Lab separation stops flat colours from being split.
    Chroma gets full weight, lightness half weight. A fixed evenly spaced sample fits
    centres; all pixels participate in the final distributions and measured counts.
    """
    sample = lab[np.linspace(0, len(lab) - 1, min(len(lab), 16384), dtype=int)]
    scale = np.array([.5, 1., 1.])
    values = sample * scale
    centres = [values[np.argmin(np.linalg.norm(values - np.median(values, axis=0), axis=1))]]
    for _ in range(7):
        distance = np.min(np.sum((values[:, None] - np.asarray(centres)) ** 2, axis=2), axis=1)
        if distance.max() <= 6 ** 2:
            break
        centres.append(values[distance.argmax()])
    centres = np.asarray(centres)
    for _ in range(32):
        labels = np.argmin(np.sum((values[:, None] - centres) ** 2, axis=2), axis=1)
        updated = np.asarray([values[labels == i].mean(0) if (labels == i).any() else c
                              for i, c in enumerate(centres)])
        if np.max(np.abs(updated - centres)) < .01:
            centres = updated
            break
        centres = updated
    centres /= scale
    labels = _labels(lab, centres)
    centres = centres[np.unique(labels)]
    return centres, _labels(lab, centres)


def _labels(lab, centres):
    # Chunking bounds atlas-sized temporary arrays.
    return np.concatenate([np.argmin(np.sum(((part[:, None] - centres) * [.5, 1, 1]) ** 2, axis=2), axis=1)
                           for part in np.array_split(lab, max(1, (len(lab) + 65535) // 65536))])


def _clothing(a, b):
    source, si = _clusters(a)
    artwork, ai = _clusters(b)
    result = []
    for i, centre in enumerate(source):
        chroma = np.linalg.norm(centre[1:])
        art_chroma = np.linalg.norm(artwork[:, 1:], axis=1)
        hue = np.arctan2(centre[2], centre[1])
        art_hue = np.arctan2(artwork[:, 2], artwork[:, 1])
        angle = np.abs(np.arctan2(np.sin(art_hue - hue), np.cos(art_hue - hue)))
        # Neutral fabric cannot own a coloured fabric's grade, nor vice versa.
        compatible = ((art_chroma < 10) == (chroma < 10)) & ((angle <= np.pi / 6) | (chroma < 10))
        distance = np.linalg.norm((artwork - centre) * [.5, 1, 1], axis=1)
        j = int(np.argmin(np.where(compatible, distance, np.inf)))
        entry = {"source_centre": centre.tolist(), "source_pixels": int((si == i).sum()),
                 "status": "unmatched"}
        if compatible.any():
            curves, cap = _curves(a[si == i], b[ai == j])
            entry.update(status="fitted", artwork_cluster=j, artwork_centre=artwork[j].tolist(),
                         artwork_pixels=int((ai == j).sum()), curves=curves, cap_lab=cap)
        result.append(entry)
    return result


def _mapped(lab, grade):
    target = np.column_stack([np.interp(lab[:, c], curve["source"], curve["target"])
                              for c, curve in enumerate(grade["curves"])])
    delta = target - lab
    cap = float(grade["cap_lab"])
    if not np.isfinite(cap) or cap < 0:
        raise ValueError("palette cap must be finite and nonnegative")
    delta *= np.minimum(1, cap / np.maximum(np.linalg.norm(delta, axis=1), np.finfo(float).tiny))[:, None]
    return lab + delta


def _clothing_target(lab, clusters):
    labels = _labels(lab, np.asarray([c["source_centre"] for c in clusters]))
    target = lab.copy()
    for i, cluster in enumerate(clusters):
        hit = labels == i
        if hit.any() and cluster["status"] == "fitted":
            target[hit] = _mapped(lab[hit], cluster)
    # Preserve every pixel's chroma floor and exact source hue. Strongly chromatic accents
    # (gold, brass, saturated gems) are excluded altogether, including their lightness.
    chroma = np.linalg.norm(lab[:, 1:], axis=1)
    target_chroma = np.linalg.norm(target[:, 1:], axis=1)
    target[:, 1:] = lab[:, 1:] * np.maximum(1, target_chroma / np.maximum(chroma, 1e-8))[:, None]
    # Fade the correction before full accent exclusion, avoiding a hard colour
    # contour where a shaded metal or dyed fabric crosses the chroma cutoff.
    t = np.clip((chroma - 20) / 10, 0, 1)
    strength = 1 - t * t * (3 - 2 * t)
    target = lab + (target - lab) * strength[:, None]
    target[chroma >= 30] = lab[chroma >= 30]
    return target


def _colour_rgb(lab, target, original):
    """Keep the chroma floor through RGB gamut conversion and 8-bit export.

    Lab/RGB lookup tables and quantisation can otherwise lose chroma even on a
    hue-preserving Lab ray. Correct only the measured deficit; retain the source
    when the requested lightness/chroma cannot be represented in the RGB gamut.
    """
    shape = target.shape
    source_chroma = np.linalg.norm(lab[..., 1:], axis=-1)
    unchanged = np.all(target == lab, axis=-1)
    target = target.astype(np.float32).copy()
    for _ in range(4):
        rgb = np.round(cv2.cvtColor(target.reshape(-1, 1, 3), cv2.COLOR_Lab2RGB) * 255) / 255
        decoded = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).reshape(shape)
        chroma = np.linalg.norm(decoded[..., 1:], axis=-1)
        bad = (chroma < source_chroma) & ~unchanged
        if not bad.any():
            break
        current = np.linalg.norm(target[..., 1:][bad], axis=-1)
        target[..., 1:][bad] *= ((current + source_chroma[bad] - chroma[bad] + .05)
                               / np.maximum(current, 1e-8))[..., None]
    rgb = rgb.reshape(shape)
    rgb[unchanged | bad] = original[unchanged | bad]
    return rgb


def fit(source, artwork, source_masks, artwork_masks):
    """Fit independent marginal Lab quantile maps; return JSON-serialisable measured caps.

    The cap is sqrt(mean_q(sum_Lab((art_quantile - source_quantile)**2))). Identical
    distributions therefore have zero cap. Missing/empty regions are recorded, not substituted.
    """
    source, artwork = _rgb(source), _rgb(artwork)
    source_masks = _masks(source_masks, source.shape[:2])
    artwork_masks = _masks(artwork_masks, artwork.shape[:2])
    src = cv2.cvtColor(source, cv2.COLOR_RGB2Lab)
    dst = cv2.cvtColor(artwork, cv2.COLOR_RGB2Lab)
    result = {}
    for name in REGIONS:
        if name not in source_masks or name not in artwork_masks:
            result[name] = {"status": "missing_mask"}
            continue
        a, b = src[source_masks[name]], dst[artwork_masks[name]]
        counts = {"source_pixels": len(a), "artwork_pixels": len(b)}
        if not len(a) or not len(b):
            result[name] = {"status": "empty_mask", **counts}
            continue
        curves, cap = _curves(a, b)
        qa, qb = np.quantile(a, QUANTILES, axis=0), np.quantile(b, QUANTILES, axis=0)
        result[name] = {"status": "fitted", **counts, "curves": curves,
                        "cap_lab": cap,
                        "rms_L": float(np.sqrt(np.mean((qb[:, 0] - qa[:, 0]) ** 2))),
                        "rms_ab": float(np.sqrt(np.mean(np.sum((qb[:, 1:] - qa[:, 1:]) ** 2, axis=1))))}
        if name == "rest":
            result[name]["clusters"] = _clothing(a, b)
    return result


def apply(rgb, grades, masks, *, enabled=False, weights=None):
    """Apply only explicitly labelled pixels. Disabled and zero-cap paths are exact identities."""
    if not enabled:
        return rgb
    original = _rgb(rgb)
    masks = _masks(masks, original.shape[:2])
    if weights is not None:
        total = np.zeros(original.shape[:2], np.float32)
        for name, weight in weights.items():
            if name not in masks:
                raise ValueError("palette weight has no region mask")
            weight = np.asarray(weight)
            if weight.shape != total.shape or not np.isfinite(weight).all() or (weight < 0).any() or (weight > 1).any():
                raise ValueError("palette weights must be finite H x W arrays in [0, 1]")
            if (weight[~masks[name]] != 0).any():
                raise ValueError("palette weights must stay inside labelled support")
            total += weight
        if (total > 1 + 1e-6).any():
            raise ValueError("palette weights must sum to at most one")
    out = original.copy()
    for name, mask in masks.items():
        grade = grades.get(name, {})
        if grade.get("status") != "fitted" or not mask.any():
            continue
        cap = float(grade["cap_lab"])
        if not np.isfinite(cap) or cap < 0:
            raise ValueError("palette cap must be finite and nonnegative")
        if cap == 0:
            continue
        pixels = original[mask].reshape(-1, 1, 3)
        lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2Lab).reshape(-1, 3)
        target = (_clothing_target(lab, grade["clusters"]) if "clusters" in grade else _mapped(lab, grade))
        delta = target - lab
        if weights is not None:
            delta *= np.asarray(weights.get(name, np.zeros(mask.shape)))[mask, None]
        if "clusters" in grade:
            out[mask] = _colour_rgb(lab, lab + delta, original[mask])
        else:
            out[mask] = cv2.cvtColor((lab + delta).astype(np.float32).reshape(-1, 1, 3),
                                    cv2.COLOR_Lab2RGB).reshape(-1, 3)
        unchanged = np.all(delta == 0, axis=1)
        values = out[mask]
        values[unchanged] = original[mask][unchanged]
        out[mask] = values
    if weights is not None:
        out[total == 0] = original[total == 0]
    return out


def complete_face(rgb, face, head):
    """Retain the semantic face and close/grow connected skin within the head.

    Per-channel Tukey chroma fences reject eye/hair outliers in the seed. New pixels must be at
    least as light as the seed's lower quartile, preventing dark hair from joining skin; existing
    shadowed face pixels are always retained. Use the final ungraded render, not the source probe.
    Empty face masks stay empty; a head/helmet is never substituted for a missing face.
    """
    from scipy import ndimage

    face, head = np.asarray(face, bool), np.asarray(head, bool)
    if not face.any():
        return face.copy()
    lab = cv2.cvtColor(_rgb(rgb), cv2.COLOR_RGB2Lab)
    q25, q75 = np.percentile(lab[face], (25, 75), axis=0)
    spread = q75 - q25
    candidate = (head & (lab[..., 0] >= q25[0]) &
                 ((lab[..., 1:] >= q25[1:] - 1.5 * spread[1:]) &
                  (lab[..., 1:] <= q75[1:] + 1.5 * spread[1:])).all(axis=2))
    grown = ndimage.binary_propagation(face, mask=candidate | face)
    radius_px = max(1, int(round(np.sqrt(face.sum()) / 50)))
    closed = ndimage.binary_closing(grown, iterations=radius_px)
    return face | (closed & head) | grown


def feather(mask, width, *, domain=None):
    """Cubic distance ramp inside support, with zero value at its outer pixel centres.

    Nine pixels at a 1536-pixel render spread the measured worst scalp correction (7.36 Lab)
    below its natural boundary gradient (1.34 Lab/pixel): 1.5 * 7.36 / 1.34 = 8.24.
    Width scales with camera resolution. Projection precedes atlas sampling, avoiding UV scale.
    """
    from scipy import ndimage

    if not np.isfinite(width) or width <= 0:
        raise ValueError("palette feather width must be positive and finite")
    mask = np.asarray(mask, bool)
    support = mask
    if domain is not None:
        domain = np.asarray(domain, bool)
        if domain.shape != mask.shape or (mask & ~domain).any():
            raise ValueError("palette feather domain must contain the mask and match its dimensions")
        # Background/gutters have no graded surface. They must not create a dark ungraded rim.
        support = mask | ~domain
    if domain is not None and support.all():
        return mask.astype(np.float32)
    distance = ndimage.distance_transform_edt(np.pad(support, 1, constant_values=domain is not None))[1:-1, 1:-1]
    t = np.clip((distance - 1) / width, 0, 1)
    return (t * t * (3 - 2 * t) * mask).astype(np.float32)


def project_fields(prep, camera, figure, fields):
    """Project smooth render fields onto visible texels, tapering grazing-angle support."""
    from .character import texel_maps
    from .cameras import axes, project, px_per_unit

    positions, normals, coverage = texel_maps(Path(prep))
    x, y, depth = project(camera, positions)
    xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
    h, w = figure.shape
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    xi, yi = np.clip(xi, 0, w - 1), np.clip(yi, 0, h - 1)
    front_depth = np.full((h, w), np.inf)
    valid = coverage & inside
    np.minimum.at(front_depth, (yi[valid], xi[valid]), depth[valid])
    visible = valid & figure[yi, xi] & (depth <= front_depth[yi, xi] + 1 / px_per_unit(camera))
    # A smooth ramp over the last 15 degrees before tangency removes the front/side grade step.
    facing = np.clip(-(normals @ axes(camera)[2]) / np.sin(np.deg2rad(15)), 0, 1)
    facing = facing * facing * (3 - 2 * facing)
    result = {}
    for name, field in fields.items():
        sampled = cv2.remap(field.astype(np.float32), (x - .5).astype(np.float32),
                            (y - .5).astype(np.float32), cv2.INTER_LINEAR)
        result[name] = (sampled * facing * visible).astype(np.float32)
    return result


def _image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255


def extend_margins(rgb, coverage):
    """Extend graded triangle colours into atlas gutters, as Texture.save does for every export."""
    from scipy import ndimage

    rgb = _rgb(rgb)
    coverage = np.asarray(coverage, bool)
    if coverage.shape != rgb.shape[:2] or not coverage.any():
        raise ValueError("palette atlas coverage must match the texture and contain covered texels")
    out = rgb.copy()
    margin = ~coverage
    nearest = ndimage.distance_transform_edt(margin, return_distances=False, return_indices=True)
    out[margin] = rgb[nearest[0][margin], nearest[1][margin]]
    return out


def transfer_mask(mask, source_figure, target_figure):
    """Transfer a semantic mask between foreground bounding boxes, without colour inference."""
    out = np.zeros(target_figure.shape, bool)
    if not source_figure.any() or not target_figure.any():
        raise ValueError("palette alignment needs nonempty artwork and render foregrounds")
    sy, sx = np.where(source_figure)
    ty, tx = np.where(target_figure)
    source = mask[sy.min():sy.max() + 1, sx.min():sx.max() + 1]
    out[ty.min():ty.max() + 1, tx.min():tx.max() + 1] = cv2.resize(
        source.astype(np.uint8), (int(tx.max() - tx.min() + 1), int(ty.max() - ty.min() + 1)),
        interpolation=cv2.INTER_NEAREST) > 0
    return out


def prepare_manifest(run, prep, camera, folder):
    """Build run-local palette inputs from stage-2 masks and the setup's existing SAM masks.

    Face is a skin proxy; unknown head/hair and body skin are not separately calibrated. The two
    face masks are intersected for fitting only, to avoid fitting beard to bare skin.
    Application retains the complete render face rather than cutting it with artwork alignment.
    Only the visible front atlas surface is labelled; the other views never participate in fitting.
    This makes no model calls and uses no per-character configuration.
    """
    import shutil
    from .character import texel_maps
    from .cameras import axes, project, px_per_unit

    run, prep, folder = Path(run), Path(prep), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    def load(path, shape=None):
        with Image.open(path) as image:
            value = np.asarray(image.convert("L")) > 127
        if shape is not None and value.shape != shape:
            raise ValueError(f"palette input mask dimensions differ: {path.name}")
        return value

    def save(mask, name):
        Image.fromarray(mask.astype(np.uint8) * 255).save(folder / name)
        return name

    artwork = run / "views" / "front.png"
    with Image.open(artwork) as image:
        shape = (image.height, image.width)
    art_figure = load(run / "masks" / "front.png", shape)
    art_head = load(run / "masks" / "front-head.png", shape)
    face_path = run / "masks" / "front-face.png"
    art_face = load(face_path, shape) if face_path.is_file() else np.zeros(shape, bool)
    probe = prep.parent / "probe"
    with Image.open(probe / "probe0.png") as image:
        render_figure = np.asarray(image.convert("RGBA"))[..., 3] > 127
    if render_figure.shape != (camera["res"][1], camera["res"][0]):
        raise ValueError("palette probe and front camera dimensions differ")
    render_face = load(probe / "palette-face.png", render_figure.shape) & render_figure
    render_head = load(probe / "palette-head.png", render_figure.shape)
    save(art_face & art_figure, "original-art-skin.png")
    save(render_face, "original-render-skin.png")
    render_masks = {"skin": render_face & transfer_mask(art_face, art_figure, render_figure),
                    "rest": render_figure & ~(render_head | render_face)}
    art_masks = {"skin": art_face & art_figure & transfer_mask(render_face, render_figure, art_figure),
                 "rest": art_figure & ~(art_head | art_face)}
    application_face = render_face if art_face.any() else np.zeros_like(render_face)
    application_masks = {"skin": application_face, "rest": render_masks["rest"] & ~application_face}
    positions, normals, coverage = texel_maps(prep)
    x, y, depth = project(camera, positions)
    xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
    h, w = render_figure.shape
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    xi, yi = np.clip(xi, 0, w - 1), np.clip(yi, 0, h - 1)
    front_depth = np.full((h, w), np.inf)
    valid = coverage & inside
    np.minimum.at(front_depth, (yi[valid], xi[valid]), depth[valid])
    visible = (valid & render_figure[yi, xi] & (normals @ axes(camera)[2] < 0) &
               (depth <= front_depth[yi, xi] + 1 / px_per_unit(camera)))
    shutil.copyfile(artwork, folder / "artwork.png")
    data = {"atlas_layout_sha256": hashlib.sha256((prep / "pos.exr").read_bytes()).hexdigest(),
            "artwork": "artwork.png", "artwork_masks": {}, "render_masks": {}, "atlas_masks": {},
            "application_masks": {}, "feather_pixels_at_1536": 9,
            "completion_head": save(render_head & render_figure, "completion-head.png"),
            "source": "stage-2 artwork masks and stage-3f setup masks",
            "skin_proxy": "face", "hair": "missing_mask",
            "artwork_face_mask": "present" if face_path.is_file() else "missing_mask"}
    for name in render_masks:
        data["application_masks"][name] = save(application_masks[name], f"application-{name}.png")
        data["artwork_masks"][name] = save(art_masks[name], f"art-{name}.png")
        data["render_masks"][name] = save(render_masks[name], f"render-{name}.png")
        data["atlas_masks"][name] = save(visible & render_masks[name][yi, xi], f"atlas-{name}.png")
    target = folder / "manifest.json"
    target.write_text(json.dumps(data, indent=2) + "\n")
    return target


def export_texture(manifest, prep, texture, camera, folder, *, timeout=900, protected_head=None):
    """Grade an export using a saved-mask manifest, without any model request.

    The manifest binds atlas masks to the preparation's position-map SHA256, and gives artwork,
    artwork_masks, render_masks and atlas_masks paths relative to itself. Masks must be disjoint.
    Only the front-visible, labelled atlas area is changed when the supplied masks cover that area.
    """
    from .backend import blender

    manifest, prep, folder = Path(manifest), Path(prep), Path(folder)
    data = json.loads(manifest.read_text())
    digest = hashlib.sha256((prep / "pos.exr").read_bytes()).hexdigest()
    if data.get("atlas_layout_sha256") != digest:
        raise ValueError("palette masks do not match the prepared atlas layout")
    folder.mkdir(parents=True, exist_ok=True)
    blender("render", {"blend": str(prep / "work.blend"), "texture": str(texture),
                       "cams": [camera], "out": str(folder)}, folder / "blender.log", timeout=timeout)

    def load_masks(key):
        result = {}
        for name, rel in data[key].items():
            with Image.open(manifest.parent / rel) as image:
                result[name] = np.asarray(image.convert("L")) > 127
        return result

    source = _image(folder / (camera["name"] + ".png"))
    grades = fit(source, _image(manifest.parent / data["artwork"]),
                 load_masks("render_masks"), load_masks("artwork_masks"))
    atlas = _image(texture)
    atlas_masks = _masks(load_masks("atlas_masks"), atlas.shape[:2])
    for name, grade in grades.items():
        grade["atlas_pixels"] = int(atlas_masks[name].sum()) if name in atlas_masks else 0
        grade["applied"] = bool(grade["atlas_pixels"] and grade.get("cap_lab", 0) > 0)
    # Explicit legacy atlas manifests have no render-space application support. Feather those
    # in atlas pixels; automatically generated manifests use the camera-space path below.
    with Image.open(prep / "cov.png") as image:
        coverage = np.asarray(image.convert("L")) > 127
    weights = {name: feather(mask, 9 * max(atlas.shape[:2]) / 1536, domain=coverage)
               for name, mask in atlas_masks.items()}
    if "application_masks" in data:
        with Image.open(folder / (camera["name"] + ".png")) as image:
            figure = np.asarray(image.convert("RGBA"))[..., 3] > 127
        width = float(data["feather_pixels_at_1536"]) * max(figure.shape) / 1536
        application = _masks(load_masks("application_masks"), source.shape[:2])
        # Saved support comes from the setup render; rasterized silhouette edges can differ.
        # Only the current rendered surface can receive a projected colour correction.
        application = {name: mask & figure for name, mask in application.items()}
        if "completion_head" in data and "skin" in application:
            with Image.open(manifest.parent / data["completion_head"]) as image:
                head = np.asarray(image.convert("L")) > 127
            application["skin"] = complete_face(source, application["skin"], head & figure)
            for name in application:
                if name != "skin":
                    application[name] &= ~application["skin"]
        for name, mask in application.items():
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"application-{name}.png")
        fields = {name: feather(mask, width, domain=figure) for name, mask in application.items()}
        weights = project_fields(prep, camera, figure, fields)
        atlas_masks = {name: weight > 0 for name, weight in weights.items()}
        for name, grade in grades.items():
            grade["feather_render_pixels"] = width
            grade["atlas_pixels"] = int(atlas_masks[name].sum()) if name in atlas_masks else 0
            grade["applied"] = bool(grade["atlas_pixels"] and grade.get("cap_lab", 0) > 0)
    if protected_head is not None:
        protection = np.asarray(protected_head, np.float32)
        if (protection.shape != coverage.shape or not np.isfinite(protection).all()
                or (protection < 0).any() or (protection > 1).any()):
            raise ValueError("palette head protection must be finite atlas weights in [0, 1]")
        for name in weights:
            weights[name] *= 1 - protection
        for name, grade in grades.items():
            grade['protected_head_pixels'] = int((atlas_masks.get(name, np.zeros_like(coverage)) & (protection == 1)).sum())
            grade['graded_atlas_pixels'] = int((weights[name] > 0).sum()) if name in weights else 0
            grade['applied'] = bool(grade['graded_atlas_pixels'] and grade.get('cap_lab', 0) > 0)
    graded = apply(atlas, grades, atlas_masks, enabled=True, weights=weights)
    graded = extend_margins(graded, coverage)
    target = folder / "texture.png"
    Image.fromarray(np.round(np.clip(graded, 0, 1) * 255).astype(np.uint8)).save(target)
    (folder / "grades.json").write_text(json.dumps(grades, indent=2) + "\n")
    return target, grades
