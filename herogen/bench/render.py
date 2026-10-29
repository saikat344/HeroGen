"""Render one known 3D character into a ground-truth run folder.

The run folder has the stage-2 layout of docs/ARCHITECTURE.md, so the building stages read it unchanged, plus
the true mesh and the evaluation views:

    gt/body.glb                      the true mesh, posed, in the contract frame (feet at Z = 0, facing -Y)
    cameras.json                     every camera: azimuth, role (building / evaluation), axes, pixel mapping
    views/{front,left,back,right}.png, views/{front-left,back-left,back-right,front-right}-45.png
                                     building views: flat unlit base colour on a white background
    views-lit/<view>.png             the same views lit (camera-fixed key and fill light, soft ambient)
    masks/<view>.png                 white figure on black
    depth/<view>.npy                 float32 depth in studs from the plane through the image centre, NaN off
                                     the figure (see cameras.depth_of)
    views/meta.json                  pixels_per_unit, height_units, image_size, mask_method "render"
    heldout/front-{left,right}-{25,65}.png
                                     evaluation views, with `-lit.png`, `-mask.png` and `-depth.npy`
    stage-2h.json                    the evaluation views' record, also "reference": "ground-truth"
    stage-2.json                     the stage record: "reference": "ground-truth", checks, pose, asset hash
    logs/bench-render.log            Blender's output

No building stage reads `heldout/`, `gt/` or `depth/`. Masks and depth come from casting one orthographic ray
through each pixel centre against the mesh written to `gt/body.glb`, so they describe that mesh exactly; the
colour renders use the same cameras, and the `render_mask_off_edge_px` check confirms the two line up.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from herogen.blender import bl
from herogen.blender.record import StageRecord, check

from . import BACKGROUND, DEFAULT_SIZE, HEIGHT_STUDS
from .rig import POSE_CONVERGED_DEG

JOB_SCRIPT = Path(__file__).with_name("bench_job.py")
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_TYPES = (".glb", ".gltf", ".fbx", ".blend")
SIZE_RANGE = (32, 8192)         # square image size in pixels
RUN_MARKERS = ("stage-2.json", "cameras.json")  # files that show a folder is a bench run folder
POSES = ("auto", "a-pose", "rest", "as-shipped")

# Gates. The mask comes from rays through pixel centres and the colour render's coverage from area sampling,
# so they may differ only on pixels at the mask's edge (render_mask_off_edge_px counts the others: zero).
HEIGHT_TOLERANCE_PX = 1.0       # front mask height vs height_units * pixels_per_unit
FACING_MIN_X = 0.999            # cos of the right-to-left axis's angle from +X (about 2.6 degrees)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


HOME_PATH = re.compile(r"(?i)\b[A-Z]:[\\/]+Users[\\/]+[^\\/\s'\"]+|(?:/Users|/home)/[^/\s'\"]+")


def scrub_paths(text: str, places: dict[str, Path]) -> str:
    """Replace the given folders (either slash direction, any case) by their placeholder names, then any
    remaining home-folder path by `~`, so a log names no person and no machine layout."""
    for name, path in sorted(places.items(), key=lambda kv: -len(str(kv[1]))):
        for form in {str(path), str(path).replace("\\", "/"), str(path).replace("/", "\\")}:
            text = re.sub(re.escape(form), lambda _m, name=name: name, text, flags=re.IGNORECASE)
    return HOME_PATH.sub("~", text)


def run_blender(job: dict, work: Path, log: Path, timeout: float,
                places: dict[str, Path] | None = None) -> dict:
    """Run the bench job in headless Blender (scripts embedded in the asset are never run)."""
    blender = bl.find_blender()
    if blender is None:
        raise bl.BlenderError("Blender not found (set HEROGEN_BLENDER to the Blender 4.5 executable)")
    job_path = work / "job.json"
    result_path = work / "job.result.json"
    job_path.write_text(json.dumps(job, indent=2))
    cmd = [blender, "--background", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
           "--python", str(JOB_SCRIPT), "--", str(job_path)]
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise bl.BlenderError(f"Blender timed out after {timeout:.0f} s") from exc
    log.write_text(scrub_paths(proc.stdout + "\n--- stderr ---\n" + proc.stderr, places or {}),
                   encoding="utf-8")
    if proc.returncode != 0 or not result_path.exists():
        tail = (proc.stdout + proc.stderr)[-2000:]
        raise bl.BlenderError(f"Blender failed (exit {proc.returncode}); log {log}\n{tail}")
    result = json.loads(result_path.read_text())
    if result.get("ok") is False:
        raise ValueError(result["error"])
    return result


def composite(rgba_path: Path, background=BACKGROUND) -> tuple[Image.Image, np.ndarray]:
    """An RGBA render over a plain background; also returns the alpha in [0, 1]."""
    with Image.open(rgba_path) as im:
        a = np.asarray(im.convert("RGBA"), dtype=np.float64)
    alpha = a[..., 3:4] / 255.0
    rgb = a[..., :3] * alpha + np.asarray(background, dtype=np.float64) * (1.0 - alpha)
    return Image.fromarray(np.clip(np.round(rgb), 0, 255).astype(np.uint8), "RGB"), alpha[..., 0]


def mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def near_edge(mask: np.ndarray) -> np.ndarray:
    """Pixels within one pixel (8-neighbourhood) of the mask's boundary, on either side."""
    pad = np.pad(mask, 1)
    shifts = [pad[1 + dr: 1 + dr + mask.shape[0], 1 + dc: 1 + dc + mask.shape[1]]
              for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    grown = np.logical_or.reduce(shifts)
    shrunk = np.logical_and.reduce(shifts)
    return grown & ~shrunk


def off_edge_disagreement(coverage: np.ndarray, mask: np.ndarray, margin: float = 0.0) -> int:
    """Pixels, away from the mask's edge, where the colour render's coverage (0..1) is on the other side of
    one half from the ray-cast mask by more than `margin`. Edge pixels can differ (area coverage against a
    ray through the pixel centre), and so can a sub-pixel sliver that slips between pixel centres while its
    sampled coverage reads about one half; `margin` (see `coverage_margin`) leaves those out. A camera
    offset puts whole pixels on the wrong side, far beyond it."""
    wrong = np.where(mask, coverage < 0.5 - margin, coverage > 0.5 + margin)
    return int((wrong & ~near_edge(mask)).sum())


def coverage_margin(samples: int) -> float:
    """Two samples' worth of the coverage a render with `samples` samples per pixel reports, at most 0.25."""
    return min(0.25, 2.0 / max(1, samples))


def visible_transparent(alpha_render: Path, mask: np.ndarray) -> int:
    """Figure pixels (away from the edge) where the texture under the pixel is transparent (alpha below one
    half): surface the renders show as solid although its texture says it is see-through."""
    with Image.open(alpha_render) as im:
        grey = np.asarray(im.convert("RGBA"), dtype=np.float64)[..., 0] / 255.0
    return int((mask & ~near_edge(mask) & (grey < 0.5)).sum())


def border_pixels(mask: np.ndarray) -> int:
    return int(mask[0].sum() + mask[-1].sum() + mask[1:-1, 0].sum() + mask[1:-1, -1].sum())


def view_paths(cam: dict) -> dict[str, str]:
    """Run-folder paths of one view's outputs."""
    v = cam["view"]
    if cam["role"] == "building":
        return {"colour": f"views/{v}.png", "lit": f"views-lit/{v}.png", "mask": f"masks/{v}.png",
                "depth": f"depth/{v}.npy"}
    return {"colour": f"heldout/{v}.png", "lit": f"heldout/{v}-lit.png", "mask": f"heldout/{v}-mask.png",
            "depth": f"heldout/{v}-depth.npy"}


def facing_check(facing: dict) -> dict:
    """The rig's right-to-left axis after the yaw must point along +X (so the figure faces -Y). A figure
    without a named rig is turned by the given yaw alone, which nothing here can confirm: recorded, not
    gated."""
    if not facing.get("checked"):
        return check("facing_left_axis_x", None, None, passed=True, gate=False,
                     note="no rig with named left and right limbs: yaw given by the operator, checked by eye")
    x = float(facing["left_axis_after_yaw"][0])
    return check("facing_left_axis_x", round(x, 6), FACING_MIN_X, axis=facing["rig_axis"])


def pose_checks(pose: str, limbs: list[dict]) -> list[dict]:
    """An A-pose has all four limbs, each converged on its target: the angle between the achieved and the
    target direction (3D for an arm, seen from the front for a leg) under `POSE_CONVERGED_DEG`."""
    if pose != "a-pose":
        return []
    have = {(x["limb"], x["side"]) for x in limbs}
    want = [("arm", "L"), ("arm", "R"), ("leg", "L"), ("leg", "R")]
    missing = [f"{limb} {side}" for limb, side in want if (limb, side) not in have]
    out = [check("pose_limbs_posed", len(want) - len(missing), len(want), missing=missing)]
    for x in limbs:
        out.append(check(f"pose_{x['limb']}_{x['side']}_error_deg", x["error_deg"], POSE_CONVERGED_DEG,
                         passed=x["error_deg"] <= POSE_CONVERGED_DEG and x["converged"],
                         converged=x["converged"], steps=x["steps"], basis=x["error_basis"]))
    return out


def check_inputs(asset: Path, out: Path, *, pose: str, include, exclude, size, height, samples_flat,
                 samples_lit, overwrite: bool) -> None:
    """Every problem with the arguments that can be found before Blender starts, as a ValueError (or
    FileNotFoundError) with a message that says what to change."""
    if not asset.is_file():
        raise FileNotFoundError(f"asset not found: {asset}")
    if asset.suffix.lower() not in ASSET_TYPES:
        raise ValueError(f"unsupported asset type {asset.suffix!r} (use {', '.join(ASSET_TYPES)})")
    if pose not in POSES:
        raise ValueError(f"pose must be one of {', '.join(POSES)}")
    for flag, pattern in (("--include", include), ("--exclude", exclude)):
        if pattern is not None:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"{flag} is not a valid regular expression: {exc}") from None
    if not isinstance(size, int) or not SIZE_RANGE[0] <= size <= SIZE_RANGE[1]:
        raise ValueError(f"--size must be a whole number of pixels from {SIZE_RANGE[0]} to {SIZE_RANGE[1]}")
    if not (isinstance(height, (int, float)) and math.isfinite(height) and height > 0):
        raise ValueError("--height must be a positive number of studs")
    for flag, n in (("--samples-flat", samples_flat), ("--samples-lit", samples_lit)):
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"{flag} must be at least 1")
    if out.exists() and not out.is_dir():
        raise ValueError(f"--out {out} is a file, not a folder")
    if out.is_dir() and any(out.iterdir()):
        if not overwrite:
            raise ValueError(f"run folder {out} is not empty; pass --overwrite to replace it")
        if not any((out / m).exists() for m in RUN_MARKERS):
            raise ValueError(f"run folder {out} is not empty and does not look like a bench run folder "
                             f"(no {' or '.join(RUN_MARKERS)}); refusing to clear it")


def heldout_record(out: Path, cams: list[dict], borders: dict, off_edge: dict, asset_name: str) -> dict:
    """`stage-2h.json`: the evaluation views, rendered from the true mesh with exact cameras. Only scoring
    reads `heldout/`; the record says so and carries each view's azimuth and files."""
    rec = StageRecord(out, "2h", "heldout")
    views = [c for c in cams if c["role"] == "evaluation"]
    for c in views:
        v = c["view"]
        rec.add(check(f"{v}_border_px", borders[v], 0, passed=borders[v] == 0))
        rec.add(check(f"{v}_render_mask_off_edge_px", off_edge[v], 0, passed=off_edge[v] == 0))
    rec.data["models"] = []
    rec.data["outputs"] = [f for c in views for f in c["files"].values()]
    rec.data["cost_basis"] = "local compute only (Blender); rendered with stage 2"
    rec.note(f"ground-truth evaluation views of {asset_name}, exact cameras (see cameras.json)")
    return rec.finish({
        "reference": "ground-truth",
        "details": {
            "views": [{"view": c["view"], "azimuth_deg": c["azimuth_deg"],
                       "elevation_deg": c["elevation_deg"], "files": c["files"]} for c in views],
            "cameras": "cameras.json",
            "read_by": "scoring only; no building stage reads heldout/",
        },
    })


def render_asset(asset: Path, out: Path, *, include: str | None = None, exclude: str | None = None,
                 yaw_deg: float | None = None, pose: str = "auto", material_images: dict | None = None,
                 material_vcols: dict | None = None, image_dirs: list | None = None,
                 bind_unrigged: bool = False,
                 size: int = DEFAULT_SIZE, height: float = HEIGHT_STUDS, samples_flat: int = 16,
                 samples_lit: int = 128, device: str = "auto", timeout: float = 3600.0,
                 overwrite: bool = False) -> dict:
    """Render `asset` into the run folder `out` and return its stage-2 record (see the module docstring).
    `out` must be empty or missing; with `overwrite`, an existing bench run folder there is cleared first."""
    asset, out = Path(asset), Path(out)
    check_inputs(asset, out, pose=pose, include=include, exclude=exclude, size=size, height=height,
                 samples_flat=samples_flat, samples_lit=samples_lit, overwrite=overwrite)
    if out.is_dir() and any(out.iterdir()):
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    rec = StageRecord(out, 2, "turnaround")
    with tempfile.TemporaryDirectory(prefix="herogen-bench-") as tmp:
        work = Path(tmp)
        job = {
            "repo_root": str(REPO_ROOT), "asset": str(asset.resolve()),
            "include": include, "exclude": exclude,
            "yaw_deg": yaw_deg, "pose": pose, "material_images": material_images or {},
            "material_vcols": material_vcols or {},
            "image_dirs": [str(Path(d).resolve()) for d in image_dirs or []],
            "bind_unrigged": bind_unrigged, "height": height, "size": size, "samples_flat": samples_flat,
            "samples_lit": samples_lit, "device": device, "gt_glb": str((out / "gt" / "body.glb").resolve()),
            "work": str(work / "views"),
        }
        places = {"<work>": work.resolve(), "<run>": out.resolve(), "<asset-dir>": asset.resolve().parent,
                  "<repo>": REPO_ROOT, "<blender>": Path(bl.find_blender() or ".").resolve().parent}
        res = run_blender(job, work, out / "logs" / "bench-render.log", timeout, places)
        cams = res["cameras"]
        outputs = ["gt/body.glb", "cameras.json", "views/meta.json"]
        ious, borders, off_edge, see_through = {}, {}, {}, {}
        for cam in cams:
            paths = view_paths(cam)
            cam["files"] = paths
            with np.load(work / "views" / f"{cam['view']}.npz") as arr:
                mask, depth = arr["mask"], arr["depth"]
            flat, alpha = composite(work / "views" / f"{cam['view']}-flat.png")
            lit, _ = composite(work / "views" / f"{cam['view']}-lit.png")
            for key, img in (("colour", flat), ("lit", lit), ("mask", mask_image(mask))):
                p = out / paths[key]
                p.parent.mkdir(parents=True, exist_ok=True)
                img.save(p)
            (out / paths["depth"]).parent.mkdir(parents=True, exist_ok=True)
            np.save(out / paths["depth"], depth.astype(np.float32))
            outputs += list(paths.values())
            ious[cam["view"]] = round(iou(alpha >= 0.5, mask), 5)
            off_edge[cam["view"]] = off_edge_disagreement(alpha, mask, coverage_margin(samples_flat))
            see_through[cam["view"]] = visible_transparent(work / "views" / f"{cam['view']}-alpha.png", mask)
            borders[cam["view"]] = border_pixels(mask)

    ppu, size = res["pixels_per_unit"], int(size)
    building = [c["view"] for c in cams if c["role"] == "building"]
    evaluation = [c["view"] for c in cams if c["role"] == "evaluation"]
    (out / "cameras.json").write_text(json.dumps({
        "frame": "contract: +Z up, feet at Z = 0, facing -Y, the character's left at +X, "
                 "X/Y bounding box centred",
        "units": "studs",
        "height_units": height,
        "pixels_per_unit": ppu,
        "image_size": size,
        "pixel_mapping": "col = (p - c).right * ppu + size/2; row = size/2 - (p - c).up * ppu; "
                         "c = image_centre_world; pixel (r, c) covers [c, c+1) x [r, r+1)",
        "depth": "(p - image_centre_world).forward in studs; NaN where no surface",
        "building": building,
        "evaluation": evaluation,
        "cameras": cams,
    }, indent=2) + "\n")
    (out / "views").mkdir(parents=True, exist_ok=True)
    (out / "views" / "meta.json").write_text(json.dumps({
        "pixels_per_unit": ppu,
        "height_units": height,
        "image_size": size,
        "views": building,
        "heldout_views": evaluation,
        "mask_method": {v: "render" for v in building + evaluation},
        "mask_method_detail": {v: "orthographic ray through each pixel centre against gt/body.glb"
                               for v in building + evaluation},
        "reference": "ground-truth",
    }, indent=2) + "\n")

    front = np.asarray(Image.open(out / "masks" / "front.png")) > 127
    rows = np.flatnonzero(front.any(axis=1))
    front_h = int(rows[-1] - rows[0] + 1) if len(rows) else 0
    expect_h = height * ppu
    rec.add(check("gt_triangles", res["triangles"], 1))
    rec.add(check("figure_in_frame_border_px", max(borders.values()), 0, passed=max(borders.values()) == 0,
                  per_view=borders))
    rec.add(check("front_height_px_error", round(abs(front_h - expect_h), 3), HEIGHT_TOLERANCE_PX,
                  passed=abs(front_h - expect_h) <= HEIGHT_TOLERANCE_PX, measured_px=front_h,
                  expected_px=round(expect_h, 3)))
    rec.add(check("render_mask_off_edge_px", max(off_edge.values()), 0, passed=max(off_edge.values()) == 0,
                  per_view=off_edge))
    rec.add(check("render_mask_iou_min", min(ious.values()), None, gate=False, per_view=ious,
                  note="edge pixels weigh more at small sizes; render_mask_off_edge_px is the gate"))
    rec.add(check("missing_images", len(res["missing_images"]), 0, passed=not res["missing_images"],
                  names=res["missing_images"]))
    rec.add(check("visible_transparent_texels_px", max(see_through.values()), 0,
                  passed=max(see_through.values()) == 0, per_view=see_through,
                  image_alpha_transparent_share=res["image_alpha_transparent_share"],
                  note="figure pixels whose texture alpha is below 0.5 (edge pixels excluded)"))
    rec.add(facing_check(res["facing"]))
    rec.add(check("skipped_visible_geometry", len(res["skipped_geometry"]), None, gate=False,
                  objects=res["skipped_geometry"],
                  note="visible in the file but not meshes the bench renders; not in gt/body.glb either"))
    for c in pose_checks(res["pose"], res["limbs"]):
        rec.add(c)
    rec.data["models"] = []
    rec.data["tools"] = [f"Blender {res['blender']}", f"Cycles ({res['device']})"]
    rec.data["outputs"] = outputs
    rec.data["cost_basis"] = "local compute only (Blender); no paid generation"
    rec.note(f"ground-truth render of {asset.name}; pose {res['pose']}")
    heldout_record(out, cams, borders, off_edge, asset.name)
    return rec.finish({
        "reference": "ground-truth",
        "details": {
            "asset": asset.name,
            "asset_sha256": sha256(asset),
            "objects": res["objects"],
            "rigs": res["rigs"],
            "pose_requested": pose,
            "pose": res["pose"],
            "limbs": res["limbs"],
            "yaw_deg": res["yaw_deg"],
            "facing": res["facing"],
            "skipped_geometry": res["skipped_geometry"],
            "contract_scale": res["contract_scale"],
            "include": include, "exclude": exclude,
            "material_sources": res["materials"],
            "material_images": {k: os.path.basename(v) for k, v in (material_images or {}).items()},
            "material_vcols": material_vcols or {},
            "bind_unrigged": bind_unrigged,
            "triangles": res["triangles"],
            "vertices": res["vertices"],
            "bounds_min": res["bounds_min"],
            "bounds_max": res["bounds_max"],
            "pixels_per_unit": ppu,
            "image_size": size,
            "samples": {"flat": samples_flat, "lit": samples_lit},
            "device": res["device"],
            "timings_s": res["timings"],
            "blender_report": res["report"],
        },
    })
