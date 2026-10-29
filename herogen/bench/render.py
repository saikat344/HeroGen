"""Render one known 3D character into a ground-truth run folder.

The run folder has the stage-2 layout of docs/ARCHITECTURE.md, so the building stages read it unchanged, plus
the true mesh and the evaluation views:

    gt/body.glb                      the true mesh, posed, in the contract frame (feet at Z = 0, facing -Y)
    cameras.json                     every camera: azimuth, role (building / evaluation), axes, pixel mapping
    views/{front,left,back,right}.png, views/az{045,135,225,315}.png
                                     building views: flat unlit base colour on a white background
    views-lit/<view>.png             the same views lit (camera-fixed key and fill light, soft ambient)
    masks/<view>.png                 white figure on black
    depth/<view>.npy                 float32 depth in studs from the plane through the image centre, NaN off
                                     the figure (see cameras.depth_of)
    views/meta.json                  pixels_per_unit, height_units, image_size, mask_method "render"
    heldout/az{025,335,065,295}.png  evaluation views, with `-lit.png`, `-mask.png` and `-depth.npy`
    stage-2.json                     the stage record: "reference": "ground-truth", checks, pose, asset hash
    logs/bench-render.log            Blender's output

No building stage reads `heldout/`, `gt/` or `depth/`. Masks and depth come from casting one orthographic ray
through each pixel centre against the mesh written to `gt/body.glb`, so they describe that mesh exactly; the
colour renders use the same cameras, and the `render_mask_iou_min` check confirms the two line up.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from herogen.blender import bl
from herogen.blender.record import StageRecord, check

from . import BACKGROUND, DEFAULT_SIZE, HEIGHT_STUDS

JOB_SCRIPT = Path(__file__).with_name("bench_job.py")
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_TYPES = (".glb", ".gltf", ".fbx", ".blend")
POSES = ("auto", "a-pose", "rest", "as-shipped")

# Gates. The mask comes from rays through pixel centres and the colour render's coverage from area sampling,
# so they differ only on edge pixels a surface half-covers.
RENDER_MASK_IOU_MIN = 0.97
HEIGHT_TOLERANCE_PX = 1.0       # front mask height vs height_units * pixels_per_unit
POSE_TOLERANCE_DEG = 1.0        # achieved limb angle vs the A-pose target
FACING_MIN_X = 0.999            # cos of the right-to-left axis's angle from +X (about 2.6 degrees)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_blender(job: dict, work: Path, log: Path, timeout: float) -> dict:
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
    log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
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
    out = []
    if pose != "a-pose":
        return out
    from .rig import A_POSE_ARM_DEG, A_POSE_LEG_DEG

    for limb in limbs:
        if limb["limb"] == "arm":
            value, target = limb["after_deg_below_horizontal"], A_POSE_ARM_DEG
        else:
            value, target = limb["after_frontal_deg_from_vertical"], A_POSE_LEG_DEG
        err = abs(value - target)
        out.append(check(f"pose_{limb['limb']}_{limb['side']}_deg", value, target,
                         passed=err <= POSE_TOLERANCE_DEG and limb["after_outward"],
                         tolerance=POSE_TOLERANCE_DEG))
    return out


def render_asset(asset: Path, out: Path, *, include: str | None = None, exclude: str | None = None,
                 yaw_deg: float | None = None, pose: str = "auto", material_images: dict | None = None,
                 material_vcols: dict | None = None, image_dirs: list | None = None,
                 bind_unrigged: bool = False,
                 size: int = DEFAULT_SIZE, height: float = HEIGHT_STUDS, samples_flat: int = 16,
                 samples_lit: int = 128, device: str = "auto", timeout: float = 3600.0) -> dict:
    """Render `asset` into the run folder `out` and return its stage-2 record (see the module docstring)."""
    asset = Path(asset)
    if not asset.is_file():
        raise FileNotFoundError(f"asset not found: {asset}")
    if asset.suffix.lower() not in ASSET_TYPES:
        raise ValueError(f"unsupported asset type {asset.suffix!r} (use {', '.join(ASSET_TYPES)})")
    if pose not in POSES:
        raise ValueError(f"pose must be one of {', '.join(POSES)}")
    out = Path(out)
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
        res = run_blender(job, work, out / "logs" / "bench-render.log", timeout)
        cams = res["cameras"]
        outputs, ious, borders = ["gt/body.glb", "cameras.json", "views/meta.json"], {}, {}
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
    rec.add(check("render_mask_iou_min", min(ious.values()), RENDER_MASK_IOU_MIN, per_view=ious))
    rec.add(check("missing_images", len(res["missing_images"]), 0, passed=not res["missing_images"],
                  names=res["missing_images"]))
    rec.add(facing_check(res["facing"]))
    for c in pose_checks(res["pose"], res["limbs"]):
        rec.add(c)
    rec.data["models"] = []
    rec.data["tools"] = [f"Blender {res['blender']}", f"Cycles ({res['device']})"]
    rec.data["outputs"] = outputs
    rec.data["cost_basis"] = "local compute only (Blender); no paid generation"
    rec.note(f"ground-truth render of {asset.name}; pose {res['pose']}")
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
