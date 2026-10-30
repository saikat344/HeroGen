"""Training data for the face and head: many characters rendered with the bench's exact cameras, plus head
close-ups and per-pixel labels, resumable and several at a time.

A manifest (JSON) lists the characters:

    {"seed": "...", "heldout_fraction": 0.1,
     "characters": [{"id": "objv-0a1b...", "asset": "raw/objaverse/0a1b....glb", "group": "...",
                     "split": "train" | "heldout", "pose": "auto", "yaw_deg": null, "include": null, ...}]}

`asset` is relative to the manifest's folder. One character's folder, `<out>/<id>/`:

    gt/body.glb              the posed mesh in the contract frame (feet at Z = 0, facing -Y), with textures
    gt/mesh.npz              the same triangles as the label maps: vertices, triangles, per-corner UVs (the
                             render UV map), per-triangle material index, per-vertex head weight
    gt/textures/*.png        the base-colour images each material reads
    views/<view>-flat.png    unlit base colour, RGBA on transparent
    views/<view>-lit.png     lit (camera-fixed key and fill, soft ambient), RGBA
    views/<view>.npz         per pixel centre: mask, depth (float16 studs, see cameras.depth_of; NaN off the
                             figure), uv (float16), material (uint8: 0 none, else index + 1), head (uint8: the
                             share of skin weight that follows the head bone, 0-255), face (face material)
    logs/render.log          Blender's output
    sample.json              written last, so its presence marks a finished character: what was rendered,
                             from what, with the cameras and checks, and `usable` with the reasons when a
                             sample is not fit for training (see `usability`)
    failed.json              instead of sample.json, when the character could not be rendered

Views: the bench's 12 (8 building, 4 evaluation) and 6 head close-ups (`head-front`, `head-front-left-45`,
`head-left`, `head-back`, `head-right`, `head-front-right-45`), which centre on the head region (the rig's
head bone and everything below it) at one pixel scale.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from herogen.blender import bl

from . import DEFAULT_SIZE, HEIGHT_STUDS
from .render import (
    ASSET_TYPES,
    REPO_ROOT,
    border_pixels,
    coverage_margin,
    off_edge_disagreement,
    run_blender,
    sha256,
)

SAMPLE = "sample.json"
FAILED = "failed.json"
DEFAULT_HEAD_SIZE = 1024
DEFAULT_SAMPLES_LIT = 64
# A sample is usable for training when its head can be learnt from: a head region from the rig, at the top
# of an upright figure, filling a fair share of the head close-up, and more than one flat colour.
HEAD_TOP_SHARE = 0.6        # the head region's centre sits above this share of the figure's height
HEAD_MIN_SHARE = 0.05       # head pixels in the `head-front` close-up, as a share of the image
HEAD_MIN_COLOURS = 4        # distinct colours (5 bits a channel) on the head in the `head-front` flat render
# Render/label disagreement allowed away from the mask's edge, as a share of the figure's pixels. The bench
# gates at 0; here a close-up magnifies alpha-clipped textures enough that a few texels sit at the cutoff
# itself (3-4 of 460,000 pixels in bench C15's head close-ups), where Cycles' float32 and the labels' float64
# sampling may round either way. A camera or scale error puts thousands of pixels on the wrong side.
OFF_EDGE_MAX_SHARE = 2e-5
OPTIONS = ("pose", "yaw_deg", "include", "exclude", "material_images", "material_vcols", "material_alpha",
           "bind_unrigged")


def heldout_draw(group: str, seed: str, fraction: float) -> bool:
    """True when `group` falls in the held-out share: a seeded hash, so a group's split never changes as
    characters are added, and every character of a group (variants of one model) lands on the same side."""
    h = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16)
    return h / float(1 << 64) < fraction


def status(dest: Path) -> str:
    """'done', 'failed' or 'todo' for a character folder."""
    if (dest / SAMPLE).is_file():
        return "done"
    if (dest / FAILED).is_file():
        return "failed"
    return "todo"


def view_checks(views: Path, cams: list[dict], samples_flat: int) -> tuple[dict, bool]:
    """Per view: pixels where the flat render and the mask disagree away from the mask's edge (at most
    `OFF_EDGE_MAX_SHARE` of the mask: the render and the labels line up) and, for whole-figure views, mask
    pixels on the image border (must be 0: the figure is all in frame). Head close-ups crop the body below
    the head, so their border is recorded only."""
    out, ok = {}, True
    margin = coverage_margin(samples_flat)
    for cam in cams:
        with np.load(views / f"{cam['view']}.npz") as arr:
            mask = arr["mask"]
            head = int((arr["head"] >= 128).sum())
        alpha = np.asarray(Image.open(views / f"{cam['view']}-flat.png").convert("RGBA"))[..., 3] / 255.0
        rec = {"off_edge_px": off_edge_disagreement(alpha, mask, margin), "border_px": border_pixels(mask),
               "mask_px": int(mask.sum()), "head_px": head}
        rec["ok"] = view_ok(rec, cam["role"])
        ok = ok and rec["ok"]
        out[cam["view"]] = rec
    return out, ok


def view_ok(rec: dict, role: str) -> bool:
    """Something is seen, the render and the labels line up, and a whole-figure view has it all in frame."""
    lined_up = rec["off_edge_px"] <= OFF_EDGE_MAX_SHARE * rec["mask_px"]
    return rec["mask_px"] > 0 and lined_up and (role == "head" or rec["border_px"] == 0)


def invisible_holes(res: dict, views: Path) -> list[str]:
    """Materials whose constant alpha below the cutoff made them all hole, when nothing at all is seen in the
    front view; [] otherwise."""
    holes = sorted(m for m, a in (res.get("alpha") or {}).items()
                   if a.get("mode") == "clip" and a.get("image") is None)
    if not holes:
        return []
    with np.load(views / "front.npz") as arr:
        return [] if arr["mask"].any() else holes


def usability(sample: dict, flat: np.ndarray, head: np.ndarray) -> list[str]:
    """Why a rendered sample is not usable for training (empty when it is). `flat` is the `head-front` flat
    render (RGBA), `head` its head-region mask."""
    why = []
    h = sample.get("head") or {}
    if h.get("source") != "rig":
        why.append("no head region from the rig")
    else:
        top = float(h["centre"][2]) / float(sample["height_units"])
        if top < HEAD_TOP_SHARE:
            why.append(f"head centre at {top:.2f} of the height: the figure is not upright")
    share = float(head.mean()) if head.size else 0.0
    if share < HEAD_MIN_SHARE:
        why.append(f"head fills {share:.3f} of the head-front close-up")
    solid = head & (flat[..., 3] == 255)
    codes = (flat[solid][:, :3] >> 3).astype(np.int32) @ np.array([1024, 32, 1])
    colours = len(np.unique(codes))
    if colours < HEAD_MIN_COLOURS:
        why.append(f"{colours} colour(s) on the head: untextured")
    if sample.get("pose") != "a-pose":
        why.append(f"not posed ({sample.get('pose')})")
    elif not all(limb.get("converged") for limb in sample.get("limbs", [])):
        why.append("a limb did not reach the A-pose")
    return why


def render_character(entry: dict, root: Path, out: Path, *, size: int = DEFAULT_SIZE,
                     head_size: int = DEFAULT_HEAD_SIZE, height: float = HEIGHT_STUDS, samples_flat: int = 16,
                     samples_lit: int = DEFAULT_SAMPLES_LIT, device: str = "auto",
                     timeout: float = 1800.0) -> dict:
    """Render one manifest entry into `out/<id>/` unless it is already done; returns a short status."""
    cid = entry["id"]
    dest = out / cid
    if status(dest) == "done":
        return {"id": cid, "status": "done", "skipped": True}
    asset = (root / entry["asset"]).resolve()
    if dest.exists():
        shutil.rmtree(dest)
    work = dest / ".work"
    work.mkdir(parents=True)
    t0 = time.time()
    try:
        if not asset.is_file() or asset.suffix.lower() not in ASSET_TYPES:
            raise ValueError(f"asset missing or not one of {', '.join(ASSET_TYPES)}: {entry['asset']}")
        job = {
            "repo_root": str(REPO_ROOT), "asset": str(asset),
            "include": entry.get("include"), "exclude": entry.get("exclude"),
            "yaw_deg": entry.get("yaw_deg"), "pose": entry.get("pose", "auto"),
            "material_images": entry.get("material_images") or {},
            "material_vcols": entry.get("material_vcols") or {},
            "material_alpha": entry.get("material_alpha") or {},
            "image_dirs": [str((root / d).resolve()) for d in entry.get("image_dirs") or []],
            "bind_unrigged": bool(entry.get("bind_unrigged")), "height": height, "size": size,
            "samples_flat": samples_flat, "samples_lit": samples_lit, "device": device,
            "gt_glb": str((dest / "gt" / "body.glb").resolve()), "work": str((work / "views").resolve()),
            "dataset": {"head_size": head_size},
        }
        places = {"<work>": work.resolve(), "<run>": dest.resolve(), "<asset-dir>": asset.parent,
                  "<repo>": REPO_ROOT, "<blender>": Path(bl.find_blender() or ".").resolve().parent}
        res = run_blender(job, work, dest / "logs" / "render.log", timeout, places)
        retry = invisible_holes(res, work / "views") if not job["material_alpha"] else []
        if retry:
            # FBX import can leave a material at alpha 0 (a transparency setting read as opacity), which hides
            # the whole figure; render it again with those materials opaque
            shutil.rmtree(work / "views")
            job["material_alpha"] = {m: "opaque" for m in retry}
            res = run_blender(job, work, dest / "logs" / "render.log", timeout, places)
            res["alpha_retry"] = {"materials": retry, "why": "constant alpha below the cutoff hid every view"}
    except (ValueError, bl.BlenderError) as exc:
        (dest / FAILED).write_text(json.dumps({"id": cid, "error": str(exc)[-2000:],
                                               "seconds": round(time.time() - t0, 1)}, indent=2))
        shutil.rmtree(work, ignore_errors=True)
        return {"id": cid, "status": "failed", "error": str(exc).splitlines()[0][:300]}
    shutil.move(str(work / "views"), str(dest / "views"))
    shutil.rmtree(work, ignore_errors=True)
    cams = res["cameras"]
    checks, ok = view_checks(dest / "views", cams, samples_flat)
    sample = {
        "id": cid, "ok": ok, "split": entry.get("split"), "group": entry.get("group"),
        "asset": entry["asset"], "asset_sha256": sha256(asset),
        "source": {k: entry.get(k) for k in ("source", "licence", "author", "url") if k in entry},
        "options": {k: entry[k] for k in OPTIONS if entry.get(k) not in (None, {}, [], False)},
        "seconds": round(time.time() - t0, 1),
        "image_size": size, "head_size": head_size, "height_units": height,
        "samples": {"flat": samples_flat, "lit": samples_lit},
        "pose": res["pose"], "limbs": res["limbs"], "yaw_deg": res["yaw_deg"], "facing": res["facing"],
        "head": res["head"], "materials": res["materials_index"], "textures": res["textures"],
        "alpha": res["alpha"], "skipped_geometry": res["skipped_geometry"],
        "triangles": res["triangles"], "vertices": res["vertices"], "device": res["device"],
        "timings_s": res["timings"], "cameras": cams, "checks": checks,
    }
    if "alpha_retry" in res:
        sample["alpha_retry"] = res["alpha_retry"]
        sample["options"]["material_alpha"] = {m: "opaque" for m in res["alpha_retry"]["materials"]}
    flat = np.asarray(Image.open(dest / "views" / "head-front-flat.png").convert("RGBA"))
    with np.load(dest / "views" / "head-front.npz") as arr:
        head = arr["head"] >= 128
    sample["unusable_reasons"] = usability(sample, flat, head)
    sample["usable"] = ok and not sample["unusable_reasons"]
    (dest / SAMPLE).write_text(json.dumps(sample, indent=2))
    return {"id": cid, "status": "done", "ok": ok, "usable": sample["usable"], "seconds": sample["seconds"]}


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    chars = data["characters"] if isinstance(data, dict) else data
    seen = set()
    for c in chars:
        if not c.get("id") or "/" in c["id"] or "\\" in c["id"] or c["id"] in (".", ".."):
            raise ValueError(f"bad character id {c.get('id')!r}")
        if c["id"] in seen:
            raise ValueError(f"duplicate character id {c['id']!r}")
        seen.add(c["id"])
    return chars


def run_batch(manifest: Path, out: Path, *, workers: int = 2, split: str | None = None,
              ids: list[str] | None = None, limit: int | None = None, retry_failed: bool = False,
              log=print, **render_kw) -> dict:
    """Render every manifest entry not yet done, `workers` at a time (each a Blender process sharing the
    GPU). Finished characters are skipped, so a stopped batch picks up where it left off; failed ones are
    retried only with `retry_failed`. Returns counts."""
    root = Path(manifest).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    chars = load_manifest(manifest)
    if split:
        chars = [c for c in chars if c.get("split") == split]
    if ids:
        want = set(ids)
        chars = [c for c in chars if c["id"] in want]
    todo = [c for c in chars if status(out / c["id"]) == "todo"
            or (retry_failed and status(out / c["id"]) == "failed")]
    if limit is not None:
        todo = todo[:limit]
    counts = {"queued": len(todo), "done": 0, "ok": 0, "usable": 0, "failed": 0}
    lock = threading.Lock()
    t0 = time.time()

    def one(entry):
        try:
            r = render_character(entry, root, out, **render_kw)
        except Exception as exc:  # noqa: BLE001 - one bad character must not stop the batch
            (out / entry["id"]).mkdir(parents=True, exist_ok=True)
            (out / entry["id"] / FAILED).write_text(json.dumps({"id": entry["id"], "error": repr(exc)}))
            r = {"id": entry["id"], "status": "failed", "error": repr(exc)[:300]}
        with lock:
            counts["done" if r["status"] == "done" else "failed"] += 1
            counts["ok"] += bool(r.get("ok"))
            counts["usable"] += bool(r.get("usable"))
            n = counts["done"] + counts["failed"]
            rate = (time.time() - t0) / n
            log(f"[{n}/{len(todo)}] {r['id']}: {r['status']}"
                + (f" ok={r.get('ok')} usable={r.get('usable')} {r.get('seconds')}s" if r["status"] == "done"
                   else f" {r.get('error')}")
                + f" (eta {rate * (len(todo) - n) / 60:.0f} min)")
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(one, todo))
    return counts
