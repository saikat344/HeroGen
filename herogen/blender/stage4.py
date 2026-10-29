"""Stage 4: joints, R15 sections, high/low LODs and the pose test."""
from __future__ import annotations

import colorsys
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import (HEIGHT_STUDS, PARENT, PIVOT_JOINT, SECTIONS, TRIANGLES_HIGH_MAX, TRIANGLES_LOW_MAX, VIEWS, bl,
               covers, joints, pose, split, texture)
from .carve import winding_inside
from .stage3 import IOU_GATE, iou
from .views import load_mask
from .record import StageRecord, check
from .stage3 import work_dir
from .views import load_views

# Largest allowed joint gap (studs) in the gated poses. 0.15 studs is 2.7% of the 5.5-stud body: about the
# width of a finger, and (estimate) about 10 px on a phone at ZBurst's 10-stud review distance. Measured worst gap per
# pose (synthetic body, high LOD): flat caps 0.15-0.24, spherical caps 0.05-0.12 (low LOD 0.04-0.08).
GAP_GATE = 0.15
# A 30-degree waist twist of a rigid torso whose cross-section is about twice as wide as deep always leaves a
# ledge as wide as the difference of the half-axes (about 0.18 studs here), whatever the cap; it is measured
# and reported, not gated.
UNGATED_POSES = {"twist_turn"}
JOINT_ERROR_GATE = 0.10        # synthetic runs only: worst joint error vs the known truth, studs
# Texture gates, per observed colour view, textured rest body rendered flat vs the view inside both silhouettes.
# Synthetic painted humanoid (known colours): mean abs error 1.9-5.2 (0-255), SSIM 0.80-0.87. Controls on the
# same renders: a flat mean colour gives error 65-67; the opposite view mirrored 18.7-79.6; the render shifted
# 6 px (about 0.075 studs) 18.8-30.9 and SSIM 0.51-0.70. Error <= 12 sits between the worst good view and the
# nearest control. SSIM cannot tell the synthetic bands from their mirror (0.85), so it gates registration:
# >= 0.70 passes every good view (synthetic min 0.80, pencil sketch 0.80) and fails the 6 px shift.
TEX_ERROR_GATE = 12.0
TEX_SSIM_GATE = 0.70
# Cap resolution per LOD (rings, first-ring points). On the synthetic body the caps cost about 3.0k of the
# high LOD's 23.9k triangles and 1.3k of the low LOD's 5.8k.
DOME = {"high": (2, 12), "low": (1, 8)}
# Shoulder covers. Worst shoulder gap that stays visible (gated poses, high LOD, radius 2.5 cut radii):
#   owner UpperTorso vs UpperArm: pencil sketch 0.083 vs 0.129, stage-2 stand-in 0.085 vs 0.180 (radius 2.0),
#   synthetic 0.035 vs 0.060. A pad that stays on the torso keeps covering the socket while the arm swings
#   under it; a pad on the arm swings away with it and uncovers the socket.
# Outset: pushing the pad 0.06 studs out everywhere cut the synthetic top-view rest IoU to 0.899 (0.971 with no
#   outset), so each pad vertex is pushed out only while it stays inside every observed mask grown by 3 px.
#   Pencil sketch, worst visible gap high/low and worst observed rest IoU: 0 px tolerance 0.148/0.157, 0.904;
#   3 px 0.125/0.140, 0.901; 6 px 0.125/0.140, 0.894 (fails the 0.90 IoU gate).
COVER_OWNER = "UpperTorso"
COVER_RADIUS = 2.5
COVER_OUTSET = 0.06
COVER_MASK_TOLERANCE_PX = 3
# Atlas sizes: ZBurst's hero LODs use 1024 maps for High and 512 for Low.
ATLAS = {"high": 1024, "low": 512}
TARGET_FRACTIONS = (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85)


def section_colors() -> dict[str, list[float]]:
    out = {}
    for i, name in enumerate(SECTIONS):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.45, 0.85)
        out[name] = [round(r, 3), round(g, 3), round(b, 3)]
    return out


def _cam(view: str, forward, target, scale: float, res: int = 512) -> dict:
    f = np.asarray(forward, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return {"view": view, "right": right.tolist(), "up": up.tolist(), "forward": f.tolist(),
            "location": (np.asarray(target) - f * 40).tolist(), "ortho_scale": scale, "resolution": res}


def pose_cameras(height: float) -> list[dict]:
    t = [0.0, 0.0, height * 0.5]
    return [_cam("quarter", [-0.6, 1.0, -0.35], t, height * 1.35), _cam("side", [-1.0, 0.0, 0.0], t, height * 1.35)]


def _save_lod(path: Path, secs: dict, uvs: dict | None = None) -> None:
    arrs = {}
    for n, s in secs.items():
        if len(s.faces) == 0:
            continue
        arrs[f"{n}__verts"] = s.verts
        arrs[f"{n}__faces"] = s.faces
        arrs[f"{n}__pivot"] = s.pivot
        if uvs is not None and n in uvs:
            arrs[f"{n}__uv"] = uvs[n]
    np.savez(path, **arrs)


def _open_edges(faces: np.ndarray) -> int:
    if len(faces) == 0:
        return 0
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return int((c == 1).sum())


def _comparison_sheet(colour_views: dict, renders: dict, path: Path) -> Path | None:
    """Turnaround views (top row) over the textured rest renders from the same cameras (bottom row)."""
    from PIL import Image

    names = [v for v in texture.COLOUR_VIEWS if v in colour_views and v in renders]
    if not names:
        return None
    tile = 512
    sheet = Image.new("RGB", (tile * len(names), tile * 2), (255, 255, 255))
    for i, v in enumerate(names):
        sheet.paste(Image.fromarray(colour_views[v][1].astype(np.uint8)).resize((tile, tile)), (tile * i, 0))
        with Image.open(renders[v]) as im:
            sheet.paste(im.convert("RGB").resize((tile, tile)), (tile * i, tile))
    sheet.save(path)
    return path


def run_stage4(run: Path, height: float = HEIGHT_STUDS, timeout: float = 900.0) -> dict:
    rec = StageRecord(run, 4, "game body")
    work = work_dir(run)
    mesh_npz = work / "03-mesh.npz"
    mesh_src = mesh_npz if mesh_npz.exists() else run / "03-mesh.glb"
    if not mesh_src.exists():
        raise FileNotFoundError("stage 4 needs stage 3's mesh (03-mesh.glb)")
    vs = load_views(run, height)
    sk = joints.estimate(vs, height)
    for n in sk.notes:
        rec.note(n)
    budgets = {"high": TRIANGLES_HIGH_MAX, "low": TRIANGLES_LOW_MAX}

    # Decimate the whole body to a ladder of targets in one Blender call, then section each candidate and
    # keep the largest whose sectioned total (cuts and caps included) fits the budget.
    targets, outs = {}, {}
    for lod, budget in budgets.items():
        for k, frac in enumerate(TARGET_FRACTIONS):
            key = f"{lod}{k}"
            targets[key] = int(budget * frac)
            outs[key] = str((work / f"04-dec-{key}.npz").resolve())
    # coarse bodies the shoulder covers are shaped from (high LOD covers, low LOD covers)
    for key, n_tri in (("coverhigh", 2700), ("coverlow", 1200)):
        targets[key] = n_tri
        outs[key] = str((work / f"04-dec-{key}.npz").resolve())
    dec = bl.run_job({"tasks": [{"task": "decimate", "mesh": str(mesh_src.resolve()), "targets": targets,
                                 "out": outs}]}, work, "stage4-decimate", timeout=timeout)
    rec.data["tools"] = [f"blender {dec['blender']}"]
    method = vs.meta.get("mask_method") or {}
    observed = []
    for name, v in vs.views.items():
        if "visual_hull" in str(method.get(name, "")):
            continue
        m = v.mask.copy()
        for _ in range(COVER_MASK_TOLERANCE_PX):     # grow the mask by the tolerance (4-neighbour steps)
            m[1:] |= m[:-1].copy(); m[:-1] |= m[1:].copy(); m[:, 1:] |= m[:, :-1].copy(); m[:, :-1] |= m[:, 1:].copy()
        observed.append(replace(v, mask=m))

    def in_masks(pts: np.ndarray) -> np.ndarray:
        ok = np.ones(len(pts), dtype=bool)
        for v in observed:
            ok &= v.inside(pts)
        return ok

    # Covers are shaped from a coarse decimation of the body (2.7k triangles for the high LOD's covers, 1.2k
    # for the low LOD's) so they stay cheap; they count in both budgets.
    chosen, lod_secs, bare_secs, lod_info, cover_info = {}, {}, {}, {}, {}
    for lod, budget in budgets.items():
        cover_src = np.load(outs["cover" + lod])
        best = None
        for k in range(len(TARGET_FRACTIONS)):
            key = f"{lod}{k}"
            d = np.load(outs[key])
            secs, rig = split.build_sections(d["verts"], d["faces"], sk, cap="dome", rings=DOME[lod][0],
                                             ring_points=DOME[lod][1])
            covered, cinfo = covers.add_covers(secs, cover_src["verts"], cover_src["faces"], owner=COVER_OWNER,
                                               radius=COVER_RADIUS, outset=COVER_OUTSET, inside=in_masks)
            total = sum(len(s.faces) for s in covered.values())
            if best is None or (total <= budget and (best[1] > budget or total > best[1])) \
                    or (best[1] > budget and total < best[1]):
                best = (key, total, covered, secs, cinfo, len(d["faces"]))
        chosen[lod] = best[0]
        lod_secs[lod], bare_secs[lod], cover_info[lod] = best[2], best[3], best[4]
        lod_info[lod] = {"body_triangles_after_decimation": best[5], "triangles": best[1],
                         "cap_triangles": sum(s.cap_triangles for s in best[3].values()),
                         "cover_triangles": sum(c["triangles"] for c in best[4].values()),
                         "dome_rings": DOME[lod][0], "dome_ring_points": DOME[lod][1], "atlas_size": ATLAS[lod]}

    # Pose test. Gated: the gap that stays visible with the covers on (both LODs). Reported alongside: the
    # visible gap without covers, the purely geometric gap without covers, and flat caps.
    poses = pose.pose_library(pose.bone_directions(sk))
    gaps = {lod: pose.joint_gaps(lod_secs[lod], poses, visible=True) for lod in lod_secs}
    gaps_nocover_visible = pose.joint_gaps(bare_secs["high"], poses, visible=True)
    gaps_nocover_geometric = pose.joint_gaps(bare_secs["high"], poses)
    d_hi = np.load(outs[chosen["high"]])
    flat_secs, _ = split.build_sections(d_hi["verts"], d_hi["faces"], sk, cap="flat")
    gaps_flat = pose.joint_gaps(flat_secs, poses)

    # UV atlases (Blender smart project, all sections of a LOD packed together), then colour projection.
    body = run / "04-body"
    (body / "pose-renders").mkdir(parents=True, exist_ok=True)
    (body / "rest-renders").mkdir(parents=True, exist_ok=True)
    lod_paths, uv_paths = {}, {}
    for lod, secs in lod_secs.items():
        lod_paths[lod] = work / f"04-{lod}.npz"
        uv_paths[lod] = work / f"04-{lod}-uv.npz"
        _save_lod(lod_paths[lod], secs)
    bl.run_job({"tasks": [{"task": "uv", "lods": {k: str(v.resolve()) for k, v in lod_paths.items()},
                           "out": {k: str(v.resolve()) for k, v in uv_paths.items()}, "margin": 0.004}]},
               work, "stage4-uv", timeout=timeout)
    colour_views = texture.observed_colour_views(vs, run)
    atlas_paths, tex_stats, zbuf = {}, {}, None
    from PIL import Image
    for lod in ("high", "low"):
        u = np.load(uv_paths[lod])
        uvs = {k.split("__")[0]: u[k] for k in u.files}
        _save_lod(lod_paths[lod], lod_secs[lod], uvs)
        if colour_views:
            atlas, stats, zbuf = texture.bake(lod_secs[lod], uvs, colour_views, ATLAS[lod], zbuf)
            atlas_paths[lod] = body / f"atlas-{lod}.png"
            Image.fromarray(atlas).save(atlas_paths[lod])
            tex_stats[lod] = stats

    # Blender: textured GLBs, rest renders from the contract cameras, pose renders.
    rest_cams = [vs.views[v].camera() for v in VIEWS if v in vs.views]
    pivots = {n: s.pivot for n, s in lod_secs["high"].items()}
    pose_jobs = [{"name": name, "matrices": {n: m.tolist() for n, m in pose.world_matrices(loc, pivots).items()}}
                 for name, loc in poses.items()]
    res = bl.run_job({"tasks": [{
        "task": "body", "lods": {k: str(v.resolve()) for k, v in lod_paths.items()},
        "glb": {"high": str((body / "high.glb").resolve()), "low": str((body / "low.glb").resolve())},
        "atlas": {k: str(v.resolve()) for k, v in atlas_paths.items()},
        "colors": section_colors(), "poses": pose_jobs, "pose_cameras": pose_cameras(height),
        "pose_dir": str((body / "pose-renders").resolve()), "pose_lod": "high",
        "rest_cameras": rest_cams, "rest_dir": str((body / "rest-renders").resolve())}]},
        work, "stage4-body", timeout=timeout)
    bres = res["results"][0]

    # Checks.
    hi, lo = lod_secs["high"], lod_secs["low"]
    nonempty = sum(1 for n in SECTIONS if len(hi[n].faces) and len(lo[n].faces))
    rec.add(check("sections_non_empty", nonempty, 15))
    for lod, budget in budgets.items():
        rec.add(check(f"triangles_{lod}", lod_info[lod]["triangles"], budget,
                      passed=lod_info[lod]["triangles"] <= budget, rule="<="))
    open_e = {f"{lod}:{n}": _open_edges(s.faces) for lod, ss in lod_secs.items() for n, s in ss.items()}
    rec.add(check("open_edges", sum(open_e.values()), 0, passed=sum(open_e.values()) == 0, rule="=="))
    pivot_rows = {}
    worst_off = 0.0
    all_inside = True
    for n in SECTIONS:
        s = hi[n]
        jn = PIVOT_JOINT[n]
        est = np.asarray(sk.joints[jn], float)
        rim = [r["points"] for r in s.rims if r["neighbour"] == PARENT[n] and len(r["points"])]
        rim_r = float(np.linalg.norm(np.concatenate(rim) - s.pivot, axis=1).mean()) if rim else None
        inside = bool(winding_inside(s.pivot[None], s.verts, s.faces)[0]) if len(s.faces) else False
        off = float(np.linalg.norm(s.pivot - est))
        ok = inside and (rim_r is None or off <= max(rim_r, 0.05))
        all_inside &= ok
        if rim_r:
            worst_off = max(worst_off, off / rim_r)
        pivot_rows[n] = {"inside_section": inside, "offset_from_mask_joint": round(off, 4),
                         "rim_radius": None if rim_r is None else round(rim_r, 4), "ok": ok}
    rec.add(check("pivots_in_joint_region", int(sum(r["ok"] for r in pivot_rows.values())), 15,
                  rule="pivot inside its section and within one cut radius of the mask joint",
                  worst_offset_over_radius=round(worst_off, 3)))
    gated = {lod: max(v for p, row in g.items() if p not in UNGATED_POSES for v in row.values())
             for lod, g in gaps.items()}
    for lod in ("high", "low"):
        rec.add(check(f"max_joint_gap_{lod}", round(gated[lod], 4), GAP_GATE, passed=gated[lod] <= GAP_GATE,
                      rule="<=", poses=sorted(p for p in poses if p not in UNGATED_POSES)))
    for p in sorted(UNGATED_POSES):
        rec.add(check(f"max_joint_gap_high_{p}", round(max(gaps["high"][p].values()), 4), GAP_GATE,
                      passed=max(gaps["high"][p].values()) <= GAP_GATE, gate=False, rule="<="))
    for name, g in (("visible_no_covers", gaps_nocover_visible), ("geometric_no_covers", gaps_nocover_geometric)):
        val = max(v for p, row in g.items() if p not in UNGATED_POSES for v in row.values())
        rec.add(check(f"max_joint_gap_high_{name}", round(val, 4), GAP_GATE, passed=val <= GAP_GATE, gate=False,
                      rule="<=", note="comparison only"))
    # The rest silhouette with covers must still match the masks (observed views gated like stage 3).
    rest_iou = {}
    for v, path in bres["rest_silhouettes"].items():
        rest_iou[v] = round(iou(load_mask(Path(path)), vs.views[v].mask), 4)
        derived = "visual_hull" in str(method.get(v, ""))
        rec.add(check(f"rest_iou_with_covers_{v}", rest_iou[v], IOU_GATE, gate=not derived))
    # Texture: the textured rest body rendered flat from each observed colour view against that view.
    tex_metrics = {}
    for v, path in bres["rest_textured"].items():
        if v not in colour_views:
            continue
        with Image.open(path) as im:
            rend = np.asarray(im.convert("RGB"), dtype=np.float64)
        rmask = load_mask(Path(bres["rest_silhouettes"][v]))
        tex_metrics[v] = texture.compare(rend, rmask, colour_views[v][1], vs.views[v].mask)
        rec.add(check(f"texture_colour_error_{v}", tex_metrics[v]["mean_abs_error"], TEX_ERROR_GATE,
                      passed=tex_metrics[v]["mean_abs_error"] <= TEX_ERROR_GATE, rule="<="))
        rec.add(check(f"texture_ssim_{v}", tex_metrics[v]["ssim"], TEX_SSIM_GATE))
    if not colour_views:
        rec.note("no observed colour views: body left untextured")
    sheet = _comparison_sheet(colour_views, bres["rest_textured"], body / "textured-vs-views.png")
    poke = sum(s.cap_points_outside for s in hi.values())
    rec.add(check("cap_points_outside_neighbour_high", poke, 0, passed=poke == 0, gate=False, rule="==",
                  note="cap samples left outside the neighbour at rest after flattening (visible poke-through)"))
    flat_gated = max(v for p, row in gaps_flat.items() if p not in UNGATED_POSES for v in row.values())
    rec.add(check("max_joint_gap_high_flat_caps", round(flat_gated, 4), GAP_GATE,
                  passed=flat_gated <= GAP_GATE, gate=False, rule="<=",
                  note="comparison only: the same cut closed with flat fans"))
    truth_path = run / "synthetic" / "truth.json"
    joint_err = None
    if truth_path.exists():
        truth = json.loads(truth_path.read_text())["joints"]
        errs = joints.joint_errors(sk.as_json(), truth)
        joint_err = {"mean": round(float(np.mean(list(errs.values()))), 4),
                     "max": round(float(max(errs.values())), 4),
                     "per_joint": {k: round(v, 4) for k, v in errs.items()}}
        piv_err = {n: round(float(np.linalg.norm(hi[n].pivot - np.asarray(truth[PIVOT_JOINT[n]]))), 4)
                   for n in SECTIONS if PIVOT_JOINT[n] in truth}
        joint_err["pivot_vs_truth_max"] = max(piv_err.values())
        joint_err["pivot_vs_truth_mean"] = round(float(np.mean(list(piv_err.values()))), 4)
        rec.add(check("joint_error_max_synthetic", joint_err["max"], JOINT_ERROR_GATE,
                      passed=joint_err["max"] <= JOINT_ERROR_GATE, rule="<=", mean=joint_err["mean"]))

    sections_json = {
        "units": "studs", "height": height,
        "frame": "Blender: +Z up, feet at z=0, faces -Y, x/y centred on the figure's bounding box; "
                 "GLBs are exported Y-up (glTF), front along +Z",
        "sections": [{
            "name": n, "parent": PARENT[n], "joint": PIVOT_JOINT[n],
            "pivot": [round(float(c), 4) for c in hi[n].pivot],
            "triangles_high": int(len(hi[n].faces)), "triangles_low": int(len(lo[n].faces)),
            "cap_triangles_high": hi[n].cap_triangles, "cap_triangles_low": lo[n].cap_triangles,
            "cap_points_outside_high": hi[n].cap_points_outside,
            "bounds_min": [round(float(c), 4) for c in hi[n].verts.min(axis=0)] if len(hi[n].verts) else None,
            "bounds_max": [round(float(c), 4) for c in hi[n].verts.max(axis=0)] if len(hi[n].verts) else None,
            "pivot_check": pivot_rows[n],
        } for n in SECTIONS],
        "joints": sk.as_json(),
        "ends": {k: [round(float(c), 4) for c in v] for k, v in sk.ends.items()},
        "joint_evidence": sk.evidence,
        "joint_method": joints.__doc__.strip(),
        "split_method": split.__doc__.strip(),
        "budgets": {"triangles_high_max": TRIANGLES_HIGH_MAX, "triangles_low_max": TRIANGLES_LOW_MAX},
        "lods": lod_info,
        "pose_test": {"method": pose.__doc__.strip(), "visible_gap_method": pose.joint_gaps.__doc__.strip(),
                      "gate_studs": GAP_GATE, "ungated_poses": sorted(UNGATED_POSES),
                      "gaps_high": gaps["high"], "gaps_low": gaps["low"],
                      "gaps_high_visible_no_covers": gaps_nocover_visible,
                      "gaps_high_geometric_no_covers": gaps_nocover_geometric, "gaps_high_flat_caps": gaps_flat},
        "covers": {"method": covers.__doc__.strip(), "owner": COVER_OWNER, "radius_cut_radii": COVER_RADIUS,
                   "outset_studs": COVER_OUTSET, "pieces_high": cover_info["high"], "pieces_low": cover_info["low"],
                   "contract_note": "covers are merged into their owner section so each GLB keeps the 15 R15 "
                                    "parts; they are listed under sections.json \"covers\""},
        "texture": {"method": texture.__doc__.strip(), "atlas_sizes": ATLAS,
                    "atlases": {k: str(v.relative_to(run)) for k, v in atlas_paths.items()},
                    "colour_views": sorted(colour_views), "bake": tex_stats, "rest_metrics": tex_metrics,
                    "gates": {"mean_abs_error_max": TEX_ERROR_GATE, "ssim_min": TEX_SSIM_GATE}},
        "rest_iou_with_covers": rest_iou,
        "joint_error_synthetic": joint_err,
        "checks": rec.data["checks"],
    }
    (body / "sections.json").write_text(json.dumps(sections_json, indent=2))
    rec.data["outputs"] = (["04-body/high.glb", "04-body/low.glb", "04-body/sections.json"]
                           + [str(v.relative_to(run)) for v in atlas_paths.values()]
                           + ([str(sheet.relative_to(run))] if sheet else [])
                           + [str(Path(p).relative_to(run.resolve())) for p in bres["pose_renders"]]
                           + [str(Path(p).relative_to(run.resolve())) for p in bres["rest_textured"].values()])
    rec.data["body"] = {"lods": lod_info, "decimation": dec["results"][0], "chosen_targets": chosen,
                        "max_gap_by_pose_high": {p: max(r.values()) for p, r in gaps["high"].items()},
                        "max_gap_by_pose_low": {p: max(r.values()) for p, r in gaps["low"].items()},
                        "max_gap_by_pose_high_flat": {p: max(r.values()) for p, r in gaps_flat.items()},
                        "max_gap_by_pose_high_visible_no_covers":
                            {p: max(r.values()) for p, r in gaps_nocover_visible.items()},
                        "max_gap_by_pose_high_geometric_no_covers":
                            {p: max(r.values()) for p, r in gaps_nocover_geometric.items()},
                        "covers": cover_info, "texture": {"bake": tex_stats, "rest_metrics": tex_metrics},
                        "rest_iou_with_covers": rest_iou,
                        "joint_error_synthetic": joint_err, "joint_evidence": sk.evidence}
    return rec.finish()
