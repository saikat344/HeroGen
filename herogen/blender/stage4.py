"""Stage 4: joints, R15 sections, the LODs a target profile asks for (`herogen.blender.PROFILES`) and the pose
test."""
from __future__ import annotations

import colorsys
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import (DEFAULT_PROFILE, HEIGHT_STUDS, PARENT, PIVOT_JOINT, PROFILES, SECTIONS, VIEWS, bl, body_joints,
               covers, cuts, joints, pose, split, texture)
from .carve import winding_inside
from .stage3 import IOU_GATE, iou
from .views import load_mask
from .record import StageRecord, check, write_atomic
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
# Comparison variant only (atlas-high-side-trust.png): weight of a side view that did not shape the mesh. On the
# pencil figure's hybrid (the pinned runners' results, in-game high body, numpy flat renders, colour error front /
# left): 1.0 11.7 / 14.2, 0.3 10.4 / 16.4, 0.1 9.6 / 18.4, 0.03 9.0 / 20.6.
SIDE_TRUST = 0.3
TARGET_FRACTIONS = (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85)
# How each profile's LODs are built: the triangle budget (None: none, the stage 3 mesh undecimated; "groups": the
# avatar part groups), the cap and cover resolution ("high" or "low" settings above) and the atlas size. An
# avatar part group has little room (Torso 1,750 with seven caps and both covers), so it takes the low settings.
LOD_BUILD = {
    "best": {"high": {"budget": None, "dome": "high", "cover": "high", "atlas": ATLAS["high"]}},
    "roblox-ingame": {"high": {"budget": PROFILES["roblox-ingame"]["lods"]["high"], "dome": "high",
                               "cover": "high", "atlas": ATLAS["high"]},
                      "low": {"budget": PROFILES["roblox-ingame"]["lods"]["low"], "dome": "low", "cover": "low",
                              "atlas": ATLAS["low"]}},
    "roblox-avatar": {"high": {"budget": "groups", "dome": "low", "cover": "low", "atlas": ATLAS["high"]}},
}
# Whole-body decimation targets for the avatar profile; each part group then takes the candidate with the most
# triangles that fits its own budget (sections are separate closed meshes, so groups can come from different
# candidates). The ladder reaches down to 1,000: on the pencil-sketch hybrid a waist cut moved half a torso radius
# up made the Torso group 1,752 triangles at the 1,500 target, over its 1,750.
AVATAR_TARGETS = (1000, 1200, 1500, 2000, 2700, 3600, 4800, 6400, 8500, 11000, 15000, 20000, 27000, 36000, 48000,
                  64000)
UNDECIMATED = 10 ** 12          # a target no mesh reaches: the decimate task keeps the mesh as it is


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


def _budgets_record(profile: str) -> dict | None:
    """The profile's triangle budgets as `sections.json` records them (None under `best`)."""
    spec = PROFILES[profile]
    if profile == "best":
        return None
    if "groups" in spec:
        return {"triangles_total_max": spec["triangles_total"],
                "groups": {g: {"sections": list(secs), "triangles_max": n} for g, (secs, n) in spec["groups"].items()},
                "texture_max_px": spec["texture_max_px"], "watertight_parts": spec["watertight_parts"],
                "not_built": list(spec["not_built"])}
    return {"triangles_high_max": spec["lods"]["high"], "triangles_low_max": spec["lods"]["low"]}


def _section_groups(build: dict, profile: str) -> dict:
    """Part group -> its sections, for the candidate selection of one LOD."""
    if build["budget"] == "groups":
        return {g: secs for g, (secs, _) in PROFILES[profile]["groups"].items()}
    return {"body": SECTIONS}


def _pick(cands: list[dict], build: dict, profile: str) -> dict[str, int]:
    """Section -> index of the candidate it is taken from. One budget: the largest candidate whose total fits
    (else the smallest); no budget: the only candidate; part groups: per group, the candidate with the most
    triangles in that group that fits the group's budget (else the fewest)."""
    budget = build["budget"]
    if budget is None:
        return {n: 0 for n in SECTIONS}
    if budget != "groups":
        best = None
        for i, c in enumerate(cands):
            total = c["total"]
            if best is None or (total <= budget and (best[1] > budget or total > best[1])) \
                    or (best[1] > budget and total < best[1]):
                best = (i, total)
        return {n: best[0] for n in SECTIONS}
    out = {}
    for group, (secs, limit) in PROFILES[profile]["groups"].items():
        counts = [sum(len(c["covered"][n].faces) for n in secs) for c in cands]
        fits = [i for i, k in enumerate(counts) if k <= limit]
        i = max(fits, key=lambda j: counts[j]) if fits else min(range(len(cands)), key=lambda j: counts[j])
        out.update({n: i for n in secs})
    return out


def run_stage4(run: Path, height: float = HEIGHT_STUDS, timeout: float = 900.0,
               profile: str = DEFAULT_PROFILE, cut_search: bool = True) -> dict:
    """Stage 4 for one target profile (`herogen.blender.PROFILES`; default `best`). `cut_search=False` keeps every
    cut at its joint (for comparison; the record says so)."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r} (known: {', '.join(PROFILES)})")
    builds = LOD_BUILD[profile]
    # An earlier record must not outlive a run that stops: nothing after stage 4 may read a stale pass.
    (run / "stage-4.json").unlink(missing_ok=True)
    rec = StageRecord(run, 4, "game body")
    work = work_dir(run)
    mesh_npz = work / "03-mesh.npz"
    mesh_src = mesh_npz if mesh_npz.exists() else run / "03-mesh.glb"
    if not mesh_src.exists():
        raise FileNotFoundError("stage 4 needs stage 3's mesh (03-mesh.glb)")
    vs = load_views(run, height)
    sk = joints.estimate(vs, height)
    silhouette_joints = sk.as_json()
    jrec = body_joints.load_joints(run / "03-joints.json")
    joints_source = "silhouette"
    if jrec is not None and jrec.get("source") == "sam3d-body":
        sk = body_joints.apply_to_skeleton(sk, jrec)
        joints_source = "sam3d-body"
    s3_path = run / "stage-3.json"
    s3 = json.loads(s3_path.read_text()) if s3_path.exists() else {}
    mesh_source = s3.get("mesh_source", "carve")
    shape_views = set(s3.get("shape_views") or vs.views)
    source = None
    src_uv, src_tex = work / "03-source-uv.npz", work / "03-source-texture.png"
    if mesh_source == "sam3d-objects" and src_uv.exists() and src_tex.exists():
        import importlib.util

        if importlib.util.find_spec("scipy") is None:
            raise ModuleNotFoundError("the sam3d-objects body's texture fallback needs scipy (pip install \".[remote]\")")
        from PIL import Image as _Image

        d = np.load(src_uv)
        with _Image.open(src_tex) as im:
            source = texture.SourceColour(d["verts"], d["faces"], d["corner_uv"],
                                          np.asarray(im.convert("RGB"), dtype=np.float64))
    for n in sk.notes:
        rec.note(n)
    budgets = {lod: b["budget"] for lod, b in builds.items()}
    if profile == "best":
        rec.note("profile best: no triangle budget, the high body keeps the stage 3 mesh's full detail, no low LOD")
    if PROFILES[profile].get("not_built"):
        rec.note(f"profile {profile}: meets the specification's triangle, watertight-part and texture limits; not "
                 "built: " + ", ".join(PROFILES[profile]["not_built"]))

    # Decimate the whole body to a ladder of targets in one Blender call, then section each candidate and
    # keep the largest whose sectioned total (cuts and caps included) fits the budget.
    targets, outs, ladder = {}, {}, {}
    for lod, budget in budgets.items():
        if budget is None:
            steps = [UNDECIMATED]
        elif budget == "groups":
            steps = list(AVATAR_TARGETS)
        else:
            steps = [int(budget * frac) for frac in TARGET_FRACTIONS]
        ladder[lod] = []
        for k, n_tri in enumerate(steps):
            key = f"{lod}{k}"
            targets[key] = n_tri
            outs[key] = str((work / f"04-dec-{key}.npz").resolve())
            ladder[lod].append(key)
    # coarse bodies the shoulder covers are shaped from (high LOD covers, low LOD covers; the high ones also for
    # the cut search), and the reference body the cuts are placed on
    for key, n_tri in (("coverhigh", 2700), ("coverlow", 1200), ("cutsearch", cuts.REFERENCE_TRIANGLES)):
        if key != "coverlow" or any(b["cover"] == "low" for b in builds.values()):
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

    # Where to cut: a cut moved along its normal only where the gate poses show a gain beyond noise, found once on
    # the reference body and used for every LOD (see cuts). The held-out poses never choose; the record shows the
    # reference body on both pose sets with the cuts unmoved and moved.
    bone_dirs = pose.bone_directions(sk)
    poses = pose.pose_library(bone_dirs)
    heldout = pose.heldout_poses(bone_dirs)
    gate_poses = {k: v for k, v in poses.items() if k not in UNGATED_POSES}
    ref, cover_ref = np.load(outs["cutsearch"]), np.load(outs["coverhigh"])
    cover_kw = {"owner": COVER_OWNER, "radius": COVER_RADIUS, "outset": COVER_OUTSET, "inside": in_masks}
    if cut_search:
        cut_offsets, cut_record = cuts.choose(ref["verts"], ref["faces"], sk, gate_poses,
                                              (cover_ref["verts"], cover_ref["faces"]), cover_kw, DOME["high"], GAP_GATE)
    else:
        cut_offsets = {}
        cut_record = {"method": "not searched: every cut at its joint (comparison run)", "chosen_fraction": {},
                      "unmoved_worst_gap": cuts.worst_by_joint(ref["verts"], ref["faces"], sk, {}, {"gate": gate_poses},
                                                               (cover_ref["verts"], cover_ref["faces"]), cover_kw,
                                                               DOME["high"])["gate"]}
    ref_sets = {}
    for label, offs in (("unmoved", {}), ("moved", cut_offsets)):
        sets = {"heldout": heldout} if label == "unmoved" else {"gate": gate_poses, "heldout": heldout}
        ref_sets[label] = cuts.worst_by_joint(ref["verts"], ref["faces"], sk, offs, sets,
                                              (cover_ref["verts"], cover_ref["faces"]), cover_kw, DOME["high"])
    ref_sets["unmoved"]["gate"] = cut_record["unmoved_worst_gap"]
    cut_record["reference_worst_gap"] = {label: {k: {j: round(v, 4) for j, v in d.items()} for k, d in sets.items()}
                                         for label, sets in ref_sets.items()}
    moved = {j: f for j, f in cut_record["chosen_fraction"].items() if f}
    rec.note("cuts moved along their normals (fractions of the child bone's radius): "
             + (", ".join(f"{j} {f:+.2f}" for j, f in moved.items()) if moved else "none"))

    # Covers are shaped from a coarse decimation of the body (2.7k triangles for the high settings' covers, 1.2k
    # for the low settings') so they stay cheap; they count in the budgets.
    chosen, lod_secs, bare_secs, lod_info, cover_info, picks = {}, {}, {}, {}, {}, {}
    for lod, build in builds.items():
        cover_src = np.load(outs["cover" + build["cover"]])
        rings, ring_points = DOME[build["dome"]]
        cands = []
        for key in ladder[lod]:
            d = np.load(outs[key])
            secs, rig = split.build_sections(d["verts"], d["faces"], sk, cap="dome", rings=rings,
                                             ring_points=ring_points, cut_offsets=cut_offsets)
            covered, cinfo = covers.add_covers(secs, cover_src["verts"], cover_src["faces"], owner=COVER_OWNER,
                                               radius=COVER_RADIUS, outset=COVER_OUTSET, inside=in_masks)
            cands.append({"key": key, "covered": covered, "bare": secs, "covers": cinfo,
                          "total": sum(len(s.faces) for s in covered.values()), "decimated": len(d["faces"])})
        pick = _pick(cands, build, profile)
        picks[lod] = {n: cands[i]["key"] for n, i in pick.items()}
        lod_secs[lod] = {n: cands[pick[n]]["covered"][n] for n in SECTIONS}
        bare_secs[lod] = {n: cands[pick[n]]["bare"][n] for n in SECTIONS}
        owner = cands[pick[COVER_OWNER]]
        cover_info[lod] = owner["covers"]
        groups = _section_groups(build, profile)
        chosen[lod] = (cands[pick[SECTIONS[0]]]["key"] if len(groups) == 1
                       else {g: cands[pick[secs[0]]]["key"] for g, secs in groups.items()})
        decimated = ({g: cands[pick[secs[0]]]["decimated"] for g, secs in groups.items()} if len(groups) > 1
                     else cands[pick[SECTIONS[0]]]["decimated"])
        lod_info[lod] = {"body_triangles_after_decimation": decimated,
                         "triangles": sum(len(s.faces) for s in lod_secs[lod].values()),
                         "cap_triangles": sum(s.cap_triangles for s in bare_secs[lod].values()),
                         "cover_triangles": sum(c["triangles"] for c in cover_info[lod].values()),
                         "dome_rings": rings, "dome_ring_points": ring_points, "atlas_size": build["atlas"],
                         "budget": build["budget"] if build["budget"] != "groups" else "avatar part groups"}
        if len(groups) > 1:
            lod_info[lod]["triangles_by_group"] = {g: sum(len(lod_secs[lod][n].faces) for n in secs)
                                                   for g, secs in groups.items()}

    # Pose test. Gated: the gap that stays visible with the covers on (both LODs). Reported alongside: the
    # visible gap without covers, the purely geometric gap without covers, and flat caps.
    gaps = {lod: pose.joint_gaps(lod_secs[lod], poses, visible=True) for lod in lod_secs}
    gaps_heldout = {lod: pose.joint_gaps(lod_secs[lod], heldout, visible=True) for lod in lod_secs}
    gaps_nocover_visible = pose.joint_gaps(bare_secs["high"], poses, visible=True)
    gaps_nocover_geometric = pose.joint_gaps(bare_secs["high"], poses)
    flat_by_key = {}
    for key in sorted(set(picks["high"].values())):
        d_hi = np.load(outs[key])
        flat_by_key[key], _ = split.build_sections(d_hi["verts"], d_hi["faces"], sk, cap="flat",
                                                   cut_offsets=cut_offsets)
    flat_secs = {n: flat_by_key[picks["high"][n]][n] for n in SECTIONS}
    gaps_flat = pose.joint_gaps(flat_secs, poses)

    # UV atlases (Blender smart project, all sections of a LOD packed together), then colour projection.
    body = run / "04-body"
    (body / "pose-renders").mkdir(parents=True, exist_ok=True)
    (body / "rest-renders").mkdir(parents=True, exist_ok=True)
    # Comparison atlases are written only for a hybrid body; one left by an earlier run of another mesh source
    # would be scored as a variant of this body.
    for stale in body.glob("atlas-high-*.png"):
        stale.unlink()
    for lod in ("high", "low"):            # a LOD this profile does not build must not survive from a past run
        if lod not in builds:
            for stale in (body / f"{lod}.glb", body / f"atlas-{lod}.png"):
                if stale.exists():
                    stale.unlink()
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
    for lod in builds:
        u = np.load(uv_paths[lod])
        uvs = {k.split("__")[0]: u[k] for k in u.files}
        _save_lod(lod_paths[lod], lod_secs[lod], uvs)
        if colour_views:
            size = builds[lod]["atlas"]
            atlas, stats, zbuf = texture.bake(lod_secs[lod], uvs, colour_views, size, zbuf, source=source)
            atlas_paths[lod] = body / f"atlas-{lod}.png"
            Image.fromarray(atlas).save(atlas_paths[lod])
            tex_stats[lod] = stats
            if source is not None and lod == "high":
                # comparison atlases, scored by `score` as variants: no source fallback (its effect), and the
                # side views down-weighted (they did not shape this mesh, so their colours land on oblique
                # surfaces at the wrong depth)
                plain, pstats, _ = texture.bake(lod_secs[lod], uvs, colour_views, size, zbuf)
                Image.fromarray(plain).save(body / f"atlas-{lod}-no-source-fallback.png")
                tex_stats[lod + "_no_source_fallback"] = pstats
                trust = {v: SIDE_TRUST for v in colour_views if v in ("left", "right") and v not in shape_views}
                if trust:
                    tr, tstats, _ = texture.bake(lod_secs[lod], uvs, colour_views, size, zbuf, source=source,
                                                 trust=trust)
                    Image.fromarray(tr).save(body / f"atlas-{lod}-side-trust.png")
                    tex_stats[lod + "_side_trust"] = {**tstats, "trust": trust}

    # Blender: textured GLBs, rest renders from the contract cameras, pose renders.
    rest_cams = [vs.views[v].camera() for v in VIEWS if v in vs.views]
    pivots = {n: s.pivot for n, s in lod_secs["high"].items()}
    pose_jobs = [{"name": name, "matrices": {n: m.tolist() for n, m in pose.world_matrices(loc, pivots).items()}}
                 for name, loc in poses.items()]
    res = bl.run_job({"tasks": [{
        "task": "body", "lods": {k: str(v.resolve()) for k, v in lod_paths.items()},
        "glb": {lod: str((body / f"{lod}.glb").resolve()) for lod in builds},
        "atlas": {k: str(v.resolve()) for k, v in atlas_paths.items()},
        "colors": section_colors(), "poses": pose_jobs, "pose_cameras": pose_cameras(height),
        "pose_dir": str((body / "pose-renders").resolve()), "pose_lod": "high",
        "rest_cameras": rest_cams, "rest_dir": str((body / "rest-renders").resolve())}]},
        work, "stage4-body", timeout=timeout)
    bres = res["results"][0]

    # Checks.
    hi, lo = lod_secs["high"], lod_secs.get("low")
    nonempty = sum(1 for n in SECTIONS if all(len(ss[n].faces) for ss in lod_secs.values()))
    rec.add(check("sections_non_empty", nonempty, 15))
    for lod, budget in budgets.items():
        n_tri = lod_info[lod]["triangles"]
        if budget is None:
            rec.add(check(f"triangles_{lod}", n_tri, None, gate=False,
                          note=f"profile {profile}: no triangle budget"))
        elif budget == "groups":
            spec = PROFILES[profile]
            rec.add(check("triangles_total", n_tri, spec["triangles_total"],
                          passed=n_tri <= spec["triangles_total"], rule="<="))
            for group, (secs, limit) in spec["groups"].items():
                k = lod_info[lod]["triangles_by_group"][group]
                rec.add(check(f"triangles_{group}", k, limit, passed=k <= limit, rule="<=", sections=list(secs)))
        else:
            rec.add(check(f"triangles_{lod}", n_tri, budget, passed=n_tri <= budget, rule="<="))
    # Every part of every LOD closed and manifold after welding, under every profile (the avatar specification
    # requires it; an open part is a defect anywhere).
    from .score import mesh_sanity

    not_closed = [f"{lod}:{n}" for lod, ss in lod_secs.items() for n in SECTIONS
                  if not mesh_sanity(ss[n].verts, ss[n].faces, 1e-6 * height)["watertight"]]
    rec.add(check("watertight_parts", len(not_closed), 0, passed=not not_closed, rule="==",
                  note="parts with a boundary or non-manifold edge after welding: " + ", ".join(not_closed)
                  if not_closed else "every part closed and manifold"))
    if PROFILES[profile].get("texture_max_px"):
        biggest = max(b["atlas"] for b in builds.values())
        rec.add(check("texture_max_px", biggest, PROFILES[profile]["texture_max_px"],
                      passed=biggest <= PROFILES[profile]["texture_max_px"], rule="<="))
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
                  rule=">=", note="pivot inside its section and within one cut radius of the mask joint",
                  worst_offset_over_radius=round(worst_off, 3)))
    gated = {lod: max(v for p, row in g.items() if p not in UNGATED_POSES for v in row.values())
             for lod, g in gaps.items()}
    for lod in builds:
        rec.add(check(f"max_joint_gap_{lod}", round(gated[lod], 4), GAP_GATE, passed=gated[lod] <= GAP_GATE,
                      rule="<=", poses=sorted(p for p in poses if p not in UNGATED_POSES)))
        held = max(v for row in gaps_heldout[lod].values() for v in row.values())
        rec.add(check(f"max_joint_gap_{lod}_heldout", round(held, 4), GAP_GATE, passed=held <= GAP_GATE, gate=False,
                      rule="<=", poses=sorted(heldout),
                      note="poses nothing is chosen on (the cut search never sees them); reported, not gated"))
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
        unseen = v not in shape_views and not derived
        rec.add(check(f"rest_iou_with_covers_{v}", rest_iou[v], IOU_GATE, gate=not derived,
                      **({"note": f"shape not built from this view (mesh source {mesh_source})"} if unseen else {})))
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
        "units": "studs", "height": height, "mesh_source": mesh_source, "joints_source": joints_source,
        "shape_views": sorted(shape_views),
        "frame": "Blender: +Z up, feet at z=0, faces -Y, x/y centred on the figure's bounding box; "
                 "GLBs are exported Y-up (glTF), front along +Z",
        "sections": [{
            "name": n, "parent": PARENT[n], "joint": PIVOT_JOINT[n],
            "pivot": [round(float(c), 4) for c in hi[n].pivot],
            "triangles_high": int(len(hi[n].faces)), "triangles_low": int(len(lo[n].faces)) if lo else None,
            "cap_triangles_high": hi[n].cap_triangles, "cap_triangles_low": lo[n].cap_triangles if lo else None,
            "cap_points_outside_high": hi[n].cap_points_outside,
            "bounds_min": [round(float(c), 4) for c in hi[n].verts.min(axis=0)] if len(hi[n].verts) else None,
            "bounds_max": [round(float(c), 4) for c in hi[n].verts.max(axis=0)] if len(hi[n].verts) else None,
            "pivot_check": pivot_rows[n],
        } for n in SECTIONS],
        "joints": sk.as_json(),
        "ends": {k: [round(float(c), 4) for c in v] for k, v in sk.ends.items()},
        "joint_evidence": sk.evidence,
        "joint_method": (body_joints.__doc__ if joints_source == "sam3d-body" else joints.__doc__).strip(),
        "silhouette_joints": silhouette_joints,
        "split_method": split.__doc__.strip(),
        "profile": profile,
        "budgets": _budgets_record(profile),
        "code": rec.data["code"],
        "lods": lod_info,
        "cuts": cut_record,
        "pose_test": {"method": pose.__doc__.strip(), "visible_gap_method": pose.joint_gaps.__doc__.strip(),
                      "gate_studs": GAP_GATE, "ungated_poses": sorted(UNGATED_POSES),
                      "gaps_high": gaps["high"], "gaps_low": gaps.get("low"),
                      "gaps_high_heldout": gaps_heldout["high"], "gaps_low_heldout": gaps_heldout.get("low"),
                      "gaps_high_visible_no_covers": gaps_nocover_visible,
                      "gaps_high_geometric_no_covers": gaps_nocover_geometric, "gaps_high_flat_caps": gaps_flat},
        "covers": {"method": covers.__doc__.strip(), "owner": COVER_OWNER, "radius_cut_radii": COVER_RADIUS,
                   "outset_studs": COVER_OUTSET, "pieces_high": cover_info["high"],
                   "pieces_low": cover_info.get("low"),
                   "contract_note": "covers are merged into their owner section so each GLB keeps the 15 R15 "
                                    "parts; they are listed under sections.json \"covers\""},
        "texture": {"method": texture.__doc__.strip(), "atlas_sizes": {lod: b["atlas"] for lod, b in builds.items()},
                    "atlases": {k: str(v.relative_to(run)) for k, v in atlas_paths.items()},
                    "colour_views": sorted(colour_views), "bake": tex_stats, "rest_metrics": tex_metrics,
                    "gates": {"mean_abs_error_max": TEX_ERROR_GATE, "ssim_min": TEX_SSIM_GATE}},
        "rest_iou_with_covers": rest_iou,
        "joint_error_synthetic": joint_err,
        "checks": rec.data["checks"],
    }
    write_atomic(body / "sections.json", json.dumps(sections_json, indent=2))
    rec.data["outputs"] = ([f"04-body/{lod}.glb" for lod in builds] + ["04-body/sections.json"]
                           + [str(v.relative_to(run)) for v in atlas_paths.values()]
                           + ([str(sheet.relative_to(run))] if sheet else [])
                           + [str(Path(p).relative_to(run.resolve())) for p in bres["pose_renders"]]
                           + [str(Path(p).relative_to(run.resolve())) for p in bres["rest_textured"].values()])
    rec.data["mesh_source"] = mesh_source
    rec.data["joints_source"] = joints_source
    rec.data["profile"] = {"name": profile, "lods": sorted(builds), "budgets": _budgets_record(profile)}
    rec.data["body"] = {"lods": lod_info, "decimation": dec["results"][0], "chosen_targets": chosen,
                        "sections_from": picks, "cuts": cut_record,
                        "max_gap_by_pose_high": {p: max(r.values()) for p, r in gaps["high"].items()},
                        "max_gap_by_pose_high_heldout": {p: max(r.values()) for p, r in gaps_heldout["high"].items()},
                        "max_gap_by_pose_low": ({p: max(r.values()) for p, r in gaps["low"].items()}
                                                if "low" in gaps else None),
                        "max_gap_by_pose_high_flat": {p: max(r.values()) for p, r in gaps_flat.items()},
                        "max_gap_by_pose_high_visible_no_covers":
                            {p: max(r.values()) for p, r in gaps_nocover_visible.items()},
                        "max_gap_by_pose_high_geometric_no_covers":
                            {p: max(r.values()) for p, r in gaps_nocover_geometric.items()},
                        "covers": cover_info, "texture": {"bake": tex_stats, "rest_metrics": tex_metrics},
                        "rest_iou_with_covers": rest_iou,
                        "joint_error_synthetic": joint_err, "joint_evidence": sk.evidence}
    return rec.finish()
