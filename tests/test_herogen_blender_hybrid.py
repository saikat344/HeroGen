"""The hybrid body (herogen/blender): the numpy rasteriser, orienting a SAM 3D Objects source mesh from the front
view, SAM 3D Body joint mapping, stage 3 from replayed remote results and the fallback to the local sources, the
source-texture fallback and stage 4 on the hybrid mesh.

Remote models are never called: replay entries are written by the tests themselves from synthetic data. Tests
that need Blender skip cleanly without it; tests that need scipy (the hybrid mesh path) skip without it.
Nothing generated here is written inside the repository.
"""
from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False

from herogen.blender import bl, body_joints, carve, glb, joints, raster, synth, texture, views
from herogen.blender.split import Section
from herogen.blender.remote import client as rclient

if HAVE_SCIPY:
    from herogen.blender import sam3d_mesh

HAVE_BLENDER = bl.find_blender() is not None


def png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def rotation(yaw: float, pitch: float) -> np.ndarray:
    a, b = np.radians(yaw), np.radians(pitch)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return rz @ rx


class HybridBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="herogen-hybrid-test-"))
        cls.rundir = cls.tmp / "run"
        cls.truth = synth.write_run(cls.rundir, synth.humanoid(), size=384, ppu=60.0)
        cls.vs = views.load_views(cls.rundir, 5.5)
        cls.verts, cls.faces, _ = carve.build_mesh(cls.vs, 5.5, 96)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def source_glb(cls, path: Path, yaw_deg: float = 180.0, pitch_deg: float = 6.0) -> None:
        """The synthetic body as an image-to-3D model might return it: unit size, turned and tilted, with a
        texture that is red on the figure's front half and blue on its back half."""
        v = cls.verts
        front = v[:, 1] < np.median(v[:, 1])
        uv = np.where(front[:, None], [0.25, 0.5], [0.75, 0.5])
        tex = np.zeros((16, 16, 3), np.uint8)
        tex[:, :8] = (220, 30, 30)
        tex[:, 8:] = (30, 30, 220)
        w = (v - v.mean(axis=0)) @ rotation(yaw_deg, pitch_deg).T / 5.5
        glb.write_glb(path, [("source", w, cls.faces, uv)], texture_png=png(tex))

    @classmethod
    def replay_entries(cls, run: Path, cache: Path, body: dict | None = None, source_yaw: float = 0.0) -> None:
        """Replay entries, made from synthetic data, for the two remote calls stage 3 makes on `run`."""
        from herogen.blender.stage3 import BODY_BOX_SOURCE, body_box

        img, msk = run / "views" / "front.png", run / "masks" / "front.png"
        s_obj = rclient.objects_summary(img, msk)
        d = cache / rclient.request_key(s_obj)
        d.mkdir(parents=True)
        cls.source_glb(d / "mesh.glb", yaw_deg=source_yaw, pitch_deg=0.0)
        (d / "entry.json").write_text(json.dumps({"request": s_obj, "cost_usd_estimate": 0.2, "run": {}}))
        vs = views.load_views(run, 5.5)
        s_body = rclient.body_summary(img, body_box(vs), BODY_BOX_SOURCE)
        d = cache / rclient.request_key(s_body)
        d.mkdir(parents=True)
        (d / "joints.json").write_text(json.dumps(body or synthetic_body_result(cls.truth, vs.views["front"])))
        (d / "entry.json").write_text(json.dumps({"request": s_body, "cost_usd_estimate": 0.02, "run": {}}))


def synthetic_body_result(truth: dict, front, depth: float = 1000.0) -> dict:
    """A SAM 3D Body `joints.json` an ideal model would give for the synthetic figure: a camera far in front
    (nearly orthographic) whose image frame is the front view's."""
    zc = 2.75
    f_len = front.pps * depth
    c = np.array([front.cmid, front.vref - zc * front.pps])

    def cam(p):
        x, y, z = p
        return np.array([x, -(z - zc), depth + y])

    def uv(q):
        return (f_len * q[:2] / q[2] + c).tolist()

    j = {k: np.asarray(v, float) for k, v in truth["joints"].items()}
    kp = {}
    for name, key in body_joints.LIMBS.items():
        q = cam(j[name])
        kp[key] = {"xyz": q.tolist(), "uv": uv(q)}
    kp["neck"] = {"xyz": cam(j["Neck"]).tolist(), "uv": uv(cam(j["Neck"]))}
    skel = np.zeros((127, 3))
    skel[:, 2] = depth
    for name, idx in body_joints.TRUNK.items():
        skel[idx] = cam(j[name])
    return {"focal_length": f_len, "principal_point": c.tolist(), "mhr70": kp,
            "mhr_skeleton_joint_coords": skel.tolist(), "mesh_top_uv": uv(cam(j["HeadTop"]))}


class TestRaster(HybridBase):
    def test_a_mesh_outside_the_view_renders_empty(self):
        fr = raster.render_view(self.vs.views["front"], self.verts + np.array([500.0, 0.0, 500.0]), self.faces)
        self.assertFalse(fr.mask.any())

    def test_raster_matches_the_mask_and_depth(self):
        f = self.vs.views["front"]
        fr = raster.render_view(f, self.verts, self.faces)
        union = np.logical_or(fr.mask, f.mask).sum()
        self.assertGreater(np.logical_and(fr.mask, f.mask).sum() / union, 0.93)
        ok = fr.mask             # the nearest depth is the smallest Y on each ray
        self.assertTrue(np.all(fr.depth[ok] >= self.verts[:, 1].min() - 1e-6))
        self.assertTrue(np.all(np.isinf(fr.depth[~ok])))


@unittest.skipUnless(HAVE_SCIPY, "scipy is required for the hybrid mesh path")
class TestSourceMesh(HybridBase):
    def test_load_source_welds_through_the_shared_reader(self):
        p = self.tmp / "load.glb"
        self.source_glb(p)
        src = sam3d_mesh.load_source(p)
        self.assertEqual(len(src.faces), len(self.faces))
        self.assertEqual(len(src.verts), len(np.unique(np.round(self.verts, 6), axis=0)))
        self.assertEqual(src.corner_uv.shape, (len(self.faces), 3, 2))
        self.assertEqual(src.texture.shape, (16, 16, 3))

    def test_orient_recovers_facing_and_frame_from_the_front_view(self):
        p = self.tmp / "src.glb"
        self.source_glb(p)
        src = sam3d_mesh.load_source(p)
        f = self.vs.views["front"]
        img = np.zeros(f.mask.shape + (3,))
        img[f.mask] = (220, 30, 30)            # the front view is red: only the right facing matches
        v, info = sam3d_mesh.orient(src, f, img, 5.5)
        self.assertEqual(info["facing_deg"], 180.0)
        self.assertAlmostEqual(float(v[:, 2].min()), 0.0, places=6)
        self.assertAlmostEqual(float(v[:, 2].max() - v[:, 2].min()), 5.5, places=6)
        self.assertGreater(info["front_iou_fitted"], 0.90)
        self.assertGreaterEqual(info["front_iou_half_res"]["fitted"], info["front_iou_half_res"]["facing_only"])
        self.assertGreater(info["front_iou_final_numpy"], 0.95)
        # front surface faces -Y: the red (front) texels sit at smaller Y than the blue ones
        u = src.corner_uv[:, 0, 0]
        cy = v[src.faces].mean(axis=1)[:, 1]
        self.assertLess(cy[u < 0.5].mean(), cy[u > 0.5].mean())

    def test_conform_pulls_a_shrunk_outline_onto_the_mask(self):
        f = self.vs.views["front"]
        v = self.verts.copy()
        v[:, 0] *= 0.93
        before = sam3d_mesh.iou(raster.render_view(f, v, self.faces).mask, f.mask)
        v2, info = sam3d_mesh.conform(v, self.faces, f)
        after = sam3d_mesh.iou(raster.render_view(f, v2, self.faces).mask, f.mask)
        self.assertGreater(after, before + 0.02)
        self.assertTrue(np.allclose(v2[:, 1], v[:, 1]))          # depth is untouched
        self.assertEqual(len(info["iou_by_iteration"]), info["iterations"] + 1)

    def test_remesh_is_closed_and_manifold(self):
        v, f, info = sam3d_mesh.remesh(self.verts, self.faces, 5.5, nz=96)
        edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        self.assertTrue((counts == 2).all())
        self.assertAlmostEqual(float(v[:, 2].max() - v[:, 2].min()), 5.5, places=6)
        self.assertEqual(info["triangles"], len(f))


class TestBodyJoints(HybridBase):
    def test_mapping_recovers_the_synthetic_joints(self):
        body = synthetic_body_result(self.truth, self.vs.views["front"])
        for method in body_joints.Y_METHODS:
            pts, ev = body_joints.from_body(body, self.vs.views["front"], self.verts, self.faces, 5.5, method)
            err_xz = {k: float(np.linalg.norm((pts[k] - np.asarray(self.truth["joints"][k]))[[0, 2]]))
                      for k in body_joints.JOINT_NAMES}
            self.assertLess(max(err_xz.values()), 0.03, (method, err_xz))
            err_y = {k: abs(pts[k][1] - self.truth["joints"][k][1]) for k in body_joints.JOINT_NAMES
                     if k != "HeadTop"}
            self.assertLess(max(err_y.values()), 0.15, (method, err_y))
            self.assertEqual(set(ev), set(body_joints.JOINT_NAMES))
        with self.assertRaises(ValueError):
            body_joints.from_body(body, self.vs.views["front"], self.verts, self.faces, 5.5, "nope")

    def test_skeleton_override_and_cross_check(self):
        sk = joints.estimate(self.vs, 5.5)
        sil = sk.as_json()
        rec = {"source": "sam3d-body", "joints": {k: list(v) for k, v in self.truth["joints"].items()},
               "evidence": {k: "truth" for k in self.truth["joints"]}}
        sk2 = body_joints.apply_to_skeleton(sk, rec)
        self.assertTrue(np.allclose(sk2.ends["HipCentre"], (np.asarray(rec["joints"]["LeftHip"])
                                                            + np.asarray(rec["joints"]["RightHip"])) / 2))
        d = body_joints.cross_check(rec["joints"], sil)
        self.assertEqual(len(d), 16)
        self.assertLess(max(d.values()), body_joints.FLAG_STUDS)
        with self.assertRaises(KeyError):
            body_joints.apply_to_skeleton(joints.estimate(self.vs, 5.5), {"joints": {"Neck": [0, 0, 4]}})


def cell_box():
    """A closed box (centre (0, 0, 1.5), half size 0.5) with each side in its own atlas cell of a 3 x 2 grid.
    Returns (verts, faces, corner_uv [M, 3, 2], side of each face as (axis, sign))."""
    verts, faces, corner, side = [], [], [], []
    k = 0
    for axis in range(3):
        for sign in (-1.0, 1.0):
            a, b = [i for i in range(3) if i != axis]
            quad = []
            for (p, q) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                v = np.zeros(3)
                v[axis], v[a], v[b] = sign * 0.5, p * 0.5, q * 0.5
                quad.append(v + [0, 0, 1.5])
            n = np.cross(quad[1] - quad[0], quad[2] - quad[0])
            order = [0, 1, 2, 3] if n[axis] * sign > 0 else [0, 3, 2, 1]
            base = len(verts)
            verts += [quad[i] for i in order]
            cu, cv = (k % 3) / 3, (k // 3) / 2
            cuv = [np.array([cu + 0.02 + (0.29 if i in (1, 2) else 0), cv + 0.02 + (0.46 if i in (2, 3) else 0)])
                   for i in order]
            for tri in ((0, 1, 2), (0, 2, 3)):
                faces.append([base + t for t in tri])
                corner.append([cuv[t] for t in tri])
                side.append((axis, sign))
            k += 1
    return np.array(verts), np.array(faces), np.array(corner), side


@unittest.skipUnless(HAVE_SCIPY, "scipy is required for the texture fallback")
class TestTextureFallback(unittest.TestCase):
    def test_source_colour_lookup(self):
        v, f, corner, side = cell_box()
        tex = np.zeros((8, 8, 3))
        tex[:, :4] = (255, 0, 0)
        tex[:, 4:] = (0, 0, 255)
        corner = np.where(np.array([s == (0, -1.0) for s in side])[:, None, None], [0.2, 0.5], [0.8, 0.5])
        corner = np.broadcast_to(corner, (len(f), 3, 2)).copy()
        sc = texture.SourceColour(v, f, corner, tex)
        col, dist = sc(np.array([[-0.6, 0.0, 1.5], [0.6, 0.0, 1.5]]))
        self.assertGreater(col[0, 0], 200)          # nearest the -X side: red
        self.assertGreater(col[1, 2], 200)          # nearest the +X side: blue
        np.testing.assert_allclose(dist, [0.1, 0.1], atol=0.05)

    def test_unseen_surfaces_take_the_source_texture(self):
        v, f, corner, side = cell_box()
        sec = {"Box": Section("Box", v, f, np.zeros(3), [], 0)}
        front = views.View("front", np.ones((64, 64), bool), 20.0, 32.0, 64.0)
        red = np.zeros((64, 64, 3))
        red[:] = (230, 20, 20)
        vw = {"front": (front, red)}
        blue = np.zeros((8, 8, 3))
        blue[:] = (20, 20, 230)
        src = texture.SourceColour(v, f, corner, blue)
        back_cell = [k for k, s in enumerate(side) if s == (1, 1.0)][0] // 2   # the +Y side faces away
        cu, cv = (back_cell % 3) / 3 + 0.16, (back_cell // 3) / 2 + 0.25
        size = 64
        at = (int((1 - cv) * size), int(cu * size))       # atlas row 0 is v = 1
        plain, stats, zbuf = texture.bake(sec, {"Box": corner}, vw, size)
        mixed, mstats, _ = texture.bake(sec, {"Box": corner}, vw, size, zbuf, source=src)
        self.assertGreater(plain[at][0], 150)               # no source: the front view's colour smears round
        self.assertGreater(mixed[at][2], 150)               # with a source: the source's own colour
        self.assertGreater(mstats["texels_source_only"], 0.0)
        self.assertIn("texels_best_facing_below_fallback", stats)
        front_cell = [k for k, s in enumerate(side) if s == (1, -1.0)][0] // 2
        fu, fv = (front_cell % 3) / 3 + 0.16, (front_cell // 3) / 2 + 0.25
        fat = (int((1 - fv) * size), int(fu * size))
        self.assertGreater(mixed[fat][0], 150)              # the face the view sees keeps the view's colour
        trusted, _, _ = texture.bake(sec, {"Box": corner}, vw, size, zbuf, source=src, trust={"front": 0.3})
        self.assertGreater(trusted[fat][0], 150)            # one view: its weight normalises out


@unittest.skipUnless(HAVE_SCIPY, "scipy is required for the hybrid mesh path")
class TestStage3SourcesWithoutBlender(HybridBase):
    """The two remote sources of stage 3 from replay, up to (not including) the Blender silhouette renders."""

    def test_objects_mesh_and_body_joints_from_replay(self):
        from herogen.blender import stage3
        from herogen.blender.record import StageRecord

        run = self.tmp / "sources"
        synth.write_run(run, synth.humanoid(), size=384, ppu=60.0)
        cache = self.tmp / "sources-cache"
        self.replay_entries(run, cache)
        vs = views.load_views(run, 5.5)
        remote = rclient.RemoteClient(cache)
        rec = StageRecord(run, 3, "base mesh")
        verts, faces, info, res = stage3._objects_mesh(run, vs, 5.5, remote, rec)
        self.assertTrue(res.path("mesh.glb").exists())
        self.assertFalse((run / "03-source.glb").exists())          # kept by run_stage3 once the source is used
        self.assertTrue((stage3.work_dir(run) / "03-source-uv.npz").exists())
        self.assertEqual(info["facing_deg"], 0.0)
        self.assertTrue(info["remote"]["cached"])
        self.assertEqual(rec.data["models"], ["sam-3d-objects"])
        self.assertEqual(rec.data["cost_usd"], 0.0)              # replayed: no new spend
        self.assertIn("replayed", rec.data["cost_basis"])
        sk = joints.estimate(vs, 5.5)
        j = stage3._body_joints(run, vs, 5.5, verts, faces, remote, rec, sk)
        self.assertEqual(j["source"], "sam3d-body")
        self.assertEqual(j["joint_sources"]["LeftShoulder"], "silhouette (X/Z), mesh (Y)")
        self.assertEqual(j["joint_sources"]["LeftElbow"], "sam3d-body")
        self.assertIn("silhouette", j["sources"])
        self.assertEqual(set(j["distance_to_silhouette"]), set(sk.joints))
        on_disk = json.loads((run / "03-joints.json").read_text())
        self.assertEqual(on_disk["joints"].keys(), j["joints"].keys())
        # the person box is the front mask's, recorded with its source; no detector runs by default
        self.assertEqual(j["remote"]["person_box"], stage3.body_box(vs))
        self.assertEqual(j["remote"]["box_source"], stage3.BODY_BOX_SOURCE)
        self.assertIsNone(j["remote"]["detector"])
        self.assertEqual(j["remote"]["code"]["dinov3"], rclient.runner_fingerprint("sam3d-body")["dinov3"])
        # with the optional detector the request (and so the cache entry) differs, and the record says so
        img = run / "views" / "front.png"
        s_det = rclient.body_summary(img, stage3.body_box(vs), stage3.BODY_BOX_SOURCE, "vitdet")
        default = cache / rclient.request_key(rclient.body_summary(img, stage3.body_box(vs), stage3.BODY_BOX_SOURCE))
        d = cache / rclient.request_key(s_det)
        d.mkdir()
        shutil.copy(default / "joints.json", d / "joints.json")
        (d / "entry.json").write_text(json.dumps({"request": s_det, "cost_usd_estimate": 0.02,
                                                  "run": {"runs": {"job": {"box": "detector (vitdet)"}}}}))
        jd = stage3._body_joints(run, vs, 5.5, verts, faces, remote, rec, sk, body_detector="vitdet")
        self.assertEqual((jd["remote"]["detector"], jd["remote"]["box_source"]), ("vitdet", "detector (vitdet)"))
        self.assertEqual(jd["remote"]["key"], d.name)

    def corrupt_run(self, name: str, glb_bytes: bytes = b"not a glb") -> tuple[Path, Path]:
        """A run whose cached SAM 3D Objects result is not a readable GLB."""
        run = self.tmp / name
        synth.write_run(run, synth.humanoid(), size=384, ppu=60.0)
        cache = self.tmp / f"{name}-cache"
        self.replay_entries(run, cache)
        key = rclient.request_key(rclient.objects_summary(run / "views" / "front.png", run / "masks" / "front.png"))
        (cache / key / "mesh.glb").write_bytes(glb_bytes)
        return run, cache

    def test_a_corrupt_cached_glb_is_a_remote_error_that_writes_nothing(self):
        from herogen.blender import stage3
        from herogen.blender.record import StageRecord

        run, cache = self.corrupt_run("corrupt")
        vs = views.load_views(run, 5.5)
        rec = StageRecord(run, 3, "base mesh")
        with self.assertRaises(rclient.RemoteError) as cm:
            stage3._remote_source("sam3d-objects", run, stage3._objects_mesh, run, vs, 5.5,
                                  rclient.RemoteClient(cache), rec)
        self.assertIn("sam3d-objects: GlbError", str(cm.exception))
        self.assertNotIn(str(self.tmp), str(cm.exception))         # no local paths in what reaches a record
        self.assertFalse((run / "03-source.glb").exists())
        self.assertFalse((stage3.work_dir(run) / "03-source-uv.npz").exists())
        self.assertEqual(rec.data["models"], ["sam-3d-objects"])  # the call is recorded even though it failed

    def test_a_mesh_that_misses_the_front_view_is_a_remote_error(self):
        from herogen.blender import stage3
        from herogen.blender.record import StageRecord

        run = self.tmp / "miss"
        synth.write_run(run, synth.humanoid(), size=384, ppu=60.0)
        cache = self.tmp / "miss-cache"
        self.replay_entries(run, cache)
        vs = views.load_views(run, 5.5)
        away = self.verts + np.array([500.0, 0.0, 500.0])      # nothing of it lands in the front view: empty mask
        with self.assertRaises(rclient.RemoteError) as cm:
            stage3._remote_source("sam3d-body", run, stage3._body_joints, run, vs, 5.5, away, self.faces,
                                  rclient.RemoteClient(cache), StageRecord(run, 3, "base mesh"),
                                  joints.estimate(vs, 5.5))
        self.assertIn("covers no pixel of the front view", str(cm.exception))
        self.assertFalse((run / "03-joints.json").exists())

    def test_a_failed_source_without_fallback_leaves_no_ok_record(self):
        from herogen.blender.stage3 import run_stage3

        run, cache = self.corrupt_run("stale")
        (run / "stage-3.json").write_text(json.dumps({"stage": 3, "ok": True}))      # an earlier, passed run
        (run / "03-source.glb").write_bytes(b"earlier source")
        with self.assertRaises(rclient.RemoteError):
            run_stage3(run, nz=96, mesh_source="sam3d-objects", joints_source="sam3d-body",
                       remote=rclient.RemoteClient(cache), fallback=False, timeout=300)
        self.assertFalse((run / "stage-3.json").exists())        # stage 4 cannot pick up the earlier record
        self.assertFalse((run / "03-source.glb").exists())
        self.assertEqual([p.name for p in run.glob("*.part")], [])


@unittest.skipUnless(HAVE_SCIPY and HAVE_BLENDER, "Blender and scipy are required")
class TestStage3HybridBlender(HybridBase):
    def test_stage3_from_replay_then_fallback_when_the_cache_is_empty(self):
        from herogen.blender.stage3 import run_stage3

        run = self.tmp / "e2e"
        synth.write_run(run, synth.humanoid(), size=384, ppu=60.0)
        cache = self.tmp / "e2e-cache"
        self.replay_entries(run, cache)
        rec = run_stage3(run, nz=96, mesh_source="sam3d-objects", joints_source="sam3d-body",
                         remote=rclient.RemoteClient(cache), fallback=False, timeout=300)
        self.assertEqual(rec["mesh_source"], "sam3d-objects")
        self.assertEqual(rec["joints_source"], "sam3d-body")
        self.assertEqual(rec["shape_views"], ["front"])
        chk = {c["name"]: c for c in rec["checks"]}
        self.assertTrue(chk["silhouette_iou_front"]["gate"])
        self.assertGreater(chk["silhouette_iou_front"]["value"], 0.90)
        for v in ("back", "left", "right"):     # observed views are gated, with a note that the shape never saw them
            self.assertTrue(chk[f"silhouette_iou_{v}"]["gate"], v)
            self.assertIn("shape not built from this view", chk[f"silhouette_iou_{v}"]["note"])
        # Built from the front view alone, the shape does not match the side masks (the fit picks a small pitch
        # the front view cannot tell from none), so the gated side views fail and so does the stage.
        for v in ("left", "right"):
            self.assertFalse(chk[f"silhouette_iou_{v}"]["pass"], v)
        self.assertEqual(chk["front_view_present"]["value"], 1)
        self.assertEqual(chk["open_edges"]["value"], 0)
        self.assertIn("03-source.glb", rec["outputs"])
        self.assertEqual(json.loads((run / "03-joints.json").read_text())["source"], "sam3d-body")
        self.assertEqual(rec["cost_usd"], 0.0)
        self.assertFalse(rec["ok"])                                   # the side views fail
        # an empty cache: the remote sources fall back to carve and silhouette, and the record says so
        rec2 = run_stage3(run, nz=96, mesh_source="sam3d-objects", joints_source="sam3d-body",
                          remote=rclient.RemoteClient(self.tmp / "empty"), timeout=300)
        self.assertEqual((rec2["mesh_source"], rec2["mesh_source_requested"]), ("carve", "sam3d-objects"))
        self.assertEqual(rec2["joints_source"], "silhouette")
        self.assertIn("fell back", rec2["notes"])
        self.assertFalse((run / "03-source.glb").exists())       # no stale source from the previous run
        self.assertEqual(json.loads((run / "03-joints.json").read_text())["source"], "silhouette")
        with self.assertRaises(rclient.ReplayMiss):
            run_stage3(run, nz=96, mesh_source="sam3d-objects", remote=rclient.RemoteClient(self.tmp / "empty"),
                       fallback=False, timeout=300)

    def test_corrupt_remote_results_fall_back_cleanly(self):
        """A corrupt cached GLB and an unreadable SAM 3D Body result: stage 3 falls back to the carve and the
        silhouette joints, passes, and leaves no source mesh or source texture behind."""
        from herogen.blender import stage3

        run = self.tmp / "corrupt-e2e"
        synth.write_run(run, synth.humanoid(), size=384, ppu=60.0)
        cache = self.tmp / "corrupt-e2e-cache"
        self.replay_entries(run, cache)
        img, msk = run / "views" / "front.png", run / "masks" / "front.png"
        (cache / rclient.request_key(rclient.objects_summary(img, msk)) / "mesh.glb").write_bytes(b"glTF broken")
        vs = views.load_views(run, 5.5)
        key = rclient.request_key(rclient.body_summary(img, stage3.body_box(vs), stage3.BODY_BOX_SOURCE))
        (cache / key / "joints.json").write_text("{ not json")
        rec = stage3.run_stage3(run, nz=96, mesh_source="sam3d-objects", joints_source="sam3d-body",
                                remote=rclient.RemoteClient(cache), timeout=300)
        self.assertTrue(rec["ok"])
        self.assertEqual((rec["mesh_source"], rec["joints_source"]), ("carve", "silhouette"))
        self.assertIn("sam3d-objects failed (sam3d-objects: GlbError", rec["notes"])
        self.assertIn("sam3d-body failed (sam3d-body: JSONDecodeError", rec["notes"])
        self.assertFalse((run / "03-source.glb").exists())
        self.assertFalse((stage3.work_dir(run) / "03-source-texture.png").exists())
        self.assertEqual(json.loads((run / "03-joints.json").read_text())["source"], "silhouette")
        self.assertEqual(json.loads((run / "stage-3.json").read_text())["mesh_source"], "carve")


@unittest.skipUnless(HAVE_SCIPY and HAVE_BLENDER, "Blender and scipy are required")
class TestStage4HybridBlender(HybridBase):
    def test_stage4_cuts_and_textures_the_hybrid_mesh(self):
        from herogen.blender.stage3 import run_stage3
        from herogen.blender.stage4 import run_stage4

        run = self.tmp / "e2e4"
        synth.write_run(run, synth.humanoid(), size=384, ppu=60.0)
        cache = self.tmp / "e2e4-cache"
        self.replay_entries(run, cache)
        rec3 = run_stage3(run, nz=96, mesh_source="sam3d-objects", joints_source="sam3d-body",
                          remote=rclient.RemoteClient(cache), fallback=False, timeout=300)
        self.assertFalse(rec3["ok"])                                  # the side views fail (stage 3 test)
        rec4 = run_stage4(run, timeout=600, profile="roblox-ingame")
        chk = {c["name"]: c for c in rec4["checks"]}
        self.assertEqual((rec4["mesh_source"], rec4["joints_source"]), ("sam3d-objects", "sam3d-body"))
        self.assertEqual(chk["open_edges"]["value"], 0)
        self.assertEqual(chk["sections_non_empty"]["value"], 15)
        self.assertTrue(chk["rest_iou_with_covers_front"]["gate"])
        for v in ("back", "left", "right"):
            self.assertTrue(chk[f"rest_iou_with_covers_{v}"]["gate"], v)      # observed: gated
            self.assertTrue(chk[f"texture_colour_error_{v}"]["gate"], v)      # colour seen: still gated
        self.assertTrue(chk["triangles_high"]["gate"])
        self.assertLessEqual(chk["triangles_high"]["value"], 24000)
        sections = json.loads((run / "04-body" / "sections.json").read_text())
        self.assertEqual(sections["mesh_source"], "sam3d-objects")
        self.assertEqual(sections["joints_source"], "sam3d-body")
        self.assertEqual(sections["shape_views"], ["front"])
        self.assertIn("silhouette_joints", sections)
        for suffix in ("", "-no-source-fallback", "-side-trust"):
            self.assertTrue((run / "04-body" / f"atlas-high{suffix}.png").exists(), suffix)
        bake = sections["texture"]["bake"]
        self.assertIn("texels_source_blend", bake["high"])
        self.assertEqual(set(bake["high_side_trust"]["trust"]), {"left", "right"})
        # The scorer (numpy) measures the stage 4 body as stage 4's Blender renders do.
        from herogen.blender import score

        assert score.main(["--run", str(run)]) == 0
        scores = {b["name"]: b for b in json.loads((run / "05-scores.json").read_text())["bodies"]}
        self.assertEqual(set(scores), {"hybrid", "hybrid-no-source-fallback", "hybrid-side-trust"})
        hyb = scores["hybrid"]
        for v in ("front", "back", "left", "right"):
            self.assertAlmostEqual(hyb["views"][v]["iou"], chk[f"rest_iou_with_covers_{v}"]["value"], delta=0.003)
            self.assertAlmostEqual(hyb["views"][v]["colour_error"], chk[f"texture_colour_error_{v}"]["value"],
                                   delta=1.0)
            self.assertAlmostEqual(hyb["views"][v]["ssim"], chk[f"texture_ssim_{v}"]["value"], delta=0.03)
        self.assertEqual((hyb["views"]["left"]["seen_shape"], hyb["views"]["left"]["seen_colour"]), (False, True))
        # stage 4 again on the carve: the hybrid's comparison atlases must not survive as stale variants
        rec3 = run_stage3(run, nz=96, timeout=300)
        self.assertEqual(rec3["mesh_source"], "carve")
        run_stage4(run, timeout=600, profile="roblox-ingame")
        self.assertEqual(sorted(p.name for p in (run / "04-body").glob("atlas-high*.png")), ["atlas-high.png"])
        self.assertEqual(json.loads((run / "04-body" / "sections.json").read_text())["mesh_source"], "carve")


if __name__ == "__main__":
    unittest.main()
