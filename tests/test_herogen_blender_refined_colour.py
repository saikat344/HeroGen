"""Synthetic provenance, frame and primary-colour checks without external model calls."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from herogen.blender import glb, refined_colour, sam3d_mesh, texture
from herogen.blender.split import Section


def png(colour):
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(stream, format="PNG")
    return stream.getvalue()


@unittest.skipUnless(importlib.util.find_spec("scipy") is not None, "scipy is required for source colour")
class RefinedSourceTests(unittest.TestCase):
    def test_exact_uv_preserves_fine_stripes_between_dense_samples(self):
        verts = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        corner = verts[faces, :2]
        image = np.broadcast_to(((np.arange(256) // 2 % 2) * 255)[None, :, None], (256, 256, 3)).astype(float)
        x = (np.arange(16, 112) + .5) / 256
        points = np.column_stack((x, np.full(len(x), .2), np.full(len(x), .01)))
        expected = image[0, np.arange(16, 112)]
        exact = texture.SourceColour(verts, faces, corner, image, exact=True)
        sampled = texture.SourceColour(verts, faces, corner, image)
        colour, distance = exact(points)
        np.testing.assert_allclose(colour, expected, atol=1e-9)
        np.testing.assert_allclose(distance, .01, atol=1e-12)
        self.assertGreater(np.abs(sampled(points)[0] - expected).mean(), 100)
        self.assertEqual(exact(np.empty((0, 3)))[0].shape, (0, 3))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.work = self.run / "blender-work"
        self.work.mkdir()
        self.verts = np.array([[-.3, -.2, 0], [.4, -.2, 0], [0, .5, 0], [0, 0, 1.0], [.2, .1, .6]])
        self.faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 4, 1]])
        self.uv = np.full((len(self.verts), 2), .5)
        self.source_path = self.run / "03-source.glb"
        glb.write_glb(self.source_path, [("raw", self.verts, self.faces, self.uv)], texture_png=png("green"))
        raw = sam3d_mesh.load_source(self.source_path)
        self.linear = sam3d_mesh.rotation(37, -13, 7).T * 5.5
        target = sam3d_mesh.perspective_to_ortho(raw.verts @ self.linear, 2, .5, 5.5)
        target[:, 0] += .15 * np.sin(target[:, 2])
        target[:, 1] += .12 * target[:, 2] ** 2
        self.target = target + [.13, -.27, .4]
        self.target_faces = raw.faces
        self.save_npz()
        self.write_refined()
        self.sha = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        self.write_record()

    def save_npz(self, verts=None):
        np.savez(self.work / "03-source-uv.npz", verts=self.target if verts is None else verts,
                 faces=self.target_faces, corner_uv=np.full((len(self.target_faces), 3, 2), .5))

    def write_record(self, **changes):
        (self.run / "stage-3f.json").write_text(json.dumps({"status": "complete", "source_sha256": self.sha,
                                                           **changes}))

    def write_refined(self, offset=0):
        # Split every corner and reverse the face order, as a new UV atlas is free to do.
        corners = (self.verts[self.faces[::-1]] + offset).reshape(-1, 3)
        faces = np.arange(len(corners)).reshape(-1, 3)
        uv = np.full((len(corners), 2), .5)
        glb.write_glb(self.run / "03-refined.glb", [("refined", corners, faces, uv)], texture_png=png("red"))

    def load(self):
        return refined_colour.load(self.run, self.work, 5.5)

    def assertRejected(self, reason):
        primary, info = self.load()
        self.assertIsNone(primary)
        self.assertFalse(info["used"])
        self.assertIn(reason, info["reason"])
        return info

    def test_refined_used_with_split_reordered_corners_and_recorded_measurements(self):
        primary, info = self.load()
        self.assertIsNotNone(primary, info)
        self.assertTrue(info["used"])
        self.assertIsNone(info["reason"])
        self.assertEqual(info["stage3f_source_sha256"], self.sha)
        self.assertEqual(info["current_source_sha256"], self.sha)
        self.assertEqual(info["refined_glb_sha256"], hashlib.sha256((self.run / "03-refined.glb").read_bytes()).hexdigest())
        check = info["source_correspondence"]
        self.assertNotIn("frame_fit", info)
        self.assertEqual(check["method"], "piecewise-linear source correspondence")
        self.assertLess(check["geometry_distance_raw"]["max"], check["threshold_raw"])
        self.assertLess(check["placement_distance_studs"]["max"], 1e-10)
        self.assertLess(check["placed_mesh_distance_studs"]["max"], 1e-10)
        colour, _ = primary(self.target[:1])
        np.testing.assert_allclose(colour, [[255, 0, 0]])
        json.dumps(info, allow_nan=False)

    def test_incomplete_status_never_uses_stale_glb(self):
        for status in ("skipped", "running", "error", "cap_reached", None):
            with self.subTest(status=status):
                self.write_record(status=status)
                self.assertRejected("not complete")

    def test_hash_mismatch_never_loads_stale_refined_geometry(self):
        self.source_path.write_bytes(b"replacement source")
        with mock.patch.object(sam3d_mesh, "load_source", side_effect=AssertionError("must not load")):
            info = self.assertRejected("does not match")
        self.assertEqual(info["stage3f_source_sha256"], self.sha)
        self.assertNotEqual(info["current_source_sha256"], self.sha)

    def test_missing_record_source_refined_or_npz_retains_views(self):
        for name, reason in (("stage-3f.json", "no stage-3f"), ("03-source.glb", "no 03-source.glb"),
                             ("03-refined.glb", "no 03-refined.glb"),
                             ("blender-work/03-source-uv.npz", "no 03-source-uv.npz")):
            with self.subTest(file=name):
                path = self.run / name
                content = path.read_bytes()
                path.unlink()
                self.assertRejected(reason)
                path.write_bytes(content)

    def test_affine_fit_cannot_represent_the_nonaffine_fixture(self):
        raw = sam3d_mesh.load_source(self.source_path).verts
        design = np.c_[raw, np.ones(len(raw))]
        affine, _, _, _ = np.linalg.lstsq(design, self.target, rcond=None)
        error = np.linalg.norm(design @ affine - self.target, axis=1)
        self.assertGreater(error.max(), 100 * 5.5e-6)
        self.assertIsNotNone(self.load()[0])

    def test_mismatched_faces_reject_correspondence(self):
        for faces in (self.target_faces[::-1], self.target_faces[:, ::-1], self.target_faces.astype(float)):
            with self.subTest(faces=faces):
                np.savez(self.work / "03-source-uv.npz", verts=self.target, faces=faces)
                self.assertRejected("faces do not match")

    def test_subdivided_reordered_corners_use_closest_triangle_barycentrics(self):
        # Each triangle gets a new centre vertex: none of these centres exists in the raw vertex list.
        tri = self.verts[self.faces[::-1]]
        centre = tri.mean(axis=1)
        subdivided = np.stack([np.stack((tri[:, i], tri[:, (i + 1) % 3], centre), axis=1)
                               for i in range(3)], axis=1).reshape(-1, 3)
        faces = np.arange(len(subdivided)).reshape(-1, 3)
        glb.write_glb(self.run / "03-refined.glb", [("split", subdivided, faces,
                      np.full((len(subdivided), 2), .5))], texture_png=png("red"))
        with mock.patch.object(texture, "SourceColour", wraps=texture.SourceColour) as constructor:
            primary, info = self.load()
        self.assertIsNotNone(primary, info)
        expected = self.target[self.target_faces].mean(axis=1)
        from herogen.blender.meshindex import MeshIndex
        placed, placed_faces = constructor.call_args.args[:2]
        self.assertLess(MeshIndex(placed, placed_faces).distance(expected).max(), 2e-6)

    def test_surface_samples_reject_a_bridge_even_with_valid_vertices(self):
        # The raw tetrahedron's interior is not its surface, although every bridge vertex is a raw vertex.
        faces = np.array([[0, 4, 3]])
        glb.write_glb(self.run / "03-refined.glb", [("bridge", self.verts, faces, self.uv)],
                      texture_png=png("red"))
        info = self.assertRejected("surface distance exceeds")
        self.assertGreater(info["source_correspondence"]["geometry_distance_raw"]["max"], .001)

    def test_displaced_refined_surface_fails_the_exact_surface_check(self):
        self.write_refined(offset=np.array([0, 0, 1e-4]))
        info = self.assertRejected("surface distance exceeds")
        check = info["source_correspondence"]
        self.assertGreater(check["geometry_distance_raw"]["max"], check["threshold_raw"])

    def test_raw_tolerance_scales_with_source_size_and_accepts_float32_roundtrips(self):
        original = self.verts.copy()
        rotation = sam3d_mesh.rotation(23, -13, 7)
        for scale in (.1, 1, 10):
            with self.subTest(scale=scale):
                self.verts = original * scale
                glb.write_glb(self.source_path, [("raw", self.verts, self.faces, self.uv)],
                              texture_png=png("green"))
                self.sha = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
                self.write_record()
                rounded = self.verts.astype(np.float32).astype(float)
                rounded = ((rounded @ rotation.T).astype(np.float32).astype(float) @ rotation)
                glb.write_glb(self.run / "03-refined.glb", [("rounded", rounded, self.faces, self.uv)],
                              texture_png=png("red"))
                primary, info = self.load()
                self.assertIsNotNone(primary, info)
                check = info["source_correspondence"]
                self.assertLess(check["geometry_distance_raw"]["size_fraction"]["max"], 1e-7)
                self.write_refined(offset=np.array([0, 0, 1e-4 * check["raw_size"]]))
                self.assertRejected("surface distance exceeds")

    def test_bad_records_and_geometry_have_clear_fallback_reasons(self):
        for value in ("{", "[]"):
            (self.run / "stage-3f.json").write_text(value)
            self.assertRejected("object" if value == "[]" else "could not be read")
        self.write_record()
        (self.run / "03-refined.glb").write_bytes(b"broken")
        self.assertRejected("could not be read")

    def test_nonfinite_corner_uvs_fall_back_before_baking(self):
        uv = self.uv.copy()
        uv[0, 0] = np.nan
        glb.write_glb(self.run / "03-refined.glb", [("refined", self.verts, self.faces, uv)],
                      texture_png=png("red"))
        self.assertRejected("non-finite corner UVs")

    def test_missing_refined_texture_falls_back(self):
        glb.write_glb(self.run / "03-refined.glb", [("refined", self.verts, self.faces, self.uv)])
        self.assertRejected("no base-colour texture")

    def test_degenerate_or_nonfinite_correspondence_falls_back(self):
        target = np.zeros_like(self.target)
        self.save_npz(target)
        self.assertRejected("degenerate")
        target[0, 0] = np.nan
        self.save_npz(target)
        self.assertRejected("correspondence is invalid")

    def test_stage3_entry_deletes_refinement_even_when_the_build_stops(self):
        from herogen.blender.stage3 import run_stage3
        self.assertIsNotNone(self.load()[0])
        with self.assertRaises(ValueError):
            run_stage3(self.run, mesh_source="carve", joints_source="silhouette", fallback=False)
        self.assertRejected("no 03-refined.glb")

    def test_primary_covers_every_texel_and_ignores_views_fallback_and_closeups(self):
        primary, info = self.load()
        self.assertIsNotNone(primary, info)
        sec = {"Head": Section("Head", self.target, self.target_faces, np.zeros(3), [], 0)}
        uv = np.broadcast_to([[.1, .1], [.9, .1], [.1, .9]], (len(self.faces), 3, 2)).copy()
        fallback = mock.Mock(side_effect=AssertionError("fallback must not be sampled"))
        with mock.patch.object(texture, "_project", side_effect=AssertionError("views must not be projected")), \
                mock.patch.object(texture, "_closeup_colour", side_effect=AssertionError("no close-ups")):
            atlas, stats, zbuf = texture.bake(sec, {"Head": uv}, {"front": (None, None)}, 32,
                                              {"front": np.zeros((2, 2))}, source=fallback,
                                              closeups={"front": object()}, primary=primary)
        covered = texture._texels(self.target[self.target_faces], uv, 32)[0]
        colours = atlas.reshape(-1, 3)[covered]
        np.testing.assert_allclose(colours, np.broadcast_to([255, 0, 0], colours.shape), atol=1)
        self.assertEqual(stats["texels_primary"], 1)
        self.assertFalse(stats["closeups_applied"])
        self.assertGreater(stats["texels_used"], 0)
        self.assertIn("primary_nearest_distance_max", stats)
        self.assertIn("front", zbuf)
        fallback.assert_not_called()

    def test_comparison_atlases_are_excluded_from_downloads_and_bundles(self):
        import zipfile
        from herogen.lab.runindex import bodies, deliverables
        from herogen.service.bundle import glb_path, roblox_bundle
        body = self.run / "04-body"
        body.mkdir()
        (body / "high.glb").write_bytes(self.source_path.read_bytes())
        (body / "atlas-high.png").write_bytes(png("red"))
        for name in ("atlas-high-views.png", "atlas-head-high-views.png"):
            (body / name).write_bytes(png("blue"))
        (self.run / "stage-4.json").write_text('{"profile":{"name":"roblox-ingame"}}')
        rig = self.run / "06-roblox"
        rig.mkdir()
        (rig / "HeroGen-R15.rbxmx").write_text('<roblox version="4"/>')
        (rig / "bind_meshes.luau").write_text("-- Bind the imported body.\n")
        (self.run / "stage-6.json").write_text(json.dumps({"ok": True, "outputs": [
            "06-roblox/HeroGen-R15.rbxmx", "06-roblox/bind_meshes.luau",
            "04-body/atlas-high-views.png", "04-body/atlas-head-high-views.png"]}))
        listed = bodies(self.run)[0]
        paths = [f["path"] for f in listed["files"]] + [d.get("path", "") for d in deliverables(self.run, listed, None)]
        self.assertFalse(any("views.png" in path for path in paths))
        self.assertEqual(glb_path(self.run), body / "high.glb")
        with zipfile.ZipFile(io.BytesIO(roblox_bundle(self.run))) as bundle:
            self.assertFalse(any("views.png" in name for name in bundle.namelist()))

    def test_primary_with_degenerate_or_subpixel_uvs_returns_a_blank_bake(self):
        sec = {"Head": Section("Head", self.target, self.target_faces, np.zeros(3), [], 0)}
        for triangle in (np.zeros((3, 2)), np.array([[.1, .1], [.1001, .1], [.1, .1001]])):
            uv = np.broadcast_to(triangle, (len(self.faces), 3, 2))
            primary = mock.Mock(side_effect=AssertionError("no covered texels to sample"))
            atlas, stats, _ = texture.bake(sec, {"Head": uv}, {}, 32, primary=primary)
            self.assertFalse(atlas.any())
            self.assertEqual(stats["texels_used"], 0)
            self.assertIsNone(stats["primary_nearest_distance_max"])
            self.assertIn("no UV texels to bake", stats["notes"])
            primary.assert_not_called()


if __name__ == "__main__":
    unittest.main()
