"""Close-ups in the texture bake (herogen/blender/texture.py) and the scorer's Head texture variants
(herogen/blender/score.py).

A synthetic head (a UV sphere seen by a front view whose face region is flat red)
gets close-ups of the front view whose face is yellow (a flat colour change, which the bake must not take) or a
checkerboard finer than the view (detail, which it must add where the texture is finer than the view).
Nothing is sent to a model and nothing generated is written inside the repository.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.blender import score, texture, views  # noqa: E402
from herogen.pipeline import imaging  # noqa: E402
from herogen.blender.split import Section

SIZE, PPS, CMID, VREF = 256, 160.0, 128.0, 250.0  # a real view's scale; z = 0 at row 250, head rows 10-170
CENTRE, RADIUS = np.array([0.0, 0.0, 1.0]), 0.5
RED, GREEN, BLUE = (230.0, 20.0, 20.0), (20.0, 200.0, 20.0), (20.0, 20.0, 230.0)


def sphere(n_lat: int = 16, n_lon: int = 32):
    """A closed UV sphere (outward winding) and one atlas cell per triangle."""
    verts = [CENTRE + [0, 0, RADIUS], CENTRE - [0, 0, RADIUS]]
    for i in range(1, n_lat):
        th = np.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * np.pi * j / n_lon
            d = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
            verts.append(CENTRE + RADIUS * d)
    verts = np.array(verts)

    def ring(i, j):
        return 2 + (i - 1) * n_lon + j % n_lon

    faces = []
    for j in range(n_lon):
        faces.append([0, ring(1, j), ring(1, j + 1)])
        faces.append([1, ring(n_lat - 1, j + 1), ring(n_lat - 1, j)])
    for i in range(1, n_lat - 1):
        for j in range(n_lon):
            a, b, c, d = ring(i, j), ring(i, j + 1), ring(i + 1, j), ring(i + 1, j + 1)
            faces += [[a, c, d], [a, d, b]]
    faces = np.array(faces)
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum("ij,ij->i", n, tri.mean(axis=1) - CENTRE) < 0
    faces[flip] = faces[flip][:, ::-1]
    g = int(np.ceil(np.sqrt(len(faces))))
    cell = 1.0 / g
    k = np.arange(len(faces))
    u0, v0 = (k % g) * cell, (k // g) * cell
    m = 0.15 * cell
    corner = np.stack([np.stack([u0 + m, v0 + m], 1), np.stack([u0 + cell - m, v0 + m], 1),
                       np.stack([u0 + m, v0 + cell - m], 1)], axis=1)
    return verts, faces, corner


def disk(view: str) -> np.ndarray:
    """The head's silhouette in a view."""
    cols = (np.arange(SIZE) + 0.5 - CMID) / PPS
    rows = (VREF - (np.arange(SIZE) + 0.5)) / PPS
    u, v = np.meshgrid(cols, rows)
    return u ** 2 + (v - CENTRE[2]) ** 2 <= RADIUS ** 2


def face_mask(half_width: float = 0.45) -> np.ndarray:
    """A face region in the front view: |x| < half_width, 0.7 <= z <= 1.3."""
    cols = (np.arange(SIZE) + 0.5 - CMID) / PPS
    rows = (VREF - (np.arange(SIZE) + 0.5)) / PPS
    u, v = np.meshgrid(cols, rows)
    return disk("front") & (np.abs(u) < half_width) & (v >= 0.7) & (v <= 1.3)


def colour_views():
    front = views.View("front", disk("front"), PPS, CMID, VREF)
    left = views.View("left", disk("left"), PPS, CMID, VREF)
    fimg = np.zeros((SIZE, SIZE, 3))
    fimg[:] = BLUE
    fimg[face_mask()] = RED
    limg = np.zeros((SIZE, SIZE, 3))
    limg[:] = GREEN
    return {"front": (front, fimg), "left": (left, limg)}


class FaceBase(unittest.TestCase):
    def setUp(self):
        self.verts, self.faces, self.corner = sphere()
        self.secs = {"Head": Section("Head", self.verts, self.faces, CENTRE, [], 0)}
        self.uvs = {"Head": self.corner}
        self.views = colour_views()
        tri = self.verts[self.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.normals = n / np.linalg.norm(n, axis=1)[:, None]
        self.centres = tri.mean(axis=1)

    def triangle(self, normal) -> int:
        """The triangle near the head's equator whose normal is closest to `normal`."""
        normal = np.asarray(normal, float) / np.linalg.norm(normal)
        score = self.normals @ normal - 5 * np.abs(self.centres[:, 2] - CENTRE[2])
        return int(np.argmax(score))

    def colour(self, atlas: np.ndarray, t: int) -> np.ndarray:
        uv = self.corner[t].mean(axis=0)
        n = atlas.shape[0]
        return atlas[int((1 - uv[1]) * n), int(uv[0] * n)].astype(float)

    def bake(self, **kw):
        return texture.bake(self.secs, self.uvs, self.views, 512, **kw)


YELLOW = (240.0, 220.0, 30.0)


def checker(n: int, period: int) -> np.ndarray:
    """A black and white checkerboard of ``period`` px squares, float RGB."""
    yy, xx = np.mgrid[0:n, 0:n]
    c = (((yy // period) + (xx // period)) % 2) * 255.0
    return np.repeat(c[..., None], 3, axis=2)


def closeup(alpha_box=(0, 0, SIZE, SIZE), pattern: bool = True, k: int = 4) -> texture.Closeup:
    """A close-up of the whole front view at k times its resolution. Its face is yellow instead of red (a flat
    colour change the view does not show) or, with ``pattern``, a checkerboard of k/2 close-up px squares (half a
    view pixel: detail above what the view can show)."""
    _, fimg = colour_views()["front"]
    img = np.repeat(np.repeat(fimg, k, axis=0), k, axis=1).copy()
    big_face = np.repeat(np.repeat(face_mask(), k, axis=0), k, axis=1)
    img[big_face] = YELLOW
    if pattern:
        img[big_face] = checker(img.shape[0], max(1, k // 2))[big_face]
    A = np.array([[1.0 / k, 0.0, 0.0], [0.0, 1.0 / k, 0.0]])
    alpha = np.zeros((SIZE, SIZE))
    l, t, r, b = alpha_box
    alpha[t:b, l:r] = 1.0
    return texture.Closeup("front", img, A, alpha)


class TestCloseupDetail(unittest.TestCase):
    """`texture._closeup_colour`: a close-up adds only detail above the view's resolution."""

    view = SimpleNamespace(pps=PPS)

    def points(self, n=64):
        """Points along a row of the face, in `_project`'s pixel-centre based coordinates, on the centre row of a
        line of checkerboard squares (a row on the squares' edge would average two lines of squares)."""
        return np.full(n, 120.125), np.linspace(110.0, 146.0, n)

    def test_a_flat_colour_change_is_not_taken(self):
        cu = closeup(pattern=False)
        row, col = self.points()
        base = np.tile(np.array(RED), (len(row), 1))
        out, a = texture._closeup_colour(cu, self.view, row, col, base, 4 * PPS)
        self.assertTrue(np.all(a > 0.99))
        self.assertLess(np.abs(out - base).max(), 1.0)        # the yellow face is not painted in

    def test_detail_finer_than_the_view_is_added(self):
        cu = closeup()
        row, col = self.points()
        base = np.tile(np.array(RED), (len(row), 1))
        out, _ = texture._closeup_colour(cu, self.view, row, col, base, 4 * PPS)   # texels 4x finer
        self.assertGreater(np.ptp(out[:, 0]), 150)                                 # the checkerboard shows
        self.assertLess(abs(float(out[:, 0].mean()) - RED[0]), 25)                 # around the view's colour

    def test_a_texture_no_finer_than_the_view_is_unchanged(self):
        cu = closeup()
        row, col = self.points()
        base = np.tile(np.array(RED), (len(row), 1))
        out, a = texture._closeup_colour(cu, self.view, row, col, base, PPS)
        self.assertTrue(np.array_equal(out, base))
        self.assertTrue(np.all(a == 0))

    def test_detail_finer_than_the_texels_averages_out(self):
        # squares of 1 close-up px, texels of 2 close-up px: the prefilter keeps them from aliasing
        cu = closeup(k=8)
        cu = texture.Closeup("front", checker(cu.img.shape[0], 1), cu.A, cu.alpha)
        row, col = self.points()
        base = np.full((len(row), 3), 128.0)
        out, _ = texture._closeup_colour(cu, self.view, row, col, base, 4 * PPS)
        self.assertLess(np.abs(out - base).max(), 20)

    def test_outside_the_footprint_nothing_changes(self):
        cu = closeup(alpha_box=(0, 0, 10, 10))
        row, col = self.points()
        base = np.tile(np.array(RED), (len(row), 1))
        out, a = texture._closeup_colour(cu, self.view, row, col, base, 4 * PPS)
        self.assertTrue(np.all(a == 0))
        self.assertTrue(np.allclose(out, base))


class TestCloseupBake(FaceBase):
    def face_texels(self, tex):
        """The texels around the centroid of the triangle facing the front camera (inside the triangle)."""
        t = self.triangle([0, -1, 0])
        u, v = self.corner[t].mean(axis=0)
        n = tex.shape[0]
        r, c = int((1 - v) * n), int(u * n)
        return tex[r - 5:r + 6, c - 5:c + 6].reshape(-1, 3).astype(float)

    def test_the_bake_adds_the_close_ups_detail_to_the_face(self):
        plain, _, _ = texture.bake(self.secs, self.uvs, self.views, 2048)
        with_cu, stats, _ = texture.bake(self.secs, self.uvs, self.views, 2048, closeups={"front": closeup(k=8)})
        p, c = self.face_texels(plain), self.face_texels(with_cu)
        self.assertLess(p[:, 0].std(), 5)                 # the view's face is flat red
        self.assertGreater(c[:, 0].std(), 30)             # the close-up's checkerboard is in the texture
        self.assertEqual(stats["closeups"]["views"], ["front"])
        self.assertGreater(stats["closeups"]["texels_with_closeup_detail"], 0)


class TestEmptySection(FaceBase):
    def test_an_empty_section_bakes_a_blank_texture(self):
        # stage 4 bakes the Head alone: an empty Head must reach stage 4's gates, not stop the bake
        empty = {"Head": Section("Head", self.verts, np.zeros((0, 3), dtype=np.int64), CENTRE, [], 0)}
        tex, stats, _ = texture.bake(empty, {"Head": np.zeros((0, 3, 2))}, self.views, 64,
                                     closeups={"front": closeup()})
        self.assertEqual(tex.shape, (64, 64, 3))
        self.assertFalse(tex.any())
        self.assertEqual(stats["texels_used"], 0)


class TestLoadCloseups(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        (self.run / "views").mkdir()
        (self.run / "masks").mkdir()
        _, fimg = colour_views()["front"]
        Image.fromarray(fimg.astype(np.uint8)).save(self.run / "views" / "front.png")
        cu = closeup()
        Image.fromarray(cu.img.astype(np.uint8)).save(self.run / "views" / "closeup-front.png")
        foot = disk("front")
        imaging.to_mask_image(foot).save(self.run / "masks" / "closeup-front.png")
        self.vs = views.ViewSet({"front": views.View("front", disk("front"), PPS, CMID, VREF)}, [], {}, PPS, 160, [])
        self.rec = {"views": {"front": {
            "used": True, "file": "views/closeup-front.png", "footprint": "masks/closeup-front.png",
            "A": cu.A.tolist(), "crop_box": [0, 0, 200, SIZE],
            "view_sha256": hashlib.sha256((self.run / "views" / "front.png").read_bytes()).hexdigest()},
            "left": {"used": False, "reason": "NCC in the view 0.5 < 0.97"}}}

    def tearDown(self):
        self.tmp.cleanup()

    def write(self):
        (self.run / "views" / "closeup.json").write_text(json.dumps(self.rec))

    def test_loads_the_used_close_ups_with_a_feathered_footprint(self):
        self.write()
        cus, info = texture.load_closeups(self.run, self.vs)
        self.assertEqual(sorted(cus), ["front"])
        self.assertEqual(info["used"], ["front"])
        a = cus["front"].alpha
        foot = disk("front")
        inner = imaging.erode(foot, 4)
        self.assertTrue(np.allclose(a[inner & (np.arange(SIZE) < 195)[None, :]], 1.0))
        self.assertTrue(np.all(a[:, 200:] == 0))                     # never past the crop box
        self.assertTrue(np.all(a[~imaging.dilate(foot, 10)] < 1e-3))
        self.assertTrue(((a > 0) & (a < 1)).any())                   # a soft edge

    def test_a_close_up_of_an_older_view_is_not_used(self):
        self.rec["views"]["front"]["view_sha256"] = "0" * 64
        self.write()
        cus, info = texture.load_closeups(self.run, self.vs)
        self.assertEqual(cus, {})
        self.assertIn("another front view", info["notes"][0])

    def test_rejected_views_keep_their_reason(self):
        self.write()
        _, info = texture.load_closeups(self.run, self.vs)
        self.assertIn("left: NCC in the view 0.5 < 0.97", info["notes"])

    def test_invalid_transforms_and_crops_fall_back(self):
        for key, value in (("A", [[0, 0, 0], [0, 0, 0]]),
                           ("A", [[float("nan"), 0, 0], [0, 1, 0]]),
                           ("crop_box", [10, 0, 0, SIZE])):
            with self.subTest(key=key, value=value):
                old = self.rec["views"]["front"][key]
                self.rec["views"]["front"][key] = value
                self.write()
                cus, info = texture.load_closeups(self.run, self.vs)
                self.assertEqual(cus, {})
                self.assertIn("invalid transform or crop", info["notes"][0])
                self.rec["views"]["front"][key] = old

    def test_wrong_json_types_fall_back(self):
        for record in ([], {"views": []}, {"views": {"front": [1]}}):
            with self.subTest(record=record):
                (self.run / "views" / "closeup.json").write_text(json.dumps(record))
                cus, info = texture.load_closeups(self.run, self.vs)
                self.assertEqual(cus, {})
                self.assertIn("could not be read", info["notes"][0])

    def test_no_record_means_no_close_ups(self):
        cus, info = texture.load_closeups(self.run, self.vs)
        self.assertEqual(cus, {})
        self.assertIn("stage 2f made none", info["notes"][0])

    def test_a_broken_record_is_reported(self):
        del self.rec["views"]["front"]["A"]
        self.write()
        cus, info = texture.load_closeups(self.run, self.vs)
        self.assertEqual(cus, {})
        self.assertIn("could not be read", info["notes"][0])


class TestScoreHeadTexture(unittest.TestCase):
    """`score.body_colour`: an atlas variant replaces the body atlas; the Head's own texture is replaced only by
    a Head texture variant."""

    def glb(self):
        mats = [SimpleNamespace(name="BodyAtlas", image=0, base_color=(1, 1, 1, 1)),
                SimpleNamespace(name=score.HEAD_MATERIAL, image=1, base_color=(1, 1, 1, 1))]
        imgs = [np.full((4, 4, 4), 100.0), np.full((4, 4, 4), 200.0)]
        return SimpleNamespace(materials=mats, meshes=[SimpleNamespace(colors=None)],
                               merged_face_material=lambda: np.array([0, 1]),
                               merged_uv=lambda: np.zeros((3, 2)), image=lambda i: imgs[i])

    def test_textures(self):
        faces = np.array([[0, 1, 2], [0, 1, 2]])
        own = score.body_colour(self.glb(), faces)
        self.assertAlmostEqual(own.textures[0][0, 0, 0], score._to_linear(np.array(100 / 255))[()], places=6)
        atlas = np.full((4, 4, 3), 10.0)
        head = np.full((4, 4, 3), 250.0)
        v = score.body_colour(self.glb(), faces, atlas)
        self.assertAlmostEqual(v.textures[0][0, 0, 0], score._to_linear(np.array(10 / 255))[()], places=6)
        self.assertAlmostEqual(v.textures[1][0, 0, 0], score._to_linear(np.array(200 / 255))[()], places=6)
        vh = score.body_colour(self.glb(), faces, atlas, head)
        self.assertAlmostEqual(vh.textures[1][0, 0, 0], score._to_linear(np.array(250 / 255))[()], places=6)

    def test_body_override_keeps_the_head_texture_in_the_profile_limit(self):
        import io
        from unittest.mock import patch
        from herogen.blender.glb import load_glb, write_glb

        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            buf = io.BytesIO()
            Image.new("RGB", (4096, 2)).save(buf, format="PNG")
            verts = np.array([[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]])
            body = write_glb(run / "head.glb", [("Head", verts, np.array([[0, 1, 2]]),
                                                np.zeros((3, 2)))], texture_png=buf.getvalue())
            glb = load_glb(body)
            glb.materials[0].name = score.HEAD_MATERIAL
            small = run / "small.png"
            Image.new("RGB", (8, 8)).save(small)
            with patch.object(score, "load_glb", return_value=glb):
                kept = score.score_body(run, "kept", body, atlas=small, views=[], profile="roblox-avatar")
                replaced = score.score_body(run, "replaced", body, atlas=small, head_atlas=small,
                                             views=[], profile="roblox-avatar")
            checks = {c["name"]: c for c in kept["profile"]["checks"]}
            self.assertEqual(kept["atlas"], 4096)
            self.assertFalse(checks["texture_max_px"]["pass"])
            self.assertEqual(replaced["atlas"], 8)
            self.assertEqual(replaced["head_atlas_image"], "small.png")

    def test_discover_pairs_a_variant_with_its_head_texture(self):
        with tempfile.TemporaryDirectory() as d:
            body = Path(d) / "04-body"
            body.mkdir()
            (body / "high.glb").write_bytes(b"glTF")
            for name in ("atlas-high.png", "atlas-head-high.png", "atlas-high-no-closeup.png",
                         "atlas-head-high-no-closeup.png", "atlas-high-side-trust.png"):
                Image.new("RGB", (4, 4)).save(body / name)
            specs = {s.name: s for s in score.discover_bodies(Path(d))}
            self.assertEqual(sorted(specs), ["carve", "carve-no-closeup", "carve-side-trust"])
            self.assertEqual(specs["carve-no-closeup"].head_atlas.name, "atlas-head-high-no-closeup.png")
            self.assertIsNone(specs["carve-side-trust"].head_atlas)
            self.assertIsNone(specs["carve"].head_atlas)


if __name__ == "__main__":
    unittest.main()
