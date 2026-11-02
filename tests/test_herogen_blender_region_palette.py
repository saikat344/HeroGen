"""Synthetic saved-mask palette calibration; no network or model calls."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

try:
    from herogen.blender.refine import region_palette as palette
    HAVE_REFINE = True
except ImportError:
    HAVE_REFINE = False


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class RegionPaletteTests(unittest.TestCase):
    def setUp(self):
        self.source = np.full((8, 8, 3), (.25, .2, .15), np.float32)
        self.art = np.full((8, 8, 3), (.6, .5, .4), np.float32)
        self.skin = np.indices((8, 8))[0] < 4
        self.masks = {"skin": self.skin, "rest": ~self.skin}

    def test_disabled_path_is_identity_without_validating_optional_inputs(self):
        self.assertIs(palette.apply(self.source, None, None), self.source)

    def test_regions_are_independent_even_when_clothing_dominates(self):
        self.art[self.skin] = self.source[self.skin]
        grades = palette.fit(self.source, self.art, self.masks, self.masks)
        self.assertEqual(grades["skin"]["cap_lab"], 0)
        self.assertGreater(grades["rest"]["cap_lab"], 0)
        out = palette.apply(self.source, grades, self.masks, enabled=True)
        np.testing.assert_array_equal(out[self.skin], self.source[self.skin])
        self.assertLess(np.abs(out[~self.skin] - self.art[~self.skin]).mean(), .005)

    def test_cap_is_measured_from_each_regions_quantile_distance(self):
        import cv2
        grades = palette.fit(self.source, self.art, self.masks, self.masks)
        a = cv2.cvtColor(self.source, cv2.COLOR_RGB2Lab)[0, 0]
        b = cv2.cvtColor(self.art, cv2.COLOR_RGB2Lab)[0, 0]
        self.assertAlmostEqual(grades["skin"]["cap_lab"], float(np.linalg.norm(b - a)), places=4)

    def test_missing_and_empty_masks_are_reported_and_unchanged(self):
        masks = {"skin": np.zeros_like(self.skin)}
        grades = palette.fit(self.source, self.art, masks, masks)
        self.assertEqual(grades["skin"]["status"], "empty_mask")
        self.assertEqual(grades["hair"]["status"], "missing_mask")
        np.testing.assert_array_equal(palette.apply(self.source, grades, self.masks, enabled=True), self.source)

    def test_unlabelled_pixels_are_byte_identical(self):
        masks = {"skin": self.skin}
        grades = palette.fit(self.source, self.art, masks, masks)
        out = palette.apply(self.source, grades, masks, enabled=True)
        np.testing.assert_array_equal(out[~self.skin], self.source[~self.skin])

    def test_bad_masks_and_rgb_fail_loudly(self):
        for masks in ({"wrong": self.skin}, {"skin": self.skin, "hair": self.skin},
                      {"skin": self.skin[:2]}):
            with self.subTest(masks=list(masks)), self.assertRaises(ValueError):
                palette.fit(self.source, self.art, masks, self.masks)
        for value in (np.full((8, 8, 3), np.nan), self.source * -1, self.source[..., 0]):
            with self.assertRaises(ValueError):
                palette.fit(value, self.art, self.masks, self.masks)

    def test_repeated_source_quantiles_remain_finite(self):
        self.art[::2] = .8
        grades = palette.fit(self.source, self.art, self.masks, self.masks)
        self.assertTrue(np.isfinite(palette.apply(self.source, grades, self.masks, enabled=True)).all())
        json.dumps(grades, allow_nan=False)

    def test_changed_island_colours_extend_into_gutters_without_changing_other_islands(self):
        rgb = np.zeros((5, 9, 3), np.float32)
        cov = np.zeros((5, 9), bool)
        cov[2, 1] = cov[2, 7] = True
        rgb[2, 1] = (.8, .6, .5)
        rgb[2, 7] = (.1, .2, .3)
        result = palette.extend_margins(rgb, cov)
        np.testing.assert_array_equal(result[cov], rgb[cov])
        np.testing.assert_array_equal(result[2, 0], rgb[2, 1])
        np.testing.assert_array_equal(result[2, 8], rgb[2, 7])
        with self.assertRaises(ValueError):
            palette.extend_margins(rgb, np.zeros_like(cov))

    def test_feather_has_zero_boundary_and_full_interior_with_smooth_ramp(self):
        mask = np.zeros((40, 60), bool)
        mask[3:37, 5:55] = True
        weight = palette.feather(mask, 9)
        self.assertEqual(weight[3, 20], 0)
        self.assertEqual(weight[15, 20], 1)
        self.assertTrue((weight[~mask] == 0).all())
        self.assertLess(np.abs(np.diff(weight[:, 20])).max(), 1.5 / 9)
        for width in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                palette.feather(mask, width)

    def test_feather_ignores_silhouette_and_gutters_but_retains_internal_transition(self):
        domain = np.zeros((40, 40), bool)
        domain[5:35, 5:35] = True
        mask = domain.copy()
        mask[25:] = False
        weight = palette.feather(mask, 9, domain=domain)
        self.assertEqual(weight[5, 15], 1)
        self.assertEqual(weight[24, 15], 0)
        self.assertEqual(weight[23, 15] > 0, True)
        self.assertFalse(weight[~mask].any())
        with self.assertRaisesRegex(ValueError, "domain"):
            palette.feather(domain, 9, domain=mask)

    def test_weighted_grade_tapers_and_protects_zero_weight_exactly(self):
        grades = palette.fit(self.source, self.art, self.masks, self.masks)
        weight = np.tile(np.linspace(0, 1, 8), (8, 1))
        weights = {name: weight * mask for name, mask in self.masks.items()}
        out = palette.apply(self.source, grades, self.masks, enabled=True, weights=weights)
        np.testing.assert_array_equal(out[:, 0], self.source[:, 0])
        self.assertLess(np.abs(out[:, -1] - self.art[:, -1]).mean(), .005)
        self.assertTrue((np.diff(out[1, :, 0]) > 0).all())
        for bad in (np.ones((2, 2)), np.full((8, 8), np.nan), np.full((8, 8), -1),
                    np.ones((8, 8))):
            with self.assertRaises(ValueError):
                palette.apply(self.source, grades, self.masks, enabled=True, weights={"skin": bad})
        self.assertIs(palette.apply(self.source, None, None, weights={"invalid": None}), self.source)

    def test_face_completion_fills_connected_skin_without_crossing_hair_or_head(self):
        rgb = np.full((40, 40, 3), (.7, .5, .35), np.float32)
        rgb[:10] = (.1, .15, .3)
        head = np.zeros((40, 40), bool)
        head[2:35, 5:35] = True
        seed = np.zeros((40, 40), bool)
        seed[14:30, 10:18] = True
        result = palette.complete_face(rgb, seed, head)
        self.assertTrue(result[14:30, 18:34].all())
        self.assertFalse(result[:9].any())
        self.assertFalse(result[~head].any())
        self.assertTrue(result[seed].all())
        self.assertFalse(palette.complete_face(rgb, np.zeros_like(seed), head).any())

    def test_face_completion_includes_lighter_scalp_but_rejects_dark_same_chroma_hair(self):
        import cv2
        lab = np.full((50, 50, 3), (15, 10, 20), np.float32)
        seed = np.zeros((50, 50), bool)
        seed[20:40, 15:35] = True
        lab[seed, 0] = np.tile(np.linspace(45, 65, 20), 20)
        lab[seed, 1] = np.tile(np.linspace(6, 22, 20), 20)
        lab[seed, 2] = np.tile(np.linspace(18, 26, 20), 20)
        lab[8:20, 15:35] = (70, 10, 20)
        # An existing shadow remains part of the face despite the growth luminance bound.
        lab[30:35, 20:25] = (20, 10, 20)
        rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        completed = palette.complete_face(rgb, seed, np.ones(seed.shape, bool))
        self.assertTrue(completed[8:20, 15:35].all())
        self.assertTrue(completed[seed].all())
        self.assertFalse(completed[:, :10].any())

    def test_export_validates_layout_before_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pos.exr").write_bytes(b"layout")
            (root / "manifest.json").write_text('{"atlas_layout_sha256":"different"}')
            with mock.patch("herogen.blender.refine.backend.blender") as render, self.assertRaisesRegex(ValueError, "layout"):
                palette.export_texture(root / "manifest.json", root, None, None, root)
            render.assert_not_called()

    def test_export_uses_only_saved_masks_and_records_missing_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pos.exr").write_bytes(b"layout")
            for name, rgb in (("art", self.art), ("texture", self.source), ("body+0", self.source)):
                Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(root / (name + ".png"))
            Image.fromarray(self.skin.astype(np.uint8) * 255).save(root / "skin.png")
            Image.new("L", (8, 8), 255).save(root / "cov.png")
            masks = {"skin": "skin.png"}
            data = {"atlas_layout_sha256": hashlib.sha256(b"layout").hexdigest(), "artwork": "art.png",
                    "artwork_masks": masks, "render_masks": masks, "atlas_masks": masks}
            (root / "manifest.json").write_text(json.dumps(data))
            with mock.patch("herogen.blender.refine.backend.blender") as render:
                target, grades = palette.export_texture(root / "manifest.json", root, root / "texture.png",
                                                       {"name": "body+0"}, root)
            self.assertTrue(target.is_file())
            self.assertEqual(grades["hair"]["status"], "missing_mask")
            self.assertEqual(render.call_count, 1)

    def test_export_clips_saved_application_to_current_silhouette(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pos.exr").write_bytes(b"layout")
            for name, rgb in (("art", self.art), ("texture", self.source)):
                Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(root / (name + ".png"))
            rgba = np.full((8, 8, 4), 200, np.uint8)
            rgba[..., 3] = 255
            rgba[0, :, 3] = 0
            Image.fromarray(rgba).save(root / "body+0.png")
            Image.fromarray(self.skin.astype(np.uint8) * 255).save(root / "skin.png")
            Image.new("L", (8, 8), 255).save(root / "cov.png")
            Image.new("L", (8, 8), 255).save(root / "application.png")
            masks = {"skin": "skin.png"}
            data = {"atlas_layout_sha256": hashlib.sha256(b"layout").hexdigest(),
                    "artwork": "art.png", "artwork_masks": masks, "render_masks": masks,
                    "atlas_masks": masks, "application_masks": {"skin": "application.png"},
                    "completion_head": "application.png", "feather_pixels_at_1536": 9}
            (root / "manifest.json").write_text(json.dumps(data))
            with mock.patch("herogen.blender.refine.backend.blender"), mock.patch.object(
                    palette, "project_fields", side_effect=lambda prep, camera, figure, fields: fields) as project:
                target, _ = palette.export_texture(root / "manifest.json", root, root / "texture.png",
                                                   {"name": "body+0"}, root)
            self.assertTrue(target.is_file())
            fields = project.call_args.args[3]
            self.assertFalse(fields["skin"][0].any())
            self.assertTrue((fields["skin"][1:] > 0).all())
            with Image.open(root / "application-skin.png") as mask:
                self.assertFalse(np.asarray(mask)[0].any())


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class GeneratedManifestTests(unittest.TestCase):
    def setUp(self):
        from herogen.blender.refine.cameras import camera
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prep = self.root / "work" / "prep"
        self.probe = self.prep.parent / "probe"
        for folder in (self.prep, self.probe, self.root / "views", self.root / "masks"):
            folder.mkdir(parents=True, exist_ok=True)
        self.folder = self.root / "generated"
        self.cam = camera("body+0", [0, 0, 0], 0, 4, (4, 4))
        Image.new("RGB", (4, 4), "tan").save(self.root / "views/front.png")
        Image.new("RGBA", (4, 4), (80, 60, 40, 255)).save(self.probe / "probe0.png")
        self.face = np.zeros((4, 4), bool)
        self.face[0, :2] = True
        self.head = np.zeros((4, 4), bool)
        self.head[:2] = True
        for name, value in (("front", np.ones((4, 4), bool)), ("front-head", self.head),
                            ("front-face", self.face)):
            self.save(self.root / "masks" / (name + ".png"), value)
        self.save(self.probe / "palette-face.png", self.face)
        self.save(self.probe / "palette-head.png", self.head)
        (self.prep / "pos.exr").write_bytes(b"synthetic-layout")
        yy, xx = np.indices((4, 4))
        self.positions = np.stack([xx - 1.5, np.zeros((4, 4)), 1.5 - yy], axis=-1)
        self.normals = np.zeros_like(self.positions)
        self.normals[..., 1] = -1
        self.maps = mock.patch("herogen.blender.refine.character.texel_maps", return_value=(
            self.positions, self.normals, np.ones((4, 4), bool)))
        self.maps.start()
        self.addCleanup(self.maps.stop)

    def save(self, path, mask):
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)

    def build(self):
        path = palette.prepare_manifest(self.root, self.prep, self.cam, self.folder)
        return json.loads(path.read_text())

    def mask(self, name):
        return np.asarray(Image.open(self.folder / name).convert("L")) > 127

    def test_generated_masks_are_disjoint_and_bound_to_current_layout(self):
        data = self.build()
        self.assertEqual(data["atlas_layout_sha256"], hashlib.sha256(b"synthetic-layout").hexdigest())
        self.assertEqual((self.folder / "artwork.png").read_bytes(),
                         (self.root / "views/front.png").read_bytes())
        for prefix in ("art", "render", "atlas"):
            skin, rest = self.mask(prefix + "-skin.png"), self.mask(prefix + "-rest.png")
            np.testing.assert_array_equal(skin, self.face)
            np.testing.assert_array_equal(rest, ~self.head)
            self.assertFalse((skin & rest).any())

    def test_missing_art_face_is_recorded_and_never_uses_head_as_skin(self):
        (self.root / "masks/front-face.png").unlink()
        data = self.build()
        self.assertEqual(data["artwork_face_mask"], "missing_mask")
        for prefix in ("art", "render", "atlas"):
            self.assertFalse(self.mask(prefix + "-skin.png").any())

    def test_back_facing_texels_are_not_graded(self):
        self.normals[0, 0, 1] = 1
        self.build()
        self.assertFalse(self.mask("atlas-skin.png")[0, 0])
        self.assertTrue(self.mask("render-skin.png")[0, 0])

    def test_face_intersection_preserves_original_masks_for_evaluation(self):
        face = self.face.copy()
        face[0, 0] = False
        self.save(self.probe / "palette-face.png", face)
        self.build()
        np.testing.assert_array_equal(self.mask("art-skin.png"), face)
        # Fitting keeps the conservative overlap, while application retains the whole saved face.
        self.assertTrue(self.mask("application-skin.png")[face].all())
        np.testing.assert_array_equal(self.mask("original-art-skin.png"), self.face)

    def test_partial_art_alignment_does_not_cut_the_render_face(self):
        partial = self.face.copy()
        partial[0, 0] = False
        self.save(self.root / "masks/front-face.png", partial)
        self.build()
        self.assertFalse(self.mask("render-skin.png")[0, 0])
        self.assertTrue(self.mask("application-skin.png")[0, 0])

    def test_projected_fields_protect_backfaces_and_uncovered_texels(self):
        self.normals[0, 0, 1] = 1
        fields = {"skin": np.ones((4, 4), np.float32)}
        result = palette.project_fields(self.prep, self.cam, np.ones((4, 4), bool), fields)
        self.assertEqual(result["skin"][0, 0], 0)
        self.assertEqual(result["skin"][2, 2], 1)
        self.normals[2, 2] = [1, 0, 0]
        result = palette.project_fields(self.prep, self.cam, np.ones((4, 4), bool), fields)
        self.assertEqual(result["skin"][2, 2], 0)

    def test_invalid_or_absent_required_masks_fail_loudly(self):
        Image.new("L", (2, 2)).save(self.root / "masks/front-head.png")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.build()
        (self.root / "masks/front-head.png").unlink()
        with self.assertRaises(FileNotFoundError):
            self.build()

    def test_transfer_requires_nonempty_foregrounds(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            palette.transfer_mask(self.face, np.zeros((4, 4), bool), self.head)


if __name__ == "__main__":
    unittest.main()
