"""Synthetic texture-refinement checks, with no model service calls."""
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from herogen.blender import bl
from herogen.blender import refine as stage
from herogen.blender.final_model import final_model
from herogen.pipeline.client import ApiError, CallLog, IMAGE_MODEL, SAM_MODEL, ModelClient, ReplayMiss, SamInstance
try:
    from herogen.blender.refine import cameras, character, imaging as refine, prompts
    HAVE_REFINE = True
except ImportError:
    HAVE_REFINE = False

def disc(shape, cx, cy, r):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r * r


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class CameraTests(unittest.TestCase):
    def test_front_camera_maps_centre_to_image_centre_and_up_to_up(self):
        cam = cameras.camera("c", [0.1, 0.2, 0.3], 0, 2.0, (100, 200))
        x, y, _ = cameras.project(cam, np.array([[0.1, 5.0, 0.3], [0.1, 0.2, 0.8]]))
        np.testing.assert_allclose([x[0], y[0]], [50, 100])
        np.testing.assert_allclose([x[1], y[1]], [50, 100 - 0.5 * 100])   # 200 px over 2 units

    def test_turned_camera_depth_grows_away_from_camera(self):
        cam = cameras.camera("c", [0, 0, 0], 90, 1.0, (64, 64))
        _, _, fwd = cameras.axes(cam)
        _, _, d = cameras.project(cam, np.array([fwd * 0.5, -fwd * 0.5]))
        self.assertGreater(d[0], d[1])

    def test_position_codec_round_trips(self):
        p = np.array([[[0.25, -1.5, 1.9, 1.0]]], np.float32)
        enc = p.copy()
        enc[..., :3] = (p[..., :3] - cameras.POS_LO) / cameras.POS_EXT
        dec, valid = cameras.decode_pos(enc)
        np.testing.assert_allclose(dec, p[..., :3], atol=1e-6)
        self.assertTrue(valid.all())


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class GateTests(unittest.TestCase):
    def test_chamfer_is_zero_for_identical_and_grows_with_shift(self):
        a = disc((128, 128), 64, 64, 30)
        self.assertEqual(refine.chamfer(a, a), (0.0, 0.0))
        med, p90 = refine.chamfer(a, disc((128, 128), 70, 64, 30))
        self.assertGreater(med, 1.0)
        self.assertLessEqual(p90, 7.0)

    def test_chamfer_ignores_the_image_border(self):
        a = np.zeros((64, 64), bool)
        a[20:, 10:50] = True            # touches the bottom border
        b = a.copy()
        b[-2:] = False                  # differs only next to the border
        self.assertEqual(refine.chamfer(a, b)[0], 0.0)

    def test_figure_mask_finds_the_figure_on_the_grey(self):
        img = np.full((64, 64, 3), refine.C.BG, np.float32)
        m = disc((64, 64), 32, 32, 20)
        img[m] = (0.8, 0.6, 0.5)
        got = refine.figure_mask(img)
        self.assertGreater((got & m).sum() / (got | m).sum(), 0.97)

    def test_part_gate_rejects_a_face_painted_where_the_render_has_none(self):
        fig = disc((100, 100), 50, 50, 45)
        none = {"face": np.zeros_like(fig)}
        face = {"face": disc((100, 100), 50, 40, 15)}
        ok, rec, _ = refine.part_gate(none, face, fig)
        self.assertFalse(ok)
        self.assertEqual(rec["face"]["outside"], 1.0)
        ok, rec, _ = refine.part_gate(face, face, fig)
        self.assertTrue(ok)
        ok, rec, _ = refine.part_gate(none, none, fig)
        self.assertTrue(ok)
        self.assertIsNone(rec["face"])

    def test_part_gate_accepts_a_sharper_face_inside_a_soft_one(self):
        fig = disc((100, 100), 50, 50, 45)
        soft = {"face": disc((100, 100), 50, 50, 30)}      # face and beard on the soft render
        sharp = {"face": disc((100, 100), 50, 40, 15)}     # face only on the repaint
        ok, rec, _ = refine.part_gate(soft, sharp, fig)
        self.assertTrue(ok)
        self.assertLess(rec["face"]["iou"], 0.3)


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class ColourTests(unittest.TestCase):
    def test_colour_match_takes_render_palette_and_keeps_repaint_detail(self):
        rng = np.random.default_rng(0)
        h = w = 256
        both = np.ones((h, w), bool)
        render = np.zeros((h, w, 3), np.float32)
        render[:, : w // 2] = (0.2, 0.3, 0.7)        # SAM's blue shirt
        render[:, w // 2:] = (0.85, 0.65, 0.5)       # SAM's skin
        repaint = np.zeros_like(render)
        repaint[:, : w // 2] = (0.35, 0.45, 0.7)     # repaint's lighter blue
        repaint[:, w // 2:] = (0.95, 0.8, 0.75)      # repaint's paler skin
        detail = (rng.random((h, w)) > 0.97)          # sharp dark specks only the repaint has
        repaint[detail] *= 0.3
        out = refine.colour_match(render, repaint, both)
        inner = np.zeros((h, w), bool)
        inner[20:-20, 20:100] = True
        plain = inner & ~detail
        np.testing.assert_allclose(out[plain].mean(0), (0.2, 0.3, 0.7), atol=0.04)
        contrast = out[plain].mean() - out[inner & detail].mean()
        self.assertGreater(contrast, 0.15)


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class HairModeTests(unittest.TestCase):
    def test_interior_agreement_keeps_detail_and_drops_a_swapped_region(self):
        rng = np.random.default_rng(1)
        h = w = 256
        fig = disc((h, w), 128, 128, 110)
        render = np.full((h, w, 3), refine.C.BG, np.float32)
        render[fig] = (0.25, 0.15, 0.1)                  # soft dark hair
        repaint = render.copy()
        strands = fig & (rng.random((h, w)) > 0.8)
        repaint[strands] = (0.35, 0.22, 0.15)            # fine strands: same colour at low frequency
        ear = disc((h, w), 170, 128, 25)
        repaint[ear] = (0.9, 0.7, 0.6)                   # skin painted where the render has hair
        frac, agree, interior = refine.interior_agreement(render, repaint, fig, fig)
        self.assertLess(agree[disc((h, w), 170, 128, 10)].max(), 0.1)
        self.assertGreater(agree[disc((h, w), 80, 128, 20)].min(), 0.9)
        self.assertGreater(frac, 0.8)
        self.assertFalse(interior[0, 0])


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class RegistrationTests(unittest.TestCase):
    def test_affine_recovers_a_translation_and_brightness_change(self):
        import cv2
        yy, xx = np.mgrid[:128, :128]
        mask = disc((128, 128), 64, 64, 45)
        render = np.full((128, 128, 3), character.BG, np.float32)
        render[mask] = np.stack([0.2 + xx[mask] / 300, 0.2 + yy[mask] / 400,
                                0.4 + np.sin(xx[mask] / 8) / 9], -1)
        edit = cv2.warpAffine(render, np.float32([[1, 0, 4], [0, 1, -3]]), (128, 128),
                              borderMode=cv2.BORDER_REPLICATE) * 0.9 + 0.03
        _, _, warped, rec = refine.register(render, mask.astype(float), edit)
        affine = np.array(rec["affine_render_to_edit"])
        np.testing.assert_allclose(affine[:, 2], (4, -3), atol=0.7)
        self.assertLess(np.abs(warped[mask] - (render[mask] * 0.9 + 0.03)).mean(), 0.01)

    def test_empty_colour_overlap_is_finite(self):
        rgb = np.ones((16, 16, 3), np.float32) * 0.2
        np.testing.assert_array_equal(refine.colour_match(rgb, rgb, np.zeros((16, 16), bool)), rgb)

    def test_camera_set_and_frozen_prompts(self):
        import hashlib

        self.assertEqual({k: hashlib.sha256(v.encode()).hexdigest() for k, v in prompts.PROMPTS.items()}, {
            "RF2R": "794b08622f4ff650261513c39aee5f8ac9b92e6c9bfaf65a1334c96b86d4b224",
            "RF3H": "f4f45c11280cac98d8234ee729650cffe1bcc73c8f6fc844df9b407248936f2f",
        })
        info = {"height": 2, "body_centre": [0, 0, 1], "hi": [1, 1, 2],
                "head_centre": [0, 0, 1.8], "head_side": 0.4}
        cams = character.cameras_for(info)
        self.assertEqual(len(stage.REFINE_ORDER), 12)
        self.assertEqual(len(stage.HAIR_ORDER), 8)
        for name in (*stage.REFINE_ORDER, *stage.HAIR_ORDER):
            self.assertIn(name, cams)
        self.assertIn("in profile", prompts.prompt("RF2R", 270))
        self.assertIn("back and side of the head", prompts.prompt("RF3H", 225))
        self.assertIn("second image", prompts.prompt("RF3H", 180))

    def test_hair_weight_cannot_write_face_scalp_or_outline(self):
        shape = (128, 128)
        fig = disc(shape, 64, 64, 50)
        head = disc(shape, 64, 50, 30)
        hair = head & (np.indices(shape)[1] < 55)
        face = head & (np.indices(shape)[1] > 70)
        rgb = np.full((*shape, 3), character.BG, np.float32)
        rgb[fig] = (0.2, 0.15, 0.1)
        inp, edit = Image.fromarray((rgb * 255).astype(np.uint8)), Image.fromarray((rgb * 255).astype(np.uint8))
        def segment(image, text):
            m = head if text == "head" else hair
            return [SamInstance(m, (0, 0, 128, 128))]
        client = SimpleNamespace(segment=segment)
        yy, xx = np.mgrid[:128, :128]
        weight = refine.hair_weight(client, inp, edit, rgb, rgb, fig, xx + .5, yy + .5, face)
        self.assertEqual(weight[face].max(), 0)
        self.assertEqual(weight[head & (xx > 60)].max(), 0)
        self.assertEqual(weight[~fig].max(), 0)
        self.assertGreater(weight[hair & (xx < 50)].max(), .5)
        client.segment = lambda image, text: [SamInstance(head if text == "head" else np.zeros(shape, bool),
                                                        (0, 0, 128, 128))]
        allowed = refine.hair_allowed(client, inp, edit, xx + .5, yy + .5, shape)
        self.assertFalse(allowed[head].any())

    def test_projection_preserves_occluded_backfacing_and_unseen_texels(self):
        tex = object.__new__(refine.Texture)
        tex.P = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0], [5, 0, 0]], float)
        tex.N = np.array([[0, -1, 0], [0, -1, 0], [0, 1, 0], [0, -1, 0]], float)
        tex.idx = np.arange(4)
        tex.Q = np.ones(4) * 0.01
        tex.height = 1
        tex.T = np.full((2, 2, 3), 0.2)
        cam = cameras.camera("c", [0, 0, 0], 0, 1, (32, 32))
        count = tex.apply(cam, np.ones((32, 32, 3)), np.ones((32, 32)), np.zeros((32, 32)),
                          np.ones((32, 32), bool), np.ones((32, 32)))
        self.assertEqual(count, 1)
        np.testing.assert_array_equal(tex.T.reshape(-1, 3)[1:], 0.2)
        self.assertGreater(tex.T[0, 0].min(), .9)


class BudgetTests(unittest.TestCase):
    def test_cap_boundary_and_prior_spend(self):
        client = ModelClient(Path("unused"))
        budget = stage.Budget(client, .02, prior=.01)
        budget.before_attempt(IMAGE_MODEL)
        client.log.add(IMAGE_MODEL, "edit", "k", False)
        with self.assertRaises(stage.BudgetStop):
            budget.before_attempt(SAM_MODEL)
        for cap in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                stage.Budget(client, cap)
        self.assertAlmostEqual(stage.cost_estimate(), .365)
        self.assertAlmostEqual(stage.cost_estimate(True), .765)

    def test_unknown_outcome_retry_is_capped_before_http(self):
        from herogen.pipeline import client as module, retry
        with tempfile.TemporaryDirectory() as tmp:
            limiter = mock.Mock()
            limiter.acquire.return_value = ("slot", 0)
            client = ModelClient(tmp, mode="live", policy=retry.RetryPolicy(max_attempts=4, sleep=lambda s: None,
                                                                           base_s=0), limiter=limiter)
            client.before_api_attempt = stage.Budget(client, .02).before_attempt
            with mock.patch.dict(os.environ, {"MODEL_API_KEY": "test-placeholder"}), \
                    mock.patch.object(module, "_open", side_effect=urllib.error.URLError(TimeoutError())) as request:
                with self.assertRaises(stage.BudgetStop):
                    client.edit_image("p", [Image.new("RGB", (8, 8))])
            self.assertEqual(request.call_count, 2)
            self.assertEqual(client.log.cost_usd(), .02)
            self.assertEqual(limiter.release.call_count, 2)

    def test_replay_at_zero_cap_misses_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = ModelClient(tmp)
            client.before_api_attempt = stage.Budget(client, 0).before_attempt
            with self.assertRaises(ReplayMiss):
                client.segment(Image.new("RGB", (8, 8)), "head")

    def test_cached_edit_and_segment_are_free_at_zero_cap(self):
        import hashlib
        from herogen.pipeline.client import png_bytes, request_key
        image = Image.new("RGB", (8, 8), "red")
        digest = hashlib.sha256(png_bytes(image)).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            client = ModelClient(tmp)
            client.before_api_attempt = stage.Budget(client, 0).before_attempt
            edit = {"endpoint": "images/edits", "model": IMAGE_MODEL, "prompt": "frozen",
                    "images_sha256": [digest, digest], "n": 1, "size": "1024x1024", "output_format": "png"}
            client._save_png("cached.png", image)
            client._save_entry(request_key(edit), {"image": "cached.png"})
            segment = {"endpoint": "responses", "model": SAM_MODEL, "text": "head", "image_sha256": digest,
                       "metadata": {"include_confidence": "true"}}
            client._save_entry(request_key(segment), {"instances": []})
            self.assertEqual(client.edit_image("frozen", [image, image]).getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(client.segment(image, "head"), [])
            self.assertEqual(client.log.cost_usd(), 0)
            self.assertTrue(all(c["cached"] for c in client.log.calls))


class FakeClient(ModelClient):
    def __init__(self, cache, reject=False, error=None):
        super().__init__(cache)
        self.reject, self.error = reject, error
        self.inputs = []

    def _call(self, model, kind):
        if self.before_api_attempt:
            self.before_api_attempt(model)
        self.log.add(model, kind, "mock-request", False)

    def segment(self, image, text):
        self._call(SAM_MODEL, "segment")
        return []

    def edit_image(self, prompt, images, **kwargs):
        self.inputs.append((prompt, images, kwargs))
        self._call(IMAGE_MODEL, "edit")
        if self.error:
            raise self.error
        return Image.new("RGB", images[0].size, (115, 115, 115)) if self.reject else images[0].copy()


@unittest.skipUnless(HAVE_REFINE, "texture-refine dependencies are required")
class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        (self.run / "03-source.glb").write_bytes(b"source")
        (self.run / "stage-3.json").write_text('{"ok":true}')
        Image.new("RGB", (64, 64), "blue").save(self.run / "01-cover.png")
        self.client = FakeClient(self.run / "cache")
        self.tex = mock.Mock(height=2)
        self.tex.apply.return_value = 17
        self.tex.save.side_effect = lambda path: Path(path)
        self.info = {"height": 2, "body_centre": [0, 0, 1], "hi": [1, 1, 2],
                     "head_centre": [0, 0, 1.8], "head_side": 0.4}
        self.renders = []
        def blender(mode, args, log, **kwargs):
            if mode == "render":
                self.renders.append(args["texture"])
                folder = Path(args["out"])
                folder.mkdir(parents=True, exist_ok=True)
                mask = disc((64, 64), 32, 32, 22)
                rgba = np.full((64, 64, 4), 0, np.uint8)
                rgba[mask] = (60, 80, 100, 255)
                Image.fromarray(rgba).save(folder / (args["cams"][0]["name"] + ".png"))
            if mode == "export":
                Path(args["out"]).write_bytes(b"refined")
        patches = [mock.patch.object(character, "setup", return_value=self.info),
                   mock.patch.object(character, "prep", return_value=self.run / "prep"),
                   mock.patch.object(character, "HEAD_RES", (64, 64)),
                   mock.patch.object(character, "BODY_RES", (64, 64)),
                   mock.patch.object(refine, "Texture", return_value=self.tex),
                   mock.patch("herogen.blender.refine.backend.blender", side_effect=blender),
                   mock.patch.object(cameras, "read_exr", return_value=np.ones((64, 64, 4), np.float32)),
                   mock.patch.object(refine, "register", side_effect=self.identity)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    @staticmethod
    def identity(rgb, alpha, edit):
        yy, xx = np.indices(alpha.shape)
        return xx + .5, yy + .5, edit, {"scale": 1.0}

    def test_success_records_models_cost_views_and_sequential_textures(self):
        rec = stage.run_stage3f(self.run, self.client)
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["outputs"], ["03-refined.glb"])
        self.assertEqual(len(rec["views"]), 20)
        self.assertTrue(all(v["accepted"] and v["tries"] == 1 for v in rec["views"].values()))
        self.assertEqual(set(rec["models"]), {IMAGE_MODEL, SAM_MODEL})
        self.assertGreater(rec["cost_usd"], .3)
        self.assertIn("replayed (cached) calls cost 0", rec["cost_basis"])
        self.assertEqual(final_model(self.run), self.run / "03-refined.glb")
        self.assertNotEqual(self.renders[0], self.renders[1])
        self.assertIn("head+0/texture.png", self.renders[1])
        self.assertTrue(all(len(images) == 2 for _, images, _ in self.client.inputs))
        self.assertTrue(all(images[1].getpixel((0, 0)) == (0, 0, 255) for _, images, _ in self.client.inputs))
        self.assertIsNone(self.client.before_api_attempt)

    def test_all_rejected_is_recorded_fallback_with_attempts(self):
        self.client.reject = True
        (self.run / "03-refined.glb").write_bytes(b"stale")
        rec = stage.run_stage3f(self.run, self.client)
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["outputs"], [])
        self.assertFalse((self.run / "03-refined.glb").exists())
        self.assertEqual(final_model(self.run).name, "03-source.glb")
        self.assertTrue(all(not v["accepted"] and 1 <= v["tries"] <= 3 for v in rec["views"].values()))
        self.assertIn("outline", rec["views"]["head+0"]["reason"])
        self.assertEqual([x[2]["attempt"] for x in self.client.inputs[:3]], [0, 1, 2])

    def test_cap_stops_calls_and_keeps_completed_views(self):
        rec = stage.run_stage3f(self.run, self.client, cap_usd=.02)
        self.assertEqual(rec["status"], "cap_reached")
        self.assertLessEqual(rec["cost_usd"], .02)
        self.assertTrue(rec["views"]["head+0"]["accepted"])
        self.assertFalse(rec["views"]["head+40"]["accepted"])
        self.assertEqual(rec["outputs"], ["03-refined.glb"])
        before = len(self.client.inputs)
        rec2 = stage.run_stage3f(self.run, self.client, cap_usd=.02)
        self.assertEqual(rec2["budget"]["prior_usd"], rec["cost_usd"])
        self.assertEqual(len(self.client.inputs), before + 1)  # blocked before the simulated API call
        self.assertEqual(rec2["cost_usd"], 0)

    def test_model_and_replay_errors_propagate_and_record_spend(self):
        for error in (ApiError("model failed"), ReplayMiss("no entry")):
            with self.subTest(error=type(error).__name__):
                self.client.error = error
                with self.assertRaises(type(error)):
                    stage.run_stage3f(self.run, self.client)
                rec = json.loads((self.run / "stage-3f.json").read_text())
                self.assertFalse(rec["ok"])
                self.assertEqual(rec["status"], "error")
                self.assertEqual(rec["views"]["head+0"]["reason"], type(error).__name__)
                self.assertGreater(rec["cost_usd"], 0)
                self.assertFalse((self.run / "03-refined.glb").exists())

    def test_interrupted_attempt_reservation_survives_a_retry(self):
        self.client.error = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            stage.run_stage3f(self.run, self.client, cap_usd=.01)
        interrupted = json.loads((self.run / "stage-3f.json").read_text())
        self.assertEqual(interrupted["budget"]["spent_usd_total"], .01)
        self.assertEqual(interrupted["budget"]["pending_attempt_usd"], .01)
        self.client.error = None
        rec = stage.run_stage3f(self.run, self.client, cap_usd=.01)
        self.assertEqual(rec["status"], "cap_reached")
        self.assertEqual(rec["cost_usd"], 0)
        self.assertEqual(rec["budget"]["prior_usd"], .01)

    def test_other_profiles_make_no_calls(self):
        rec = stage.run_stage3f(self.run, self.client, profile="roblox-ingame")
        self.assertEqual(rec["status"], "skipped")
        self.assertEqual(self.client.log.calls, [])
        self.tex.apply.assert_not_called()

    def test_no_source_is_skipped(self):
        (self.run / "03-source.glb").unlink()
        rec = stage.run_stage3f(self.run, self.client)
        self.assertEqual(rec["reason"], "no SAM source mesh")
        self.assertEqual(rec["cost_usd"], 0)


class ConsumerTests(unittest.TestCase):
    def test_default_selection_and_roblox_isolation(self):
        from herogen.lab.runindex import bodies, deliverables
        from herogen.service.bundle import glb_path
        from herogen.service.runner import plan_for
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "04-body").mkdir()
            for name in ("04-body/high.glb", "03-source.glb", "03-refined.glb"):
                (run / name).write_bytes(b"synthetic")
            self.assertEqual(final_model(run).name, "03-refined.glb")
            self.assertEqual(glb_path(run).name, "03-refined.glb")
            body = bodies(run)[0]
            self.assertEqual(body["high"], "03-refined.glb")
            self.assertEqual(deliverables(run, body, None)[0]["path"], "03-refined.glb")
            (run / "stage-4.json").write_text('{"profile":{"name":"roblox-ingame"}}')
            self.assertEqual(final_model(run).relative_to(run).as_posix(), "04-body/high.glb")
            self.assertEqual(bodies(run)[0]["high"], "04-body/high.glb")
            self.assertIn("3f", [s.key for s in plan_for("sketch", "best")])
            self.assertNotIn("3f", [s.key for s in plan_for("sketch", "roblox-ingame")])


@unittest.skipUnless(HAVE_REFINE and bl.find_blender(), "Blender and texture-refine dependencies are required")
class BlenderRefineTests(unittest.TestCase):
    def test_prep_render_export_preserves_frame_and_base_colour_image(self):
        from herogen.blender.glb import load_glb
        from herogen.blender.refine.backend import blender
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.glb"
            # Give the synthetic cube a real UV layer and embedded base-colour image inside Blender.
            from herogen.blender import bl as runner
            script = root / "source.py"
            script.write_text('''import bpy
from pathlib import Path
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(location=(0.2, -0.3, 0.7), scale=(0.2, 0.3, 0.7))
o = bpy.context.object
m = bpy.data.materials.new("m")
m.use_nodes = True
i = bpy.data.images.new("colour", 32, 32)
i.generated_color = (0.2, 0.4, 0.7, 1)
i.pack()
t = m.node_tree.nodes.new("ShaderNodeTexImage")
t.image = i
m.node_tree.links.new(t.outputs["Color"], m.node_tree.nodes.get("Principled BSDF").inputs["Base Color"])
o.data.materials.append(m)
bpy.ops.export_scene.gltf(filepath=str(Path(__file__).with_name("source.glb")), export_format="GLB")
''')
            import subprocess
            proc = subprocess.run([runner.find_blender(), "--background", "--factory-startup", "--python-exit-code", "1",
                                   "--python", str(script)], capture_output=True, timeout=120)
            if proc.returncode < 0:
                self.skipTest(f"Blender cannot start in this environment (signal {-proc.returncode})")
            self.assertEqual(proc.returncode, 0, proc.stdout.decode(errors="replace")[-1500:])
            log = root / "blender.log"
            blender("prep", {"glb": str(source), "out": str(root / "prep"), "tex": 4096, "head": None}, log)
            blender("export", {"blend": str(root / "prep/work.blend"), "texture": str(root / "prep/base.png"),
                               "out": str(root / "refined.glb")}, log)
            before, after = load_glb(source), load_glb(root / "refined.glb")
            self.assertEqual(before.triangles, after.triangles)
            for axis in (0, 1, 2):
                np.testing.assert_allclose(np.unique(before.meshes[0].verts[:, axis]),
                                           np.unique(after.meshes[0].verts[:, axis]), atol=1e-6)
            self.assertIn((4096, 4096), after.image_sizes)
            cam = cameras.camera("front", [0.2, -0.3, .7], 0, 2, (128, 128))
            blender("render", {"blend": str(root / "prep/work.blend"), "texture": str(root / "prep/base.png"),
                               "cams": [cam], "out": str(root / "render"), "pos": True}, log)
            self.assertTrue((root / "render/front.png").is_file())
            self.assertTrue((root / "render/front.pos.exr").is_file())


class RunnerTests(unittest.TestCase):
    def test_cli_default_orders_refine_after_mesh_and_before_body(self):
        from herogen.blender.__main__ import main
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "stage-2.json").write_text('{"ok":true}')
            order = []
            result = {"ok": True, "checks": [], "seconds": 0}
            def called(name):
                def fn(*args, **kwargs):
                    order.append((name, kwargs))
                    return result
                return fn
            with mock.patch("herogen.blender.stage3.run_stage3", side_effect=called("3")), \
                    mock.patch.object(stage, "run_stage3f", side_effect=called("3f")), \
                    mock.patch("herogen.blender.stage4.run_stage4", side_effect=called("4")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["run", "--run", str(run), "--refine-cap-usd", "0.42"]), 0)
            self.assertEqual([name for name, _ in order], ["3", "3f", "4"])
            self.assertEqual(order[1][1]["cap_usd"], .42)
            order.clear()
            with mock.patch("herogen.blender.stage3.run_stage3", side_effect=called("3")), \
                    mock.patch.object(stage, "run_stage3f", side_effect=called("3f")), \
                    mock.patch("herogen.blender.stage4.run_stage4", side_effect=called("4")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["run", "--run", str(run), "--profile", "roblox-ingame"]), 0)
            self.assertEqual([name for name, _ in order], ["3", "4"])

    def test_cli_model_error_is_nonzero_without_traceback(self):
        import contextlib
        import io
        from herogen.blender.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "stage-3.json").write_text('{"ok":true}')
            for error in (ApiError("model failure"), ReplayMiss("cache miss")):
                stderr = io.StringIO()
                with mock.patch.object(stage, "run_stage3f", side_effect=error), contextlib.redirect_stderr(stderr):
                    self.assertEqual(main(["run", "--run", str(run), "--stages", "3f"]), 1)
                self.assertIn(type(error).__name__, stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_service_cap_configuration_is_finite_and_defaults_to_best(self):
        from herogen.service.config import ConfigError, from_env
        settings = from_env({}, require_key=False)
        self.assertEqual(settings.default_profile, "best")
        self.assertEqual(settings.refine_cap_usd, .80)
        configured = from_env({"HEROGEN_SERVICE_REFINE_CAP_USD": "0.42"}, require_key=False)
        self.assertEqual(configured.refine_cap_usd, .42)
        for raw in ("nan", "inf", "-1", "bad"):
            with self.assertRaises(ConfigError):
                from_env({"HEROGEN_SERVICE_REFINE_CAP_USD": raw}, require_key=False)

    def test_service_runs_refinement_with_model_key_and_cache(self):
        from herogen.service.runner import PLAN, CliRunner, StageContext
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            ctx = StageContext("j", "sketch", "best", "", "", run, run / "input.png", run / "cache",
                               run / "remote-cache")
            spec = next(s for s in PLAN if s.key == "3f")
            runner = CliRunner("test-placeholder")
            runner._flags["herogen.blender"] = "--profile --refine --refine-cache"
            argv, stdin = runner.command(spec, ctx, {}, run)
            self.assertEqual(argv[argv.index("--stages") + 1], "3f")
            self.assertEqual(argv[argv.index("--refine") + 1], "live")
            self.assertEqual(argv[argv.index("--refine-cache") + 1], str(ctx.cache_dir))
            self.assertIsNone(stdin)
            captured = {}
            def process(*args, **kwargs):
                captured.update(kwargs)
                (run / "stage-3f.json").write_text('{"ok":true}')
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            with mock.patch("herogen.service.runner.subprocess.run", side_effect=process):
                self.assertEqual(runner.run(spec, ctx).state, "passed")
            self.assertEqual(captured["env"]["MODEL_API_KEY"], "test-placeholder")
