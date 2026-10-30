"""Stage 2f, the face source (herogen/pipeline/facesource.py): crops, registration, gates and the fallback.

Every image is drawn here: a figure with a round head, two eyes and a mouth, and close-ups made from its own
crop (shifted, rescaled or replaced by something else) by a stand-in for the Model API. Nothing is sent to a
model and nothing generated is written inside the repository.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

from herogen.pipeline import facesource as F
from herogen.pipeline import imaging
from herogen.pipeline.client import ApiError, CallLog, ReplayMiss, SamInstance
from herogen.pipeline.runfolder import RunFolder

SIZE = 512
HEAD_C, HEAD_R = (256.0, 120.0), 34.0       # the head disk in every view (pixel-edge coordinates)


def draw_view(view: str) -> tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray]:
    """(image, figure mask, face mask, head mask) of a little figure: a body box and a head with features."""
    img = Image.new("RGB", (SIZE, SIZE), (250, 250, 250))
    d = ImageDraw.Draw(img)
    cx, cy = HEAD_C
    d.rectangle((cx - 50, cy + HEAD_R - 4, cx + 50, 470), fill=(40, 70, 160))           # body
    d.ellipse((cx - HEAD_R, cy - HEAD_R, cx + HEAD_R, cy + HEAD_R), fill=(235, 190, 160))
    d.pieslice((cx - HEAD_R, cy - HEAD_R, cx + HEAD_R, cy + HEAD_R), 180, 360, fill=(90, 50, 20))  # hair
    if view == "front":
        for ex in (-12, 12):
            d.ellipse((cx + ex - 5, cy + 2, cx + ex + 5, cy + 10), fill=(20, 20, 20))
        d.rectangle((cx - 9, cy + 18, cx + 9, cy + 22), fill=(170, 30, 40))
    else:
        s = 1 if view == "right" else -1
        d.ellipse((cx + s * 18 - 4, cy + 2, cx + s * 18 + 4, cy + 10), fill=(20, 20, 20))
        d.polygon([(cx + s * HEAD_R, cy + 8), (cx + s * (HEAD_R + 8), cy + 14), (cx + s * HEAD_R, cy + 18)],
                  fill=(225, 175, 150))
    a = np.asarray(img).astype(int)
    fig = np.abs(a - 250).max(axis=2) > 10
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] + 0.5
    head = ((xx - cx) ** 2 + (yy - cy) ** 2 <= HEAD_R ** 2) & fig
    face = head & (yy > cy - 2) & (np.abs(xx - cx) < 26)
    return img, fig, face, head


def write_run(root: Path, parts: bool = True) -> RunFolder:
    run = RunFolder(root, "r")
    for v in F.VIEWS:
        img, fig, face, head = draw_view(v)
        img.save(run.file(f"views/{v}.png"))
        imaging.to_mask_image(fig).save(run.file(f"masks/{v}.png"))
        if v == "front" and parts:
            imaging.to_mask_image(face).save(run.file("masks/front-face.png"))
            imaging.to_mask_image(head).save(run.file("masks/front-head.png"))
    return run


def upscale_from(crop: Image.Image, shift_px: float = 0.0, scale: float = 1.0, out: int = 640) -> Image.Image:
    """A 'close-up': the crop resampled to ``out`` px, optionally shifted (crop pixels) and zoomed about its
    centre, as a model that slightly misplaces its output would."""
    a = np.asarray(crop, dtype=np.float64)
    n = a.shape[0]
    k = n / out
    # close-up pixel centre -> crop coordinates
    M = np.array([[k / scale, 0.0, n / 2 - (out / 2) * k / scale - shift_px],
                  [0.0, k / scale, n / 2 - (out / 2) * k / scale - 0.5 * shift_px]])
    return Image.fromarray(np.clip(F.warp(a, M, (out, out)), 0, 255).astype(np.uint8))


class FakeClient:
    """The two calls stage 2f makes: an edit (answered by ``answer`` from the input crop) and SAM `head`
    (the head disk). ``fail`` raises the Model API's error instead."""

    def __init__(self, answer=upscale_from, fail: bool = False):
        self.log = CallLog()
        self.answer = answer
        self.fail = fail
        self.prompts: list[str] = []

    def edit_image(self, prompt, images, *, size="1024x1024", attempt=0):
        if self.fail:
            raise ApiError("images/edits failed: HTTP 500")
        self.prompts.append(prompt)
        self.log.add("muse-image-1.0", "edit", str(len(self.prompts)), cached=False)
        return self.answer(images[0])

    def segment(self, image, text):
        self.log.add("sam-3.1", "segment", text, cached=False)
        _, _, _, head = draw_view("left")
        return [SamInstance(mask=head, box=imaging.bbox(head), confidence=0.9)]


class TestCrops(unittest.TestCase):
    def test_front_crop_is_a_square_around_face_and_head(self):
        face, head = (230.0, 118.0, 282.0, 154.0), (222.0, 86.0, 290.0, 154.0)
        box = F.front_crop_box(face, head)
        l, t, r, b = box
        self.assertAlmostEqual(r - l, b - t)
        self.assertLessEqual(l, face[0])
        self.assertGreaterEqual(r, face[2])
        self.assertLess(t, head[1])                       # the head top, plus the top padding
        self.assertAlmostEqual(b, t + (r - l))
        self.assertAlmostEqual(b - (154 + 0.45 * 36), (r - l) * (1 - F.CROP_TOP_PAD) - (154 + 0.45 * 36 - 86),
                               places=6)

    def test_head_box_far_above_the_face_is_bounded(self):
        face = (230.0, 118.0, 282.0, 154.0)                # 36 px tall
        tall = (222.0, 0.0, 290.0, 154.0)                  # a spire or a raised arm read as head
        l, t, r, b = F.front_crop_box(face, tall)
        self.assertGreaterEqual(t + F.CROP_TOP_PAD * (r - l), 118 - 1.2 * 36 - 1e-9)

    def test_no_face_uses_the_head_box(self):
        box = F.front_crop_box(None, (200.0, 80.0, 300.0, 180.0))
        self.assertAlmostEqual(box[2] - box[0], 100 * (1 + 2 * F.CROP_MARGIN))
        spire = F.front_crop_box(None, (240.0, 0.0, 280.0, 160.0))   # much taller than wide: sits on its bottom
        self.assertAlmostEqual(spire[3], 160 + 4.0)
        self.assertIsNone(F.front_crop_box(None, None))

    def test_profile_crop_follows_the_baseline(self):
        _, ffig, _, _ = draw_view("front")
        _, sfig, _, _ = draw_view("left")
        fbox = (200.0, 70.0, 312.0, 182.0)
        shifted = np.zeros_like(sfig)
        shifted[:-6] = sfig[6:]                            # the side figure stands 6 px higher
        box = F.profile_crop_box(fbox, ffig, shifted)
        self.assertAlmostEqual(box[1], 70.0 - 6)
        self.assertGreaterEqual(box[2] - box[0], 112.0)
        self.assertIsNone(F.profile_crop_box(fbox, ffig, np.zeros_like(sfig)))

    def test_crop_outside_the_image_takes_the_border_colour(self):
        img, _, _, _ = draw_view("front")
        c = np.asarray(F.crop(img, (-100.0, -100.0, 100.0, 100.0), 64))
        self.assertEqual(tuple(c[2, 2]), (250, 250, 250))
        self.assertEqual(c.shape, (64, 64, 3))


class TestRegistration(unittest.TestCase):
    def setUp(self):
        img, fig, face, head = draw_view("front")
        self.view = np.asarray(img, dtype=np.float64)
        self.face, self.head, self.fig = face, head, fig
        self.box = F.front_crop_box(imaging.bbox(face), imaging.bbox(head))
        self.crop = F.crop(img, self.box, F.INPUT_SIDE)
        self.region = imaging.dilate(head, 4)

    def corner_error(self, A, shift_px, scale=1.0):
        """Largest distance (view px) between A's and the true transform's images of the close-up's corners."""
        S, side = 640, self.box[2] - self.box[0]
        l, t = self.box[:2]
        # true: close-up px -> crop px (inverse of upscale_from's map) -> view px
        n = F.INPUT_SIDE
        kc = n / S
        Mi = np.linalg.inv(np.array([[kc / scale, 0, n / 2 - (S / 2) * kc / scale - shift_px],
                                     [0, kc / scale, n / 2 - (S / 2) * kc / scale - 0.5 * shift_px], [0, 0, 1]]))
        crop_to_view = np.array([[side / n, 0, l], [0, side / n, t], [0, 0, 1]])
        true = (crop_to_view @ np.linalg.inv(Mi))[:2]
        corners = np.array([[0, 0, 1], [S, 0, 1], [0, S, 1], [S, S, 1]], float).T
        return float(np.abs(np.asarray(A) @ corners - true @ corners).max())

    def test_exact_upscale_keeps_the_crop_box(self):
        cu = np.asarray(upscale_from(self.crop), dtype=np.float64)
        r = F.register(cu, self.view, self.box, self.region, self.face)
        self.assertGreater(r["ncc_view"], 0.97)
        self.assertLess(self.corner_error(r["A"], 0.0), 0.5)
        self.assertAlmostEqual(r["scale_ratio"], 1.0, delta=0.01)

    def test_a_shifted_close_up_is_brought_back(self):
        shift = 24.0                                       # crop px: about 3 view px
        cu = np.asarray(upscale_from(self.crop, shift_px=shift, scale=1.03), dtype=np.float64)
        r = F.register(cu, self.view, self.box, self.region, self.face)
        self.assertTrue(r["used_fit"], r)
        self.assertGreater(r["ncc_work"]["fitted"], r["ncc_work"]["crop_box"])
        self.assertLess(self.corner_error(r["A"], shift, 1.03), 0.6)
        self.assertGreater(r["ncc_view"], 0.97)

    def test_another_picture_fails_the_gate(self):
        other = np.asarray(upscale_from(self.crop), dtype=np.float64)[::-1]        # upside down
        r = F.register(other, self.view, self.box, self.region, self.face)
        self.assertLess(r["ncc_view"], F.NCC_MIN["front"])


class TestStage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_close_ups_are_made_registered_and_recorded(self):
        run = write_run(self.root)
        client = FakeClient()
        rec = F.stage2f_face_source(run, client)
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["stage"], "2f")
        self.assertEqual(rec["details"]["used_views"], ["front", "left", "right"])
        self.assertEqual(client.prompts, [F.PROMPTS["FC4"], F.PROMPTS["PC4"], F.PROMPTS["PC4"]])
        saved = json.loads((run.path / F.RECORD).read_text())
        for v in F.VIEWS:
            r = saved["views"][v]
            self.assertTrue(r["used"], r)
            self.assertGreaterEqual(r["ncc_view"], F.NCC_MIN[v])
            for key in ("file", "input", "footprint"):
                self.assertTrue((run.path / r[key]).exists(), r[key])
            with Image.open(run.path / r["footprint"]) as im:
                foot = np.asarray(im) > 127
            self.assertTrue(foot.any())
            self.assertFalse((foot & ~F.box_mask(foot.shape, r["crop_box"])).any())    # inside the crop
        self.assertTrue((self.root / "r" / "stage-2f.json").exists())
        self.assertAlmostEqual(rec["cost_usd"], 3 * 0.01 + 2 * 0.0025)
        # the views stage 2 wrote are untouched
        img, _, _, _ = draw_view("front")
        self.assertTrue(np.array_equal(np.asarray(Image.open(run.path / "views/front.png")), np.asarray(img)))

    def test_a_close_up_that_does_not_match_is_not_used(self):
        run = write_run(self.root)
        rec = F.stage2f_face_source(run, FakeClient(answer=lambda c: upscale_from(c).rotate(180)))
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["details"]["used_views"], [])
        self.assertIn("NCC in the view", rec["details"]["closeups"]["front"]["reason"])
        self.assertFalse(list((run.path / "views").glob("closeup-*")))
        # every close-up gated out: the record is still written, with each view's reason for stage 4
        saved = json.loads((run.path / F.RECORD).read_text())
        self.assertEqual(saved["used_views"], [])
        for v in F.VIEWS:
            self.assertFalse(saved["views"][v]["used"])
            self.assertIn("NCC in the view", saved["views"][v]["reason"])
        self.assertIn(F.RECORD, rec["outputs"])
        self.assertIn("plain views", rec["notes"])

    def test_a_reframed_close_up_is_not_used(self):
        run = write_run(self.root)
        rec = F.stage2f_face_source(run, FakeClient(answer=lambda c: upscale_from(c, scale=0.6)))
        reasons = [r.get("reason", "") for r in rec["details"]["closeups"].values()]
        self.assertTrue(any("reframed" in x or "NCC" in x for x in reasons), reasons)
        self.assertNotIn("front", rec["details"]["used_views"])

    def test_a_model_failure_falls_back_and_never_raises(self):
        run = write_run(self.root)
        F.stage2f_face_source(run, FakeClient())                    # an earlier, successful run
        rec = F.stage2f_face_source(run, FakeClient(fail=True))
        self.assertTrue(rec["ok"])
        self.assertIn("ApiError", rec["details"]["fallback"])
        self.assertEqual(rec["details"]["used_views"], [])
        saved = json.loads((run.path / F.RECORD).read_text())          # nothing stale for stage 4 to read
        self.assertEqual((saved["views"], saved["used_views"]), ({}, []))
        self.assertIn("ApiError", saved["fallback"])
        self.assertFalse(list((run.path / "views").glob("closeup-*")))
        self.assertFalse(list((run.path / "masks").glob("closeup-*")))

    def test_a_replay_miss_is_raised(self):
        # a replay with no cached answer is a broken cache, not a model failure: it stops the run
        run = write_run(self.root)
        F.stage2f_face_source(run, FakeClient())                    # an earlier, successful run
        with mock.patch.object(F, "make_face_source", side_effect=ReplayMiss("no cached response for key k")):
            with self.assertRaises(ReplayMiss):
                F.stage2f_face_source(run, FakeClient())
        self.assertFalse(list((run.path / "views").glob("closeup-*")))   # nothing stale left behind
        rec = json.loads((run.path / "stage-2f.json").read_text())       # the earlier record is replaced
        self.assertEqual((rec["ok"], rec["status"]), (False, "error"))
        self.assertIn("ReplayMiss", rec["error"])
        self.assertEqual((rec["outputs"], rec["details"]["used_views"]), ([], []))

    def test_results_show_a_replay_miss_as_failed(self):
        from herogen.pipeline.results import collect

        run = write_run(self.root)
        self.assertTrue(F.stage2f_face_source(run, FakeClient())["ok"])
        with mock.patch.object(F, "make_face_source", side_effect=ReplayMiss("no cached response for key k")):
            with self.assertRaises(ReplayMiss):
                F.stage2f_face_source(run, FakeClient())
        row = [s for s in collect(run)["stages"] if s["stage"] == "2f"][0]
        self.assertNotEqual(row["status"], "pass")
        self.assertFalse(any("closeup" in str(x) for x in row.get("outputs") or row.get("images") or []))

    def test_the_cli_reports_a_fallback_on_stderr_and_goes_on(self):
        from herogen.pipeline.__main__ import run_pipeline

        write_run(self.root)
        err, out = io.StringIO(), io.StringIO()
        with mock.patch.object(F, "make_face_source", side_effect=ApiError("images/edits failed: HTTP 500")),                 contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            status = run_pipeline(None, "r", ["2f"], mode="replay", runs=str(self.root))
        self.assertEqual(status, 0)
        self.assertIn("stage 2f face source fell back: ApiError: images/edits failed: HTTP 500", err.getvalue())
        self.assertIn("fell back to the plain views", out.getvalue())
        self.assertNotIn("close-ups used for", out.getvalue())

    def test_the_cli_stops_on_a_replay_miss(self):
        from herogen.pipeline.__main__ import run_pipeline

        write_run(self.root)
        with mock.patch.object(F, "make_face_source", side_effect=ReplayMiss("no cached response for key k")),                 contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_pipeline(None, "r", ["2f"], mode="replay", runs=str(self.root)), 1)

    def test_no_part_masks_means_no_crop(self):
        run = write_run(self.root, parts=False)
        rec = F.stage2f_face_source(run, FakeClient())
        self.assertTrue(rec["ok"])
        self.assertIn("no face or head mask", rec["details"]["fallback"])

    def test_missing_stage_2_is_recorded(self):
        run = RunFolder(self.root, "empty")
        rec = F.stage2f_face_source(run, FakeClient())
        self.assertTrue(rec["ok"])
        self.assertIn("FileNotFoundError", rec["details"]["fallback"])


class TestHelpers(unittest.TestCase):
    def test_blur_keeps_a_constant_and_spreads_a_dot(self):
        flat = np.full((20, 20), 7.0)
        self.assertTrue(np.allclose(F.blur(flat, 2.0), 7.0))
        dot = np.zeros((21, 21))
        dot[10, 10] = 1
        b = F.blur(dot, 1.5)
        self.assertAlmostEqual(b.sum(), 1.0, places=6)
        self.assertAlmostEqual(b[10, 12] / b[10, 10], math.exp(-0.5 * (2 / 1.5) ** 2), places=2)

    def test_ncc(self):
        a = np.random.default_rng(0).random((30, 30))
        m = np.ones_like(a, bool)
        self.assertAlmostEqual(F.ncc(a, 3 * a + 1, m), 1.0, places=6)
        self.assertAlmostEqual(F.ncc(a, -a, m), -1.0, places=6)
        self.assertEqual(F.ncc(a, np.zeros_like(a), m), 0.0)


if __name__ == "__main__":
    unittest.main()
