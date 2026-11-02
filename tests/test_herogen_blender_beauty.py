"""Synthetic beauty-reference gates, resumption and camera selection."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from herogen.blender.refine import beauty
from herogen.pipeline.client import IMAGE_MODEL, SAM_MODEL, ApiError, CallLog, ReplayMiss


class BeautyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run, self.out = self.root / "run", self.root / "refs"
        for name in ("views", "masks"):
            (self.run / name).mkdir(parents=True)
        self.mask = np.zeros((64, 64), bool)
        self.mask[8:56, 20:44] = True
        self.mask[20:40, 10:20] = True
        for name in beauty.STAGE2.values():
            Image.new("RGB", (64, 64), "white").save(self.run / f"views/{name}.png")
            Image.fromarray(self.mask.astype(np.uint8) * 255).save(self.run / f"masks/{name}.png")
        Image.new("RGB", (64, 64), "blue").save(self.run / "01-cover.png")
        rgba = np.full((64, 64, 4), 200, np.uint8)
        rgba[..., 3] = self.mask * 255
        Image.fromarray(rgba).save(self.root / "guide.png")
        self.guides = {a: self.root / "guide.png" for a in (45, -45)}
        self.client = SimpleNamespace(
            edit_image=mock.Mock(return_value=Image.new("RGB", (64, 64), "red")),
            segment=mock.Mock(side_effect=lambda image, text: [SimpleNamespace(mask=self.mask)]),
            log=SimpleNamespace(calls=[], cost_usd=lambda start: 0))

    def generate(self):
        return beauty.generate(self.run, self.client, self.out, self.guides)

    def test_frozen_prompt_and_wrapped_nearest_yaw(self):
        self.assertEqual(hashlib.sha256(beauty.BT1.encode()).hexdigest(),
                         "9ec676576f09c9c266071edd3114ea3eea3c2453ca254349385b5c9fd87877e8")
        self.assertEqual(beauty.nearest(40), 45)
        self.assertEqual(beauty.nearest(270), -90)
        self.assertEqual(beauty.nearest(360), 0)

    def test_facial_features_are_measured_only_inside_the_face(self):
        face = np.zeros((64, 64), bool)
        face[10:30, 20:40] = True
        face[10:15, 20:25] = False
        calls = []

        def segment(image, text):
            calls.append((image.size, text))
            if text == "face":
                return [SimpleNamespace(mask=face)]
            mask = np.zeros(image.size[::-1], bool)
            if text == "eyes":
                mask[8:10, 8:10] = True
            elif text == "mouth":
                mask[:3, :3] = True  # outside the actual face despite lying in its bounding box
            return [SimpleNamespace(mask=mask)]

        self.client.segment.side_effect = segment
        got = beauty.facial_features(self.client, Image.new("RGB", (64, 64)))
        self.assertEqual(got, {"eyes": True, "eyebrows": False, "mouth": False})
        self.assertEqual(calls, [((64, 64), "face"), ((20, 20), "eyes"),
                                 ((20, 20), "eyebrows"), ((20, 20), "mouth")])

    def test_facial_features_skip_covered_faces_and_propagate_errors(self):
        image = Image.new("RGB", (64, 64))
        self.client.segment.return_value = []
        self.client.segment.side_effect = None
        self.assertFalse(any(beauty.facial_features(self.client, image).values()))
        self.assertEqual(self.client.segment.call_count, 1)
        for error in (ApiError("failed"), ReplayMiss("missing")):
            self.client.segment.side_effect = error
            with self.assertRaises(type(error)):
                beauty.facial_features(self.client, image)

    def test_accepts_six_views_and_resumes_without_paid_calls(self):
        rec = self.generate()
        self.assertTrue(all(v["accepted"] for v in rec["views"].values()))
        self.assertEqual([rec["views"][str(a)]["threshold"] for a in beauty.ANGLES],
                         [.9, .85, .85, .9, .9, .9])
        self.assertEqual(self.client.edit_image.call_count, 6)
        self.generate()
        self.assertEqual(self.client.edit_image.call_count, 6)
        self.assertEqual(self.client.segment.call_count, 12)
        loaded = beauty.load(self.out)
        crop, selected = beauty.reference(self.out, loaded, -40, head=True)
        self.assertEqual(selected["yaw"], -45)
        self.assertLess(crop.width, 64)
        self.assertEqual(len(selected["image_sha256"]), 64)

    def test_rejects_after_three_attempts_and_never_substitutes(self):
        self.client.segment.side_effect = lambda image, text: []
        rec = self.generate()
        self.assertEqual(self.client.edit_image.call_count, 18)
        self.assertTrue(all(not v["accepted"] and len(v["attempts"]) == 3 for v in rec["views"].values()))
        self.generate()
        self.assertEqual(self.client.edit_image.call_count, 18)
        self.assertTrue(all(v is None for v in beauty.load(self.out)["fallbacks"].values()))

    def test_model_and_replay_errors_propagate_and_checkpoint(self):
        for error in (ApiError("failed"), ReplayMiss("missing")):
            with self.subTest(error=type(error).__name__):
                self.client.edit_image.side_effect = error
                with self.assertRaises(type(error)):
                    self.generate()
                rec = json.loads((self.out / "references.json").read_text())
                self.assertFalse(rec["views"]["0"]["accepted"])
                self.assertEqual(rec["error_type"], type(error).__name__)

    def test_paid_image_with_failed_segmentation_reuses_same_attempt(self):
        self.client.segment.side_effect = ApiError("segmentation failed")
        with self.assertRaises(ApiError):
            self.generate()
        self.client.segment.side_effect = lambda image, text: [SimpleNamespace(mask=self.mask)]
        self.generate()
        attempts = [call.kwargs["attempt"] for call in self.client.edit_image.call_args_list]
        self.assertEqual(attempts[:2], [0, 0])
        self.assertEqual(len(attempts), 7)  # the real client serves the first image from its request cache

    def test_reference_hash_and_facing_are_rechecked(self):
        self.generate()
        path = self.out / "references.json"
        rec = json.loads(path.read_text())
        # Mirroring cannot flip the facing of a front or back view, so a mirrored saved mask
        # stays usable at 0 and 180; the same mask is wrong-facing at a lateral yaw.
        for yaw in (0, 180):
            Image.fromarray(self.mask[:, ::-1].astype(np.uint8) * 255).save(self.out / f"mask{yaw:+d}-try1.png")
        loaded = beauty.load(self.out)
        self.assertTrue(loaded["views"]["0"]["accepted"])
        self.assertTrue(loaded["views"]["180"]["accepted"])
        self.assertEqual(loaded["fallbacks"]["0"], 0)
        Image.fromarray(self.mask[:, ::-1].astype(np.uint8) * 255).save(self.out / "mask+45-try1.png")
        loaded = beauty.load(self.out)
        self.assertFalse(loaded["views"]["45"]["accepted"])
        self.assertFalse(loaded["views"]["45"]["attempts"][0]["correct_facing"])
        self.assertEqual(loaded["fallbacks"]["45"], 0)
        rec["sha256"] = "changed"
        path.write_text(json.dumps(rec))
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.generate()

    def test_diagonal_threshold_does_not_loosen_cardinal_gate(self):
        near = self.mask.copy()
        near[8:14] = False  # IoU 0.875
        self.client.segment.side_effect = lambda image, text: [SimpleNamespace(mask=near)]
        rec = self.generate()
        self.assertFalse(rec["views"]["0"]["accepted"])
        self.assertTrue(rec["views"]["45"]["accepted"])
        self.assertLess(rec["views"]["0"]["attempts"][0]["iou"], .9)

    def test_style_selection_best_valid_mirror_tie_and_empty_fallback(self):
        rec = self.generate()
        v = rec["views"]["45"]
        v["attempts"].append({"try": 2, "image": v["image"], "iou": 1})
        near = self.mask.copy()
        near[8:10] = False
        Image.fromarray(near.astype(np.uint8) * 255).save(self.out / "mask+45-try1.png")
        Image.fromarray(self.mask.astype(np.uint8) * 255).save(self.out / "mask+45-try2.png")
        (self.out / "references.json").write_text(json.dumps(rec))
        self.assertEqual(beauty.load(self.out)["views"]["45"]["mask"], "mask+45-try2.png")
        Image.fromarray(self.mask[:, ::-1].astype(np.uint8) * 255).save(self.out / "mask+45-try2.png")
        loaded = beauty.load(self.out)
        self.assertEqual(loaded["views"]["45"]["mask"], "mask+45-try1.png")
        self.assertFalse(loaded["views"]["45"]["attempts"][1]["correct_facing"])
        for yaw in beauty.ANGLES:
            Image.new("L", (64, 64), 255).save(self.out / f"expected{yaw:+d}.png")
        loaded = beauty.load(self.out)
        self.assertIsNone(beauty.selected_yaw(loaded, 0))
        with self.assertRaisesRegex(ValueError, "standard recipe"):
            beauty.reference(self.out, loaded, 0)

    def test_centroid_alignment_removes_offset_from_the_facing_test(self):
        self.out.mkdir()
        body = np.zeros((64, 64), bool)
        body[10:50, 24:40] = True
        arm = np.zeros((64, 64), bool)
        arm[24:34, 12:22] = True
        expected = body | arm
        Image.fromarray(expected.astype(np.uint8) * 255).save(self.out / "expected+45.png")
        # Correctly facing but sitting 2 px off-centre, as a retained prop shifts a silhouette.
        Image.fromarray(beauty._translate(expected, 0, 2).astype(np.uint8) * 255).save(
            self.out / "mask+45-try1.png")
        Image.new("RGB", (64, 64)).save(self.out / "candidate.png")
        rec = {"sha256": beauty.BT1_SHA256, "views": {"45": {
            "image": "candidate.png", "head_box": [0, 0, 20, 20],
            "attempts": [{"try": 1, "image": "candidate.png"}]}}}
        (self.out / "references.json").write_text(json.dumps(rec))
        trial = beauty.load(self.out)["views"]["45"]["attempts"][0]
        self.assertTrue(trial["style_usable"])
        self.assertGreater(trial["facing_iou"], trial["style_iou"])
        self.assertAlmostEqual(trial["facing_iou"], 1.0)
        self.assertGreater(trial["facing_iou"], trial["mirror_iou"])

    def test_head_crop_extends_into_the_neck_and_clamps_to_the_image(self):
        rec = self.generate()
        rec["views"]["0"]["head_box"] = [10, 10, 30, 30]
        (self.out / "references.json").write_text(json.dumps(rec))
        loaded = beauty.load(self.out)
        self.assertEqual(loaded["views"]["0"]["head_box"], [10, 10, 30, 35])
        crop, _ = beauty.reference(self.out, loaded, 0, head=True)
        self.assertEqual(crop.size, (20, 25))
        rec["views"]["0"]["head_box"] = [10, 52, 30, 62]
        (self.out / "references.json").write_text(json.dumps(rec))
        self.assertEqual(beauty.load(self.out)["views"]["0"]["head_box"], [10, 52, 30, 64])

    def test_rear_quarter_cameras_never_receive_face_crops(self):
        self.generate()
        loaded = beauty.load(self.out)
        self.assertEqual(beauty.nearest(135), 90)  # the tie that used to bind the profile crop
        for angle, yaw in ((135, 180), (-135, 180), (150, 180), (180, 180), (90, 90), (0, 0)):
            self.assertEqual(beauty.selected_yaw(loaded, angle), yaw)
        _, info = beauty.reference(self.out, loaded, -135, head=True)
        self.assertEqual((info["yaw"], info["fallback"]), (180, True))
        loaded["views"]["180"]["accepted"] = False
        self.assertIsNone(beauty.selected_yaw(loaded, 135))
        with self.assertRaisesRegex(ValueError, "standard recipe"):
            beauty.reference(self.out, loaded, 135)
        self.assertEqual(beauty.selected_yaw(loaded, 90), 90)

    def test_style_iou_boundary_is_inclusive_and_not_below(self):
        self.out.mkdir()
        expected = np.zeros((64, 64), np.uint8)
        expected[10:20, 5:15] = 255
        mask = expected.copy()
        mask[17:20] = 0
        Image.fromarray(expected).save(self.out / "expected+0.png")
        Image.fromarray(mask).save(self.out / "mask+0-try1.png")
        Image.new("RGB", (64, 64)).save(self.out / "candidate.png")
        rec = {"sha256": beauty.BT1_SHA256, "views": {"0": {
            "image": "candidate.png", "head_box": [0, 0, 20, 20],
            "attempts": [{"try": 1, "image": "candidate.png"}]}}}
        (self.out / "references.json").write_text(json.dumps(rec))
        loaded = beauty.load(self.out)
        self.assertEqual(loaded["views"]["0"]["attempts"][0]["style_iou"], .7)
        self.assertTrue(loaded["views"]["0"]["accepted"])
        mask[10, 5] = 0
        Image.fromarray(mask).save(self.out / "mask+0-try1.png")
        self.assertFalse(beauty.load(self.out)["views"]["0"]["accepted"])

    def test_missing_and_invalid_selected_head_crop_are_loud(self):
        rec = self.generate()
        path = self.out / "references.json"
        rec["views"]["0"].pop("head_box")
        path.write_text(json.dumps(rec))
        with self.assertRaisesRegex(ValueError, "no head crop"):
            beauty.load(self.out)
        rec["views"]["0"]["head_box"] = [0, 0, 1000, 5]
        path.write_text(json.dumps(rec))
        with self.assertRaisesRegex(ValueError, "invalid head crop"):
            beauty.load(self.out)

    def test_returned_resolution_keeps_palette_mask_at_image_size(self):
        self.client.edit_image.return_value = Image.new("RGB", (128, 128), "red")
        mask = np.repeat(np.repeat(self.mask, 2, 0), 2, 1)
        self.client.segment.side_effect = lambda image, text: [SimpleNamespace(mask=mask)]
        rec = self.generate()
        front = rec["views"]["0"]
        self.assertEqual(front["attempts"][0]["iou"], 1)
        with Image.open(self.out / front["mask"]) as image:
            self.assertEqual(image.size, (128, 128))


class FastClient:
    """A cache-backed client shape that `isolated_client` can copy; each yaw draws its own image."""

    def __init__(self, cache_dir, mask, *, unusable=(), headless=(), failing=(), failing_sam=(), barrier=None,
                 error=None, once=False):
        self.cache_dir, self.mask, self.log = Path(cache_dir), mask, CallLog()
        self.unusable, self.headless, self.failing, self.barrier = unusable, headless, failing, barrier
        self.failing_sam, self.error = failing_sam, error or ApiError("edit failed")
        self.once, self.failed = once, set()   # once: each failing request fails only on its first send
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.edits, self.segments = [], []

    def fails(self, request):
        if self.once and request in self.failed:
            return False
        self.failed.add(request)
        return True

    def _load(self, key):
        return None

    def _save_entry(self, key, entry):
        pass

    @staticmethod
    def yaw(prompt):
        return int(prompt.split("at yaw ", 1)[1].split(" ", 1)[0])

    def edit_image(self, prompt, images, *, size, attempt):
        yaw = self.yaw(prompt)
        self.edits.append((yaw, attempt))
        self.log.add(IMAGE_MODEL, "edit", f"edit{yaw}", False)
        if self.barrier is not None:
            self.barrier.wait()  # every yaw must be in flight at once
        if yaw in self.failing and self.fails(("edit", yaw)):
            raise self.error
        return Image.new("RGB", images[0].size, (100 + yaw // 2, 50, 50))

    def segment(self, image, text):
        yaw = (image.getpixel((0, 0))[0] - 100) * 2
        self.segments.append((yaw, text))
        self.log.add(SAM_MODEL, "segment", f"{text}{yaw}", False)
        if (yaw, text) in self.failing_sam and self.fails((text, yaw)):
            raise self.error
        if text == "character" and yaw in self.unusable:
            return []
        if text == "head":
            if yaw in self.headless:
                return []
            head = np.zeros(self.mask.shape, bool)
            head[8:20, 24:40] = True
            return [SimpleNamespace(mask=head)]
        return [SimpleNamespace(mask=self.mask)]


class FastGenerationTests(unittest.TestCase):
    setUp = BeautyTests.setUp

    def fast(self, **kwargs):
        client = FastClient(self.root / "cache", self.mask, **kwargs)
        self.fast_client = client
        return beauty.generate_fast(self.run, client, self.out), client

    def test_one_sample_per_cardinal_in_parallel_without_diagonal_guides(self):
        import threading
        rec, client = self.fast(barrier=threading.Barrier(4, timeout=10))
        self.assertEqual(sorted(client.edits), sorted((a, 0) for a in beauty.FAST_ANGLES))
        self.assertEqual(sorted(client.segments), sorted((a, t) for a in beauty.FAST_ANGLES
                                                         for t in ("character", "head")))
        self.assertEqual((rec["angles"], rec["attempts_per_angle"], rec["fallback"]),
                         ([0, 90, -90, 180], 1, "none"))
        self.assertEqual(rec["selected"], {str(a): True for a in beauty.FAST_ANGLES})
        self.assertEqual(rec["status"], "complete")
        # The isolated calls reach the caller's ledger: four images and eight SAM queries.
        self.assertEqual(client.log.cost_usd(), 4 * .01 + 8 * .0025)
        self.assertEqual(rec["cost_usd"], client.log.cost_usd())
        self.assertEqual([rec["views"][str(a)]["threshold"] for a in beauty.FAST_ANGLES], [.9] * 4)
        loaded = beauty.load(self.out)
        for angle in beauty.FAST_ANGLES:
            self.assertEqual(beauty.selected_yaw(loaded, angle), angle)
        self.assertTrue(json.loads((self.out / "references.json").read_text())["selected"]["0"])

    def test_rejected_yaw_is_not_retried_and_keeps_the_standard_recipe(self):
        rec, client = self.fast(unusable=(90,))
        self.assertEqual(sorted(client.edits), sorted((a, 0) for a in beauty.FAST_ANGLES))
        self.assertNotIn((90, "head"), client.segments)
        self.assertFalse(rec["selected"]["90"])
        self.assertEqual(rec["status"], "rejected")
        loaded = beauty.load(self.out)
        self.assertIsNone(beauty.selected_yaw(loaded, 90))
        with self.assertRaisesRegex(ValueError, "standard recipe"):
            beauty.reference(self.out, loaded, 90)
        self.assertEqual([beauty.selected_yaw(loaded, a) for a in (0, -90, 180)], [0, -90, 180])
        # Saved references without the fast record still borrow the nearest usable yaw.
        loaded.pop("fallback")
        self.assertEqual(beauty.selected_yaw(loaded, 90), 0)

    def test_selected_sample_without_a_head_is_unusable(self):
        rec, _ = self.fast(headless=(180,))
        self.assertFalse(rec["views"]["180"]["attempts"][0]["head_detected"])
        self.assertFalse(rec["selected"]["180"])
        self.assertIsNone(beauty.selected_yaw(beauty.load(self.out), 180))

    # The three model errors seen in held-out runs: a read timeout, an HTTP 500 and a content-policy 400.
    TIMEOUT = ApiError("images/edits failed after 1 attempts: TimeoutError: The read operation timed out",
                       network_error="TimeoutError")
    SERVER = ApiError("images/edits failed after 1 attempts: HTTP 500: internal server error", http_status=500)
    POLICY = ApiError("images/edits failed: HTTP 400: filtered (not retried)", http_status=400,
                      error_code="content_policy_violation")
    ERRORS = (TIMEOUT, SERVER, POLICY)

    @staticmethod
    def fields(error):
        return {"error": "ApiError", "http_status": error.http_status, "error_code": error.error_code,
                "network_error": error.network_error, "message": str(error)}

    def test_model_error_on_a_yaw_leaves_it_without_a_reference(self):
        for error in self.ERRORS:
            with self.subTest(error=str(error)):
                rec, client = self.fast(failing=(-90,), error=error)
                resent = error is not self.POLICY   # a transient error is resent once; a 4xx never
                self.assertEqual(sorted(client.edits), sorted([(a, 0) for a in beauty.FAST_ANGLES]
                                                              + [(-90, 0)] * resent))
                self.assertNotIn(-90, [yaw for yaw, _ in client.segments])
                saved = json.loads((self.out / "references.json").read_text())
                expected = {"request": "BT1 image edit", **self.fields(error),
                            **({"retried": True} if resent else {}), "fallback": "standard recipe"}
                self.assertEqual(saved["views"]["-90"]["error"], expected)
                self.assertEqual(saved["views"]["-90"]["attempts"], [])
                self.assertEqual(saved["contained_errors"], [{"yaw": -90, **expected}])
                retries = [{"yaw": -90, "request": "image edit", **self.fields(error), "retry": self.fields(error)}]
                self.assertEqual(saved["retries"], retries if resent else [])
                self.assertEqual(saved["views"]["-90"].get("retries"),
                                 [{k: v for k, v in retries[0].items() if k != "yaw"}] if resent else None)
                self.assertEqual((saved["status"], saved["selected"]["-90"]), ("rejected", False))
                self.assertNotIn("error_type", saved)
                loaded = beauty.load(self.out)
                self.assertIsNone(beauty.selected_yaw(loaded, -90))
                self.assertEqual([beauty.selected_yaw(loaded, a) for a in (0, 90, 180)], [0, 90, 180])
                # No substitute sample: one image per yaw, plus the single resend of a transient error.
                self.assertEqual(sum(c["kind"] == "edit" for c in client.log.calls), 4 + resent)
                self.assertEqual(rec["cost_usd"], client.log.cost_usd())

    def test_transient_error_then_success_keeps_the_reference(self):
        for error in (self.TIMEOUT, self.SERVER):
            with self.subTest(error=str(error)):
                rec, client = self.fast(failing=(90,), failing_sam=((180, "head"),), error=error, once=True)
                self.assertEqual(sorted(client.edits), sorted([(a, 0) for a in beauty.FAST_ANGLES] + [(90, 0)]))
                self.assertEqual(rec["selected"], {str(a): True for a in beauty.FAST_ANGLES})
                self.assertEqual(rec["contained_errors"], [])
                self.assertEqual([(e["yaw"], e["request"], e.get("concept"), e["retry"]) for e in rec["retries"]],
                                 [(90, "image edit", None, "ok"), (180, "SAM query", "head", "ok")])
                self.assertEqual(rec["views"]["90"]["retries"][0]["http_status"], error.http_status)
                self.assertEqual([beauty.selected_yaw(beauty.load(self.out), a) for a in beauty.FAST_ANGLES],
                                 list(beauty.FAST_ANGLES))
                # Both sends of each resent request are on the caller's ledger: five images, nine SAM queries.
                self.assertAlmostEqual(client.log.cost_usd(), 5 * .01 + 9 * .0025)

    def test_content_policy_error_is_never_resent(self):
        rec, client = self.fast(failing=(90,), error=self.POLICY, once=True)
        self.assertEqual(sorted(client.edits), sorted((a, 0) for a in beauty.FAST_ANGLES))
        self.assertEqual(rec["retries"], [])
        self.assertEqual([e["yaw"] for e in rec["contained_errors"]], [90])

    def test_sam_errors_on_a_yaw_leave_it_without_a_reference(self):
        rec, client = self.fast(failing_sam=((90, "character"), (180, "head")), error=self.SERVER)
        saved = json.loads((self.out / "references.json").read_text())
        self.assertEqual([(e["yaw"], e["request"], e["image"], e["retried"]) for e in saved["contained_errors"]],
                         [(90, "SAM character outline", "beauty+90-try1.png", True),
                          (180, "SAM head crop", "beauty+180-try1.png", True)])
        for yaw in ("90", "180"):
            self.assertEqual(saved["views"][yaw]["attempts"], [])
            self.assertNotIn("image", saved["views"][yaw])
        loaded = beauty.load(self.out)
        self.assertEqual([beauty.selected_yaw(loaded, a) for a in beauty.FAST_ANGLES], [0, None, -90, None])
        self.assertEqual(rec["selected"], {"0": True, "90": False, "-90": True, "180": False})

    def test_front_model_error_turns_the_beauty_arm_off(self):
        rec, _ = self.fast(failing=(0,), error=self.POLICY)
        self.assertEqual([(e["yaw"], e["fallback"], e["error_code"]) for e in rec["contained_errors"]],
                         [(0, "beauty off", "content_policy_violation")])
        self.assertIsNone(beauty.selected_yaw(beauty.load(self.out), 0))

    def test_replay_miss_drains_every_yaw_and_propagates(self):
        with self.assertRaises(ReplayMiss):
            self.fast(failing=(-90,), error=ReplayMiss("no cache entry"))
        client = self.fast_client
        self.assertEqual(sorted(client.edits), sorted((a, 0) for a in beauty.FAST_ANGLES))
        rec = json.loads((self.out / "references.json").read_text())
        self.assertEqual((rec["status"], rec["error_type"]), ("error", "ReplayMiss"))
        self.assertEqual(rec["views"]["-90"]["attempts"], [])
        self.assertNotIn("error", rec["views"]["-90"])
        self.assertTrue(all(len(rec["views"][str(a)]["attempts"]) == 1 for a in (0, 90, 180)))
        self.assertEqual(rec["cost_usd"], client.log.cost_usd())
        self.assertEqual(sum(c["kind"] == "edit" for c in client.log.calls), 4)

    def test_folder_is_rebuilt_for_each_run(self):
        self.out.mkdir()
        (self.out / "beauty+45-try1.png").write_bytes(b"stale")
        self.fast()
        self.assertFalse((self.out / "beauty+45-try1.png").exists())
        self.assertNotIn("45", json.loads((self.out / "references.json").read_text())["views"])


if __name__ == "__main__":
    unittest.main()
