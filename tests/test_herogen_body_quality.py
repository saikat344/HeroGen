"""Synthetic geometry and mocked strict visual responses for body acceptance measurements."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from herogen.blender import body_quality as q
from herogen.pipeline.client import ClientError


def quality_client(score=75):
    """A strict-schema client substitute for synthetic construction tests; no paid calls."""
    from herogen.pipeline.client import CallLog

    client = mock.Mock()
    client.log = CallLog()
    client.compare_images.return_value = {"usable_probability": score, "reason": "synthetic solid body"}
    return client


def prism(sides=4):
    angle = np.arange(sides) * 2 * np.pi / sides + np.pi / 4
    ring = np.column_stack((np.cos(angle), np.sin(angle)))
    v = np.vstack([np.column_stack((ring, np.full(sides, z))) for z in (0, 1)])
    f = []
    for i in range(sides):
        j = (i + 1) % sides
        f.extend([(i, j, j + sides), (i, j + sides, i + sides)])
    return v, np.asarray(f)


class CrossSectionTests(unittest.TestCase):
    def test_distinguishes_box_from_rounded_body(self):
        box = q.cross_sections(*prism())
        rounded = q.cross_sections(*prism(128))
        self.assertAlmostEqual(box["rectangularity"], 1.0)
        self.assertAlmostEqual(rounded["rectangularity"], np.pi / 4, delta=0.001)
        self.assertEqual(len(box["sections"]), 3)

    def test_descriptor_is_translation_and_scale_invariant(self):
        v, f = prism(16)
        self.assertAlmostEqual(q.cross_sections(v, f)["rectangularity"],
                               q.cross_sections(v * [2, 0.1, 5] + [6, -4, 7], f)["rectangularity"])

    def test_missing_section_is_explicit(self):
        v = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 1]])
        self.assertIsNone(q.cross_sections(v, np.array([[0, 1, 2]]))["rectangularity"])

    def test_bad_meshes_raise_clear_errors(self):
        v, f = prism()
        for bad in (v * np.nan, np.zeros_like(v)):
            with self.assertRaises(ValueError):
                q.cross_sections(bad, f)


class VisualTests(unittest.TestCase):
    def images(self):
        return {n: Image.new("L", (32, 32), 100) for n in q.YAWS}

    def test_prompt_is_pinned_and_schema_and_parameters_are_fixed(self):
        self.assertEqual(hashlib.sha256(q.PROMPT.encode()).hexdigest(), q.PROMPT_SHA256)
        client = mock.Mock()
        client.compare_images.return_value = {"usable_probability": 75, "reason": "solid body"}
        result = q.visual_score(client, Image.new("RGB", (32, 32)), self.images())
        self.assertEqual(result["usable_probability"], 75)
        kwargs = client.compare_images.call_args.kwargs
        self.assertEqual(kwargs["schema"], q.SCHEMA)
        self.assertEqual(kwargs["model"], "muse-spark-1.3")
        self.assertEqual(client.compare_images.call_args.args[1][0].size, (1280, 688))

    def test_invalid_or_failed_answer_never_passes(self):
        client = mock.Mock()
        for answer in ({}, {"usable_probability": True, "reason": "yes"},
                       {"usable_probability": 101, "reason": "yes"},
                       {"usable_probability": 80, "reason": " "},
                       {"usable_probability": 80, "reason": "yes", "extra": True}):
            client.compare_images.return_value = answer
            with self.assertRaises(ClientError):
                q.visual_score(client, Image.new("RGB", (32, 32)), self.images())
        client.compare_images.side_effect = ClientError("model unavailable")
        with self.assertRaisesRegex(ClientError, "model unavailable"):
            q.visual_score(client, Image.new("RGB", (32, 32)), self.images())

    def test_missing_views_raise_before_the_call(self):
        client = mock.Mock()
        with self.assertRaises(KeyError):
            q.visual_score(client, Image.new("RGB", (32, 32)), {})
        client.compare_images.assert_not_called()

    def test_six_renders_share_a_scale(self):
        v, f = prism(16)
        images = q.render_views(v, f, size=48)
        self.assertEqual(list(images), list(q.YAWS))
        for im in images.values():
            self.assertEqual(im.size, (48, 48))
            self.assertLess(np.asarray(im).min(), 255)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        Image.new("RGB", (32, 32), "white").save(self.run / "00-sketch.png")

    def test_calibrated_boundary_and_recorded_rejection_reason(self):
        from herogen.blender.body_acceptance import assess
        from herogen.blender.record import StageRecord

        for score, accepted in ((17, False), (18, True)):
            rec = StageRecord(self.run, 3, "base mesh")
            assess(self.run, *prism(16), rec, client=quality_client(score))
            result = rec.finish()
            self.assertEqual(result["ok"], accepted)
            self.assertEqual(result["checks"][0]["reason"], "synthetic solid body")
            self.assertTrue((self.run / "03-quality" / "sheet.png").is_file())
            self.assertEqual(len(result["outputs"]), 7)

    def test_model_error_is_loud_and_preserves_cost_in_failed_record(self):
        from herogen.blender.body_acceptance import BodyQualityError, assess
        from herogen.blender.record import StageRecord

        client = quality_client()

        def fail(*args, **kwargs):
            client.log.add("muse-spark-1.3", "compare", "synthetic", False, cost_usd=0.01)
            raise ClientError("responses: incomplete")

        client.compare_images.side_effect = fail
        rec = StageRecord(self.run, 3, "base mesh")
        rec.data["cost_usd"] = 0.1
        with self.assertRaisesRegex(BodyQualityError, "incomplete"):
            assess(self.run, *prism(16), rec, client=client)
        result = json.loads((self.run / "stage-3.json").read_text())
        self.assertFalse(result["ok"])
        self.assertEqual(result["cost_usd"], 0.11)
        self.assertIn("incomplete", result["error"])
        self.assertIn("muse-spark-1.3", result["models"])

    def test_quality_error_never_retries_another_constructor(self):
        from herogen.blender.body_acceptance import BodyQualityError
        from herogen.blender import stage3

        with mock.patch.object(stage3, "_run_stage3", side_effect=BodyQualityError("model failed")) as build:
            with self.assertRaises(BodyQualityError):
                stage3.run_stage3(self.run)
        self.assertEqual(build.call_count, 1)

    def test_missing_sketch_is_a_recorded_error(self):
        from herogen.blender.body_acceptance import BodyQualityError, assess
        from herogen.blender.record import StageRecord

        (self.run / "00-sketch.png").unlink()
        client = quality_client()
        with self.assertRaises(BodyQualityError):
            assess(self.run, *prism(16), StageRecord(self.run, 3, "base mesh"), client=client)
        client.compare_images.assert_not_called()


if __name__ == "__main__":
    unittest.main()
