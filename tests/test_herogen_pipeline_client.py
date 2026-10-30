"""The model client keeps a paid SAM response even when decoding it fails, and asks Muse Spark about images with a
strict schema, checking every answer against it."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.pipeline import client as client_mod  # noqa: E402
from herogen.pipeline.client import ApiError, ClientError, ModelClient, ReplayMiss, schema_errors  # noqa: E402

_RESPONSE = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "<masks>"}]}]}


class SamPaidResponseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "cache"
        self.image = Image.new("RGB", (16, 16), "white")

    def tearDown(self):
        self.tmp.cleanup()

    def test_failed_decode_keeps_the_response_and_bills_once(self):
        posts = []

        def fake_post(_client, endpoint, body, **_):
            posts.append(endpoint)
            return _RESPONSE

        def broken_decode(text, size):
            raise ClientError("no parser")

        with mock.patch.object(ModelClient, "_post", fake_post), \
                mock.patch.object(client_mod, "decode_sam_output", broken_decode):
            c = ModelClient(self.cache, mode="live")
            with self.assertRaises(ClientError):
                c.segment(self.image, "character")
            with self.assertRaises(ClientError):
                c.segment(self.image, "character")  # decodes the saved text again, no second POST
        self.assertEqual(posts, ["responses"])
        self.assertEqual([call["cached"] for call in c.log.calls], [False, True])
        self.assertAlmostEqual(c.log.cost_usd(), client_mod.PRICE_PER_IMAGE_USD[client_mod.SAM_MODEL])

        with mock.patch.object(client_mod, "decode_sam_output", lambda text, size: []):
            replay = ModelClient(self.cache, mode="replay")
            self.assertEqual(replay.segment(self.image, "character"), [])  # recovered from the cache offline


SCHEMA = {"name": "verdict", "schema": {
    "type": "object",
    "properties": {"score": {"type": "number"}, "kind": {"type": "string", "enum": ["a", "b"]},
                   "tags": {"type": "array", "items": {"type": "string"}}},
    "required": ["score", "kind", "tags"], "additionalProperties": False}}


def _answer(text: str, status: str = "completed") -> dict:
    return {"status": status, "usage": {"input_tokens": 1000, "output_tokens": 200},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


class CompareImagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "cache"
        self.image = Image.new("RGB", (2048, 1024), "white")

    def tearDown(self):
        self.tmp.cleanup()

    def _live(self, resp):
        posts = []

        def fake_post(_client, endpoint, body, **kw):
            posts.append((endpoint, body, kw))
            return resp

        with mock.patch.object(ModelClient, "_post", fake_post):
            c = ModelClient(self.cache, mode="live")
            try:
                return c.compare_images("prompt", [self.image], schema=SCHEMA), c, posts
            except ClientError as e:
                return e, c, posts

    def test_answer_is_recorded_priced_and_replayed(self):
        out, c, posts = self._live(_answer('{"score": 0.25, "kind": "a", "tags": ["x"]}'))
        self.assertEqual(out, {"score": 0.25, "kind": "a", "tags": ["x"]})
        endpoint, body, kw = posts[0]
        self.assertEqual((endpoint, kw["model"], kw["kind"]), ("responses", client_mod.TEXT_MODEL, "compare"))
        self.assertEqual(body["text"]["format"]["strict"], True)
        self.assertTrue(body["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertAlmostEqual(c.log.cost_usd(), (1000 * 1.25 + 200 * 4.25) / 1e6)
        replay = ModelClient(self.cache, mode="replay")
        self.assertEqual(replay.compare_images("prompt", [self.image], schema=SCHEMA), out)
        self.assertEqual(replay.log.cost_usd(), 0)
        with self.assertRaises(ReplayMiss):
            replay.compare_images("another prompt", [self.image], schema=SCHEMA)

    def test_a_non_conforming_answer_is_never_guessed(self):
        for text, why in (('{"score": 0.2, "kind": "c", "tags": []}', "not one of"),
                          ('{"score": 0.2, "tags": []}', "kind is missing"),
                          ('{"score": "high", "kind": "a", "tags": []}', "should be number"),
                          ('{"score": 1, "kind": "a", "tags": [], "extra": 1}', "extra is not in the schema"),
                          ("not json", "is not JSON"), ("", "no output text")):
            self.tmp.cleanup()
            self.tmp = tempfile.TemporaryDirectory()
            self.cache = Path(self.tmp.name) / "cache"
            err, _, _ = self._live(_answer(text))
            self.assertIsInstance(err, ClientError, text)
            self.assertIn(why, str(err))

    def test_an_incomplete_answer_is_a_failure_and_not_cached(self):
        err, c, _ = self._live(_answer('{"score": 0.2, "kind": "a", "tags": []}', status="incomplete"))
        self.assertIsInstance(err, ApiError)
        self.assertGreater(c.log.cost_usd(), 0)  # answered, so billed
        self.assertFalse(list(self.cache.glob("*.json")))

    def test_schema_errors(self):
        self.assertEqual(schema_errors({"score": 1, "kind": "b", "tags": ["t"]}, SCHEMA["schema"]), [])
        self.assertEqual(schema_errors(True, {"type": "number"}), ["answer should be number, got bool"])
        self.assertIn("can't be checked", schema_errors(1, {"type": "number", "minimum": 0})[0])


if __name__ == "__main__":
    unittest.main()
