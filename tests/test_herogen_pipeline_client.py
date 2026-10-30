"""The model client keeps a paid SAM response even when decoding it fails."""

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
from herogen.pipeline.client import ClientError, ModelClient  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
