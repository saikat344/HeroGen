"""The Sapiens 2 pose client: record/replay, retries and failures, with a fake Modal app (no network)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.pipeline import sapiens  # noqa: E402
from herogen.pipeline.client import ReplayMiss  # noqa: E402

ANSWER = {"names": ["nose", "left_wrist"], "keypoints": [[10.0, 20.0], [30.5, 40.25]], "scores": [0.9, 0.4],
          "box": [0.0, 0.0, 64.0, 64.0], "gpu": "fake", "load_s": 1.0, "seconds": 2.0}


class _Remote:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def spawn(self, png, box):
        self.calls.append(box)
        return _Call(self.outcomes.pop(0))


class _Call:
    def __init__(self, out):
        self.out = out

    def get(self, timeout):
        assert timeout == sapiens.CALL_TIMEOUT_S
        if isinstance(self.out, Exception):
            raise self.out
        return self.out


class _Pose:
    def __init__(self, outcomes):
        self.keypoints = _Remote(outcomes)


class PoseClientTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "cache"
        self.img = Image.new("RGB", (64, 64), (200, 10, 10))

    def tearDown(self):
        self.tmp.cleanup()

    def _live(self, outcomes):
        c = sapiens.PoseClient(self.cache, mode="live")
        pose = _Pose(outcomes)
        opened = []

        def fake_open(client):
            opened.append(1)
            client._pose = pose

        return c, pose, opened, mock.patch.object(sapiens.PoseClient, "_open", fake_open)

    def test_live_then_replay(self):
        c, pose, opened, patch = self._live([ANSWER])
        with patch, mock.patch.object(sapiens.time, "sleep"):
            kps = c.keypoints(self.img, [0, 0, 64, 64])
        self.assertEqual(kps["left_wrist"], (30.5, 40.25, 0.4))
        self.assertEqual(len(opened), 1)
        self.assertFalse(c.log.calls[0]["cached"])
        self.assertAlmostEqual(c.log.calls[0]["cost_usd"], 2.0 * sapiens.PRICE_PER_SECOND, places=6)
        entry = json.loads(next(self.cache.glob("sapiens-*.json")).read_text())
        self.assertEqual(entry["request"]["model"], sapiens.POSE_MODEL)
        self.assertEqual(entry["request"]["box"], [0.0, 0.0, 64.0, 64.0])
        again = sapiens.PoseClient(self.cache, mode="replay")
        self.assertEqual(again.keypoints(self.img, [0, 0, 64, 64]), kps)
        self.assertTrue(again.log.calls[0]["cached"])

    def test_replay_miss(self):
        c = sapiens.PoseClient(self.cache, mode="replay")
        with self.assertRaises(ReplayMiss):
            c.keypoints(self.img, [0, 0, 64, 64])

    def test_key_covers_box_and_image(self):
        c, pose, _, patch = self._live([ANSWER, ANSWER, ANSWER])
        with patch, mock.patch.object(sapiens.time, "sleep"):
            c.keypoints(self.img, [0, 0, 64, 64])
            c.keypoints(self.img, [1, 0, 64, 64])
            c.keypoints(Image.new("RGB", (64, 64)), [0, 0, 64, 64])
            c.keypoints(self.img, [0, 0, 64, 64])  # cached
        self.assertEqual(len(pose.keypoints.calls), 3)
        self.assertEqual(len(list(self.cache.glob("sapiens-*.json"))), 3)

    def test_retries_then_succeeds(self):
        c, pose, opened, patch = self._live([TimeoutError("no container"), ANSWER])
        with patch, mock.patch.object(sapiens.time, "sleep") as sleep:
            c.keypoints(self.img, [0, 0, 64, 64])
        self.assertEqual(len(opened), 2)  # the app is opened again after a failure
        self.assertEqual([a.args[0] for a in sleep.call_args_list], [0.0, sapiens.RETRY_WAIT_S[0]])

    def test_every_try_fails(self):
        c, _, _, patch = self._live([RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])
        with patch, mock.patch.object(sapiens.time, "sleep"), self.assertRaises(sapiens.PoseError) as e:
            c.keypoints(self.img, [0, 0, 64, 64])
        self.assertIn("a", str(e.exception))
        self.assertEqual(list(self.cache.glob("sapiens-*.json")), [])  # a failure is never cached

    def test_missing_modal_is_a_pose_error(self):
        c = sapiens.PoseClient(self.cache, mode="live")

        def no_modal(_client):
            raise ImportError("No module named 'modal'")

        with mock.patch.object(sapiens.PoseClient, "_open", no_modal), self.assertRaises(sapiens.PoseError):
            c.keypoints(self.img, [0, 0, 64, 64])

    def test_pins_follow_the_app(self):
        pins = sapiens.app_pins()
        self.assertEqual(pins["model"], sapiens.POSE_MODEL)
        self.assertEqual(len(pins["code"]), 40)
        self.assertEqual(len(pins["weights"]["sha256"]), 64)
        self.assertEqual(len(pins["app_sha256"]), 16)

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            sapiens.PoseClient(self.cache, mode="other")


if __name__ == "__main__":
    unittest.main()
