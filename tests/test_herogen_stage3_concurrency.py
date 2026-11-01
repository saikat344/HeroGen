"""Stage 3's Body inference overlaps Objects while conversion and accounting stay ordered."""
from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from herogen.blender import stage3
from herogen.blender.record import StageRecord
from herogen.blender.remote.client import RemoteError, ReplayMiss


class BodyPrefetchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        mask = np.zeros((16, 16), bool)
        mask[2:14, 4:12] = True
        self.vs = SimpleNamespace(views={"front": SimpleNamespace(mask=mask)})
        self.rec = StageRecord(self.run, 3, "base mesh")
        self.result = SimpleNamespace(cached=False, entry={"cost_usd_estimate": .01})

    def test_body_is_in_flight_before_objects_and_consumed_once(self):
        started, finish = threading.Event(), threading.Event()

        def body(*args, **kwargs):
            started.set()
            if not finish.wait(5):
                raise AssertionError("Body was not released by Objects")
            return self.result

        def objects(*args, **kwargs):
            self.assertTrue(started.wait(5))
            finish.set()
            return "mesh"

        remote = SimpleNamespace(body=Mock(side_effect=body), objects=Mock(side_effect=objects))
        with stage3._prefetch_body(self.run, self.vs, remote, self.rec, None, enabled=True) as proxy:
            self.assertEqual(proxy.objects("rgb", "mask"), "mesh")
            self.assertIs(proxy.body(), self.result)
            # The caller accounts only after the Objects result, exactly as in the serial path.
            self.assertEqual(self.rec.data["models"], [])
            stage3._account(self.rec, "sam-3d-body", proxy.body())
        remote.body.assert_called_once_with(self.run / "views/front.png", [4, 2, 12, 14],
                                            stage3.BODY_BOX_SOURCE, detector=None)
        self.assertEqual(self.rec.data["cost_usd"], .01)

    def test_early_objects_failure_drains_and_accounts_body(self):
        remote = SimpleNamespace(body=Mock(return_value=self.result))
        with self.assertRaisesRegex(ValueError, "mesh failed"):
            with stage3._prefetch_body(self.run, self.vs, remote, self.rec, None, enabled=True):
                raise ValueError("mesh failed")
        self.assertEqual(self.rec.data["models"], ["sam-3d-body"])
        self.assertEqual(self.rec.data["cost_usd"], .01)

    def test_unused_remote_failure_and_replay_miss_stay_loud(self):
        for error in (RemoteError, ReplayMiss):
            remote = SimpleNamespace(body=Mock(side_effect=error("Body failed")))
            with self.subTest(error=error), self.assertRaisesRegex(error, "Body failed"):
                with stage3._prefetch_body(self.run, self.vs, remote, self.rec, None, enabled=True):
                    pass

    def test_two_failures_preserve_the_original_error_and_record_the_drained_failure(self):
        primary = RemoteError("Objects failed")
        secondary = ReplayMiss("Body replay missed")
        remote = SimpleNamespace(body=Mock(side_effect=secondary))
        with self.assertRaisesRegex(RemoteError, "Objects failed") as caught:
            with stage3._prefetch_body(self.run, self.vs, remote, self.rec, None, enabled=True):
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertIs(caught.exception.__cause__, secondary)
        self.assertIn("Body replay missed", self.rec.data["notes"])

    def test_serial_option_does_not_launch_body(self):
        remote = Mock()
        with stage3._prefetch_body(self.run, self.vs, remote, self.rec, None, enabled=False) as returned:
            self.assertIs(returned, remote)
        remote.body.assert_not_called()


if __name__ == "__main__":
    unittest.main()
