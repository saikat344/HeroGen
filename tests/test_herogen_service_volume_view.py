"""The web's view of the Modal data volume: refreshes never queue behind each other, and a waiting reload is not
starved by a stream of new readers."""

import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from herogen.service.storage import ModalVolumeView
from herogen.service.worker import prewarm_before_shape


class Vol:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.reloads = 0

    def reload(self) -> None:
        time.sleep(self.delay)
        self.reloads += 1

    def commit(self) -> None:
        pass


class VolumeViewTest(unittest.TestCase):
    def test_a_throttled_refresh_does_not_wait_for_one_under_way(self):
        vol = Vol(delay=0.3)
        view = ModalVolumeView(vol, min_interval_s=0.0)
        first = threading.Thread(target=view.refresh)
        first.start()
        time.sleep(0.05)
        t = time.monotonic()
        view.refresh()   # another request: reads the current view instead of queueing a second reload
        self.assertLess(time.monotonic() - t, 0.1)
        first.join()
        self.assertEqual(vol.reloads, 1)

    def test_a_forced_refresh_still_reloads(self):
        vol = Vol()
        view = ModalVolumeView(vol)
        view.refresh()
        view.refresh(0)
        self.assertEqual(vol.reloads, 2)

    def test_new_readers_wait_behind_a_waiting_reload(self):
        vol = Vol()
        view = ModalVolumeView(vol, min_interval_s=0.0)
        order: list[str] = []
        release = threading.Event()

        def long_read():
            with view.reading():
                order.append("read 1")
                release.wait(2)

        def late_read():
            with view.reading():
                order.append("read 2")

        def refresh():
            view.refresh()
            order.append("reload")

        a = threading.Thread(target=long_read)
        a.start()
        time.sleep(0.05)
        b = threading.Thread(target=refresh)
        b.start()
        time.sleep(0.05)
        c = threading.Thread(target=late_read)
        c.start()
        time.sleep(0.05)
        self.assertEqual(order, ["read 1"], "the late reader waits behind the reload")
        release.set()
        for t in (a, b, c):
            t.join(2)
        self.assertEqual(order, ["read 1", "reload", "read 2"])
        self.assertEqual((view.readers, view.waiting, view.writer, view.refreshing), (0, 0, False, False))

    def test_a_failed_reload_releases_the_view(self):
        class Broken(Vol):
            def reload(self):
                raise RuntimeError("open file")

        view = ModalVolumeView(Broken(), min_interval_s=0.0)
        with self.assertRaises(RuntimeError):
            view.refresh()
        self.assertEqual((view.waiting, view.writer, view.refreshing), (0, False, False))
        with view.reading():
            self.assertEqual(view.readers, 1)


class PrewarmBeforeShapeTest(unittest.TestCase):
    PLAN = [SimpleNamespace(key=k) for k in ("p", "0", "1", "2", "3", "3f", "4")]

    def calls(self, key, plan=PLAN, remote="1"):
        with mock.patch.dict(os.environ, {"MODAL_IS_REMOTE": remote} if remote else {}, clear=False), \
                mock.patch("herogen.blender.remote.prewarm") as warm, \
                mock.patch("threading.Thread") as thread:
            if not remote:
                os.environ.pop("MODAL_IS_REMOTE", None)
            prewarm_before_shape(plan, SimpleNamespace(key=key))
            return thread.call_args_list, warm

    def test_starts_the_warmup_when_the_turnaround_starts_in_a_modal_container(self):
        calls, warm = self.calls("2")
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0].kwargs["target"], warm)
        self.assertTrue(calls[0].kwargs["daemon"])

    def test_nothing_for_other_stages_local_runs_or_plans_without_a_shape(self):
        self.assertEqual(self.calls("1")[0], [])
        self.assertEqual(self.calls("3")[0], [])
        self.assertEqual(self.calls("2", remote=None)[0], [])
        self.assertEqual(self.calls("2", plan=self.PLAN[:4])[0], [])


if __name__ == "__main__":
    unittest.main()
