"""The checks of the Modal build-log smoke test, run on one local folder (no Modal needed): events written while
an `EventPump` mirrors them arrive in order and on time, nothing but the log is written, and a delete while
pumping stays deleted."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "buildlog"))

import modal_smoke as smoke  # noqa: E402

from herogen.service.snapshot import LocalVolume  # noqa: E402
from herogen.service.storage import VolumeView  # noqa: E402


class SmokeChecksTest(unittest.TestCase):
    def test_a_short_local_run_passes_every_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, local = Path(tmp) / "volume", Path(tmp) / "local" / "events.jsonl"
            flags: dict = {}
            out: dict = {}
            deleted = lambda: flags.get("deleted", False)  # noqa: E731
            w = threading.Thread(target=lambda: out.update(w=smoke.write_events(
                LocalVolume(), root, local, 2.5, deleted, pump_s=0.1, emit_every_s=0.05)))
            w.start()
            time.sleep(0.2)
            r = threading.Thread(target=lambda: out.update(r=smoke.read_events(VolumeView(), root, 3.5, deleted,
                                                                               poll_s=0.05)))
            r.start()
            time.sleep(1.2)
            flags["deleted"] = True
            held = smoke.delete_run(LocalVolume(), root)
            w.join()
            r.join()
            left = sorted(str(p) for p in root.rglob("*") if p.is_file())
            summary = smoke.check(out["w"], out["r"], held, left, max_lag_s=1.0, settle_s=0.5,
                                  raced=False)
            self.assertGreater(summary["events"], 10)

    def test_check_names_what_failed(self):
        good = {"seen": [0, 1, 2], "lags": [1.0], "bad": 0, "after_delete": [(1.0, False), (20.0, False)]}
        written = {"emitted": 3, "pump_stopped": True}
        held = [f"{smoke.RUN}/events.jsonl"]
        smoke.check(written, good, held, [])
        for change, message in (({"bad": 1}, "garbled"), ({"seen": [0, 2]}, "lost"), ({"lags": [9.0]}, "slowest"),
                                ({"after_delete": [(1.0, False), (20.0, True)]}, "came back")):
            with self.subTest(message=message), self.assertRaisesRegex(AssertionError, message):
                smoke.check(written, {**good, **change}, held, [])
        with self.assertRaisesRegex(AssertionError, "more than the log"):
            smoke.check(written, good, held + [f"{smoke.RUN}/events.jsonl.part"], [])


if __name__ == "__main__":
    unittest.main()
