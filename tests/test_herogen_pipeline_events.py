"""The build log's event file: schema whitelist, size caps, no-op without a target, safe concurrent appends from
threads and processes, and readers that never see a partial line."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import CancelledError
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from herogen.pipeline import events  # noqa: E402


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


class EventsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / events.FILE
        env = {k: v for k, v in os.environ.items() if not k.startswith("HEROGEN_EVENTS_")}
        patch = mock.patch.dict(os.environ, {**env, events.ENV_FILE: str(self.path)}, clear=True)
        patch.start()
        self.addCleanup(patch.stop)

    def test_no_target_is_a_no_op(self):
        with mock.patch.dict(os.environ, {events.ENV_FILE: ""}):
            self.assertIsNone(events.emit("local", step="0", what="input_checked"))
            with events.call("muse-image", "cover_art") as c:
                c.usd = 0.01
            self.assertIsNone(events.image("01-cover.png", step="1"))
        self.assertFalse(self.path.exists())

    def test_emit_writes_one_valid_line_with_step_and_attempt_context(self):
        with mock.patch.dict(os.environ, {events.ENV_STEP: "2", events.ENV_ATTEMPT: "3"}):
            ev = events.emit("decision", what="view_result", view="body+90", try_=2, ok=False, why="outline_moved",
                             m={"median_px": 4.666666666, "median_limit": 6})
        self.assertEqual(lines(self.path), [ev])
        self.assertEqual((ev["v"], ev["kind"], ev["step"], ev["a"], ev["try"]), (1, "decision", "2", 3, 2))
        self.assertEqual(ev["m"]["median_px"], 4.666667)
        with events.step("3f"):
            self.assertEqual(events.emit("local", what="views_rendered")["step"], "3f")
        with events.step("5"):
            self.assertNotIn("step", events.emit("job_end", state="succeeded"))
        self.assertIsNone(events.emit("job_end", step="5", state="succeeded"))

    def test_schema_whitelist_drops_anything_outside_it(self):
        bad = [
            dict(kind="local", what="has spaces"),            # identifiers only, never text
            dict(kind="local", what="Users/me/run"),          # never a path
            dict(kind="call_start", who="other-model"),       # closed model vocabulary
            dict(kind="local", prompt="draw a fox"),          # no such field
            dict(kind="local", m={"x": float("nan")}),        # finite numbers only
            dict(kind="local", m={"x": "1"}),
            dict(kind="local", s=-1),
            dict(kind="nonsense"),
            dict(kind="call_start", target="a samurai fox"),  # SAM concepts are a closed set
            dict(kind="checks", failed=["ok"] * 21),
            dict(kind="check", ok=1),                          # booleans are booleans
            dict(kind="local", step="9"),
        ]
        for fields in bad:
            with self.subTest(fields=fields):
                kind = fields.pop("kind")
                self.assertIsNone(events.emit(kind, **fields))
                with self.assertRaises(ValueError):
                    events.validate({"v": 1, "t": 1, "kind": kind, **fields})
        self.assertFalse(self.path.exists())
        self.assertEqual(events.target_for("a samurai fox"), "other")
        self.assertEqual(events.target_for("feet"), "feet")
        self.assertEqual(events.slug("Full-body sketch"), "full_body_sketch")

    def test_every_documented_example_validates(self):
        examples = [
            {"v": 1, "t": 1, "kind": "job_start", "a": 1, "of": 2},
            {"v": 1, "t": 1, "kind": "requeue", "a": 1, "next": 2, "of": 2, "step": "3", "code": "gpu_unavailable"},
            {"v": 1, "t": 1, "kind": "gpu_end", "step": "3", "id": "5e6f7a8b9c0d", "who": "sam-3d-objects",
             "what": "3d_shape", "outcome": "ok", "s": 172.9, "usd": 0.101, "gpu": "a100-40gb",
             "m": {"load_s": 60.4, "infer_s": 96.1}},
            {"v": 1, "t": 1, "kind": "checks", "step": "2", "n": 13, "passed": 10, "gated": 4, "failed": [],
             "advisory": ["top_depth_ratio"]},
            {"v": 1, "t": 1, "kind": "check", "step": "0", "name": "min_side_px", "ok": True, "gate": True,
             "op": ">=", "m": {"value": 1280, "limit": 256}},
        ]
        for ev in examples:
            self.assertEqual(events.validate(ev), ev)
        with self.assertRaises(ValueError):
            events.validate({"v": 1, "t": 1, "kind": "job_start", "step": "1"})

    def test_line_size_cap(self):
        self.assertIsNone(events.emit("checks", step="4", n=1, failed=["x" * 48] * 20, advisory=["y" * 48] * 20))
        self.assertFalse(self.path.exists())

    def test_run_cap_keeps_only_step_boundaries_and_outcomes_after_one_truncated_marker(self):
        with mock.patch.object(events, "MAX_EVENTS", 5):
            for _ in range(7):
                events.emit("local", step="4", what="part_split")
            events.emit("step_end", step="4", outcome="passed", s=1.0, usd=0)
            events.emit("job_end", state="succeeded")
            events.emit("local", step="4", what="part_split")
        kinds = [e["kind"] for e in lines(self.path)]
        self.assertEqual(kinds, ["local"] * 5 + ["truncated", "step_end", "job_end"])

    def test_byte_cap(self):
        with mock.patch.object(events, "MAX_BYTES", 200):
            for _ in range(10):
                events.emit("local", step="4", what="part_split")
        kinds = [e["kind"] for e in lines(self.path)]
        self.assertEqual(kinds.count("truncated"), 1)
        self.assertLess(len(kinds), 10)

    def test_concurrent_appends_from_threads_never_interleave(self):
        def work(n):
            for i in range(50):
                events.emit("local", step="3f", what="repaint_checked", m={"thread": n, "i": i})
        threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        got = lines(self.path)
        self.assertEqual(len(got), 400)
        self.assertEqual(sorted((e["m"]["thread"], e["m"]["i"]) for e in got),
                         sorted((n, i) for n in range(8) for i in range(50)))

    def test_concurrent_appends_from_processes_never_interleave(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); from herogen.pipeline import events\n"
                "for i in range(100): events.emit('local', step='4', what='part_split', m={'p': int(sys.argv[2]), 'i': i})")
        procs = [subprocess.Popen([sys.executable, "-c", code, str(ROOT), str(p)], env=dict(os.environ))
                 for p in range(4)]
        for p in procs:
            self.assertEqual(p.wait(timeout=120), 0)
        got = lines(self.path)
        self.assertEqual(len(got), 400)
        self.assertEqual(len({(e["m"]["p"], e["m"]["i"]) for e in got}), 400)

    def test_reader_skips_partial_last_line_and_resumes(self):
        events.emit("local", step="0", what="input_checked")
        events.emit("image", step="0", what="sketch")
        with open(self.path, "ab") as f:
            f.write(b'{"v":1,"t":1,"kind":"loc')         # a writer mid-line
        got, cursor, more = events.read(self.path, 0)
        self.assertEqual([e["kind"] for _, e in got], ["local", "image"])
        self.assertFalse(more)
        self.assertEqual(cursor, got[-1][0])
        self.assertEqual(events.read(self.path, cursor), ([], cursor, False))
        with open(self.path, "ab") as f:
            f.write(b'al","step":"0","what":"x"}\n')
        got2, cursor2, _ = events.read(self.path, cursor)
        self.assertEqual([(e["kind"], e["what"]) for _, e in got2], [("local", "x")])
        self.assertEqual(cursor2, self.path.stat().st_size)

    def test_reader_skips_invalid_lines_and_pages(self):
        with open(self.path, "ab") as f:
            f.write(b"not json\n")
            f.write(b'{"v":1,"t":1,"kind":"local","prompt":"secret idea"}\n')
        for i in range(5):
            events.emit("local", step="4", what="part_split", n=i)
        got, cursor, more = events.read(self.path, 0, limit=2)
        self.assertEqual([e["n"] for _, e in got], [0, 1])
        self.assertTrue(more)
        got, cursor, more = events.read(self.path, cursor, limit=10)
        self.assertEqual([e["n"] for _, e in got], [2, 3, 4])
        self.assertFalse(more)
        self.assertNotIn("secret", json.dumps([e for _, e in got]))
        self.assertEqual(events.read(Path(self.tmp.name) / "missing.jsonl", 0), ([], 0, False))

    def test_call_pairs_start_and_end_and_classifies_failures(self):
        with events.call("muse-image", "turnaround", step="2") as c:
            c.add_cost(0.01)
        with self.assertRaises(TimeoutError), events.call("sam-3.1", "cover_art", step="1", target="feet"):
            raise TimeoutError("socket")
        with self.assertRaises(CancelledError), events.call("muse-image", "repaint", step="3f", view="body+0", try_=2):
            raise CancelledError()
        with events.call("sam-3.1", "cover_art", step="1"):
            events.mark_reused()
        with events.call("sam-3d-objects", "3d_shape", gpu=True, step="3") as c:
            c.m["load_s"] = 60.4
            c.extra["gpu"] = "a100-40gb"
        got = lines(self.path)
        self.assertEqual([(e["kind"], e.get("outcome"), e.get("code")) for e in got], [
            ("call_start", None, None), ("call_end", "ok", None),
            ("call_start", None, None), ("call_end", "failed", "timeout"),
            ("call_start", None, None), ("call_end", "cancelled", None),
            ("call_start", None, None), ("call_end", "reused", None),
            ("gpu_wait", None, None), ("gpu_end", "ok", None)])
        for start, end in zip(got[::2], got[1::2]):
            self.assertEqual(start["id"], end["id"])
        self.assertEqual(len({e["id"] for e in got}), 5)
        self.assertEqual(got[1]["usd"], 0.01)
        self.assertNotIn("s", got[7])            # a reused result is never timed
        self.assertEqual(got[9]["m"], {"load_s": 60.4})
        self.assertEqual(got[5]["view"], "body+0")
        self.assertIsNone(events.current_call())

    def test_images_are_announced_only_for_files_the_job_shows(self):
        events.image("views/front.png", step="2")
        events.image("masks/front.png", step="2")
        events.image("p-attempts/0-design.png", step="p")
        self.assertEqual([e["what"] for e in lines(self.path)], ["front"])

    def test_unwritable_target_never_raises(self):
        with mock.patch.dict(os.environ, {events.ENV_FILE: str(Path(self.tmp.name) / "gone" / events.FILE)}):
            self.assertIsNone(events.emit("local", step="0", what="input_checked"))
        self.assertFalse((Path(self.tmp.name) / "gone").exists())


if __name__ == "__main__":
    unittest.main()
