"""Versioned outputs: every stage 3/4 record names the code that built it, and the scorer compares bodies from
anywhere (frozen builds of earlier milestones) and records their `version.json`. Temporary folders only; the
git checks make a throwaway repository of their own."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from herogen.blender import record
from herogen.blender import score as S

from test_herogen_blender_score import ARM, BODY, make_run, write_body

HAVE_GIT = shutil.which("git") is not None


class CodeVersion(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="herogen-version-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @unittest.skipUnless(HAVE_GIT, "git is not installed")
    def test_commit_and_dirty_in_a_checkout(self):
        repo = self.tmp / "repo"
        repo.mkdir()
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "nobody",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "nobody", "GIT_CONFIG_GLOBAL": "/dev/null"}

        def git(*args):
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env, timeout=30)

        git("init", "-q")
        (repo / "code.py").write_text("x = 1\n")
        git("add", "code.py")
        git("-c", "commit.gpgsign=false", "commit", "-q", "-m", "init")
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
        self.assertEqual(record.code_version(repo), {"commit": head, "dirty": False})
        (repo / "untracked.png").write_bytes(b"x")                 # untracked files are not code changes
        self.assertFalse(record.code_version(repo)["dirty"])
        (repo / "code.py").write_text("x = 2\n")
        self.assertEqual(record.code_version(repo), {"commit": head, "dirty": True})

    def test_none_outside_a_checkout_or_without_git(self):
        self.assertIsNone(record.code_version(self.tmp))
        path = os.environ.get("PATH", "")
        os.environ["PATH"] = str(self.tmp)                        # no git on PATH: still no crash
        try:
            self.assertIsNone(record.code_version())
        finally:
            os.environ["PATH"] = path

    def test_stage_records_carry_the_code_version(self):
        rec = record.StageRecord(self.tmp, 3, "base mesh")
        self.assertIn("code", rec.data)
        rec.finish()
        out = json.loads((self.tmp / "stage-3.json").read_text())
        self.assertEqual(out["code"], record.code_version())
        here = record.code_version()
        if here is not None:                                       # these tests run from a checkout
            self.assertEqual(len(here["commit"]), 40)
            self.assertIsInstance(here["dirty"], bool)


class VersionedBodies(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="herogen-versions-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def frozen(self, label: str | None, boxes, extra: dict | None = None) -> Path:
        d = self.tmp / "versions" / "sketch" / (label or "unlabelled")
        d.mkdir(parents=True)
        glb = write_body(d / "high.glb", boxes)
        if label is not None:
            (d / "version.json").write_text(json.dumps({"label": label, "frozen": "2026-10-29T09:00:00-07:00",
                                                        "code": {"commit": "a" * 40, "dirty": False},
                                                        "secret_path": "/somewhere/private", **(extra or {})}))
        return glb

    def test_any_number_of_bodies_from_outside_the_run_with_their_labels(self):
        run = make_run(self.tmp)
        m0 = self.frozen("m0-carve", (BODY,))
        m1 = self.frozen("m1-hybrid", (BODY, ARM), {"code": {"commit": "b" * 40, "dirty": True}})
        bare = self.frozen(None, (BODY, ARM))
        args = ["--run", str(run), "--body", f"m0={m0}", "--body", f"m1={m1}", "--body", f"bare={bare}"]
        self.assertEqual(S.main(args), 0)
        data = json.loads((run / "05-scores.json").read_text())
        by = {b["name"]: b for b in data["bodies"]}
        self.assertEqual(set(by), {"m0", "m1", "bare"})
        self.assertEqual(by["m0"]["version"], {"label": "m0-carve", "frozen": "2026-10-29T09:00:00-07:00",
                                               "code": {"commit": "a" * 40, "dirty": False}})
        self.assertEqual(by["m1"]["version"]["code"], {"commit": "b" * 40, "dirty": True})
        self.assertIsNone(by["bare"]["version"])
        for b in by.values():
            self.assertEqual(b["glb"], "high.glb")               # outside the run: no folder path recorded
            self.assertNotIn("secret_path", json.dumps(b))
        self.assertGreater(by["m1"]["views"][S.HELDOUT_VIEW]["iou"], by["m0"]["views"][S.HELDOUT_VIEW]["iou"])
        self.assertIn("code", data)

    def test_only_the_version_format_fields_are_read(self):
        """`frozen` and `code: {commit, dirty}`, the format frozen versions are written in; the older top-level
        `commit`, `dirty` and `created` are not version.json fields and are not recorded."""
        run = make_run(self.tmp)
        glb = self.frozen("old-style", (BODY,), {"commit": "c" * 40, "dirty": True, "created": "2026-01-01"})
        v = S.score_body(run, "x", glb)["version"]
        self.assertEqual(v["code"], {"commit": "a" * 40, "dirty": False})
        self.assertEqual(v["frozen"], "2026-10-29T09:00:00-07:00")
        for k in ("commit", "dirty", "created"):
            self.assertNotIn(k, v)

    def test_unreadable_version_file_is_reported_not_fatal(self):
        run = make_run(self.tmp)
        glb = self.frozen("x", (BODY,))
        glb.with_name("version.json").write_text("{not json")
        e = S.score_body(run, "x", glb)
        self.assertEqual(e["version"]["label"], None)
        self.assertIn("could not be read", e["version"]["error"])


class AtomicWrites(unittest.TestCase):
    """Records, joints, sections, scores and cache entries are written in one step: a write that fails part-way
    leaves the previous file whole and no temporary file behind."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="herogen-atomic-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_failed_stage_record_write_leaves_the_previous_record(self):
        (self.tmp / "stage-3.json").write_text(json.dumps({"stage": 3, "ok": True}))
        rec = record.StageRecord(self.tmp, 3, "base mesh")
        real = os.replace

        def broken(src, dst):
            raise OSError("disk full")

        record.os.replace = broken
        try:
            with self.assertRaises(OSError):
                rec.finish()
        finally:
            record.os.replace = real
        self.assertEqual(json.loads((self.tmp / "stage-3.json").read_text()), {"stage": 3, "ok": True})
        self.assertEqual(list(self.tmp.glob("*.part")), [])
        rec.finish()
        self.assertEqual(json.loads((self.tmp / "stage-3.json").read_text())["stage"], 3)

    def test_every_writer_goes_through_write_atomic(self):
        import inspect

        from herogen.blender import body_joints, score, stage4
        from herogen.blender.remote import client

        self.assertIn("write_atomic(path", inspect.getsource(record.StageRecord.finish))
        self.assertIn("write_atomic(path", inspect.getsource(body_joints.write_joints))
        self.assertIn("write_atomic(path", inspect.getsource(score.write_scores))
        self.assertIn("return _run_stage4(", inspect.getsource(stage4.run_stage4))
        self.assertIn('write_atomic(body / "sections.json"', inspect.getsource(stage4._run_stage4))
        self.assertIn("write_atomic(entry_path", inspect.getsource(client.RemoteClient._call))


if __name__ == "__main__":
    unittest.main()
