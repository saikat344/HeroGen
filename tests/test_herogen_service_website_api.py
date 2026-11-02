"""Website data the hero list, gallery and landing page read: hero card thumbnails, small gallery images and
queue positions."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from test_herogen_service_support import HAVE_WEB, png, web_test
from test_herogen_service_web import WebCase


@web_test
class HeroThumbnailTest(WebCase):
    def setUp(self) -> None:
        super().setUp()
        self.alice = self.signin()
        self.job = self.make_job(self.alice).json()
        self.jid = self.job["id"]
        self.uid = self.store.job_get(self.jid)["user_id"]
        self.run = self.svc.storage.run_dir(self.uid, self.jid)

    def listed(self, browser=None) -> dict:
        jobs = (browser or self.alice).get("/api/jobs").json()["jobs"]
        return {j["id"]: j for j in jobs}

    def write(self, stage: str, name: str, colour: tuple[int, int, int]) -> bytes:
        from PIL import Image
        import io

        self.run.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        Image.new("RGB", (64, 96), colour).save(buf, format="PNG")
        (self.run / name).write_bytes(buf.getvalue())
        (self.run / f"stage-{stage}.json").write_text(json.dumps({"stage": int(stage), "ok": True,
                                                                   "outputs": [name]}))
        return buf.getvalue()

    def fetch(self, url: str, browser=None):
        r = (browser or self.alice).get(url)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers["content-type"], "image/png")
        return r

    def test_cover_then_sketch_then_upload_then_none(self):
        upload = self.svc.storage.input_path(self.uid, self.jid).read_bytes()
        url = self.listed()[self.jid]["thumbnail_url"]
        self.assertRegex(url, rf"^/api/jobs/{self.jid}/thumbnail\?v=[0-9a-f]{{16}}$")
        self.assertEqual(self.fetch(url).content, upload, "a queued upload shows the uploaded sketch")
        self.assertEqual(self.alice.get(f"/api/jobs/{self.jid}").json()["thumbnail_url"], url)

        sketch = self.write("0", "00-sketch.png", (200, 10, 10))
        url_sketch = self.listed()[self.jid]["thumbnail_url"]
        self.assertNotEqual(url_sketch, url, "the URL changes with the image")
        self.assertEqual(self.fetch(url_sketch).content, sketch)

        cover = self.write("1", "01-cover.png", (10, 200, 10))
        url_cover = self.listed()[self.jid]["thumbnail_url"]
        self.assertEqual(self.fetch(url_cover).content, cover)
        self.assertEqual(self.alice.get(f"/api/jobs/{self.jid}").json()["thumbnail_url"], url_cover)

        # A cover being redrawn is not shown, as in the job's own images: the sketch is.
        with self.store.lock:
            self.store.db.execute("INSERT INTO job_stages(job_id, stage, state, attempt) VALUES (?, '1', 'running', 1)",
                                  (self.jid,))
        self.assertEqual(self.listed()[self.jid]["thumbnail_url"], url_sketch)
        # A record still marked running does not count either.
        (self.run / "stage-0.json").write_text(json.dumps({"stage": 0, "status": "running"}))
        self.assertEqual(self.listed()[self.jid]["thumbnail_url"], url)

        prompt = self.alice.post("/api/jobs", data={"kind": "prompt", "prompt": "a knight"},
                                 files={"x": ("a", b"a")}, headers={"Idempotency-Key": "key-00000002"}).json()
        self.assertIsNone(self.listed()[prompt["id"]]["thumbnail_url"], "nothing to show yet")
        self.assertEqual(self.alice.get(f"/api/jobs/{prompt['id']}/thumbnail").status_code, 404)

    def test_finished_hero_shows_its_cover(self):
        self.finish(self.jid)
        url = self.listed()[self.jid]["thumbnail_url"]
        self.assertEqual(self.fetch(url).content, (self.run / "01-cover.png").read_bytes())

    def test_cache_headers_follow_the_version(self):
        url = self.listed()[self.jid]["thumbnail_url"]
        self.assertEqual(self.fetch(url).headers["cache-control"], "private, max-age=31536000, immutable")
        stale = self.fetch(f"/api/jobs/{self.jid}/thumbnail?v=0000000000000000")
        self.assertEqual(stale.headers["cache-control"], "private, no-cache")

    def test_other_workspaces_never_see_the_image(self):
        self.finish(self.jid)
        url = self.listed()[self.jid]["thumbnail_url"]
        bob = self.signin("bob@example.com")
        self.assertEqual(bob.get(url).status_code, 404)
        self.assertEqual(bob.get(f"/api/jobs/{self.jid}/thumbnail").status_code, 404)
        self.assertEqual(bob.get("/api/jobs").json()["jobs"], [])
        bob_job = self.make_job(bob, key="key-00000003").json()
        self.assertEqual(self.alice.get(self.listed(bob)[bob_job["id"]]["thumbnail_url"]).status_code, 404)
        from test_herogen_service_web import Browser
        self.assertEqual(Browser(self.app).get(url).status_code, 401)
        self.assertEqual(self.alice.get("/api/jobs/j_notanid/thumbnail").status_code, 404)

    def test_linked_images_are_refused(self):
        outside = self.tmp / "outside.png"
        outside.write_bytes(png())
        self.run.mkdir(parents=True, exist_ok=True)
        (self.run / "stage-1.json").write_text(json.dumps({"stage": 1, "ok": True}))
        (self.run / "01-cover.png").symlink_to(outside)
        upload = self.svc.storage.input_path(self.uid, self.jid)
        self.assertEqual(self.fetch(self.listed()[self.jid]["thumbnail_url"]).content, upload.read_bytes())
        upload.unlink()
        upload.symlink_to(outside)
        self.assertIsNone(self.listed()[self.jid]["thumbnail_url"])

    def test_listing_reads_no_images_and_lists_no_folders(self):
        self.finish(self.jid)
        for i in range(3):
            self.make_job(self.alice, key=f"key-1000000{i}")
        with mock.patch.object(Path, "glob", side_effect=AssertionError("folder listing")), \
                mock.patch.object(Path, "iterdir", side_effect=AssertionError("folder listing")), \
                mock.patch("herogen.service.web.version_token", side_effect=AssertionError("image hashing")), \
                mock.patch.object(self.store, "job_stages", side_effect=AssertionError("one query per job")):
            jobs = self.listed()
        self.assertEqual(len(jobs), 4)
        self.assertTrue(all(j["thumbnail_url"] for j in jobs.values()))


if __name__ == "__main__":
    unittest.main()
