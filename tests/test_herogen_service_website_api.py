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
        self.assertEqual(r.headers["content-type"], "image/webp")
        return r

    def assertShows(self, response, source: bytes) -> None:
        """The response is a WebP of `source`: the same picture, at most 512 px on its long side and never
        enlarged, within lossy-compression error."""
        import io
        from PIL import Image, ImageChops, ImageStat

        shown = Image.open(io.BytesIO(response.content))
        original = Image.open(io.BytesIO(source)).convert("RGB")
        self.assertEqual(shown.format, "WEBP")
        scale = min(1, 512 / max(original.size))
        self.assertEqual(max(shown.size), round(max(original.size) * scale))
        diff = ImageStat.Stat(ImageChops.difference(shown.convert("RGB"), original.resize(shown.size))).mean
        self.assertLess(max(diff), 8, diff)

    def test_cover_then_sketch_then_upload_then_none(self):
        upload = self.svc.storage.input_path(self.uid, self.jid).read_bytes()
        url = self.listed()[self.jid]["thumbnail_url"]
        self.assertRegex(url, rf"^/api/jobs/{self.jid}/thumbnail\?v=[0-9a-f]{{16}}$")
        self.assertShows(self.fetch(url), upload)   # a queued upload shows the uploaded sketch
        self.assertEqual(self.alice.get(f"/api/jobs/{self.jid}").json()["thumbnail_url"], url)

        sketch = self.write("0", "00-sketch.png", (200, 10, 10))
        url_sketch = self.listed()[self.jid]["thumbnail_url"]
        self.assertNotEqual(url_sketch, url, "the URL changes with the image")
        self.assertShows(self.fetch(url_sketch), sketch)

        cover = self.write("1", "01-cover.png", (10, 200, 10))
        url_cover = self.listed()[self.jid]["thumbnail_url"]
        self.assertShows(self.fetch(url_cover), cover)
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
        self.assertShows(self.fetch(url), (self.run / "01-cover.png").read_bytes())

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
        self.assertShows(self.fetch(self.listed()[self.jid]["thumbnail_url"]), upload.read_bytes())
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

    def put_cover(self, content: bytes) -> None:
        self.run.mkdir(parents=True, exist_ok=True)
        (self.run / "01-cover.png").write_bytes(content)
        (self.run / "stage-1.json").write_text(json.dumps({"stage": 1, "ok": True, "outputs": ["01-cover.png"]}))

    def thumb_folder(self) -> Path:
        return self.svc.storage.thumb_dir(self.uid, self.jid)

    def thumb_files(self) -> list[str]:
        return sorted(p.name for p in self.thumb_folder().iterdir()) if self.thumb_folder().is_dir() else []

    def test_thumbnail_is_a_512px_webp_made_once_and_cached(self):
        import io
        from PIL import Image

        self.put_cover(png(1400, 900))
        url = self.listed()[self.jid]["thumbnail_url"]
        token = url.split("v=")[1]
        first = self.fetch(url)
        image = Image.open(io.BytesIO(first.content))
        self.assertEqual((image.format, image.size), ("WEBP", (512, 329)))
        self.assertLess(len(first.content), (self.run / "01-cover.png").stat().st_size)
        self.assertEqual(first.headers["cache-control"], "private, max-age=31536000, immutable")
        self.assertEqual(self.thumb_files(), [f"{token}.webp"])
        stamp = (self.thumb_folder() / f"{token}.webp").stat().st_mtime_ns
        with mock.patch("PIL.Image.open", side_effect=AssertionError("made again")):
            again = self.fetch(url)
        self.assertEqual(again.content, first.content)
        self.assertEqual(self.thumb_files(), [f"{token}.webp"])
        self.assertEqual((self.thumb_folder() / f"{token}.webp").stat().st_mtime_ns, stamp)

    def test_a_new_cover_gets_a_new_version_and_image(self):
        self.put_cover(png(1400, 900))
        old = self.listed()[self.jid]["thumbnail_url"]
        first = self.fetch(old).content
        self.put_cover(png(900, 1300, colour=(200, 0, 0)))
        new = self.listed()[self.jid]["thumbnail_url"]
        self.assertNotEqual(new, old)
        second = self.fetch(new)
        self.assertNotEqual(second.content, first)
        self.assertEqual(second.headers["cache-control"], "private, max-age=31536000, immutable")
        self.assertEqual(self.thumb_files(), [f"{new.split('v=')[1]}.webp"], "the older copy is removed")
        stale = self.alice.get(old)   # an old link shows what is there now, without the long cache
        self.assertEqual((stale.status_code, stale.content, stale.headers["cache-control"]),
                         (200, second.content, "private, no-cache"))

    def test_another_user_gets_404_even_when_cached(self):
        self.put_cover(png(1400, 900))
        url = self.listed()[self.jid]["thumbnail_url"]
        self.fetch(url)
        bob = self.signin("bob@example.com")
        self.assertEqual(bob.get(url).status_code, 404)
        self.assertEqual(bob.get(f"/api/jobs/{self.jid}/thumbnail").status_code, 404)

    def test_corrupt_and_oversized_images_are_a_404_not_a_500(self):
        import io
        from PIL import Image

        self.put_cover(b"\x89PNG\r\n\x1a\n not really a picture")
        self.assertEqual(self.alice.get(self.listed()[self.jid]["thumbnail_url"]).status_code, 404)
        self.put_cover(png(300, 300)[:200])   # cut short
        self.assertEqual(self.alice.get(self.listed()[self.jid]["thumbnail_url"]).status_code, 404)
        bomb = io.BytesIO()
        Image.new("1", (9000, 9000)).save(bomb, format="PNG")   # 81 million pixels in a few KB
        self.assertLess(len(bomb.getvalue()), 100_000)
        self.put_cover(bomb.getvalue())
        with mock.patch("PIL.PngImagePlugin.PngImageFile.load", side_effect=AssertionError("decoded a bomb")):
            self.assertEqual(self.alice.get(self.listed()[self.jid]["thumbnail_url"]).status_code, 404)
        self.assertEqual(self.thumb_files(), [])
        self.put_cover(png(400, 400))   # a good cover afterwards works
        self.assertShows(self.fetch(self.listed()[self.jid]["thumbnail_url"]), png(400, 400))

    def test_deleting_the_hero_removes_its_thumbnail(self):
        self.put_cover(png(1400, 900))
        url = self.listed()[self.jid]["thumbnail_url"]
        self.fetch(url)
        self.assertEqual(len(self.thumb_files()), 1)
        self.assertEqual(self.alice.send("DELETE", f"/api/jobs/{self.jid}").status_code, 200)
        self.assertFalse(self.thumb_folder().exists())
        self.assertEqual(self.alice.get(url).status_code, 404)

    def test_files_are_read_and_written_under_the_volume_lock(self):
        """A Modal volume cannot reload while any file on it is open: the source and the cached copy are read and
        the copy is written with the read lock held, the decode holds none, and the commit comes after the write."""
        import os
        from herogen.service import gallery
        from herogen.service.storage import ModalVolumeView, Storage

        class Vol:
            def reload(self):
                pass

            def commit(self):
                events.append(("commit", view.readers))

        events: list[tuple[str, int]] = []
        view = ModalVolumeView(Vol())
        self.svc.storage = Storage(self.s.data_dir, view)
        self.put_cover(png(1400, 900))
        url = self.listed()[self.jid]["thumbnail_url"]
        data = str(self.s.data_dir.resolve())
        orig_read, orig_replace, orig_decode = Path.read_bytes, os.replace, gallery.webp_thumbnail

        def spy_read(path):
            if str(path.resolve()).startswith(data):
                events.append((f"read {path.suffix}", view.readers))
            return orig_read(path)

        def spy_replace(src, dst, *a, **kw):
            if str(dst).startswith(data):
                events.append(("write", view.readers))
            return orig_replace(src, dst, *a, **kw)

        def spy_decode(raw):
            events.append(("decode", view.readers))
            return orig_decode(raw)

        with mock.patch.object(Path, "read_bytes", spy_read), mock.patch("os.replace", spy_replace), \
                mock.patch.object(gallery, "webp_thumbnail", spy_decode):
            made = self.fetch(url)
            events.append(("second request", view.readers))
            cached = self.fetch(url)
        self.assertEqual(events, [("read .png", 1), ("decode", 0), ("write", 1), ("commit", 0),
                                  ("second request", 0), ("read .webp", 1)])
        self.assertEqual(cached.content, made.content)
        self.assertEqual(view.readers, 0, "lock released")


@web_test
class QueuePositionTest(WebCase):
    def setUp(self) -> None:
        super().setUp()
        self.alice = self.signin()
        self.bob = self.signin("bob@example.com")
        self.n = 0

    def create(self, b) -> str:
        self.n += 1
        r = self.make_job(b, key=f"key-q{self.n:07d}")
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()["id"]

    def place(self, b, jid) -> tuple:
        v = b.get(f"/api/jobs/{jid}").json()
        listed = {j["id"]: j for j in b.get("/api/jobs").json()["jobs"]}[jid]
        self.assertEqual((listed["queue_position"], listed["queued_ahead"]), (v["queue_position"], v["queued_ahead"]))
        return v["queue_position"], v["queued_ahead"]

    def line(self, jobs) -> list[str]:
        """Waiting jobs in the order the service reports, from each owner's own view."""
        places = {jid: self.place(b, jid)[0] for b, jid in jobs}
        return sorted((j for j, p in places.items() if p), key=places.get)

    def test_positions_follow_the_dispatch_order_across_workspaces(self):
        from herogen.service.worker import dispatch_queued
        a1, a2 = self.create(self.alice), self.create(self.alice)
        self.assertEqual(self.submitted, [a1, a2], "two workers start; the rest wait")
        b1, a3, b2 = self.create(self.bob), self.create(self.alice), self.create(self.bob)
        jobs = [(self.alice, a1), (self.alice, a2), (self.bob, b1), (self.alice, a3), (self.bob, b2)]
        self.assertEqual(self.place(self.alice, a1), (0, 0), "dispatched: a worker is starting it")
        self.assertEqual(self.place(self.bob, b1), (1, 0))
        self.assertEqual(self.place(self.alice, a3), (2, 1))
        self.assertEqual(self.place(self.bob, b2), (3, 2))
        body = self.bob.get("/api/jobs").text + self.bob.get(f"/api/jobs/{b2}").text
        for other in (a1, a2, a3):
            self.assertNotIn(other, body, "only counts cross workspaces")

        # Each freed slot goes to the job that stood first.
        self.alice.post(f"/api/jobs/{a1}/cancel", json={})
        expected = self.line(jobs)
        self.submitted.clear()
        dispatch_queued(self.store, self.s, self.svc.submit)
        self.assertEqual(self.submitted, expected[:1])
        self.assertEqual(self.place(self.bob, b1), (0, 0))
        self.assertEqual(self.place(self.alice, a3), (1, 0))

        # Unclaimed dispatches go back in line at their own place once their grace runs out.
        self.clock.advance(301)
        expected = self.line(jobs)
        self.assertEqual(expected, [a2, b1, a3, b2])
        self.submitted.clear()
        dispatch_queued(self.store, self.s, self.svc.submit)
        self.assertEqual(self.submitted, expected[:2])

    def test_only_queued_jobs_report_a_place(self):
        a1 = self.create(self.alice)
        self.assertEqual(self.place(self.alice, a1), (0, 0))
        self.finish(a1)
        self.assertEqual(self.store.job_get(a1)["state"], "succeeded")
        self.assertEqual(self.place(self.alice, a1), (None, None))
        a2, a3, a4 = self.create(self.alice), self.create(self.alice), self.create(self.alice)
        self.alice.post(f"/api/jobs/{a4}/cancel", json={})
        self.assertEqual(self.place(self.alice, a4), (None, None))
        self.alice.send("DELETE", f"/api/jobs/{a2}", json={})
        self.assertEqual(self.store.queue_positions([a2, a3, "j_notanid", "../x"]), {a3: 0})

    def test_running_jobs_report_no_place(self):
        a1 = self.create(self.alice)
        self.assertIsNotNone(self.store.job_claim(a1, "call-1", 600, 3))
        self.assertEqual(self.place(self.alice, a1), (None, None))


if __name__ == "__main__":
    unittest.main()
