"""Version API ownership, admission, CSRF, immutable copies, stop and continuation."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from test_herogen_service_support import HAVE_WEB, web_test
from test_herogen_service_web import Browser, WebCase

if HAVE_WEB:
    from herogen.service.fakes import FakeRunner
    from herogen.service.store import Conflict, NotFound, new_id
    from herogen.service.worker import run_job


@web_test
class TestVersions(WebCase):
    def setUp(self):
        super().setUp()
        self.b = self.signin()
        self.parent = self.make_job(self.b).json()["id"]
        self.assertEqual(self.finish(self.parent), "succeeded")
        self.uid = self.store.job_get(self.parent)["user_id"]

    def redo(self, stage="cover", note="green coat", key="redo-key-0001", browser=None, **kw):
        return (browser or self.b).post(f"/api/jobs/{self.parent}/redo", json={"stage": stage, "note": note},
                                       headers={"Idempotency-Key": key}, **kw)

    def test_cover_redo_stops_and_continue_keeps_paid_work(self):
        before = self.svc.storage.run_dir(self.uid, self.parent).joinpath("01-cover.png").read_bytes()
        r = self.redo()
        self.assertEqual(r.status_code, 201, r.text)
        j = r.json()
        self.assertEqual((j["parent_job"], j["from_stage"], j["note"], j["stop_after"]),
                         (self.parent, "1", "green coat", "1"))
        runner = FakeRunner()
        self.assertEqual(self.finish(j["id"], runner), "stopped")
        self.assertEqual(runner.calls, ["1"])
        rows = {r["stage"]: r for r in self.store.job_stages(j["id"])}
        self.assertEqual(rows["0"]["cost_usd"], 0)
        self.assertAlmostEqual(self.store.job_get(j["id"])["cost_usd"], .0125)
        used = self.b.get("/api/quota").json()["jobs_total"]
        r = self.b.post(f"/api/jobs/{j['id']}/continue")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["stop_after"], None)
        runner.calls.clear()
        self.assertEqual(self.finish(j["id"], runner), "succeeded")
        self.assertEqual(runner.calls, ["2", "3", "3f", "4", "6", "5"])
        self.assertAlmostEqual(self.store.job_get(j["id"])["cost_usd"], .0375)
        self.assertEqual(self.b.get("/api/quota").json()["jobs_total"], used)
        self.assertEqual(self.svc.storage.run_dir(self.uid, self.parent).joinpath("01-cover.png").read_bytes(), before)

    def test_turnaround_redo_reuses_cover_and_regenerates_whole_sheet(self):
        r = self.redo("turnaround", "left view: satchel missing")
        self.assertEqual(r.status_code, 201, r.text)
        runner = FakeRunner()
        self.assertEqual(self.finish(r.json()["id"], runner), "stopped")
        self.assertEqual(runner.calls, ["2"])
        run = self.svc.storage.run_dir(self.uid, r.json()["id"])
        self.assertEqual((run / "01-cover.png").read_bytes(),
                         (self.svc.storage.run_dir(self.uid, self.parent) / "01-cover.png").read_bytes())
        self.assertTrue(all((run / "views" / f"{v}.png").is_file() for v in ("front", "right", "back", "left")))
        self.assertAlmostEqual(self.store.job_get(r.json()["id"])["cost_usd"], .025)

    def test_unconfirmed_copy_is_kept_then_purged_only_without_a_job(self):
        from unittest.mock import patch
        import time
        from herogen.service.store import StoreError
        from herogen.service.worker import purge

        create = self.store.job_create
        def busy(**kwargs):
            raise StoreError("The service is busy.", "busy")
        with patch.object(self.store, "job_create", busy):
            self.assertEqual(self.redo().status_code, 503)
        runs = self.svc.storage.runs_root(self.uid)
        orphan = next(p for p in runs.iterdir() if p.name != self.parent)
        old = time.time() - 2 * 86400
        os.utime(orphan, (old, old))
        os.utime(runs / self.parent, (old, old))
        out = purge(self.store, self.s, self.svc.storage)
        self.assertEqual(out["orphan_runs"], 1)
        self.assertFalse(orphan.exists())
        self.assertTrue((runs / self.parent).exists(), "an owned old folder is retained")
        def saved_then_busy(**kwargs):
            create(**kwargs)
            raise StoreError("The service is busy.", "busy")
        with patch.object(self.store, "job_create", saved_then_busy):
            self.assertEqual(self.redo().status_code, 503)
        child = next(j for j in self.store.jobs_list(self.uid) if j["id"] != self.parent)
        os.utime(runs / child["id"], (old, old))
        self.assertEqual(purge(self.store, self.s, self.svc.storage)["orphan_runs"], 0)
        self.assertTrue((runs / child["id"]).exists())
        again = self.redo()
        self.assertEqual((again.status_code, again.json()["id"]), (200, child["id"]))

    def test_redo_is_idempotent_and_changed_request_conflicts(self):
        first = self.redo()
        second = self.redo()
        self.assertEqual((first.status_code, second.status_code), (201, 200))
        self.assertEqual(first.json()["id"], second.json()["id"])
        self.assertEqual(self.redo(note="red coat").status_code, 409)
        self.assertEqual(len(list(self.svc.storage.runs_root(self.uid).iterdir())), 2)

    def test_versions_include_parent_siblings_and_descendants(self):
        first = self.redo().json()["id"]
        self.finish(first)
        second = self.redo(key="redo-key-0002").json()["id"]
        self.finish(second)
        self.store.user_set_quota("alice@example.com", 8, 8)
        child = self.b.post(f"/api/jobs/{first}/redo", json={"stage": "cover", "note": "blue coat"},
                            headers={"Idempotency-Key": "redo-key-0003"}).json()["id"]
        for target in (self.parent, first, second, child):
            r = self.b.get(f"/api/jobs/{target}/versions")
            self.assertEqual(r.status_code, 200)
            versions = r.json()["versions"]
            self.assertEqual({j["id"] for j in versions}, {self.parent, first, second, child})
            parent = next(j for j in versions if j["id"] == self.parent)
            self.assertEqual(parent["cover_thumbnail_url"], f"/runs/r1/{self.parent}/01-cover.png")
            self.assertEqual(next(j for j in versions if j["id"] == first)["note"], "green coat")
            self.assertEqual(next(j for j in versions if j["id"] == first)["status"], "stopped")

    def test_owner_checks_for_redo_versions_and_continue(self):
        other = self.signin("bob@example.com")
        child = self.redo().json()["id"]
        self.finish(child)
        for path in (f"/api/jobs/{self.parent}/redo", f"/api/jobs/{child}/continue"):
            r = other.post(path, json={"stage": "cover", "note": "green"},
                           headers={"Idempotency-Key": "redo-key-0004"})
            self.assertEqual(r.status_code, 404, r.text)
        self.assertEqual(other.get(f"/api/jobs/{self.parent}/versions").status_code, 404)
        with self.assertRaises(NotFound):
            self.store.job_versions(self.parent, next(u["id"] for u in self.store.users_list() if u["email"] == "bob@example.com"))

    def test_signed_out_and_csrf_checks(self):
        guest = Browser(self.app)
        guest.session()
        for path in (f"/api/jobs/{self.parent}/redo", f"/api/jobs/{self.parent}/continue"):
            self.assertEqual(guest.post(path, json={"stage": "cover"}).status_code, 401)
            self.assertEqual(self.b.post(path, csrf=False).status_code, 403)
            self.assertEqual(self.b.post(path, headers={"Origin": "http://wrong.example"}).status_code, 403)
        self.assertEqual(guest.get(f"/api/jobs/{self.parent}/versions").status_code, 401)

    def test_note_validation_and_cleaning(self):
        for note in (None, {}, 25, "a" * 301, "\u202e", "123!"):
            with self.subTest(note=note):
                r = self.redo(note=note)
                self.assertEqual((r.status_code, r.json()["error"]["code"]), (400, "bad_note"))
        r = self.redo(note="«green»\n\x00coat\u202e")
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(r.json()["note"], "green coat")

    def test_stage_and_idempotency_key_validation(self):
        for stage in ("mesh", 1, [], None):
            self.assertEqual(self.redo(stage=stage).status_code, 400)
        self.assertEqual(self.redo(key="short").status_code, 400)

    def test_daily_total_and_storage_admission(self):
        for daily, total, expected in ((1, 5, "24 hours"), (5, 1, "all 1")):
            self.store.user_set_quota("alice@example.com", daily, total)
            r = self.redo()
            self.assertEqual(r.status_code, 429, r.text)
            self.assertIn(expected, r.json()["error"]["message"])
        self.store.user_set_quota("alice@example.com", 5, 5)
        self.store.db.execute("UPDATE jobs SET bytes=? WHERE id=?", (self.s.user_storage_max_bytes, self.parent))
        self.assertEqual(self.redo().status_code, 429)
        self.assertEqual(len(list(self.svc.storage.runs_root(self.uid).iterdir())), 1, "no refused copies left")

    def test_active_generation_blocks_redo_and_continue_and_normal_create(self):
        child = self.redo().json()["id"]
        self.assertEqual(self.redo(key="redo-key-0002").status_code, 409)
        self.assertEqual(self.make_job(self.b, key="normal-key-0002").status_code, 409)
        self.finish(child)
        other = self.redo(key="redo-key-0002").json()["id"]
        self.assertEqual(self.b.post(f"/api/jobs/{child}/continue").status_code, 409)
        self.store.job_cancel(other, self.uid)
        self.assertEqual(self.b.post(f"/api/jobs/{child}/continue").status_code, 200)

    def test_continue_is_idempotent_and_storage_checked(self):
        child = self.redo().json()["id"]
        self.finish(child)
        self.store.db.execute("UPDATE jobs SET bytes=? WHERE id=?", (self.s.user_storage_max_bytes, child))
        self.assertEqual(self.b.post(f"/api/jobs/{child}/continue").status_code, 429)
        self.store.db.execute("UPDATE jobs SET bytes=1 WHERE id=?", (child,))
        self.assertEqual(self.b.post(f"/api/jobs/{child}/continue").status_code, 200)
        self.assertEqual(self.b.post(f"/api/jobs/{child}/continue").status_code, 200)
        self.assertEqual(self.b.post(f"/api/jobs/{self.parent}/continue").status_code, 409)

    def test_continuation_has_its_own_retry_allowance_without_resetting_fences(self):
        child = self.redo().json()["id"]
        self.assertEqual(self.finish(child), "stopped")
        old_attempt = self.store.job_get(child)["attempts"]
        self.b.post(f"/api/jobs/{child}/continue")
        self.assertEqual(self.finish(child, FakeRunner(fail={"2": "transient"})), "queued")
        self.assertEqual(self.finish(child), "succeeded")
        self.assertEqual(self.store.job_get(child)["attempts"], old_attempt + 2)
        self.assertFalse(self.store.job_stage_set(child, old_attempt, "1", "failed"))

    def test_failed_gate_rejects_version_and_cannot_continue(self):
        child = self.redo().json()["id"]
        self.assertEqual(self.finish(child, FakeRunner(fail={"1": "reject"})), "rejected")
        self.assertEqual(self.b.post(f"/api/jobs/{child}/continue").status_code, 409)
        self.assertEqual(self.store.job_get(self.parent)["state"], "succeeded")

    def test_missing_output_rejected_without_new_job(self):
        (self.svc.storage.run_dir(self.uid, self.parent) / "00-sketch.png").unlink()
        r = self.redo()
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (409, "not_ready"))
        self.assertEqual(len(self.store.jobs_list(self.uid)), 1)

    def test_deleted_parent_does_not_erase_owned_descendants(self):
        child = self.redo().json()["id"]
        self.finish(child)
        self.b.send("DELETE", f"/api/jobs/{self.parent}")
        r = self.b.get(f"/api/jobs/{child}/versions")
        self.assertEqual([j["id"] for j in r.json()["versions"]], [child])
        self.assertEqual(self.b.get(f"/api/jobs/{self.parent}/versions").status_code, 404)

    def test_redo_never_starts_side_work_or_later_groups(self):
        child = self.redo("turnaround").json()["id"]
        groups, side = [], []
        from herogen.service.worker import InPlaceWorkdir, make_context, run_group
        runner = FakeRunner()
        def execute(job, attempt, group, resume):
            groups.append(group)
            return run_group(self.store, self.s, job, attempt, group, resume, runner,
                             InPlaceWorkdir(make_context(self.s, self.svc.storage, job)))
        self.assertEqual(run_job(self.store, self.s, self.svc.storage, child, "stop", execute,
                                 heldout=side.append), "stopped")
        self.assertEqual((groups, side), (["model"], []))

    def test_parallel_redos_admit_only_one_job(self):
        parent = self.store.job_get(self.parent)
        def create(i):
            try:
                return self.store.job_create(job_id=new_id("j"), user_id=self.uid, client_key=f"parallel-{i}",
                       kind=parent["kind"], profile=parent["profile"], input_sha=parent["input_sha"],
                       prompt=parent["prompt"], name=parent["name"], parent_job=self.parent, from_stage="1",
                       note="green coat", jobs_per_day=10, jobs_total=10, storage_max_bytes=10**10)[1]
            except Conflict:
                return False
        with ThreadPoolExecutor(8) as pool:
            self.assertEqual(sum(pool.map(create, range(8))), 1)
