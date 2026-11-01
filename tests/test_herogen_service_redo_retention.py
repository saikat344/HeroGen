"""Redo appearance text is forgotten on deletion and retention expiry."""

from herogen.pipeline.redo import fork_run
from herogen.service.store import new_id
from herogen.service.worker import purge
from tests.test_herogen_service_jobs import JobCase


class RedoRetentionTest(JobCase):
    def child(self):
        parent = self.new_job("prompt", name="Ranger")
        self.assertEqual(self.go(parent), "succeeded")
        child_id = new_id("j")
        note = f"scarlet hood for {self.user['email']}"
        fork_run(self.storage.runs_root(self.user["id"]), child_id,
                 self.storage.run_dir(self.user["id"], parent["id"]), 1, note)
        child, _ = self.store.job_create(
            job_id=child_id, user_id=self.user["id"], client_key="redo-retention-0001",
            kind=parent["kind"], profile=parent["profile"], input_sha=parent["input_sha"],
            prompt=parent["prompt"], name=parent["name"], parent_job=parent["id"], from_stage="1",
            note=note, jobs_per_day=10, jobs_total=10, storage_max_bytes=10**10)
        self.assertEqual(self.go(child), "stopped")
        self.assertEqual(self.store.job_get(child_id)["note"], note)
        return child_id

    def assert_forgotten(self, child_id):
        self.assertEqual(self.store.job_get(child_id)["note"], "")
        admin = next(j for j in self.store.jobs_admin_list() if j["id"] == child_id)
        self.assertEqual(admin["note"], "")
        self.assertEqual(self.store.job_get(child_id)["counts_quota"], 1)

    def test_stopped_redo_retention_purges_note_with_other_inputs(self):
        child_id = self.child()
        self.clock.advance(self.s.retention_days * 86400 + 1)
        self.assertIn(child_id, purge(self.store, self.s, self.storage)["jobs"])
        self.assert_forgotten(child_id)
        row = self.store.job_get(child_id)
        self.assertEqual((row["bytes"], row["prompt"], row["name"]), (-1, "", ""))

    def test_job_deletion_forgets_note_before_file_cleanup(self):
        child_id = self.child()
        deleted = self.store.job_delete(child_id, self.user["id"])
        self.assertEqual(deleted["note"], "")
        self.assert_forgotten(child_id)

    def test_account_deletion_forgets_notes_including_deleted_jobs(self):
        child_id = self.child()
        self.store.job_delete(child_id, self.user["id"])
        # A previously deleted row still belongs to the account's erasure scope.
        self.store.db.execute("UPDATE jobs SET note=? WHERE id=?", (self.user["email"], child_id))
        self.assertIn(child_id, self.store.user_delete(self.user["id"]))
        self.assert_forgotten(child_id)
        self.assertNotIn(self.user["email"], str([dict(r) for r in self.store.db.execute("SELECT * FROM jobs")]))
