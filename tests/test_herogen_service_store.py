"""Metadata store: job state machine, idempotency, quotas, claims and attempts, sessions, retention, snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import secrets
import shutil
import sqlite3
import sys
import threading
from pathlib import Path
from typing import ClassVar

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from test_herogen_service_support import Clock, TempDirCase, make_user

from herogen.service.snapshot import SnapshotConflict, Snapshotter, SnapshotUnreadable, Superseded
from herogen.service.store import (
    TRANSITIONS,
    Conflict,
    InvalidTransition,
    NotFound,
    QuotaExceeded,
    Store,
    StoreError,
    new_id,
    valid_id,
)


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class StoreCase(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        self.clock = Clock()
        self.store = Store(self.tmp / "meta.db", clock=self.clock)
        self.user = make_user(self.store)

    def tearDown(self) -> None:
        self.store.close()
        super().tearDown()

    def create(self, key="k" * 8, **kw):
        args = dict(job_id=new_id("j"), user_id=self.user["id"], client_key=key, kind="sketch",
                    profile="roblox-ingame", jobs_per_day=3, jobs_total=5, storage_max_bytes=10 ** 9,
                    input_sha="a" * 64)
        args.update(kw)
        return self.store.job_create(**args)


class TestMethodSets(TempDirCase):
    def test_admin_methods_are_separate(self):
        self.assertFalse(Store.PUBLIC & Store.ADMIN)
        for name in Store.PUBLIC | Store.ADMIN:
            self.assertTrue(callable(getattr(Store, name)), name)
        for name in ("allow_add", "invite_create", "user_set_status", "user_set_quota", "audit_list", "users_list"):
            self.assertIn(name, Store.ADMIN)

    def test_invite_validity_ignores_the_address(self):
        store = Store(self.tmp / "m.db", clock=Clock())
        store.invite_create("a" * 64, 1, 3600, email="bob@example.com")
        self.assertTrue(store.invite_valid("a" * 64))
        self.assertFalse(store.invite_valid("b" * 64))
        for email in ("bob@example.com", "eve@example.com"):
            self.assertFalse(store.signin_eligibility(email, "a" * 64)["invalid_invite"], email)
        self.assertFalse(store.signin_eligibility("eve@example.com", "a" * 64)["eligible"])
        make_user(store, "carol@example.com")
        for email in ("carol@example.com", "nobody@example.com"):
            self.assertTrue(store.signin_eligibility(email, "b" * 64)["invalid_invite"], email)
        store.close()

    def test_allowlist_audit_holds_no_address(self):
        store = Store(self.tmp / "m.db", clock=Clock())
        store.allow_add("bob@example.com", "", "tag123")
        store.allow_remove("bob@example.com", "tag123")
        self.assertNotIn("bob@example.com", json.dumps(store.audit_list()))
        store.close()


class TestIds(TempDirCase):
    def test_ids_are_path_safe_and_typed(self):
        for p in ("u", "j", "i"):
            i = new_id(p)
            self.assertTrue(valid_id(i, p))
            self.assertRegex(i, r"^[uji]_[a-z2-7]{26}$")
        self.assertFalse(valid_id("j_../../etc", "j"))
        self.assertFalse(valid_id(new_id("u"), "j"))
        self.assertFalse(valid_id(None, "j"))
        with self.assertRaises(ValueError):
            new_id("x")

    def test_errors_keep_their_code_across_pickling(self):
        e = pickle.loads(pickle.dumps(StoreError("nope", "account")))
        self.assertEqual((str(e), e.code), ("nope", "account"))
        q = pickle.loads(pickle.dumps(QuotaExceeded("full")))
        self.assertIsInstance(q, QuotaExceeded)
        self.assertEqual(q.code, "quota")


class TestJobs(StoreCase):
    def test_create_is_idempotent_per_key(self):
        a, created = self.create()
        self.assertTrue(created)
        b, created2 = self.create(job_id=new_id("j"))
        self.assertFalse(created2)
        self.assertEqual(a["id"], b["id"])
        with self.assertRaises(Conflict):
            self.create(job_id=new_id("j"), input_sha="b" * 64)
        other = make_user(self.store, "bob@example.com")
        c, created3 = self.create(user_id=other["id"], job_id=new_id("j"))
        self.assertTrue(created3, "keys are per user")
        self.assertNotEqual(c["id"], a["id"])

    def test_daily_and_total_quota(self):
        for i in range(3):
            self.create(key=f"key-{i:04d}")
        with self.assertRaises(QuotaExceeded) as cm:
            self.create(key="key-0003")
        self.assertIn("24 hours", str(cm.exception))
        u = self.store.usage(self.user["id"], 3, 5)
        self.assertEqual((u["jobs_today"], u["jobs_total"]), (3, 3))
        self.assertAlmostEqual(u["next_daily_slot_at"], self.clock.t + 86400)
        self.clock.advance(86401)
        self.create(key="key-0004")
        self.create(key="key-0005")
        with self.assertRaises(QuotaExceeded) as cm:
            self.create(key="key-0006")
        self.assertIn("all 5", str(cm.exception))

    def test_per_user_quota_override_and_refund(self):
        self.store.user_set_quota("alice@example.com", 1, None)
        j, _ = self.create(key="key-0000")
        with self.assertRaises(QuotaExceeded):
            self.create(key="key-0001")
        # A job that fails for our reasons before spending anything gives its quota back.
        claim = self.store.job_claim(j["id"], "c1", 60, 2)
        self.store.job_finish(j["id"], claim["attempt"], "failed", error_code="stage_error", refund=True,
                              error_message="It broke.")
        self.assertEqual(self.store.job_get(j["id"])["error_message"],
                         "It broke. It does not use up one of today's heroes, but it still counts toward your total.")
        self.create(key="key-0002")

    def test_a_job_that_spent_money_is_never_refunded(self):
        j, _ = self.create(key="key-0000")
        c = self.store.job_claim(j["id"], "c", 60, 2)
        self.store.job_stage_set(j["id"], 1, "1", "passed", cost_usd=0.0125)
        self.store.job_finish(j["id"], c["attempt"], "failed", error_code="stage_error", refund=True,
                              error_message="It broke.")
        job = self.store.job_get(j["id"])
        self.assertEqual((job["counts_quota"], job["refunded_at"], job["error_message"]), (1, None, "It broke."))
        k, _ = self.create(key="key-0001")
        c = self.store.job_claim(k["id"], "c", 60, 2)
        self.store.job_finish(k["id"], c["attempt"], "failed", error_code="x", refund=True, cost_usd=0.001)
        self.assertEqual(self.store.job_get(k["id"])["counts_quota"], 1, "a cost reported at the finish counts too")

    def test_out_of_attempts_after_spending_counts(self):
        j, _ = self.create()
        self.store.job_claim(j["id"], "c1", 60, 2)
        self.store.job_stage_set(j["id"], 1, "1", "passed", cost_usd=0.02)
        self.assertEqual(self.store.job_requeue(j["id"], 1, 2, "timeout", "2", "Slow."), "queued")
        self.store.job_claim(j["id"], "c2", 60, 2)
        self.assertEqual(self.store.job_requeue(j["id"], 2, 2, "timeout", "2", "Slow."), "failed")
        self.assertEqual(self.store.job_get(j["id"])["counts_quota"], 1)

    def test_refunds_are_capped_per_day(self):
        self.store.user_set_quota("alice@example.com", 10, 10)
        refunded = []
        for i in range(4):
            j, _ = self.create(key=f"key-{i:04d}")
            c = self.store.job_claim(j["id"], "c", 60, 2)
            self.store.job_finish(j["id"], c["attempt"], "failed", error_code="x", refund=True)
            refunded.append(self.store.job_get(j["id"])["counts_quota"] == 0)
        self.assertEqual(refunded, [True, True, False, False], "at most REFUNDS_PER_DAY (2) a day")
        self.clock.advance(86401)
        j, _ = self.create(key="key-0009")
        c = self.store.job_claim(j["id"], "c", 60, 2)
        self.store.job_finish(j["id"], c["attempt"], "failed", error_code="x", refund=True)
        self.assertEqual(self.store.job_get(j["id"])["counts_quota"], 0)

    def test_refunds_give_back_the_day_not_the_lifetime_total(self):
        self.store.user_set_quota("alice@example.com", 10, 3)
        for i in range(3):
            j, _ = self.create(key=f"key-{i:04d}")
            c = self.store.job_claim(j["id"], "c", 60, 2)
            self.store.job_finish(j["id"], c["attempt"], "failed", error_code="x", refund=True)
        u = self.store.usage(self.user["id"], 3, 5)
        self.assertEqual((u["jobs_today"], u["jobs_total"]), (1, 3), "two refunded today, all three count in total")
        with self.assertRaises(QuotaExceeded) as cm:
            self.create(key="key-0009")
        self.assertIn("all 3", str(cm.exception))

    def test_parallel_creates_never_exceed_the_quota(self):
        import concurrent.futures as cf

        def one(i):
            try:
                return self.create(key=f"race-{i:04d}", job_id=new_id("j"))[1]
            except QuotaExceeded:
                return "quota"

        with cf.ThreadPoolExecutor(16) as ex:
            got = list(ex.map(one, range(16)))
        self.assertEqual(got.count(True), 3)
        self.assertEqual(got.count("quota"), 13)
        self.assertEqual(len(self.store.jobs_list(self.user["id"])), 3)

    def test_storage_limit(self):
        j, _ = self.create(key="key-0000", storage_max_bytes=100)
        c = self.store.job_claim(j["id"], "c", 60, 2)
        self.store.job_finish(j["id"], c["attempt"], "succeeded", bytes_=150)
        with self.assertRaises(QuotaExceeded) as cm:
            self.create(key="key-0001", storage_max_bytes=100)
        self.assertIn("storage", str(cm.exception))

    def test_disabled_account_cannot_create(self):
        self.store.user_set_status("alice@example.com", "disabled")
        with self.assertRaises(StoreError) as cm:
            self.create()
        self.assertEqual(cm.exception.code, "account")

    def test_claim_once_and_fence_stale_attempts(self):
        j, _ = self.create()
        c1 = self.store.job_claim(j["id"], "call-1", 60, 2)
        self.assertEqual(c1["attempt"], 1)
        self.assertIsNone(self.store.job_claim(j["id"], "call-dup", 60, 2), "a duplicate worker does nothing")
        self.assertTrue(self.store.job_current(j["id"], 1))
        self.assertFalse(self.store.job_current(j["id"], 2))
        self.assertTrue(self.store.job_stage_set(j["id"], 1, "0", "passed", seconds=1.0))
        self.assertEqual(self.store.job_requeue(j["id"], 1, 2, "timeout", "1", "slow"), "queued")
        self.assertFalse(self.store.job_stage_set(j["id"], 1, "1", "passed"), "attempt 1 is stale now")
        self.assertFalse(self.store.job_finish(j["id"], 1, "succeeded"))
        c2 = self.store.job_claim(j["id"], "call-2", 60, 2)
        self.assertEqual(c2["attempt"], 2)
        self.assertEqual(self.store.job_requeue(j["id"], 2, 2, "timeout", "1", "slow"), "failed")
        job = self.store.job_get(j["id"])
        self.assertEqual((job["state"], job["counts_quota"]), ("failed", 0))

    def test_out_of_attempts_fails_at_claim_before_paying(self):
        j, _ = self.create()
        self.store.job_claim(j["id"], "c1", 60, 1)
        self.store.job_requeue(j["id"], 1, 5, "x", None, "x")   # a larger limit queued it again
        self.assertIsNone(self.store.job_claim(j["id"], "c2", 60, 1))
        self.assertEqual(self.store.job_get(j["id"])["state"], "failed")

    def test_lease_expiry_requeues_then_fails(self):
        j, _ = self.create()
        self.store.job_claim(j["id"], "c1", 60, 2)
        self.assertEqual(self.store.reconcile_stale(2), [])
        self.clock.advance(61)
        self.assertEqual(self.store.reconcile_stale(2), [(j["id"], "queued")])
        self.store.job_claim(j["id"], "c2", 60, 2)
        self.clock.advance(61)
        self.assertEqual(self.store.reconcile_stale(2), [(j["id"], "failed")])

    def test_stage_lease_extends_job_lease(self):
        j, _ = self.create()
        self.store.job_claim(j["id"], "c1", 60, 2)
        self.store.job_stage_set(j["id"], 1, "3", "running", lease_s=7200)
        self.clock.advance(3600)
        self.assertEqual(self.store.reconcile_stale(2), [])
        self.assertEqual(self.store.job_get(j["id"])["stage"], "3")

    def test_group_enter_limits_restarts(self):
        j, _ = self.create()
        self.store.job_claim(j["id"], "c1", 60, 2)
        self.assertTrue(self.store.group_enter(j["id"], 1, "model", 2))
        self.assertTrue(self.store.group_enter(j["id"], 1, "model", 2))
        self.assertFalse(self.store.group_enter(j["id"], 1, "model", 2))
        self.assertTrue(self.store.group_enter(j["id"], 1, "blender", 2))
        self.assertFalse(self.store.group_enter(j["id"], 2, "export", 2), "not the live attempt")

    def test_transitions_are_enforced(self):
        j, _ = self.create()
        c = self.store.job_claim(j["id"], "c", 60, 2)
        self.store.job_finish(j["id"], c["attempt"], "succeeded")
        with self.store._tx() as cx:
            row = cx.execute("SELECT * FROM jobs WHERE id=?", (j["id"],)).fetchone()
            for new in ("running", "queued", "failed", "canceled"):
                with self.assertRaises(InvalidTransition):
                    self.store._set_state(cx, row, new)
        for state, nxt in TRANSITIONS.items():
            if state in ("succeeded", "rejected", "failed", "canceled"):
                self.assertEqual(nxt, set(), f"{state} is final")
        with self.assertRaises(StoreError):
            self.store.job_finish(j["id"], 1, "queued")

    def test_cancel_refunds_a_job_that_never_started(self):
        j, _ = self.create()
        out = self.store.job_cancel(j["id"], self.user["id"])
        self.assertEqual((out["state"], out["counts_quota"]), ("canceled", 0))
        k, _ = self.create(key="key-0001")
        self.store.job_claim(k["id"], "c", 60, 2)
        out = self.store.job_cancel(k["id"], self.user["id"])
        self.assertEqual((out["state"], out["counts_quota"]), ("canceled", 1))
        self.assertFalse(self.store.job_current(k["id"], 1), "a canceled job's worker stops")

    def test_jobs_are_private_to_their_owner(self):
        j, _ = self.create()
        other = make_user(self.store, "bob@example.com")
        self.assertIsNone(self.store.job_get(j["id"], other["id"]))
        with self.assertRaises(NotFound):
            self.store.job_cancel(j["id"], other["id"])
        with self.assertRaises(NotFound):
            self.store.job_delete(j["id"], other["id"])
        self.assertEqual(self.store.jobs_list(other["id"]), [])

    def test_dispatch_respects_running_cap_and_grace(self):
        jobs = [self.create(key=f"key-{i:04d}")[0] for i in range(3)]
        due = self.store.jobs_to_dispatch(2)
        self.assertEqual([d["id"] for d in due], [jobs[0]["id"], jobs[1]["id"]])
        for d in due:
            self.store.job_dispatch(d["id"], "call")
        self.assertEqual(self.store.jobs_to_dispatch(2), [], "two are starting")
        self.clock.advance(301)
        self.assertEqual(len(self.store.jobs_to_dispatch(2)), 2, "never claimed: dispatch again after the grace")

    def test_retention_and_purge(self):
        j, _ = self.create()
        c = self.store.job_claim(j["id"], "c", 60, 2)
        self.store.job_finish(j["id"], c["attempt"], "succeeded", bytes_=10)
        self.assertEqual(self.store.retention_due(30 * 86400), [])
        self.clock.advance(31 * 86400)
        due = self.store.retention_due(30 * 86400)
        self.assertEqual([d["id"] for d in due], [j["id"]])
        self.store.job_mark_purged(j["id"])
        self.assertEqual(self.store.retention_due(30 * 86400), [])
        self.assertIsNone(self.store.job_get(j["id"], self.user["id"]), "purged jobs are gone for the user")
        self.assertEqual(self.store.usage(self.user["id"], 3, 5)["jobs_total"], 1, "they still count for quota")
        counts = self.store.purge_expired(86400)
        self.assertGreaterEqual(counts["audit"], 1)

    def test_account_deletion_forgets_the_address_everywhere(self):
        self.store.invite_create("f" * 64, 1, 3600, email="alice@example.com")
        self.store.audit_add("system", "test", "alice@example.com", {"note": "alice@example.com was here"})
        self.store.user_delete(self.user["id"])
        blob = json.dumps([dict(r) for t in ("users", "allowlist", "invites", "audit", "login_tokens")
                           for r in self.store.db.execute(f"SELECT * FROM {t}")])
        self.assertNotIn("alice@example.com", blob)
        self.assertIsNotNone(self.store.invite_list()[0]["revoked_at"], "the bound invite cannot pass to anyone")

    def test_account_deletion(self):
        j, _ = self.create()
        self.store.job_claim(j["id"], "c", 60, 2)
        ids = self.store.user_delete(self.user["id"])
        self.assertEqual(ids, [j["id"]])
        self.assertIsNone(self.store.user_get(self.user["id"]))
        job = self.store.job_get(j["id"])
        self.assertEqual(job["state"], "canceled")
        self.assertIsNotNone(job["deleted_at"])
        self.assertIsNone(self.store.session_get(h("ses-alice@example.com"), 3600))
        emails = [r[0] for r in self.store.db.execute("SELECT email FROM users")]
        self.assertNotIn("alice@example.com", emails, "the address is forgotten")
        # The address can be invited again as a new account.
        again = make_user(self.store, "alice@example.com", session="ses-again")
        self.assertNotEqual(again["id"], self.user["id"])


class TestSessions(StoreCase):
    def test_expiry_idle_and_revocation(self):
        sess = h("ses-alice@example.com")
        self.assertEqual(self.store.session_get(sess, 1000)["id"], self.user["id"])
        self.clock.advance(900)
        self.assertIsNotNone(self.store.session_get(sess, 1000), "seen within the idle window")
        self.clock.advance(999)
        self.assertIsNotNone(self.store.session_get(sess, 1000), "last-seen moved forward")
        self.clock.advance(1001)
        self.assertIsNone(self.store.session_get(sess, 1000), "idle too long")
        self.clock.advance(-1001)
        self.clock.advance(3600)
        self.assertIsNone(self.store.session_get(sess, 10 ** 6), "absolute expiry")

    def test_disable_revokes_sessions(self):
        sess = h("ses-alice@example.com")
        self.store.user_set_status("alice@example.com", "disabled")
        self.assertIsNone(self.store.session_get(sess, 1000))
        self.store.user_set_status("alice@example.com", "active")
        self.assertIsNone(self.store.session_get(sess, 1000), "re-enabling does not revive old sessions")

    def test_revoke(self):
        sess = h("ses-alice@example.com")
        self.store.session_revoke(sess)
        self.assertIsNone(self.store.session_get(sess, 1000))


class TestSchema(TempDirCase):
    def test_a_version_1_database_is_upgraded(self):
        from herogen.service import store as store_mod

        path = self.tmp / "old.db"
        con = sqlite3.connect(str(path))
        for stmt in filter(None, (x.strip() for x in store_mod.SCHEMA.split(";"))):
            con.execute(stmt.replace(", refunded_at REAL", ""))
        con.execute("INSERT INTO jobs(id, user_id, client_key, kind, profile, state, created_at) "
                    "VALUES ('j_x', 'u_x', 'k', 'sketch', 'best', 'succeeded', 1.0)")
        con.execute("PRAGMA user_version=1")
        con.commit()
        con.close()
        st = Store(path)
        self.assertEqual(st.db.execute("PRAGMA user_version").fetchone()[0], store_mod.SCHEMA_VERSION)
        self.assertIsNone(st.job_get("j_x")["refunded_at"], "existing rows keep their data")
        st.close()

    def test_an_unknown_version_is_refused(self):
        path = self.tmp / "new.db"
        con = sqlite3.connect(str(path))
        con.execute("PRAGMA user_version=99")
        con.close()
        with self.assertRaises(RuntimeError) as cm:
            Store(path)
        self.assertIn("schema version 99", str(cm.exception))


class SimVolume:
    """Per-container views of one shared Volume, as a Modal Volume behaves: reload copies the shared files in;
    commit copies out the files this container changed and removes the ones it deleted, file by file (the last
    commit of a file wins). Hooks run before a commit, between its copies and its deletions, and after a given
    reload."""

    def __init__(self, shared: Path, view: Path) -> None:
        self.shared, self.view = shared, view
        self.base: dict = {}
        self.before_commit = None
        self.mid_commit = None
        self.after_reload: dict = {}
        self.reloads = 0

    def _hashes(self, root: Path) -> dict:
        return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*") if p.is_file()}

    def reload(self) -> None:
        self.reloads += 1
        if self.view.exists():
            shutil.rmtree(self.view)
        shutil.copytree(self.shared, self.view)
        self.base = self._hashes(self.view)
        hook = self.after_reload.pop(self.reloads, None)
        if hook:
            hook()

    def commit(self) -> None:
        if self.before_commit:
            hook, self.before_commit = self.before_commit, None
            hook()
        now = self._hashes(self.view)
        for rel, digest in now.items():
            if self.base.get(rel) != digest:
                dst = self.shared / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.view / rel, dst)
        if self.mid_commit:
            hook, self.mid_commit = self.mid_commit, None
            hook()
        for rel in self.base:
            if rel not in now and (self.shared / rel).exists():
                (self.shared / rel).unlink()
        self.base = now


def newest_state(shared_meta: Path) -> list[str]:
    """The allowlist in the newest readable snapshot on the shared Volume, chosen as any new store would."""
    reader = Snapshotter(shared_meta, shared_meta.parent / f"reader-{secrets.token_hex(4)}.db")
    reader.restore()
    if not reader.local.exists():
        return []
    con = sqlite3.connect(str(reader.local))
    try:
        return sorted(r[0] for r in con.execute("SELECT email FROM allowlist"))
    finally:
        con.close()


class Container:
    """One store container: its own view of the shared Volume, its Snapshotter and its Store, and the retry loop
    of modal_app.StoreService._run (restore and retry on SnapshotConflict, "busy" on Superseded)."""

    opened: ClassVar[list] = []    # every container a test made, closed in ContainerCase.tearDown

    def __init__(self, root: Path, name: str) -> None:
        self.vol = SimVolume(root / "shared", root / f"view-{name}")
        self.vol.reload()
        self.snap = Snapshotter(self.vol.view / "meta", root / f"local-{name}.db", self.vol,
                                owner=name.encode().hex())
        self.snap.acquire()
        self.snap.restore()
        self.store = Store(self.snap.local, persist=self.snap.save)
        Container.opened.append(self)

    def write(self, email: str) -> str:
        for _ in range(3):
            try:
                self.store.allow_add(email)
                return "ok"
            except Superseded:
                return "busy"
            except SnapshotConflict:
                self.store.close()
                self.snap.restore()
                self.store = Store(self.snap.local, persist=self.snap.save)
        return "busy"


class ContainerCase(TempDirCase):
    def tearDown(self) -> None:
        while Container.opened:
            Container.opened.pop().store.close()
        super().tearDown()


class TestSnapshotInterleavings(ContainerCase):
    def setUp(self) -> None:
        super().setUp()
        (self.tmp / "shared").mkdir()
        self.meta = self.tmp / "shared" / "meta"

    def test_a_write_racing_a_new_store_is_never_reported_saved_and_lost(self):
        """A passes its checks; B takes the lease and passes its checks; A commits and reads back; then B commits.
        A must not report success for a write that B's state lacks."""
        a = Container(self.tmp, "a")
        self.assertEqual(a.write("first@example.com"), "ok")
        gate, ready, state = threading.Event(), threading.Event(), {}

        def start_b():
            b = Container(self.tmp, "b")
            commit = b.vol.commit

            def held():
                ready.set()
                gate.wait(10)
                commit()

            b.vol.commit = held
            state["thread"] = threading.Thread(target=lambda: state.update(b=b.write("from-b@example.com")))
            state["thread"].start()
            ready.wait(10)

        a.vol.before_commit = start_b
        a.vol.after_reload[a.vol.reloads + 2] = gate.set   # A's read-back after its commit; then B commits
        result_a = a.write("from-a@example.com")
        state["thread"].join(10)
        rows = newest_state(self.meta)
        self.assertEqual(state["b"], "ok")
        self.assertIn("from-b@example.com", rows)
        self.assertFalse(result_a == "ok" and "from-a@example.com" not in rows, "A's write was lost silently")
        self.assertEqual(result_a, "busy")
        self.assertEqual(a.write("again@example.com"), "busy")

    def test_a_stale_stores_late_commit_never_replaces_an_acknowledged_write(self):
        """A stale store's late commit: A passes its pre-check and stalls before its commit lands; B starts,
        writes and is acknowledged; then A's commit lands. B's acknowledged write must stay in the newest state,
        for B's next write and for a store started afterwards."""
        a = Container(self.tmp, "a")
        self.assertEqual(a.write("first@example.com"), "ok")
        state = {}

        def b_starts_and_writes():
            state["b"] = Container(self.tmp, "b")
            state["ack"] = state["b"].write("from-b@example.com")

        a.vol.before_commit = b_starts_and_writes
        result_a = a.write("from-a@example.com")
        b = state["b"]
        self.assertEqual((state["ack"], result_a), ("ok", "busy"))
        self.assertIn("from-b@example.com", newest_state(self.meta), "A's late commit replaced B's write")
        self.assertEqual(b.write("later@example.com"), "ok", "B is not thrown back to A's snapshot")
        self.assertEqual(newest_state(self.meta), ["first@example.com", "from-b@example.com", "later@example.com"])
        c = Container(self.tmp, "c")
        self.assertEqual([r["email"] for r in c.store.allow_list()],
                         ["first@example.com", "from-b@example.com", "later@example.com"])

    def test_a_crash_after_the_upload_loses_nothing_acknowledged(self):
        """A commits its snapshot and dies before confirming: the write was not acknowledged, and the next store
        starts from the newest snapshot on the Volume, which is consistent."""
        a = Container(self.tmp, "a")
        self.assertEqual(a.write("first@example.com"), "ok")

        def crash():
            raise RuntimeError("container killed")

        a.vol.after_reload[a.vol.reloads + 2] = crash      # the read-back after the commit never completes
        with self.assertRaises(RuntimeError):
            a.store.allow_add("unconfirmed@example.com")
        b = Container(self.tmp, "b")
        self.assertIn("first@example.com", [r["email"] for r in b.store.allow_list()])
        self.assertEqual(b.write("from-b@example.com"), "ok")
        self.assertIn("from-b@example.com", newest_state(self.meta))

    def test_a_crash_before_the_commit_leaves_the_volume_as_it_was(self):
        a = Container(self.tmp, "a")
        self.assertEqual(a.write("first@example.com"), "ok")
        before = sorted(p.name for p in self.meta.iterdir())

        def crash():
            raise RuntimeError("container killed")

        a.vol.before_commit = crash
        with self.assertRaises(RuntimeError):
            a.store.allow_add("never@example.com")
        self.assertEqual(sorted(p.name for p in self.meta.iterdir()), before)
        self.assertEqual(newest_state(self.meta), ["first@example.com"])

    def test_a_superseded_store_never_writes_again(self):
        a = Container(self.tmp, "a")
        self.assertEqual(a.write("first@example.com"), "ok")
        b = Container(self.tmp, "b")
        files = sorted(p.name for p in self.meta.iterdir())
        for i in range(3):
            self.assertEqual(a.write(f"stale{i}@example.com"), "busy")
        self.assertEqual(sorted(p.name for p in self.meta.iterdir()), files, "the stale store wrote nothing")
        self.assertEqual(b.write("from-b@example.com"), "ok")
        self.assertEqual(newest_state(self.meta), ["first@example.com", "from-b@example.com"])

    def test_restore_during_a_concurrent_commit_sees_a_whole_state(self):
        """B starts while A's commit is half applied (its new snapshot copied, its deletions not yet): B restores a
        complete snapshot, and A, now outranked, does not report its write saved."""
        a = Container(self.tmp, "a")
        for i in range(4):
            self.assertEqual(a.write(f"w{i}@example.com"), "ok")
        state = {}
        a.vol.mid_commit = lambda: state.update(b=Container(self.tmp, "b"))
        self.assertEqual(a.write("from-a@example.com"), "busy")
        b = state["b"]
        restored = [r["email"] for r in b.store.allow_list()]
        self.assertIn("from-a@example.com", restored, "B restored A's new snapshot, which was already whole")
        self.assertEqual(b.write("from-b@example.com"), "ok")
        self.assertEqual(newest_state(self.meta), sorted(restored + ["from-b@example.com"]))

    def test_two_stores_starting_together(self):
        """Both read the same lease number. The larger owner id wins; the other is refused at start or at its
        first save, whichever sees the winner's lease first."""
        (self.tmp / "shared" / "meta").mkdir()
        va, vb = SimVolume(self.tmp / "shared", self.tmp / "va"), SimVolume(self.tmp / "shared", self.tmp / "vb")
        va.reload()
        vb.reload()
        sa = Snapshotter(va.view / "meta", self.tmp / "la.db", va, owner="61")
        sb = Snapshotter(vb.view / "meta", self.tmp / "lb.db", vb, owner="62")
        # b computes its epoch from a view taken before a's lease landed, so both take epoch 1; b's lease lands
        # after a's start-up check passed.
        reload_b = vb.reload
        calls = {"n": 0}

        def stale_first_reload():
            calls["n"] += 1
            if calls["n"] > 1:
                reload_b()

        vb.reload = stale_first_reload
        sa.acquire()
        sb.acquire()
        self.assertEqual(sa.epoch[0], sb.epoch[0])
        sa.restore()
        con = sqlite3.connect(":memory:")
        with self.assertRaises(Superseded):
            sa.save(con)                          # a's first save sees b's larger lease: a never writes
        con.close()
        sb.restore()
        sbs = Store(sb.local, persist=sb.save)
        sbs.allow_add("from-b@example.com")
        self.assertEqual(newest_state(self.meta), ["from-b@example.com"])
        sbs.close()


class TestSnapshotReads(ContainerCase):
    """A snapshot that cannot be read right now is not corrupt: the store must not start, never fall back."""

    def setUp(self) -> None:
        super().setUp()
        (self.tmp / "shared").mkdir()
        self.meta = self.tmp / "shared" / "meta"
        a = Container(self.tmp, "a")
        for i in range(5):
            self.assertEqual(a.write(f"u{i}@example.com"), "ok")   # all five acknowledged
        a.store.close()
        self.acked = [f"u{i}@example.com" for i in range(5)]

    def flaky_connect(self, failures: int):
        """Make sqlite3.connect raise a disk I/O error for the next `failures` read-only opens; returns the undo."""
        real = sqlite3.connect
        left = [failures]

        def flaky(target, *a, **k):
            if "mode=ro" in str(target) and left[0] > 0:
                left[0] -= 1
                raise sqlite3.OperationalError("disk I/O error")
            return real(target, *a, **k)

        sqlite3.connect = flaky
        self.addCleanup(setattr, sqlite3, "connect", real)
        return lambda: setattr(sqlite3, "connect", real)

    def test_transient_open_errors_stop_the_start_and_lose_nothing(self):
        for failures in (1, 3):
            undo = self.flaky_connect(failures)
            files = sorted(p.name for p in self.meta.iterdir())
            with self.assertLogs("herogen.service", "ERROR") as logs, self.assertRaises(SnapshotUnreadable):
                Container(self.tmp, f"b{failures}")
            self.assertIn("will not start", "\n".join(logs.output))
            self.assertEqual(sorted(p.name for p in self.meta.iterdir() if p.name.startswith("meta-")),
                             [f for f in files if f.startswith("meta-")], "nothing was written or removed")
            undo()
        c = Container(self.tmp, "c")                      # a healthy restart later
        self.assertEqual([r["email"] for r in c.store.allow_list()], self.acked)
        self.assertEqual(c.write("after@example.com"), "ok")
        self.assertEqual(newest_state(self.meta), sorted(self.acked + ["after@example.com"]))

    def test_an_io_error_reading_the_file_stops_the_start(self):
        import herogen.service.snapshot as snapmod

        real = snapmod.shutil.copyfile

        def eio(src, dst, *a, **k):
            if str(dst).endswith(".candidate"):          # the snapshot read, not the simulated Volume's copying
                raise OSError(5, "Input/output error")
            return real(src, dst, *a, **k)

        snapmod.shutil.copyfile = eio
        try:
            with self.assertLogs("herogen.service", "ERROR"), self.assertRaises(SnapshotUnreadable):
                Container(self.tmp, "b")
        finally:
            snapmod.shutil.copyfile = real
        self.assertEqual([r["email"] for r in Container(self.tmp, "c").store.allow_list()], self.acked)

    def test_a_short_read_stops_the_start_and_loses_nothing(self):
        import herogen.service.snapshot as snapmod

        real = snapmod.shutil.copyfile

        def short(src, dst, *a, **k):
            if str(dst).endswith(".candidate"):          # the copy returns early without an error
                data = Path(src).read_bytes()
                Path(dst).write_bytes(data[: len(data) // 2])
                return dst
            return real(src, dst, *a, **k)

        snapmod.shutil.copyfile = short
        try:
            with self.assertLogs("herogen.service", "ERROR") as logs, self.assertRaises(SnapshotUnreadable):
                Container(self.tmp, "b")
        finally:
            snapmod.shutil.copyfile = real
        self.assertIn("read back", "\n".join(logs.output))
        self.assertEqual(list(self.meta.glob("quarantine-*")), [], "a short read is not corruption")
        self.assertEqual([r["email"] for r in Container(self.tmp, "c").store.allow_list()], self.acked)

    def test_a_corrupt_newest_snapshot_is_quarantined_and_survives_pruning(self):
        newest = max(self.meta.glob("meta-*.db"))
        newest.write_bytes(newest.read_bytes()[:4096])     # torn: the header survives, the pages do not
        with self.assertLogs("herogen.service", "ERROR") as logs:
            b = Container(self.tmp, "b")
        self.assertIn("is corrupt", "\n".join(logs.output))
        quarantined = list(self.meta.glob("quarantine-*"))
        self.assertEqual([q.name.endswith(newest.name) for q in quarantined], [True], "kept, under quarantine")
        self.assertFalse(newest.exists())
        self.assertEqual([r["email"] for r in b.store.allow_list()], self.acked[:4], "the previous snapshot")
        self.assertEqual(b.write("b1@example.com"), "ok")
        b.store.close()
        c = Container(self.tmp, "c")                       # another container, with no memory of the corruption
        for i in range(4):
            self.assertEqual(c.write(f"c{i}@example.com"), "ok")
            self.assertTrue(quarantined[0].exists(), f"pruned after write {i + 1}")
        self.assertEqual(len(list(self.meta.glob("meta-*.db"))), 3, "quarantined files use no kept slot")

    def test_never_restores_to_empty_while_snapshots_exist(self):
        for p in self.meta.glob("meta-*.db"):
            p.write_bytes(b"not a database" * 100)
        with self.assertLogs("herogen.service", "ERROR"), self.assertRaises(SnapshotUnreadable) as cm:
            Container(self.tmp, "b")
        self.assertIn("refusing to start from an empty database", str(cm.exception))
        self.assertEqual(len(list(self.meta.glob("quarantine-*"))), 3, "all three kept, under quarantine")
        with self.assertLogs("herogen.service", "ERROR"), self.assertRaises(SnapshotUnreadable):
            Container(self.tmp, "c")                       # and every later start refuses too
        (self.meta / "meta-3-62.db").write_bytes(b"")       # a snapshot name this version does not know
        with self.assertLogs("herogen.service", "ERROR"), self.assertRaises(SnapshotUnreadable) as cm:
            Container(self.tmp, "d")
        self.assertIn("unrecognised", str(cm.exception))

    def test_a_volume_with_nothing_starts_empty(self):
        shutil.rmtree(self.tmp / "shared")
        (self.tmp / "shared").mkdir()
        self.assertEqual(Container(self.tmp, "fresh").store.allow_list(), [])

    def test_snapshots_need_no_side_files(self):
        (self.meta / "meta-00000001-61-0000000001.db-shm").write_bytes(b"x")   # left by an older version
        b = Container(self.tmp, "b")
        self.assertEqual(b.write("after@example.com"), "ok")
        for p in self.meta.glob("meta-*.db"):
            self.assertEqual(p.read_bytes()[18:20], b"\x01\x01", f"{p.name} has a WAL header")
        self.assertEqual([p.name for p in self.meta.iterdir() if not p.name.endswith(".db")
                          and not p.name.startswith("lease-")], [], "no -wal, -shm or journal files")

    def test_check_snapshot_tells_corrupt_from_unreadable(self):
        import herogen.service.snapshot as snapmod
        from herogen.service.snapshot import check_snapshot

        good = max(self.meta.glob("meta-*.db"))
        self.assertTrue(check_snapshot(good))
        real = snapmod.sqlite3.connect
        for name, corrupt in (("SQLITE_IOERR", False), ("SQLITE_CANTOPEN", False), ("SQLITE_BUSY", False),
                              ("SQLITE_CORRUPT", True), ("SQLITE_NOTADB", True)):
            def fail(*a, _name=name, **k):
                e = sqlite3.DatabaseError(f"simulated {_name}")
                e.sqlite_errorname = _name
                raise e

            snapmod.sqlite3.connect = fail
            try:
                if corrupt:
                    self.assertFalse(check_snapshot(good), name)
                else:
                    with self.assertRaises(SnapshotUnreadable, msg=name):
                        check_snapshot(good)
            finally:
                snapmod.sqlite3.connect = real


class TestOperatorQuarantine(ContainerCase):
    """The way past a snapshot that never becomes readable: an operator quarantines it by name."""

    def setUp(self) -> None:
        super().setUp()
        (self.tmp / "shared").mkdir()
        self.meta = self.tmp / "shared" / "meta"
        a = Container(self.tmp, "a")
        for i in range(3):
            self.assertEqual(a.write(f"u{i}@example.com"), "ok")
        a.store.close()
        self.vol = SimVolume(self.tmp / "shared", self.tmp / "view-operator")

    def wedge_newest(self) -> Path:
        """Corrupt the newest snapshot so that SQLite reports it as an open error (a bad schema format number)."""
        newest = max(self.meta.glob("meta-*.db"))
        data = bytearray(newest.read_bytes())
        data[44:48] = b"\x00\x00\x00\x09"
        newest.write_bytes(bytes(data))
        return newest

    def test_quarantining_a_wedged_snapshot_lets_the_store_start_from_the_next(self):
        from herogen.service.snapshot import quarantine_snapshot, snapshot_files

        wedged = self.wedge_newest()
        with self.assertLogs("herogen.service", "ERROR"), self.assertRaises(SnapshotUnreadable):
            Container(self.tmp, "b")                       # unreadable: the store will not start
        self.vol.reload()
        self.assertIn(wedged.name, snapshot_files(self.vol.view / "meta", self.vol))
        with self.assertLogs("herogen.service", "WARNING") as logs:
            out = quarantine_snapshot(self.vol.view / "meta", self.vol, wedged.name)
        self.assertIn("operator quarantined", "\n".join(logs.output))
        self.assertEqual((out["quarantined"], out["allow_empty"]), (wedged.name, False))
        self.assertFalse(wedged.exists())
        self.assertTrue((self.meta / out["as"]).exists(), "kept, under quarantine")
        c = Container(self.tmp, "c")
        self.assertEqual([r["email"] for r in c.store.allow_list()], ["u0@example.com", "u1@example.com"])
        self.assertEqual(c.write("after@example.com"), "ok")

    def test_the_operators_lease_stops_a_running_store(self):
        from herogen.service.snapshot import quarantine_snapshot

        b = Container(self.tmp, "b")
        oldest = min(self.meta.glob("meta-*.db"))
        quarantine_snapshot(self.vol.view / "meta", self.vol, oldest.name)
        self.assertEqual(b.write("late@example.com"), "busy", "the running store is superseded")

    def test_the_last_snapshot_needs_an_explicit_flag(self):
        from herogen.service.snapshot import quarantine_snapshot

        names = sorted(p.name for p in self.meta.glob("meta-*.db"))
        for n in names[:-1]:
            quarantine_snapshot(self.vol.view / "meta", self.vol, n)
        with self.assertRaises(ValueError) as cm:
            quarantine_snapshot(self.vol.view / "meta", self.vol, names[-1])
        self.assertIn("--allow-empty", str(cm.exception))
        self.assertTrue((self.meta / names[-1]).exists(), "nothing changed")
        with self.assertRaises(ValueError):
            quarantine_snapshot(self.vol.view / "meta", self.vol, "../../etc/passwd")
        out = quarantine_snapshot(self.vol.view / "meta", self.vol, names[-1], allow_empty=True)
        self.assertTrue(out["allow_empty"])
        with self.assertLogs("herogen.service", "WARNING") as logs:
            c = Container(self.tmp, "c")
        self.assertIn("an operator allowed it", "\n".join(logs.output))
        self.assertEqual(c.store.allow_list(), [])
        self.assertEqual(c.write("fresh@example.com"), "ok")
        self.assertEqual(list(self.meta.glob("allow-empty-*")), [], "the permission is used up by the first save")
        self.assertEqual(len(list(self.meta.glob("quarantine-*"))), 3, "every quarantined file is still there")


class TestSnapshots(TempDirCase):
    def snap(self, name: str, **kw) -> Snapshotter:
        return Snapshotter(self.tmp / "vol" / "meta", self.tmp / name / "meta.db", **kw)

    def test_every_write_is_snapshotted_and_restorable(self):
        snap = self.snap("a")
        snap.acquire()
        snap.restore()
        store = Store(snap.local, persist=snap.save)
        make_user(store)
        gen = snap.head[2]
        self.assertGreaterEqual(gen, 3)
        store.close()
        snap2 = self.snap("b")
        snap2.acquire()
        self.assertEqual(snap2.restore(), gen)
        s2 = Store(snap2.local, persist=snap2.save)
        self.assertEqual([u["email"] for u in s2.users_list()], ["alice@example.com"])
        s2.allow_add("bob@example.com")
        meta = self.tmp / "vol" / "meta"
        self.assertLessEqual(len(list(meta.glob("meta-*.db"))), 3)
        self.assertEqual(len(list(meta.glob("lease-*"))), 1, "older leases are cleared by the holder")
        s2.close()

    def test_a_stale_copy_is_refused_then_restored(self):
        a, b = self.snap("a"), self.snap("b")
        a.acquire()
        a.restore()
        sa = Store(a.local, persist=a.save)
        make_user(sa)
        b.epoch, b.owner, b.last = a.epoch, a.owner, None   # b holds a's lease but not its newest snapshot
        con = sqlite3.connect(str(a.local))
        with self.assertRaises(SnapshotConflict) as cm:
            b.save(con)
        self.assertNotIsInstance(cm.exception, Superseded)
        con.close()
        sa.close()

    def test_corrupt_newest_snapshot_falls_back(self):
        a = self.snap("a")
        a.acquire()
        a.restore()
        sa = Store(a.local, persist=a.save)
        make_user(sa)
        sa.close()
        key = a.head
        a._snap_path(key).write_bytes(b"not a database" * 100)
        b = self.snap("b")
        b.acquire()
        self.assertEqual(b.restore(), key[2] - 1)
        sb = Store(b.local, persist=b.save)
        sb.allow_add("bob@example.com")     # the corrupt file is skipped, not taken as the newest
        self.assertGreater(b.head[:2], key[:2])
        sb.close()

    def test_owner_ids_are_hex(self):
        with self.assertRaises(ValueError):
            self.snap("x", owner="../evil")
