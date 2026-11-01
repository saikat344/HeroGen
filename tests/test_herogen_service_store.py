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
from contextlib import closing
from typing import ClassVar
from unittest.mock import Mock, patch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from test_herogen_service_support import Clock, TempDirCase, make_user

from herogen.service.snapshot import SnapshotConflict, Snapshotter, SnapshotUnreadable, Superseded
from herogen.service.store import (
    TRANSITIONS,
    Conflict,
    InvalidTransition,
    NotFound,
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
                    profile="roblox-ingame", storage_max_bytes=10 ** 9,
                    input_sha="a" * 64)
        args.update(kw)
        return self.store.job_create(**args)


class TestMethodSets(TempDirCase):
    def test_admin_methods_are_separate(self):
        self.assertFalse(Store.PUBLIC & Store.ADMIN)
        for name in Store.PUBLIC | Store.ADMIN:
            self.assertTrue(callable(getattr(Store, name)), name)
        for name in ("generation_update", "user_set_status", "audit_list", "users_list"):
            self.assertIn(name, Store.ADMIN)


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
        q = pickle.loads(pickle.dumps(StoreError("full")))
        self.assertIsInstance(q, StoreError)
        self.assertEqual(q.code, "store_error")


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


    def test_storage_limit(self):
        j, _ = self.create(key="key-0000", storage_max_bytes=100)
        c = self.store.job_claim(j["id"], "c", 60, 2)
        self.store.job_finish(j["id"], c["attempt"], "succeeded", bytes_=150)
        with self.assertRaises(StoreError) as cm:
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
        self.assertEqual(job["state"], "failed")

    def test_out_of_attempts_fails_at_claim_before_paying(self):
        j, _ = self.create()
        self.store.job_claim(j["id"], "c1", 60, 1)
        self.store.job_requeue(j["id"], 1, 5, "x", None, "x")   # a larger limit queued it again
        self.assertIsNone(self.store.job_claim(j["id"], "c2", 60, 1))
        self.assertEqual(self.store.job_get(j["id"])["state"], "failed")

    def test_lease_expiry_fails(self):
        j, _ = self.create()
        self.store.job_claim(j['id'], 'lost', 60, 2)
        self.clock.advance(60)
        self.assertEqual(self.store.reconcile_stale(2), [(j['id'], 'failed')])
        self.assertIsNone(self.store.job_claim(j['id'], 'late', 60, 2))

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


    def test_renewing_a_lease_extends_it_only_for_its_holder(self):
        lease = lambda sql, *a: self.store._one(sql, a)["lease_until"]  # noqa: E731
        j, _ = self.create()
        self.store.job_claim(j["id"], "c", 60, 2)
        self.store.job_stage_set(j["id"], 1, "3", "running", lease_s=4000)
        t = self.clock.t
        self.assertTrue(self.store.job_renew(j["id"], 1, 600))
        self.assertEqual(lease("SELECT lease_until FROM jobs WHERE id=?", j["id"]), t + 4000, "a longer lease stays")
        self.clock.advance(3900)
        self.assertTrue(self.store.job_renew(j["id"], 1, 600))
        self.assertEqual(lease("SELECT lease_until FROM jobs WHERE id=?", j["id"]), self.clock.t + 600)
        self.assertFalse(self.store.job_renew(j["id"], 2, 600), "not the live attempt")
        self.store.job_cancel(j['id'], self.user['id'])
        self.assertFalse(self.store.job_renew(j['id'], 1, 600))

    def test_a_reused_stage_keeps_what_it_spent(self):
        j, _ = self.create()
        self.store.job_claim(j["id"], "c", 60, 2)
        self.store.job_stage_set(j["id"], 1, "1", "passed", seconds=3.0, cost_usd=0.02)
        self.store.job_requeue(j["id"], 1, 2, "timeout", "3", "slow")
        self.store.job_claim(j["id"], "c2", 60, 2)
        self.store.job_stage_set(j["id"], 2, "1", "passed", message="kept from an earlier attempt")
        row = {r["stage"]: r for r in self.store.job_stages(j["id"])}["1"]
        self.assertEqual((row["cost_usd"], row["seconds"]), (0.02, 3.0))


    def running(self, stage="1"):
        """A job whose attempt 1 is running `stage`, with stage 0 passed and the held-out views running beside."""
        j, _ = self.create(key=new_id("j")[2:])
        self.store.job_claim(j["id"], "c", 60, 2)
        self.store.job_stage_set(j["id"], 1, "0", "passed", seconds=1.0)
        self.store.job_stage_set(j["id"], 1, stage, "running", lease_s=60)

        return j["id"]

    def states(self, job_id):
        return {r["stage"]: r["state"] for r in self.store.job_stages(job_id)}

    def test_an_attempt_that_ends_without_its_worker_closes_its_running_stage(self):
        # Only the live attempt may write a stage, so a stage left running when the attempt ended some other way
        # used to stay running for good, and the page polled every 3 s forever.
        a = self.running()
        self.store.job_cancel(a, self.user["id"])
        self.assertEqual(self.states(a), {"0": "passed", "1": "canceled"})
        b = self.running()
        self.store.job_delete(b, self.user["id"])
        self.assertEqual(self.states(b)["1"], "canceled")
        c = self.running()
        self.clock.advance(61)
        self.store.reconcile_stale(2)
        self.assertEqual((self.store.job_get(c)["state"], self.states(c)["1"]), ("failed", "failed"))
        d = self.running()
        self.assertEqual(self.store.job_requeue(d, 1, 1, "timeout", "1", "slow"), "failed")
        self.assertEqual(self.states(d)["1"], "failed")
        e = self.running("3")
        self.store.user_delete(self.user["id"])
        self.assertEqual(self.states(e)["3"], "canceled")
        for job in (a, b, c, d, e):
            self.assertNotIn("running", [s for k, s in self.states(job).items() if k != "2h"], job)

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
        jobs = [self.create(key=f"key-{i:04d}", user_id=make_user(self.store, f"user{i}@example.com")["id"])[0]
                for i in range(3)]
        self.store.generation_update(actor="fixture", max_running=2)
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

        counts = self.store.purge_expired(86400)
        self.assertGreaterEqual(counts["audit"], 1)


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
    def old_database(self, version: int) -> Path:
        """A database made with the schema of an older version, holding one job and its reservation."""
        from herogen.service import migrations as store_mod

        schema = store_mod.SCHEMA.replace(store_mod.ADMISSION_SCHEMA, '').replace(store_mod.ANALYTICS_SCHEMA, '')
        if version < 8:
            schema = schema.replace("  continued_after INTEGER NOT NULL DEFAULT 0, parent_job TEXT, from_stage TEXT, "
                                    "note TEXT NOT NULL DEFAULT '', stop_after TEXT,\n", "")
        if version < 5:
            schema = schema.replace(", kind TEXT NOT NULL DEFAULT 'account'", '')
            schema = schema.replace(', budget_required INTEGER NOT NULL DEFAULT 0', '')
        else:
            # v9 already has the phase-aware admission tables; older stores have the legacy ones.
            admission = store_mod.ADMISSION_SCHEMA if version >= 9 else store_mod.LEGACY_ADMISSION_SCHEMA
            if version < 7:
                admission = admission.replace(", process TEXT NOT NULL DEFAULT ''", '')
                admission = ';'.join(stmt for stmt in admission.split(';')
                                     if 'CREATE TABLE admission_calls' not in stmt
                                     and 'CREATE TABLE admission_processes' not in stmt)
            if version == 5:
                admission = admission.replace('user_id TEXT NOT NULL, client_key',
                                              'user_id TEXT NOT NULL, code_id TEXT, client_key')
                admission = ';'.join(stmt for stmt in admission.split(';') if 'budget_overhead' not in stmt)
            schema += admission
        if version >= 10:
            schema += store_mod.ANALYTICS_SCHEMA
        if version < 4:
            schema = schema.replace("CREATE TABLE refine_budgets (job_id TEXT PRIMARY KEY, state TEXT NOT NULL);", "")
        if version < 3:
            schema = schema.replace("claim TEXT, lease_until REAL,", "")
        if version < 2:
            schema = schema.replace(", refunded_at REAL", "")
        path = self.tmp / f"v{version}.db"
        con = sqlite3.connect(str(path))
        for stmt in filter(None, (x.strip() for x in schema.split(";"))):
            con.execute(stmt)
        con.execute("INSERT INTO jobs(id, user_id, client_key, kind, profile, state, created_at) "
                    "VALUES ('j_x', 'u_x', 'k', 'sketch', 'best', 'succeeded', 1.0)")
        con.execute("INSERT INTO job_stages(job_id, stage, state, attempt, cost_usd) VALUES ('j_x', '1', 'passed', 1, "
                    "0.02)")
        if version >= 5:
            code_col = ",code_id" if version == 5 else ""
            code_val = ",'code'" if version == 5 else ""
            con.execute(f"INSERT INTO admissions(job_id,user_id{code_col},client_key,amount_micro,cost_signature,"
                        f"created_at) VALUES ('j_x','u_x'{code_val},'k',123,'signature',1.0)")
            con.execute("INSERT INTO admission_work(token,job_id,kind,state,created_at) "
                        "VALUES ('w1','j_x','modal','sent',1.0)")
        if version >= 4:
            con.execute("INSERT INTO refine_budgets VALUES ('j_x', '{\"spent_usd_total\":0.2}')")
        if version >= 5:
            con.execute("INSERT INTO admission_policy VALUES (1,1000,100,5,1234,1,'commit','signature','fixture')")
        if version >= 6:
            con.execute("INSERT INTO budget_overhead VALUES (1,50,'fixture',1234)")
        if version >= 7:
            con.execute("INSERT INTO admission_calls(job_id,token,receipt) VALUES ('j_x','w1','{}')")
            con.execute("INSERT INTO admission_processes(token,host,pid) VALUES ('w1','fixture',12)")
        if version >= 11:
            for stmt in store_mod.MIGRATIONS[10]:
                con.execute(stmt)
            con.execute("UPDATE generation_settings SET enabled=1,limit_usd=42,running_estimate_usd=2,max_running=3 WHERE id=1")
        if version >= 12:
            for stmt in store_mod.MIGRATIONS[11]:
                con.execute(stmt)
        con.execute(f"PRAGMA user_version={version}")
        con.commit()
        con.close()
        return path

    def test_older_databases_are_upgraded_keeping_their_rows(self):
        from herogen.service import store as store_mod

        for version in range(1, store_mod.SCHEMA_VERSION):
            path = self.old_database(version)
            with sqlite3.connect(path) as before:
                removed = ('admissions','admission_work','admission_calls','admission_processes',
                           'admission_policy','budget_overhead','refine_budgets')
                snapshots = {}
                for table in removed:
                    if before.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                        cols = [r[1] for r in before.execute('PRAGMA table_info('+table+')')]
                        # v5 had a retired code_id; subsequent immutable migrations intentionally remove it.
                        cols = [c for c in cols if c != 'code_id']
                        snapshots[table] = (cols, before.execute('SELECT '+','.join(cols)+' FROM '+table).fetchall())
            st = Store(path)
            for table in removed:
                self.assertIsNone(st.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone())
                self.assertIsNotNone(st.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", ('legacy_'+table,)).fetchone())
                if table in snapshots:
                    cols, rows = snapshots[table]
                    self.assertEqual([tuple(r) for r in st.db.execute('SELECT '+','.join(cols)+' FROM legacy_'+table)], rows)
            self.assertFalse(st.generation_get()['spend_limits_enabled'])
            if version == 11:
                self.assertTrue(st.generation_get()['enabled'])
                self.assertEqual(st.generation_get()['limit_usd'], 42)
                self.assertEqual(st.generation_get()['max_running'], 3)
                self.assertEqual(st.generation_get()['running_estimate_usd'], 2)
            self.assertEqual(st.db.execute("PRAGMA user_version").fetchone()[0], store_mod.SCHEMA_VERSION)
            self.assertIsNone(st.job_get("j_x")["parent_job"])
            self.assertEqual(st.job_get("j_x")["note"], "")
            self.assertIsNone(st.job_get("j_x")["refunded_at"], "existing rows keep their data")
            self.assertEqual(st.job_stages("j_x")[0]["cost_usd"], 0.02)
            cols = [r[1] for r in st.db.execute("PRAGMA table_info(job_stages)")]
            self.assertIn("claim", cols)
            self.assertEqual(st.db.execute("SELECT COUNT(*) FROM analytics_events").fetchone()[0], 0,
                             f"v{version} gains an empty analytics table")
            if version >= 5:
                kept = st.db.execute("SELECT amount_micro, closed, cost_signature FROM legacy_admissions "
                                     "WHERE job_id='j_x'").fetchone()
                self.assertEqual(tuple(kept), (123, 0, 'signature'), f"v{version} keeps its reservation")
                work = st.db.execute("SELECT state, process FROM legacy_admission_work WHERE token='w1'").fetchone()
                self.assertEqual(tuple(work), ('sent', ''), f"v{version} keeps its work ledger")
            st.close()


    def test_an_unknown_version_is_refused(self):
        path = self.tmp / "new.db"
        con = sqlite3.connect(str(path))
        con.execute("PRAGMA user_version=99")
        con.close()
        connection = sqlite3.connect(str(path))
        self.addCleanup(connection.close)
        with patch("herogen.service.store.sqlite3.connect", return_value=connection):
            with self.assertRaises(RuntimeError) as cm:
                Store(path)
        self.assertIn("schema version 99", str(cm.exception))
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_connection_is_closed_when_configuration_fails(self):
        connection = Mock()
        error = sqlite3.OperationalError("configuration failed")
        connection.execute.side_effect = error
        with patch("herogen.service.store.sqlite3.connect", return_value=connection):
            with self.assertRaises(sqlite3.OperationalError) as cm:
                Store(self.tmp / "configuration.db")
        self.assertIs(cm.exception, error)
        connection.close.assert_called_once_with()

    def test_connection_is_closed_when_persist_fails(self):
        connection = sqlite3.connect(str(self.tmp / "persist.db"), isolation_level=None)
        self.addCleanup(connection.close)
        error = KeyboardInterrupt()
        with patch("herogen.service.store.sqlite3.connect", return_value=connection):
            with self.assertRaises(KeyboardInterrupt) as cm:
                Store(self.tmp / "persist.db", persist=Mock(side_effect=error))
        self.assertIs(cm.exception, error)
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


class SnapshotStore(Store):
    def write_marker(self, label):
        self.audit_add('fixture', 'snapshot_test', label)

    def markers(self):
        return [{'email': r['subject']} for r in reversed(self.audit_list(1000)) if r['action'] == 'snapshot_test']


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
        return sorted(r[0] for r in con.execute("SELECT subject FROM audit WHERE action='snapshot_test'"))
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
        self.store = SnapshotStore(self.snap.local, persist=self.snap.save)
        Container.opened.append(self)

    def write(self, email: str) -> str:
        for _ in range(3):
            try:
                self.store.write_marker(email)
                return "ok"
            except Superseded:
                return "busy"
            except SnapshotConflict:
                self.store.close()
                self.snap.restore()
                self.store = SnapshotStore(self.snap.local, persist=self.snap.save)
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
        self.assertEqual([r["email"] for r in c.store.markers()],
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
            a.store.write_marker("unconfirmed@example.com")
        b = Container(self.tmp, "b")
        self.assertIn("first@example.com", [r["email"] for r in b.store.markers()])
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
            a.store.write_marker("never@example.com")
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
        restored = [r["email"] for r in b.store.markers()]
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
        sbs = SnapshotStore(sb.local, persist=sb.save)
        sbs.write_marker("from-b@example.com")
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
        self.assertEqual([r["email"] for r in c.store.markers()], self.acked)
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
        self.assertEqual([r["email"] for r in Container(self.tmp, "c").store.markers()], self.acked)

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
        self.assertEqual([r["email"] for r in Container(self.tmp, "c").store.markers()], self.acked)

    def test_a_corrupt_newest_snapshot_is_quarantined_and_survives_pruning(self):
        newest = max(self.meta.glob("meta-*.db"))
        newest.write_bytes(newest.read_bytes()[:4096])     # torn: the header survives, the pages do not
        with self.assertLogs("herogen.service", "ERROR") as logs:
            b = Container(self.tmp, "b")
        self.assertIn("is corrupt", "\n".join(logs.output))
        quarantined = list(self.meta.glob("quarantine-*"))
        self.assertEqual([q.name.endswith(newest.name) for q in quarantined], [True], "kept, under quarantine")
        self.assertFalse(newest.exists())
        self.assertEqual([r["email"] for r in b.store.markers()], self.acked[:4], "the previous snapshot")
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
        self.assertEqual(Container(self.tmp, "fresh").store.markers(), [])

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
        self.assertEqual([r["email"] for r in c.store.markers()], ["u0@example.com", "u1@example.com"])
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
        self.assertEqual(c.store.markers(), [])
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
        store = SnapshotStore(snap.local, persist=snap.save)
        make_user(store)
        gen = snap.head[2]
        self.assertGreaterEqual(gen, 3)
        store.close()
        snap2 = self.snap("b")
        snap2.acquire()
        self.assertEqual(snap2.restore(), gen)
        s2 = SnapshotStore(snap2.local, persist=snap2.save)
        self.assertEqual([u["email"] for u in s2.users_list()], ["alice@example.com"])
        s2.write_marker("bob@example.com")
        meta = self.tmp / "vol" / "meta"
        self.assertLessEqual(len(list(meta.glob("meta-*.db"))), 3)
        self.assertEqual(len(list(meta.glob("lease-*"))), 1, "older leases are cleared by the holder")
        s2.close()

    def test_a_stale_copy_is_refused_then_restored(self):
        a, b = self.snap("a"), self.snap("b")
        a.acquire()
        a.restore()
        sa = SnapshotStore(a.local, persist=a.save)
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
        sa = SnapshotStore(a.local, persist=a.save)
        make_user(sa)
        sa.close()
        key = a.head
        a._snap_path(key).write_bytes(b"not a database" * 100)
        b = self.snap("b")
        b.acquire()
        self.assertEqual(b.restore(), key[2] - 1)
        sb = SnapshotStore(b.local, persist=b.save)
        sb.write_marker("bob@example.com")     # the corrupt file is skipped, not taken as the newest
        self.assertGreater(b.head[:2], key[:2])
        sb.close()

    def test_owner_ids_are_hex(self):
        with self.assertRaises(ValueError):
            self.snap("x", owner="../evil")
