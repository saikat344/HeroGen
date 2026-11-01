"""The metadata store: guest workspaces, admin sign-in, jobs, stages and audit log.

One SQLite database (WAL journal) written by one process at a time. Every write runs in a `BEGIN IMMEDIATE`
transaction under a lock and calls `persist()` afterwards, which on Modal snapshots the database to its Volume
before the call returns (see `snapshot.py`); locally the database file itself is the durable copy.

The store never sees a secret: sign-in links and sessions arrive as keyed hashes (`auth.py`), and
nothing here sends email or touches run folders. Rows come back as plain dicts so they cross a Modal call
unchanged. Rate-limit counters live in memory only: they are lost when the process restarts, which only
forgives a few requests.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from herogen.service.migrations import SCHEMA as ADMISSION_SCHEMA, LEGACY_SCHEMA as LEGACY_ADMISSION_SCHEMA, PHASE_MIGRATION
from herogen.service.generation import GenerationStore, SCHEMA as GENERATION_SCHEMA
from herogen.service.analytics import AnalyticsStore, SCHEMA as ANALYTICS_SCHEMA, PUBLIC as ANALYTICS_PUBLIC

SCHEMA_VERSION = 12
# Steps that bring an older database up to SCHEMA_VERSION, by the version they start from.
MIGRATIONS = {
    11: ("ALTER TABLE generation_settings ADD COLUMN spend_limits_enabled INTEGER NOT NULL DEFAULT 0",),
    6: ("ALTER TABLE admission_work ADD COLUMN process TEXT NOT NULL DEFAULT ''",
        next(s.strip() for s in LEGACY_ADMISSION_SCHEMA.split(';') if 'CREATE TABLE admission_calls' in s),
        next(s.strip() for s in LEGACY_ADMISSION_SCHEMA.split(';') if 'CREATE TABLE admission_processes' in s)),
    1: ("ALTER TABLE jobs ADD COLUMN refunded_at REAL",),
    2: ("ALTER TABLE job_stages ADD COLUMN claim TEXT", "ALTER TABLE job_stages ADD COLUMN lease_until REAL"),
    3: ("CREATE TABLE refine_budgets (job_id TEXT PRIMARY KEY, state TEXT NOT NULL)",),
    4: ("ALTER TABLE users ADD COLUMN kind TEXT NOT NULL DEFAULT 'account'",
        "ALTER TABLE jobs ADD COLUMN budget_required INTEGER NOT NULL DEFAULT 0",
        *tuple(s.strip() for s in LEGACY_ADMISSION_SCHEMA.replace(", process TEXT NOT NULL DEFAULT ''", '').split(';')
               if s.strip() and 'CREATE TABLE admission_calls' not in s and 'CREATE TABLE admission_processes' not in s)),
    5: ("UPDATE users SET kind='guest' WHERE kind='judge'",
        "DROP TABLE IF EXISTS judge_workspaces", "DROP TABLE IF EXISTS judge_codes",
        "ALTER TABLE admissions RENAME TO admissions_old",
        next(stmt.strip() for stmt in LEGACY_ADMISSION_SCHEMA.split(';') if 'CREATE TABLE admissions (' in stmt),
        "INSERT INTO admissions SELECT job_id,user_id,client_key,amount_micro,cost_signature,actual_micro,main_done,side_required,side_done,closed,created_at,evidence FROM admissions_old",
        "DROP TABLE admissions_old",
        next(stmt.strip() for stmt in LEGACY_ADMISSION_SCHEMA.split(';') if 'CREATE TABLE IF NOT EXISTS budget_overhead' in stmt)),
    7: ("ALTER TABLE jobs ADD COLUMN parent_job TEXT",
        "ALTER TABLE jobs ADD COLUMN from_stage TEXT",
        "ALTER TABLE jobs ADD COLUMN note TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE jobs ADD COLUMN stop_after TEXT",
        "ALTER TABLE jobs ADD COLUMN continued_after INTEGER NOT NULL DEFAULT 0"),
    8: PHASE_MIGRATION,
    10: (*tuple(f"ALTER TABLE {name} RENAME TO legacy_{name}" for name in (
        'admissions', 'admission_work', 'admission_calls', 'admission_processes',
        'admission_policy', 'budget_overhead', 'refine_budgets')),
        GENERATION_SCHEMA.strip(),
        "INSERT INTO generation_settings(id,max_running) VALUES (1, 5)"),
    9: tuple(s.strip() for s in ANALYTICS_SCHEMA.split(';') if s.strip()),
}
ID_RE = re.compile(r"^[uji]_[a-z2-7]{26}$")
ACTIVE_STATES = ("queued", "running")
TERMINAL_STATES = ("succeeded", "rejected", "failed", "canceled", "stopped")
STATES = ACTIVE_STATES + TERMINAL_STATES
# The only state changes a job may make. Deleting a job is a flag (`deleted_at`), not a state.
TRANSITIONS = {
    "queued": {"running", "canceled", "failed"},
    "running": {"stopped", "succeeded", "rejected", "failed", "canceled", "queued"},
    "stopped": {"queued"}, "succeeded": set(), "rejected": set(), "failed": set(), "canceled": set(),
}
STAGE_STATES = ("pending", "running", "passed", "rejected", "failed", "skipped", "unavailable", "canceled")
# What a stage the attempt was still running becomes when the attempt ends without recording it (a cancel, a
# delete, the sweep requeueing or failing a lost worker): requeued, it runs again; otherwise it never finishes.
STAGE_WHEN_ENDED = {"queued": "pending", "canceled": "canceled"}
DISPATCH_GRACE_S = 300.0    # a queued job dispatched this recently is taken to be starting
SEEN_EVERY_S = 300.0        # how often a session's last-seen time is written
DAY_S = 86400.0

SCHEMA = """
CREATE TABLE users (
  id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, status TEXT NOT NULL DEFAULT 'active',
  created_at REAL NOT NULL, invite_id TEXT, terms_version TEXT, terms_accepted_at REAL, adult_confirmed_at REAL,
  quota_daily INTEGER, quota_total INTEGER, last_login_at REAL, deleted_at REAL, kind TEXT NOT NULL DEFAULT 'account'
);
CREATE TABLE allowlist (email TEXT PRIMARY KEY, added_at REAL NOT NULL, note TEXT NOT NULL DEFAULT '');
CREATE TABLE invites (
  id TEXT PRIMARY KEY, code_hash TEXT NOT NULL UNIQUE, created_at REAL NOT NULL, expires_at REAL NOT NULL,
  max_uses INTEGER NOT NULL, uses INTEGER NOT NULL DEFAULT 0, email TEXT, note TEXT NOT NULL DEFAULT '',
  revoked_at REAL
);
CREATE TABLE login_tokens (
  hash TEXT PRIMARY KEY, email TEXT NOT NULL, invite_id TEXT, created_at REAL NOT NULL, expires_at REAL NOT NULL,
  used_at REAL
);
CREATE INDEX login_tokens_email ON login_tokens(email);
CREATE TABLE sessions (
  hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL,
  last_seen_at REAL NOT NULL, revoked_at REAL
);
CREATE INDEX sessions_user ON sessions(user_id);
CREATE TABLE jobs (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, client_key TEXT NOT NULL, kind TEXT NOT NULL,
  input_sha TEXT NOT NULL DEFAULT '', prompt TEXT NOT NULL DEFAULT '', name TEXT NOT NULL DEFAULT '',
  profile TEXT NOT NULL, state TEXT NOT NULL, created_at REAL NOT NULL, dispatched_at REAL, call_id TEXT,
  started_at REAL, finished_at REAL, attempts INTEGER NOT NULL DEFAULT 0, lease_until REAL, stage TEXT,
  error_code TEXT, error_stage TEXT, error_message TEXT, counts_quota INTEGER NOT NULL DEFAULT 1,
  seconds REAL, cost_usd REAL, cost_detail TEXT, code_commit TEXT NOT NULL DEFAULT '', bytes INTEGER,
  continued_after INTEGER NOT NULL DEFAULT 0, parent_job TEXT, from_stage TEXT, note TEXT NOT NULL DEFAULT '', stop_after TEXT,
  deleted_at REAL, refunded_at REAL, budget_required INTEGER NOT NULL DEFAULT 0, UNIQUE(user_id, client_key)
);
CREATE INDEX jobs_user ON jobs(user_id, created_at);
CREATE INDEX jobs_state ON jobs(state);
CREATE TABLE job_stages (
  job_id TEXT NOT NULL, stage TEXT NOT NULL, state TEXT NOT NULL, attempt INTEGER NOT NULL,
  started_at REAL, seconds REAL, cost_usd REAL, message TEXT NOT NULL DEFAULT '', claim TEXT, lease_until REAL,
  PRIMARY KEY (job_id, stage)
);
CREATE TABLE job_runs (
  job_id TEXT NOT NULL, attempt INTEGER NOT NULL, grp TEXT NOT NULL, entries INTEGER NOT NULL,
  PRIMARY KEY (job_id, attempt, grp)
);
CREATE TABLE refine_budgets (job_id TEXT PRIMARY KEY, state TEXT NOT NULL);
CREATE TABLE audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
  subject TEXT NOT NULL DEFAULT '', detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX audit_at ON audit(at);
"""


SCHEMA += ADMISSION_SCHEMA + ANALYTICS_SCHEMA


class StoreError(Exception):
    """A request the store refuses. `code` is stable and safe to show; the message is for the user."""

    code = "store_error"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code

    def __reduce__(self):  # keep the code when the error crosses a Modal call
        return (type(self), (str(self), self.code))


class Conflict(StoreError):
    code = "conflict"


class InvalidTransition(StoreError):
    code = "invalid_transition"


class NotFound(StoreError):
    code = "not_found"


def new_id(prefix: str) -> str:
    """`u_`, `j_` or `i_` and 26 random base32 characters (130 bits): safe as a path segment."""
    if prefix not in ("u", "j", "i"):
        raise ValueError(prefix)
    return f"{prefix}_" + base64.b32encode(secrets.token_bytes(17)).decode().lower().rstrip("=")[:26]


def valid_id(value: Any, prefix: str) -> bool:
    return isinstance(value, str) and bool(ID_RE.match(value)) and value.startswith(prefix + "_")


class RateLimiter:
    """Sliding-window counters in memory, keyed by bucket."""

    def __init__(self, clock: Callable[[], float], max_buckets: int = 100_000) -> None:
        self.clock = clock
        self.max_buckets = max_buckets
        self.hits: dict[str, deque[float]] = {}
        self.lock = threading.Lock()

    def hit(self, bucket: str, limit: int, window_s: float) -> tuple[bool, int]:
        """(allowed, seconds until the next request would be allowed)."""
        now = self.clock()
        with self.lock:
            if len(self.hits) > self.max_buckets:
                self.hits = {k: q for k, q in self.hits.items() if q and q[-1] > now - 3600}
            q = self.hits.setdefault(bucket, deque())
            while q and q[0] <= now - window_s:
                q.popleft()
            if len(q) >= limit:
                return False, max(1, int(q[0] + window_s - now + 0.999))
            q.append(now)
            return True, 0


class Store(GenerationStore, AnalyticsStore):
    # Methods the service's own functions may invoke by name (modal_app.StoreService.call).
    PUBLIC = frozenset({
        "rate_hit", "login_token_create", "login_token_info",
        "login_complete", "session_get", "session_revoke", "user_get", "user_delete",
        "job_create", "job_get", "jobs_list", "job_dispatch", "job_claim", "job_current", "job_stage_set",
        "group_enter", "job_finish", "job_requeue", "job_cancel", "job_delete", "job_stages", "jobs_to_dispatch",
        "reconcile_stale", "retention_due", "job_mark_purged", "purge_expired", "audit_add",
        "job_renew",
        "job_continue", "job_versions",
        "guest_open", "generation_get",
    }) | ANALYTICS_PUBLIC
    # Methods reached through StoreService.admin_call with the hosted admin key.
    ADMIN = frozenset({
        "users_list", "user_set_status", "audit_list", "jobs_admin_list",
        "generation_update",
    })

    def __init__(self, path: Path | str, clock: Callable[[], float] = time.time,
                 persist: Callable[[sqlite3.Connection], None] | None = None,
                 max_running: int | None = None, spend_limits_enabled: bool | None = None) -> None:
        self.default_max_running = max_running if max_running is not None else int(os.environ.get("HEROGEN_SERVICE_MAX_RUNNING", "5"))
        self.default_spend_limits = (spend_limits_enabled if spend_limits_enabled is not None else
                                     os.environ.get("HEROGEN_SERVICE_SPEND_LIMITS", "false").lower() in ("true", "1", "yes"))
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.persist = persist
        self.lock = threading.RLock()
        self.rates = RateLimiter(clock)
        self.db = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False, timeout=30)
        try:
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA foreign_keys=ON")
            self._migrate()
        except BaseException:
            self.db.close()
            raise

    def close(self) -> None:
        with self.lock:
            self.db.close()

    # ---- plumbing ---------------------------------------------------------------------------------------

    def _migrate(self) -> None:
        with self.lock:
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if version == SCHEMA_VERSION:
                return
            if version > SCHEMA_VERSION or (version and any(v not in MIGRATIONS
                                                            for v in range(version, SCHEMA_VERSION))):
                raise RuntimeError(f"the metadata database has schema version {version}; this code knows "
                                   f"{SCHEMA_VERSION} and can upgrade from {sorted(MIGRATIONS)} only")
            self.db.execute("BEGIN IMMEDIATE")
            try:
                if version == 0:
                    for stmt in filter(None, (s.strip() for s in SCHEMA.split(";"))):
                        self.db.execute(stmt)
                else:
                    for v in range(version, SCHEMA_VERSION):
                        for stmt in MIGRATIONS[v]:
                            self.db.execute(stmt)
                if version == 0:
                    for v in (10, 11):
                        for stmt in MIGRATIONS[v]:
                            self.db.execute(stmt)
                    self.db.execute("UPDATE generation_settings SET enabled=1 WHERE id=1")
                if version <= 11:
                    self.db.execute("UPDATE generation_settings SET spend_limits_enabled=? WHERE id=1",
                                    (int(self.default_spend_limits),))
                self.db.execute('UPDATE generation_settings SET max_running=? WHERE id=1', (self.default_max_running,))
                self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
        if self.persist:
            self.persist(self.db)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            if self.persist:
                self.persist(self.db)

    def _one(self, sql: str, args: tuple = ()) -> dict[str, Any] | None:
        with self.lock:
            row = self.db.execute(sql, args).fetchone()
        return dict(row) if row else None

    def _all(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def _audit(self, c: sqlite3.Connection, actor: str, action: str, subject: str = "", detail: Any = "") -> None:
        text = detail if isinstance(detail, str) else json.dumps(detail, sort_keys=True)
        c.execute("INSERT INTO audit(at, actor, action, subject, detail) VALUES (?,?,?,?,?)",
                  (self.clock(), actor, action, subject, text[:2000]))

    def audit_add(self, actor: str, action: str, subject: str = "", detail: Any = "") -> None:
        with self._tx() as c:
            self._audit(c, actor, action, subject, detail)

    def rate_hit(self, bucket: str, limit: int, window_s: float) -> tuple[bool, int]:
        return self.rates.hit(bucket, limit, window_s)


    # ---- sign-in ----------------------------------------------------------------------------------------


    def login_token_create(self, token_hash: str, email: str, ttl_s: float, actor: str = "") -> None:
        now = self.clock()
        with self._tx() as c:
            c.execute("INSERT INTO login_tokens(hash, email, created_at, expires_at) VALUES (?,?,?,?)",
                      (token_hash, email, now, now + ttl_s))
            self._audit(c, actor or "anonymous", "signin_link_sent")

    def _live_token(self, c: sqlite3.Connection, token_hash: str) -> dict[str, Any] | None:
        row = c.execute("SELECT * FROM login_tokens WHERE hash=?", (token_hash,)).fetchone()
        if row is None or row["used_at"] is not None or row["expires_at"] <= self.clock():
            return None
        return dict(row)

    def login_token_info(self, token_hash: str) -> dict[str, Any] | None:
        with self.lock:
            tok = self._live_token(self.db, token_hash)
            return {"email": tok["email"], "expires_at": tok["expires_at"]} if tok else None

    def login_complete(self, token_hash: str, session_hash: str, *, admin_emails, session_ttl_s: float):
        with self._tx() as c:
            tok = self._live_token(c, token_hash)
            if tok is None or tok['email'] not in admin_emails:
                return {'status': 'invalid'}
            user = c.execute('SELECT * FROM users WHERE email=? AND deleted_at IS NULL', (tok['email'],)).fetchone()
            if user and user['status'] != 'active':
                return {'status': 'invalid'}
            uid = user['id'] if user else new_id('u')
            now = self.clock()
            if not user:
                c.execute('INSERT INTO users(id,email,created_at) VALUES (?,?,?)', (uid,tok['email'],now))
            c.execute('UPDATE login_tokens SET used_at=? WHERE email=? AND used_at IS NULL', (now,tok['email']))
            c.execute('INSERT INTO sessions(hash,user_id,created_at,expires_at,last_seen_at) VALUES (?,?,?,?,?)', (session_hash,uid,now,now+session_ttl_s,now))
            c.execute('UPDATE users SET last_login_at=? WHERE id=?', (now,uid))
            self._audit(c,uid,'signed_in',uid)
            return {'status':'ok', 'user':self.user_get(uid)}

    @staticmethod
    def _public_user(row: dict[str, Any]) -> dict[str, Any]:
        user = {k: row[k] for k in ("id", "email", "status", "created_at", "last_login_at")}
        user['kind'] = row.get('kind', 'account')
        if user['kind'] == 'guest':
            user['email'] = None
        return user

    def session_get(self, session_hash: str, idle_s: float) -> dict[str, Any] | None:
        """The signed-in user for a session, or None when the session is unknown, revoked, expired or idle."""
        now = self.clock()
        row = self._one("SELECT s.*, u.status AS u_status, u.deleted_at AS u_deleted FROM sessions s "
                        "JOIN users u ON u.id = s.user_id WHERE s.hash=?", (session_hash,))
        if (row is None or row["revoked_at"] is not None or row["expires_at"] <= now
                or row["last_seen_at"] + idle_s <= now or row["u_status"] != "active" or row["u_deleted"]):
            return None
        if now - row["last_seen_at"] >= SEEN_EVERY_S:
            with self._tx() as c:
                c.execute("UPDATE sessions SET last_seen_at=? WHERE hash=?", (now, session_hash))
        user = self.user_get(row["user_id"])
        return user

    def session_revoke(self, session_hash: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE sessions SET revoked_at=? WHERE hash=? AND revoked_at IS NULL",
                      (self.clock(), session_hash))

    # ---- users ------------------------------------------------------------------------------------------

    def user_get(self, user_id: str) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM users WHERE id=? AND deleted_at IS NULL", (user_id,))
        return self._public_user(row) if row else None

    def users_list(self) -> list[dict[str, Any]]:
        rows = self._all("SELECT u.*, (SELECT COUNT(*) FROM jobs j WHERE j.user_id=u.id) AS jobs FROM users u "
                         "WHERE deleted_at IS NULL ORDER BY created_at")
        return [{**self._public_user(r), "jobs": r["jobs"]} for r in rows]

    def _user_by_email(self, c: sqlite3.Connection, email: str) -> sqlite3.Row:
        row = c.execute("SELECT * FROM users WHERE email=? AND deleted_at IS NULL", (email,)).fetchone()
        if row is None:
            raise NotFound("no account with that address")
        return row

    def user_set_status(self, email: str, status: str) -> dict[str, Any]:
        if status not in ("active", "disabled"):
            raise StoreError("status must be active or disabled")
        with self._tx() as c:
            row = self._user_by_email(c, email)
            c.execute("UPDATE users SET status=? WHERE id=?", (status, row["id"]))
            if status == "disabled":
                c.execute("UPDATE sessions SET revoked_at=? WHERE user_id=? AND revoked_at IS NULL",
                          (self.clock(), row["id"]))
            self._audit(c, "admin", f"user_{status}", row["id"])
        return self.user_get(row["id"]) or {}


    def user_delete(self, user_id: str) -> list[str]:
        """Delete an account: the address is forgotten, sessions end, every job is marked deleted. Returns the job
        ids whose files the caller must remove."""
        now = self.clock()
        with self._tx() as c:
            row = c.execute("SELECT * FROM users WHERE id=? AND deleted_at IS NULL", (user_id,)).fetchone()
            if row is None:
                raise NotFound("no such account")
            jobs = [r["id"] for r in c.execute("SELECT id FROM jobs WHERE user_id=?", (user_id,))]
            for r in c.execute("SELECT id FROM jobs WHERE user_id=? AND state='running'", (user_id,)).fetchall():
                self._end_stages(c, r["id"], "canceled")
            c.execute("UPDATE jobs SET note='', deleted_at=COALESCE(deleted_at, ?), state=CASE WHEN state IN "
                      "('queued','running') THEN 'canceled' ELSE state END WHERE user_id=?", (now, user_id))
            c.execute("UPDATE sessions SET revoked_at=? WHERE user_id=? AND revoked_at IS NULL", (now, user_id))
            c.execute("UPDATE users SET email=?, deleted_at=?, status='disabled' WHERE id=?",
                      (f"deleted:{user_id}", now, user_id))
            # Forget the address in links and audit entries.
            email = row["email"]
            c.execute("DELETE FROM login_tokens WHERE email=?", (email,))
            c.execute("UPDATE audit SET subject='' WHERE subject=?", (email,))
            c.execute("UPDATE audit SET detail='' WHERE instr(detail, ?) > 0", (email,))
            self._audit(c, user_id, "account_deleted", user_id, {"jobs": len(jobs)})
        return jobs


    # ---- jobs -------------------------------------------------------------------------------------------

    def job_create(self, *, job_id: str, user_id: str, client_key: str, kind: str, profile: str,
                   storage_max_bytes: int, input_sha: str = "", prompt: str = "",
                   name: str = "", code_commit: str = "",
                   parent_job: str | None = None, from_stage: str | None = None,
                   note: str = "") -> tuple[dict[str, Any], bool]:
        """Create a queued job, or return the one this user already made with the same `client_key`.

        Idempotent: the same key and request returns the first job and `False`; the same key with a different
        request raises Conflict. The spend guard and storage limit are checked in the insertion transaction."""
        if kind not in ("sketch", "prompt") or not valid_id(job_id, "j"):
            raise StoreError("bad job request")
        if parent_job is not None:
            from herogen.pipeline.redo import clean_note

            note = clean_note(note)
            if from_stage not in ("1", "2"):
                raise StoreError("Choose cover or turnaround to redo.")
        elif from_stage is not None or note:
            raise StoreError("Choose an earlier hero to redo.")
        with self._tx() as c:
            if parent_job is not None:
                parent = c.execute("SELECT * FROM jobs WHERE id=? AND user_id=? AND deleted_at IS NULL",
                                   (parent_job, user_id)).fetchone()
                if parent is None:
                    raise NotFound("No such hero.")
                if parent["state"] in ACTIVE_STATES:
                    raise Conflict("Wait for this hero to stop before making a new version.")
                if (kind, input_sha, prompt, name, profile) != tuple(parent[k] for k in (
                        "kind", "input_sha", "prompt", "name", "profile")):
                    raise StoreError("The new version must use the earlier hero's input.")
            old = c.execute("SELECT * FROM jobs WHERE user_id=? AND client_key=?", (user_id, client_key)).fetchone()
            if old is not None:
                same = (old["kind"], old["input_sha"], old["prompt"], old["name"], old["profile"], old["parent_job"], old["from_stage"], old["note"]) == (
                    kind, input_sha, prompt, name, profile, parent_job, from_stage, note)
                if not same:
                    raise Conflict("this request key was already used for a different hero")
                return dict(old), False
            user = c.execute("SELECT status,kind FROM users WHERE id=? AND deleted_at IS NULL", (user_id,)).fetchone()
            if user is None or user["status"] != "active":
                raise StoreError("this account cannot start jobs", "account")
            used = c.execute("SELECT COALESCE(SUM(bytes),0) FROM jobs WHERE user_id=? AND deleted_at IS NULL", (user_id,)).fetchone()[0]
            if used >= storage_max_bytes:
                raise StoreError("Your saved heroes use all of your storage; delete some to make room.", "storage_full")
            self._guard_generation(c)
            c.execute("INSERT INTO jobs(id, user_id, client_key, kind, input_sha, prompt, name, profile, state, "
                      "created_at, code_commit, parent_job, from_stage, note, stop_after) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (job_id, user_id, client_key, kind, input_sha, prompt, name, profile, "queued", self.clock(),
                       code_commit, parent_job, from_stage, note, from_stage))
            self._audit(c, user_id, "job_created", job_id, {"kind": kind, "profile": profile})
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row), True


    def job_continue(self, job_id: str, user_id: str, storage_max_bytes: int, *,
                     code_commit: str = '') -> dict[str, Any]:
        """Resume a stopped version once, preserving stage costs and granting a fresh retry allowance."""
        with self._tx() as c:
            job = c.execute("SELECT * FROM jobs WHERE id=? AND user_id=? AND deleted_at IS NULL",
                            (job_id, user_id)).fetchone()
            if job is None:
                raise NotFound("No such hero.")
            if job["state"] in ACTIVE_STATES and job["continued_after"] > 0 and job["stop_after"] is None:
                return dict(job)  # an HTTP retry never starts another generation
            if job["state"] != "stopped":
                raise Conflict("Only a version stopped after cover or turnaround can be continued.")
            user = c.execute("SELECT status,kind FROM users WHERE id=? AND deleted_at IS NULL", (user_id,)).fetchone()
            if user is None or user["status"] != "active":
                raise StoreError("This account cannot start jobs.", "account")
            used = c.execute("SELECT COALESCE(SUM(bytes),0) FROM jobs WHERE user_id=? AND deleted_at IS NULL",
                             (user_id,)).fetchone()[0]
            if used >= storage_max_bytes:
                raise StoreError("Your saved heroes use all of your storage. Delete some to make room.", "storage_full")
            self._guard_generation(c)
            self._set_state(c, job, "queued", continued_after=job["attempts"], stop_after=None, finished_at=None, dispatched_at=None,
                            call_id=None, lease_until=None, error_code=None, error_stage=None, error_message=None)
            self._audit(c, user_id, "job_continued", job_id)
            return dict(c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def job_versions(self, job_id: str, user_id: str) -> list[dict[str, Any]]:
        """The owned version family, including descendants; deleted versions are never returned."""
        with self.lock:
            jobs = {r["id"]: dict(r) for r in self.db.execute("SELECT * FROM jobs WHERE user_id=?",
                                                            (user_id,))}
            if job_id not in jobs or jobs[job_id]["deleted_at"] is not None:
                raise NotFound("No such hero.")
            root, seen = job_id, set()
            while jobs[root]["parent_job"] in jobs and root not in seen:
                seen.add(root)
                root = jobs[root]["parent_job"]
            family = {root}
            for _ in range(len(jobs)):
                added = {jid for jid, j in jobs.items() if j["parent_job"] in family} - family
                if not added:
                    break
                family.update(added)
            return sorted((jobs[j] for j in family if jobs[j]["deleted_at"] is None),
                          key=lambda j: (j["created_at"], j["id"]))

    def job_get(self, job_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        if user_id is None:
            return self._one("SELECT * FROM jobs WHERE id=?", (job_id,))
        return self._one("SELECT * FROM jobs WHERE id=? AND user_id=? AND deleted_at IS NULL", (job_id, user_id))

    def jobs_list(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM jobs WHERE user_id=? AND deleted_at IS NULL ORDER BY created_at DESC LIMIT ?",
                         (user_id, limit))

    def jobs_admin_list(self, limit: int = 50, state: str | None = None) -> list[dict[str, Any]]:
        if state:
            return self._all("SELECT * FROM jobs WHERE state=? ORDER BY created_at DESC LIMIT ?", (state, limit))
        return self._all("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))

    def job_stages(self, job_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT stage, state, attempt, started_at, seconds, cost_usd, message FROM job_stages "
                         "WHERE job_id=?", (job_id,))

    def _set_state(self, c: sqlite3.Connection, job: sqlite3.Row | dict, new: str, **cols: Any) -> None:
        old = job["state"]
        if new != old and new not in TRANSITIONS[old]:
            raise InvalidTransition(f"a {old} job cannot become {new}")
        cols["state"] = new
        sets = ", ".join(f"{k}=?" for k in cols)
        c.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*cols.values(), job["id"]))
        if old == "running" and new != "running":
            self._end_stages(c, job["id"], STAGE_WHEN_ENDED.get(new, "failed"))

    @staticmethod
    def _end_stages(c: sqlite3.Connection, job_id: str, state: str) -> None:
        """Close the job's own stages left running when its attempt ends. Only the live attempt may write a stage
        (`job_stage_set`), so after a cancel, a delete or the sweep's requeue or failure nothing else ever would,
        and the stage would show as running for good (the page keeps polling while a stage runs). Side stages
        (`claim` set) from historical records are left alone."""
        c.execute("UPDATE job_stages SET state=?, lease_until=NULL WHERE job_id=? AND state='running' AND claim IS "
                  "NULL", (state, job_id))

    def job_dispatch(self, job_id: str, call_id: str) -> bool:
        """Record that a worker was started for a queued job."""
        with self._tx() as c:
            n = c.execute("UPDATE jobs SET dispatched_at=?, call_id=? WHERE id=? AND state='queued'",
                          (self.clock(), call_id, job_id)).rowcount
        return n > 0

    def jobs_to_dispatch(self, max_running: int) -> list[dict[str, Any]]:
        """Queued jobs a worker should be started for now, oldest first, keeping at most `max_running` jobs running
        or starting. A job dispatched less than DISPATCH_GRACE_S ago counts as starting."""
        now = self.clock()
        with self._tx() as c:
            self._expire_leases(c)
            max_running = c.execute("SELECT max_running FROM generation_settings WHERE id=1").fetchone()[0]
            busy = c.execute(
                "SELECT COUNT(*) FROM jobs WHERE state='running' OR (state='queued' AND dispatched_at>?)",
                (now - DISPATCH_GRACE_S,)).fetchone()[0]
            free = max(0, max_running - busy)
            if not free:
                return []
            rows = [dict(r) for r in c.execute(
                "SELECT * FROM jobs WHERE state='queued' AND deleted_at IS NULL AND (dispatched_at IS NULL OR "
                "dispatched_at<=?) ORDER BY created_at,rowid LIMIT ?", (now - DISPATCH_GRACE_S, free))]
            for row in rows:
                c.execute('UPDATE jobs SET dispatched_at=? WHERE id=?', (now, row['id']))
            return rows

    @staticmethod
    def _attempts_in_phase(job) -> int:
        return job["attempts"] - job["continued_after"]

    def job_claim(self, job_id: str, call_id: str, lease_s: float, max_attempts: int) -> dict[str, Any] | None:
        """A worker's first call. Starts the next attempt of a queued job and returns {"attempt", "job"}; returns
        None for a job that is running, finished, deleted or unknown, so a duplicate or re-run worker (Modal may
        start the same input again after a crash) does nothing. A job out of attempts fails here instead, before
        any paid call."""
        now = self.clock()
        with self._tx() as c:
            job = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if job is None or job["state"] != "queued" or job["deleted_at"] is not None:
                return None
            if self._attempts_in_phase(job) >= max_attempts:
                self._fail_ours(c, job, "attempts", job["stage"], "The job stopped unexpectedly several times.")
                return None
            attempt = job["attempts"] + 1
            self._set_state(c, job, "running", attempts=attempt, lease_until=now + lease_s, call_id=call_id,
                            started_at=job["started_at"] or now)
            self._audit(c, "system", "job_claimed", job_id, {"attempt": attempt})
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return {"attempt": attempt, "job": dict(row)}

    def _current(self, c: sqlite3.Connection, job_id: str, attempt: int) -> sqlite3.Row | None:
        job = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if job is None or job["state"] != "running" or job["attempts"] != attempt or job["deleted_at"] is not None:
            return None
        return job

    def job_current(self, job_id: str, attempt: int, lease_s: float | None = None, stage: str | None = None) -> bool:
        """True while `attempt` is the job's live attempt (not canceled, finished, deleted or superseded); extends
        the lease and records the stage when given."""
        with self._tx() as c:
            job = self._current(c, job_id, attempt)
            if job is None:
                return False
            if lease_s is not None:
                c.execute("UPDATE jobs SET lease_until=?, stage=COALESCE(?, stage) WHERE id=?",
                          (self.clock() + lease_s, stage, job_id))
        return True

    def job_renew(self, job_id: str, attempt: int, lease_s: float) -> bool:
        """The orchestrator's heartbeat: keep the live attempt's lease at least `lease_s` ahead (a longer stage
        lease is kept). False once `attempt` is not the live one, and renewing stops."""
        with self._tx() as c:
            if self._current(c, job_id, attempt) is None:
                return False
            c.execute("UPDATE jobs SET lease_until=MAX(COALESCE(lease_until, 0), ?) WHERE id=?",
                      (self.clock() + lease_s, job_id))
        return True


    def group_enter(self, job_id: str, attempt: int, group: str, max_entries: int) -> bool:
        """Count a start of one stage group for one attempt. False once it has started more than `max_entries`
        times (Modal starts an input again after its container dies, without limit) or when the attempt is not
        live; the caller then stops before any paid call."""
        with self._tx() as c:
            if self._current(c, job_id, attempt) is None:
                return False
            c.execute("INSERT INTO job_runs(job_id, attempt, grp, entries) VALUES (?,?,?,1) ON CONFLICT(job_id, "
                      "attempt, grp) DO UPDATE SET entries=entries+1", (job_id, attempt, group))
            n = c.execute("SELECT entries FROM job_runs WHERE job_id=? AND attempt=? AND grp=?",
                          (job_id, attempt, group)).fetchone()[0]
        return n <= max_entries

    def job_stage_set(self, job_id: str, attempt: int, stage: str, state: str, *, seconds: float | None = None,
                      cost_usd: float | None = None, message: str = "", lease_s: float | None = None) -> bool:
        if state not in STAGE_STATES:
            raise StoreError(f"bad stage state {state!r}")
        now = self.clock()
        with self._tx() as c:
            if self._current(c, job_id, attempt) is None:
                return False
            c.execute("INSERT INTO job_stages(job_id, stage, state, attempt, started_at, seconds, cost_usd, message) "
                      "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(job_id, stage) DO UPDATE SET state=excluded.state, "
                      "attempt=excluded.attempt, started_at=COALESCE(CASE WHEN excluded.state='running' THEN "
                      "excluded.started_at END, job_stages.started_at), "
                      "seconds=COALESCE(excluded.seconds, job_stages.seconds), "
                      "cost_usd=COALESCE(excluded.cost_usd, job_stages.cost_usd), message=excluded.message",
                      (job_id, stage, state, attempt, now, seconds, cost_usd, message[:2000]))
            c.execute("UPDATE jobs SET stage=?, lease_until=COALESCE(?, lease_until) WHERE id=?",
                      (stage, now + lease_s if lease_s else None, job_id))
        return True

    def job_finish(self, job_id: str, attempt: int, state: str, *, error_code: str | None = None,
                   error_stage: str | None = None, error_message: str | None = None, seconds: float | None = None,
                   cost_usd: float | None = None, cost_detail: Any = None, bytes_: int | None = None,
                   ) -> bool:
        """End the live attempt and record its cost."""
        if state not in ("succeeded", "rejected", "failed", "stopped"):
            raise StoreError(f"bad final state {state!r}")
        with self._tx() as c:
            job = self._current(c, job_id, attempt)
            if job is None:
                return False
            self._set_state(c, job, state, finished_at=self.clock(), error_code=error_code, error_stage=error_stage,
                            error_message=error_message, seconds=seconds, cost_usd=cost_usd,
                            cost_detail=json.dumps(cost_detail) if cost_detail is not None else None,
                            bytes=bytes_, lease_until=None)
            self._audit(c, "system", f"job_{state}", job_id, {"stage": error_stage, "code": error_code,
                                                               "attempt": attempt})
        return True

    def job_requeue(self, job_id: str, attempt: int, max_attempts: int, error_code: str, error_stage: str | None,
                    error_message: str) -> str | None:
        """After an infrastructure error: queue the job for another attempt, or fail it when it is out of attempts
        Returns the new state, or None when `attempt` is not the live
        attempt."""
        with self._tx() as c:
            job = self._current(c, job_id, attempt)
            if job is None:
                return None
            return self._requeue(c, job, max_attempts, error_code, error_stage, error_message)

    def _requeue(self, c: sqlite3.Connection, job: sqlite3.Row, max_attempts: int, error_code: str,
                 error_stage: str | None, error_message: str) -> str:
        if self._attempts_in_phase(job) >= max_attempts:
            self._fail_ours(c, job, error_code, error_stage, error_message)
            return "failed"
        self._set_state(c, job, "queued", lease_until=None, dispatched_at=None, call_id=None,
                        cost_usd=self._recorded_cost(c, job["id"]), error_code=error_code,
                        error_stage=error_stage, error_message=error_message)
        self._audit(c, "system", "job_requeued", job["id"], {"code": error_code, "stage": error_stage})
        return "queued"


    def _fail_ours(self, c: sqlite3.Connection, job: sqlite3.Row, error_code: str, error_stage: str | None,
                   error_message: str) -> None:
        """Fail a job and retain its recorded stage costs."""
        self._set_state(c, job, "failed", finished_at=self.clock(), error_code=error_code, error_stage=error_stage,
                        cost_usd=self._recorded_cost(c, job["id"]), error_message=error_message, lease_until=None)
        self._audit(c, "system", "job_failed", job["id"], {"code": error_code, "stage": error_stage})

    def _recorded_cost(self, c: sqlite3.Connection, job_id: str) -> float:
        """Cumulative main-stage spending, including on retry or worker loss."""
        return round(c.execute("SELECT COALESCE(SUM(cost_usd),0) FROM job_stages WHERE job_id=? AND stage!='2h'", (job_id,)).fetchone()[0], 6)

    def _expire_leases(self, c):
        out = []
        for job in c.execute("SELECT * FROM jobs WHERE state='running' AND lease_until<=?", (self.clock(),)).fetchall():
            self._fail_ours(c, job, 'worker_lost', job['stage'], 'The worker stopped responding. Please create another hero.')
            out.append((job['id'], 'failed'))
        return out

    def reconcile_stale(self, max_attempts):
        """End lost workers; only their recorded stage cost remains charged."""
        with self._tx() as c:
            return self._expire_leases(c)

    def job_cancel(self, job_id: str, user_id: str) -> dict[str, Any]:
        with self._tx() as c:
            job = c.execute("SELECT * FROM jobs WHERE id=? AND user_id=? AND deleted_at IS NULL",
                            (job_id, user_id)).fetchone()
            if job is None:
                raise NotFound("no such job")
            if job["state"] in ACTIVE_STATES:
                self._set_state(c, job, "canceled", finished_at=self.clock(), lease_until=None,
                                cost_usd=self._recorded_cost(c, job_id))
                self._audit(c, user_id, "job_canceled", job_id)
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row)

    def job_delete(self, job_id: str, user_id: str) -> dict[str, Any]:
        """Cancel the job if it is still active and mark it deleted; the caller removes its files."""
        self.job_cancel(job_id, user_id)
        with self._tx() as c:
            c.execute("UPDATE jobs SET note='', deleted_at=? WHERE id=? AND user_id=?", (self.clock(), job_id, user_id))
            self._audit(c, user_id, "job_deleted", job_id)
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row)

    # ---- retention --------------------------------------------------------------------------------------

    def retention_due(self, retention_s: float) -> list[dict[str, Any]]:
        """Finished jobs older than the retention period, and deleted jobs whose files may still exist."""
        cutoff = self.clock() - retention_s
        return self._all("SELECT id, user_id FROM jobs WHERE bytes IS NOT -1 AND ((state IN "
                         "('succeeded','rejected','failed','canceled','stopped') AND COALESCE(finished_at, created_at)<?) "
                         "OR deleted_at IS NOT NULL)", (cutoff,))

    def job_mark_purged(self, job_id: str) -> None:
        """The job's files are gone: keep the row for cost accounting and audit but drop its inputs."""
        with self._tx() as c:
            c.execute("UPDATE jobs SET bytes=-1, prompt='', name='', note='', deleted_at=COALESCE(deleted_at, ?) WHERE id=?",
                      (self.clock(), job_id))
            self._audit(c, "system", "job_purged", job_id)

    def purge_expired(self, audit_retention_s: float, analytics_retention_s: float = 30 * DAY_S) -> dict[str, int]:
        now = self.clock()
        with self._tx() as c:
            t = c.execute("DELETE FROM login_tokens WHERE expires_at<? OR used_at<?", (now - DAY_S, now - DAY_S)).rowcount
            s = c.execute("DELETE FROM sessions WHERE expires_at<? OR revoked_at<?", (now, now - DAY_S)).rowcount
            a = c.execute("DELETE FROM audit WHERE at<?", (now - audit_retention_s,)).rowcount
            ev = c.execute("DELETE FROM analytics_events WHERE at<?", (now - analytics_retention_s,)).rowcount
        return {"login_tokens": t, "sessions": s, "audit": a, "analytics_events": ev}

    def audit_list(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))


def ensure_private(path: Path) -> None:
    """Make a local database file readable by its owner only."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
