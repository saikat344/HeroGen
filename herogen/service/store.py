"""The metadata store: users, allowlist, invites, sign-in links, sessions, jobs, stages, audit log.

One SQLite database (WAL journal) written by one process at a time. Every write runs in a `BEGIN IMMEDIATE`
transaction under a lock and calls `persist()` afterwards, which on Modal snapshots the database to its Volume
before the call returns (see `snapshot.py`); locally the database file itself is the durable copy.

The store never sees a secret: sign-in links, sessions and invite codes arrive as keyed hashes (`auth.py`), and
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

SCHEMA_VERSION = 2
# Steps that bring an older database up to SCHEMA_VERSION, by the version they start from.
MIGRATIONS = {
    1: ("ALTER TABLE jobs ADD COLUMN refunded_at REAL",),
}
ID_RE = re.compile(r"^[uji]_[a-z2-7]{26}$")
ACTIVE_STATES = ("queued", "running")
TERMINAL_STATES = ("succeeded", "rejected", "failed", "canceled")
STATES = ACTIVE_STATES + TERMINAL_STATES
# The only state changes a job may make. Deleting a job is a flag (`deleted_at`), not a state.
TRANSITIONS = {
    "queued": {"running", "canceled", "failed"},
    "running": {"succeeded", "rejected", "failed", "canceled", "queued"},
    "succeeded": set(), "rejected": set(), "failed": set(), "canceled": set(),
}
STAGE_STATES = ("pending", "running", "passed", "rejected", "failed", "skipped")
DISPATCH_GRACE_S = 300.0    # a queued job dispatched this recently is taken to be starting
SEEN_EVERY_S = 300.0        # how often a session's last-seen time is written
DAY_S = 86400.0
REFUNDS_PER_DAY = 2         # failed jobs per user per 24 hours that may give their quota back
# Shown on a refunded job: the day's slot comes back, the job still counts toward the lifetime total.
REFUND_NOTE = " It does not use up one of today's heroes, but it still counts toward your total."

SCHEMA = """
CREATE TABLE users (
  id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, status TEXT NOT NULL DEFAULT 'active',
  created_at REAL NOT NULL, invite_id TEXT, terms_version TEXT, terms_accepted_at REAL, adult_confirmed_at REAL,
  quota_daily INTEGER, quota_total INTEGER, last_login_at REAL, deleted_at REAL
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
  deleted_at REAL, refunded_at REAL, UNIQUE(user_id, client_key)
);
CREATE INDEX jobs_user ON jobs(user_id, created_at);
CREATE INDEX jobs_state ON jobs(state);
CREATE TABLE job_stages (
  job_id TEXT NOT NULL, stage TEXT NOT NULL, state TEXT NOT NULL, attempt INTEGER NOT NULL,
  started_at REAL, seconds REAL, cost_usd REAL, message TEXT NOT NULL DEFAULT '', PRIMARY KEY (job_id, stage)
);
CREATE TABLE job_runs (
  job_id TEXT NOT NULL, attempt INTEGER NOT NULL, grp TEXT NOT NULL, entries INTEGER NOT NULL,
  PRIMARY KEY (job_id, attempt, grp)
);
CREATE TABLE audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
  subject TEXT NOT NULL DEFAULT '', detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX audit_at ON audit(at);
"""


class StoreError(Exception):
    """A request the store refuses. `code` is stable and safe to show; the message is for the user."""

    code = "store_error"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code

    def __reduce__(self):  # keep the code when the error crosses a Modal call
        return (type(self), (str(self), self.code))


class QuotaExceeded(StoreError):
    code = "quota"


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


class Store:
    # Methods the service's own functions may invoke by name (modal_app.StoreService.call).
    PUBLIC = frozenset({
        "rate_hit", "invite_valid", "signin_eligibility", "login_token_create", "login_token_info",
        "login_complete", "session_get", "session_revoke", "user_get", "usage", "user_delete",
        "job_create", "job_get", "jobs_list", "job_dispatch", "job_claim", "job_current", "job_stage_set",
        "group_enter", "job_finish", "job_requeue", "job_cancel", "job_delete", "job_stages", "jobs_to_dispatch",
        "reconcile_stale", "retention_due", "job_mark_purged", "purge_expired", "audit_add",
    })
    # Methods only the admin function may invoke (modal_app.StoreService.admin_call, which needs the admin key).
    ADMIN = frozenset({
        "allow_add", "allow_remove", "allow_list", "invite_create", "invite_list", "invite_revoke",
        "users_list", "user_set_status", "user_set_quota", "audit_list", "jobs_admin_list",
    })

    def __init__(self, path: Path | str, clock: Callable[[], float] = time.time,
                 persist: Callable[[sqlite3.Connection], None] | None = None,
                 refunds_per_day: int = REFUNDS_PER_DAY) -> None:
        self.refunds_per_day = refunds_per_day
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.persist = persist
        self.lock = threading.RLock()
        self.rates = RateLimiter(clock)
        self.db = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._migrate()

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

    # ---- allowlist and invites --------------------------------------------------------------------------

    def allow_add(self, email: str, note: str = "", subject: str = "") -> None:
        """`subject` is what the audit log records for the address: a keyed digest (`Signer.tag`), never the
        address itself."""
        with self._tx() as c:
            c.execute("INSERT INTO allowlist(email, added_at, note) VALUES (?,?,?) "
                      "ON CONFLICT(email) DO UPDATE SET note=excluded.note", (email, self.clock(), note))
            self._audit(c, "admin", "allow_add", subject)

    def allow_remove(self, email: str, subject: str = "") -> bool:
        with self._tx() as c:
            n = c.execute("DELETE FROM allowlist WHERE email=?", (email,)).rowcount
            self._audit(c, "admin", "allow_remove", subject)
        return n > 0

    def allow_list(self) -> list[dict[str, Any]]:
        return self._all("SELECT email, added_at, note FROM allowlist ORDER BY added_at")

    def invite_create(self, code_hash: str, max_uses: int, ttl_s: float, email: str | None = None,
                      note: str = "") -> dict[str, Any]:
        if max_uses < 1 or ttl_s <= 0:
            raise StoreError("an invite needs at least one use and a positive lifetime")
        iid = new_id("i")
        now = self.clock()
        with self._tx() as c:
            c.execute("INSERT INTO invites(id, code_hash, created_at, expires_at, max_uses, email, note) "
                      "VALUES (?,?,?,?,?,?,?)", (iid, code_hash, now, now + ttl_s, max_uses, email, note))
            self._audit(c, "admin", "invite_create", iid, {"max_uses": max_uses, "bound": bool(email)})
        return self._invite_row(iid)

    def _invite_row(self, iid: str) -> dict[str, Any]:
        row = self._one("SELECT id, created_at, expires_at, max_uses, uses, email, note, revoked_at FROM invites "
                        "WHERE id=?", (iid,))
        if row is None:
            raise NotFound("no such invite")
        return row

    def invite_list(self) -> list[dict[str, Any]]:
        return self._all("SELECT id, created_at, expires_at, max_uses, uses, email, note, revoked_at FROM invites "
                         "ORDER BY created_at")

    def invite_revoke(self, invite_id: str) -> bool:
        with self._tx() as c:
            n = c.execute("UPDATE invites SET revoked_at=? WHERE id=? AND revoked_at IS NULL",
                          (self.clock(), invite_id)).rowcount
            self._audit(c, "admin", "invite_revoke", invite_id)
        return n > 0

    def _open_invite(self, c: sqlite3.Connection, invite_hash: str) -> sqlite3.Row | None:
        row = c.execute("SELECT * FROM invites WHERE code_hash=?", (invite_hash,)).fetchone()
        if (row is None or row["revoked_at"] is not None or row["expires_at"] <= self.clock()
                or row["uses"] >= row["max_uses"]):
            return None
        return row

    def invite_valid(self, invite_hash: str) -> bool:
        """Whether an invite code exists and is open (not revoked, expired or used up), judged from the code alone:
        the answer never depends on an address. Read-only."""
        with self.lock:
            return self._open_invite(self.db, invite_hash) is not None

    def _usable_invite(self, c: sqlite3.Connection, invite_id: str | None, email: str) -> dict[str, Any] | None:
        if not invite_id:
            return None
        row = c.execute("SELECT * FROM invites WHERE id=?", (invite_id,)).fetchone()
        if (row is None or row["revoked_at"] is not None or row["expires_at"] <= self.clock()
                or row["uses"] >= row["max_uses"] or (row["email"] and row["email"] != email)):
            return None
        return dict(row)

    # ---- sign-in ----------------------------------------------------------------------------------------

    def signin_eligibility(self, email: str, invite_hash: str | None) -> dict[str, Any]:
        """Whether a sign-in link may be sent to `email`: an active account, an allowlisted address, or an invite
        code usable by this address. `invalid_invite` says whether the code itself is not open, judged from the code
        alone (see `invite_valid`); a code bound to another address is open but gives this address nothing."""
        with self.lock:
            c = self.db
            user = c.execute("SELECT status FROM users WHERE email=? AND deleted_at IS NULL", (email,)).fetchone()
            invite_id = None
            invalid_invite = False
            if invite_hash:
                row = self._open_invite(c, invite_hash)
                invalid_invite = row is None
                inv = self._usable_invite(c, row["id"] if row else None, email)
                invite_id = inv["id"] if inv else None
            if user is not None:
                return {"eligible": user["status"] == "active", "invite_id": None, "invalid_invite": invalid_invite,
                        "existing": True}
            allowed = c.execute("SELECT 1 FROM allowlist WHERE email=?", (email,)).fetchone() is not None
            return {"eligible": allowed or invite_id is not None, "invite_id": None if allowed else invite_id,
                    "invalid_invite": invalid_invite, "existing": False}

    def login_token_create(self, token_hash: str, email: str, ttl_s: float, invite_id: str | None = None,
                           actor: str = "") -> None:
        now = self.clock()
        with self._tx() as c:
            c.execute("INSERT INTO login_tokens(hash, email, invite_id, created_at, expires_at) VALUES (?,?,?,?,?)",
                      (token_hash, email, invite_id, now, now + ttl_s))
            self._audit(c, actor or "anonymous", "signin_link_sent", "", {"invite": bool(invite_id)})

    def _live_token(self, c: sqlite3.Connection, token_hash: str) -> dict[str, Any] | None:
        row = c.execute("SELECT * FROM login_tokens WHERE hash=?", (token_hash,)).fetchone()
        if row is None or row["used_at"] is not None or row["expires_at"] <= self.clock():
            return None
        return dict(row)

    def login_token_info(self, token_hash: str, terms_version: str) -> dict[str, Any] | None:
        """What the confirmation page shows before a link is used: its address and whether terms must be
        accepted. Nothing is consumed."""
        with self.lock:
            tok = self._live_token(self.db, token_hash)
            if tok is None:
                return None
            user = self.db.execute("SELECT * FROM users WHERE email=? AND deleted_at IS NULL",
                                   (tok["email"],)).fetchone()
        new = user is None
        return {"email": tok["email"], "new_account": new, "expires_at": tok["expires_at"],
                "needs_terms": new or (user["terms_version"] or "") != terms_version}

    def login_complete(self, token_hash: str, session_hash: str, *, terms_version: str, accept_terms: bool,
                       adult: bool, session_ttl_s: float) -> dict[str, Any]:
        """Use a sign-in link: consume it, create the account if this is its first use, and open a session, all in
        one transaction. Returns {"status": "ok", "user": ...} or {"status": <reason>} with nothing consumed
        unless the status is "ok" (so a user who still has to accept the terms can do so with the same link)."""
        now = self.clock()
        with self._tx() as c:
            tok = self._live_token(c, token_hash)
            if tok is None:
                return {"status": "invalid"}
            email = tok["email"]
            user = c.execute("SELECT * FROM users WHERE email=? AND deleted_at IS NULL", (email,)).fetchone()
            if user is not None and user["status"] != "active":
                return {"status": "disabled"}
            needs_terms = user is None or (user["terms_version"] or "") != terms_version
            if needs_terms and not (accept_terms and adult and terms_version):
                return {"status": "needs_terms"}
            if user is None:
                allowed = c.execute("SELECT 1 FROM allowlist WHERE email=?", (email,)).fetchone() is not None
                inv = None if allowed else self._usable_invite(c, tok["invite_id"], email)
                if not allowed and inv is None:
                    return {"status": "not_invited"}
                uid = new_id("u")
                c.execute("INSERT INTO users(id, email, created_at, invite_id, terms_version, terms_accepted_at, "
                          "adult_confirmed_at) VALUES (?,?,?,?,?,?,?)",
                          (uid, email, now, inv["id"] if inv else None, terms_version, now, now))
                if inv:
                    c.execute("UPDATE invites SET uses=uses+1 WHERE id=?", (inv["id"],))
                self._audit(c, uid, "account_created", uid, {"invite": inv["id"] if inv else None})
            else:
                uid = user["id"]
                if needs_terms:
                    c.execute("UPDATE users SET terms_version=?, terms_accepted_at=?, adult_confirmed_at=? WHERE id=?",
                              (terms_version, now, now, uid))
                    self._audit(c, uid, "terms_accepted", uid, {"version": terms_version})
            # Every outstanding link for this address dies with the one used.
            c.execute("UPDATE login_tokens SET used_at=? WHERE email=? AND used_at IS NULL", (now, email))
            c.execute("INSERT INTO sessions(hash, user_id, created_at, expires_at, last_seen_at) VALUES (?,?,?,?,?)",
                      (session_hash, uid, now, now + session_ttl_s, now))
            c.execute("UPDATE users SET last_login_at=? WHERE id=?", (now, uid))
            self._audit(c, uid, "signed_in", uid)
            row = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        return {"status": "ok", "user": self._public_user(dict(row))}

    @staticmethod
    def _public_user(row: dict[str, Any]) -> dict[str, Any]:
        return {k: row[k] for k in ("id", "email", "status", "created_at", "terms_version", "quota_daily",
                                    "quota_total", "last_login_at")}

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
        return self.user_get(row["user_id"])

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

    def user_set_quota(self, email: str, daily: int | None, total: int | None) -> dict[str, Any]:
        with self._tx() as c:
            row = self._user_by_email(c, email)
            c.execute("UPDATE users SET quota_daily=?, quota_total=? WHERE id=?", (daily, total, row["id"]))
            self._audit(c, "admin", "user_quota", row["id"], {"daily": daily, "total": total})
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
            c.execute("UPDATE jobs SET deleted_at=COALESCE(deleted_at, ?), state=CASE WHEN state IN "
                      "('queued','running') THEN 'canceled' ELSE state END WHERE user_id=?", (now, user_id))
            c.execute("UPDATE sessions SET revoked_at=? WHERE user_id=? AND revoked_at IS NULL", (now, user_id))
            c.execute("UPDATE users SET email=?, deleted_at=?, status='disabled' WHERE id=?",
                      (f"deleted:{user_id}", now, user_id))
            # Forget the address everywhere it may be: links, the allowlist, invites bound to it (revoked, so the
            # code does not become usable by anyone) and any audit entry naming it.
            email = row["email"]
            c.execute("DELETE FROM login_tokens WHERE email=?", (email,))
            c.execute("DELETE FROM allowlist WHERE email=?", (email,))
            c.execute("UPDATE invites SET email=NULL, revoked_at=COALESCE(revoked_at, ?) WHERE email=?", (now, email))
            c.execute("UPDATE audit SET subject='' WHERE subject=?", (email,))
            c.execute("UPDATE audit SET detail='' WHERE instr(detail, ?) > 0", (email,))
            self._audit(c, user_id, "account_deleted", user_id, {"jobs": len(jobs)})
        return jobs

    def usage(self, user_id: str, jobs_per_day: int, jobs_total: int) -> dict[str, Any]:
        with self.lock:
            return self._usage(self.db, user_id, jobs_per_day, jobs_total)

    def _usage(self, c: sqlite3.Connection, user_id: str, jobs_per_day: int, jobs_total: int) -> dict[str, Any]:
        now = self.clock()
        user = c.execute("SELECT quota_daily, quota_total FROM users WHERE id=?", (user_id,)).fetchone()
        day_limit = user["quota_daily"] if user and user["quota_daily"] is not None else jobs_per_day
        total_limit = user["quota_total"] if user and user["quota_total"] is not None else jobs_total
        day = c.execute("SELECT COUNT(*) FROM jobs WHERE user_id=? AND counts_quota=1 AND created_at>?",
                        (user_id, now - DAY_S)).fetchone()[0]
        # A refund gives back the day's slot only: refunded jobs still count toward the lifetime total.
        total = c.execute("SELECT COUNT(*) FROM jobs WHERE user_id=? AND (counts_quota=1 OR refunded_at IS NOT NULL)",
                          (user_id,)).fetchone()[0]
        oldest = c.execute("SELECT MIN(created_at) FROM jobs WHERE user_id=? AND counts_quota=1 AND created_at>?",
                           (user_id, now - DAY_S)).fetchone()[0]
        stored = c.execute("SELECT COALESCE(SUM(bytes),0) FROM jobs WHERE user_id=? AND deleted_at IS NULL",
                           (user_id,)).fetchone()[0]
        return {"jobs_today": day, "jobs_per_day": day_limit, "jobs_total": total, "jobs_total_limit": total_limit,
                "next_daily_slot_at": (oldest + DAY_S) if day >= day_limit and oldest else None,
                "stored_bytes": stored}

    # ---- jobs -------------------------------------------------------------------------------------------

    def job_create(self, *, job_id: str, user_id: str, client_key: str, kind: str, profile: str,
                   jobs_per_day: int, jobs_total: int, storage_max_bytes: int, input_sha: str = "", prompt: str = "",
                   name: str = "", code_commit: str = "") -> tuple[dict[str, Any], bool]:
        """Create a queued job, or return the one this user already made with the same `client_key`.

        Idempotent: the same key and request returns the first job and `False`; the same key with a different
        request raises Conflict. The quota (jobs in the last 24 hours and in total, counting jobs that cost money)
        and the storage limit are checked in the same transaction, before anything is paid for."""
        if kind not in ("sketch", "prompt") or not valid_id(job_id, "j"):
            raise StoreError("bad job request")
        with self._tx() as c:
            old = c.execute("SELECT * FROM jobs WHERE user_id=? AND client_key=?", (user_id, client_key)).fetchone()
            if old is not None:
                same = (old["kind"], old["input_sha"], old["prompt"], old["name"], old["profile"]) == (
                    kind, input_sha, prompt, name, profile)
                if not same:
                    raise Conflict("this request key was already used for a different hero")
                return dict(old), False
            user = c.execute("SELECT status FROM users WHERE id=? AND deleted_at IS NULL", (user_id,)).fetchone()
            if user is None or user["status"] != "active":
                raise StoreError("this account cannot start jobs", "account")
            u = self._usage(c, user_id, jobs_per_day, jobs_total)
            if u["jobs_total"] >= u["jobs_total_limit"]:
                raise QuotaExceeded(f"you have used all {u['jobs_total_limit']} heroes of this beta account")
            if u["jobs_today"] >= u["jobs_per_day"]:
                raise QuotaExceeded(f"you have made {u['jobs_per_day']} heroes in the last 24 hours; try again later")
            if u["stored_bytes"] >= storage_max_bytes:
                raise QuotaExceeded("your saved heroes use all of your storage; delete some to make room")
            c.execute("INSERT INTO jobs(id, user_id, client_key, kind, input_sha, prompt, name, profile, state, "
                      "created_at, code_commit) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (job_id, user_id, client_key, kind, input_sha, prompt, name, profile, "queued", self.clock(),
                       code_commit))
            self._audit(c, user_id, "job_created", job_id, {"kind": kind, "profile": profile})
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row), True

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
        with self.lock:
            busy = self.db.execute(
                "SELECT COUNT(*) FROM jobs WHERE state='running' OR (state='queued' AND dispatched_at>?)",
                (now - DISPATCH_GRACE_S,)).fetchone()[0]
            free = max(0, max_running - busy)
            if not free:
                return []
            return [dict(r) for r in self.db.execute(
                "SELECT * FROM jobs WHERE state='queued' AND deleted_at IS NULL AND (dispatched_at IS NULL OR "
                "dispatched_at<=?) ORDER BY created_at LIMIT ?", (now - DISPATCH_GRACE_S, free))]

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
            if job["attempts"] >= max_attempts:
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
                      "excluded.started_at END, job_stages.started_at), seconds=excluded.seconds, "
                      "cost_usd=excluded.cost_usd, message=excluded.message",
                      (job_id, stage, state, attempt, now, seconds, cost_usd, message[:2000]))
            c.execute("UPDATE jobs SET stage=?, lease_until=COALESCE(?, lease_until) WHERE id=?",
                      (stage, now + lease_s if lease_s else None, job_id))
        return True

    def job_finish(self, job_id: str, attempt: int, state: str, *, error_code: str | None = None,
                   error_stage: str | None = None, error_message: str | None = None, seconds: float | None = None,
                   cost_usd: float | None = None, cost_detail: Any = None, bytes_: int | None = None,
                   refund: bool = False) -> bool:
        """End the live attempt as succeeded, rejected or failed. `refund` asks for a failed job to give its quota
        back because the failure was the service's, not the sketch's; `_refund_ok` decides."""
        if state not in ("succeeded", "rejected", "failed"):
            raise StoreError(f"bad final state {state!r}")
        with self._tx() as c:
            job = self._current(c, job_id, attempt)
            if job is None:
                return False
            refunded = state == "failed" and refund and self._refund_ok(c, job, cost_usd)
            if refunded and error_message is not None:
                error_message += REFUND_NOTE
            self._set_state(c, job, state, finished_at=self.clock(), error_code=error_code, error_stage=error_stage,
                            error_message=error_message, seconds=seconds, cost_usd=cost_usd,
                            cost_detail=json.dumps(cost_detail) if cost_detail is not None else None,
                            bytes=bytes_, counts_quota=0 if refunded else job["counts_quota"], lease_until=None,
                            refunded_at=self.clock() if refunded else None)
            self._audit(c, "system", f"job_{state}", job_id, {"stage": error_stage, "code": error_code,
                                                               "attempt": attempt})
        return True

    def job_requeue(self, job_id: str, attempt: int, max_attempts: int, error_code: str, error_stage: str | None,
                    error_message: str) -> str | None:
        """After an infrastructure error: queue the job for another attempt, or fail it when it is out of attempts
        (refunded when `_refund_ok` allows). Returns the new state, or None when `attempt` is not the live
        attempt."""
        with self._tx() as c:
            job = self._current(c, job_id, attempt)
            if job is None:
                return None
            return self._requeue(c, job, max_attempts, error_code, error_stage, error_message)

    def _requeue(self, c: sqlite3.Connection, job: sqlite3.Row, max_attempts: int, error_code: str,
                 error_stage: str | None, error_message: str) -> str:
        if job["attempts"] >= max_attempts:
            self._fail_ours(c, job, error_code, error_stage, error_message)
            return "failed"
        self._set_state(c, job, "queued", lease_until=None, dispatched_at=None, call_id=None, error_code=error_code,
                        error_stage=error_stage, error_message=error_message)
        self._audit(c, "system", "job_requeued", job["id"], {"code": error_code, "stage": error_stage})
        return "queued"

    def _refund_ok(self, c: sqlite3.Connection, job: sqlite3.Row, extra_cost: float | None = None) -> bool:
        """A failed job gives its quota back only if it spent nothing (its stages' recorded cost, model calls and
        compute together, is zero) and the user has had fewer than `refunds_per_day` refunds in the last 24 hours.
        Otherwise a stage that fails after paid calls could be used to run paid work without limit."""
        spent = c.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM job_stages WHERE job_id=?",
                          (job["id"],)).fetchone()[0]
        if spent + (extra_cost or 0.0) > 0:
            return False
        recent = c.execute("SELECT COUNT(*) FROM jobs WHERE user_id=? AND refunded_at>?",
                           (job["user_id"], self.clock() - DAY_S)).fetchone()[0]
        return recent < self.refunds_per_day

    def _fail_ours(self, c: sqlite3.Connection, job: sqlite3.Row, error_code: str, error_stage: str | None,
                   error_message: str) -> None:
        """Fail a job for the service's reasons, refunding it when `_refund_ok` allows."""
        refunded = self._refund_ok(c, job)
        self._set_state(c, job, "failed", finished_at=self.clock(), error_code=error_code, error_stage=error_stage,
                        error_message=error_message + (REFUND_NOTE if refunded else ""),
                        counts_quota=0 if refunded else job["counts_quota"], lease_until=None,
                        refunded_at=self.clock() if refunded else None)
        self._audit(c, "system", "job_failed", job["id"], {"code": error_code, "stage": error_stage,
                                                            "refunded": refunded})

    def reconcile_stale(self, max_attempts: int) -> list[tuple[str, str]]:
        """Running jobs whose lease ran out (the worker died without reporting): requeue or fail each one."""
        now = self.clock()
        out = []
        with self._tx() as c:
            for job in c.execute("SELECT * FROM jobs WHERE state='running' AND lease_until<?", (now,)).fetchall():
                out.append((job["id"], self._requeue(c, job, max_attempts, "worker_lost", job["stage"],
                                                     "The worker stopped responding.")))
        return out

    def job_cancel(self, job_id: str, user_id: str) -> dict[str, Any]:
        with self._tx() as c:
            job = c.execute("SELECT * FROM jobs WHERE id=? AND user_id=? AND deleted_at IS NULL",
                            (job_id, user_id)).fetchone()
            if job is None:
                raise NotFound("no such job")
            if job["state"] in ACTIVE_STATES:
                # A job that never started costs nothing and gives its quota back.
                refund = job["state"] == "queued" and job["attempts"] == 0
                self._set_state(c, job, "canceled", finished_at=self.clock(), lease_until=None,
                                counts_quota=0 if refund else job["counts_quota"])
                self._audit(c, user_id, "job_canceled", job_id)
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row)

    def job_delete(self, job_id: str, user_id: str) -> dict[str, Any]:
        """Cancel the job if it is still active and mark it deleted; the caller removes its files."""
        self.job_cancel(job_id, user_id)
        with self._tx() as c:
            c.execute("UPDATE jobs SET deleted_at=? WHERE id=? AND user_id=?", (self.clock(), job_id, user_id))
            self._audit(c, user_id, "job_deleted", job_id)
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row)

    # ---- retention --------------------------------------------------------------------------------------

    def retention_due(self, retention_s: float) -> list[dict[str, Any]]:
        """Finished jobs older than the retention period, and deleted jobs whose files may still exist."""
        cutoff = self.clock() - retention_s
        return self._all("SELECT id, user_id FROM jobs WHERE bytes IS NOT -1 AND ((state IN "
                         "('succeeded','rejected','failed','canceled') AND COALESCE(finished_at, created_at)<?) "
                         "OR deleted_at IS NOT NULL)", (cutoff,))

    def job_mark_purged(self, job_id: str) -> None:
        """The job's files are gone: keep the row (quota, cost and audit need it) but drop its inputs."""
        with self._tx() as c:
            c.execute("UPDATE jobs SET bytes=-1, prompt='', name='', deleted_at=COALESCE(deleted_at, ?) WHERE id=?",
                      (self.clock(), job_id))
            self._audit(c, "system", "job_purged", job_id)

    def purge_expired(self, audit_retention_s: float) -> dict[str, int]:
        now = self.clock()
        with self._tx() as c:
            t = c.execute("DELETE FROM login_tokens WHERE expires_at<? OR used_at<?", (now - DAY_S, now - DAY_S)).rowcount
            s = c.execute("DELETE FROM sessions WHERE expires_at<? OR revoked_at<?", (now, now - DAY_S)).rowcount
            a = c.execute("DELETE FROM audit WHERE at<?", (now - audit_retention_s,)).rowcount
        return {"login_tokens": t, "sessions": s, "audit": a}

    def audit_list(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))


def ensure_private(path: Path) -> None:
    """Make a local database file readable by its owner only."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
