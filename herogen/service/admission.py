"""Durable judge allowances and conservative generation admission, owned by the metadata store.

Amounts are integer micro-USD reservation bounds, not provider invoices. No automatic failure refund releases
money. An operator reconciles a reservation from provider evidence; uncertain worker exits retain capacity.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
import re

SCHEMA = '''
CREATE TABLE judge_codes (
 id TEXT PRIMARY KEY, code_hash TEXT NOT NULL UNIQUE, allowance INTEGER NOT NULL, used INTEGER NOT NULL DEFAULT 0,
 max_workspaces INTEGER NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL, revoked_at REAL,
 note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE judge_workspaces (user_id TEXT PRIMARY KEY, code_id TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE admission_policy (
 id INTEGER PRIMARY KEY CHECK(id=1), limit_micro INTEGER NOT NULL, reserve_micro INTEGER NOT NULL,
 max_active INTEGER NOT NULL, expires_at REAL NOT NULL, enabled INTEGER NOT NULL, code_commit TEXT NOT NULL,
 cost_signature TEXT NOT NULL, bound_note TEXT NOT NULL
);
CREATE TABLE admissions (
 job_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, code_id TEXT, client_key TEXT NOT NULL,
 amount_micro INTEGER NOT NULL, cost_signature TEXT NOT NULL, actual_micro INTEGER, main_done INTEGER NOT NULL DEFAULT 0,
 side_required INTEGER NOT NULL DEFAULT 1, side_done INTEGER NOT NULL DEFAULT 0,
 closed INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, evidence TEXT NOT NULL DEFAULT '',
 UNIQUE(code_id, client_key)
);
CREATE TABLE admission_work (
 token TEXT PRIMARY KEY, job_id TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL,
 created_at REAL NOT NULL, finished_at REAL
);
CREATE INDEX admission_work_job ON admission_work(job_id);
'''
PUBLIC = {'public_demo_redeem', 'judge_redeem', 'judge_info', 'admission_status', 'admission_work_begin', 'admission_work_end',
          'admission_main_done', 'admission_side_prepare', 'admission_side_done'}
ADMIN = {'judge_code_create', 'judge_codes_list', 'judge_code_revoke', 'judge_code_allowance',
         'budget_configure', 'budget_report', 'budget_reconcile'}


def micro_usd(value) -> int:
    try:
        amount = Decimal(str(value))
        if not amount.is_finite() or amount < 0 or amount * 1_000_000 != (amount * 1_000_000).to_integral_value():
            raise ValueError
        return int(amount * 1_000_000)
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError('USD amounts must be non-negative with at most six decimal places.') from None


class AdmissionStore:
    def judge_code_create(self, code_hash: str, allowance: int, expires_at: float, max_workspaces: int = 8,
                          note: str = '') -> dict:
        from herogen.service.store import new_id
        if not re.fullmatch(r'[a-f0-9]{64}', code_hash) or not isinstance(allowance, int) or allowance < 1:
            raise ValueError('A code needs a keyed hash and a positive generation allowance.')
        if not math.isfinite(expires_at) or expires_at <= self.clock() or not 1 <= max_workspaces <= 100:
            raise ValueError('A code needs a future expiry and 1–100 workspaces.')
        key = new_id('i')
        with self._tx() as c:
            c.execute('INSERT INTO judge_codes(id,code_hash,allowance,max_workspaces,created_at,expires_at,note) '
                      'VALUES (?,?,?,?,?,?,?)', (key, code_hash, allowance, max_workspaces, self.clock(), expires_at, note[:400]))
            self._audit(c, 'admin', 'judge_code_created', key, {'allowance': allowance, 'expires_at': expires_at})
        return {'id': key, 'allowance': allowance, 'expires_at': expires_at}

    def judge_codes_list(self) -> list[dict]:
        return self._all('SELECT id,allowance,used,max_workspaces,created_at,expires_at,revoked_at,note FROM judge_codes')

    def judge_code_revoke(self, code_id: str) -> bool:
        with self._tx() as c:
            n = c.execute('UPDATE judge_codes SET revoked_at=? WHERE id=? AND revoked_at IS NULL',
                          (self.clock(), code_id)).rowcount
            self._audit(c, 'admin', 'judge_code_revoked', code_id)
        return bool(n)

    def judge_code_allowance(self, code_id: str, allowance: int, note: str) -> None:
        if allowance < 0 or not note.strip():
            raise ValueError('Set a non-negative allowance and explain the adjustment.')
        with self._tx() as c:
            if not c.execute('UPDATE judge_codes SET allowance=? WHERE id=?', (allowance, code_id)).rowcount:
                raise ValueError('Judge code not found.')
            self._audit(c, 'admin', 'judge_allowance_changed', code_id, {'allowance': allowance, 'note': note[:400]})

    def judge_redeem(self, code_hash: str, session_hash: str, session_ttl_s: float, old_session_hash: str = '') -> dict:
        from herogen.service.store import StoreError, new_id
        now = self.clock()
        with self._tx() as c:
            code = c.execute('SELECT * FROM judge_codes WHERE code_hash=?', (code_hash,)).fetchone()
            if code is None or code['revoked_at'] is not None or code['expires_at'] <= now:
                raise StoreError('This judge code is invalid or has expired.', 'judge_code')
            prior = c.execute('SELECT s.user_id FROM sessions s JOIN judge_workspaces w ON w.user_id=s.user_id '
                              'JOIN users u ON u.id=w.user_id WHERE s.hash=? AND w.code_id=? AND s.revoked_at IS NULL '
                              'AND s.expires_at>? AND u.deleted_at IS NULL AND u.status=\'active\'',
                              (old_session_hash, code['id'], now)).fetchone()
            if prior:
                uid = prior['user_id']
            else:
                count = c.execute('SELECT COUNT(*) FROM judge_workspaces WHERE code_id=?', (code['id'],)).fetchone()[0]
                if count >= code['max_workspaces']:
                    raise StoreError('This code has reached its browser-workspace limit. Contact the project owner.', 'judge_workspaces')
                uid = new_id('u')
                # Internal namespace, never a deliverable email address or an accepted account agreement.
                c.execute('INSERT INTO users(id,email,created_at,kind) VALUES (?,?,?,?)',
                          (uid, f'workspace:{uid}', now, 'judge'))
                c.execute('INSERT INTO judge_workspaces VALUES (?,?,?)', (uid, code['id'], now))
            c.execute('INSERT INTO sessions(hash,user_id,created_at,expires_at,last_seen_at) VALUES (?,?,?,?,?)',
                      (session_hash, uid, now, now + session_ttl_s, now))
            if old_session_hash:
                c.execute('UPDATE sessions SET revoked_at=? WHERE hash=?', (now, old_session_hash))
            self._audit(c, uid, 'judge_workspace_opened', code['id'])
        return self.user_get(uid)

    def public_demo_redeem(self, session_hash: str, session_ttl_s: float, old_session_hash: str,
                           allowance: int, code_commit: str, cost_signature: str) -> dict:
        """A browser allowance is convenience, not identity or Sybil protection. Global reservations bind spend."""
        import secrets
        from herogen.service.store import StoreError
        with self.lock:
            prior = self._one('SELECT j.code_hash FROM sessions s JOIN judge_workspaces w ON w.user_id=s.user_id '
                              'JOIN judge_codes j ON j.id=w.code_id WHERE s.hash=? AND s.revoked_at IS NULL '
                              'AND s.expires_at>? AND j.revoked_at IS NULL AND j.expires_at>?',
                              (old_session_hash, self.clock(), self.clock()))
            if prior:
                return self.judge_redeem(prior['code_hash'], session_hash, session_ttl_s, old_session_hash)
            status = self._admission_status(self.db, code_commit, cost_signature)
            if not status['accepting']:
                raise StoreError('Live generation is currently unavailable. No job was queued. Inspect the public gallery or try again later.', status['reason'])
            policy = self._one('SELECT expires_at FROM admission_policy WHERE id=1')
            secret_hash = secrets.token_hex(32)
            self.judge_code_create(secret_hash, allowance, policy['expires_at'], 1, 'Public demo workspace')
            return self.judge_redeem(secret_hash, session_hash, session_ttl_s, old_session_hash)

    def judge_info(self, user_id: str) -> dict | None:
        row = self._one('SELECT j.id,j.allowance,j.used,j.expires_at,j.revoked_at FROM judge_workspaces w '
                        'JOIN judge_codes j ON j.id=w.code_id WHERE w.user_id=?', (user_id,))
        return None if row is None else {'remaining': max(0, row['allowance'] - row['used']),
                                        'expires_at': row['expires_at'], 'valid': row['revoked_at'] is None and row['expires_at'] > self.clock()}

    def budget_configure(self, limit_micro: int, reserve_micro: int, max_active: int, expires_at: float,
                         code_commit: str, bound_note: str, cost_signature: str, provider_caps_verified: bool = False, enabled: bool = True) -> None:
        if (not isinstance(limit_micro, int) or not isinstance(reserve_micro, int) or reserve_micro <= 0
                or limit_micro < reserve_micro or not 1 <= max_active <= 32 or not math.isfinite(expires_at)
                or expires_at <= self.clock() or not re.fullmatch(r'[a-f0-9]{40}', code_commit)
                or not re.fullmatch(r'[a-f0-9]{64}', cost_signature) or not bound_note.strip() or (enabled and not provider_caps_verified)):
            raise ValueError('Budget needs a positive verified job bound, sufficient funds, capacity, future expiry, exact commit and provider-cap confirmation.')
        with self._tx() as c:
            c.execute('INSERT INTO admission_policy VALUES (1,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET '
                      'limit_micro=excluded.limit_micro,reserve_micro=excluded.reserve_micro,max_active=excluded.max_active,'
                      'expires_at=excluded.expires_at,enabled=excluded.enabled,code_commit=excluded.code_commit,cost_signature=excluded.cost_signature,bound_note=excluded.bound_note',
                      (limit_micro, reserve_micro, max_active, expires_at, int(enabled), code_commit, cost_signature, bound_note[:1000]))
            self._audit(c, 'admin', 'generation_budget_configured', '', {'limit_micro': limit_micro, 'reserve_micro': reserve_micro,
                                                                     'max_active': max_active, 'commit': code_commit})

    def _admission_status(self, c, code_commit: str, cost_signature: str) -> dict:
        policy = c.execute('SELECT * FROM admission_policy WHERE id=1').fetchone()
        if policy is None or not policy['enabled'] or policy['expires_at'] <= self.clock():
            return {'accepting': False, 'reason': 'generation_paused'}
        if not code_commit or policy['code_commit'] != code_commit or policy['cost_signature'] != cost_signature:
            return {'accepting': False, 'reason': 'cost_bound_unverified'}
        used = c.execute('SELECT COALESCE(SUM(COALESCE(actual_micro,amount_micro)),0) FROM admissions').fetchone()[0]
        active = c.execute('SELECT COUNT(*) FROM admissions WHERE closed=0').fetchone()[0]
        legacy = c.execute("SELECT COUNT(*) FROM jobs WHERE budget_required=0 AND state IN ('queued','running') AND deleted_at IS NULL").fetchone()[0]
        if used + policy['reserve_micro'] > policy['limit_micro']:
            return {'accepting': False, 'reason': 'generation_budget_exhausted'}
        if active + legacy >= policy['max_active']:
            return {'accepting': False, 'reason': 'generation_capacity', 'retry_after': 5}
        return {'accepting': True, 'reason': 'available'}

    def admission_status(self, code_commit: str, cost_signature: str = '') -> dict:
        with self.lock:
            return self._admission_status(self.db, code_commit, cost_signature)

    def _admit(self, c, job_id: str, user_id: str, client_key: str, code_commit: str, cost_signature: str) -> None:
        from herogen.service.store import StoreError, Conflict
        code = c.execute('SELECT j.* FROM judge_workspaces w JOIN judge_codes j ON j.id=w.code_id WHERE w.user_id=?',
                         (user_id,)).fetchone()
        if not code and c.execute("SELECT kind FROM users WHERE id=?", (user_id,)).fetchone()['kind'] == 'judge':
            raise StoreError('Judge workspace permission is unavailable.', 'judge_code')
        if code:
            previous = c.execute('SELECT user_id FROM admissions WHERE code_id=? AND client_key=?', (code['id'], client_key)).fetchone()
            if previous and previous['user_id'] != user_id:
                raise Conflict('This request belongs to another private browser workspace. Return to that browser to recover it.')
            if code['revoked_at'] is not None or code['expires_at'] <= self.clock():
                raise StoreError('Judge access has expired or was revoked.', 'judge_code')
            if code['used'] >= code['allowance']:
                raise StoreError('This judge code has used its generation allowance. Existing results remain available.', 'judge_allowance')
            if c.execute('SELECT 1 FROM admissions WHERE code_id=? AND closed=0', (code['id'],)).fetchone():
                raise StoreError('This code already has a generation in progress or awaiting safe reconciliation. No new job was queued.', 'judge_active')
        elif c.execute('SELECT 1 FROM admissions WHERE user_id=? AND closed=0', (user_id,)).fetchone():
            raise StoreError('This workspace already has an active generation. No new job was queued.', 'judge_active')
        status = self._admission_status(c, code_commit, cost_signature)
        if not status['accepting']:
            messages = {'generation_paused': 'Live generation is temporarily paused. No job was queued.',
                        'cost_bound_unverified': 'Live generation is paused while this version’s cost bound is verified. No job was queued.',
                        'generation_budget_exhausted': 'The live-generation budget is currently allocated. Existing results remain available. No job was queued.',
                        'generation_capacity': 'All live-generation slots are occupied. No job was queued; please try again shortly.'}
            raise StoreError(messages[status['reason']], status['reason'])
        reserve = c.execute('SELECT reserve_micro FROM admission_policy WHERE id=1').fetchone()[0]
        c.execute('INSERT INTO admissions(job_id,user_id,code_id,client_key,amount_micro,cost_signature,created_at) VALUES (?,?,?,?,?,?,?)',
                  (job_id, user_id, code['id'] if code else None, client_key, reserve, cost_signature, self.clock()))
        if code:
            c.execute('UPDATE judge_codes SET used=used+1 WHERE id=?', (code['id'],))
        self._audit(c, user_id, 'generation_reserved', job_id, {'micro_usd': reserve})

    def admission_work_begin(self, job_id: str, token: str, kind: str, cost_signature: str = '') -> None:
        from herogen.service.store import StoreError
        with self._tx() as c:
            row = c.execute('SELECT closed,cost_signature FROM admissions WHERE job_id=?', (job_id,)).fetchone()
            if row is None or row['closed'] or row['cost_signature'] != cost_signature:
                raise StoreError('This generation has no open compute reservation.', 'generation_paused')
            if kind != 'heldout' and c.execute("SELECT 1 FROM admission_work WHERE job_id=? AND kind!='heldout' AND state!='closed'", (job_id,)).fetchone():
                raise StoreError('Earlier compute has not been confirmed stopped. Operator reconciliation is required.', 'generation_uncertain')
            c.execute('INSERT INTO admission_work(token,job_id,kind,state,created_at) VALUES (?,?,?,?,?)',
                      (token, job_id, kind, 'open', self.clock()))

    def _close_admission(self, c, job_id: str) -> None:
        c.execute('UPDATE admissions SET closed=1 WHERE job_id=? AND main_done=1 AND (side_required=0 OR side_done=1) '
                  'AND NOT EXISTS(SELECT 1 FROM admission_work WHERE job_id=? AND state!=\'closed\')', (job_id, job_id))

    def admission_work_end(self, job_id: str, token: str, certain: bool) -> None:
        with self._tx() as c:
            c.execute('UPDATE admission_work SET state=?,finished_at=? WHERE token=? AND job_id=? AND state=\'open\'',
                      ('closed' if certain else 'uncertain', self.clock(), token, job_id))
            self._close_admission(c, job_id)

    def admission_main_done(self, job_id: str) -> None:
        with self._tx() as c:
            c.execute('UPDATE admissions SET main_done=1 WHERE job_id=?', (job_id,))
            self._close_admission(c, job_id)

    def admission_side_prepare(self, job_id: str) -> None:
        with self._tx() as c:
            c.execute('UPDATE admissions SET side_required=1 WHERE job_id=? AND closed=0', (job_id,))

    def admission_side_done(self, job_id: str) -> None:
        with self._tx() as c:
            c.execute('UPDATE admissions SET side_done=1 WHERE job_id=?', (job_id,))
            self._close_admission(c, job_id)

    def budget_report(self) -> dict:
        return {'policy': self._one('SELECT * FROM admission_policy WHERE id=1'),
                'admissions': self._all('SELECT * FROM admissions ORDER BY created_at'),
                'work': self._all("SELECT job_id,kind,state,created_at,finished_at FROM admission_work WHERE state!='closed'")}

    def budget_reconcile(self, job_id: str, actual_micro: int, evidence: str, work_stopped: bool = False) -> None:
        if not isinstance(actual_micro, int) or actual_micro < 0 or not evidence.strip():
            raise ValueError('Reconciliation requires a non-negative cost and provider evidence.')
        with self._tx() as c:
            row = c.execute('SELECT * FROM admissions WHERE job_id=?', (job_id,)).fetchone()
            if row is None:
                raise ValueError('No reservation exists for this job.')
            job = c.execute('SELECT state FROM jobs WHERE id=?', (job_id,)).fetchone()
            if job and job['state'] in ('running', 'queued'):
                raise ValueError('Cannot reconcile an active job.')
            if not row['closed'] and not work_stopped:
                raise ValueError('Confirm that all main, side and uncertain remote work has stopped first.')
            if actual_micro > row['amount_micro']:
                c.execute('UPDATE admission_policy SET enabled=0 WHERE id=1')
            c.execute('UPDATE admissions SET actual_micro=?,closed=1,evidence=? WHERE job_id=?',
                      (actual_micro, evidence[:1000], job_id))
            self._audit(c, 'admin', 'generation_budget_reconciled', job_id,
                        {'actual_micro': actual_micro, 'evidence': evidence[:1000], 'work_stopped': work_stopped})
