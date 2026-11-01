"""Private guest workspaces and conservative generation admission, owned by the metadata store.

Amounts are integer micro-USD ledger bounds rounded up, not provider invoices. The periodic worker settles
terminal jobs after main and required side work finish, every recorded worker/process has stopped, and the
latest sent deadline plus 60 seconds has passed (latest worker start plus 60 seconds if no call was sent).
Completed calls use known costs; sent calls without completion and unknown calls use full price (Modal calls
use function timeout times price per second). Missing process/ledger evidence retains the reservation. Automatic
settlement uses the ledger bound, never a provider invoice; manual reconciliation wins if it commits first.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
import re
import json

SCHEMA = '''
CREATE TABLE admission_policy (
 id INTEGER PRIMARY KEY CHECK(id=1), limit_micro INTEGER NOT NULL, reserve_micro INTEGER NOT NULL,
 max_active INTEGER NOT NULL, expires_at REAL NOT NULL, enabled INTEGER NOT NULL, code_commit TEXT NOT NULL,
 cost_signature TEXT NOT NULL, bound_note TEXT NOT NULL
);
CREATE TABLE admissions (
 job_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, client_key TEXT NOT NULL,
 amount_micro INTEGER NOT NULL, cost_signature TEXT NOT NULL, actual_micro INTEGER, main_done INTEGER NOT NULL DEFAULT 0,
 side_required INTEGER NOT NULL DEFAULT 1, side_done INTEGER NOT NULL DEFAULT 0,
 closed INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, evidence TEXT NOT NULL DEFAULT ''
);
CREATE TABLE admission_work (
 token TEXT PRIMARY KEY, job_id TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL,
 created_at REAL NOT NULL, finished_at REAL, process TEXT NOT NULL DEFAULT ''
);
CREATE INDEX admission_work_job ON admission_work(job_id);
CREATE TABLE admission_calls (id INTEGER PRIMARY KEY, job_id TEXT NOT NULL, token TEXT NOT NULL, receipt TEXT NOT NULL);
CREATE TABLE admission_processes (token TEXT NOT NULL, host TEXT NOT NULL, pid INTEGER NOT NULL,
 stopped INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(token,host,pid));
CREATE TABLE IF NOT EXISTS budget_overhead (id INTEGER PRIMARY KEY CHECK(id=1), amount_micro INTEGER NOT NULL, evidence TEXT NOT NULL, updated_at REAL NOT NULL);
'''
PUBLIC = {'guest_open', 'admission_status', 'admission_work_begin', 'admission_work_end',
          'admission_main_done', 'admission_side_prepare', 'admission_side_done'}
PUBLIC |= {'admission_record', 'admission_candidates', 'admission_auto_settle'}
ADMIN = {'budget_configure', 'budget_report', 'budget_reconcile', 'budget_overhead'}


def micro_usd(value) -> int:
    try:
        amount = Decimal(str(value))
        if not amount.is_finite() or amount < 0 or amount * 1_000_000 != (amount * 1_000_000).to_integral_value():
            raise ValueError
        return int(amount * 1_000_000)
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError('USD amounts must be non-negative with at most six decimal places.') from None


class AdmissionStore:
    def guest_open(self, session_hash: str, session_ttl_s: float, idle_s: float, old_session_hash: str = '') -> dict:
        """Reuse a valid browser session; guest access alone never reserves or starts compute."""
        from herogen.service.store import new_id
        with self._tx() as c:
            now = self.clock()
            prior = c.execute("SELECT s.user_id FROM sessions s JOIN users u ON u.id=s.user_id "
                "WHERE s.hash=? AND s.revoked_at IS NULL AND s.expires_at>? AND s.last_seen_at>? "
                "AND u.deleted_at IS NULL AND u.status='active' AND u.kind='guest'",
                (old_session_hash, now, now-idle_s)).fetchone()
            uid = prior['user_id'] if prior else new_id('u')
            if not prior:
                c.execute('INSERT INTO users(id,email,created_at,kind) VALUES (?,?,?,?)',
                          (uid, f'workspace:{uid}', now, 'guest'))
            c.execute('INSERT INTO sessions(hash,user_id,created_at,expires_at,last_seen_at) VALUES (?,?,?,?,?)',
                      (session_hash, uid, now, now + session_ttl_s, now))
            if old_session_hash:
                c.execute('UPDATE sessions SET revoked_at=? WHERE hash=?', (now, old_session_hash))
            self._audit(c, uid, 'guest_workspace_opened')
        return self.user_get(uid)

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
        used += c.execute('SELECT COALESCE(SUM(amount_micro),0) FROM budget_overhead').fetchone()[0]
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
        from herogen.service.store import StoreError
        if c.execute('SELECT 1 FROM admissions WHERE user_id=? AND closed=0', (user_id,)).fetchone():
            raise StoreError('This workspace already has an active generation or work awaiting safe reconciliation. No new job was queued.', 'workspace_active')
        status = self._admission_status(c, code_commit, cost_signature)
        if not status['accepting']:
            messages = {'generation_paused': 'Live generation is temporarily paused. No job was queued.',
                        'cost_bound_unverified': 'Live generation is paused while this version’s cost bound is verified. No job was queued.',
                        'generation_budget_exhausted': 'The live-generation budget is currently allocated. Existing results remain available. No job was queued.',
                        'generation_capacity': 'All live-generation slots are occupied. No job was queued; please try again shortly.'}
            raise StoreError(messages[status['reason']], status['reason'])
        reserve = c.execute('SELECT reserve_micro FROM admission_policy WHERE id=1').fetchone()[0]
        # A continuation gets a fresh retry allowance. Reserve its full bound in addition to all prior
        # spending on this job, including a reconciled amount, before reopening its compute receipt.
        c.execute('INSERT INTO admissions(job_id,user_id,client_key,amount_micro,cost_signature,created_at) VALUES (?,?,?,?,?,?) '
                  'ON CONFLICT(job_id) DO UPDATE SET amount_micro=COALESCE(admissions.actual_micro,admissions.amount_micro)+excluded.amount_micro, '
                  'actual_micro=NULL,cost_signature=excluded.cost_signature,main_done=0,side_required=1,side_done=0,closed=0,evidence=\'\'',
                  (job_id, user_id, client_key, reserve, cost_signature, self.clock()))
        self._audit(c, user_id, 'generation_reserved', job_id, {'micro_usd': reserve})

    def admission_work_begin(self, job_id: str, token: str, kind: str, cost_signature: str = '',
                             process: dict | None = None) -> None:
        from herogen.service.store import StoreError
        with self._tx() as c:
            row = c.execute('SELECT closed,cost_signature FROM admissions WHERE job_id=?', (job_id,)).fetchone()
            if row is None or row['closed'] or row['cost_signature'] != cost_signature:
                raise StoreError('This generation has no open compute reservation.', 'generation_paused')
            if kind != 'heldout' and c.execute("SELECT 1 FROM admission_work WHERE job_id=? AND kind!='heldout' AND state!='closed'", (job_id,)).fetchone():
                raise StoreError('Earlier compute has not been confirmed stopped. Operator reconciliation is required.', 'generation_uncertain')
            c.execute('INSERT INTO admission_work(token,job_id,kind,state,created_at,process) VALUES (?,?,?,?,?,?)',
                      (token, job_id, kind, 'open', self.clock(), json.dumps(process) if process else ''))
            if process and process.get('compute_price_per_second'):
                receipt = self._compute_receipt(token, kind, process)
                c.execute('INSERT INTO admission_calls(job_id,token,receipt) VALUES (?,?,?)',
                          (job_id, token, json.dumps(receipt)))

    def _compute_receipt(self, token: str, kind: str, process: dict) -> dict:
        timeout, price = process['compute_timeout_s'], process['compute_price_per_second']
        return {'call_id': token + '-compute', 'model': 'service-' + kind, 'endpoint': 'modal/worker',
                'key': token, 'outcome': 'sent', 'timeout_s': timeout, 'price_per_second': price,
                'price_usd': timeout * price, 'deadline': self.clock() + timeout}

    def admission_record(self, job_id: str, token: str, action: str, **kwargs) -> None:
        """Durably acknowledge a receipt before a subprocess may send paid work."""
        finishing = action == 'process_stopped' or (action == 'call' and kwargs['row'].get('outcome') != 'sent')
        with self._tx() as c:
            work = c.execute("SELECT 1 FROM admission_work w JOIN admissions a USING(job_id) "
                             "WHERE w.job_id=? AND token=? AND (state='open' OR (? AND state='uncertain')) "
                             "AND a.actual_micro IS NULL AND a.closed=0", (job_id, token, finishing)).fetchone()
            if not work:
                raise ValueError('No live reservation for this receipt')
            if action == 'call':
                row = kwargs['row']
                allowed = {'time', 'call_id', 'model', 'endpoint', 'key', 'outcome', 'deadline', 'price_usd',
                           'timeout_s', 'price_per_second', 'attempt', 'status', 'seconds', 'slot_wait_s',
                           'wait_s', 'wait_reason', 'error', 'cost_usd', 'unknown'}
                if set(row) - allowed:
                    raise ValueError('Unsupported call receipt fields')
                c.execute('INSERT INTO admission_calls(job_id,token,receipt) VALUES (?,?,?)',
                          (job_id, token, json.dumps(row, sort_keys=True)))
            elif action == 'process_start':
                c.execute('INSERT INTO admission_processes(token,host,pid) VALUES (?,?,?)',
                          (token, kwargs['host'], kwargs['pid']))
            elif action == 'process_stopped':
                c.execute('UPDATE admission_processes SET stopped=1 WHERE token=? AND host=? AND pid=?',
                          (token, kwargs['host'], kwargs['pid']))
            else:
                raise ValueError('Unknown admission receipt')

    def _admission_snapshot(self, job_id: str) -> dict:
        work = self._all('SELECT * FROM admission_work WHERE job_id=? ORDER BY token', (job_id,))
        processes = self._all('SELECT p.* FROM admission_processes p JOIN admission_work w USING(token) '
                              'WHERE w.job_id=? ORDER BY token,host,pid', (job_id,))
        calls = self._all('SELECT receipt FROM admission_calls WHERE job_id=? ORDER BY id', (job_id,))
        return {'work': work, 'processes': processes, 'calls': [json.loads(r['receipt']) for r in calls]}

    def admission_candidates(self) -> list[dict]:
        with self.lock:
            rows = self._all("SELECT a.job_id,a.created_at FROM admissions a JOIN jobs j ON j.id=a.job_id "
                             "WHERE actual_micro IS NULL AND j.state NOT IN ('queued','running') AND a.main_done=1 "
                             "AND (a.side_required=0 OR a.side_done=1)")
            return [{**r, **self._admission_snapshot(r['job_id'])} for r in rows]

    def admission_auto_settle(self, job_id: str, snapshot: dict) -> bool:
        """Compare the worker's stopped-process evidence atomically with manual reconciliation and new work."""
        from herogen.pipeline.ledger import bound
        cost, calls, full, deadline = bound(snapshot['calls'])
        if not snapshot['work'] or any(not w['process'] for w in snapshot['work']):
            return False
        if not snapshot['calls']:
            deadline = max(w['created_at'] for w in snapshot['work'])
        if self.clock() <= deadline + 60:
            return False
        evidence = f'auto-settled: {calls} calls, {full} charged at full price, ledger {self.path.name}#admission_calls/{job_id}'
        return self.budget_reconcile(job_id, cost, evidence, work_stopped=True, _auto_snapshot=snapshot)

    def _terminal_admission(self, c, job_id: str) -> None:
        job = c.execute('SELECT state,deleted_at FROM jobs WHERE id=?', (job_id,)).fetchone()
        if job and job['state'] in ('succeeded', 'rejected', 'failed', 'canceled', 'stopped'):
            c.execute('UPDATE admissions SET main_done=1 WHERE job_id=?', (job_id,))
            if job['state'] != 'succeeded' or job['deleted_at'] is not None:
                # A queued side worker can no longer claim this job; running paid work retains its receipt.
                c.execute('UPDATE admissions SET side_done=1 WHERE job_id=?', (job_id,))
            self._close_admission(c, job_id)

    def _close_admission(self, c, job_id: str) -> None:
        c.execute('UPDATE admissions SET closed=1 WHERE job_id=? AND main_done=1 AND (side_required=0 OR side_done=1) '
                  'AND NOT EXISTS(SELECT 1 FROM admission_work WHERE job_id=? AND state!=\'closed\')', (job_id, job_id))

    def admission_work_end(self, job_id: str, token: str, certain: bool, worker_stopped: bool = False,
                           compute_seconds: float | None = None) -> None:
        with self._tx() as c:
            c.execute('UPDATE admission_work SET state=?,finished_at=? WHERE token=? AND job_id=? AND state=\'open\'',
                      ('closed' if certain else 'uncertain', self.clock(), token, job_id))
            if worker_stopped:
                row = c.execute('SELECT process FROM admission_work WHERE token=? AND job_id=?', (token, job_id)).fetchone()
                if row and row['process']:
                    process = {**json.loads(row['process']), 'stopped': True}
                    c.execute('UPDATE admission_work SET process=? WHERE token=?', (json.dumps(process), token))
                    if process.get('compute_price_per_second') and compute_seconds is not None:
                        sent = c.execute("SELECT receipt FROM admission_calls WHERE token=? ORDER BY id LIMIT 1", (token,)).fetchone()
                        receipt = {**json.loads(sent['receipt']), 'outcome': 'ok', 'seconds': compute_seconds,
                                   'cost_usd': compute_seconds * process['compute_price_per_second']}
                        c.execute('INSERT INTO admission_calls(job_id,token,receipt) VALUES (?,?,?)',
                                  (job_id, token, json.dumps(receipt)))
            self._close_admission(c, job_id)

    def admission_main_done(self, job_id: str) -> None:
        with self._tx() as c:
            c.execute('UPDATE admissions SET main_done=1 WHERE job_id=?', (job_id,))
            self._close_admission(c, job_id)

    def admission_side_prepare(self, job_id: str) -> None:
        with self._tx() as c:
            c.execute('UPDATE admissions SET side_required=1 WHERE job_id=? AND closed=0', (job_id,))

    def admission_side_done(self, job_id: str, unavailable_message: str = '') -> None:
        with self._tx() as c:
            if unavailable_message:
                job = c.execute('SELECT attempts FROM jobs WHERE id=?', (job_id,)).fetchone()
                if job:
                    c.execute("INSERT INTO job_stages(job_id,stage,state,attempt,message) VALUES (?,'2h','unavailable',?,?) "
                              "ON CONFLICT(job_id,stage) DO NOTHING", (job_id, job['attempts'], unavailable_message[:2000]))
            c.execute('UPDATE admissions SET side_done=1 WHERE job_id=?', (job_id,))
            self._close_admission(c, job_id)

    def budget_overhead(self, amount_micro: int, evidence: str) -> None:
        """Record cumulative non-job charges from provider evidence, never a fabricated invoice estimate."""
        if type(amount_micro) is not int or amount_micro < 0 or not evidence.strip():
            raise ValueError('Non-job spending needs a non-negative amount and evidence.')
        with self._tx() as c:
            c.execute('INSERT INTO budget_overhead VALUES (1,?,?,?) ON CONFLICT(id) DO UPDATE SET '
                      'amount_micro=excluded.amount_micro,evidence=excluded.evidence,updated_at=excluded.updated_at',
                      (amount_micro, evidence[:1000], self.clock()))
            self._audit(c, 'admin', 'budget_overhead_recorded', '', {'amount_micro': amount_micro})

    def budget_report(self) -> dict:
        with self.lock:
            policy = self._one('SELECT * FROM admission_policy WHERE id=1')
            admissions = self._all('SELECT * FROM admissions ORDER BY created_at')
            for row in admissions:
                row['auto_settled'] = row['actual_micro'] is not None and row['evidence'].startswith('auto-settled: ')
            overhead = self._one('SELECT * FROM budget_overhead WHERE id=1')
            committed = sum(a['actual_micro'] if a['actual_micro'] is not None else a['amount_micro'] for a in admissions)
            committed += overhead['amount_micro'] if overhead else 0
            active = [a for a in admissions if not a['closed']]
            failures = self._one("SELECT COUNT(*) AS n FROM jobs WHERE state='failed' AND finished_at>=?", (self.clock()-1800,))['n']
            work = self._all("SELECT job_id,kind,state,created_at,finished_at FROM admission_work WHERE state!='closed'")
            alerts = []
            if policy:
                fraction = committed / policy['limit_micro']
                if fraction >= .9: alerts.append('budget_90_percent')
                elif fraction >= .75: alerts.append('budget_75_percent')
                elif fraction >= .5: alerts.append('budget_50_percent')
                if committed + policy['reserve_micro'] > policy['limit_micro']: alerts.append('budget_unavailable')
                if len(active) >= policy['max_active']: alerts.append('capacity_full')
                if policy['expires_at'] <= self.clock(): alerts.append('policy_expired')
            else: alerts.append('policy_missing')
            if any(w['state'] == 'uncertain' for w in work): alerts.append('work_uncertain')
            if failures >= 2: alerts.append('repeated_failures')
            return {'policy': policy, 'admissions': admissions, 'work': work, 'overhead': overhead,
                    'summary': {'committed_micro': committed, 'active': len(active), 'recent_failures': failures,
                                'oldest_active_seconds': max((self.clock()-a['created_at'] for a in active), default=0),
                                'alerts': alerts}}

    def budget_reconcile(self, job_id: str, actual_micro: int, evidence: str, work_stopped: bool = False,
                         *, _auto_snapshot: dict | None = None) -> bool:
        if not isinstance(actual_micro, int) or actual_micro < 0 or not evidence.strip():
            raise ValueError('Reconciliation requires a non-negative cost and evidence.')
        with self._tx() as c:
            row = c.execute('SELECT * FROM admissions WHERE job_id=?', (job_id,)).fetchone()
            if row is None:
                raise ValueError('No reservation exists for this job.')
            if _auto_snapshot is not None and (row['actual_micro'] is not None
                    or not row['main_done'] or (row['side_required'] and not row['side_done'])
                    or self._admission_snapshot(job_id) != _auto_snapshot):
                return False
            job = c.execute('SELECT state FROM jobs WHERE id=?', (job_id,)).fetchone()
            if job and job['state'] in ('running', 'queued'):
                raise ValueError('Cannot reconcile an active job.')
            outstanding = c.execute("SELECT 1 FROM admission_work WHERE job_id=? AND state!='closed'", (job_id,)).fetchone()
            if (not row['closed'] or outstanding) and not work_stopped:
                raise ValueError('Confirm that all main, side and uncertain remote work has stopped first.')
            if work_stopped:
                c.execute("UPDATE admission_work SET state='closed',finished_at=? WHERE job_id=? AND state!='closed'", (self.clock(), job_id))
            if actual_micro > row['amount_micro']:
                c.execute('UPDATE admission_policy SET enabled=0 WHERE id=1')
            c.execute('UPDATE admissions SET actual_micro=?,closed=1,evidence=? WHERE job_id=?',
                      (actual_micro, evidence[:1000], job_id))
            self._audit(c, 'admin', 'generation_budget_reconciled', job_id,
                        {'actual_micro': actual_micro, 'evidence': evidence[:1000], 'work_stopped': work_stopped})
            if _auto_snapshot is not None:
                self._audit(c, 'service', 'auto_settled', job_id, {'actual_micro': actual_micro, 'evidence': evidence})
        return True
