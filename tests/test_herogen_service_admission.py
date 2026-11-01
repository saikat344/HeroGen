"""Judge isolation, atomic admission and conservative reservation lifecycle; no paid work."""
from __future__ import annotations
import concurrent.futures
import hashlib
import threading
from test_herogen_service_support import Clock, TempDirCase, make_user, settings, png
from herogen.service.store import Store, StoreError, new_id
from herogen.service.admission import micro_usd, ADMIN, PUBLIC
from herogen.service.fakes import FakeRunner
from herogen.service.storage import Storage
from herogen.service.worker import run_job, run_group, InPlaceWorkdir, make_context


class TestAdmission(TempDirCase):
    def setUp(self):
        super().setUp()
        self.clock = Clock()
        self.s = settings(self.tmp, code_commit='a' * 40)
        self.store = Store(self.s.meta_path, clock=self.clock)
        self.seq = 0
        self.configure()

    def tearDown(self):
        self.store.close()
        super().tearDown()

    def configure(self, limit=100_000_000, reserve=10_000_000, capacity=1, **kw):
        args = dict(limit_micro=limit, reserve_micro=reserve, max_active=capacity, expires_at=self.clock()+86400*14,
                    code_commit=self.s.code_commit, cost_signature=self.s.cost_signature, bound_note='Synthetic test bound',
                    provider_caps_verified=True)
        args.update(kw)
        self.store.budget_configure(**args)

    def code(self, allowance=2):
        self.seq += 1
        h = hashlib.sha256(str(self.seq).encode()).hexdigest()
        row = self.store.judge_code_create(h, allowance, self.clock()+86400*14)
        return h, row

    def redeem(self, code, old=''):
        token = hashlib.sha256(new_id('j').encode()).hexdigest()
        return self.store.judge_redeem(code, token, 86400*30, old), token

    def job(self, user, key=None):
        return self.store.job_create(job_id=new_id('j'), user_id=user['id'], client_key=key or new_id('j'),
            kind='sketch', profile='best', jobs_per_day=100, jobs_total=100, storage_max_bytes=10**10,
            input_sha='x', code_commit=self.s.code_commit, cost_signature=self.s.cost_signature)

    def close(self, job):
        self.store.job_cancel(job['id'], job['user_id'])

    def test_private_workspaces_shared_allowance_and_no_fake_agreement(self):
        code, row = self.code()
        a, sa = self.redeem(code); b, sb = self.redeem(code)
        self.assertNotEqual(a['id'], b['id']); self.assertIsNone(a['email'])
        self.assertEqual(self.store._one('SELECT terms_version FROM users WHERE id=?', (a['id'],))['terms_version'], None)
        j, created = self.job(a, 'same-request')
        self.assertTrue(created); self.assertIsNone(self.store.job_get(j['id'], b['id']))
        self.assertEqual(self.store.jobs_list(b['id']), [])
        with self.assertRaises(StoreError) as e: self.job(b)
        self.assertEqual(e.exception.code, 'judge_active')
        with self.assertRaises(StoreError) as e: self.job(b, 'same-request')
        self.assertEqual(e.exception.code, 'conflict')
        self.close(j)
        k, _ = self.job(b); self.close(k)
        c, _ = self.redeem(code)
        with self.assertRaises(StoreError) as e: self.job(c)
        self.assertEqual(e.exception.code, 'judge_allowance')
        self.assertEqual(self.store.judge_info(a['id'])['remaining'], 0)
        self.assertNotIn('code_hash', self.store.judge_codes_list()[0])
        self.assertEqual(self.store.judge_redeem(code, 'c'*64, 3600, sa)['id'], a['id'])
        self.assertIsNone(self.store.session_get(sa, 86400))

    def test_atomic_global_capacity_and_idempotency(self):
        users = [self.redeem(self.code()[0])[0] for _ in range(2)]
        barrier = threading.Barrier(2)
        def create(user):
            barrier.wait()
            try: return self.job(user, 'request')[0]
            except StoreError as e: return e.code
        with concurrent.futures.ThreadPoolExecutor(2) as pool: result = list(pool.map(create, users))
        self.assertEqual(sum(isinstance(x, dict) for x in result), 1)
        self.assertIn('generation_capacity', result)
        j = next(x for x in result if isinstance(x, dict))
        u = next(u for u in users if u['id'] == j['user_id'])
        old, created = self.job(u, 'request')
        self.assertFalse(created); self.assertEqual(old['id'], j['id'])
        self.assertEqual(len(self.store.budget_report()['admissions']), 1)

    def test_cancel_preserves_money_reconcile_does_not_refund_allowance(self):
        self.configure(limit=10_000_000)
        u, _ = self.redeem(self.code()[0]); j, _ = self.job(u); self.close(j)
        self.assertTrue(self.store.budget_report()['admissions'][0]['closed'])
        self.assertEqual(self.store.admission_status(self.s.code_commit, self.s.cost_signature)['reason'], 'generation_budget_exhausted')
        self.store.budget_reconcile(j['id'], 0, 'Synthetic provider evidence: no calls dispatched')
        self.assertTrue(self.store.admission_status(self.s.code_commit, self.s.cost_signature)['accepting'])
        self.assertEqual(self.store.judge_info(u['id'])['remaining'], 1)
        self.store.close(); self.store = Store(self.s.meta_path, clock=self.clock)
        self.assertEqual(self.store.judge_info(u['id'])['remaining'], 1)

    def test_unknown_work_and_side_race_hold_capacity(self):
        u, _ = self.redeem(self.code()[0]); j, _ = self.job(u)
        self.store.admission_work_begin(j['id'], 'main', 'model', self.s.cost_signature)
        self.store.admission_main_done(j['id'])
        self.store.admission_work_end(j['id'], 'main', True)
        self.assertFalse(self.store.budget_report()['admissions'][0]['closed'], 'side may still be dispatched')
        self.store.admission_work_begin(j['id'], 'side', 'heldout', self.s.cost_signature)
        self.store.admission_side_done(j['id'])
        self.store.admission_work_end(j['id'], 'side', False)
        self.assertFalse(self.store.budget_report()['admissions'][0]['closed'])
        self.close(j)
        with self.assertRaises(ValueError): self.store.budget_reconcile(j['id'], 1, 'Uncertain remote work')
        self.store.budget_reconcile(j['id'], 1, 'All remote calls confirmed stopped', work_stopped=True)
        self.assertTrue(self.store.budget_report()['admissions'][0]['closed'])
        with self.assertRaises(StoreError): self.store.admission_work_begin(j['id'], 'late', 'heldout', self.s.cost_signature)

    def test_expiry_revocation_version_and_config_fail_closed(self):
        code, row = self.code(); u, token = self.redeem(code)
        self.assertIsNotNone(self.store.session_get(token, 86400*30))
        self.assertEqual(self.store.admission_status('b'*40, self.s.cost_signature)['reason'], 'cost_bound_unverified')
        self.assertEqual(self.store.admission_status(self.s.code_commit, self.s.with_(max_attempts=3).cost_signature)['reason'], 'cost_bound_unverified')
        self.store.judge_code_revoke(row['id'])
        self.assertIsNone(self.store.session_get(token, 86400*30))
        with self.assertRaises(StoreError): self.job(u)
        code, _ = self.code(); _, token = self.redeem(code)
        self.clock.advance(86400*15)
        self.assertIsNone(self.store.session_get(token, 86400*30))
        self.assertFalse(self.store.admission_status(self.s.code_commit, self.s.cost_signature)['accepting'])

    def test_no_admin_privilege_and_no_missing_mapping_bypass(self):
        self.assertFalse(ADMIN & PUBLIC)
        code, _ = self.code(); u, _ = self.redeem(code)
        self.store.db.execute('DELETE FROM judge_workspaces WHERE user_id=?', (u['id'],))
        with self.assertRaises(StoreError) as e: self.job(u)
        self.assertEqual(e.exception.code, 'judge_code')

    def test_bounds_and_exceeded_reservation_pause_service(self):
        with self.assertRaises(ValueError): self.configure(provider_caps_verified=False)
        with self.assertRaises(ValueError): micro_usd('NaN')
        with self.assertRaises(ValueError): micro_usd('0.0000001')
        self.assertEqual(micro_usd('100'), 100_000_000)
        u, _ = self.redeem(self.code()[0]); j, _ = self.job(u); self.close(j)
        self.store.budget_reconcile(j['id'], 10_000_001, 'Measured provider total exceeds bound')
        self.assertEqual(self.store.admission_status(self.s.code_commit, self.s.cost_signature)['reason'], 'generation_paused')

    def test_worker_success_closes_capacity_but_keeps_reservation(self):
        u, _ = self.redeem(self.code()[0]); j, _ = self.job(u)
        storage = Storage(self.s.data_dir); storage.save_input(u['id'], j['id'], png())
        runner = FakeRunner()
        def execute(job, attempt, group, resume):
            return run_group(self.store, self.s, job, attempt, group, resume, runner,
                             InPlaceWorkdir(make_context(self.s, storage, job)))
        self.assertEqual(run_job(self.store, self.s, storage, j['id'], 'call', execute), 'succeeded')
        reservation = self.store.budget_report()['admissions'][0]
        self.assertTrue(reservation['closed']); self.assertIsNone(reservation['actual_micro'])
        self.assertFalse(self.store.budget_report()['work'])


from test_herogen_service_web import WebCase, Browser


class TestJudgeHTTP(WebCase):
    def build(self, **kw):
        super().build(judge_access=True, code_commit='a'*40, **kw)
        self.store.budget_configure(100_000_000, 10_000_000, 1, self.clock()+86400*14,
            self.s.code_commit, 'Synthetic bound; fake workers only', self.s.cost_signature, provider_caps_verified=True)

    def enter(self, code=None):
        from herogen.service.auth import new_invite_code, normalize_invite
        b = Browser(self.app); b.session()
        if code is None:
            code = new_invite_code()
            self.store.judge_code_create(self.svc.signer.hash('judge-code', normalize_invite(code)), 2, self.clock()+86400*14)
        r = b.post('/api/judge/access', json={'code': code})
        self.assertEqual(r.status_code, 200, r.text)
        b.csrf = r.json()['csrf']
        return b, code

    def test_gate_private_history_csrf_allowance_and_recovery(self):
        b = Browser(self.app); b.session()
        self.assertEqual(self.make_job(b).status_code, 401)
        self.assertEqual(b.post('/api/judge/access', json={'code':'bad'}, csrf=False).status_code, 403)
        a, code = self.enter(); b, _ = self.enter(code)
        self.assertEqual(a.session()['user']['kind'], 'judge')
        self.assertNotIn('code', a.session().get('judge', {}))
        r = self.make_job(a); self.assertEqual(r.status_code, 201, r.text)
        jid = r.json()['id']
        self.assertEqual(self.make_job(a).status_code, 200)
        self.assertEqual(b.get('/api/jobs/'+jid).status_code, 404)
        self.assertEqual(b.get('/api/jobs/'+jid+'/download/glb').status_code, 404)
        self.assertEqual(self.make_job(b, 'different-key').json()['error']['code'], 'judge_active')
        self.assertEqual(len(self.submitted), 1)
        self.store.job_cancel(jid, self.store.job_get(jid)['user_id'])
        r = self.make_job(b, 'different-key'); self.assertEqual(r.status_code, 201, r.text)
        self.store.job_cancel(r.json()['id'], self.store.job_get(r.json()['id'])['user_id'])
        self.assertEqual(self.make_job(a, 'third-request').status_code, 429)
        self.assertEqual(a.session()['judge']['remaining'], 0)

    def test_capacity_has_retry_header_and_no_dispatch_or_credit_consumption(self):
        a, _ = self.enter(); b, _ = self.enter()
        self.assertEqual(self.make_job(a).status_code, 201)
        r = self.make_job(b)
        self.assertEqual(r.status_code, 503); self.assertEqual(r.headers['retry-after'], '5')
        self.assertEqual(r.json()['error']['code'], 'generation_capacity')
        self.assertEqual(b.session()['judge']['remaining'], 2)
        self.assertEqual(len(self.submitted), 1)

    def test_paused_budget_and_admin_path(self):
        a, _ = self.enter()
        self.store.db.execute('UPDATE admission_policy SET enabled=0')
        r = self.make_job(a)
        self.assertEqual(r.status_code, 503); self.assertEqual(r.json()['error']['code'], 'generation_paused')
        self.assertEqual(len(self.submitted), 0)
        self.assertEqual(a.get('/admin/signin').status_code, 200)


class TestPublicAccess(WebCase):
    def build(self, **kw):
        super().build(public_demo=True, code_commit='a'*40, **kw)
        self.store.budget_configure(100_000_000, 10_000_000, 1, self.clock()+86400*14,
            self.s.code_commit, 'Synthetic bound', self.s.cost_signature, provider_caps_verified=True)

    def test_public_browser_allowance_reuse_and_global_budget(self):
        a, b = Browser(self.app), Browser(self.app)
        for browser in (a, b):
            browser.session()
            r = browser.post('/api/demo/access', json={})
            self.assertEqual(r.status_code, 200, r.text)
            browser.csrf = r.json()['csrf']
        self.assertEqual(len(self.store.judge_codes_list()), 2)
        r = self.make_job(a); self.assertEqual(r.status_code, 201, r.text)
        jid = r.json()['id']
        self.assertEqual(b.get('/api/jobs/'+jid).status_code, 404)
        self.assertEqual(self.make_job(b).json()['error']['code'], 'generation_capacity')
        a.session(); r = a.post('/api/demo/access', json={})
        self.assertEqual(r.status_code, 200); a.csrf = r.json()['csrf']
        self.assertEqual(len(self.store.judge_codes_list()), 2, 're-entry preserves allowance and history')
        self.store.job_cancel(jid, self.store.job_get(jid)['user_id'])
        self.assertEqual(self.make_job(a, 'new-attempt').json()['error']['code'], 'judge_allowance')
        self.assertEqual(self.make_job(b).status_code, 201)

    def test_unfunded_gate_is_explicit_and_csrf_required(self):
        a = Browser(self.app); a.session()
        self.assertEqual(a.post('/api/demo/access', json={}, csrf=False).status_code, 403)
        self.store.db.execute('UPDATE admission_policy SET enabled=0')
        r = a.post('/api/demo/access', json={})
        self.assertEqual(r.status_code, 503); self.assertEqual(r.json()['error']['code'], 'generation_paused')
        self.assertEqual(self.store.judge_codes_list(), [])
        self.assertEqual(self.submitted, [])
