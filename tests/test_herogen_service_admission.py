"""Private guest sessions, concurrent budget admission and conservative work accounting; no paid calls."""
from __future__ import annotations
import concurrent.futures
import secrets
import threading
from test_herogen_service_support import Clock, TempDirCase, settings, png
from herogen.service.store import Store, StoreError, new_id
from herogen.service.admission import micro_usd, ADMIN, PUBLIC
from herogen.service.fakes import FakeRunner
from herogen.service.storage import Storage
from herogen.service.worker import run_job, run_group, InPlaceWorkdir, make_context
from test_herogen_service_web import WebCase, Browser

class TestAdmission(TempDirCase):
    def setUp(self):
        super().setUp(); self.clock=Clock(); self.s=settings(self.tmp, code_commit='a'*40)
        self.store=Store(self.s.meta_path, clock=self.clock); self.configure()
    def tearDown(self):
        self.store.close(); super().tearDown()
    def configure(self, **kw):
        args=dict(limit_micro=1_000_000_000, reserve_micro=10_000_000, max_active=5,
                  expires_at=self.clock()+86400*14, code_commit=self.s.code_commit,
                  cost_signature=self.s.cost_signature, bound_note='Synthetic test bound', provider_caps_verified=True)
        args.update(kw); self.store.budget_configure(**args)
    def guest(self, old=''):
        token=secrets.token_hex(32)
        return self.store.guest_open(token, 86400*30, 86400*7, old), token
    def job(self, user, key=None):
        return self.store.job_create(job_id=new_id('j'), user_id=user['id'], client_key=key or secrets.token_hex(8),
            kind='sketch', profile='best', jobs_per_day=1, jobs_total=1, storage_max_bytes=10**10,
            input_sha='x', code_commit=self.s.code_commit, cost_signature=self.s.cost_signature)
    def close(self, job): self.store.job_cancel(job['id'],job['user_id'])
    def status(self): return self.store.admission_status(self.s.code_commit,self.s.cost_signature)

    def test_private_history_and_session_rotation(self):
        a, token=self.guest(); b,_=self.guest(); j,_=self.job(a)
        self.assertEqual(a['kind'],'guest'); self.assertIsNone(a['email'])
        self.assertIsNone(self.store.job_get(j['id'],b['id'])); self.assertEqual(self.store.jobs_list(b['id']),[])
        self.assertIsNone(self.store._one('SELECT terms_version FROM users WHERE id=?',(a['id'],))['terms_version'])
        again,_=self.guest(token); self.assertEqual(again['id'],a['id'])
        self.assertIsNone(self.store.session_get(token,86400))
        self.assertFalse(ADMIN & PUBLIC); self.assertFalse(any('judge' in method for method in PUBLIC | ADMIN))

    def test_five_global_slots_atomic_and_one_per_workspace(self):
        users=[self.guest()[0] for _ in range(6)]; barrier=threading.Barrier(6)
        def create(u):
            barrier.wait()
            try: return self.job(u,'same-key')[0]
            except StoreError as e: return e.code
        with concurrent.futures.ThreadPoolExecutor(6) as pool: results=list(pool.map(create,users))
        self.assertEqual(sum(isinstance(x,dict) for x in results),5); self.assertIn('generation_capacity',results)
        j=next(x for x in results if isinstance(x,dict)); u=next(u for u in users if u['id']==j['user_id'])
        old,created=self.job(u,'same-key'); self.assertFalse(created); self.assertEqual(old['id'],j['id'])
        with self.assertRaises(StoreError) as e: self.job(u)
        self.assertEqual(e.exception.code,'workspace_active')
        self.assertEqual(len(self.store.budget_report()['admissions']),5)

    def test_guest_can_generate_again_after_finishing_without_beta_lifetime_limit(self):
        u,_=self.guest()
        for _ in range(3): j,_=self.job(u); self.close(j)
        self.assertEqual(len(self.store.jobs_list(u['id'])),3)

    def test_reservations_and_non_job_charges_enforce_total(self):
        self.configure(limit_micro=10_000_000)
        u,_=self.guest(); j,_=self.job(u); self.close(j)
        self.assertEqual(self.status()['reason'],'generation_budget_exhausted')
        self.store.budget_reconcile(j['id'],0,'Synthetic evidence: no paid calls')
        self.assertTrue(self.status()['accepting'])
        self.store.budget_overhead(1,'Synthetic hosting invoice')
        self.assertEqual(self.status()['reason'],'generation_budget_exhausted')
        self.store.close(); self.store=Store(self.s.meta_path,clock=self.clock)
        self.assertEqual(self.store.budget_report()['summary']['committed_micro'],1)

    def test_side_race_uncertain_work_and_safe_reconciliation(self):
        u,_=self.guest(); j,_=self.job(u)
        self.store.admission_work_begin(j['id'],'main','model',self.s.cost_signature)
        self.store.admission_main_done(j['id']); self.store.admission_work_end(j['id'],'main',True)
        self.assertFalse(self.store.budget_report()['admissions'][0]['closed'])
        self.store.admission_work_begin(j['id'],'side','heldout',self.s.cost_signature)
        self.store.admission_side_done(j['id']); self.store.admission_work_end(j['id'],'side',False); self.close(j)
        self.assertIn('work_uncertain',self.store.budget_report()['summary']['alerts'])
        with self.assertRaises(ValueError): self.store.budget_reconcile(j['id'],1,'Uncertain work')
        self.store.budget_reconcile(j['id'],1,'All remote work confirmed stopped',work_stopped=True)
        with self.assertRaises(StoreError): self.store.admission_work_begin(j['id'],'late','heldout',self.s.cost_signature)

    def test_changed_version_settings_expiry_and_exceeded_bound_pause(self):
        self.assertEqual(self.store.admission_status('b'*40,self.s.cost_signature)['reason'],'cost_bound_unverified')
        self.assertEqual(self.store.admission_status(self.s.code_commit,self.s.with_(max_attempts=3).cost_signature)['reason'],'cost_bound_unverified')
        with self.assertRaises(ValueError): self.configure(provider_caps_verified=False)
        for value in ('NaN','0.0000001'):
            with self.assertRaises(ValueError): micro_usd(value)
        self.assertEqual(micro_usd('1000'),1_000_000_000)
        u,_=self.guest(); j,_=self.job(u); self.close(j)
        self.store.budget_reconcile(j['id'],10_000_001,'Observed total exceeds bound')
        self.assertEqual(self.status()['reason'],'generation_paused')
        self.configure(); self.clock.advance(86400*15)
        self.assertEqual(self.status()['reason'],'generation_paused')

    def test_changed_refine_cap_unverifies_bound_and_refuses_workers(self):
        bumped=self.s.with_(refine_cap_usd=80)
        self.assertNotEqual(self.s.cost_signature,bumped.cost_signature)
        self.assertEqual(self.store.admission_status(self.s.code_commit,bumped.cost_signature)['reason'],'cost_bound_unverified')
        u,_=self.guest(); j,_=self.job(u)
        with self.assertRaises(StoreError) as e: self.store.admission_work_begin(j['id'],'cap-changed','model',bumped.cost_signature)
        self.assertEqual(e.exception.code,'generation_paused')

    def test_alerts_use_allocations_and_recorded_overhead(self):
        self.store.budget_overhead(500_000_000,'Synthetic non-job charges')
        self.assertIn('budget_50_percent',self.store.budget_report()['summary']['alerts'])
        self.store.budget_overhead(750_000_000,'Synthetic non-job charges')
        self.assertIn('budget_75_percent',self.store.budget_report()['summary']['alerts'])
        self.store.budget_overhead(900_000_000,'Synthetic non-job charges')
        self.assertIn('budget_90_percent',self.store.budget_report()['summary']['alerts'])

    def test_success_releases_capacity_but_retains_money(self):
        u,_=self.guest(); j,_=self.job(u); storage=Storage(self.s.data_dir); storage.save_input(u['id'],j['id'],png())
        runner=FakeRunner()
        def execute(job,attempt,group,resume):
            return run_group(self.store,self.s,job,attempt,group,resume,runner,InPlaceWorkdir(make_context(self.s,storage,job)))
        self.assertEqual(run_job(self.store,self.s,storage,j['id'],'call',execute),'succeeded')
        row=self.store.budget_report()['admissions'][0]
        self.assertTrue(row['closed']); self.assertIsNone(row['actual_micro'])

    def execute_job(self, user, runner, heldout):
        j,_=self.job(user); storage=Storage(self.s.data_dir); storage.save_input(user['id'],j['id'],png())
        def execute(job,attempt,group,resume):
            return run_group(self.store,self.s,job,attempt,group,resume,runner,InPlaceWorkdir(make_context(self.s,storage,job)))
        result=run_job(self.store,self.s,storage,j['id'],'call',execute,heldout=heldout)
        return j,result

    def test_rejected_deleted_or_unspawned_side_checks_release_certain_capacity(self):
        u,_=self.guest(); j,result=self.execute_job(u,FakeRunner(fail={'3':'reject'}),lambda job:None)
        self.assertEqual(result,'rejected'); self.assertTrue(self.store.budget_report()['admissions'][0]['closed'])
        j,result=self.execute_job(u,FakeRunner(),lambda job:None)
        self.assertEqual(result,'succeeded'); self.assertFalse(self.store.budget_report()['admissions'][-1]['closed'])
        self.store.job_delete(j['id'],u['id']); self.assertTrue(self.store.budget_report()['admissions'][-1]['closed'])
        def failed_spawn(job): raise RuntimeError('Synthetic spawn failure')
        j,result=self.execute_job(u,FakeRunner(),failed_spawn)
        self.assertEqual(result,'succeeded'); self.assertTrue(self.store.budget_report()['admissions'][-1]['closed'])
        side=next(stage for stage in self.store.job_stages(j['id']) if stage['stage']=='2h')
        self.assertEqual(side['state'],'unavailable')

    def test_crashed_budgeted_worker_stops_retry_and_alert_clears_after_confirmation(self):
        u,_=self.guest(); j,_=self.job(u)
        self.store.job_claim(j['id'],'lost',60,2)
        self.store.admission_work_begin(j['id'],'lost-work','model',self.s.cost_signature)
        self.clock.advance(120); self.store.reconcile_stale(2)
        self.assertEqual(self.store.job_get(j['id'])['state'],'failed')
        self.assertEqual(self.store.job_get(j['id'])['error_code'],'generation_uncertain')
        self.assertIn('work_uncertain',self.store.budget_report()['summary']['alerts'])
        self.assertIsNone(self.store.job_claim(j['id'],'retry',60,2))
        self.store.budget_reconcile(j['id'],0,'All remote work confirmed stopped',work_stopped=True)
        self.assertEqual(self.store.budget_report()['work'],[])
        self.assertNotIn('work_uncertain',self.store.budget_report()['summary']['alerts'])

    def test_side_lease_loss_is_uncertain_without_affecting_main_job(self):
        u,_=self.guest(); j,_=self.job(u); self.store.job_claim(j['id'],'main',1000,2)
        self.assertTrue(self.store.side_claim(j['id'],'2h','side',30))
        self.store.admission_work_begin(j['id'],'side-work','heldout',self.s.cost_signature)
        self.clock.advance(60); self.store.reconcile_stale(2)
        self.assertEqual(self.store.job_get(j['id'])['state'],'running')
        self.assertIn('work_uncertain',self.store.budget_report()['summary']['alerts'])
        self.assertFalse(self.store.budget_report()['admissions'][0]['closed'])

    def test_account_deletion_closes_unstarted_reservation_without_refunding_money(self):
        u,_=self.guest(); j,_=self.job(u); self.store.user_delete(u['id'])
        row=self.store.budget_report()['admissions'][0]
        self.assertTrue(row['closed']); self.assertIsNone(row['actual_micro'])


class TestGuestHTTP(WebCase):
    def build(self,**kw):
        super().build(public_demo=True,code_commit='a'*40,max_running_jobs=5,**kw)
        self.store.budget_configure(1_000_000_000,10_000_000,5,self.clock()+86400*14,
            self.s.code_commit,'Synthetic bound',self.s.cost_signature,provider_caps_verified=True)
    def enter(self):
        b=Browser(self.app); b.session(); r=b.post('/api/demo/access',json={})
        self.assertEqual(r.status_code,200,r.text); b.csrf=r.json()['csrf']; return b

    def redo(self, browser, parent, stage='cover', key='redo-key-0001'):
        return browser.post(f'/api/jobs/{parent}/redo', json={'stage': stage, 'note': 'green coat'},
                            headers={'Idempotency-Key': key})

    def reservation(self, jid):
        return next(row for row in self.store.budget_report()['admissions'] if row['job_id'] == jid)

    def test_guest_redo_and_continue_reserve_budget_without_consuming_another_quota_slot(self):
        for stage in ('cover', 'turnaround'):
            with self.subTest(stage=stage):
                b = self.enter()
                parent = self.make_job(b).json()['id']
                self.assertEqual(self.finish(parent), 'succeeded')
                before = self.store.budget_report()['summary']['committed_micro']
                r = self.redo(b, parent, stage)
                self.assertEqual(r.status_code, 201, r.text)
                child = r.json()['id']
                self.assertEqual(self.store.job_get(child)['budget_required'], 1)
                self.assertEqual(self.reservation(child)['amount_micro'], 10_000_000)
                self.assertFalse(self.reservation(child)['closed'])
                self.assertEqual(self.store.budget_report()['summary']['committed_micro'], before + 10_000_000)
                self.assertEqual(self.redo(b, parent, stage).status_code, 200)
                self.assertEqual(self.finish(child), 'stopped')
                self.assertTrue(self.reservation(child)['closed'])
                self.assertEqual(self.store.budget_report()['summary']['active'], 0)
                quota = b.get('/api/quota').json()['jobs_total']
                for _ in range(2):
                    r = b.post(f'/api/jobs/{child}/continue')
                    self.assertEqual(r.status_code, 200, r.text)
                    self.assertFalse(self.reservation(child)['closed'])
                    self.assertEqual(self.reservation(child)['amount_micro'], 20_000_000)
                self.assertEqual(b.get('/api/quota').json()['jobs_total'], quota)
                self.assertEqual(self.finish(child), 'succeeded')
                self.assertTrue(self.reservation(child)['closed'])

    def test_stopped_redo_finalizer_cannot_close_queued_continuation(self):
        self.assert_stale_redo_finalizer('cover', duplicate=False, mid_stage=False)

    def test_duplicate_redo_finalizer_cannot_close_running_continuation(self):
        self.assert_stale_redo_finalizer('cover', duplicate=True, mid_stage=True)

    def test_stopped_turnaround_finalizer_cannot_close_running_continuation(self):
        self.assert_stale_redo_finalizer('turnaround', duplicate=False, mid_stage=True)

    def test_duplicate_turnaround_finalizer_cannot_close_queued_continuation(self):
        self.assert_stale_redo_finalizer('turnaround', duplicate=True, mid_stage=False)

    def assert_stale_redo_finalizer(self, stage, *, duplicate, mid_stage):
        from unittest.mock import patch
        from herogen.service.worker import run_heldout

        self.store.db.execute('UPDATE admission_policy SET max_active=1')
        b = self.enter()
        parent = self.make_job(b).json()['id']
        self.assertEqual(self.finish(parent), 'succeeded')
        child = self.redo(b, parent, stage).json()['id']
        runner = FakeRunner()

        def execute(job, attempt, group, resume):
            return run_group(self.store, self.s, job, attempt, group, resume, runner,
                             InPlaceWorkdir(make_context(self.s, self.svc.storage, job)))

        def heldout(job):
            return run_heldout(self.store, self.s, job, runner,
                lambda claim: InPlaceWorkdir(make_context(self.s, self.svc.storage, job)))

        def run():
            return run_job(self.store, self.s, self.svc.storage, child, 'race', execute,
                           heldout=heldout)

        if duplicate:
            self.assertEqual(run(), 'stopped')
        paused, release = threading.Event(), threading.Event()
        original_stages = self.store.job_stages

        def pause_finalizer(jid):
            rows = original_stages(jid)
            if jid == child and self.store.job_get(jid)['state'] == 'stopped':
                paused.set()
                if not release.wait(10):
                    raise TimeoutError('Stopped finalizer was not released')
            return rows

        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            with patch.object(self.store, 'job_stages', pause_finalizer):
                stale = pool.submit(run)
                try:
                    self.assertTrue(paused.wait(10), 'Finalizer did not observe stopped redo')
                finally:
                    if not paused.is_set():
                        release.set()
            try:
                self.assertTrue(self.reservation(child)['closed'])
                old_attempt = self.store.job_get(child)['attempts']
                r = b.post(f'/api/jobs/{child}/continue')
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()['state'], 'queued')
                # A queued continuation has not claimed another attempt yet.
                self.assertEqual(self.store.job_get(child)['attempts'], old_attempt)

                def release_and_check():
                    release.set()
                    self.assertEqual(stale.result(timeout=10), None if duplicate else 'stopped')
                    row = self.reservation(child)
                    side_done = int(mid_stage and stage == 'turnaround')
                    self.assertEqual((row['main_done'], row['side_done'], row['closed']), (0, side_done, 0))
                    self.assertEqual(row['amount_micro'], 20_000_000)
                    self.assertIsNone(row['actual_micro'])
                    self.assertEqual(self.store.budget_report()['summary']['active'], 1)
                    other = self.make_job(self.enter())
                    self.assertEqual(other.status_code, 503, other.text)
                    self.assertEqual(other.json()['error']['code'], 'generation_capacity')

                original_run = runner.run
                def release_during_stage(spec, ctx):
                    if spec.key == ('2' if stage == 'cover' else '3'):
                        self.assertTrue(self.store.budget_report()['work'])
                        release_and_check()
                    return original_run(spec, ctx)

                if mid_stage:
                    with patch.object(runner, 'run', release_during_stage):
                        result = run()
                else:
                    release_and_check()
                    result = run()
                self.assertEqual(result, 'succeeded')
                rows = {row['stage']: row for row in self.store.job_stages(child)}
                self.assertEqual(rows['2h']['state'], 'passed')
                self.assertAlmostEqual(rows['2']['cost_usd'], .025)
                self.assertEqual(self.store.job_get(child)['counts_quota'], 1)
                self.assertTrue(self.reservation(child)['closed'])
                self.assertEqual(self.reservation(child)['amount_micro'], 20_000_000)
                self.assertEqual(self.store.budget_report()['work'], [])
            finally:
                release.set()
                stale.result(timeout=10)

    def test_guest_redo_and_continue_refused_by_pause_budget_and_cost_verification(self):
        b = self.enter()
        parent = self.make_job(b).json()['id']
        self.assertEqual(self.finish(parent), 'succeeded')
        child = self.redo(b, parent).json()['id']
        self.assertEqual(self.finish(child), 'stopped')
        committed = self.store.budget_report()['summary']['committed_micro']
        uid = self.store.job_get(child)['user_id']
        for reason, policy in (
            ('generation_paused', {'enabled': False}),
            ('generation_budget_exhausted', {'limit_micro': committed}),
            ('cost_bound_unverified', {'cost_signature': 'b' * 64}),
        ):
            policy_args = dict(limit_micro=1_000_000_000, reserve_micro=10_000_000, max_active=5,
                               expires_at=self.clock() + 86400, code_commit=self.s.code_commit,
                               cost_signature=self.s.cost_signature, bound_note='Synthetic bound',
                               provider_caps_verified=True)
            policy_args.update(policy)
            self.store.budget_configure(**policy_args)
            submitted = list(self.submitted)
            for action in ('redo', 'continue'):
                with self.subTest(reason=reason, action=action):
                    r = (self.redo(b, parent, key='redo-key-0002') if action == 'redo'
                         else b.post(f'/api/jobs/{child}/continue'))
                    self.assertEqual(r.status_code, 503, r.text)
                    self.assertEqual(r.json()['error']['code'], reason)
                    self.assertEqual(self.store.job_get(child)['state'], 'stopped')
                    self.assertEqual(self.store.budget_report()['summary']['committed_micro'], committed)
                    self.assertEqual(len(self.store.jobs_list(uid)), 2)
                    self.assertEqual(len(list(self.svc.storage.runs_root(uid).iterdir())), 2)
                    self.assertEqual(self.submitted, submitted)

    def test_public_account_redo_and_continue_use_admission(self):
        b = self.signin()
        parent = self.make_job(b).json()['id']
        self.assertEqual(self.finish(parent), 'succeeded')
        child = self.redo(b, parent).json()['id']
        self.assertEqual(self.store.job_get(child)['budget_required'], 1)
        self.assertEqual(self.finish(child), 'stopped')
        self.store.db.execute('UPDATE admission_policy SET enabled=0')
        for r in (self.redo(b, parent, key='redo-key-0002'), b.post(f'/api/jobs/{child}/continue')):
            self.assertEqual(r.status_code, 503, r.text)
            self.assertEqual(r.json()['error']['code'], 'generation_paused')

    def test_guest_redo_and_continue_respect_global_capacity(self):
        b = self.enter()
        parent = self.make_job(b).json()['id']
        self.assertEqual(self.finish(parent), 'succeeded')
        child = self.redo(b, parent).json()['id']
        self.assertEqual(self.finish(child), 'stopped')
        self.store.db.execute('UPDATE admission_policy SET max_active=1')
        self.assertEqual(self.make_job(self.enter()).status_code, 201)
        for r in (self.redo(b, parent, key='redo-key-0002'), b.post(f'/api/jobs/{child}/continue')):
            self.assertEqual(r.status_code, 503, r.text)
            self.assertEqual(r.json()['error']['code'], 'generation_capacity')
            self.assertEqual(r.headers['retry-after'], '5')
        self.assertEqual(self.store.job_get(child)['state'], 'stopped')
        self.assertTrue(self.reservation(child)['closed'])

    def test_guest_continuation_retains_reconciled_prior_spend(self):
        b = self.enter()
        parent = self.make_job(b).json()['id']
        self.assertEqual(self.finish(parent), 'succeeded')
        child = self.redo(b, parent).json()['id']
        self.assertEqual(self.finish(child), 'stopped')
        self.store.budget_reconcile(child, 123, 'Synthetic receipt: no paid calls')
        r = b.post(f'/api/jobs/{child}/continue')
        self.assertEqual(r.status_code, 200, r.text)
        row = self.reservation(child)
        self.assertEqual(row['amount_micro'], 10_000_123)
        self.assertIsNone(row['actual_micro'])
        self.assertEqual(self.store.budget_report()['summary']['committed_micro'], 20_000_123)

    def test_guest_continuation_cannot_reopen_uncertain_compute(self):
        b = self.enter()
        parent = self.make_job(b).json()['id']
        self.assertEqual(self.finish(parent), 'succeeded')
        child = self.redo(b, parent).json()['id']
        claim = self.store.job_claim(child, 'synthetic', 60, 2)
        self.store.admission_work_begin(child, 'uncertain', 'model', self.s.cost_signature)
        self.store.job_finish(child, claim['attempt'], 'stopped')
        self.store.admission_work_end(child, 'uncertain', False)
        r = b.post(f'/api/jobs/{child}/continue')
        self.assertEqual(r.status_code, 409, r.text)
        self.assertEqual(r.json()['error']['code'], 'workspace_active')
        self.assertEqual(self.store.job_get(child)['state'], 'stopped')
        self.assertEqual(self.reservation(child)['amount_micro'], 10_000_000)
    def test_guest_entry_is_private_csrf_protected_and_not_compute(self):
        b=Browser(self.app); b.session()
        self.assertEqual(b.post('/api/demo/access',json={},csrf=False).status_code,403)
        a,b=self.enter(),self.enter(); self.assertEqual(a.session()['user']['kind'],'guest')
        self.assertEqual(self.submitted,[])
        r=self.make_job(a); self.assertEqual(r.status_code,201,r.text); jid=r.json()['id']
        self.assertEqual(self.make_job(a).status_code,200)
        self.assertEqual(self.make_job(a,'another-key').json()['error']['code'],'workspace_active')
        self.assertEqual(b.get('/api/jobs/'+jid).status_code,404)
        self.assertEqual(b.get('/api/jobs/'+jid+'/download/glb').status_code,404)
        a.session(); r=a.post('/api/demo/access',json={}); a.csrf=r.json()['csrf']
        self.assertEqual(len(self.store.users_list()),2); self.assertEqual(a.get('/api/jobs/'+jid).status_code,200)
        self.assertEqual(a.post('/api/judge/access',json={}).status_code,404)
        self.assertEqual(a.get('/admin/signin').status_code,200)
    def test_sixth_request_is_not_queued_and_unfunded_generation_is_explicit(self):
        browsers=[self.enter() for _ in range(6)]
        for b in browsers[:5]: self.assertEqual(self.make_job(b).status_code,201)
        r=self.make_job(browsers[5]); self.assertEqual(r.status_code,503)
        self.assertEqual(r.headers['retry-after'],'5'); self.assertEqual(len(self.submitted),5)
        self.store.db.execute('UPDATE admission_policy SET enabled=0')
        r=self.make_job(browsers[5]); self.assertEqual(r.json()['error']['code'],'generation_paused')
