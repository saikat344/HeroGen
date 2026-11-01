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

class TestGuestHTTP(WebCase):
    def build(self,**kw):
        super().build(public_demo=True,code_commit='a'*40,max_running_jobs=5,**kw)
        self.store.budget_configure(1_000_000_000,10_000_000,5,self.clock()+86400*14,
            self.s.code_commit,'Synthetic bound',self.s.cost_signature,provider_caps_verified=True)
    def enter(self):
        b=Browser(self.app); b.session(); r=b.post('/api/demo/access',json={})
        self.assertEqual(r.status_code,200,r.text); b.csrf=r.json()['csrf']; return b
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
