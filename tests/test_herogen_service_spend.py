"""Transactional generation settings, terminal spend and guest-only HTTP access."""
import concurrent.futures
import json
import threading
from pathlib import Path

from test_herogen_service_support import Clock, TempDirCase, make_user
from test_herogen_service_web import WebCase, Browser
from herogen.service.store import Store, StoreError, new_id
from herogen.service.admin import run
from herogen.service.auth import Signer


class SpendGuardTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.clock = Clock()
        self.store = Store(self.tmp / 'guard.db', clock=self.clock, max_running=2)
        self.addCleanup(self.store.close)
        self.user = make_user(self.store)
        self.store.generation_update(actor=self.user['id'], limit_usd=3)

    def create(self, key=None, store=None, commit='a'*40):
        return (store or self.store).job_create(job_id=new_id('j'), user_id=self.user['id'],
            client_key=key or new_id('j'), kind='prompt', profile='best', prompt='A knight',
            storage_max_bytes=100000, code_commit=commit)

    def test_exact_refusals_and_running_estimate(self):
        self.store.generation_update(actor=self.user['id'], enabled=False)
        with self.assertRaisesRegex(StoreError, '^Live generation is paused right now\\.$'):
            self.create()
        self.store.generation_update(actor=self.user['id'], enabled=True)
        self.create(); self.create()
        status = self.store.generation_get()
        self.assertEqual((status['spent_usd'], status['running_usd'], status['remaining_usd']), (0, 3, 0))
        with self.assertRaisesRegex(StoreError, '^The live-generation budget is used up for now\\.$'):
            self.create()

    def test_every_final_state_releases_estimate_automatically(self):
        for state in ('succeeded', 'failed', 'rejected', 'canceled', 'stopped', 'crashed'):
            with self.subTest(state=state):
                job, _ = self.create()
                claim = self.store.job_claim(job['id'], 'call', 10, 2)
                self.store.job_stage_set(job['id'], claim['attempt'], '1', 'passed', cost_usd=.1)
                if state == 'canceled':
                    self.store.job_cancel(job['id'], self.user['id'])
                elif state == 'crashed':
                    self.clock.advance(11)
                    self.assertEqual(self.store.reconcile_stale(2), [(job['id'], 'failed')])
                else:
                    self.store.job_finish(job['id'], claim['attempt'], state, cost_usd=.1)
                status = self.store.generation_get()
                self.assertEqual(status['running_usd'], 0)
                self.assertAlmostEqual(status['spent_usd'], .1 * (('succeeded', 'failed', 'rejected', 'canceled', 'stopped', 'crashed').index(state)+1))
                next_job, _ = self.create()
                self.store.job_cancel(next_job['id'], self.user['id'])

    def test_expired_lease_releases_estimate_before_sweep(self):
        job, _ = self.create()
        self.store.job_claim(job['id'], 'call', 10, 2)
        self.store.job_stage_set(job['id'], 1, '1', 'running', cost_usd=.2)
        self.clock.advance(10)
        self.assertEqual(self.store.generation_get()['running_usd'], 0)
        self.assertEqual(self.store.generation_get()['spent_usd'], .2)
        self.assertEqual(self.store.job_get(job['id'])['state'], 'failed')
        self.create()

    def test_restart_changed_commit_and_years_elapsed_keep_enabled(self):
        self.store.close()
        self.store = Store(self.tmp / 'guard.db', clock=self.clock)
        self.clock.advance(20 * 365 * 86400)
        self.assertTrue(self.store.generation_get()['enabled'])
        self.create(commit='b'*40)

    def test_concurrent_connections_cannot_exceed_total_guard(self):
        stores = [Store(self.tmp / 'guard.db', clock=self.clock) for _ in range(12)]
        self.addCleanup(lambda: [s.close() for s in stores])
        barrier = threading.Barrier(len(stores))
        def start(store):
            barrier.wait()
            try:
                return self.create(store=store)[1]
            except StoreError as exc:
                self.assertEqual(exc.code, 'generation_budget_exhausted')
                return False
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(stores)) as pool:
            results = list(pool.map(start, stores))
        self.assertEqual(sum(results), 2)
        self.assertEqual(self.store.generation_get()['running_usd'], 3)

    def test_dispatch_order_and_atomic_concurrency(self):
        self.store.generation_update(actor=self.user['id'], limit_usd=100)
        jobs = [self.create()[0] for _ in range(5)]
        first = self.store.jobs_to_dispatch(99)
        self.assertEqual([j['id'] for j in first], [j['id'] for j in jobs[:2]])
        self.assertEqual(self.store.jobs_to_dispatch(99), [])
        self.store.job_claim(first[0]['id'], 'call', 60, 2)
        self.store.job_finish(first[0]['id'], 1, 'failed', cost_usd=.1)
        self.assertEqual([j['id'] for j in self.store.jobs_to_dispatch(99)], [jobs[2]['id']])

    def test_model_errors_remain_failed_stages_and_jobs(self):
        from herogen.pipeline.client import ApiError, ReplayMiss
        from herogen.service.storage import Storage
        from herogen.service.worker import InPlaceWorkdir, make_context, run_group, run_job
        from test_herogen_service_support import settings
        configured = settings(self.tmp)
        storage = Storage(configured.data_dir)
        for error in (ApiError('synthetic model failure'), ReplayMiss('synthetic cache miss')):
            class BrokenRunner:
                def run(self, spec, ctx):
                    raise error
            job, _ = self.create()
            def execute(job, attempt, group, resume):
                return run_group(self.store, configured, job, attempt, group, resume, BrokenRunner(),
                                 InPlaceWorkdir(make_context(configured, storage, job)))
            self.assertEqual(run_job(self.store, configured, storage, job['id'], 'call', execute), 'failed')
            self.assertEqual(self.store.job_stages(job['id'])[0]['state'], 'failed')
            self.assertEqual(self.store.job_get(job['id'])['error_code'], 'stage_error')

    def test_cli_partial_updates_and_audit_actor(self):
        signer = Signer(b'k'*48)
        result = json.loads(run('generation-settings --pause --limit-usd 12 --running-estimate-usd 2 --max-running 3', self.store, signer))
        self.assertFalse(result['enabled'])
        self.assertEqual((result['limit_usd'], result['running_estimate_usd'], result['max_running']), (12,2,3))
        self.assertEqual(json.loads(run('generation-settings', self.store, signer)), result)
        self.store.generation_update(actor=self.user['id'], enabled=True)
        self.assertEqual(self.store.audit_list(1)[0]['actor'], self.user['id'])


class GuestHTTPTest(WebCase):
    def test_retired_public_account_session_must_open_guest_access(self):
        browser = Browser(self.app)
        browser.session()
        email = 'retired@example.com'
        token, session = self.svc.signer.new_token(), self.svc.signer.new_token()
        self.store.login_token_create(self.svc.signer.hash('login', token), email, 600)
        self.store.login_complete(self.svc.signer.hash('login', token), self.svc.signer.hash('session', session),
                                  admin_emails=(email,), session_ttl_s=3600)
        browser.c.cookies.set('hg_session', session, domain='testserver.local', path='/')
        self.assertFalse(browser.session()['signed_in'])
        opened = browser.post('/api/demo/access', json={})
        self.assertEqual(opened.status_code, 200)
        self.assertEqual(browser.session()['user']['kind'], 'guest')

    def test_guest_double_submit_and_admin_only_email(self):
        guest = Browser(self.app); guest.session()
        r = guest.post('/api/demo/access', json={})
        guest.csrf = r.json()['csrf']
        first = self.make_job(guest)
        second = self.make_job(guest)
        self.assertEqual((first.status_code, second.status_code), (201,200))
        self.assertEqual(first.json()['id'], second.json()['id'])
        self.assertNotIn('2h', [s['key'] for s in first.json()['stages']])
        self.assertEqual(len(self.store.jobs_admin_list()), 1)
        self.assertEqual(guest.get('/api/admin/generation').status_code, 403)
        ops = self.signin('ops@example.com')
        admin_response = ops.post('/api/auth/request', json={'email':'ops@example.com'})
        count = len(self.sender.sent)
        stranger = ops.post('/api/auth/request', json={'email':'stranger@example.com'})
        self.assertEqual(stranger.json(), admin_response.json())
        self.assertEqual(len(self.sender.sent), count)
        self.assertEqual(ops.post('/api/admin/generation', json={'enabled':False}, csrf=False).status_code,403)
        self.assertEqual(ops.post('/api/admin/generation', json={'enabled':False}).status_code,200)
        self.assertEqual(self.store.audit_list(1)[0]['actor'], self.store.users_list()[-1]['id'])
