"""Private guest sessions, concurrent budget admission and conservative work accounting; no paid calls."""
from __future__ import annotations
import concurrent.futures
import secrets
import threading
import json
import sys
from unittest import mock
from test_herogen_service_support import Clock, TempDirCase, settings, png
from herogen.service.store import Store, StoreError, new_id
from herogen.service.admission import micro_usd, ADMIN, PUBLIC
from herogen.service.fakes import FakeRunner
from herogen.service.storage import Storage
from herogen.service.worker import run_job, run_group, InPlaceWorkdir, make_context
from herogen.service.worker import settle_admissions, reconcile, run_heldout
from herogen.service.processes import identity
from herogen.service.runner import CliRunner, plan_for
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

    def settlement_job(self, *, live=False, modal=False, complete=None):
        u,_=self.guest(); job,_=self.job(u)
        self.store.admission_work_begin(job['id'],'work','model',self.s.cost_signature,process=identity())
        sent=dict(call_id='call',model='synthetic',endpoint='responses',key='request-hash',
                  outcome='sent',deadline=self.clock()+120,price_usd=.01)
        if modal:
            sent.update(endpoint='modal/infer',timeout_s=1800,price_per_second=.001,price_usd=1.8)
        self.store.admission_record(job['id'],'work','call',row=sent)
        if complete is not None:
            self.store.admission_record(job['id'],'work','call',row={**sent,**complete})
        self.store.admission_work_end(job['id'],'work',False,worker_stopped=not live)
        self.close(job)
        return job,sent

    def test_sent_crash_settles_at_full_price_only_after_deadline_and_grace(self):
        job,sent=self.settlement_job()
        self.assertEqual(settle_admissions(self.store),[])
        self.clock.advance(180)
        self.assertEqual(settle_admissions(self.store),[])
        self.clock.advance(.001)
        self.assertEqual(settle_admissions(self.store),[job['id']])
        report=self.store.budget_report(); row=report['admissions'][0]
        self.assertEqual(row['actual_micro'],10_000); self.assertTrue(row['closed'])
        self.assertTrue(row['auto_settled']); self.assertEqual(report['summary']['active'],0)
        self.assertIn('1 calls, 1 charged at full price, ledger ',row['evidence'])
        self.assertEqual(settle_admissions(self.store),[])
        self.assertEqual(len([a for a in self.store.audit_list() if a['action']=='auto_settled']),1)

    def test_live_worker_and_live_descendant_retain_the_reservation(self):
        job,_=self.settlement_job(live=True); self.clock.advance(1000)
        self.assertEqual(settle_admissions(self.store),[])
        # A worker returning does not prove an orphaned child group stopped.
        self.store.db.execute('INSERT INTO admission_processes VALUES (?,?,?,0)',('work',identity()['host'],123))
        self.store.admission_work_end(job['id'],'work',False,worker_stopped=True)
        with mock.patch('herogen.service.worker.stopped',return_value=False):
            self.assertEqual(settle_admissions(self.store,process_stopped=lambda *a,**k:False),[])
        self.assertIsNone(self.store.budget_report()['admissions'][0]['actual_micro'])

    def test_unknown_modal_uses_function_timeout_times_gpu_price(self):
        job,_=self.settlement_job(modal=True,complete={'outcome':'unknown','cost_usd':.001})
        self.clock.advance(181)
        self.assertEqual(settle_admissions(self.store),[job['id']])
        self.assertEqual(self.store.budget_report()['admissions'][0]['actual_micro'],1_800_000)

    def test_normal_completion_uses_known_cost_without_counting_sent_twice(self):
        job,_=self.settlement_job(complete={'outcome':'ok','cost_usd':.0031})
        self.clock.advance(181)
        self.assertEqual(settle_admissions(self.store),[job['id']])
        row=self.store.budget_report()['admissions'][0]
        self.assertEqual(row['actual_micro'],3100); self.assertIn('0 charged at full price',row['evidence'])

    def test_manual_reconcile_wins_after_settler_read_without_double_settlement(self):
        job,_=self.settlement_job(); self.clock.advance(181)
        auto=self.store.admission_auto_settle
        def race(job_id,snapshot):
            self.store.budget_reconcile(job_id,7,'Synthetic manual evidence',work_stopped=True)
            return auto(job_id,snapshot)
        with mock.patch.object(self.store,'admission_auto_settle',side_effect=race):
            self.assertEqual(settle_admissions(self.store),[])
        row=self.store.budget_report()['admissions'][0]
        self.assertEqual(row['actual_micro'],7); self.assertFalse(row['auto_settled'])
        self.assertEqual(len([a for a in self.store.audit_list() if a['action']=='generation_budget_reconciled']),1)

    def test_expired_lease_does_not_prove_a_live_worker_stopped(self):
        u,_=self.guest(); job,_=self.job(u); self.store.job_claim(job['id'],'call',10,2)
        self.store.admission_work_begin(job['id'],'work','model',self.s.cost_signature,process=identity())
        self.clock.advance(1000)
        reconcile(self.store,self.s,lambda jid:'unused')
        self.assertIsNone(self.store.budget_report()['admissions'][0]['actual_micro'])

    def test_gone_worker_settles_after_sweep_and_foreign_or_missing_evidence_retains(self):
        job,_=self.settlement_job(live=True); self.clock.advance(181)
        self.assertEqual(settle_admissions(self.store,process_stopped=lambda *a,**k:True),[job['id']])
        u,_=self.guest(); old,_=self.job(u)
        self.store.admission_work_begin(old['id'],'old','model',self.s.cost_signature)
        self.close(old)
        self.clock.advance(1000)
        self.assertEqual(settle_admissions(self.store,process_stopped=lambda *a,**k:True),[])

    def test_changed_ledger_snapshot_cannot_settle(self):
        job,_=self.settlement_job(); self.clock.advance(181)
        snapshot=self.store._admission_snapshot(job['id'])
        self.store.db.execute("UPDATE admission_calls SET receipt=? WHERE job_id=?",
                             (json.dumps(dict(snapshot['calls'][0],deadline=self.clock()+1000)),job['id']))
        self.assertFalse(self.store.admission_auto_settle(job['id'],snapshot))

    def test_stage_subprocess_acknowledges_ledger_and_records_its_process_group(self):
        user,_=self.guest(); job,_=self.job(user); storage=Storage(self.s.data_dir)
        self.store.admission_work_begin(job['id'],'stage','model',self.s.cost_signature,process=identity())
        ctx=make_context(self.s,storage,job); ctx.run_dir.mkdir(parents=True)
        ctx.accounting=lambda action,**kw:self.store.admission_record(job['id'],'stage',action,**kw)
        script="from pathlib import Path; from herogen.pipeline import ledger; import sys; ledger.sent(Path(sys.argv[1]),model='synthetic',endpoint='responses',key='hash',timeout=1,price_usd=.01)"
        runner=CliRunner()
        with mock.patch.object(runner,'available',return_value=None), \
                mock.patch.object(runner,'command',return_value=([sys.executable,'-c',script,str(ctx.cache_dir)],None)):
            runner.run(plan_for('sketch','best')[0],ctx)
        snapshot=self.store._admission_snapshot(job['id'])
        self.assertEqual(len(snapshot['calls']),1)
        self.assertEqual(snapshot['calls'][0]['outcome'],'sent')
        self.assertEqual(len(snapshot['processes']),1); self.assertTrue(snapshot['processes'][0]['stopped'])

    def test_compute_receipts_use_elapsed_time_or_full_timeout(self):
        user,_=self.guest(); job,_=self.job(user)
        process={**identity(),'compute_timeout_s':100,'compute_price_per_second':.001}
        self.store.admission_work_begin(job['id'],'compute','model',self.s.cost_signature,process=process)
        self.store.admission_work_end(job['id'],'compute',False,worker_stopped=True,compute_seconds=2)
        self.close(job); self.clock.advance(161)
        self.assertEqual(settle_admissions(self.store),[job['id']])
        self.assertEqual(self.store.budget_report()['admissions'][0]['actual_micro'],2000)

    def test_budgeted_stage_timeout_reaps_its_registered_process(self):
        user,_=self.guest(); job,_=self.job(user); storage=Storage(self.s.data_dir)
        self.store.admission_work_begin(job['id'],'timeout','model',self.s.cost_signature,process=identity())
        ctx=make_context(self.s,storage,job); ctx.run_dir.mkdir(parents=True); ctx.timeout_s=.05
        ctx.accounting=lambda action,**kw:self.store.admission_record(job['id'],'timeout',action,**kw)
        runner=CliRunner()
        with mock.patch.object(runner,'available',return_value=None), \
                mock.patch.object(runner,'command',return_value=([sys.executable,'-c','import time; time.sleep(60)'],None)):
            result=runner.run(plan_for('sketch','best')[0],ctx)
        self.assertEqual(result.code,'timeout')
        self.assertTrue(self.store._admission_snapshot(job['id'])['processes'][0]['stopped'])

    def test_timeout_descendant_group_settles_from_another_host(self):
        user,_=self.guest(); job,_=self.job(user); storage=Storage(self.s.data_dir)
        self.store.admission_work_begin(job['id'],'timeout','model',self.s.cost_signature,
                                        process={'host':'worker-container','pid':123})
        ctx=make_context(self.s,storage,job); ctx.run_dir.mkdir(parents=True); ctx.timeout_s=.25
        ctx.accounting=lambda action,**kw:self.store.admission_record(job['id'],'timeout',action,**kw)
        script=("import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', "
                "'import time; b=bytearray(200_000_000); time.sleep(60)']); time.sleep(60)")
        runner=CliRunner()
        with mock.patch.object(runner,'available',return_value=None), \
                mock.patch.object(runner,'command',return_value=([sys.executable,'-c',script],None)):
            result=runner.run(plan_for('sketch','best')[0],ctx)
        self.assertEqual(result.code,'timeout')
        self.assertTrue(self.store._admission_snapshot(job['id'])['processes'][0]['stopped'])
        self.store.admission_work_end(job['id'],'timeout',False,worker_stopped=True)
        self.close(job); self.clock.advance(61)
        with mock.patch('herogen.service.processes.socket.gethostname',return_value='sweep-container'):
            self.assertEqual(settle_admissions(self.store),[job['id']])
        row=self.store.budget_report()['admissions'][0]
        self.assertTrue(row['closed']); self.assertEqual(row['actual_micro'],0)
        self.assertEqual(self.store.budget_report()['summary']['active'],0)
        self.job(user)

    def test_group_exit_wait_retains_unproven_processes(self):
        from herogen.service.processes import wait_stopped
        with mock.patch('herogen.service.processes.stopped',side_effect=[False,False,True]), \
                mock.patch('herogen.service.processes.time.sleep'):
            self.assertTrue(wait_stopped(identity(),group=True))
        with mock.patch('herogen.service.processes.stopped',return_value=False):
            self.assertFalse(wait_stopped(identity(),group=True,timeout=0))

    def test_windows_unproven_group_does_not_abort_later_settlement(self):
        import types
        from herogen.service import processes
        first,_=self.settlement_job()
        self.store.db.execute('INSERT INTO admission_processes VALUES (?,?,?,0)',
                              ('work',identity()['host'],123))
        user,_=self.guest(); second,_=self.job(user)
        self.store.admission_work_begin(second['id'],'other','model',self.s.cost_signature,process=identity())
        self.store.admission_work_end(second['id'],'other',True,worker_stopped=True)
        self.close(second); self.clock.advance(181)
        with mock.patch.object(processes,'os',types.SimpleNamespace(name='nt')):
            self.assertEqual(settle_admissions(self.store),[second['id']])
        self.assertIsNone(next(r for r in self.store.budget_report()['admissions']
                               if r['job_id']==first['id'])['actual_micro'])

    def test_windows_stage_transport_acknowledges_before_sending(self):
        import types
        from herogen.service import refine_budget
        user,_=self.guest(); job,_=self.job(user); storage=Storage(self.s.data_dir)
        self.store.admission_work_begin(job['id'],'ipc','model',self.s.cost_signature,process=identity())
        ctx=make_context(self.s,storage,job); ctx.run_dir.mkdir(parents=True)
        ctx.accounting=lambda action,**kw:self.store.admission_record(job['id'],'ipc',action,**kw)
        script=("from pathlib import Path; from herogen.pipeline import ledger; import sys; "
                "ledger.sent(Path(sys.argv[1]),model='synthetic',endpoint='responses',key='hash',timeout=1,price_usd=.01)")
        runner=CliRunner()
        with mock.patch.object(refine_budget,'os',types.SimpleNamespace(name='nt',path=refine_budget.os.path)), \
                mock.patch.object(runner,'available',return_value=None), \
                mock.patch.object(runner,'command',return_value=([sys.executable,'-c',script,str(ctx.cache_dir)],None)):
            runner.run(plan_for('sketch','best')[0],ctx)
        snapshot=self.store._admission_snapshot(job['id'])
        self.assertEqual(len(snapshot['calls']),1)
        self.assertEqual(snapshot['calls'][0]['outcome'],'sent')
        self.assertTrue(snapshot['processes'][0]['stopped'])

    def test_parallel_settlers_commit_once(self):
        job,_=self.settlement_job(); self.clock.advance(181)
        barrier=threading.Barrier(2); auto=self.store.admission_auto_settle
        def compete(job_id,snapshot):
            barrier.wait(timeout=10)
            return auto(job_id,snapshot)
        with mock.patch.object(self.store,'admission_auto_settle',side_effect=compete), \
                concurrent.futures.ThreadPoolExecutor(2) as pool:
            results=list(pool.map(lambda _:settle_admissions(self.store),range(2)))
        self.assertEqual(sum(r==[job['id']] for r in results),1)
        self.assertEqual(len([a for a in self.store.audit_list() if a['action']=='auto_settled']),1)

    def test_pending_side_work_and_incomplete_prices_retain_reservations(self):
        job,_=self.settlement_job(); self.clock.advance(181)
        self.store.db.execute('UPDATE admissions SET side_done=0 WHERE job_id=?',(job['id'],))
        self.assertEqual(settle_admissions(self.store),[])
        self.store.db.execute('UPDATE admissions SET side_done=1 WHERE job_id=?',(job['id'],))
        row=self.store._admission_snapshot(job['id'])['calls'][0]
        row['price_usd']=None
        self.store.db.execute('UPDATE admission_calls SET receipt=? WHERE job_id=?',(json.dumps(row),job['id']))
        self.assertEqual(settle_admissions(self.store),[])
        self.assertIsNone(self.store.budget_report()['admissions'][0]['actual_micro'])

    def test_uncertain_work_can_confirm_exit_and_completion_but_cannot_send_again(self):
        job,sent=self.settlement_job(live=True)
        self.store.db.execute('INSERT INTO admission_processes VALUES (?,?,?,0)',('work',identity()['host'],123))
        self.store.admission_record(job['id'],'work','process_stopped',host=identity()['host'],pid=123)
        self.store.admission_record(job['id'],'work','call',row={**sent,'outcome':'ok','cost_usd':.002})
        with self.assertRaises(ValueError): self.store.admission_record(job['id'],'work','call',row=sent)
        self.store.admission_work_end(job['id'],'work',False,worker_stopped=True)
        self.clock.advance(181)
        self.assertEqual(settle_admissions(self.store),[job['id']])
        self.assertEqual(self.store.budget_report()['admissions'][0]['actual_micro'],2000)

    def test_side_setup_exception_still_records_invocation_exit(self):
        user,_=self.guest(); job,_=self.execute_job(user,FakeRunner(),lambda job:None)
        def unavailable(claim): raise RuntimeError('Synthetic work folder failure')
        with self.assertRaises(RuntimeError):
            run_heldout(self.store,self.s,job,FakeRunner(),unavailable)
        work=next(w for w in self.store._admission_snapshot(job['id'])['work'] if w['kind']=='heldout')
        self.assertTrue(json.loads(work['process'])['stopped'])
        self.clock.advance(61)
        self.assertEqual(settle_admissions(self.store),[job['id']])


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
