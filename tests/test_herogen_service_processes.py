"""Offline process evidence, descendant containment and authenticated subprocess accounting."""
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from multiprocessing.connection import Client
from multiprocessing import AuthenticationError
from unittest import mock

from herogen.service import processes, refine_budget, windows_job


class TestProcessEvidence(unittest.TestCase):
    def test_live_self_is_never_signalled_and_dead_child_is_detected(self):
        self.assertFalse(processes.stopped(processes.identity()))
        with subprocess.Popen([sys.executable,'-c','pass']) as child:
            child.wait()
            self.assertTrue(processes.stopped({**processes.identity(),'pid':child.pid}))

    def test_windows_probe_uses_handle_evidence_and_never_kill(self):
        current=processes.identity()
        with mock.patch.object(processes,'os',types.SimpleNamespace(name='nt')), \
                mock.patch('herogen.pipeline.retry._alive',side_effect=[True,False]) as alive:
            self.assertFalse(processes.stopped(current))
            self.assertTrue(processes.stopped(current))
            self.assertFalse(processes.stopped(current,group=True))
        self.assertEqual(alive.call_count,2)

    def test_foreign_and_malformed_processes_retain_reservations(self):
        for process in ({'host':'foreign','pid':123},{**processes.identity(),'pid':0},
                        {**processes.identity(),'pid':True}):
            self.assertFalse(processes.stopped(process))
        with mock.patch('herogen.pipeline.retry._alive',side_effect=OSError('cannot inspect')):
            self.assertFalse(processes.stopped(processes.identity()))

    def test_stage_wrapper_cannot_start_before_acknowledgement(self):
        from herogen.service import stage_process
        with mock.patch.object(stage_process,'remote_budget',side_effect=RuntimeError('not registered')), \
                mock.patch.object(stage_process.subprocess,'call') as call:
            with self.assertRaises(RuntimeError):
                stage_process.main()
            call.assert_not_called()

    def test_windows_stage_is_assigned_before_registration_and_kills_the_job(self):
        proc=mock.MagicMock(pid=321)
        proc.poll.return_value=0
        job=mock.Mock()
        job.wait_empty.return_value=True
        with mock.patch.object(processes,'sys',types.SimpleNamespace(platform='win32',executable=sys.executable)), \
                mock.patch.object(windows_job,'WindowsJob',return_value=job), \
                mock.patch.object(processes.subprocess,'Popen') as popen:
            popen.return_value.__enter__.return_value=proc
            with processes.stage_process(['stage','arg']) as (child,stop):
                job.assign.assert_called_once_with(321)
                self.assertIs(child,proc)
                self.assertTrue(stop())
            self.assertEqual(popen.call_args.args[0],
                             [sys.executable,'-m','herogen.service.stage_process','stage','arg'])
        job.terminate.assert_called_once(); job.close.assert_called_once()

    def test_job_wait_requires_zero_active_processes(self):
        kernel=mock.Mock()
        kernel.CreateJobObjectW.return_value=11
        kernel.SetInformationJobObject.return_value=True
        active=iter([2,1,0])
        def query(handle,kind,state,size,unused):
            state._obj.active=next(active)
            return True
        kernel.QueryInformationJobObject.side_effect=query
        with mock.patch.object(windows_job,'_kernel',return_value=kernel), \
                mock.patch.object(windows_job.time,'sleep'):
            job=windows_job.WindowsJob()
            self.assertEqual(kernel.SetInformationJobObject.call_args.args[2]._obj.basic.flags,0x2000)
            self.assertTrue(job.wait_empty())
            kernel.QueryInformationJobObject.side_effect=None
            kernel.QueryInformationJobObject.return_value=False
            self.assertFalse(job.wait_empty())
            def live(handle,kind,state,size,unused):
                state._obj.active=1
                return True
            kernel.QueryInformationJobObject.side_effect=live
            self.assertFalse(job.wait_empty(timeout=0))
            job.close()
        kernel.CloseHandle.assert_called_once_with(11)


class TestConnectionBridge(unittest.TestCase):
    def test_wrong_authentication_cannot_reach_callback(self):
        callback=mock.Mock(return_value={'allowed':True})
        with refine_budget._connection_bridge(callback,'TEST_BUDGET') as env:
            address=json.loads(env['TEST_BUDGET_ADDRESS'])
            with self.assertRaises(AuthenticationError):
                Client(address,authkey=b'incorrect')
            callback.assert_not_called()
            with mock.patch.dict(os.environ,env):
                self.assertEqual(refine_budget.remote_budget('reserve',prefix='TEST_BUDGET',amount=.1),
                                 {'allowed':True})
            callback.assert_called_once_with(action='reserve',amount=.1)

    def test_failed_durable_write_cannot_authorize_a_send(self):
        from herogen.pipeline import ledger
        with tempfile.TemporaryDirectory() as folder, \
                refine_budget._connection_bridge(mock.Mock(side_effect=OSError('unavailable')),ledger.BRIDGE) as env, \
                mock.patch.dict(os.environ,env):
            from pathlib import Path
            cache=Path(folder)
            with self.assertRaisesRegex(RuntimeError,'could not be acknowledged'):
                ledger.sent(cache,model='synthetic',endpoint='responses',key='hash',timeout=1,price_usd=.01)
            self.assertFalse((cache/ledger.LEDGER).exists())


if __name__ == '__main__':
    unittest.main()
