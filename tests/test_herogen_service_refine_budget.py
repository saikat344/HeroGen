"""Offline interrupted-client regressions; AF_UNIX simulates the Windows bridge on POSIX."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

from herogen.service import refine_budget


class TestInterruptedBridge(unittest.TestCase):
    def isolated(self, code):
        # A transport deadlock must fail this test, rather than hang the test runner.
        root = Path(__file__).resolve().parents[1]
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(root), str(root / 'tests'))),
               'PYTHONDONTWRITEBYTECODE': '1', 'MODEL_API_KEY': '',
               'MODAL_TOKEN_ID': '', 'MODAL_TOKEN_SECRET': ''}
        result = subprocess.run([sys.executable, '-c', textwrap.dedent(code)], cwd=root, env=env,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr, '')  # also catches unhandled bridge-thread exceptions

    def test_unauthenticated_disconnect_and_idle_peer_recover(self):
        self.isolated('''
            import json, os
            from multiprocessing.connection import Client
            from unittest import mock
            from herogen.service import refine_budget
            callback = mock.Mock(return_value={'allowed': True})
            with mock.patch.object(refine_budget, '_AUTH_TIMEOUT', .1):
                with refine_budget._connection_bridge(callback, 'TEST_BUDGET') as env:
                    address = json.loads(env['TEST_BUDGET_ADDRESS'])
                    for disconnect in (True, False):
                        with Client(address) as peer:
                            peer.recv_bytes(256)
                            if disconnect:
                                peer.close()
                            with mock.patch.dict(os.environ, env):
                                assert refine_budget.remote_budget('reserve', prefix='TEST_BUDGET') == {'allowed': True}
                    assert callback.call_count == 2
                    assert all(c.kwargs == {'action': 'reserve'} for c in callback.call_args_list)
        ''')

    def test_shutdown_cancels_idle_authentication_and_request(self):
        self.isolated('''
            import json, threading
            from multiprocessing.connection import Client
            from unittest import mock
            from herogen.service import refine_budget
            before = set(threading.enumerate())
            for authenticate in (False, True):
                callback = mock.Mock()
                with refine_budget._connection_bridge(callback, 'TEST_BUDGET') as env:
                    key = bytes.fromhex(env['TEST_BUDGET_AUTHKEY']) if authenticate else None
                    peer = Client(json.loads(env['TEST_BUDGET_ADDRESS']), authkey=key)
                    if not authenticate:
                        peer.recv_bytes(256)
                peer.close()
                callback.assert_not_called()
            assert set(threading.enumerate()) == before
        ''')

    def test_shutdown_cancels_reply_to_peer_that_stops_reading(self):
        self.isolated('''
            import json, threading
            from multiprocessing.connection import Client
            from herogen.service import refine_budget
            before = set(threading.enumerate())
            called = threading.Event()
            def callback(**kw):
                called.set()
                return {'payload': 'x' * (8 * 1024 * 1024)}
            with refine_budget._connection_bridge(callback, 'TEST_BUDGET') as env:
                peer = Client(json.loads(env['TEST_BUDGET_ADDRESS']),
                              authkey=bytes.fromhex(env['TEST_BUDGET_AUTHKEY']))
                peer.send_bytes(b'{"action":"reserve"}')
                assert called.wait(2)
            peer.close()
            assert set(threading.enumerate()) == before
        ''')

    @unittest.skipIf(sys.platform == 'win32', 'partial stream frames apply to the AF_UNIX simulation')
    def test_partial_authentication_and_request_frames_cannot_block_shutdown(self):
        self.isolated('''
            import json, socket, struct, threading
            from multiprocessing.connection import Client
            from unittest import mock
            from herogen.service import refine_budget
            before = set(threading.enumerate())
            for authenticate in (False, True):
                callback = mock.Mock()
                with refine_budget._connection_bridge(callback, 'TEST_BUDGET') as env:
                    key = bytes.fromhex(env['TEST_BUDGET_AUTHKEY']) if authenticate else None
                    peer = Client(json.loads(env['TEST_BUDGET_ADDRESS']), authkey=key)
                    if not authenticate:
                        peer.recv_bytes(256)
                    wire = socket.socket(fileno=peer.fileno())
                    wire.sendall(struct.pack('!i', 128) + b'x')
                    wire.detach()
                peer.close()
                callback.assert_not_called()
            assert set(threading.enumerate()) == before
        ''')

    def test_stage_timeout_mid_handshake_or_send_keeps_full_price_settlement(self):
        # Uses the Windows budget_bridge dispatch on every host. POSIX supplies AF_UNIX and
        # process-group termination; native AF_PIPE and TerminateJobObject need a Windows runner.
        child = '''
import json, os, socket, struct, sys, time
from pathlib import Path
from multiprocessing.connection import Client
from herogen.pipeline import ledger
ledger.sent(Path(sys.argv[1]), model='synthetic', endpoint='responses', key='hash', timeout=1, price_usd=.01)
address = json.loads(os.environ[ledger.BRIDGE + '_ADDRESS'])
key = bytes.fromhex(os.environ[ledger.BRIDGE + '_AUTHKEY']) if sys.argv[3] == 'send' else None
peer = Client(address, authkey=key)
if key is None:
    peer.recv_bytes(256)
else:
    wire = socket.socket(fileno=peer.fileno())
    wire.sendall(struct.pack('!i', 128) + b'x')
    wire.detach()
Path(sys.argv[2]).write_text('connected')
time.sleep(60)
'''
        phases = ('handshake',) if sys.platform == 'win32' else ('handshake', 'send')
        for phase in phases:
            with self.subTest(phase=phase):
                self.isolated('''
                    import json, os, sys, types
                    from unittest import mock
                    from herogen.service import refine_budget
                    from herogen.service.processes import identity
                    from herogen.service.runner import CliRunner, plan_for
                    from herogen.service.storage import Storage
                    from herogen.service.worker import make_context, settle_admissions
                    from test_herogen_service_admission import TestAdmission

                    fixture = TestAdmission()
                    fixture.setUp()
                    try:
                        user, _ = fixture.guest()
                        job, _ = fixture.job(user)
                        store, settings = fixture.store, fixture.s
                        store.admission_work_begin(job['id'], 'ipc', 'model', settings.cost_signature,
                                                   process=identity())
                        ctx = make_context(settings, Storage(settings.data_dir), job)
                        ctx.run_dir.mkdir(parents=True)
                        ctx.timeout_s = 1
                        ctx.accounting = lambda action, **kw: store.admission_record(job['id'], 'ipc', action, **kw)
                        marker = ctx.run_dir / 'connected'
                        child = CHILD
                        runner = CliRunner()
                        win_os = types.SimpleNamespace(name='nt', path=os.path)
                        with mock.patch.object(refine_budget, 'os', win_os), \\
                                mock.patch.object(runner, 'available', return_value=None), \\
                                mock.patch.object(runner, 'command', return_value=(
                                    [sys.executable, '-c', child, str(ctx.cache_dir), str(marker), PHASE], None)):
                            out = runner.run(plan_for('sketch', 'best')[0], ctx)
                        assert out.code == 'timeout', out
                        assert marker.read_text() == 'connected'
                        assert store._admission_snapshot(job['id'])['processes'][0]['stopped']
                        store.admission_work_end(job['id'], 'ipc', False, worker_stopped=True)
                        fixture.close(job)
                        fixture.clock.advance(181)
                        assert settle_admissions(store) == [job['id']]
                        row = store.budget_report()['admissions'][0]
                        assert row['closed'] and row['actual_micro'] == 10000, row
                        assert store.budget_report()['summary']['active'] == 0
                        fixture.job(user)
                    finally:
                        fixture.tearDown()
                '''.replace('PHASE', repr(phase)).replace('CHILD', repr(child)))

    def test_windows_pending_write_is_canceled_on_shutdown(self):
        import threading
        api = mock.Mock(ERROR_IO_PENDING=997, WAIT_TIMEOUT=258)
        operation = mock.Mock()
        api.WriteFile.return_value = operation, 997
        api.WaitForMultipleObjects.return_value = 258
        done = threading.Event()
        api.WaitForMultipleObjects.side_effect = lambda *args: (done.set() or 258)
        with mock.patch.object(refine_budget.sys, 'platform', 'win32'), mock.patch.dict(sys.modules, _winapi=api):
            bounded = refine_budget._BoundedConnection(mock.Mock(), done)
            with self.assertRaises(TimeoutError):
                bounded.send_bytes(b'response')
        operation.cancel.assert_called_once_with()
        operation.GetOverlappedResult.assert_called_once_with(True)


if __name__ == '__main__':
    unittest.main()
