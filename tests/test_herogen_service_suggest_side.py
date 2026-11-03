import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from herogen.service.runner import SUGGEST_FILE, Suggestion


def started(code: str, run: Path) -> Suggestion:
    home = Path(tempfile.mkdtemp())
    with open(home / 'suggest.log', 'wb') as out:
        proc = subprocess.Popen([sys.executable, '-c', code, str(run)], stdout=out, stderr=subprocess.STDOUT)
    return Suggestion(proc, home, run)


class SuggestionSideTask(unittest.TestCase):
    def setUp(self):
        self.run = Path(tempfile.mkdtemp())

    def test_its_spend_comes_from_the_file_it_wrote(self):
        code = ("import json,sys,pathlib; pathlib.Path(sys.argv[1], %r).write_text(json.dumps({'cost_usd': 0.005}))"
                % SUGGEST_FILE)
        s = started(code, self.run)
        self.assertAlmostEqual(s.finish(), 0.005)
        self.assertFalse(s.home.exists(), 'its private home is removed')

    def test_a_failed_run_costs_nothing_and_never_raises(self):
        s = started("import sys; print('missing inputs', file=sys.stderr); sys.exit(1)", self.run)
        with self.assertLogs('herogen.service', 'WARNING'):
            self.assertEqual(s.finish(), 0.0)

    def test_a_chatty_run_is_never_blocked_by_its_own_output(self):
        # More than a pipe holds (64 KiB): with the output in a file the run ends at once instead of stalling.
        code = ("import sys; sys.stderr.write('x' * 1_000_000); sys.stderr.flush(); "
                "import json,pathlib; pathlib.Path(sys.argv[1], %r).write_text(json.dumps({'cost_usd': 0.004}))"
                % SUGGEST_FILE)
        s = started(code, self.run)
        time.sleep(2)  # nobody reads it while the game body is built beside it
        self.assertAlmostEqual(s.finish(wait_s=1.5), 0.004)

    def test_stopping_it_also_stops_the_render_it_started(self):
        # The real run starts a Blender render; stopping the run must stop that child too.
        home = Path(tempfile.mkdtemp())
        code = ("import subprocess, sys, time, pathlib; "
                "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                "pathlib.Path(sys.argv[1], 'child.pid').write_text(str(c.pid)); time.sleep(60)")
        with open(home / 'suggest.log', 'wb') as out:
            proc = subprocess.Popen([sys.executable, '-c', code, str(self.run)], stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        for _ in range(100):
            if (self.run / 'child.pid').exists():
                break
            time.sleep(0.05)
        child = int((self.run / 'child.pid').read_text())
        with self.assertLogs('herogen.service', 'WARNING'):
            Suggestion(proc, home, self.run).finish(wait_s=0)
        for _ in range(100):
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            os.kill(child, 9)
            self.fail('the child process outlived the stopped run')

    def test_a_stuck_run_is_stopped(self):
        s = started("import time; time.sleep(30)", self.run)
        with self.assertLogs('herogen.service', 'WARNING'):
            self.assertEqual(s.finish(wait_s=0.5), 0.0)
        self.assertIsNotNone(s.proc.returncode)


if __name__ == '__main__':
    unittest.main()
