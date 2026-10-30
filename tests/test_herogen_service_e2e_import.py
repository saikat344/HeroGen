"""The browser end-to-end module must stay importable without the web extras (uvicorn and the rest of
requirements-dev.txt): CI has none of them, and an import crash there reads as a failure, not a skip. A child
interpreter has the extras blocked in sys.modules, then imports the module and runs its suite; no network and no
browser are involved."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

CHILD = """
import json, sys
sys.path[:0] = [sys.argv[1]]
for name in ("starlette", "httpx", "httpx2", "uvicorn"):
    sys.modules[name] = None
import unittest
import tests.test_herogen_service_e2e as e2e
suite = unittest.defaultTestLoader.loadTestsFromModule(e2e)
result = unittest.TestResult()
suite.run(result)
print(json.dumps({"tests": result.testsRun,
                  "skipped": [[t.id(), reason] for t, reason in result.skipped],
                  "errors": [tb for _, tb in result.errors],
                  "failures": [tb for _, tb in result.failures]}))
"""


class E2EImportTest(unittest.TestCase):
    def test_without_the_web_extras_the_module_imports_and_skips(self):
        p = subprocess.run([sys.executable, "-c", CHILD, str(_ROOT)], capture_output=True, text=True,
                           cwd=_ROOT, timeout=120)
        self.assertEqual(p.returncode, 0, p.stderr)
        r = json.loads(p.stdout)
        self.assertEqual((r["errors"], r["failures"]), ([], []))
        self.assertEqual(r["tests"], len(r["skipped"]), "without the extras nothing may run")
        self.assertTrue(r["skipped"], "TestServicePage must be reported as skipped")
        for test_id, reason in r["skipped"]:
            self.assertIn("TestServicePage", test_id)
            self.assertIn("E2E_CDP_PORT", reason, "the skip must say what to set up")


if __name__ == "__main__":
    unittest.main()
