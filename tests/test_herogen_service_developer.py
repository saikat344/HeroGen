"""Exercise the optional inspector's data semantics, request lifecycle and safe rendering with Node."""
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which("node"), "Node is required for the inspector's JavaScript checks")
class DeveloperModeTest(unittest.TestCase):
    def test_inspection_semantics_and_lifecycle(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([shutil.which("node"), "--test", "tests/service_developer.test.mjs"],
                                cwd=root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
