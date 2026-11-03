"""Check stage presentation semantics without model calls or browser-only dependencies."""
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is required for generation presentation checks')
class GenerationTest(unittest.TestCase):
    def test_stage_semantics(self):
        result = subprocess.run([shutil.which('node'), '--experimental-vm-modules', '--test', 'tests/service_generation.test.mjs'],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_preview_request_lifecycle(self):
        result = subprocess.run([shutil.which('node'), '--experimental-vm-modules', '--test',
                                 'tests/service_preview.test.mjs'], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_animation_poses(self):
        result = subprocess.run([shutil.which('node'), '--test', 'tests/service_animate.test.mjs'],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(shutil.which('node'), 'Node is required for Build log checks')
class BuildLogTest(unittest.TestCase):
    def test_build_log_rows_copy_and_fallback(self):
        for name in ('tests/service_buildlog.test.mjs', 'tests/service_workbench.test.mjs'):
            result = subprocess.run([shutil.which('node'), '--experimental-vm-modules', '--test', name],
                                    cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
