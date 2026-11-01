"""Permanent source gate against retired website mechanisms."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    'admission_', 'admissions', 'budget_reconcile', 'budget_overhead', 'settle_admissions',
    'auto_settle', 'generation_uncertain', 'workspace_active', 'generation_capacity',
    'cost_bound_unverified', 'version_changed', 'policy_expiry', 'policy_expires',
    'refine_budget.py', 'HEROGEN_REFINE_BUDGET_', 'job_refine_budget', 'refine_budgets',
    'herogen-drafts', 'pendingIdempotency', 'pending_idempotency', 'was not confirmed',
    'invite', 'allowlist', 'quota', 'refine_budget', 'pending_key', 'pendingkey', 'policy expiry', 'quota_daily', 'quota_total', 'counts_quota', 'budget_required',
    'stage-2h', 'stage2h', '"2h"', "'2h'", 'heldout', 'held-out',
)


def scan():
    violations = []
    files = 0
    for path in sorted((ROOT / 'herogen/service').rglob('*')):
        if path.suffix not in ('.py', '.js', '.html', '.css', '.mjs', '.txt'):
            continue
        source = path.read_text()
        files += 1
        if path.name == 'migrations.py':
            # Immutable historical SQL and the table-retirement migration live here only.
            tree = ast.parse(source)
            if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in ast.walk(tree)):
                violations.append('migrations.py must contain only schema definitions')
            continue
        for line, text in enumerate(source.splitlines(), 1):
            for token in FORBIDDEN:
                if token.lower() in text.lower():
                    violations.append(f'{path.relative_to(ROOT)}:{line}: {token}')
    return files, violations


class NoLegacyPathsTest(unittest.TestCase):
    def test_service_has_no_retired_mechanisms(self):
        files, violations = scan()
        self.assertEqual(violations, [], '\n'.join(violations))
        self.assertGreater(files, 0)


if __name__ == '__main__':
    files, violations = scan()
    print('\n'.join(violations) if violations else f'PASS: {files} service source files scanned; 0 legacy-path references (migration SQL excepted).')
    raise SystemExit(bool(violations))
