"""Immutable schema definitions used by historical store migrations."""
LEGACY_SCHEMA = '''
CREATE TABLE admission_policy (
 id INTEGER PRIMARY KEY CHECK(id=1), limit_micro INTEGER NOT NULL, reserve_micro INTEGER NOT NULL,
 max_active INTEGER NOT NULL, expires_at REAL NOT NULL, enabled INTEGER NOT NULL, code_commit TEXT NOT NULL,
 cost_signature TEXT NOT NULL, bound_note TEXT NOT NULL
);
CREATE TABLE admissions (
 job_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, client_key TEXT NOT NULL,
 amount_micro INTEGER NOT NULL, cost_signature TEXT NOT NULL, actual_micro INTEGER, main_done INTEGER NOT NULL DEFAULT 0,
 side_required INTEGER NOT NULL DEFAULT 1, side_done INTEGER NOT NULL DEFAULT 0,
 closed INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, evidence TEXT NOT NULL DEFAULT ''
);
CREATE TABLE admission_work (
 token TEXT PRIMARY KEY, job_id TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL,
 created_at REAL NOT NULL, finished_at REAL, process TEXT NOT NULL DEFAULT ''
);
CREATE INDEX admission_work_job ON admission_work(job_id);
CREATE TABLE admission_calls (id INTEGER PRIMARY KEY, job_id TEXT NOT NULL, token TEXT NOT NULL, receipt TEXT NOT NULL);
CREATE TABLE admission_processes (token TEXT NOT NULL, host TEXT NOT NULL, pid INTEGER NOT NULL,
 stopped INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(token,host,pid));
CREATE TABLE IF NOT EXISTS budget_overhead (id INTEGER PRIMARY KEY CHECK(id=1), amount_micro INTEGER NOT NULL, evidence TEXT NOT NULL, updated_at REAL NOT NULL);
'''
# Keep historical migration statements unchanged; fresh stores apply the same extension.
PHASE_MIGRATION = (
    "ALTER TABLE admissions ADD COLUMN phase INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE admissions ADD COLUMN baseline_micro INTEGER DEFAULT 0",
    "ALTER TABLE admissions ADD COLUMN settled_phase INTEGER NOT NULL DEFAULT -1",
    "ALTER TABLE admission_work ADD COLUMN phase INTEGER NOT NULL DEFAULT 0",
    "UPDATE admissions SET baseline_micro=actual_micro,settled_phase=0 WHERE actual_micro IS NOT NULL",
    # Older open continuations have no durable split between prior spend and new receipts.
    # Retain their allocation until operator evidence establishes a cumulative baseline.
    "UPDATE admissions SET baseline_micro=NULL WHERE actual_micro IS NULL AND job_id IN "
    "(SELECT id FROM jobs WHERE continued_after>0)",
)
SCHEMA = LEGACY_SCHEMA + ';\n'.join(PHASE_MIGRATION) + ';\n'
