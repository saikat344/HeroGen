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

# Historical bootstrap and upgrade SQL is kept together, including the retirement migration.
from herogen.service.generation import SCHEMA as GENERATION_SCHEMA
from herogen.service.generation import MODEL_SPEND_SCHEMA
from herogen.service.analytics import SCHEMA as ANALYTICS_SCHEMA

ADMISSION_SCHEMA = SCHEMA
LEGACY_ADMISSION_SCHEMA = LEGACY_SCHEMA

SCHEMA_VERSION = 15
# Steps that bring an older database up to SCHEMA_VERSION, by the version they start from.
MIGRATIONS = {
    14: tuple(s.strip() for s in MODEL_SPEND_SCHEMA.split(';') if s.strip()),
    13: ("ALTER TABLE jobs ADD COLUMN build_all INTEGER NOT NULL DEFAULT 0",),
    12: ("ALTER TABLE jobs ADD COLUMN mesh_saved INTEGER NOT NULL DEFAULT 0",
         "ALTER TABLE jobs ADD COLUMN model_ready_at REAL",
         "ALTER TABLE jobs ADD COLUMN export_state TEXT NOT NULL DEFAULT 'pending'",
         "ALTER TABLE jobs ADD COLUMN export_message TEXT NOT NULL DEFAULT ''",
         "ALTER TABLE jobs ADD COLUMN export_started_at REAL",
         "ALTER TABLE jobs ADD COLUMN export_finished_at REAL",
         "ALTER TABLE jobs ADD COLUMN export_seconds REAL"),
    11: ("ALTER TABLE generation_settings ADD COLUMN spend_limits_enabled INTEGER NOT NULL DEFAULT 0",),
    6: ("ALTER TABLE admission_work ADD COLUMN process TEXT NOT NULL DEFAULT ''",
        next(s.strip() for s in LEGACY_ADMISSION_SCHEMA.split(';') if 'CREATE TABLE admission_calls' in s),
        next(s.strip() for s in LEGACY_ADMISSION_SCHEMA.split(';') if 'CREATE TABLE admission_processes' in s)),
    1: ("ALTER TABLE jobs ADD COLUMN refunded_at REAL",),
    2: ("ALTER TABLE job_stages ADD COLUMN claim TEXT", "ALTER TABLE job_stages ADD COLUMN lease_until REAL"),
    3: ("CREATE TABLE refine_budgets (job_id TEXT PRIMARY KEY, state TEXT NOT NULL)",),
    4: ("ALTER TABLE users ADD COLUMN kind TEXT NOT NULL DEFAULT 'account'",
        "ALTER TABLE jobs ADD COLUMN budget_required INTEGER NOT NULL DEFAULT 0",
        *tuple(s.strip() for s in LEGACY_ADMISSION_SCHEMA.replace(", process TEXT NOT NULL DEFAULT ''", '').split(';')
               if s.strip() and 'CREATE TABLE admission_calls' not in s and 'CREATE TABLE admission_processes' not in s)),
    5: ("UPDATE users SET kind='guest' WHERE kind='judge'",
        "DROP TABLE IF EXISTS judge_workspaces", "DROP TABLE IF EXISTS judge_codes",
        "ALTER TABLE admissions RENAME TO admissions_old",
        next(stmt.strip() for stmt in LEGACY_ADMISSION_SCHEMA.split(';') if 'CREATE TABLE admissions (' in stmt),
        "INSERT INTO admissions SELECT job_id,user_id,client_key,amount_micro,cost_signature,actual_micro,main_done,side_required,side_done,closed,created_at,evidence FROM admissions_old",
        "DROP TABLE admissions_old",
        next(stmt.strip() for stmt in LEGACY_ADMISSION_SCHEMA.split(';') if 'CREATE TABLE IF NOT EXISTS budget_overhead' in stmt)),
    7: ("ALTER TABLE jobs ADD COLUMN parent_job TEXT",
        "ALTER TABLE jobs ADD COLUMN from_stage TEXT",
        "ALTER TABLE jobs ADD COLUMN note TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE jobs ADD COLUMN stop_after TEXT",
        "ALTER TABLE jobs ADD COLUMN continued_after INTEGER NOT NULL DEFAULT 0"),
    8: PHASE_MIGRATION,
    10: (*tuple(f"ALTER TABLE {name} RENAME TO legacy_{name}" for name in (
        'admissions', 'admission_work', 'admission_calls', 'admission_processes',
        'admission_policy', 'budget_overhead', 'refine_budgets')),
        GENERATION_SCHEMA.strip(),
        "INSERT INTO generation_settings(id,max_running) VALUES (1, 5)"),
    9: tuple(s.strip() for s in ANALYTICS_SCHEMA.split(';') if s.strip()),
}
SCHEMA = """
CREATE TABLE users (
  id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, status TEXT NOT NULL DEFAULT 'active',
  created_at REAL NOT NULL, invite_id TEXT, terms_version TEXT, terms_accepted_at REAL, adult_confirmed_at REAL,
  quota_daily INTEGER, quota_total INTEGER, last_login_at REAL, deleted_at REAL, kind TEXT NOT NULL DEFAULT 'account'
);
CREATE TABLE allowlist (email TEXT PRIMARY KEY, added_at REAL NOT NULL, note TEXT NOT NULL DEFAULT '');
CREATE TABLE invites (
  id TEXT PRIMARY KEY, code_hash TEXT NOT NULL UNIQUE, created_at REAL NOT NULL, expires_at REAL NOT NULL,
  max_uses INTEGER NOT NULL, uses INTEGER NOT NULL DEFAULT 0, email TEXT, note TEXT NOT NULL DEFAULT '',
  revoked_at REAL
);
CREATE TABLE login_tokens (
  hash TEXT PRIMARY KEY, email TEXT NOT NULL, invite_id TEXT, created_at REAL NOT NULL, expires_at REAL NOT NULL,
  used_at REAL
);
CREATE INDEX login_tokens_email ON login_tokens(email);
CREATE TABLE sessions (
  hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL,
  last_seen_at REAL NOT NULL, revoked_at REAL
);
CREATE INDEX sessions_user ON sessions(user_id);
CREATE TABLE jobs (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, client_key TEXT NOT NULL, kind TEXT NOT NULL,
  input_sha TEXT NOT NULL DEFAULT '', prompt TEXT NOT NULL DEFAULT '', name TEXT NOT NULL DEFAULT '',
  profile TEXT NOT NULL, state TEXT NOT NULL, created_at REAL NOT NULL, dispatched_at REAL, call_id TEXT,
  started_at REAL, finished_at REAL, attempts INTEGER NOT NULL DEFAULT 0, lease_until REAL, stage TEXT,
  error_code TEXT, error_stage TEXT, error_message TEXT, counts_quota INTEGER NOT NULL DEFAULT 1,
  seconds REAL, cost_usd REAL, cost_detail TEXT, code_commit TEXT NOT NULL DEFAULT '', bytes INTEGER,
  continued_after INTEGER NOT NULL DEFAULT 0, parent_job TEXT, from_stage TEXT, note TEXT NOT NULL DEFAULT '', stop_after TEXT,
  build_all INTEGER NOT NULL DEFAULT 0,
  deleted_at REAL, refunded_at REAL, budget_required INTEGER NOT NULL DEFAULT 0, UNIQUE(user_id, client_key)
);
CREATE INDEX jobs_user ON jobs(user_id, created_at);
CREATE INDEX jobs_state ON jobs(state);
CREATE TABLE job_stages (
  job_id TEXT NOT NULL, stage TEXT NOT NULL, state TEXT NOT NULL, attempt INTEGER NOT NULL,
  started_at REAL, seconds REAL, cost_usd REAL, message TEXT NOT NULL DEFAULT '', claim TEXT, lease_until REAL,
  PRIMARY KEY (job_id, stage)
);
CREATE TABLE job_runs (
  job_id TEXT NOT NULL, attempt INTEGER NOT NULL, grp TEXT NOT NULL, entries INTEGER NOT NULL,
  PRIMARY KEY (job_id, attempt, grp)
);
CREATE TABLE refine_budgets (job_id TEXT PRIMARY KEY, state TEXT NOT NULL);
CREATE TABLE audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
  subject TEXT NOT NULL DEFAULT '', detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX audit_at ON audit(at);
"""


SCHEMA += ADMISSION_SCHEMA + ANALYTICS_SCHEMA + MODEL_SPEND_SCHEMA
