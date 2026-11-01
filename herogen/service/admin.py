"""Operational administration and live-generation settings for local and hosted stores."""

from __future__ import annotations

import argparse
import json
import shlex
import time
from typing import Any

from herogen.service.auth import Signer, normalize_email


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="admin", description="HeroGen service administration")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("users")
    for name in ("disable", "enable"):
        p = sub.add_parser(name)
        p.add_argument("email")
    p = sub.add_parser("jobs")
    p.add_argument("--state", default=None)
    p.add_argument("--limit", type=int, default=50)
    p = sub.add_parser("audit")
    p.add_argument("--limit", type=int, default=100)
    sub.add_parser("snapshots", help="list the metadata store's snapshot, quarantine and lease files")
    p = sub.add_parser("snapshot-quarantine",
                       help="move a snapshot the store cannot read to quarantine, so it starts from the next one")
    p.add_argument("name", help="the snapshot file name, as `snapshots` lists it")
    p.add_argument("--allow-empty", action="store_true",
                   help="allow quarantining the last snapshot; the store then starts from an empty database")
    p = sub.add_parser('generation-settings', help='show or update live-generation settings')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--enable', action='store_true')
    mode.add_argument('--pause', action='store_true')
    p.add_argument('--limit-usd')
    p.add_argument('--running-estimate-usd')
    p.add_argument('--max-running', type=int)
    p = sub.add_parser('analytics-report', help='print first-party analytics and pipeline report as JSON')
    p.add_argument('--days', type=int, default=7, help='window in days (1 to 90)')
    return ap


def _email(raw: str) -> str:
    e = normalize_email(raw)
    if e is None:
        raise SystemExit(f"not a valid email address: {raw!r}")
    return e


def _when(t: Any) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(t)) + "Z" if isinstance(t, (int, float)) else "-"


def run(argv: list[str] | str, store, signer: Signer, snapshots: tuple | None = None) -> str:
    """Run one admin command and return what it prints. `snapshots` is (folder, volume) of the metadata store's
    snapshots on Modal; the snapshot commands need it and never touch `store`, so they work while the store
    cannot start."""
    a = parser().parse_args(shlex.split(argv) if isinstance(argv, str) else argv)
    out: list[str] = []
    if a.cmd == 'generation-settings':
        if a.enable or a.pause or any(v is not None for v in (a.limit_usd, a.running_estimate_usd, a.max_running)):
            result = store.generation_update(actor='admin', enabled=True if a.enable else False if a.pause else None,
                limit_usd=a.limit_usd, running_estimate_usd=a.running_estimate_usd, max_running=a.max_running)
        else:
            result = store.generation_get()
        return json.dumps(result, indent=2)
    if a.cmd in ("snapshots", "snapshot-quarantine"):
        from herogen.service.snapshot import quarantine_snapshot, snapshot_files

        if snapshots is None:
            return "This store keeps no snapshots (dev mode writes its database directly; snapshots exist on Modal)."
        folder, volume = snapshots
        if a.cmd == "snapshots":
            return "\n".join(snapshot_files(folder, volume)) or "no snapshot files"
        try:
            res = quarantine_snapshot(folder, volume, a.name, allow_empty=a.allow_empty)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        return (f"quarantined {res['quarantined']} as {res['as']} (lease {res['lease']}); "
                + (f"the store starts from {max(res['remaining'])}" if res["remaining"]
                   else "the store may start from an empty database"))
    if a.cmd == "users":
        for u in store.users_list():
            out.append(f"{u['id']}  {u['status']}  {u['jobs']} jobs")
    elif a.cmd in ("disable", "enable"):
        u = store.user_set_status(_email(a.email), "disabled" if a.cmd == "disable" else "active")
        out.append(f"{u['email']} is {u['status']}")
    elif a.cmd == "jobs":
        for j in store.jobs_admin_list(a.limit, a.state):
            out.append(f"{j['id']}  {j['user_id']}  {j['state']:<9}  stage {j['stage'] or '-'}  attempts "
                       f"{j['attempts']}  {_when(j['created_at'])}  cost {j['cost_usd'] if j['cost_usd'] is not None else '-'}"
                       f"  {j['error_code'] or ''}".rstrip())
    elif a.cmd == "audit":
        for e in store.audit_list(a.limit):
            out.append(f"{_when(e['at'])}  {e['actor']}  {e['action']}  {e['subject']}  {e['detail']}".rstrip())
    elif a.cmd == "analytics-report":
        from herogen.service.analytics import MAX_DAYS, MIN_DAYS
        if not (MIN_DAYS <= a.days <= MAX_DAYS):
            raise SystemExit(f"--days must be {MIN_DAYS} to {MAX_DAYS}")
        return json.dumps(store.analytics_report(a.days), indent=2)
    return "\n".join(out)
