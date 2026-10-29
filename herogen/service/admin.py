"""Admin commands: invites, allowlist, users, quotas, jobs and the audit log.

Run against the dev store with `python3 -m herogen.service admin --data <dir> <command>`, or against the deployed
service with `modal run -m herogen.service.modal_app::admin --args "<command>"` from the repository root. That runs
the command in the `admin_command` function, the only function besides the store that holds the admin key the
store's admin methods require, and the signing key, so invite codes are hashed with the server's key and neither
key leaves Modal. Invite codes are printed once and only their keyed hashes are stored; the audit log records
allowlist changes by a keyed digest of the address, never the address.
"""

from __future__ import annotations

import argparse
import json
import shlex
import time
from typing import Any

from herogen.service.auth import Signer, new_invite_code, normalize_email, normalize_invite


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="admin", description="HeroGen service administration")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("invite", help="create invite codes (printed once)")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--uses", type=int, default=1, help="sign-ups each code allows")
    p.add_argument("--days", type=float, default=14.0, help="days until the code expires")
    p.add_argument("--email", default=None, help="only this address may use the code")
    p.add_argument("--note", default="")
    sub.add_parser("invites", help="list invites (codes are not stored, only their hashes)")
    p = sub.add_parser("revoke-invite")
    p.add_argument("invite_id")
    p = sub.add_parser("allow", help="let an address create an account without a code")
    p.add_argument("email")
    p.add_argument("--note", default="")
    p = sub.add_parser("disallow")
    p.add_argument("email")
    sub.add_parser("allowlist")
    sub.add_parser("users")
    for name in ("disable", "enable"):
        p = sub.add_parser(name)
        p.add_argument("email")
    p = sub.add_parser("quota", help="set a user's quota; 'default' uses the service setting")
    p.add_argument("email")
    p.add_argument("--daily", default="default")
    p.add_argument("--total", default="default")
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
    return ap


def _email(raw: str) -> str:
    e = normalize_email(raw)
    if e is None:
        raise SystemExit(f"not a valid email address: {raw!r}")
    return e


def _limit(raw: str) -> int | None:
    if raw == "default":
        return None
    n = int(raw)
    if n < 0:
        raise SystemExit("a quota cannot be negative")
    return n


def _when(t: Any) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(t)) + "Z" if isinstance(t, (int, float)) else "-"


def run(argv: list[str] | str, store, signer: Signer, snapshots: tuple | None = None) -> str:
    """Run one admin command and return what it prints. `snapshots` is (folder, volume) of the metadata store's
    snapshots on Modal; the snapshot commands need it and never touch `store`, so they work while the store
    cannot start."""
    a = parser().parse_args(shlex.split(argv) if isinstance(argv, str) else argv)
    out: list[str] = []
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
    if a.cmd == "invite":
        if not 1 <= a.count <= 100:
            raise SystemExit("--count must be 1 to 100")
        email = _email(a.email) if a.email else None
        for _ in range(a.count):
            code = new_invite_code()
            inv = store.invite_create(signer.hash("invite", normalize_invite(code)), a.uses, a.days * 86400.0, email,
                                      a.note)
            out.append(f"{code}  (id {inv['id']}, {a.uses} use(s), expires {_when(inv['expires_at'])}"
                       + (f", only {email}" if email else "") + ")")
    elif a.cmd == "invites":
        for i in store.invite_list():
            state = "revoked" if i["revoked_at"] else "expired" if i["expires_at"] < time.time() else "open"
            out.append(f"{i['id']}  {i['uses']}/{i['max_uses']} used  {state}  expires {_when(i['expires_at'])}  "
                       f"{i['email'] or ''}  {i['note']}".rstrip())
    elif a.cmd == "revoke-invite":
        out.append("revoked" if store.invite_revoke(a.invite_id) else "no open invite with that id")
    elif a.cmd == "allow":
        store.allow_add(_email(a.email), a.note, signer.tag(_email(a.email)))
        out.append(f"allowed {_email(a.email)}")
    elif a.cmd == "disallow":
        out.append("removed" if store.allow_remove(_email(a.email), signer.tag(_email(a.email)))
                   else "not on the allowlist")
    elif a.cmd == "allowlist":
        out += [f"{r['email']}  added {_when(r['added_at'])}  {r['note']}".rstrip() for r in store.allow_list()]
    elif a.cmd == "users":
        for u in store.users_list():
            q = f"quota {u['quota_daily'] if u['quota_daily'] is not None else 'default'}/day " \
                f"{u['quota_total'] if u['quota_total'] is not None else 'default'} total"
            out.append(f"{u['id']}  {u['email']}  {u['status']}  {u['jobs']} jobs  {q}  last sign-in "
                       f"{_when(u['last_login_at'])}")
    elif a.cmd in ("disable", "enable"):
        u = store.user_set_status(_email(a.email), "disabled" if a.cmd == "disable" else "active")
        out.append(f"{u['email']} is {u['status']}")
    elif a.cmd == "quota":
        u = store.user_set_quota(_email(a.email), _limit(a.daily), _limit(a.total))
        out.append(json.dumps({"email": u["email"], "quota_daily": u["quota_daily"], "quota_total": u["quota_total"]}))
    elif a.cmd == "jobs":
        for j in store.jobs_admin_list(a.limit, a.state):
            out.append(f"{j['id']}  {j['user_id']}  {j['state']:<9}  stage {j['stage'] or '-'}  attempts "
                       f"{j['attempts']}  {_when(j['created_at'])}  cost {j['cost_usd'] if j['cost_usd'] is not None else '-'}"
                       f"  {j['error_code'] or ''}".rstrip())
    elif a.cmd == "audit":
        for e in store.audit_list(a.limit):
            out.append(f"{_when(e['at'])}  {e['actor']}  {e['action']}  {e['subject']}  {e['detail']}".rstrip())
    return "\n".join(out)
