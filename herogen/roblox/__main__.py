"""CLI: `python3 -m herogen.roblox export --body <run>/04-body --out <run>/06-roblox`
and `python3 -m herogen.roblox check --body <run>/04-body --rig <file.rbxmx>`.

Exit codes: 0 passed, 1 a gate check failed, 2 the body or a file couldn't be used (one `error:` line)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import RIG_FILE, RobloxExportError


def _line(c: dict) -> str:
    status = "PASS" if c["pass"] else ("FAIL" if c.get("gate", True) else "info")
    detail = f"  {c['detail']}" if c.get("detail") else ""
    return f"  {status}  {c['name']}: {c['value']} (threshold {c['threshold']}){detail}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m herogen.roblox")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write the R15 rig, Studio scripts and README for a stage-4 body")
    e.add_argument("--body", required=True, type=Path, help="the stage-4 folder (sections.json and the GLB)")
    e.add_argument("--out", required=True, type=Path, help="the folder to write, e.g. <run>/06-roblox")
    e.add_argument("--glb", default="high.glb", help="the body GLB inside --body (default high.glb)")
    c = sub.add_parser("check", help="check an .rbxmx rig against a stage-4 body")
    c.add_argument("--body", required=True, type=Path)
    c.add_argument("--rig", required=True, type=Path)
    c.add_argument("--glb", default="high.glb")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "export":
            from .export import record_path, run_export

            rec = run_export(a.body, a.out, a.glb)
            if "error" in rec:
                print(f"error: {rec['error']}", file=sys.stderr)
                return 2
            failed = [x for x in rec["checks"] if x.get("gate", True) and not x["pass"]]
            for x in failed:
                print(_line(x), file=sys.stderr)
            passed = sum(1 for x in rec["checks"] if x["pass"])
            print(f"stage 6 {'ok' if rec['ok'] else 'FAILED'}: {a.out / RIG_FILE}, HipHeight "
                  f"{rec['rig']['hip_height']:.4f}, {passed}/{len(rec['checks'])} checks passed, "
                  f"record {record_path(a.out)}")
            return 0 if rec["ok"] else 1
        from .check import check_rig

        checks = check_rig(a.body, a.rig, a.glb)
        for x in checks:
            print(_line(x))
        ok = all(x["pass"] for x in checks if x.get("gate", True))
        passed = sum(x["pass"] for x in checks)
        print(f"{a.rig}: {'PASS' if ok else 'FAIL'}, {passed}/{len(checks)} checks passed")
        return 0 if ok else 1
    except RobloxExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc.filename or 'a file'}: {exc.strerror or exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
