"""CLI: `python3 -m herogen.versions <freeze | list | verify | compare>`.

    freeze --run R --label L [--store DIR] [--body 04-body] [--sketch S] [--milestone M] [--note TEXT]
           [--by NAME] [--allow-dirty] [--allow-unknown-code]
    list [--store DIR] [--sketch S] [--json]
    verify --version V [--store DIR] [--json]
    compare A B [--store DIR] [--json]

The store is `--store` or `$HEROGEN_VERSIONS`. A version (`V`, `A`, `B`) is a version folder, `<sketch>/<label>`
in the store, or a label found in only one sketch folder. Exit status: 0 done, 1 `verify` found a problem,
2 refused or failed (one line on stderr, never a traceback).
"""
from __future__ import annotations

import argparse
import json
import sys

from .compare import compare, render_text, to_json
from .store import VersionError, freeze, list_versions, resolve_version, verify

PROG = "python3 -m herogen.versions"


class _Parser(argparse.ArgumentParser):
    def error(self, message):                                   # one line, not usage plus a line
        raise VersionError(f"{message} (see {PROG} --help)")


def _parser() -> argparse.ArgumentParser:
    ap = _Parser(prog=PROG, description="Freeze, list, verify and compare versioned bodies.")
    sub = ap.add_subparsers(dest="cmd", required=True, parser_class=_Parser)
    f = sub.add_parser("freeze", help="freeze a run's body as a new, write-once version")
    f.add_argument("--run", required=True, help="run folder")
    f.add_argument("--label", required=True, help="new version label, e.g. m5-wta")
    f.add_argument("--store", help="version store (default: $HEROGEN_VERSIONS)")
    f.add_argument("--body", default="04-body", help="body folder in the run (default 04-body)")
    f.add_argument("--sketch", help="sketch folder in the store (default: the one holding this sketch)")
    f.add_argument("--milestone", help="milestone the version belongs to")
    f.add_argument("--note", help="free text kept in version.json")
    f.add_argument("--by", help="who froze it (default: the login name)")
    f.add_argument("--allow-dirty", action="store_true",
                   help="freeze even though the records say uncommitted code built or scored the run")
    f.add_argument("--allow-unknown-code", action="store_true",
                   help="freeze even though no record names the commit that built the body")
    ls = sub.add_parser("list", help="list the versions in the store")
    ls.add_argument("--store")
    ls.add_argument("--sketch")
    ls.add_argument("--json", action="store_true")
    v = sub.add_parser("verify", help="re-hash every file of a version")
    v.add_argument("--version", required=True, dest="ref")
    v.add_argument("--store")
    v.add_argument("--json", action="store_true")
    c = sub.add_parser("compare", help="compare two versions (deltas are B - A)")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--store")
    c.add_argument("--json", action="store_true")
    return ap


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def _run(a) -> int:
    if a.cmd == "freeze":
        meta = freeze(a.run, a.label, a.store, body=a.body, sketch=a.sketch, milestone=a.milestone, note=a.note,
                      by=a.by, allow_dirty=a.allow_dirty, allow_unknown_code=a.allow_unknown_code)
        code = meta["code"]
        state = {True: "dirty (allowed)", False: "clean", None: "dirty unknown"}
        code_s = ("code unknown (allowed)" if code is None or code["commit"] is None else
                  f"code {code['commit'][:12]} {state[code['dirty']]}")
        miss = meta["missing"]["files"] + meta["missing"]["fields"]
        print(f"frozen {meta['sketch']}/{meta['label']}: {len(meta['files_sha256'])} files, "
              f"{_mb(sum(meta['files_bytes'].values()))}, {code_s}; missing: {', '.join(miss) or 'nothing'}")
        for w in meta["warnings"]:
            print(f"warning: {w}", file=sys.stderr)
        return 0
    if a.cmd == "list":
        res = list_versions(a.store, a.sketch)
        if a.json:
            sys.stdout.write(json.dumps(res, indent=2) + "\n")
            return 0
        rows = []
        for s in res["versions"]:
            commit = (s["commit"][:12] + {True: " dirty", False: "", None: " ?"}[s["dirty"]]) if s["commit"] \
                else "not stamped"
            rows.append([f"{s['sketch']}/{s['label']}", str(s["milestone"] or ""), str(s["frozen"] or ""),
                         str(s["profile"] or "missing"), str(s["mesh_source"] or "missing"),
                         str(s["joints_source"] or "missing"), commit, str(s["files"])])
        if rows:
            head = ["version", "milestone", "frozen", "profile", "mesh", "joints", "code", "files"]
            widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
            for r in [head, *rows]:
                print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
        else:
            print(f"no versions in {res['store']}" + (f" for sketch {a.sketch}" if a.sketch else ""))
        for s in res["versions"]:
            if s["label_recorded"]:
                print(f"label mismatch: {s['sketch']}/{s['label']} has version.json label "
                      f"{s['label_recorded']!r}")
        for u in res["unfinished"]:
            print(f"unfinished freeze (not a version; safe to delete): {u}")
        for c in res["empty_claims"]:
            print(f"empty claim from an interrupted freeze (not a version; safe to remove): {c}")
        for b in res["not_versions"]:
            print(f"not a version (no readable version.json): {b}")
        return 0
    if a.cmd == "verify":
        rep = verify(resolve_version(a.ref, a.store))
        if a.json:
            sys.stdout.write(json.dumps(rep, indent=2) + "\n")
        else:
            for m in rep["mismatched"]:
                print(f"changed: {m['file']} (recorded {m['recorded'][:12]}, found {m['found'][:12]})")
            for n in rep["missing"]:
                print(f"missing: {n}")
            for n in rep["not_regular"]:
                print(f"not a regular file: {n}")
            for u in rep["unreadable"]:
                print(f"unreadable: {u['file']} ({u['error']})")
            for n in rep["unlisted"]:
                print(f"not in version.json: {n}")
            for n in rep["unsafe"]:
                print(f"unsafe file name in version.json: {n!r}")
            if rep["label_recorded"]:
                print(f"label: version.json says {rep['label_recorded']!r}, the folder is {rep['version']}")
            print(f"{rep['version']}: " + (f"ok, {rep['checked']} files match their SHA-256" if rep["ok"]
                                           else "FAILED"))
        return 0 if rep["ok"] else 1
    if a.cmd == "compare":
        res = compare(resolve_version(a.a, a.store), resolve_version(a.b, a.store))
        sys.stdout.write(to_json(res) if a.json else render_text(res))
        return 0
    raise VersionError(f"unknown command {a.cmd}")


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(_parser().parse_args(argv))
    except VersionError as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print(f"{PROG}: interrupted", file=sys.stderr)
    except OSError as exc:
        where = f" ({exc.filename})" if getattr(exc, "filename", None) else ""
        print(f"{PROG}: {exc.strerror or exc.__class__.__name__}{where}", file=sys.stderr)
    except Exception as exc:                                     # never a traceback
        print(f"{PROG}: unexpected {exc.__class__.__name__}: {exc}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
