"""CLI: `python3 -m herogen.lab serve [--runs DIR ...] [--versions DIR] [--port 8790]`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from herogen.lab.server import make_server
from herogen.lab.workspace import Workspace


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m herogen.lab", description="HeroGen lab: view runs in a browser")
    sub = ap.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="serve the lab page on 127.0.0.1 (read-only)")
    serve.add_argument("--runs", action="append", metavar="DIR",
                       help="a runs folder; repeat it to read several (default: $HEROGEN_RUNS or ./runs)")
    serve.add_argument("--versions", metavar="DIR", default=os.environ.get("HEROGEN_VERSIONS"),
                       help="a folder of frozen versions (default: $HEROGEN_VERSIONS; none when unset)")
    serve.add_argument("--port", type=int, default=8790, help="port on 127.0.0.1 (0 picks a free one)")
    args = ap.parse_args(argv)

    roots = args.runs or [os.environ.get("HEROGEN_RUNS", "runs")]
    missing = [r for r in roots if not Path(r).is_dir()]
    if missing:
        print(f"runs folder not found: {', '.join(missing)}", file=sys.stderr)
        return 2
    if len(set(Path(r).resolve() for r in roots)) != len(roots):
        print("the same runs folder is given twice", file=sys.stderr)
        return 2
    if args.versions and not Path(args.versions).is_dir():
        print(f"versions folder not found: {args.versions}", file=sys.stderr)
        return 2
    ws = Workspace([Path(r) for r in roots], Path(args.versions) if args.versions else None, labels=roots)
    try:
        server = make_server(ws, args.port)
    except OSError as e:
        print(f"cannot listen on 127.0.0.1:{args.port}: {e.strerror or e}", file=sys.stderr)
        return 2
    where = "; ".join(f"{k} = {ws.labels[k]}" for k, _ in ws.roots)
    print(f"HeroGen lab on http://127.0.0.1:{server.server_address[1]}/ (runs: {where}"
          f"{'; versions: ' + args.versions if args.versions else ''}); Ctrl-C stops it", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
