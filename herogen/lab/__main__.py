"""CLI: `python3 -m herogen.lab serve [--runs runs] [--port 8790]`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from herogen.lab.server import make_server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m herogen.lab", description="HeroGen lab: view runs in a browser")
    sub = ap.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="serve the lab page on 127.0.0.1 (read-only)")
    serve.add_argument("--runs", default=os.environ.get("HEROGEN_RUNS", "runs"),
                       help="runs folder (default: $HEROGEN_RUNS or ./runs)")
    serve.add_argument("--port", type=int, default=8790)
    args = ap.parse_args(argv)

    root = Path(args.runs)
    if not root.is_dir():
        print(f"runs folder not found: {args.runs}", file=sys.stderr)
        return 2
    server = make_server(root, args.port)
    print(f"HeroGen lab on http://127.0.0.1:{server.server_address[1]}/ (runs: {args.runs}); Ctrl-C stops it",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
