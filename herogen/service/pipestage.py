"""Stage 2 (turnaround) alone: `python3 -m herogen.service.pipestage 2 --run-id <id> --runs <root> --cache <cache>`.

The pipeline's CLI always follows a passing stage 2 with the held-out views (2h). The service runs 2h beside the
rest of the job instead (see `worker.run_heldout`), so it starts stage 2 through the same public functions the
CLI's `run_pipeline` uses (`RunFolder`, `ModelClient` in live mode with the given cache, `stage2_turnaround`),
which write the same files and the same `stage-2.json`. Exit status: 0 passed, 2 the gate failed, 1 an error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m herogen.service.pipestage")
    ap.add_argument("stage", choices=("2",))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cache", required=True, type=Path)
    a = ap.parse_args(argv)
    from herogen.pipeline.client import ModelClient
    from herogen.pipeline.runfolder import RunFolder, runs_root
    from herogen.pipeline.stages import GateFailed, stage2_turnaround

    run = RunFolder(runs_root(a.runs), a.run_id)
    try:
        rec = stage2_turnaround(run, ModelClient(a.cache, mode="live"))
    except GateFailed as e:
        print(f"stage 2: {e}; stopping", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001  (one line for the runner; the record holds the rest)
        print(f"stage 2: error: {type(e).__name__}: {e}; stopping", file=sys.stderr)
        return 1
    print(f"stage 2 {rec['name']}: ok ({rec['seconds']} s, ${rec['cost_usd']:.4f} est)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
