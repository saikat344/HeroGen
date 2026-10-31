"""Per-stage adapter: `python3 -m herogen.service.pipestage p|2 --run-id <id> --runs <root> --cache <cache>`.

Stage p reads the description as UTF-8 on stdin and calls the pipeline's stagep_prompt, which owns the prompt
versions, checks, images and stage-p.json. No description or model error text is printed: API errors may echo
the description. The runner reads the record for costs and gates, and exit codes for errors: 3 input rejected,
4 content policy, 5 transient model error, 1 permanent error. Exception messages never decide these codes.

The pipeline's CLI always follows a passing stage 2 with the held-out views (2h). The service runs 2h beside the
rest of the job instead (see `worker.run_heldout`), so it starts stage 2 through the same public functions the
CLI's `run_pipeline` uses (`RunFolder`, `ModelClient` in live mode with the given cache, `stage2_turnaround`),
which write the same files and the same `stage-2.json`. Exit status: 0 passed, 2 the gate failed, 1 an error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def prompt_error_exit(error: BaseException | None) -> int:
    """Inspect explicit exception causes: stage p wraps client errors in PromptStageError."""
    from herogen.pipeline.client import ApiError

    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, ApiError):
            if error.error_code == "content_policy_violation":
                return 4
            return 5 if error.transient else 1
        error = error.__cause__
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m herogen.service.pipestage")
    ap.add_argument("stage", choices=("p", "2"))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cache", required=True, type=Path)
    a = ap.parse_args(argv)
    from herogen.pipeline.client import ModelClient
    from herogen.pipeline.runfolder import RunFolder, runs_root
    from herogen.pipeline.prompt_sketch import stagep_prompt
    from herogen.pipeline.stages import GateFailed, InputRejected, stage2_turnaround

    run = RunFolder(runs_root(a.runs), a.run_id)
    try:
        client = ModelClient(a.cache, mode="live")
        if a.stage == "p":
            rec = stagep_prompt(run, client, sys.stdin.read())
        else:
            rec = stage2_turnaround(run, client)
    except GateFailed as e:
        print(f"stage {a.stage}: gate failed; stopping" if a.stage == "p" else f"stage 2: {e}; stopping",
              file=sys.stderr)
        return 2
    except InputRejected:
        print(f"stage {a.stage}: input rejected; stopping", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001  (one line for the runner; the record holds the rest)
        print("stage p: error; stopping" if a.stage == "p" else f"stage 2: error: {type(e).__name__}: {e}; stopping",
              file=sys.stderr)
        return prompt_error_exit(e) if a.stage == "p" else 1
    print(f"stage {a.stage} {rec['name']}: ok ({rec['seconds']} s, ${rec['cost_usd']:.4f} est)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
