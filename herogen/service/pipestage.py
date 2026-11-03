"""Per-stage adapter: `python3 -m herogen.service.pipestage p|1|2 --run-id <id> --runs <root> --cache <cache>`.

Stage p reads the description as UTF-8 on stdin and calls the pipeline's stagep_prompt, which owns the prompt
versions, checks, images and stage-p.json. No user text or model error text is printed: API errors may echo
descriptions and appearance notes. Stage p exit codes: 0 passed, 1 permanent error, 2 argparse usage error, 3 input rejected,
4 content policy, 5 transient model error, 6 gate failed. Gate failures add status "gate_failed" to the pipeline
record; input rejection has status "input_rejected" and an error, policy refusal status "error" and an error.
The runner requires matching records for rejections. Exception messages never decide these codes. Cover and turnaround use the same isolated
output and structured exit codes because their prompts can include descriptions and appearance notes.

The service calls `stage2_turnaround` and `stage1_cover` directly through this adapter, so only the
requested image stage runs and writes its record.

`reword` (fast lane, after a content-policy refusal; `herogen.pipeline.reword`) reads the refused brief on stdin,
asks Muse Spark once for a rewrite and writes `reword.json` in the run folder: status "used" with the rewrite, or
"rewrite_failed" with the cost and the error class (an unusable rewrite is not kept). Exit 0 when the
rewrite can be used, 1 otherwise; it prints no user or model text either.
Stages 1/2 use the same exit codes as stage p except that gate failures use 2; the runner requires a matching
gate-failed record to distinguish them from usage errors. Argparse help exits 0 without running the pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def prompt_error_exit(error: BaseException | None) -> int:
    """Inspect explicit exception causes, including stage p's PromptStageError wrapper."""
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
    ap.add_argument("stage", choices=("p", "1", "2", "reword"))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--turnaround-concurrency", type=int, default=4)
    a = ap.parse_args(argv)
    from herogen.pipeline.client import ModelClient
    from herogen.pipeline.runfolder import RunFolder, runs_root
    from herogen.pipeline.prompt_sketch import stagep_prompt
    from herogen.pipeline.stages import GateFailed, InputRejected, stage1_cover, stage2_turnaround

    run = RunFolder(runs_root(a.runs), a.run_id)
    if a.stage == "reword":
        return reword_main(run, a.cache, sys.stdin.read())
    try:
        client = ModelClient(a.cache, mode="live")
        if a.stage == "p":
            rec = stagep_prompt(run, client, sys.stdin.read())
        elif a.stage == "1":
            rec = stage1_cover(run, client)
        else:
            rec = stage2_turnaround(run, client, concurrency=a.turnaround_concurrency)
    except GateFailed as e:
        run.write_stage({**e.record, "status": "gate_failed"})
        print(f"stage {a.stage}: gate failed; stopping", file=sys.stderr)
        return 6 if a.stage == "p" else 2
    except InputRejected:
        print(f"stage {a.stage}: input rejected; stopping", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001  (one line for the runner; the record holds the rest)
        print(f"stage {a.stage}: error; stopping", file=sys.stderr)
        return prompt_error_exit(e)
    print(f"stage {a.stage} {rec['name']}: ok ({rec['seconds']} s, ${rec['cost_usd']:.4f} est)")
    return 0


def reword_main(run, cache: Path, original: str) -> int:
    from herogen.pipeline import reword
    from herogen.pipeline.client import ApiError, ModelClient

    client = ModelClient(cache, mode="live")
    calls0 = len(client.log.calls)
    try:
        rec = reword.rewrite(client, original)
    except Exception as e:  # noqa: BLE001  (recorded as a failed rewrite; the job stops loudly on it)
        failed = {"status": reword.FAILED, "reason": reword.REASON, "original": original,
                  "error": type(e).__name__, "cost_usd": client.log.cost_usd(calls0)}
        if isinstance(e, ApiError):
            failed.update(http_status=e.http_status, error_code=e.error_code)
        reword.write(run.path, failed)
        print("reword: failed; stopping", file=sys.stderr)
        return 1
    reword.write(run.path, rec)
    print(f"reword: ok (${rec['cost_usd']:.4f} est)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
