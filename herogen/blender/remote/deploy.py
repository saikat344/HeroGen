"""Deploy the SAM 3D GPU runners as persistent Modal apps with their fixed capacity.

`python -m herogen.blender.remote.deploy [objects] [body]` (both when none is named) runs
`modal deploy herogen/blender/remote/<runner>.py` for each runner. The deployed apps are what
`herogen.blender.remote.prewarm` warms and what `herogen.service.modal_bridge` calls
(`HEROGEN_REMOTE_DEPLOYED=1` for local runs).

The runners read their capacity from the environment when Modal loads them; this command sets it from `CAPACITY`
and ignores the caller's values, so a deploy cannot pick up a stray limit. The GPU type is each runner's own
default (the bridge refuses a request for another GPU). `--dry-run` prints the commands without deploying.
The runner files are not edited here: their source hash is part of every cached result's key.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Containers each deployed runner may run at once, and how many it keeps warm (0: only warmed on request).
# Objects is fixed at 5 so up to five heroes build their 3D shape at the same time; change it only with the
# project owner's approval.
CAPACITY = {"objects": {"MAX_CONTAINERS": 5, "MIN_CONTAINERS": 0},
            "body": {"MAX_CONTAINERS": 5, "MIN_CONTAINERS": 0}}
# Settings a deploy never takes from the caller: capacity (above) and a GPU other than the runner's default.
CLEARED = ("GPU", "MAX_CONTAINERS", "MIN_CONTAINERS", "APP")


def environment(runner: str, base: dict[str, str] | None = None) -> dict[str, str]:
    """The deploy's environment: `base` without the runner's cleared settings, plus its fixed capacity."""
    prefix = f"HEROGEN_{runner.upper()}_"
    env = {k: v for k, v in (os.environ if base is None else base).items()
           if not any(k == prefix + name for name in CLEARED)}
    env.update({prefix + name: str(value) for name, value in CAPACITY[runner].items()})
    return env


def command(runner: str, modal: str) -> list[str]:
    return [modal, "deploy", str(HERE / f"{runner}.py")]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runners", nargs="*", choices=sorted(CAPACITY), help="runners to deploy (default: all)")
    p.add_argument("--dry-run", action="store_true", help="print the commands and settings without deploying")
    a = p.parse_args(argv)
    modal = os.environ.get("HEROGEN_MODAL") or shutil.which("modal")
    if modal is None and not a.dry_run:
        print("modal CLI not found (set HEROGEN_MODAL)", file=sys.stderr)
        return 1
    for runner in a.runners or sorted(CAPACITY):
        env = environment(runner)
        settings = " ".join(f"HEROGEN_{runner.upper()}_{n}={v}" for n, v in CAPACITY[runner].items())
        cmd = command(runner, modal or "modal")
        print(f"{settings} {' '.join(cmd)}")
        if not a.dry_run:
            rc = subprocess.run(cmd, env=env).returncode
            if rc:
                print(f"{runner}: modal deploy failed with exit code {rc}", file=sys.stderr)
                return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
