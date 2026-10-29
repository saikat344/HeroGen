"""CLI: `python3 -m herogen.blender run --run <folder> [--stages 3,4]` and `... synth --run <folder>`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import HEIGHT_STUDS


def _upstream_problem(run: Path, stage: int) -> str | None:
    """Why `stage` doesn't count as passed in `run`, or None when its record says ok."""
    path = run / f"stage-{stage}.json"
    if not path.exists():
        return f"{path.name} not found"
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return f"{path.name} unreadable ({exc.__class__.__name__})"
    return None if rec.get("ok") is True else f"{path.name} has ok={rec.get('ok')!r}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m herogen.blender")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run stages 3 and/or 4 on a run folder")
    r.add_argument("--run", required=True, type=Path)
    r.add_argument("--stages", default="3,4")
    r.add_argument("--height", type=float, default=HEIGHT_STUDS, help="body height in studs")
    r.add_argument("--voxels", type=int, default=None, help="voxel grid resolution along the height")
    r.add_argument("--timeout", type=float, default=900.0, help="seconds allowed per Blender call")
    r.add_argument("--force", action="store_true", help="run even if the previous stage did not pass")
    s = sub.add_parser("synth", help="write a synthetic A-pose humanoid's masks (stage-2 output) to a run folder")
    s.add_argument("--run", required=True, type=Path)
    s.add_argument("--size", type=int, default=512)
    s.add_argument("--arm-angle", type=float, default=50.0)
    s.add_argument("--elbow-bend", type=float, default=0.0)
    s.add_argument("--head-scale", type=float, default=1.0)
    s.add_argument("--clothed", action="store_true", help="add hair, neck scarf, backpack and scarf strip")
    a = ap.parse_args(argv)
    if a.cmd == "synth":
        from . import synth

        fig = synth.humanoid(arm_angle_deg=a.arm_angle, elbow_bend_deg=a.elbow_bend, head_scale=a.head_scale,
                             clothed=a.clothed)
        synth.write_run(a.run, fig, size=a.size, ppu=a.size * 80.0 / 512)
        print(f"synthetic masks written to {a.run}")
        return 0
    stages = {int(x) for x in a.stages.split(",") if x.strip()}
    if not stages:
        ap.error("--stages names no stage (use 3, 4 or 3,4)")
    if not stages <= {3, 4}:
        ap.error("this package runs stages 3 and 4 only")
    first = min(stages)
    problem = _upstream_problem(a.run, first - 1)
    if problem and not a.force:
        print(f"stage {first} needs a passed stage {first - 1}: {problem} (use --force to run anyway)",
              file=sys.stderr)
        return 2
    ok = True
    if 3 in stages:
        from .stage3 import DEFAULT_NZ, run_stage3

        rec = run_stage3(a.run, a.height, a.voxels or DEFAULT_NZ, timeout=a.timeout)
        print(json.dumps({"stage": 3, "ok": rec["ok"], "seconds": rec["seconds"],
                          "checks": {c["name"]: c["value"] for c in rec["checks"]}}))
        ok &= rec["ok"]
        if not rec["ok"]:
            print("stage 3 gate failed; stopping", file=sys.stderr)
            return 1
    if 4 in stages:
        from .stage4 import run_stage4

        rec = run_stage4(a.run, a.height, timeout=a.timeout)
        print(json.dumps({"stage": 4, "ok": rec["ok"], "seconds": rec["seconds"],
                          "checks": {c["name"]: c["value"] for c in rec["checks"]}}))
        ok &= rec["ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
