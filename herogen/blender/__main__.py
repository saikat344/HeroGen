"""CLI: `python3 -m herogen.blender run --run <folder> [--stages 3,4] [--mesh-source ...] [--joints ...]` and
`... synth --run <folder>`."""
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
    if rec.get("ok") is not True:
        return f"{path.name} has ok={rec.get('ok')!r}"
    if stage == 2:  # an earlier version could pass stage 2 on background-threshold masks: not a measurement
        methods = (rec.get("details") or {}).get("mask_method") or {}
        fallback = sorted(v for v, m in methods.items() if str(m).startswith("threshold"))
        if fallback:
            return f"{path.name} passed on threshold masks ({', '.join(fallback)}); run stage 2 again"
    return None


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
    r.add_argument("--mesh-source", choices=("carve", "sam3d-objects"), default="carve",
                   help="stage 3 mesh: silhouette carving (default) or SAM 3D Objects from the front view")
    r.add_argument("--joints", choices=("silhouette", "sam3d-body"), default="silhouette",
                   help="joints: silhouette finder (default) or SAM 3D Body on the front view")
    r.add_argument("--y-method", choices=("mesh", "mhr3d"), default="mesh", help="sam3d-body joint depth")
    r.add_argument("--limb-roots", choices=("silhouette", "sam3d-body"), default="silhouette",
                   help="sam3d-body joints: where shoulders and hips are cut (see body_joints)")
    r.add_argument("--body-detector", choices=("none", "vitdet"), default="none",
                   help="sam3d-body joints: person box from the front mask (default) or SAM 3D Body's ViTDet "
                        "detector (weights under CC BY-SA 3.0)")
    r.add_argument("--remote", choices=("replay", "live"), default="replay",
                   help="remote models: replay the cache only (default) or call Modal and record")
    r.add_argument("--remote-cache", type=Path, default=None, help="default: <run parent>/_cache-remote")
    r.add_argument("--remote-timeout", type=float, default=1800.0, help="seconds allowed per Modal call")
    r.add_argument("--objects-gpu", default=None, help="Modal GPU for SAM 3D Objects (default A100-40GB)")
    r.add_argument("--body-gpu", default=None, help="Modal GPU for SAM 3D Body (default A10G; none = CPU)")
    r.add_argument("--no-fallback", action="store_true", help="fail instead of falling back to carve/silhouette")
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
    from .bl import BlenderError

    if 3 in stages:
        from .remote.client import DEFAULT_BODY_GPU, DEFAULT_OBJECTS_GPU, RemoteClient, RemoteError
        from .stage3 import DEFAULT_NZ, error_text, run_stage3

        remote = RemoteClient(a.remote_cache or a.run.parent / "_cache-remote", mode=a.remote,
                              timeout=a.remote_timeout, body_gpu=a.body_gpu or DEFAULT_BODY_GPU,
                              objects_gpu=a.objects_gpu or DEFAULT_OBJECTS_GPU)
        try:
            rec = run_stage3(a.run, a.height, a.voxels or DEFAULT_NZ, timeout=a.timeout,
                             mesh_source=a.mesh_source, joints_source=a.joints, remote=remote,
                             fallback=not a.no_fallback, y_method=a.y_method, limb_roots=a.limb_roots,
                             body_detector=None if a.body_detector == "none" else a.body_detector)
        except (RemoteError, BlenderError, ModuleNotFoundError, ValueError, OSError) as exc:
            print(f"stage 3 stopped: {error_text(exc, a.run)}", file=sys.stderr)
            return 1
        print(json.dumps({"stage": 3, "ok": rec["ok"], "seconds": rec["seconds"],
                          "mesh_source": rec.get("mesh_source"), "joints_source": rec.get("joints_source"),
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
