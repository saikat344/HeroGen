"""Single-pass fast-lane construction and best-model completion after local export failures."""
import json
import shutil
import sys
import time

from herogen.pipeline.client import ModelClient
from herogen.pipeline.fast import informational
from herogen.pipeline.runfolder import RunFolder

from .final_model import final_model
from .record import StageRecord
from .stage3 import error_text


def run(a, stages):
    from .remote.client import DEFAULT_BODY_GPU, DEFAULT_OBJECTS_GPU, RemoteClient
    from .stage3 import DEFAULT_NZ, run_stage3
    from .refine import run_stage3f

    folder = RunFolder(a.run.parent, a.run.name)
    if "3" in stages:
        remote = RemoteClient(a.remote_cache or a.run.parent / "_cache-remote", mode=a.remote,
                              timeout=a.remote_timeout, body_gpu=a.body_gpu or DEFAULT_BODY_GPU,
                              objects_gpu=a.objects_gpu or DEFAULT_OBJECTS_GPU, fast=True)
        try:
            rec = run_stage3(a.run, a.height, a.voxels or DEFAULT_NZ, timeout=a.timeout,
                             mesh_source=a.mesh_source, joints_source=a.joints, remote=remote,
                             fallback=False, y_method=a.y_method, limb_roots=a.limb_roots,
                             body_detector=None if a.body_detector == "none" else a.body_detector, fast=True)
            print(f"stage 3: complete ({rec['seconds']} s)")
        except Exception as exc:
            print(f"stage 3 stopped: {error_text(exc, a.run)}", file=sys.stderr)
            return 1
    if "3f" in stages:
        try:
            rec = run_stage3f(a.run, ModelClient(a.refine_cache or a.run.parent / "_cache-model",
                                                mode=a.refine, fast=True),
                              profile=a.profile, timeout=a.timeout, cap_usd=None,
                              refine_profile="fast", recipe=a.refine_recipe, user_note=a.refine_note)
            print(f"stage 3f: {rec['status']} ({rec['seconds']} s)")
        except Exception as exc:
            print(f"stage 3f stopped: {error_text(exc, a.run)}", file=sys.stderr)
            return 1
    from .stage4 import run_stage4
    from herogen.roblox.export import run_export
    from herogen.pipeline.results import stage5_results

    steps = [("4", "game body", lambda: run_stage4(a.run, a.height, timeout=a.timeout,
                                                   profile=a.profile, cut_workers=a.cut_workers, fast=True)),
             ("6", "roblox export", lambda: run_export(a.run / "04-body", a.run / "06-roblox")),
             ("5", "results", lambda: stage5_results(folder))]
    failed = []
    for stage, name, fn in steps:
        if stage not in stages:
            continue
        receipt = StageRecord(a.run, int(stage), name)
        (a.run / f"stage-{stage}.json").unlink(missing_ok=True)
        try:
            if stage == "4" and (a.run / "04-body").exists():
                shutil.rmtree(a.run / "04-body")
            fn()
            rec = folder.read_stage(stage)
            if rec is None:
                raise ValueError(f"stage {stage} did not write its record")
            if stage != "4":
                informational(rec.get("checks", []))
            rec.update(fast_lane=True, ok=not bool(rec.get("error"))
                       and (rec.get("ok", False) if stage == "4" else True))
            if rec.get("error"):
                rec["status"] = "error"
        except Exception as exc:
            rec = folder.read_stage(stage) or receipt.data
            rec.update(ok=False, status="error", error=error_text(exc, a.run), fast_lane=True)
            rec["seconds"] = round(time.monotonic() - receipt._t0, 2)
        best = final_model(a.run, "best")
        rec["final_model"] = best.relative_to(a.run).as_posix() if best else None
        folder.write_stage(rec)
        if not rec.get("ok"):
            failed.append(stage)
        if rec.get("error"):
            print(f"stage {stage} failed: {rec['error']}; keeping {rec['final_model']}", file=sys.stderr)
        else:
            print(f"stage {stage}: complete ({rec.get('seconds')} s)")
    best = final_model(a.run, "best")
    print(json.dumps({"fast_lane": True, "final_model": best.relative_to(a.run).as_posix() if best else None,
                      "failed_stages": failed}))
    # Exit 0 whenever a model exists (the result is always delivered); failed_stages names this call's failures.
    return 0 if best else 1
