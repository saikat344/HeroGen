"""Stage 3's remote models from inside a Modal worker: a stand-in for the `modal run` command.

herogen.blender's remote client runs SAM 3D Objects and SAM 3D Body by starting
`<modal> run <runner>.py::main --job <job.json> --out <dir>` (an ephemeral app per call), where `<modal>` is
`$HEROGEN_MODAL`. A Modal worker has no Modal CLI login, so the Blender worker image points `HEROGEN_MODAL` at
this bridge instead. It takes the same arguments, calls the runner's `infer` function in the runner's deployed
app (`modal deploy herogen/blender/remote/objects.py`, `.../body.py`; the app name is the runner's `APP`
constant) and writes the same files the runner's own `main` writes: `<out>/<name>/<file>`, `<out>/<name>/run.json`
and `<out>/run.json`. The client's cache, cost estimate and error handling are unchanged.

`python3 -m herogen.service.modal_bridge run <runner.py>::main --job <job.json> --out <dir>`
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

GPU_ENV = {"objects": "HEROGEN_OBJECTS_GPU", "body": "HEROGEN_BODY_GPU"}


class BridgeError(RuntimeError):
    pass


def runner_constants(path: Path) -> dict:
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def payload(runner: str, spec: dict) -> tuple[list, dict]:
    """(positional args, keyword args) for the runner's `infer`, built exactly as its `main` builds them."""
    if runner == "objects":
        jobs = [{"name": j["name"], "image": Path(j["image"]).read_bytes(), "mask": Path(j["mask"]).read_bytes()}
                for j in spec["jobs"]]
        return [jobs], {"seed": int(spec.get("seed", 42))}
    if runner == "body":
        jobs = [{"name": j["name"], "png": Path(j["image"]).read_bytes(), "bbox": j["bbox"],
                 "box_source": j.get("box_source")} for j in spec["jobs"]]
        return [jobs, spec.get("detector")], {}
    raise BridgeError(f"unknown runner {runner!r}")


def write_outputs(out: Path, log: dict, files: dict) -> None:
    for name, d in files.items():
        (out / name).mkdir(parents=True, exist_ok=True)
        for fn, b in d.items():
            if "/" in fn or fn.startswith("."):
                raise BridgeError(f"runner returned an unsafe file name {fn!r}")
            (out / name / fn).write_bytes(b)
        (out / name / "run.json").write_text(json.dumps(log["runs"][name], indent=2))
    (out / "run.json").write_text(json.dumps(log, indent=2))


def main(argv: list[str] | None = None, lookup=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) < 2 or argv[0] != "run":
        print("modal_bridge: expected `run <runner.py>::main --job <file> --out <dir>`", file=sys.stderr)
        return 2
    target, rest = argv[1], argv[2:]
    path_s, _, entry = target.partition("::")
    ap = argparse.ArgumentParser(prog="modal_bridge run")
    ap.add_argument("--job", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(rest)
    path = Path(path_s)
    runner = path.stem
    if entry != "main" or runner not in GPU_ENV or not path.is_file():
        print(f"modal_bridge: unsupported target {target!r}", file=sys.stderr)
        return 2
    app_name = runner_constants(path).get("APP")
    if not isinstance(app_name, str):
        print("modal_bridge: the runner has no APP constant", file=sys.stderr)
        return 2
    if lookup is None:
        import modal

        def lookup(app: str, fn: str):
            return modal.Function.from_name(app, fn)

    spec = json.loads(a.job.read_text())
    args, kwargs = payload(runner, spec)
    t = time.time()
    log, files = lookup(app_name, "infer").remote(*args, **kwargs)
    log["wall_seconds"] = round(time.time() - t, 1)
    log["gpu_type"] = os.environ.get(GPU_ENV[runner], "")
    write_outputs(a.out, log, files)
    print(json.dumps({k: v for k, v in log.items() if k != "runs"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
