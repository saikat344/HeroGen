"""Timed Blender operations for texture refinement, using the repository's executable discovery."""
import json
import subprocess
from pathlib import Path

from herogen.blender.bl import BlenderError, find_blender


def blender(mode, args, log, timeout=900.0):
    executable = find_blender()
    if executable is None:
        raise BlenderError("Blender not found (set HEROGEN_BLENDER)")
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    job = log.with_name(f"refine-{mode}.json")
    result = Path(str(job) + ".result.json")
    result.unlink(missing_ok=True)
    job.write_text(json.dumps(args))
    command = [executable, "--background", "--factory-startup", "--python-exit-code", "1",
               "--python", str(Path(__file__).with_name("blender_ops.py")), "--", mode, str(job)]
    with log.open("a") as output:
        try:
            proc = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise BlenderError(f"texture refinement {mode} timed out after {timeout:g} s") from exc
    if proc.returncode or not result.is_file():
        raise BlenderError(f"texture refinement {mode} failed (exit {proc.returncode}); see blender.log")
    return json.loads(result.read_text())
