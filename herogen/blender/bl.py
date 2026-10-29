"""Headless Blender runner: one subprocess per job, always with a timeout."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

JOB_SCRIPT = Path(__file__).with_name("blender_job.py")
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


class BlenderError(RuntimeError):
    pass


def find_blender() -> str | None:
    """`HEROGEN_BLENDER`, then the macOS app bundle, then `blender` on PATH."""
    for cand in (os.environ.get("HEROGEN_BLENDER"), DEFAULT_BLENDER, shutil.which("blender")):
        if cand and Path(cand).exists():
            return cand
    return None


def run_job(job: dict, work: Path, name: str, timeout: float = 600.0) -> dict:
    """Write `<work>/<name>.json`, run Blender on it and return the parsed result."""
    blender = find_blender()
    if blender is None:
        raise BlenderError("Blender not found (set HEROGEN_BLENDER)")
    work.mkdir(parents=True, exist_ok=True)
    job_path = work / f"{name}.json"
    result_path = work / f"{name}.result.json"
    log_path = work / f"{name}.log"
    if result_path.exists():
        result_path.unlink()
    job_path.write_text(json.dumps(job))
    cmd = [blender, "--background", "--factory-startup", "--python-exit-code", "1",
           "--python", str(JOB_SCRIPT), "--", str(job_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise BlenderError(f"Blender job {name} timed out after {timeout:.0f} s") from exc
    log_path.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0 or not result_path.exists():
        tail = (proc.stdout + proc.stderr)[-1500:]
        raise BlenderError(f"Blender job {name} failed (exit {proc.returncode}); log {log_path}\n{tail}")
    return json.loads(result_path.read_text())
