"""The Lab's JSON API (`/api/workspace`, `/api/runs/<root>/<run>`), served per user over that user's run folders.

The service answers with the Lab's own code when the deployed HeroGen has it: `herogen.lab.workspace.Workspace`
(the Studio interface), else `herogen.lab.runindex` (the first Lab page). Each user's runs folder is the one runs
root, key `r1`, labelled "Your heroes", so a run's id is `r1/<job-id>` exactly as the Lab names it, and the
Studio page works unchanged. Without either module a minimal summary with the same field names is built here;
the service's own page reads only those fields.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

ROOT_KEY = "r1"
ROOT_LABEL = "Your heroes"
STAGES = [("0", "input"), ("1", "cover art"), ("2", "turnaround"), ("2h", "held-out view"), ("3", "base mesh"),
          ("4", "game body"), ("5", "results")]
IMAGES = {"sketch": "00-sketch.png", "cover": "01-cover.png", "turnaround": "02-turnaround.png",
          "front": "views/front.png", "back": "views/back.png", "left": "views/left.png", "right": "views/right.png",
          "heldout": "heldout/front-left-34.png"}
PROFILES = ["best", "roblox-ingame", "roblox-avatar"]


def _module(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def backend() -> str:
    if _module("herogen.lab.workspace") is not None:
        return "workspace"
    if _module("herogen.lab.runindex") is not None:
        return "runindex"
    return "minimal"


def studio_static() -> Path | None:
    """The Studio page's static folder, when the deployed code has the Studio (its page loads `static/app.js`)."""
    lab = _module("herogen.lab")
    if lab is None or not getattr(lab, "__file__", None):
        return None
    static = Path(lab.__file__).resolve().parent / "static"
    return static if (static / "index.html").is_file() and (static / "app.js").is_file() else None


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _stage(run: Path, key: str, label: str) -> dict[str, Any]:
    rec = _read(run / f"stage-{key}.json")
    if not isinstance(rec, dict):
        return {"stage": key, "name": label, "status": "not run"}
    checks = [c for c in rec.get("checks") or [] if isinstance(c, dict)]
    return {"stage": key, "name": label, "status": "pass" if rec.get("ok") is True else "gate failed",
            "ok": rec.get("ok") is True, "seconds": rec.get("seconds"), "cost_usd": rec.get("cost_usd"),
            "started": rec.get("started"), "notes": rec.get("notes") or "",
            "checks": [{"name": c.get("name"), "value": c.get("value"), "threshold": c.get("threshold"),
                        "op": c.get("op", ">="), "pass": c.get("pass") is True, "gate": c.get("gate", True)}
                       for c in checks]}


def _files(run: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for k, rel in IMAGES.items():
        p = run / rel
        if p.is_file() and not p.is_symlink():
            out[k] = {"path": rel, "bytes": p.stat().st_size}
    return out


def _minimal_detail(run: Path, run_id: str) -> dict[str, Any]:
    stages = [_stage(run, k, label) for k, label in STAGES]
    glbs = [f"04-body/{n}" for n in ("high.glb", "low.glb") if (run / "04-body" / n).is_file()]
    secs = [s["seconds"] for s in stages if isinstance(s.get("seconds"), (int, float))]
    costs = [s["cost_usd"] for s in stages if isinstance(s.get("cost_usd"), (int, float))]
    return {"id": run_id, "root": ROOT_KEY, "name": run.name, "path": f"{ROOT_LABEL}/{run.name}", "stages": stages,
            "images": _files(run), "bodies": [{"dir": "04-body", "files": [{"path": g} for g in glbs]}] if glbs else [],
            "deliverables": [{"kind": "GLB " + g.split("/")[-1][:-4], "status": "built", "path": g} for g in glbs],
            "total_seconds": round(sum(secs), 2) if secs else None,
            "total_cost_usd": round(sum(costs), 6) if costs else None, "cost_is_estimate": True}


def _minimal_entry(run: Path, run_id: str) -> dict[str, Any]:
    status = {}
    for k, _ in STAGES:
        rec = _read(run / f"stage-{k}.json")
        status[k] = "not run" if not isinstance(rec, dict) else ("pass" if rec.get("ok") is True else "gate failed")
    return {"id": run_id, "root": ROOT_KEY, "name": run.name, "stages": status,
            "has_body": (run / "04-body" / "high.glb").is_file(), "readable": True}


def workspace(runs_root: Path, job_ids: list[str]) -> dict[str, Any]:
    """The `/api/workspace` answer for one user's runs (newest job first in `job_ids`)."""
    allowed = set(job_ids)
    kind = backend()
    if kind == "workspace":
        from herogen.lab.workspace import Workspace

        data = Workspace([runs_root], labels=[ROOT_LABEL]).workspace()
        data["runs"] = {k: v for k, v in data["runs"].items() if k.split("/", 1)[-1] in allowed}
        chars = []
        for c in data["characters"]:
            c["runs"] = [r for r in c["runs"] if r.split("/", 1)[-1] in allowed]
            c["profiles"] = {p: [r for r in ids if r.split("/", 1)[-1] in allowed]
                             for p, ids in c.get("profiles", {}).items()}
            c["profiles"] = {p: ids for p, ids in c["profiles"].items() if ids}
            if c["runs"]:
                chars.append(c)
        data["characters"] = chars
        data["versions_root"] = None
        data["unattached_versions"] = []
        return data
    runs = {}
    for jid in job_ids:
        run = runs_root / jid
        if run.is_dir() and not run.is_symlink():
            runs[f"{ROOT_KEY}/{jid}"] = (_minimal_entry(run, f"{ROOT_KEY}/{jid}") if kind == "minimal"
                                         else {**_runindex_entry(run), "id": f"{ROOT_KEY}/{jid}", "root": ROOT_KEY,
                                               "name": jid})
    chars = [{"key": f"run-{rid.replace('/', '-')}", "title": rid.split("/", 1)[1], "runs": [rid],
              "profiles": {}, "versions": []} for rid in runs]
    return {"roots": [{"key": ROOT_KEY, "label": ROOT_LABEL, "exists": runs_root.is_dir()}], "versions_root": None,
            "profiles": PROFILES, "characters": chars, "runs": runs, "unattached_versions": []}


def _runindex_entry(run: Path) -> dict[str, Any]:
    from herogen.lab import runindex

    return runindex._run_entry(run)


def run_detail(runs_root: Path, job_id: str) -> dict[str, Any] | None:
    """The `/api/runs/r1/<job-id>` answer, or None when the folder does not exist."""
    run = runs_root / job_id
    if not run.is_dir() or run.is_symlink():
        return None
    run_id = f"{ROOT_KEY}/{job_id}"
    kind = backend()
    if kind == "workspace":
        from herogen.lab.workspace import Workspace

        return Workspace([runs_root], labels=[ROOT_LABEL]).run_detail(run_id)
    if kind == "runindex":
        from herogen.lab import runindex

        detail = runindex.run_detail(run)
        detail.update({"id": run_id, "root": ROOT_KEY, "name": job_id, "path": f"{ROOT_LABEL}/{job_id}"})
        return detail
    return _minimal_detail(run, run_id)
