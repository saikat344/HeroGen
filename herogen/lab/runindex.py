"""Read-only index of run folders for the lab page: stage records, chain images, bodies and scores.

Nothing here writes to disk or creates folders. Every path handed to the page is relative to its run folder.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Stage records in pipeline order; "2h" is the held-out view (see docs/ARCHITECTURE.md).
STAGES = [
    ("0", "input"),
    ("1", "cover art"),
    ("2", "turnaround"),
    ("2h", "held-out view"),
    ("3", "base mesh"),
    ("4", "game body"),
    ("5", "results"),
]

TURNAROUND_VIEWS = ["front", "back", "left", "right"]
HELDOUT_IMAGE = "heldout/front-left-34.png"

# The chain shown left to right: one card per step, its stage record and the images it shows.
CHAIN = [
    {"key": "sketch", "label": "Sketch", "stage": "0", "images": ["00-sketch.png"]},
    {"key": "cover", "label": "Cover art", "stage": "1", "images": ["01-cover.png"]},
    {"key": "turnaround", "label": "Turnaround", "stage": "2",
     "images": [f"views/{v}.png" for v in TURNAROUND_VIEWS], "sheet": "02-turnaround.png"},
    {"key": "heldout", "label": "Held-out view", "stage": "2h", "images": [HELDOUT_IMAGE]},
    {"key": "mesh", "label": "Base mesh", "stage": "3", "images": ["03-renders/front.png"]},
    {"key": "body", "label": "Game body", "stage": "4",
     "images": ["04-body/rest-renders/textured-front.png"]},
]

# Scores (05-scores.json): rows in this order, the held-out view last.
SCORE_VIEWS = ["front", "back", "left", "right", "front-left-34"]
HELDOUT_VIEW = "front-left-34"

SECTIONS = [
    "Head", "UpperTorso", "LowerTorso",
    "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
    "RightUpperLeg", "RightLowerLeg", "RightFoot",
]

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
BODY_DIR = re.compile(r"^04-body(-[A-Za-z0-9_.-]+)?$")
SHORT_OPS = {"<", "<=", ">", ">=", "==", "!="}


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _read_json(path: Path) -> tuple[Any, str | None]:
    """(data, None) or (None, error text). A missing file is (None, None)."""
    if not path.is_file():
        return None, None
    try:
        return json.loads(path.read_text()), None
    except (OSError, ValueError) as e:
        return None, f"{path.name} could not be read ({type(e).__name__})"


def is_run_dir(path: Path) -> bool:
    if not path.is_dir() or not SAFE_NAME.match(path.name):
        return False
    return any(path.glob("stage-*.json")) or (path / "04-body").is_dir()


def run_dir(root: Path, run_id: str) -> Path | None:
    """The run folder for `run_id`, or None when the id is not a plain name or not a run inside `root`."""
    if not SAFE_NAME.match(run_id or ""):
        return None
    root_r = root.resolve()
    p = root / run_id
    try:
        p_r = p.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if p_r.parent != root_r or not is_run_dir(p):
        return None
    return p


def normalise_check(c: dict[str, Any]) -> dict[str, Any]:
    """A check as the page shows it: value, comparison and threshold ("–" when there is no threshold).

    Pipeline checks name the comparison `op`; Blender checks name it `rule`, which is sometimes a sentence
    rather than an operator. With a threshold and no operator, the comparison is `>=` (as on results.html).
    """
    threshold = c.get("threshold")
    op = c.get("op") if isinstance(c.get("op"), str) else None
    rule = c.get("rule") if isinstance(c.get("rule"), str) else None
    cmp_text = op or rule
    rule_text = None
    if cmp_text is not None and cmp_text not in SHORT_OPS:
        rule_text, cmp_text = cmp_text, None
    if threshold is None:
        limit = "–"
        cmp_text = None
    else:
        cmp_text = cmp_text or ">="
        limit = f"{cmp_text} {_fmt(threshold)}"
    return {
        "name": str(c.get("name", "")),
        "value": c.get("value"),
        "threshold": threshold,
        "cmp": cmp_text,
        "limit": limit,
        "rule_text": rule_text,
        "pass": bool(c.get("pass")),
        "gate": bool(c.get("gate", True)),
        "note": c.get("note") if isinstance(c.get("note"), str) else None,
    }


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def stage_summary(run: Path, key: str, label: str) -> dict[str, Any]:
    rec, err = _read_json(run / f"stage-{key}.json")
    if err:
        return {"stage": key, "name": label, "status": "unreadable", "error": err}
    if not isinstance(rec, dict):
        return {"stage": key, "name": label, "status": "not run"}
    checks = [normalise_check(c) for c in rec.get("checks", []) if isinstance(c, dict)]
    return {
        "stage": key,
        "name": str(rec.get("name", label)),
        "status": "pass" if rec.get("ok") else "gate failed",
        "ok": bool(rec.get("ok")),
        "seconds": _num(rec.get("seconds")),
        "cost_usd": _num(rec.get("cost_usd")),
        "cost_basis": str(rec.get("cost_basis", "")),
        "models": [str(m) for m in rec.get("models", []) if isinstance(m, str)],
        "tools": [str(m) for m in rec.get("tools", []) if isinstance(m, str)],
        "notes": str(rec.get("notes", "")),
        "label": str((rec.get("details") or {}).get("label", "")) if isinstance(rec.get("details"), dict) else "",
        "checks": checks,
        "checks_passed": sum(1 for c in checks if c["pass"]),
        "checks_failed_gate": sum(1 for c in checks if c["gate"] and not c["pass"]),
    }


def _sections_summary(body_dir: Path) -> dict[str, Any] | None:
    data, err = _read_json(body_dir / "sections.json")
    if err:
        return {"error": err}
    if not isinstance(data, dict):
        return None
    sections = []
    for s in data.get("sections", []):
        if not isinstance(s, dict):
            continue
        pivot = s.get("pivot")
        sections.append({
            "name": str(s.get("name", "")),
            "parent": s.get("parent") if isinstance(s.get("parent"), str) else None,
            "joint": s.get("joint") if isinstance(s.get("joint"), str) else None,
            "pivot": [float(x) for x in pivot] if isinstance(pivot, list) and len(pivot) == 3
            and all(_num(x) is not None for x in pivot) else None,
            "triangles_high": _num(s.get("triangles_high")),
            "triangles_low": _num(s.get("triangles_low")),
        })
    budgets = data.get("budgets") if isinstance(data.get("budgets"), dict) else {}
    return {
        "units": str(data.get("units", "")),
        "height": _num(data.get("height")),
        "mesh_source": data.get("mesh_source") if isinstance(data.get("mesh_source"), str) else None,
        "joints_source": data.get("joints_source") if isinstance(data.get("joints_source"), str) else None,
        "sections": sections,
        "budgets": {k: _num(v) for k, v in budgets.items() if _num(v) is not None},
    }


def _rel(run: Path, p: Path) -> str:
    return p.relative_to(run).as_posix()


def bodies(run: Path, score_bodies: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Every `04-body*/` folder holding a high or low GLB, `04-body/` first.

    A body takes its name from the 05-scores.json entry whose GLB it is (variants excluded), else from its
    folder name (`04-body-carve` -> `carve`, `04-body` -> `body`).
    """
    names_by_glb: dict[str, str] = {}
    for b in score_bodies or []:
        if not b.get("variant_of") and b.get("glb") and b["glb"] not in names_by_glb:
            names_by_glb[b["glb"]] = b["name"]
    found = []
    for d in sorted(p for p in run.iterdir() if p.is_dir() and BODY_DIR.match(p.name)):
        if d.is_symlink():
            continue
        high = d / "high.glb"
        low = d / "low.glb"
        if not high.is_file() and not low.is_file():
            continue
        high_rel = _rel(run, high) if high.is_file() else None
        default = "body" if d.name == "04-body" else d.name[len("04-body-"):]
        rec, _ = _read_json(d / "stage-4.json") if d.name != "04-body" else _read_json(run / "stage-4.json")
        found.append({
            "name": names_by_glb.get(high_rel or "", default),
            "dir": d.name,
            "high": high_rel,
            "low": _rel(run, low) if low.is_file() else None,
            "sections": _sections_summary(d),
            "stage4_ok": bool(rec.get("ok")) if isinstance(rec, dict) else None,
            "front_render": _rel(run, d / "rest-renders" / "textured-front.png")
            if (d / "rest-renders" / "textured-front.png").is_file() else None,
        })
    found.sort(key=lambda b: (b["dir"] != "04-body", b["dir"]))
    return found


def scores(run: Path) -> dict[str, Any] | None:
    """`05-scores.json` as the page shows it; None when the run has none."""
    raw, err = _read_json(run / "05-scores.json")
    if err:
        return {"error": err, "bodies": []}
    if raw is None:
        return None
    bodies_in = raw.get("bodies") if isinstance(raw, dict) else None
    if not isinstance(bodies_in, list):
        return {"error": "05-scores.json has no list of bodies", "bodies": []}
    out = []
    for b in bodies_in:
        if not isinstance(b, dict):
            continue
        name = str(b.get("name", ""))
        views_in = b.get("views") if isinstance(b.get("views"), dict) else {}
        views = {}
        for v in SCORE_VIEWS:
            sv = views_in.get(v)
            if not isinstance(sv, dict):
                continue
            row = {k: _num(sv.get(k)) for k in ("iou", "colour_error", "ssim", "yaw_deg", "scale")}
            row["seen"] = sv.get("seen") if isinstance(sv.get("seen"), bool) else None
            views[v] = row
        render = f"05-renders/{name}/{HELDOUT_VIEW}.png"
        out.append({
            "name": name,
            "glb": str(b.get("glb", "")),
            "variant_of": b.get("variant_of") if isinstance(b.get("variant_of"), str) else None,
            "mesh_source": b.get("mesh_source") if isinstance(b.get("mesh_source"), str) else None,
            "joints_source": b.get("joints_source") if isinstance(b.get("joints_source"), str) else None,
            "triangles": _num(b.get("triangles")),
            "triangles_low": _num(b.get("triangles_low")),
            "atlas": _num(b.get("atlas")),
            "views": views,
            "heldout_render": render if SAFE_NAME.match(name) and (run / render).is_file() else None,
        })
    return {
        "bodies": out,
        "views": SCORE_VIEWS,
        "heldout_view": HELDOUT_VIEW,
        "heldout_image": HELDOUT_IMAGE if (run / HELDOUT_IMAGE).is_file() else None,
        "comparison_image": "05-renders/heldout-comparison.png"
        if (run / "05-renders" / "heldout-comparison.png").is_file() else None,
    }


def run_detail(run: Path) -> dict[str, Any]:
    stages = {key: stage_summary(run, key, label) for key, label in STAGES}
    chain = []
    for step in CHAIN:
        images = [{"path": p, "exists": (run / p).is_file()} for p in step["images"]]
        card = {"key": step["key"], "label": step["label"], "stage": step["stage"], "images": images}
        if "sheet" in step:
            card["sheet"] = step["sheet"] if (run / step["sheet"]).is_file() else None
        chain.append(card)
    sc = scores(run)
    total_seconds = sum(s.get("seconds") or 0.0 for s in stages.values())
    total_cost = sum(s.get("cost_usd") or 0.0 for s in stages.values())
    return {
        "id": run.name,
        "stages": [stages[k] for k, _ in STAGES],
        "chain": chain,
        "bodies": bodies(run, (sc or {}).get("bodies")),
        "scores": sc,
        "total_seconds": round(total_seconds, 2),
        "total_cost_usd": round(total_cost, 6),
        "cost_is_estimate": True,
    }


def run_list(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if p.is_symlink() or not is_run_dir(p):
            continue
        status = {}
        for key, _ in STAGES:
            rec, err = _read_json(p / f"stage-{key}.json")
            status[key] = "unreadable" if err else ("not run" if not isinstance(rec, dict)
                                                    else ("pass" if rec.get("ok") else "gate failed"))
        out.append({
            "id": p.name,
            "stages": status,
            "has_body": (p / "04-body" / "high.glb").is_file() or (p / "04-body" / "low.glb").is_file(),
            "has_scores": (p / "05-scores.json").is_file(),
            "modified": max((f.stat().st_mtime for f in p.glob("stage-*.json")), default=p.stat().st_mtime),
        })
    out.sort(key=lambda r: -r["modified"])
    return out
