"""Read-only index of run folders for the lab page: stage records, chain images, bodies and scores.

Nothing here writes to disk or creates folders. Every path handed to the page is relative to its run folder.
"""

from __future__ import annotations

import json
import math
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
    """A finite number as a float, else None (bools, strings, NaN, infinities and ints too big for a float)."""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except OverflowError:
        return None
    return f if math.isfinite(f) else None


def _list(v: Any) -> list[Any]:
    """`v` when it is a list, else an empty list: record fields are only iterated after this check."""
    return v if isinstance(v, list) else []


def _dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _scalar(v: Any) -> Any:
    """A value the page can print as it is: None, bool, finite number or string; anything else as JSON text."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, (int, float)):
        return v if _num(v) is not None else None
    return json.dumps(v, allow_nan=False, default=str)


def _finite_float(text: str) -> float | None:
    f = float(text)
    return f if math.isfinite(f) else None


def parse_json(text: str) -> Any:
    """Strict-enough JSON for the page: NaN, Infinity and numbers that overflow a float become None."""
    return json.loads(text, parse_constant=lambda _: None, parse_float=_finite_float)


def _read_json(path: Path) -> tuple[Any, str | None]:
    """(data, None) or (None, error text). A missing file (or a broken link) is (None, None)."""
    try:
        if not path.is_file():
            return None, None
        return parse_json(path.read_text()), None
    except (OSError, ValueError, RecursionError) as e:
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


ASSUMED_CMP_NOTE = "no comparison recorded: shown as >=, the default for a check with a threshold"


def normalise_check(c: dict[str, Any]) -> dict[str, Any]:
    """A check as the page shows it: value, comparison and threshold ("–" when there is no threshold).

    Pipeline checks name the comparison `op`; Blender checks name it `rule`. Older Blender records put a
    sentence in `rule` instead of an operator: the sentence becomes a note and is never read as a comparison.
    With a threshold and no recorded operator, the comparison is `>=` (the default of both pipelines' `check()`
    and of results.html), flagged with `cmp_assumed` and a note saying so.
    """
    threshold = _scalar(c.get("threshold"))
    notes = [c["note"]] if isinstance(c.get("note"), str) and c["note"] else []
    cmp_text = None
    for key in ("op", "rule"):
        text = c.get(key)
        if not isinstance(text, str) or not text.strip():
            continue
        if text.strip() in SHORT_OPS:
            cmp_text = cmp_text or text.strip()
        else:
            notes.append(f"{key}: {text}")
    assumed = False
    if threshold is None:
        limit = "–"
        cmp_text = None
    else:
        if cmp_text is None:
            cmp_text, assumed = ">=", True
            notes.append(ASSUMED_CMP_NOTE)
        limit = f"{cmp_text} {_fmt(threshold)}"
    return {
        "name": str(c.get("name", "")),
        "value": _scalar(c.get("value")),
        "threshold": threshold,
        "cmp": cmp_text,
        "cmp_assumed": assumed,
        "limit": limit,
        "pass": c.get("pass") is True,
        "gate": c.get("gate", True) is not False,
        "notes": notes,
    }


def _text(v: Any) -> str:
    """A record's free-text field: strings as they are, None as empty, anything else as JSON text."""
    if v is None:
        return ""
    return v if isinstance(v, str) else str(_scalar(v))


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
    checks = [normalise_check(c) for c in _list(rec.get("checks")) if isinstance(c, dict)]
    details = _dict(rec.get("details"))
    return {
        "stage": key,
        "name": rec["name"] if isinstance(rec.get("name"), str) and rec["name"] else label,
        "status": "pass" if rec.get("ok") is True else "gate failed",
        "ok": rec.get("ok") is True,
        "seconds": _num(rec.get("seconds")),
        "cost_usd": _num(rec.get("cost_usd")),
        "cost_basis": _text(rec.get("cost_basis")),
        "models": [m for m in _list(rec.get("models")) if isinstance(m, str)],
        "tools": [m for m in _list(rec.get("tools")) if isinstance(m, str)],
        "notes": _text(rec.get("notes")),
        "label": _text(details.get("label")),
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
    for s in _list(data.get("sections")):
        if not isinstance(s, dict):
            continue
        pivot = s.get("pivot")
        sections.append({
            "name": _text(s.get("name")),
            "parent": s.get("parent") if isinstance(s.get("parent"), str) else None,
            "joint": s.get("joint") if isinstance(s.get("joint"), str) else None,
            "pivot": [_num(x) for x in pivot] if isinstance(pivot, list) and len(pivot) == 3
            and all(_num(x) is not None for x in pivot) else None,
            "triangles_high": _num(s.get("triangles_high")),
            "triangles_low": _num(s.get("triangles_low")),
        })
    # Triangle budgets belong to an opt-in profile (e.g. "roblox-ingame"); the default "best" profile has none,
    # written as null or left out. No usable budget is None, never an empty dict that reads as "within budget".
    budgets = {k: _num(v) for k, v in _dict(data.get("budgets")).items() if _num(v) is not None}
    profile = data.get("profile")
    return {
        "profile": profile.strip() if isinstance(profile, str) and profile.strip() else None,
        "units": _text(data.get("units")),
        "height": _num(data.get("height")),
        "mesh_source": data.get("mesh_source") if isinstance(data.get("mesh_source"), str) else None,
        "joints_source": data.get("joints_source") if isinstance(data.get("joints_source"), str) else None,
        "sections": sections,
        "budgets": budgets or None,
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
    try:
        dirs = sorted(p for p in run.iterdir() if BODY_DIR.match(p.name) and not p.is_symlink() and p.is_dir())
    except OSError:
        return []
    for d in dirs:
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
            "stage4_ok": rec.get("ok") is True if isinstance(rec, dict) else None,
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
        name = _text(b.get("name"))
        views_in = _dict(b.get("views"))
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
            "glb": _text(b.get("glb")),
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


def _total(values: Any) -> float | None:
    """The sum of the finite numbers in `values` (None counts as 0), or None when the sum overflows."""
    try:
        return _num(math.fsum(v or 0.0 for v in values))
    except OverflowError:
        return None


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
    total_seconds = _total(s.get("seconds") for s in stages.values())
    total_cost = _total(s.get("cost_usd") for s in stages.values())
    return {
        "id": run.name,
        "stages": [stages[k] for k, _ in STAGES],
        "chain": chain,
        "bodies": bodies(run, (sc or {}).get("bodies")),
        "scores": sc,
        "total_seconds": None if total_seconds is None else round(total_seconds, 2),
        "total_cost_usd": None if total_cost is None else round(total_cost, 6),
        "cost_is_estimate": True,
    }


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _run_entry(p: Path) -> dict[str, Any]:
    status = {}
    for key, _ in STAGES:
        rec, err = _read_json(p / f"stage-{key}.json")
        status[key] = "unreadable" if err else ("not run" if not isinstance(rec, dict)
                                                else ("pass" if rec.get("ok") is True else "gate failed"))
    # A stage file that cannot be stat()ed (a broken link, a permission error) is left out of the time.
    times = [t for t in (_mtime(f) for f in p.glob("stage-*.json")) if t is not None]
    return {
        "id": p.name,
        "stages": status,
        "has_body": (p / "04-body" / "high.glb").is_file() or (p / "04-body" / "low.glb").is_file(),
        "has_scores": (p / "05-scores.json").is_file(),
        "modified": max(times, default=_mtime(p) or 0.0),
    }


def run_list(root: Path) -> list[dict[str, Any]]:
    """Every run folder in `root`, newest first. A folder that cannot be listed is left out; one bad run
    never hides the others."""
    try:
        if not root.is_dir():
            return []
        candidates = sorted(root.iterdir())
    except OSError:
        return []
    out = []
    for p in candidates:
        try:
            if p.is_symlink() or not is_run_dir(p):
                continue
            out.append(_run_entry(p))
        except OSError:
            continue
    out.sort(key=lambda r: -r["modified"])
    return out
