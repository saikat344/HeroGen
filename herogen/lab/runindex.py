"""Read-only index of one run folder for the lab page: stage records, images, bodies, budgets and scores.

Nothing here writes to disk or creates folders. Every path handed to the page is relative to its run folder.
Every value comes from a file in the run folder; a field a record does not hold is None (or left out), never a
default the page could mistake for a measurement.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
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

# Files in a body folder the page offers as deliverables, in this order (each listed only when it exists).
BODY_FILES = ["high.glb", "low.glb", "atlas-high.png", "atlas-low.png", "sections.json"]

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


def folder_error(path: Path) -> str | None:
    """Why a folder cannot be listed ("folder could not be read (PermissionError)"), or None when it can."""
    try:
        os.listdir(path)
    except OSError as e:
        return f"folder could not be read ({type(e).__name__})"
    return None


def is_run_dir(path: Path) -> bool:
    """A plain-named folder holding a stage record or a body, or one that cannot be listed at all (it is shown as
    a run that can't be read, rather than hidden)."""
    if not SAFE_NAME.match(path.name) or path.is_symlink() or not path.is_dir():
        return False
    if folder_error(path):
        return True
    return any(path.glob("stage-*.json")) or (path / "04-body").is_dir()


def run_dir(root: Path, run_id: str) -> Path | None:
    """The run folder for `run_id`, or None when the id is not a plain name, names a symlink (as the listing
    leaves them out) or is not a run directly inside `root`."""
    if not SAFE_NAME.match(run_id or ""):
        return None
    p = root / run_id
    try:
        root_r = root.resolve(strict=True)
        p_r = p.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if p.is_symlink() or p_r.parent != root_r or not is_run_dir(p):
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
        "review": c.get("review") is True,
        "notes": notes,
        # The poses a joint-gap check measured and the sections a triangle check counted, when it lists them.
        "poses": [p for p in _list(c.get("poses")) if isinstance(p, str)] or None,
        "sections": [p for p in _list(c.get("sections")) if isinstance(p, str)] or None,
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
    if rec is None:
        return {"stage": key, "name": label, "status": "not run"}
    if not isinstance(rec, dict):
        return {"stage": key, "name": label, "status": "unreadable", "error": f"stage-{key}.json is not a JSON object"}
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
        "started": rec["started"] if isinstance(rec.get("started"), str) else None,
        "code": _code(rec.get("code")),
        "calls": _calls(rec.get("calls")),
        "mesh_source": rec.get("mesh_source") if isinstance(rec.get("mesh_source"), str) else None,
        "joints_source": rec.get("joints_source") if isinstance(rec.get("joints_source"), str) else None,
        "checks": checks,
        "checks_passed": sum(1 for c in checks if c["pass"]),
        "checks_failed_gate": sum(1 for c in checks if c["gate"] and not c["pass"]),
    }


def _code(v: Any) -> dict[str, Any] | None:
    """The `code` stamp of a record (`{"commit": ..., "dirty": ...}`), or None when the record has none."""
    if not isinstance(v, dict) or not isinstance(v.get("commit"), str) or not v["commit"]:
        return None
    return {"commit": v["commit"], "dirty": v.get("dirty") if isinstance(v.get("dirty"), bool) else None}


def _calls(v: Any) -> dict[str, Any] | None:
    """Model calls a record lists: how many, and how many were replayed from the cache. None when not listed."""
    if not isinstance(v, list):
        return None
    calls = [c for c in v if isinstance(c, dict)]
    return {
        "count": len(calls),
        "replayed": sum(1 for c in calls if c.get("cached") is True),
        "models": sorted({c["model"] for c in calls if isinstance(c.get("model"), str)}),
    }


def _vec3(v: Any) -> list[float] | None:
    if isinstance(v, list) and len(v) == 3 and all(_num(x) is not None for x in v):
        return [_num(x) for x in v]
    return None


def _gap_table(v: Any) -> dict[str, dict[str, float]] | None:
    """`{pose: {joint: gap in studs}}` from a pose-test record, numbers only; None when there is none."""
    if not isinstance(v, dict):
        return None
    out = {}
    for pose, row in v.items():
        if isinstance(pose, str) and isinstance(row, dict):
            out[pose] = {j: _num(g) for j, g in row.items() if isinstance(j, str) and _num(g) is not None}
    return out or None


GAP_TABLES = ("gaps_high", "gaps_low", "gaps_high_heldout", "gaps_low_heldout")


def _pose_test(v: Any) -> dict[str, Any] | None:
    if not isinstance(v, dict):
        return None
    return {
        "gate_studs": _num(v.get("gate_studs")),
        "ungated_poses": [p for p in _list(v.get("ungated_poses")) if isinstance(p, str)],
        **{k: _gap_table(v.get(k)) for k in GAP_TABLES},
    }


def _lods(v: Any) -> dict[str, Any]:
    """Per level of detail: the triangles the record counts, its atlas size, its budget and per-group counts."""
    out = {}
    for lod, d in _dict(v).items():
        if not isinstance(lod, str) or not isinstance(d, dict):
            continue
        groups = {k: _num(n) for k, n in _dict(d.get("triangles_by_group")).items() if _num(n) is not None}
        out[lod] = {
            "triangles": _num(d.get("triangles")),
            "atlas_size": _num(d.get("atlas_size")),
            "budget": _num(d.get("budget")),
            "triangles_by_group": groups or None,
        }
    return out


def _budget_groups(v: Any) -> dict[str, Any] | None:
    """Per-part-group budgets (`{"Head": {"sections": [...], "triangles_max": n}}`), or None."""
    out = {}
    for name, g in _dict(v).items():
        if isinstance(name, str) and isinstance(g, dict) and _num(g.get("triangles_max")) is not None:
            out[name] = {"sections": [s for s in _list(g.get("sections")) if isinstance(s, str)],
                         "triangles_max": _num(g["triangles_max"])}
    return out or None


def _not_built(budgets: Any) -> list[dict[str, str]]:
    """What a profile record says it does not build, each with the record's own words: its `not_built` list, and
    any budget entry whose value is a sentence starting "not built" (older records: `"cages": "not built: ..."`)."""
    b = _dict(budgets)
    items = [{"name": x.strip(), "reason": "listed as not built in the profile record"}
             for x in _list(b.get("not_built")) if isinstance(x, str) and x.strip()]
    for k, v in b.items():
        if isinstance(v, str) and v.lower().startswith("not built") \
                and not any(k.lower() in i["name"].lower() for i in items):
            items.append({"name": k, "reason": v})
    return items


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
            "bounds_min": _vec3(s.get("bounds_min")),
            "bounds_max": _vec3(s.get("bounds_max")),
        })
    # Triangle budgets belong to an opt-in profile (e.g. "roblox-ingame"); the default "best" profile has none,
    # written as null or left out. No usable budget is None, never an empty dict that reads as "within budget".
    budgets = {k: _num(v) for k, v in _dict(data.get("budgets")).items() if _num(v) is not None}
    profile = data.get("profile")
    raw_budgets = _dict(data.get("budgets"))
    return {
        "profile": profile.strip() if isinstance(profile, str) and profile.strip() else None,
        "units": _text(data.get("units")),
        "height": _num(data.get("height")),
        "frame": _text(data.get("frame")),
        "mesh_source": data.get("mesh_source") if isinstance(data.get("mesh_source"), str) else None,
        "joints_source": data.get("joints_source") if isinstance(data.get("joints_source"), str) else None,
        "shape_views": [v for v in _list(data.get("shape_views")) if isinstance(v, str)],
        "sections": sections,
        "budgets": budgets or None,
        "budget_groups": _budget_groups(raw_budgets.get("groups")),
        "not_built": _not_built(raw_budgets),
        "lods": _lods(data.get("lods")),
        "joints": {k: _vec3(v) for k, v in _dict(data.get("joints")).items() if _vec3(v) is not None},
        "ends": {k: _vec3(v) for k, v in _dict(data.get("ends")).items() if _vec3(v) is not None},
        "pose_test": _pose_test(data.get("pose_test")),
    }


def _rel(run: Path, p: Path) -> str:
    return p.relative_to(run).as_posix()


GLB_MAGIC = 0x46546C67
GLB_JSON = 0x4E4F534A
GLB_JSON_MAX = 32 * 1024 * 1024


def glb_summary(path: Path) -> dict[str, Any]:
    """Triangles per named node, counted from a GLB's own JSON chunk (the binary chunk is never read).

    `{"bytes", "triangles", "sections": {node name: triangles}, "nodes"}`, or `{"bytes", "error"}` when the file
    is not a readable glTF 2.0 binary. Only triangle-list primitives (mode 4, the default) are counted."""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            head = f.read(20)
            if len(head) < 20:
                return {"bytes": size, "error": "too short for a GLB"}
            magic, version, _length, chunk_len, chunk_type = struct.unpack("<IIIII", head)
            if magic != GLB_MAGIC or version != 2 or chunk_type != GLB_JSON or chunk_len > GLB_JSON_MAX:
                return {"bytes": size, "error": "not a glTF 2.0 binary"}
            doc = parse_json(f.read(chunk_len).decode("utf-8"))
    except (OSError, ValueError, RecursionError, UnicodeDecodeError) as e:
        return {"bytes": None, "error": f"could not be read ({type(e).__name__})"}
    if not isinstance(doc, dict):
        return {"bytes": size, "error": "its JSON chunk is not an object"}
    accessors = _list(doc.get("accessors"))
    meshes = _list(doc.get("meshes"))

    def count(i: Any) -> float | None:
        if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(accessors):
            return _num(_dict(accessors[i]).get("count"))
        return None

    def mesh_triangles(m: Any) -> int:
        total = 0
        for prim in _list(_dict(m).get("primitives")):
            prim = _dict(prim)
            if prim.get("mode", 4) != 4:
                continue
            n = count(prim.get("indices")) if "indices" in prim else count(_dict(prim.get("attributes")).get("POSITION"))
            total += int(n // 3) if n else 0
        return total

    per_node: dict[str, int] = {}
    nodes = _list(doc.get("nodes"))
    for node in nodes:
        node = _dict(node)
        mi = node.get("mesh")
        if isinstance(mi, int) and not isinstance(mi, bool) and 0 <= mi < len(meshes):
            name = node["name"] if isinstance(node.get("name"), str) else f"mesh {mi}"
            per_node[name] = per_node.get(name, 0) + mesh_triangles(meshes[mi])
    return {"bytes": size, "triangles": sum(per_node.values()), "sections": per_node, "nodes": len(nodes)}


def png_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from a PNG's header, or None when the file is not a readable PNG."""
    try:
        with path.open("rb") as f:
            head = f.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def file_entry(run: Path, rel: str) -> dict[str, Any] | None:
    """A file the page can show or offer: its run-relative path, size and, for a PNG, its pixel size."""
    p = run / rel
    try:
        if not p.is_file():
            return None
        size = p.stat().st_size
    except OSError:
        return None
    out: dict[str, Any] = {"path": rel, "bytes": size}
    if rel.lower().endswith(".png"):
        wh = png_size(p)
        out["px"] = list(wh) if wh else None
    return out


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
        low_rel = _rel(run, low) if low.is_file() else None
        found.append({
            "name": names_by_glb.get(high_rel or "", default),
            "dir": d.name,
            "high": high_rel,
            "low": low_rel,
            "glb": {lod: glb_summary(run / rel) for lod, rel in (("high", high_rel), ("low", low_rel)) if rel},
            "sections": _sections_summary(d),
            "stage4_ok": rec.get("ok") is True if isinstance(rec, dict) else None,
            "stage4_record": "stage-4.json" if d.name == "04-body" else f"{d.name}/stage-4.json",
            "front_render": _rel(run, d / "rest-renders" / "textured-front.png")
            if (d / "rest-renders" / "textured-front.png").is_file() else None,
            "files": [e for e in (file_entry(run, f"{d.name}/{n}") for n in BODY_FILES) if e],
        })
    found.sort(key=lambda b: (b["dir"] != "04-body", b["dir"]))
    return found


SCORE_METRICS = ("iou", "colour_error", "ssim", "yaw_deg", "scale", "pixels")


def _score_views(raw: dict[str, Any], bodies_in: list[Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """The scored views in the file's own order, with what the file says about each (held out or not, image, mask).

    Newer score files list their views (`"views": [{"view": ..., "heldout": ...}]`). Older ones do not: their views
    are the known turnaround order followed by any other view a body was scored on, and a view counts as held out
    when every body records it as unseen."""
    info: dict[str, dict[str, Any]] = {}
    for v in _list(raw.get("views")):
        v = _dict(v)
        name = v.get("view")
        if isinstance(name, str) and SAFE_NAME.match(name) and name not in info:
            info[name] = {
                "heldout": v.get("heldout") is True,
                "image": v["image"] if isinstance(v.get("image"), str) else None,
                "mask": v["mask"] if isinstance(v.get("mask"), str) else None,
                "yaw_deg": _num(v.get("yaw_deg")),
            }
    if info:
        return list(info), info
    seen_by_view: dict[str, list[Any]] = {}
    for b in bodies_in:
        for name, row in _dict(_dict(b).get("views")).items():
            if isinstance(name, str) and SAFE_NAME.match(name) and isinstance(row, dict):
                seen_by_view.setdefault(name, []).append(row.get("seen"))
    names = [v for v in SCORE_VIEWS if v in seen_by_view] + sorted(v for v in seen_by_view if v not in SCORE_VIEWS)
    old_heldout = _dict(raw.get("heldout"))
    for name in names:
        held = bool(seen_by_view[name]) and all(x is False for x in seen_by_view[name])
        image = old_heldout.get("image") if held and isinstance(old_heldout.get("image"), str) else None
        mask = old_heldout.get("mask") if held and isinstance(old_heldout.get("mask"), str) else None
        info[name] = {"heldout": held, "image": image, "mask": mask, "yaw_deg": None}
    return names, info


def read_scores(path: Path, base: Path) -> dict[str, Any] | None:
    """A score file (`05-scores.json` or a frozen `05-scores-<tag>.json`) as the page shows it; None when absent.

    Render paths are checked relative to `base`, the folder the scorer wrote `05-renders/` into."""
    raw, err = _read_json(path)
    if err:
        return {"error": err, "bodies": [], "views": [], "view_info": {}}
    if raw is None:
        return None
    bodies_in = raw.get("bodies") if isinstance(raw, dict) else None
    if not isinstance(bodies_in, list):
        return {"error": f"{path.name} has no list of bodies", "bodies": [], "views": [], "view_info": {}}
    views, info = _score_views(raw, bodies_in)
    heldout_views = [v for v in views if info[v]["heldout"]]
    out = []
    for b in bodies_in:
        if not isinstance(b, dict):
            continue
        name = _text(b.get("name"))
        views_in = _dict(b.get("views"))
        rows = {}
        for v in views:
            sv = views_in.get(v)
            if not isinstance(sv, dict):
                continue
            row: dict[str, Any] = {k: _num(sv.get(k)) for k in SCORE_METRICS}
            for k in ("seen", "seen_shape", "seen_colour"):
                row[k] = sv.get(k) if isinstance(sv.get(k), bool) else None
            rows[v] = row
        renders = {}
        if SAFE_NAME.match(name):
            for v in views:
                rel = f"05-renders/{name}/{v}.png"
                if (base / rel).is_file():
                    renders[v] = rel
        profile = _dict(b.get("profile"))
        out.append({
            "name": name,
            "glb": _text(b.get("glb")),
            "variant_of": b.get("variant_of") if isinstance(b.get("variant_of"), str) else None,
            "mesh_source": b.get("mesh_source") if isinstance(b.get("mesh_source"), str) else None,
            "joints_source": b.get("joints_source") if isinstance(b.get("joints_source"), str) else None,
            "colour_source": b.get("colour_source") if isinstance(b.get("colour_source"), str) else None,
            "triangles": _num(b.get("triangles")),
            "triangles_low": _num(b.get("triangles_low")),
            "atlas": _num(b.get("atlas")),
            "budget": _num(b.get("budget")),
            "within_budget": b.get("within_budget") if isinstance(b.get("within_budget"), bool) else None,
            "profile": profile["name"] if isinstance(profile.get("name"), str) else None,
            "views": rows,
            "renders": renders,
            "heldout_render": renders.get(heldout_views[0]) if heldout_views else None,
        })
    first_heldout = heldout_views[0] if heldout_views else None
    image = info[first_heldout]["image"] if first_heldout else None
    if first_heldout and image is None and (base / f"heldout/{first_heldout}.png").is_file():
        image = f"heldout/{first_heldout}.png"
    return {
        "file": path.name,
        "scored": raw["scored"] if isinstance(raw.get("scored"), str) else None,
        "code": _code(raw.get("code")),
        "bodies": out,
        "views": views,
        "view_info": info,
        "heldout_views": heldout_views,
        "heldout_view": first_heldout,
        "heldout_image": image if image and (base / image).is_file() else None,
        "comparison_image": "05-renders/heldout-comparison.png"
        if (base / "05-renders" / "heldout-comparison.png").is_file() else None,
    }


def scores(run: Path) -> dict[str, Any] | None:
    """`05-scores.json` as the page shows it; None when the run has none."""
    return read_scores(run / "05-scores.json", run)


def _total(values: Any) -> float | None:
    """The sum of the finite numbers in `values` (None counts as 0), or None when the sum overflows."""
    try:
        return _num(math.fsum(v or 0.0 for v in values))
    except OverflowError:
        return None


# Images the page shows, by key: the chain from the sketch to the turnaround and the masks used to frame the
# body over a view. Each is listed with its size when the file exists.
IMAGES = {
    "sketch": "00-sketch.png", "cover": "01-cover.png", "turnaround": "02-turnaround.png",
    "sketch_mask": "01-masks/sketch.png", "cover_mask": "01-masks/cover.png", "mesh_front": "03-renders/front.png",
    **{f"view_{v}": f"views/{v}.png" for v in ["front", "right", "back", "left", "top"]},
    **{f"mask_{v}": f"masks/{v}.png" for v in ["front", "right", "back", "left", "top"]},
}


def images(run: Path) -> dict[str, dict[str, Any]]:
    return {k: e for k, rel in IMAGES.items() if (e := file_entry(run, rel))}


def view_meta(run: Path) -> dict[str, Any] | None:
    """`views/meta.json`: the pixel scale shared by the turnaround views (pixels per stud) and their image size."""
    data, err = _read_json(run / "views" / "meta.json")
    if err:
        return {"error": err}
    if not isinstance(data, dict):
        return None
    return {"pixels_per_unit": _num(data.get("pixels_per_unit")), "height_units": _num(data.get("height_units")),
            "image_size": _num(data.get("image_size"))}


def _sum_sections(sec: dict[str, Any], key: str, names: list[str] | None = None) -> float | None:
    vals = [s[key] for s in sec["sections"] if (names is None or s["name"] in names) and s.get(key) is not None]
    return float(sum(vals)) if vals else None


def budget_meters(sec: dict[str, Any] | None) -> dict[str, Any]:
    """Triangle counts against the budgets the body's own record holds.

    `groups`: a total and per part group budget (the avatar profile); `lods`: one budget per level of detail (the
    in-game profile and older runs); `none`: the record holds no budget (the best-quality profile). Counts are the
    record's per-LOD totals (`lods.<lod>.triangles`, `triangles_by_group`), else the sum of its sections."""
    if not sec or "sections" not in sec:
        return {"kind": "unknown", "items": []}
    lods = sec.get("lods") or {}
    budgets = sec.get("budgets") or {}

    def lod_total(lod: str) -> float | None:
        v = (lods.get(lod) or {}).get("triangles")
        return v if v is not None else _sum_sections(sec, f"triangles_{lod}")

    if sec.get("budget_groups"):
        by_group = (lods.get("high") or {}).get("triangles_by_group") or {}
        items = []
        if budgets.get("triangles_total_max") is not None:
            items.append({"key": "total", "label": "Total", "value": lod_total("high"),
                          "limit": budgets["triangles_total_max"]})
        for name, g in sec["budget_groups"].items():
            value = by_group.get(name)
            if value is None:
                value = _sum_sections(sec, "triangles_high", g["sections"])
            items.append({"key": name, "label": re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name), "value": value,
                          "limit": g["triangles_max"], "sections": g["sections"]})
        return {"kind": "groups", "items": items}
    items = []
    for lod in ("high", "low"):
        limit = budgets.get(f"triangles_{lod}_max")
        if limit is not None:
            items.append({"key": lod, "label": lod.capitalize(), "value": lod_total(lod), "limit": limit})
    if items:
        return {"kind": "lods", "items": items}
    # The LODs the record lists, else every LOD its sections count triangles for.
    present = [lod for lod in ("high", "low") if (lod in lods if lods else lod_total(lod) is not None)]
    return {"kind": "none", "items": [{"key": lod, "label": lod.capitalize(), "value": lod_total(lod), "limit": None}
                                      for lod in present]}


DELIVERABLE_KINDS = {
    "high.glb": "GLB high", "low.glb": "GLB low", "atlas-high.png": "Texture atlas (high)",
    "atlas-low.png": "Texture atlas (low)", "sections.json": "Sections and pivots",
}


def deliverables(run: Path, body: dict[str, Any] | None, stage4: dict[str, Any] | None) -> list[dict[str, Any]]:
    """What the run's main body offers, file by file, and what it does not build, each with the reason the run's
    own files give (a LOD the profile leaves out, a format nothing writes yet, the profile's `not_built` list)."""
    out: list[dict[str, Any]] = []
    if body is None:
        return [{"kind": "Game body", "status": "not built", "reason": "stage 4 has not written 04-body/ in this run"}]
    files = {f["path"].split("/")[-1]: f for f in body["files"]}
    sec = body.get("sections") or {}
    profile = _dict((stage4 or {}).get("profile"))
    profile_lods = [x for x in _list(profile.get("lods")) if isinstance(x, str)]
    for name, kind in DELIVERABLE_KINDS.items():
        f = files.get(name)
        if f:
            item = {"kind": kind, "status": "built", **f}
            lod = name[:-4] if name.endswith(".glb") else None
            if lod and lod in body["glb"]:
                item["triangles"] = body["glb"][lod].get("triangles")
            out.append(item)
        elif name == "low.glb":
            reason = (f"profile {profile.get('name')} builds {', '.join(profile_lods)} only"
                      if profile_lods and "low" not in profile_lods else "no low.glb in this body folder")
            out.append({"kind": kind, "status": "not in profile" if profile_lods else "missing", "reason": reason})
    rbxm = sorted(p.name for p in (run / body["dir"]).glob("*.rbxm")) if (run / body["dir"]).is_dir() else []
    if rbxm:
        out.append({"kind": "R15 model for Roblox", "status": "built", **(file_entry(run, f"{body['dir']}/{rbxm[0]}")
                                                                          or {})})
    else:
        out.append({"kind": "R15 model for Roblox", "status": "not built",
                    "reason": "no .rbxm file in the body folder: nothing in the pipeline exports one yet"})
    if (run / "results.html").is_file():
        out.append({"kind": "Stage report", "status": "built", **(file_entry(run, "results.html") or {})})
    for item in sec.get("not_built") or []:
        out.append({"kind": item["name"][:1].upper() + item["name"][1:], "status": "not built",
                    "reason": item["reason"]})
    return out


def run_detail(run: Path) -> dict[str, Any]:
    stages = {key: stage_summary(run, key, label) for key, label in STAGES}
    chain = []
    for step in CHAIN:
        imgs = [{"path": p, "exists": (run / p).is_file()} for p in step["images"]]
        card = {"key": step["key"], "label": step["label"], "stage": step["stage"], "images": imgs}
        if "sheet" in step:
            card["sheet"] = step["sheet"] if (run / step["sheet"]).is_file() else None
        chain.append(card)
    sc = scores(run)
    total_seconds = _total(s.get("seconds") for s in stages.values())
    total_cost = _total(s.get("cost_usd") for s in stages.values())
    found = bodies(run, (sc or {}).get("bodies"))
    main = next((b for b in found if b["dir"] == "04-body"), None)
    stage4, _ = _read_json(run / "stage-4.json")
    return {
        "id": run.name,
        "stages": [stages[k] for k, _ in STAGES],
        "chain": chain,
        "images": images(run),
        "view_meta": view_meta(run),
        "bodies": found,
        "scores": sc,
        "budget_meters": budget_meters(main["sections"] if main else None),
        "deliverables": deliverables(run, main, stage4 if isinstance(stage4, dict) else None),
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
        status[key] = ("unreadable" if err or (rec is not None and not isinstance(rec, dict)) else "not run"
                       if rec is None else "pass" if rec.get("ok") is True else "gate failed")
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
