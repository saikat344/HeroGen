"""Held-out views of a run, read from `stage-2h.json` and `heldout/`, in every format the pipeline has written.

Three formats exist:

- `views`: a record with `details.views`, one entry per generated view (`front-left-30`, `front-right-60`, ...),
  each with its status (`accepted`, `needs review` or `rejected`), the status the checks alone gave
  (`machine_status`), a human review decision when there is one, the yaw the checks found, and its files. Its
  checks are named `<view>.<check>`.
- `single`: an older record for one view, named in `details.view` (`front-left-34`), with unprefixed checks. It
  has no review step: the view passed or failed its checks.
- `files`: no record, only images in `heldout/` (runs from before stage 2h had a record of its own).

Nothing here decides whether a view is good: every status is the record's own, or "not recorded".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from herogen.lab.runindex import (SAFE_NAME, _dict, _list, _num, _read_json, _text, normalise_check,
                                  stage_summary)

RECORD = "stage-2h.json"
STATUSES = ("accepted", "needs review", "rejected")
# Status -> the group the page shows the view in.
# A view that failed its checks in the older single-view format was never reviewed, so it is not "rejected".
GROUP = {"accepted": "accepted", "passed checks": "accepted", "needs review": "needs review",
         "rejected": "rejected", "failed checks": "failed checks", "not recorded": "not recorded"}
_FILE_VIEW = re.compile(r"^(front|back)-(left|right)-\d+$")


def view_label(name: str) -> str:
    """`front-left-30` -> `Front-left 30°`; other names as they are."""
    m = re.match(r"^(front|back)-(left|right)-(\d+)$", name)
    return f"{m.group(1).capitalize()}-{m.group(2)} {m.group(3)}°" if m else name


def _file(run: Path, rel: Any) -> str | None:
    """`rel` when it names a file inside the run's `heldout/` folder that exists, else None."""
    if not isinstance(rel, str) or not rel.startswith("heldout/"):
        return None
    parts = rel.split("/")
    if any(not SAFE_NAME.match(p) for p in parts):
        return None
    return rel if (run / rel).is_file() else None


def _decision(v: Any) -> dict[str, Any] | None:
    d = _dict(v)
    if not d:
        return None
    return {
        "decision": _text(d.get("decision")) or None,
        "by": _text(d.get("by")) or None,
        "at": d["at"] if isinstance(d.get("at"), str) else None,
        "note": _text(d.get("note")) or None,
        "from_status": _text(d.get("from_status")) or None,
    }


def _window(v: Any) -> list[float] | None:
    if isinstance(v, list) and len(v) == 2 and all(_num(x) is not None for x in v):
        return [_num(v[0]), _num(v[1])]
    return None


def _checks_for(checks: list[dict[str, Any]], view: str) -> list[dict[str, Any]]:
    prefix = view + "."
    return [c for c in checks if c["name"].startswith(prefix)]


def _summarise_view(entry: dict[str, Any]) -> dict[str, Any]:
    checks = entry["checks"]
    entry["checks_failed_gate"] = sum(1 for c in checks if c["gate"] and not c["pass"])
    entry["group"] = GROUP.get(entry["status"], "not recorded")
    entry["label"] = view_label(entry["name"])
    return entry


def _views_format(run: Path, rec: dict[str, Any], checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for name, v in _dict(_dict(rec.get("details")).get("views")).items():
        if not isinstance(name, str) or not SAFE_NAME.match(name) or not isinstance(v, dict):
            continue
        status = v.get("status") if v.get("status") in STATUSES else "not recorded"
        files = _dict(v.get("files"))
        attempts = [a for a in _list(v.get("attempts")) if isinstance(a, dict)]
        kept = v.get("kept_attempt")
        out.append(_summarise_view({
            "name": name,
            "status": status,
            "machine_status": v.get("machine_status") if v.get("machine_status") in STATUSES else None,
            "review": _decision(v.get("review")),
            "found_yaw_deg": _num(v.get("found_yaw_deg")),
            "target_yaw_deg": _num(v.get("target_yaw_deg")),
            "yaw_window_deg": _window(v.get("yaw_window_deg")),
            "facing": _text(v.get("facing")) or None,
            "image": _file(run, files.get("image")),
            "mask": _file(run, files.get("mask")),
            "attempts": len(attempts),
            "kept_attempt": kept if isinstance(kept, int) and not isinstance(kept, bool) else None,
            "checks": _checks_for(checks, name),
        }))
    return out


def _single_format(run: Path, rec: dict[str, Any], checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    name = _dict(rec.get("details")).get("view")
    if not isinstance(name, str) or not SAFE_NAME.match(name):
        return []
    passed = rec.get("ok") is True
    return [_summarise_view({
        "name": name,
        "status": "passed checks" if passed else "failed checks",
        "machine_status": None,
        "review": None,
        "found_yaw_deg": None,
        "target_yaw_deg": None,
        "yaw_window_deg": None,
        "facing": None,
        "image": _file(run, f"heldout/{name}.png"),
        "mask": _file(run, f"heldout/{name}-mask.png"),
        "attempts": len(_list(_dict(rec.get("details")).get("attempts"))),
        "kept_attempt": None,
        "checks": checks,
    })]


def _files_format(run: Path) -> list[dict[str, Any]]:
    try:
        names = sorted(p.stem for p in (run / "heldout").glob("*.png") if _FILE_VIEW.match(p.stem))
    except OSError:
        return []
    return [_summarise_view({
        "name": n, "status": "not recorded", "machine_status": None, "review": None, "found_yaw_deg": None,
        "target_yaw_deg": None, "yaw_window_deg": None, "facing": None, "image": _file(run, f"heldout/{n}.png"),
        "mask": _file(run, f"heldout/{n}-mask.png"), "attempts": None, "kept_attempt": None, "checks": [],
    }) for n in names]


def heldout(run: Path) -> dict[str, Any] | None:
    """The run's held-out views; None when it has neither a `stage-2h.json` nor any image in `heldout/`."""
    rec, err = _read_json(run / RECORD)
    if err:
        return {"format": "unreadable", "error": err, "views": [], "record": stage_summary(run, "2h", "held-out")}
    if rec is not None and not isinstance(rec, dict):
        return {"format": "unreadable", "error": f"{RECORD} is not a JSON object", "views": [],
                "record": stage_summary(run, "2h", "held-out")}
    if rec is None:
        views = _files_format(run)
        return {"format": "files", "views": views, "record": None, "notes": "", "reference_note": None,
                "read_by": None, "decisions": []} if views else None
    checks = [normalise_check(c) for c in _list(rec.get("checks")) if isinstance(c, dict)]
    details = _dict(rec.get("details"))
    if isinstance(details.get("views"), dict):
        fmt, views = "views", _views_format(run, rec, checks)
    else:
        fmt, views = "single", _single_format(run, rec, checks)
    return {
        "format": fmt,
        "views": views,
        "record": stage_summary(run, "2h", "held-out"),
        "notes": _text(rec.get("notes")),
        "reference_note": _text(details.get("reference_note")) or None,
        "read_by": _text(details.get("read_by")) or None,
        "decisions": [d for d in (_decision(x) for x in _list(details.get("decisions"))) if d],
    }
