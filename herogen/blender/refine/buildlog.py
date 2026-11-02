"""Build-log events for texture refine: each try's gate result, each view's outcome, waves and versions.

Every number comes from the measurements the stage record already holds (`views[].attempts[].checks`), so the log
and stage-3f.json always agree. Reasons are mapped to fixed identifiers; the record's sentences never leave here.
Without a build-log file (`herogen.pipeline.events`) every function is a no-op.
"""
from __future__ import annotations

from herogen.pipeline import events

STEP = "3f"

# Record reasons (`views[].reason`, `attempts[].reason`) as build-log reason identifiers.
WHY = {
    "outline moved": "outline_moved",
    "new face outside the render face": "second_face",
    "hair outline, scale or colour agreement": "hair_mismatch",
    "cancelled: lower attempt accepted": "lower_try_accepted",
    "head_not_found": "head_not_found",
    "render outline not measurable": "outline_not_measurable",
    "beauty reference has no facial features": "missing_features",
    "not attempted: API cap": "spend_cap",
    "not attempted: model error": "model_error",
    "not attempted: face unavailable": "face_unavailable",
}


def why(reason) -> str:
    reason = str(reason or "")
    if reason in WHY:
        return WHY[reason]
    if reason.startswith("missing facial features"):
        return "missing_features"
    if reason.startswith("API cap"):
        return "spend_cap"
    # Anything else is an exception class name recorded for a failed request.
    return "model_error"


def measures(trial: dict, limits: dict | None = None) -> dict:
    """A try's gate numbers as a flat map: outline overlap and drift with their limits, hair agreement and scale
    change, and the share of the repaint's face outside the render's face."""
    checks = trial.get("checks") or {}
    m = {}
    for key, value in checks.items():
        if type(value) in (int, float):
            m["overlap" if key == "iou" else key] = round(float(value), 4)
    for key, value in (limits or {}).items():
        if type(value) in (int, float):
            m[key] = value
    scale = (trial.get("registration") or {}).get("scale")
    if type(scale) in (int, float) and "agreement" in checks:
        m["scale_change"] = round(abs(float(scale) - 1), 4)
    face = (checks.get("parts") or {}).get("face")
    if isinstance(face, dict) and type(face.get("outside")) in (int, float):
        m["face_outside"] = face["outside"]
        from .imaging import PART_EXTRA_MAX
        m["face_outside_max"] = PART_EXTRA_MAX
    return dict(list(m.items())[:events.MAX_M])


def view_key(cam: dict, hair: bool) -> str:
    return ("hair-" if hair else "") + str(cam.get("name", ""))


def try_result(view: str, trial: dict, ok: bool, reason=None, limits=None, number=None) -> None:
    """One try judged against its gates (accepted, or rejected with the reason). ``number`` is the try's slot
    when it runs as one of several parallel tries; otherwise the record's own try number."""
    events.emit("decision", step=STEP, what="view_result", view=view, try_=number or trial.get("try"), ok=ok,
                why=None if ok else why(reason), m=measures(trial, limits) or None)


def try_cancelled(view: str, number: int, private: dict, calls: list) -> None:
    """A hedged try stopped because a lower try was accepted. ``paid`` is 1 when its repaint request had already
    been billed, 0 when it was stopped before the request."""
    trial = (private.get("attempts") or [{}])[-1]
    from herogen.pipeline.client import IMAGE_MODEL
    paid = any(c.get("model") == IMAGE_MODEL and not c.get("cached") and (c.get("cost_usd") or 0) > 0
               for c in calls)
    events.emit("decision", step=STEP, what="view_result", view=view, try_=trial.get("try", number), ok=False,
                why="lower_try_accepted", m={**measures(trial), "paid": int(paid)})


def view_selected(view: str, number: int, tries: int) -> None:
    """The view keeps the repaint of try ``number`` (the lowest accepted try)."""
    events.emit("decision", step=STEP, what="view_selected", view=view, try_=number, ok=True, m={"tries": tries})


def view_kept(view: str, reason, tries: int) -> None:
    """No try passed: the view keeps its original paint."""
    events.emit("decision", step=STEP, what="view_kept", view=view, ok=False, why=why(reason),
                m={"tries": tries} if tries else None)


def view_skipped(view: str, reason) -> None:
    events.emit("skip", step=STEP, what="view", view=view, why=why(reason))


def wave(number: int, views: int, tries: int, from_version: int = 0) -> None:
    m = {"wave": number, "views": views, "tries": tries}
    if from_version:
        m["from_version"] = from_version
    events.emit("decision", step=STEP, what="wave", n=views * tries, m=m)


def rendered(number: int, cameras: int, seconds: float) -> None:
    events.emit("local", step=STEP, what="views_rendered", s=seconds, m={"wave": number, "cameras": cameras})


def head_probe(found: bool, seconds: float) -> None:
    """The front render was searched for the head (the face close-up needs it)."""
    events.emit("local", step=STEP, what="head_probe", ok=bool(found), s=seconds)


def version(n: int, wave_number, accepted: int, views: int, rejected: int, pending: int) -> None:
    m = {"rejected": rejected, "pending": pending}
    if type(wave_number) is int:
        m["wave"] = wave_number
    events.emit("version", step=STEP, n=n, passed=accepted, of=views, m=m)


def skipped(reason: str) -> None:
    """The whole step kept its unrefined texture (``spend_cap``, ``views_rejected``)."""
    events.emit("skip", step=STEP, what="texture_refine", why=reason)


def extra_try(view: str, number: int) -> None:
    """Every earlier try of the view was rejected by its gates, so one more (conditional) try was started."""
    events.emit("decision", step=STEP, what="extra_try", view=view, try_=number)
