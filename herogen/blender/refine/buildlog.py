"""Build-log events for texture refine: each try's gate result, each view's outcome, waves and versions.

Every number comes from the measurements the stage record already holds (`views[].attempts[].checks`), so the log
and stage-3f.json always agree. Reasons are mapped to fixed identifiers; the record's sentences never leave here.
Without a build-log file (`herogen.pipeline.events`) every function is a no-op.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools

from herogen.pipeline import events

STEP = "3f"
_MUTED = contextvars.ContextVar("refine_buildlog_muted", default=False)


@contextlib.contextmanager
def muted():
    """No view, wave or version lines inside: an internal sub-run whose results the main run reports itself.
    Its model requests are still logged as calls (they are real requests), without a try number."""
    token = _MUTED.set(True)
    try:
        yield
    finally:
        _MUTED.reset(token)


def is_muted() -> bool:
    return _MUTED.get()

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


def _quiet(fn):
    """Emit only when the build log is on, and never let working out a line fail the stage."""
    @functools.wraps(fn)
    def emit(*args, **kwargs):
        if not events.enabled() or _MUTED.get():
            return
        try:
            fn(*args, **kwargs)
        except Exception:  # noqa: BLE001 - the log describes the stage; it must never change its outcome
            events.emit("skip", step=STEP, what="build_log_detail", why="unreadable")
    return emit


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


@_quiet
def try_result(view: str, trial: dict, ok: bool, reason=None, limits=None, number=None) -> None:
    """One try judged against its gates (accepted, or rejected with the reason). ``number`` is the try's slot
    when it runs as one of several parallel tries; otherwise the record's own try number."""
    events.emit("decision", step=STEP, what="view_result", view=view, try_=number or trial.get("try"), ok=ok,
                why=None if ok else why(reason), m=measures(trial, limits) or None)


@_quiet
def try_cancelled(view: str, number: int, private: dict, calls: list) -> None:
    """A hedged try stopped because a lower try was accepted. ``paid`` is 1 when any request of the try (its
    repaint or the SAM 3.1 checks on it) may already have been billed, 0 when it was stopped before any;
    ``paid_usd`` is their estimate. Reused (cached) answers cost nothing. A request logged without its own cost
    (the full-lane client, a timeout that may have been billed) counts at its list price."""
    from herogen.pipeline.client import call_cost
    trial = (private.get("attempts") or [{}])[-1]
    spent = sum(call_cost(c) for c in calls if not c.get("cached"))
    events.emit("decision", step=STEP, what="view_result", view=view, try_=number, ok=False,
                why="lower_try_accepted", m={**measures(trial), "paid": int(spent > 0), "paid_usd": round(spent, 6)})


@_quiet
def try_failed(view: str, number: int) -> None:
    """A try stopped on a model or replay error before its gates could judge it."""
    events.emit("decision", step=STEP, what="view_result", view=view, try_=number, ok=False, why="model_error")


def _code(error: dict) -> str | None:
    """A recorded model error's failure class: from the answer's status, else the transport failure."""
    status, network = error.get("http_status"), error.get("network_error")
    if type(status) is int:
        return events.code_for_status(status)
    return ("timeout" if network == "TimeoutError" else "network") if network else None


def _contained(error: dict) -> dict:
    """The failure class and measures of a contained model error: ``m.fallback`` is 1, with the answer's
    ``http_status`` when one came back (none for a timeout or a dropped connection), and ``m.retried`` 1
    when it came from the one resend of a transient error."""
    status = error.get("http_status")
    return {"code": _code(error),
            "m": {"fallback": 1, **({"http_status": status} if type(status) is int else {}),
                  **({"retried": 1} if error.get("retried") else {})}}


@_quiet
def retried(entry: dict, *, view: str | None = None, number: int | None = None, yaw: int | None = None) -> None:
    """A beauty-only request failed with a transient error (``code``, ``m.http_status``) and was resent once,
    identically: ``ok`` says whether the resend answered. A view's try (``view``, ``try``) or a beauty
    reference (``m.yaw``). A failed resend is then contained like any beauty-only error."""
    status = entry.get("http_status")
    ok = entry.get("retry") == "ok"
    events.emit("decision", step=STEP, what="model_retry", view=view, try_=number, ok=ok,
                why=None if ok else "model_error", code=_code(entry),
                m={"retry": 1, **({"yaw": yaw} if yaw is not None else {}),
                   **({"http_status": status} if type(status) is int else {})})


@_quiet
def try_contained(view: str, number: int, error: dict) -> None:
    """The enhanced (beauty) try's model request failed. The stage goes on: the view's ordinary tries decide,
    as if this try had been rejected."""
    events.emit("decision", step=STEP, what="view_result", view=view, try_=number, ok=False, why="model_error",
                **_contained(error))


@_quiet
def reference_contained(yaw: int, error: dict) -> None:
    """A beauty reference's model request failed. The stage goes on without a reference at that yaw (``m.yaw``):
    its views use the standard recipe, and without the front reference the beauty arm is off for the run."""
    fields = _contained(error)
    events.emit("decision", step=STEP, what="beauty_reference", ok=False, why="model_error", code=fields["code"],
                m={"yaw": yaw, **fields["m"]})


@_quiet
def face_source(source, view=None, number=None) -> None:
    """Where the face paint comes from, once the frontal tries are known: a beauty repaint, an ordinary
    (default) repaint, or the ordinary fast run's atlas when no frontal face repaint was accepted."""
    events.emit("decision", step=STEP, what="face_source", why=events.slug(source or "unknown"),
                view=view if source in ("beauty", "default") else None,
                try_=number if source in ("beauty", "default") and type(number) is int else None)


@_quiet
def view_selected(view: str, number: int, tries: int) -> None:
    """The view keeps the repaint of try ``number`` (the lowest accepted try)."""
    events.emit("decision", step=STEP, what="view_selected", view=view, try_=number, ok=True, m={"tries": tries})


@_quiet
def view_kept(view: str, reason, tries: int) -> None:
    """No try passed: the view keeps its original paint."""
    events.emit("decision", step=STEP, what="view_kept", view=view, ok=False, why=why(reason),
                m={"tries": tries} if tries else None)


@_quiet
def view_skipped(view: str, reason) -> None:
    events.emit("skip", step=STEP, what="view", view=view, why=why(reason))


@_quiet
def wave(number: int, views: int, tries: int, from_version: int = 0) -> None:
    m = {"wave": number, "views": views, "tries": tries}
    if from_version:
        m["from_version"] = from_version
    events.emit("decision", step=STEP, what="wave", n=views * tries, m=m)


@_quiet
def rendered(number: int, cameras: int, seconds: float) -> None:
    events.emit("local", step=STEP, what="views_rendered", s=seconds, m={"wave": number, "cameras": cameras})


@_quiet
def head_probe(found: bool, seconds: float) -> None:
    """The front render was searched for the head (the face close-up needs it)."""
    events.emit("local", step=STEP, what="head_probe", ok=bool(found), s=seconds)


@_quiet
def version(n: int, wave_number, accepted: int, views: int, rejected: int, pending: int) -> None:
    m = {"rejected": rejected, "pending": pending}
    if type(wave_number) is int:
        m["wave"] = wave_number
    events.emit("version", step=STEP, n=n, passed=accepted, of=views, m=m)


@_quiet
def skipped(reason: str) -> None:
    """The whole step kept its unrefined texture (``spend_cap``, ``views_rejected``)."""
    events.emit("skip", step=STEP, what="texture_refine", why=reason)


@_quiet
def extra_try(view: str, number: int) -> None:
    """Every earlier try of the view was rejected by its gates, so one more (conditional) try was started."""
    events.emit("decision", step=STEP, what="extra_try", view=view, try_=number)
