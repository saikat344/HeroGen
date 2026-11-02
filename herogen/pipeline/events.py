"""Per-run build log: what the pipeline is doing, one small JSON event per line in ``<run>/events.jsonl``.

The service sets ``HEROGEN_EVENTS_FILE`` for every stage subprocess, so model calls, GPU calls, step boundaries and
local sub-steps land in one append-only file the web layer can read while the job runs. Without that variable (and
without an explicit ``file=``) every function here does nothing, so the standalone CLIs and their tests are
unchanged.

Events are structured, never prose: ``emit`` accepts only the fields of a fixed schema (``validate``). ``kind``,
``who``, ``outcome``, ``state`` and ``op`` are closed sets; ``what``, ``why``, ``code``, ``name`` and ``target`` are
short lowercase identifiers; ``m`` maps identifiers to finite numbers. There is no free-text field, so a prompt, a
path, a key or an exception message cannot be written by accident. An event that does not fit is dropped (logged
at debug level): a build log line is never a reason for a stage to fail.

Writing: one ``write()`` of a whole line on an ``O_APPEND`` descriptor, under the exclusive file lock the call
ledger uses, so threads and processes never interleave. A reader can still see a partial last line while it is
being written; readers (``read``) only return complete lines. A run keeps at most ``MAX_EVENTS`` events or
``MAX_BYTES`` bytes; past that only the kinds in ``KEEP_PAST_CAP`` are written, after one ``truncated`` event.

Context: the step of an event comes from an explicit ``step=``, else ``step()``/``set_step()``, else
``HEROGEN_EVENTS_STEP`` (set per stage subprocess, so worker threads inherit it). The attempt comes from
``HEROGEN_EVENTS_ATTEMPT``. ``call()`` brackets one model or GPU call with a start and an end event sharing an
``id``; code inside it can add waits, retries, cost and measurements to that call through ``current_call()``.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import math
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from herogen.pipeline import filelock

ENV_FILE = "HEROGEN_EVENTS_FILE"
ENV_STEP = "HEROGEN_EVENTS_STEP"
ENV_ATTEMPT = "HEROGEN_EVENTS_ATTEMPT"
FILE = "events.jsonl"
VERSION = 1
MAX_LINE = 1024          # bytes per event line, newline included
MAX_EVENTS = 2000
MAX_BYTES = 512 * 1024
HARD_MAX_BYTES = 2 * MAX_BYTES   # even the kept kinds stop here

KINDS = frozenset((
    "job_start", "queue", "step_start", "step_end", "call_start", "call_end", "call_wait", "call_retry",
    "gpu_wait", "gpu_end", "local", "image", "check", "checks", "decision", "version", "skip", "recover",
    "error", "requeue", "cancel", "model_ready", "job_end", "truncated",
))
JOB_LEVEL = frozenset(("job_start", "queue", "requeue", "cancel", "model_ready", "job_end", "truncated"))
KEEP_PAST_CAP = frozenset(("step_start", "step_end", "error", "skip", "version", "requeue", "cancel",
                           "model_ready", "job_end"))
WHO = frozenset(("muse-image", "sam-3.1", "muse-spark", "sam-3d-objects", "sam-3d-body", "herogen", "service"))
STEPS = frozenset(("p", "0", "1", "2", "3", "3f", "4", "6", "5"))
OUTCOMES = frozenset(("ok", "failed", "reused", "cancelled", "unknown",
                      "passed", "rejected", "skipped", "kept"))
STATES = frozenset(("succeeded", "rejected", "failed", "canceled", "stopped"))
EXPORTS = frozenset(("ready", "unavailable"))
OPS = frozenset(("<", "<=", "==", ">=", ">", "!="))
TARGETS = frozenset(("character", "person", "creature", "feet", "foot", "boot", "shoe", "leg", "face", "head",
                     "hair", "other"))
# Run files that are images the job JSON shows, by the key it shows them under (`job.images`).
IMAGE_FILES = {"p-design.png": "design", "p-full-body.png": "full_body", "00-sketch.png": "sketch",
               "01-cover.png": "cover", "02-turnaround.png": "turnaround",
               **{f"views/{v}.png": v for v in ("front", "right", "back", "left", "top")}}

IDENT = re.compile(r"[a-z0-9][a-z0-9_]{0,47}")
GPU = re.compile(r"[a-z0-9][a-z0-9_-]{0,23}")
VIEW = re.compile(r"[a-z][a-z-]{0,23}[+-][0-9]{1,3}")
CALL_ID = re.compile(r"[0-9a-f]{12}")
MAX_M = 16
MAX_NAMES = 20

log = logging.getLogger(__name__)


def _int(lo: int, hi: int):
    def check(v: Any) -> Any:
        if type(v) is not int or not lo <= v <= hi:
            raise ValueError("an integer in range")
        return v
    return check


def _num(v: Any) -> Any:
    if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
        raise ValueError("a finite number >= 0")
    return round(v, 6) if type(v) is float else v


def _seconds(v: Any) -> Any:
    v = _num(v)
    return round(v, 3) if type(v) is float else v


def _bool(v: Any) -> bool:
    if type(v) is not bool:
        raise ValueError("a boolean")
    return v


def _pattern(rx: re.Pattern):
    def check(v: Any) -> str:
        if type(v) is not str or not rx.fullmatch(v):
            raise ValueError(f"a string matching {rx.pattern}")
        return v
    return check


def _one_of(values: frozenset):
    def check(v: Any) -> str:
        if type(v) is not str or v not in values:
            raise ValueError(f"one of {sorted(values)}")
        return v
    return check


def _measures(v: Any) -> dict[str, Any]:
    if type(v) is not dict or not v or len(v) > MAX_M:
        raise ValueError(f"a map of 1 to {MAX_M} measurements")
    out = {}
    for k, x in v.items():
        if type(k) is not str or not IDENT.fullmatch(k):
            raise ValueError("measurement names are identifiers")
        if type(x) not in (int, float) or not math.isfinite(x):
            raise ValueError("measurements are finite numbers")
        out[k] = round(x, 6) if type(x) is float else x
    return out


def _names(v: Any) -> list[str]:
    if type(v) is not list or len(v) > MAX_NAMES:
        raise ValueError(f"a list of at most {MAX_NAMES} names")
    return [_pattern(IDENT)(x) for x in v]


_ident = _pattern(IDENT)
FIELDS: dict[str, Any] = {
    "v": _int(VERSION, VERSION), "t": _int(0, 10**14), "kind": _one_of(KINDS), "step": _one_of(STEPS),
    "a": _int(1, 99), "id": _pattern(CALL_ID), "who": _one_of(WHO), "what": _ident, "target": _one_of(TARGETS),
    "view": _pattern(VIEW), "try": _int(1, 9), "outcome": _one_of(OUTCOMES), "code": _ident, "why": _ident,
    "s": _seconds, "usd": _num, "ok": _bool, "gate": _bool, "op": _one_of(OPS), "name": _ident, "m": _measures,
    "i": _int(1, 99), "of": _int(1, 999), "n": _int(0, 100000), "passed": _int(0, 100000),
    "gated": _int(0, 100000), "failed": _names, "advisory": _names, "attempt": _int(1, 99), "next": _int(1, 99),
    "state": _one_of(STATES), "export": _one_of(EXPORTS), "gpu": _pattern(GPU),
}
REQUIRED = ("v", "t", "kind")


def validate(event: Any) -> dict[str, Any]:
    """The event as it may be stored or served (numbers rounded, keys in schema order); `ValueError` names the
    first field that is not in the schema."""
    if type(event) is not dict:
        raise ValueError("an event is a JSON object")
    for k in REQUIRED:
        if k not in event:
            raise ValueError(f"missing {k}")
    out: dict[str, Any] = {}
    for k in FIELDS:
        if k in event:
            try:
                out[k] = FIELDS[k](event[k])
            except ValueError as e:
                raise ValueError(f"{k}: must be {e}") from None
    extra = set(event) - set(FIELDS)
    if extra:
        raise ValueError(f"fields not in the schema: {sorted(extra)}")
    if out["kind"] in JOB_LEVEL and out["kind"] != "requeue" and "step" in out:
        raise ValueError(f"{out['kind']} is a job-level event and has no step")
    return out


def encode(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


# ---- context -------------------------------------------------------------------------------------------------

_step: contextvars.ContextVar[str | None] = contextvars.ContextVar("herogen_events_step", default=None)
_call: contextvars.ContextVar[Call | None] = contextvars.ContextVar("herogen_events_call", default=None)


def target() -> str | None:
    """The file events go to, or None when the build log is off."""
    return os.environ.get(ENV_FILE) or None


def enabled() -> bool:
    return target() is not None


@contextlib.contextmanager
def step(key: Any) -> Iterator[None]:
    token = _step.set(str(key))
    try:
        yield
    finally:
        _step.reset(token)


def set_step(key: Any) -> None:
    """Set the step for the rest of this context (a stage function that owns its thread)."""
    _step.set(str(key))


def current_step() -> str | None:
    return _step.get() or os.environ.get(ENV_STEP) or None


def _attempt() -> int | None:
    raw = os.environ.get(ENV_ATTEMPT, "")
    return int(raw) if raw.isdigit() else None


def who_for(model: str) -> str:
    """The schema's `who` for a model id ("muse-image-1.0" -> "muse-image")."""
    for who in ("muse-image", "muse-spark", "sam-3d-objects", "sam-3d-body", "sam-3.1"):
        if model.startswith(who):
            return who
    return {"sam3d-objects": "sam-3d-objects", "sam3d-body": "sam-3d-body"}.get(model, "herogen")


def slug(text: Any, fallback: str = "other") -> str:
    """A code-defined label ("full-body sketch", "3D shape") as a schema identifier, else ``fallback``."""
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:48]
    return s if IDENT.fullmatch(s) else fallback


def target_for(text: Any) -> str:
    """The SAM 3.1 concept as one of the closed `TARGETS`: never the text itself unless it is in the set."""
    s = slug(text)
    return s if s in TARGETS else "other"


# ---- writing -------------------------------------------------------------------------------------------------

class _Tally:
    """Per process: how many complete lines a file had when this process last looked, so the cap check reads only
    the bytes appended since (by any process)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.seen: dict[str, tuple[int, int, int, bool]] = {}   # path -> (inode, bytes, lines, truncated)

    def scan(self, path: str, fd: int) -> tuple[int, int, bool]:
        st = os.fstat(fd)
        with self.lock:
            ino, size, lines, truncated = self.seen.get(path, (st.st_ino, 0, 0, False))
        if ino != st.st_ino or size > st.st_size:
            size, lines, truncated = 0, 0, False
        if st.st_size > size:
            with open(path, "rb") as r:
                r.seek(size)
                chunk = r.read(st.st_size - size)
            lines += chunk.count(b"\n")
            truncated = truncated or b'"kind":"truncated"' in chunk
            size += len(chunk)
        with self.lock:
            self.seen[path] = (st.st_ino, size, lines, truncated)
        return size, lines, truncated


_tally = _Tally()


def _append(path: str, line: bytes, kind: str) -> bool:
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        with filelock.exclusive(fd):
            size, lines, truncated = _tally.scan(path, fd)
            if size >= HARD_MAX_BYTES:
                return False
            if (size >= MAX_BYTES or lines >= MAX_EVENTS) and kind not in KEEP_PAST_CAP:
                if not truncated:
                    os.write(fd, encode({"v": VERSION, "t": _now_ms(), "kind": "truncated"}))
                return False
            return os.write(fd, line) == len(line)
    finally:
        os.close(fd)


def _now_ms() -> int:
    return int(time.time() * 1000)


def emit(kind: str, *, file: str | os.PathLike | None = None, step: Any = None, **fields: Any) -> dict | None:
    """Append one event and return it as written, or None when the build log is off or the event was dropped.
    ``try_`` is accepted for the ``try`` field; fields set to None are left out. Never raises."""
    path = os.fspath(file) if file is not None else target()
    if not path:
        return None
    try:
        if "try_" in fields:
            fields["try"] = fields.pop("try_")
        event: dict[str, Any] = {"v": VERSION, "t": _now_ms(), "kind": kind}
        key = step if step is not None or kind in JOB_LEVEL else current_step()
        if key is not None:
            event["step"] = str(key)
        attempt = fields.pop("a", None) or _attempt()
        if attempt is not None:
            event["a"] = attempt
        event.update({k: v for k, v in fields.items() if v is not None})
        event = validate(event)
        line = encode(event)
        if len(line) > MAX_LINE:
            raise ValueError(f"event is {len(line)} bytes, over {MAX_LINE}")
        if not _append(path, line, kind):
            return None
        return event
    except (ValueError, TypeError, OSError) as e:
        log.debug("build log event %s dropped: %s", kind, e)
        return None


def image(rel: str, *, step: Any = None) -> dict | None:
    """An `image` event for a run file the job JSON shows (`IMAGE_FILES`); other files are not announced."""
    key = IMAGE_FILES.get(str(rel))
    return emit("image", step=step, what=key) if key else None


# ---- calls ---------------------------------------------------------------------------------------------------

def code_for(exc: BaseException) -> str:
    """A failure class for an exception, from its type and transport attributes only (never its message)."""
    status = getattr(exc, "http_status", None)
    if type(status) is int:
        return code_for_status(status)
    name = type(exc).__name__
    return {"ReplayMiss": "replay_miss", "RemoteError": "remote_error", "TimeoutError": "timeout",
            "CancelledError": "cancelled"}.get(name, "error")


def code_for_status(status: int) -> str:
    if status in (408, 409, 425, 429):
        return f"http_{status}"
    return "http_5xx" if 500 <= status <= 599 else "http_4xx" if 400 <= status <= 499 else "error"


class Call:
    """One model or GPU call between its start and end events. Set ``outcome``, ``code``, ``usd`` or ``m`` before
    the block ends to change what the end event says."""

    def __init__(self, who: str, what: str, *, gpu: bool, step: Any, file: str | None, **fields: Any) -> None:
        self.id = secrets.token_hex(6)
        self.who, self.what, self.gpu = who, what, gpu
        self.step = step if step is not None else current_step()
        self.file = file if file is not None else target()
        self.fields = {("try" if k == "try_" else k): v for k, v in fields.items() if v is not None}
        self.outcome: str | None = None
        self.code: str | None = None
        self.usd = 0.0
        self.m: dict[str, Any] = {}
        self.extra: dict[str, Any] = {}
        self.t0 = time.monotonic()

    @property
    def active(self) -> bool:
        return self.file is not None

    def emit(self, kind: str, **fields: Any) -> dict | None:
        if not self.active:
            return None
        return emit(kind, file=self.file, step=self.step, id=self.id, who=self.who, **fields)

    def add_cost(self, usd: Any) -> None:
        if type(usd) in (int, float) and math.isfinite(usd) and usd > 0:
            self.usd += usd

    def start(self) -> None:
        self.emit("gpu_wait" if self.gpu else "call_start", what=self.what, **self.fields)

    def end(self, exc: BaseException | None) -> None:
        if not self.active:
            return
        outcome = self.outcome or ("ok" if exc is None else "cancelled" if code_for(exc) == "cancelled"
                                   else "failed")
        fields: dict[str, Any] = {"what": self.what, **self.fields, "outcome": outcome}
        if outcome in ("failed", "unknown"):
            fields["code"] = self.code or (code_for(exc) if exc is not None else "error")
        if outcome != "reused":
            fields["s"] = time.monotonic() - self.t0
        fields["usd"] = 0.0 if outcome == "reused" else self.usd
        if self.m:
            fields["m"] = self.m
        fields.update(self.extra)
        self.emit("gpu_end" if self.gpu else "call_end", **fields)


@contextlib.contextmanager
def call(who: str, what: str, *, gpu: bool = False, step: Any = None, file: str | os.PathLike | None = None,
         **fields: Any) -> Iterator[Call]:
    """Bracket one call: ``call_start``/``call_end`` (``gpu_wait``/``gpu_end`` with ``gpu=True``), sharing an id.
    The end says ``failed`` (with a ``code``) when the block raises, ``cancelled`` for a cancelled future."""
    c = Call(who, what, gpu=gpu, step=step, file=os.fspath(file) if file is not None else None, **fields)
    c.start()
    token = _call.set(c)
    try:
        yield c
    except BaseException as e:
        c.end(e)
        raise
    else:
        c.end(None)
    finally:
        _call.reset(token)


def current_call() -> Call | None:
    """The call this code runs inside (same thread), or None."""
    return _call.get()


def mark_reused() -> None:
    """The current call was answered from the cache."""
    c = _call.get()
    if c is not None:
        c.outcome = "reused"


# ---- reading -------------------------------------------------------------------------------------------------

def read(path: Path, offset: int = 0, *, limit: int = 500, max_bytes: int = 256 * 1024
         ) -> tuple[list[tuple[int, dict[str, Any]]], int, bool]:
    """Complete, valid events from byte ``offset``: ([(end offset, event)], next offset, more).

    A trailing partial line is left for the next read. Lines that are not valid JSON or not in the schema are
    skipped (the offset still moves past them). At most ``limit`` events and ``max_bytes`` of file per read;
    ``more`` is True when complete lines remain after the returned ones."""
    try:
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            if offset >= size:
                return [], min(offset, size), False
            f.seek(offset)
            chunk = f.read(min(max_bytes, size - offset))
    except FileNotFoundError:
        return [], 0, False
    end = chunk.rfind(b"\n")
    if end < 0:
        if len(chunk) >= max_bytes:   # one line longer than a whole page: never valid, step over it
            return [], offset + len(chunk), offset + len(chunk) < size
        return [], offset, False
    out: list[tuple[int, dict[str, Any]]] = []
    pos = offset
    for raw in chunk[: end + 1].split(b"\n")[:-1]:
        nxt = pos + len(raw) + 1
        if len(out) >= limit:
            return out, pos, True
        pos = nxt
        if len(raw) + 1 > MAX_LINE:
            continue
        try:
            out.append((pos, validate(json.loads(raw))))
        except (ValueError, UnicodeDecodeError):
            continue
    return out, pos, offset + len(chunk) < size   # unread bytes may hold more complete lines
