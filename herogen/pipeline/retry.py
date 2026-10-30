"""Retry policy and the shared per-key limiter for Model API calls (used by `client.ModelClient._post`).

Retry policy (`RetryPolicy`):
- Retried: HTTP 408, 409, 425, 429 and 5xx except 501 and 505 (which say the request can never work), and
  network errors (no HTTP status). Never any other 4xx, and never a 409 on a paid generation (Muse Image). A 200 answer whose status is "failed" with error code
  `rate_limit_exceeded` or `server_error` is retried as a 429 or a 500.
- Wait before a retry: the server's `Retry-After` (seconds or an HTTP date; a non-finite or unreadable value
  is ignored) plus 0-1 s of jitter when it sent one; else the largest rate-limit reset header it sent (`retry-after-ms`, `ratelimit-reset`,
  `x-ratelimit-reset`, `x-ratelimit-reset-requests`, `x-ratelimit-reset-tokens`) plus the same jitter; else
  full-jitter exponential backoff, a random time between 0 and min(60 s, 2 s x 2^n) before retry n + 1.
- Limits: at most 8 attempts within a per-call deadline (10 minutes for SAM 3.1, 15 for Muse Image), separate
  from each attempt's socket timeout. A wait that would end past the deadline is not started; the call fails
  at once.

Shared limiter (`SharedLimiter`): every process that calls with one API key shares one limit on calls in
flight, kept in a small JSON state file under HEROGEN_LIMITER_DIR (default ~/.cache/herogen/api-limiter),
named by a SHA-256 prefix of the key (never the key itself) and guarded by an exclusive file lock
(`herogen.pipeline.filelock`). A 429 halves the limit (never below 1) and, when its caller starts the wait it
was given, holds every process off for that wait, at most HOLD_MAX_S (60 s); every
SUCCESSES_PER_STEP successful calls since the last 429 raise the limit by one, up to its maximum. A slot whose
process has died is freed when another process next looks.
"""

from __future__ import annotations

import email.utils
import hashlib
import json
import math
import os
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from herogen.pipeline import filelock

RETRY_STATUSES = frozenset({408, 409, 425, 429})
NEVER_RETRY_5XX = frozenset({501, 505})
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 60.0
MAX_ATTEMPTS = 8
JITTER_S = 1.0


def retryable(status: int | None, *, paid_edit: bool = False) -> bool:
    """Whether an attempt that ended with HTTP ``status`` (None: no response) may be retried. ``paid_edit``: a
    paid call that is not idempotent (a Muse Image generation), never retried on a 409 conflict."""
    if status is None:
        return True
    if paid_edit and status == 409:
        return False
    return status in RETRY_STATUSES or (500 <= status <= 599 and status not in NEVER_RETRY_5XX)


def maybe_billed(status: int | None, *, paid_edit: bool) -> bool:
    """Whether a failed HTTP ``status`` on a paid generation may still have generated and billed the image: any
    5xx but 503 (the service refused it unprocessed), so the attempt is counted as an unknown outcome."""
    return paid_edit and status is not None and 500 <= status <= 599 and status != 503


# A 200 answer whose status is "failed" with one of these error codes is retried as this HTTP status. Any
# other answer that did not complete is returned, and the client stops on it (`client.require_completed`).
FAILED_ANSWER_AS_HTTP = {"rate_limit_exceeded": 429, "server_error": 500}


def failed_answer_status(resp: object) -> int | None:
    """The HTTP status to retry a 200 answer as, when it says it failed for a transient reason; else None."""
    if not isinstance(resp, dict) or resp.get("status") != "failed" or not isinstance(resp.get("error"), dict):
        return None
    return FAILED_ANSWER_AS_HTTP.get(str(resp["error"].get("code")))


def _duration_s(text: str) -> float | None:
    """Seconds in "1.5", "20ms", "6m0s", "1h2m3.5s"; None when unreadable."""
    t = text.strip().lower()
    try:
        v = float(t)
    except ValueError:
        pass
    else:
        return max(0.0, v) if math.isfinite(v) else None
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", t)
    if not parts or "".join(n + u for n, u in parts) != t:
        return None
    scale = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
    return sum(float(n) * scale[u] for n, u in parts)


def retry_after_s(headers: Mapping[str, str], now_epoch: float) -> float | None:
    """`Retry-After` in seconds from now: delta seconds or an HTTP date; None when absent or unreadable."""
    v = _get(headers, "retry-after")
    if v is None:
        return None
    v = v.strip()
    if re.fullmatch(r"\d+(\.\d+)?", v):
        d = float(v)
        return d if math.isfinite(d) else None
    try:
        when = email.utils.parsedate_to_datetime(v)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if when is None:
        return None
    try:
        d = when.timestamp() - now_epoch
    except (OverflowError, OSError, ValueError):
        return None
    return max(0.0, d) if math.isfinite(d) else None


RESET_HEADERS = ("retry-after-ms", "ratelimit-reset", "x-ratelimit-reset", "x-ratelimit-reset-requests",
                 "x-ratelimit-reset-tokens")


def reset_wait_s(headers: Mapping[str, str], now_epoch: float) -> float | None:
    """The largest rate-limit reset wait the server sent, in seconds; None when it sent none."""
    waits = []
    for name in RESET_HEADERS:
        v = _get(headers, name)
        if v is None:
            continue
        if name == "retry-after-ms":
            d = _duration_s(v)
            d = None if d is None else d / 1000.0
        else:
            d = _duration_s(v)
            if d is not None and d > 1e11:  # an epoch time in milliseconds
                d = max(0.0, d / 1000.0 - now_epoch)
            elif d is not None and d > 1e9:  # an epoch time in seconds, not a delta
                d = max(0.0, d - now_epoch)
        if d is not None:
            waits.append(d)
    return max(waits) if waits else None


def _get(headers: Mapping[str, str], name: str) -> str | None:
    for k, v in headers.items():
        if k.lower() == name:
            return v
    return None


@dataclass
class RetryPolicy:
    """How long to wait before each retry, and when to stop. ``sleep``, ``clock`` (monotonic, for the
    deadline), ``wall`` (epoch seconds, for HTTP dates) and ``rand`` are injectable for tests."""

    max_attempts: int = MAX_ATTEMPTS
    base_s: float = BACKOFF_BASE_S
    cap_s: float = BACKOFF_CAP_S
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    wall: Callable[[], float] = time.time
    rand: random.Random = field(default_factory=random.Random)

    def wait(self, retry: int, headers: Mapping[str, str]) -> tuple[float, str]:
        """(seconds, reason) to wait before retry number ``retry`` (0 for the first retry)."""
        ra = retry_after_s(headers, self.wall())
        if ra is not None:
            return ra + self.rand.uniform(0.0, JITTER_S), "Retry-After"
        rs = reset_wait_s(headers, self.wall())
        if rs is not None:
            return rs + self.rand.uniform(0.0, JITTER_S), "rate-limit reset header"
        return self.rand.uniform(0.0, min(self.cap_s, self.base_s * 2.0 ** retry)), "backoff"


LIMITER_DIR_ENV = "HEROGEN_LIMITER_DIR"
# The longest a 429 may hold every process on a key off, whatever wait the server asked for (the backoff cap):
# a bad or huge Retry-After must never block the key for longer. A call that gives up instead of waiting
# (its wait would pass its time limit) holds no one off.
HOLD_MAX_S = BACKOFF_CAP_S
LIMIT_START = 4
LIMIT_MAX = 8
SUCCESSES_PER_STEP = 20


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _SYNCHRONIZE = 0x00100000
    _ERROR_ACCESS_DENIED = 5
    _WAIT_TIMEOUT = 0x102

    def _alive(pid: int) -> bool:
        # Never os.kill(pid, 0) here: on Windows it terminates the process instead of probing it.
        if not 0 < pid <= 0xFFFFFFFF:
            return False
        handle = _kernel32.OpenProcess(_SYNCHRONIZE, False, pid)
        if not handle:
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED  # it exists, but belongs to someone else
        try:
            return _kernel32.WaitForSingleObject(handle, 0) == _WAIT_TIMEOUT  # not yet signalled: running
        finally:
            _kernel32.CloseHandle(handle)
else:
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


class SharedLimiter:
    """A limit on calls in flight shared by every process using one API key (see the module docstring).

    ``stale_s``: a slot held longer than this is freed (its attempt must have ended: the socket timeout plus a
    margin)."""

    def __init__(self, directory: Path | str, key_id: str, *, start: int = LIMIT_START, maximum: int = LIMIT_MAX,
                 stale_s: float = 300.0, poll_s: float = 0.2, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.dir = Path(directory)
        self.state_path = self.dir / f"{key_id}.json"
        self.lock_path = self.dir / f"{key_id}.lock"
        self.start, self.maximum = max(1, start), max(1, maximum)
        self.stale_s, self.poll_s, self.clock, self.sleep = stale_s, poll_s, clock, sleep
        self._local = threading.Lock()
        self._n = 0
        self._token = uuid.uuid4().hex[:8]  # slot ids stay unique across limiter objects in one process

    @staticmethod
    def key_id(api_key: str) -> str:
        """A name for a key's state file: a SHA-256 prefix, from which the key can't be recovered."""
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def for_key(cls, api_key: str, **kw) -> "SharedLimiter":
        directory = os.environ.get(LIMITER_DIR_ENV) or str(Path.home() / ".cache" / "herogen" / "api-limiter")
        return cls(directory, cls.key_id(api_key), **kw)

    def _update(self, change: Callable[[dict, float], object]) -> object:
        """Apply ``change(state, now)`` under the exclusive file lock and save the state."""
        self.dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            filelock.lock(fd)
            try:
                state = json.loads(self.state_path.read_text())
            except (OSError, ValueError):
                state = {}
            state.setdefault("limit", self.start)
            state["limit"] = max(1, min(int(state["limit"]), self.maximum))
            state.setdefault("slots", {})
            state.setdefault("hold_until", 0.0)
            now0 = self.clock()
            hold = float(state["hold_until"]) if isinstance(state["hold_until"], (int, float)) else 0.0
            # A hold from a damaged file or an earlier version never outlasts HOLD_MAX_S from now.
            state["hold_until"] = min(hold, now0 + HOLD_MAX_S) if math.isfinite(hold) else 0.0
            state.setdefault("successes", 0)
            now = self.clock()
            state["slots"] = {s: v for s, v in state["slots"].items()
                              if now - v["since"] < self.stale_s and _alive(int(v["pid"]))}
            out = change(state, now)
            tmp = self.state_path.with_name(f".{self.state_path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(state))
            os.replace(tmp, self.state_path)
            return out
        finally:
            filelock.unlock(fd)
            os.close(fd)

    def acquire(self, deadline: float) -> tuple[str | None, float]:
        """Wait for a slot until ``deadline`` (this limiter's clock). Returns (slot id, seconds waited); the
        id is None when no slot came free in time."""
        with self._local:
            self._n += 1
            slot = f"{os.getpid()}-{self._token}-{self._n}"
        t0 = self.clock()
        while True:
            def take(state: dict, now: float) -> float:
                if now >= state["hold_until"] and len(state["slots"]) < state["limit"]:
                    state["slots"][slot] = {"pid": os.getpid(), "since": now}
                    return 0.0
                return max(self.poll_s, state["hold_until"] - now)
            wait = float(self._update(take))  # type: ignore[arg-type]
            if wait == 0.0:
                return slot, self.clock() - t0
            if self.clock() + min(wait, self.poll_s) > deadline:
                return None, self.clock() - t0
            self.sleep(min(wait, max(self.poll_s, 1.0)))

    def release(self, slot: str, *, rate_limited: bool = False, hold_s: float = 0.0, ok: bool = False) -> int:
        """Free ``slot``. A 429 (``rate_limited``) halves the limit and holds every process off for
        ``hold_s``, at most HOLD_MAX_S (a non-finite or negative hold is none); a success counts toward raising
        it. Returns the limit now."""
        hold = min(max(0.0, hold_s), HOLD_MAX_S) if math.isfinite(hold_s) else 0.0

        def done(state: dict, now: float) -> int:
            state["slots"].pop(slot, None)
            if rate_limited:
                state["limit"] = max(1, state["limit"] // 2)
                state["successes"] = 0
                state["hold_until"] = max(state["hold_until"], now + hold)
            elif ok:
                state["successes"] += 1
                if state["successes"] >= SUCCESSES_PER_STEP and state["limit"] < self.maximum:
                    state["limit"] += 1
                    state["successes"] = 0
            return state["limit"]
        return int(self._update(done))  # type: ignore[arg-type]
