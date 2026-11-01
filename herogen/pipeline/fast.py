"""Explicit speed/demo mode and per-request timing for its stage records."""
from copy import copy
from datetime import datetime
from functools import wraps
import os
import time


def enabled(fast=False):
    return bool(fast or os.environ.get("HEROGEN_FAST_LANE") == "1")


def informational(checks):
    for check in checks:
        check["gate"] = False
    return checks


def tracked(model, kind, *, isolate=False):
    """Keep cache hits, transport attempts and failures in the caller's existing call log."""
    def decorate(fn):
        @wraps(fn)
        def call(self, *args, **kwargs):
            if not getattr(self, "fast", False):
                return fn(self, *args, **kwargs)
            # Each request owns its receipts, including retries. Concurrent requests share the caller's
            # cache and policy but never annotate or settle one another's log slices.
            caller_log = self.log
            if isolate:
                self = copy(self)
                self.log = type(caller_log)()
            start = len(self.log.calls)
            began = datetime.now().astimezone().isoformat(timespec="microseconds")
            t0 = time.monotonic()
            error = None
            try:
                return fn(self, *args, **kwargs)
            except Exception as exc:
                error = type(exc).__name__
                raise
            finally:
                rows = self.log.calls[start:]
                if not rows:
                    row = {"model": model, "kind": kind, "cached": False, "cost_usd": 0.0}
                    self.log.calls.append(row)
                    rows = [row]
                purpose = getattr(self, "purpose", kind)
                if kind == "segment":
                    purpose = f"segment {args[1] if len(args) > 1 else kwargs['text']} ({purpose})"
                for row in rows:
                    row.setdefault("stage", getattr(self, "stage", None))
                    row.setdefault("purpose", purpose)
                    row.setdefault("model", model)
                    row.setdefault("kind", kind)
                    if row.get("cached"):
                        row["cost_usd"] = 0.0
                    elif "cost_usd" not in row:
                        from herogen.pipeline.client import call_cost
                        row["cost_usd"] = call_cost(row)
                    row.setdefault("started", began)
                    row.setdefault("wall_seconds", round(time.monotonic() - t0, 6))
                    row.setdefault("outcome", "ok")
                if error:
                    rows[-1]["outcome"] = error
                if isolate:
                    caller_log.calls.extend(rows)
        return call
    return decorate
