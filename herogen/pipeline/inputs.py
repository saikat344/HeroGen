"""Input checks: is the sketch usable at all? Run in stages 0 and 1 before any image is generated.

Each check is a measured check (`checks.check`) with a ``verdict``: "pass", "warn" or "reject". A rejected
check is gated and stops the run with the stage's status "input_rejected"; a warning is recorded with a
``message`` for the user and the run goes on.
"""

from __future__ import annotations

from typing import Any

from herogen.pipeline import checks as C

Check = dict[str, Any]

VERDICTS = ("pass", "warn", "reject")

# Stage 0. Muse Image works from the sketch's pixels; a side under 256 px leaves too little to keep.
MIN_SIDE_PX = 256


def input_check(name: str, value: float, threshold: float, op: str, on_fail: str, message: str,
                note: str = "") -> Check:
    """A check whose failure has the verdict ``on_fail`` ("reject" or "warn"). Only a check that rejects is
    gated. ``message`` tells the user what is wrong and what to do; it is kept only when the check fails."""
    if on_fail not in ("reject", "warn"):
        raise ValueError(f"on_fail must be 'reject' or 'warn', not {on_fail!r}")
    out = C.check(name, value, threshold, op=op, gate=on_fail == "reject", note=note)
    out["verdict"] = "pass" if out["pass"] else on_fail
    if not out["pass"]:
        out["message"] = message
    return out


def rejections(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.get("verdict") == "reject"]


def warnings(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.get("verdict") == "warn"]


def rejection_error(checks: list[Check]) -> str:
    """The error of a rejected input: every rejecting check's message, in order."""
    return "; ".join(c["message"] for c in rejections(checks))


def size_check(width: int, height: int) -> Check:
    side = min(width, height)
    return input_check(
        "min_side_px", side, MIN_SIDE_PX, ">=", "reject",
        f"the sketch is too small: its shorter side is {side} px and HeroGen needs at least {MIN_SIDE_PX} px. "
        "Scan or export it at a higher resolution",
    )
