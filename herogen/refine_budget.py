"""Serializable per-run texture-refinement spend and pending attempts."""
from __future__ import annotations

import copy
import math


class BudgetStop(RuntimeError):
    """The next attempt would exceed the refinement cap."""


def update(state, action, *, token="", amount=0.0, cap=0.80, prior=0.0):
    """Return a new state. Only an explicit settlement replaces an attempt's reserved cost.

    Tokens make writes idempotent when a store snapshot must be retried. A lost acknowledgement never
    authorizes a request; its reservation remains charged even if the request was never sent.
    """
    if any(not math.isfinite(v) or v < 0 for v in (amount, cap, prior)):
        raise ValueError("texture-refine budget amounts must be finite and nonnegative")
    result = copy.deepcopy(state) if state is not None else {"legacy_usd": prior, "attempts": {}}
    attempts = result["attempts"]
    total = round(result["legacy_usd"] + sum(a["reserved_usd"] if a["cost_usd"] is None else a["cost_usd"]
                                            for a in attempts.values()), 6)
    result["allowed"] = True
    if action == "reserve":
        if not token:
            raise ValueError("a reservation needs a token")
        if token in attempts:
            if attempts[token]["reserved_usd"] != amount:
                raise ValueError("reservation token reused with a different amount")
        elif round(total + amount, 6) > cap:
            result["allowed"] = False
        else:
            attempts[token] = {"reserved_usd": amount, "cost_usd": None}
    elif action == "settle":
        entry = attempts[token]
        if amount > entry["reserved_usd"]:
            raise ValueError("settlement exceeds its reservation")
        if entry["cost_usd"] is not None and entry["cost_usd"] != amount:
            raise ValueError("reservation already settled with a different cost")
        entry["cost_usd"] = amount
    elif action != "read":
        raise ValueError("unknown refinement budget operation")
    result["spent_usd_total"] = round(result["legacy_usd"] + sum(
        a["reserved_usd"] if a["cost_usd"] is None else a["cost_usd"] for a in attempts.values()), 6)
    result["pending_attempt_usd"] = round(sum(a["reserved_usd"] for a in attempts.values()
                                             if a["cost_usd"] is None), 6)
    return result
