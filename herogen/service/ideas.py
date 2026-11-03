"""Where the website's suggested hero ideas come from (`POST /api/ideas`, `herogen.pipeline.idea`).

The web layer calls one zero-argument function (`Services.ideas`) and gets back a record it can show and count:
``{"status": "ok", "idea", "cost_usd", ...}`` or ``{"status": "failed", "error", "cost_usd", ...}``. `live` makes
the Muse Spark call in this process with ``MODEL_API_KEY`` (dev mode with the local runner, and the Modal
``suggest_idea`` function, which holds the model secret the web container does not); `fake` hands out canned
ideas at no cost (dev mode with the fake runner, and tests). `transportable` keeps only the fields the web layer
reads, so no model text other than a checked idea crosses a process boundary.
"""

from __future__ import annotations

import itertools
import os
import tempfile
import threading
from typing import Any

UNAVAILABLE = "unavailable"   # no model key here: nothing was called
FIELDS = ("status", "idea", "error", "cost_usd", "model", "attempts", "seconds")


def transportable(rec: dict[str, Any]) -> dict[str, Any]:
    """The record's shown and counted fields only (the prompt version as its name)."""
    out = {k: rec[k] for k in FIELDS if k in rec}
    version = rec.get("prompt_version")
    if isinstance(version, dict) and isinstance(version.get("name"), str):
        out["prompt_version"] = version["name"]
    return out


def unknown_cost_usd() -> float:
    """What a request whose outcome is unknown (it failed between here and the model) is counted at: every attempt
    `herogen.pipeline.idea.suggest` may make, each at its output-token cap."""
    from herogen import prompts
    from herogen.pipeline import idea
    from herogen.pipeline.client import answer_cost_usd

    step = prompts.selected("hero_idea").step("idea")
    return round(idea.ATTEMPTS * answer_cost_usd(step["model"], None, step["params"]["max_output_tokens"]), 6)


def live() -> dict[str, Any]:
    """One idea from Muse Spark, in the fast lane's single-attempt mode, through a throwaway cache (every request
    is new, so nothing is ever replayed). The call shares the key's rate limiter with the pipeline."""
    from herogen.pipeline import idea
    from herogen.pipeline.client import ENV_KEY, ModelClient

    if not os.environ.get(ENV_KEY, "").strip():
        return {"status": idea.FAILED, "error": UNAVAILABLE, "cost_usd": 0.0}
    with tempfile.TemporaryDirectory(prefix="herogen-idea-") as cache:
        return transportable(idea.suggest(ModelClient(cache, mode="live", fast=True)))


FAKE_IDEAS = (
    "A cheerful snail mail carrier with a teal shell, a mustard canvas satchel and brass goggles, in a red knit "
    "scarf and tiny leather boots. Full body, standing, plain background.",
    "A dreamy cloud shepherd in a fluffy white wool poncho with sky-blue trim, a curled willow crook and silver "
    "bells on her belt. Full body, standing, plain background.",
    "A brave frog knight in polished copper armour with lily-pad green enamel, a short oak spear and a pale yellow "
    "cape. Full body, standing, plain background.",
)


class Fake:
    """Canned ideas in turn, at no cost (dev mode's fake runner and the tests)."""

    def __init__(self, ideas: tuple[str, ...] = FAKE_IDEAS) -> None:
        self._next = itertools.cycle(ideas)
        self._lock = threading.Lock()

    def __call__(self) -> dict[str, Any]:
        with self._lock:
            text = next(self._next)
        return {"status": "ok", "idea": text, "cost_usd": 0.0, "model": "fake", "attempts": 1, "seconds": 0.0,
                "prompt_version": "fake"}


__all__ = ["FAKE_IDEAS", "Fake", "UNAVAILABLE", "live", "transportable", "unknown_cost_usd"]
