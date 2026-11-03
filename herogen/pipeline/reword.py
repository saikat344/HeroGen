"""A character brief reworded once after the image model refused it under its content policy (fast lane only).

When Muse Image refuses a request that carried the user's brief (stage p, or stage 1's colour brief) with
``content_policy_violation``, the service (`herogen.service.runner`) asks Muse Spark, once per job, to rewrite the
brief as an original character with no brand, franchise, character, real-person or logo references
(``brief-reword-v1`` in ``herogen/prompts``), then runs the refused step once more with the rewrite. Nothing here
retries anything: one rewrite call, recorded in full.

``reword.json`` in the run folder is the job's record of it: the original brief, the rewrite, the names the model
said it removed, the reason, its cost and its status. While its status is ``used``, the rewrite is the job's brief:
stage p draws from it, stage 1's colour brief reads it (`brief`), and a cover redo copies it with the run
(`herogen.pipeline.redo.fork_run`). Without the file nothing changes, so the full lane, uploads and runs that were
never refused behave exactly as before.

A rewrite is refused here (``RewordRejected``), never repaired, when it is empty, too long, unchanged, or still
contains a name the model listed as removed.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from herogen import prompts

FILE = "reword.json"
REASON = "content_policy"
USED, REFUSED_AGAIN, FAILED = "used", "refused_again", "rewrite_failed"


class RewordRejected(ValueError):
    """The model's rewrite can't be used: empty, over the limit, unchanged, or still naming what it removed."""

    def __init__(self, message: str, candidate: str = "") -> None:
        super().__init__(message)
        self.candidate = candidate


def read(run_dir: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((Path(run_dir) / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write(run_dir: Path, data: dict[str, Any]) -> None:
    p = Path(run_dir) / FILE
    part = p.with_name(p.name + ".part")
    part.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(part, p)


def brief(run_dir: Path) -> str | None:
    """The reworded brief while it is the job's brief (status ``used``), else None."""
    data = read(run_dir)
    if data and data.get("status") == USED and isinstance(data.get("reworded"), str) and data["reworded"]:
        return data["reworded"]
    return None


def request_prompt(version: prompts.PromptVersion, text: str) -> str:
    """The rewrite request with the cleaned brief in its one slot."""
    slot = version.data["user_text"]["slot"]
    template = version.step("reword")["prompt"]
    if template.count(slot) != 1:
        raise prompts.RegistryError(f"{version.name}: the reword prompt must hold exactly one {slot}")
    return template.replace(slot, text)


def _mentions(text: str, name: str) -> bool:
    """Whether ``text`` still holds ``name``, ignoring case, spaces and punctuation ("Iron-Man" holds "Ironman").
    A name under three letters must appear as a whole word."""
    letters = "".join(re.findall(r"\w", name.casefold()))
    if len(letters) < 3:
        return bool(letters) and re.search(rf"(?<!\w){re.escape(letters)}(?!\w)", text.casefold()) is not None
    return letters in "".join(re.findall(r"\w", text.casefold()))


def check(original: str, answer: dict[str, Any]) -> tuple[str, list[str]]:
    """(rewrite, removed names) from the model's answer, cleaned with the brief's own text rules; raises
    `RewordRejected` when it can't be used."""
    from herogen.pipeline.prompt_sketch import PromptRejected, clean_user_text

    rules = prompts.selected("prompt_sketch").data["user_text"]
    removed = [str(n).strip() for n in answer.get("removed_names") or [] if str(n).strip()]
    raw = answer.get("reworded")
    candidate = " ".join(str(raw).split()) if isinstance(raw, str) else ""
    try:
        text = clean_user_text(raw, max_chars=int(rules["max_chars"]), delimiters=(rules["open"], rules["close"]))
    except PromptRejected as e:
        raise RewordRejected(f"the rewrite can't be used: {e}", candidate) from None
    if text.casefold() == " ".join(original.split()).casefold():
        raise RewordRejected("the rewrite is the same as the brief", text)
    still = [n for n in removed if _mentions(text, n)]
    if still:
        raise RewordRejected(f"the rewrite still names {len(still)} of the removed names", text)
    return text, removed


def rewrite(client, original: str, *, version: prompts.PromptVersion | None = None) -> dict[str, Any]:
    """One Muse Spark call: the rewrite of ``original`` as a record (status ``used``). A Model API failure
    propagates as it is; an unusable answer raises `RewordRejected`."""
    version = version or prompts.selected("brief_reword")
    step = version.step("reword")
    p = step["params"]
    t0 = time.monotonic()
    calls0 = len(client.log.calls)
    client.purpose = "reword brief"   # names the call in the build log and the call records
    answer = client.compare_images(request_prompt(version, original), [], schema=step["schema"], model=step["model"],
                                   reasoning_effort=p["reasoning_effort"], max_output_tokens=p["max_output_tokens"])
    text, removed = check(original, answer)
    return {"status": USED, "reason": REASON, "original": original, "reworded": text, "removed_names": removed,
            "model": step["model"], "prompt_version": version.ref(),
            "cost_usd": client.log.cost_usd(calls0), "seconds": round(time.monotonic() - t0, 2)}


__all__ = ["FILE", "RewordRejected", "brief", "check", "read", "request_prompt", "rewrite", "write"]
