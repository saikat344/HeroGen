"""One suggested hero description for the website's Describe box, from Muse Spark.

The website's "Suggest an idea" button (``POST /api/ideas``, `herogen.service.web`) asks for one playful, original
hero described the way the pipeline draws well: one full-body standing character, clear colours and materials, no
brand, franchise or real-person names, family friendly, at most the description's character limit
(``hero-idea-v2`` in ``herogen/prompts``). Each request picks a random seed theme and mood from the prompt version,
so two clicks give different ideas.

`suggest` makes one Muse Spark call and, when the reply can't be used (empty, too short or long, a refusal, or an
answer that does not fit the schema), one more with a fresh seed; nothing else is retried. It never raises for a
model failure: the record says ``ok`` with the idea, or ``failed`` with an error class (``refused`` for a
content-policy refusal, ``model_error``, ``unusable``) and, either way, what the calls cost, so the caller can count
the spend. No model or error text is kept in a failed record.
"""

from __future__ import annotations

import random
import re
import time
from typing import Any

from herogen import prompts

OK, FAILED = "ok", "failed"
REFUSED, MODEL_ERROR, UNUSABLE = "refused", "model_error", "unusable"
ATTEMPTS = 2   # one call, and one more only when the reply can't be used

_QUOTES = "\"'`“”‘’«»„‟‹›"
_LABEL = re.compile(r"^(?:hero\s+)?(?:idea|description|character|suggestion)\s*[:\-–—]\s*", re.IGNORECASE)
_MARKDOWN = re.compile(r"(\*\*|__|`+|^#{1,6}\s+|^\s*(?:[-*•]|\d+[.)])\s+)", re.MULTILINE)
_REFUSAL = re.compile(
    r"\b(?:i\s*(?:am|'m|’m)\s*(?:sorry|unable|not able)|i\s*(?:can(?:no|'|’)?t|cannot|won(?:'|’)?t)\b"
    r"|as an ai\b|i apologi[sz]e|sorry,|unable to (?:help|assist|comply|create|provide)"
    r"|(?:against|violates?) (?:my|the) (?:guidelines|policy|policies))",
    re.IGNORECASE)


class IdeaRejected(ValueError):
    """The model's reply can't be shown: empty, too short or long, a refusal, or not a description."""


def pick_seed(version: prompts.PromptVersion, rng: random.Random | None = None) -> tuple[str, str]:
    """(theme, mood) for one request, at random from the version's seed lists."""
    rng = rng or random.SystemRandom()
    seed = version.data["seed"]
    return rng.choice(seed["themes"]), rng.choice(seed["moods"])


def request_prompt(version: prompts.PromptVersion, theme: str, mood: str) -> str:
    """The request with the seed theme and mood in their one slot each."""
    seed = version.data["seed"]
    template = version.step("idea")["prompt"]
    for slot in (seed["theme_slot"], seed["mood_slot"]):
        if template.count(slot) != 1:
            raise prompts.RegistryError(f"{version.name}: the idea prompt must hold exactly one {slot}")
    return template.replace(seed["theme_slot"], theme).replace(seed["mood_slot"], mood)


def check(answer: Any, version: prompts.PromptVersion | None = None) -> str:
    """The idea from the model's answer, cleaned (quotes, markdown and a leading label removed, then the
    description's own text rules); raises `IdeaRejected` when it can't be used."""
    from herogen.pipeline.prompt_sketch import PromptRejected, clean_user_text

    version = version or prompts.selected("hero_idea")
    limits = version.data["limits"]
    rules = prompts.selected("prompt_sketch").data["user_text"]
    raw = answer.get("description") if isinstance(answer, dict) else None
    if not isinstance(raw, str):
        raise IdeaRejected("the reply holds no description")
    text = _MARKDOWN.sub("", raw)
    text = " ".join(text.split())
    for _ in range(3):   # nested wrappers such as **"..."** or a label inside quotes
        before = text
        text = _LABEL.sub("", text.strip().strip(_QUOTES).strip())
        if text == before:
            break
    if _REFUSAL.search(text):
        raise IdeaRejected("the reply is a refusal")
    try:
        text = clean_user_text(text, max_chars=min(int(limits["max_chars"]), int(rules["max_chars"])),
                               delimiters=(rules["open"], rules["close"]))
    except PromptRejected as e:
        raise IdeaRejected(f"the reply can't be used: {e}") from None
    if len(text) < int(limits["min_chars"]):
        raise IdeaRejected(f"the reply is {len(text)} characters long; ideas need at least {limits['min_chars']}")
    return text


def suggest(client, *, version: prompts.PromptVersion | None = None,
            rng: random.Random | None = None) -> dict[str, Any]:
    """Ask Muse Spark for one hero idea (a second call only when the first reply can't be used). Returns a record:
    ``{"status": "ok", "idea", ...}`` or ``{"status": "failed", "error", ...}``, both with ``cost_usd``, ``model``,
    ``prompt_version``, ``attempts`` and ``seconds``."""
    from herogen.pipeline.client import ApiError, ClientError

    version = version or prompts.selected("hero_idea")
    step = version.step("idea")
    p = step["params"]
    t0 = time.monotonic()
    calls0 = len(client.log.calls)
    client.purpose = "suggest idea"   # names the call in the call records
    rec: dict[str, Any] = {"model": step["model"], "prompt_version": version.ref()}
    error = UNUSABLE
    attempts = 0
    for attempts in range(1, ATTEMPTS + 1):
        theme, mood = pick_seed(version, rng)
        try:
            answer = client.compare_images(request_prompt(version, theme, mood), [], schema=step["schema"],
                                           model=step["model"], reasoning_effort=p["reasoning_effort"],
                                           max_output_tokens=p["max_output_tokens"])
            idea = check(answer, version)
        except ApiError as e:
            if e.error_code == "content_policy_violation":
                error = REFUSED
                break
            if e.http_status is None and e.network_error is None:
                # Answered but incomplete (it ran out of output tokens, say): an unusable reply, asked once more.
                error = UNUSABLE
                continue
            # Any other failed call ends it: the same request would likely fail the same way.
            error = MODEL_ERROR
            break
        except (IdeaRejected, ClientError):
            # ClientError here is an answer that could not be parsed or does not fit the schema.
            error = UNUSABLE
            continue
        rec.update(status=OK, idea=idea, seed={"theme": theme, "mood": mood})
        break
    if rec.get("status") != OK:
        rec.update(status=FAILED, error=error)
    rec.update(attempts=attempts, cost_usd=client.log.cost_usd(calls0), seconds=round(time.monotonic() - t0, 2))
    return rec


__all__ = ["FAILED", "IdeaRejected", "MODEL_ERROR", "OK", "REFUSED", "UNUSABLE", "check", "pick_seed",
           "request_prompt", "suggest"]
