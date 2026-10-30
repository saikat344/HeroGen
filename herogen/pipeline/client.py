"""The one model client for herogen/pipeline: Muse Image, SAM 3.1 and Muse Spark image questions, with record and
replay.

Modes:
- ``live``: call Meta's Model API, then save a request summary and the response to the cache.
- ``replay``: read the cache only. A missing entry raises ``ReplayMiss``; nothing touches the network.

Only results are cached. A call that fails raises ``ApiError`` and leaves no entry, so a replay never repeats
a failure (an entry an earlier version wrote for a failed call replays as ``ReplayMiss``).

The cache key is a hash of the request (model, endpoint, prompt, parameters and the SHA-256 of every input
image), so a replay returns exactly what the live call returned for the same inputs. Images are stored as
PNG files next to a JSON entry; SAM masks are stored decoded (one PNG per instance) so replay needs no mask
decoder. The API key is read from ``MODEL_API_KEY`` at call time, sent only in the Authorization header,
and never stored, printed or logged.

Live calls follow the retry policy and the shared per-key limiter in `herogen.pipeline.retry`. Every HTTP
attempt is appended to ``<cache>/calls-ledger.jsonl`` (model, endpoint, request key, status, outcome, time
taken, wait before the next attempt; never a header or the key). An attempt that may have reached the model
without an answer coming back (it timed out or was cut off on our side) is recorded there and in the call log
as outcome "unknown": it may be billed, so the cost estimate counts it. A Muse Spark call is priced by the
token usage it reports, with a count it did not report at its bound (`answer_cost_usd`): so a failed answer
that is retried is priced from its usage, and an attempt with no answer at its output-token cap.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import io
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from herogen.pipeline import filelock, retry

API_BASE = "https://api.meta.ai/v1"
IMAGE_MODEL = "muse-image-1.0"
SAM_MODEL = "sam-3.1"
TEXT_MODEL = "muse-spark-1.3"

# List-price estimates in USD used to report a cost per stage. Estimates, not billed actuals.
PRICE_PER_IMAGE_USD = {IMAGE_MODEL: 0.01, SAM_MODEL: 2.50 / 1000.0}
# Text and vision calls are priced by tokens: (input, output) USD per million tokens; reasoning counts as output.
PRICE_PER_MTOK_USD = {TEXT_MODEL: (1.25, 4.25)}
COST_BASIS = (
    "estimate: call count x configured list price (muse-image $0.01/image, sam-3.1 $0.0025/image; "
    "muse-spark-1.3 $1.25/$4.25 per million input/output tokens)"
)

ENV_KEY = "MODEL_API_KEY"
TIMEOUT_S = 240.0  # socket timeout of one attempt
# Per-call deadline over all attempts and waits. One SAM call has sat on an open connection for about 4 minutes
# before answering, so a deadline allows a slow attempt plus retries. A Muse Spark comparison is a responses
# call like SAM's and has its deadline.
DEADLINE_S = {SAM_MODEL: 600.0, IMAGE_MODEL: 900.0, TEXT_MODEL: 600.0}
LEDGER = "calls-ledger.jsonl"

log = logging.getLogger(__name__)


class ClientError(RuntimeError):
    """A model call failed (HTTP error, bad response, or no key in live mode)."""


class ReplayMiss(ClientError):
    """Replay mode found no cache entry for a request."""


class ApiError(ClientError):
    """The Model API itself failed: an HTTP or network error after the retries, or a response that can't be
    read. Never cached as a result, so a replay never repeats it. A stage stops on it; the one caller that
    records it instead is the held-out face cue, which then sends the view to human review."""


@dataclass
class SamInstance:
    """One SAM 3.1 instance: a full-image boolean mask plus its box (left, top, right, bottom; exclusive)."""

    mask: np.ndarray
    box: tuple[int, int, int, int]
    confidence: float | None = None


@dataclass
class CallLog:
    """What a client did, for stage records: models used, image count per model, estimated cost."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def add(self, model: str, kind: str, key: str, cached: bool, outcome: str = "",
            cost_usd: float | None = None) -> None:
        """``outcome`` "unknown": the call may have reached the model but no answer came back (counted in the
        cost, since it may be billed). ``cost_usd``: the call's own estimate (token-priced models), used in
        place of the per-image price."""
        call: dict[str, Any] = {"model": model, "kind": kind, "key": key, "cached": cached}
        if outcome:
            call["outcome"] = outcome
        if cost_usd is not None:
            call["cost_usd"] = round(cost_usd, 6)
        self.calls.append(call)

    def cost_usd(self, since: int = 0) -> float:
        return round(sum(call_cost(c) for c in self.calls[since:] if not c["cached"]), 6)

    def models(self, since: int = 0) -> list[str]:
        seen: list[str] = []
        for c in self.calls[since:]:
            if c["model"] not in seen:
                seen.append(c["model"])
        return seen


def call_cost(call: dict[str, Any]) -> float:
    """Estimated USD of one logged call: its token-priced cost when it has one, else the per-image price."""
    if "cost_usd" in call:
        return float(call["cost_usd"])
    return PRICE_PER_IMAGE_USD.get(call["model"], 0.0)


def token_cost_usd(model: str, usage: dict[str, Any] | None) -> float:
    if model not in PRICE_PER_MTOK_USD:
        raise ClientError(f"no token price configured for {model}")
    usage = usage or {}
    pin, pout = PRICE_PER_MTOK_USD[model]
    return (int(usage.get("input_tokens") or 0) * pin + int(usage.get("output_tokens") or 0) * pout) / 1e6


def answer_cost_usd(model: str, usage: Any, max_output_tokens: int) -> float:
    """Estimated USD of a token-priced attempt that may be billed, from the usage it reported. A count it did not
    report (no usage, or no such field) is taken at its bound: output at ``max_output_tokens``, the most the call
    allows, and input at 0. So an attempt that may be billed is never counted at $0 unless it reported that."""
    u = usage if isinstance(usage, dict) else {}

    def count(name: str) -> int | None:
        v = u.get(name)
        return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None

    out = count("output_tokens")
    return token_cost_usd(model, {"input_tokens": count("input_tokens") or 0,
                                  "output_tokens": max_output_tokens if out is None else out})


def jpeg_bytes(img: Image.Image, max_side: int, quality: int) -> bytes:
    """RGB JPEG, scaled down (Lanczos) so its longer side is at most ``max_side``."""
    im = img.convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def request_key(summary: dict[str, Any]) -> str:
    """Stable cache key for a request summary (which holds image hashes, never image bytes)."""
    canon = json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha(canon.encode("utf-8"))[:24]


def _scrub(text: str) -> str:
    key = os.environ.get(ENV_KEY, "").strip()
    if key and len(key) >= 4:
        text = text.replace(key, "redacted")
    return re.sub(r"(?i)bearer\s+\S+", "bearer redacted", text)


class ModelClient:
    """Muse Image generate/edit, SAM 3.1 text-prompt segmentation and Muse Spark image comparison, through one
    record/replay cache."""

    def __init__(self, cache_dir: Path | str, mode: str = "replay", log: CallLog | None = None, *,
                 policy: retry.RetryPolicy | None = None, limiter: retry.SharedLimiter | None = None) -> None:
        """``policy`` and ``limiter`` default to `retry.RetryPolicy()` and the shared limiter of the key in use."""
        if mode not in ("live", "replay"):
            raise ValueError(f"mode must be 'live' or 'replay', got {mode!r}")
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.log = log if log is not None else CallLog()
        self.policy = policy or retry.RetryPolicy()
        self.limiter = limiter

    # ---- public calls -------------------------------------------------------------------------------

    def generate_image(self, prompt: str, *, size: str = "1024x1024", attempt: int = 0) -> Image.Image:
        """Muse Image generation. ``attempt`` > 0 asks again for the same request (a new sample, its own cache key);
        the API takes no seed, so a new sample is the only way to draw again."""
        body = {"model": IMAGE_MODEL, "prompt": prompt, "n": 1, "size": size, "output_format": "png"}
        summary: dict[str, Any] = {"endpoint": "images/generations", **body}
        if attempt:
            summary["attempt"] = attempt
        return self._image_call("images/generations", body, summary, kind="generate")

    def edit_image(
        self, prompt: str, images: list[Image.Image], *, size: str = "1024x1024", attempt: int = 0
    ) -> Image.Image:
        """Muse Image edit. ``attempt`` > 0 asks again for the same request (a new sample, its own cache key)."""
        pngs = [png_bytes(im.convert("RGB")) for im in images]
        body = {
            "model": IMAGE_MODEL,
            "prompt": prompt,
            "images": [{"image_url": _data_url(p)} for p in pngs],
            "n": 1,
            "size": size,
            "output_format": "png",
        }
        summary = {
            "endpoint": "images/edits",
            "model": IMAGE_MODEL,
            "prompt": prompt,
            "images_sha256": [_sha(p) for p in pngs],
            "n": 1,
            "size": size,
            "output_format": "png",
        }
        if attempt:
            summary["attempt"] = attempt
        return self._image_call("images/edits", body, summary, kind="edit")

    def segment(self, image: Image.Image, text: str) -> list[SamInstance]:
        """SAM 3.1 concept segmentation: every instance of ``text`` in ``image``, as full-size masks."""
        rgb = image.convert("RGB")
        png = png_bytes(rgb)
        summary = {
            "endpoint": "responses",
            "model": SAM_MODEL,
            "text": text,
            "image_sha256": _sha(png),
            "metadata": {"include_confidence": "true"},
        }
        key = request_key(summary)
        entry = self._load(key)
        if entry is not None and "api_error" in entry and self.mode == "replay":
            # Written by an earlier version, which cached failed calls. A failed call is not a result, so
            # replay has nothing to return for this request; a live run asks again.
            raise ReplayMiss(f"no SAM result cached for key {key} (text={text!r}), only a failed call "
                             f"({_scrub(str(entry['api_error']))[:200]}); make it again live")
        if entry is not None and "api_error" not in entry:
            self.log.add(SAM_MODEL, "segment", key, cached=True)
            if "instances" not in entry:  # a paid response saved before its masks were decoded
                return self._decode_and_save(key, entry, rgb.size)
            return self._load_instances(key, entry, rgb.size)
        if self.mode == "replay":
            raise ReplayMiss(f"no cached SAM response for key {key} (text={text!r})")
        body = {
            "model": SAM_MODEL,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text},
                        {"type": "input_image", "image_url": _data_url(png)},
                    ],
                }
            ],
            "metadata": {"include_confidence": "true"},
        }
        started = time.monotonic()
        # A failed call raises `ApiError` and is never cached: a failure is not a result to replay.
        resp = self._post("responses", body, model=SAM_MODEL, kind="segment", key=key)
        # The call is billed now, so log it and keep the raw response before decoding can fail.
        self.log.add(SAM_MODEL, "segment", key, cached=False)
        # An incomplete or failed response can carry no output, which would read as "nothing found".
        require_completed("responses", resp, required=True)
        entry = {"request": summary, "seconds": round(time.monotonic() - started, 2),
                 "output_text": _output_text(resp)}
        self._save_entry(key, entry)
        return self._decode_and_save(key, entry, rgb.size)

    def compare_images(
        self,
        prompt: str,
        images: list[Image.Image],
        *,
        schema: dict[str, Any],
        model: str = TEXT_MODEL,
        reasoning_effort: str = "low",
        max_output_tokens: int = 3000,
        image_max_side: int = 1024,
        jpeg_quality: int = 90,
        attempt: int = 0,
    ) -> dict[str, Any]:
        """A Responses API call with images and a strict JSON schema; returns the parsed JSON object.

        Images are sent as JPEG data URLs scaled to ``image_max_side``. Cost is estimated from the token usage
        (`answer_cost_usd`: a count the answer did not report is taken at its bound).
        The answer is checked against ``schema`` here too (``schema_errors``), live and on replay: an empty,
        unparsable or non-conforming answer raises ``ClientError`` naming what is wrong (it is never guessed).
        An incomplete answer raises `ApiError` and is never cached.
        """
        jpgs = [jpeg_bytes(im, image_max_side, jpeg_quality) for im in images]
        summary: dict[str, Any] = {
            "endpoint": "responses",
            "model": model,
            "prompt": prompt,
            "images_sha256": [_sha(j) for j in jpgs],
            "schema": schema,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
        }
        if attempt:
            summary["attempt"] = attempt
        key = request_key(summary)
        entry = self._load(key)
        if entry is not None:
            self.log.add(model, "compare", key, cached=True)
            return _conforming(_parse_json_object(entry["output_text"], key), schema, key)
        if self.mode == "replay":
            raise ReplayMiss(f"no cached responses entry for key {key} (model={model})")
        if model not in PRICE_PER_MTOK_USD:  # checked before the paid call, so its cost is never lost
            raise ClientError(f"no token price configured for {model}")
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content += [{"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(j).decode("ascii")}
                    for j in jpgs]
        body = {
            "model": model,
            "input": [{"type": "message", "role": "user", "content": content}],
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": reasoning_effort},
            "text": {"format": {"type": "json_schema", "name": schema["name"], "strict": True,
                                "schema": schema["schema"]}},
        }
        started = time.monotonic()
        # An attempt that may be billed is priced from the usage it reported, if any (else at the output cap).
        resp = self._post("responses", body, model=model, kind="compare", key=key,
                          usage_cost_usd=lambda usage: answer_cost_usd(model, usage, max_output_tokens))
        # Answered, so it may be billed: priced from its usage, with any count it did not report at its bound.
        cost = answer_cost_usd(model, resp.get("usage"), max_output_tokens)
        self.log.add(model, "compare", key, cached=False, cost_usd=cost)
        # An incomplete answer (out of output tokens, say) is a failure, not a result: raised and never cached.
        require_completed("responses", resp, required=True)
        entry = {"request": summary, "seconds": round(time.monotonic() - started, 2), "status": resp.get("status"),
                 "output_text": _output_text(resp), "usage": resp.get("usage"), "cost_usd": round(cost, 6)}
        self._save_entry(key, entry)
        return _conforming(_parse_json_object(entry["output_text"], key), schema, key)

    def _decode_and_save(self, key: str, entry: dict[str, Any], size: tuple[int, int]) -> list[SamInstance]:
        instances = decode_sam_output(entry["output_text"], size)
        entry["instances"] = [
            {"box": list(inst.box), "confidence": inst.confidence, "mask": f"{key}-mask-{i}.png"}
            for i, inst in enumerate(instances)
        ]
        for i, inst in enumerate(instances):
            self._save_png(f"{key}-mask-{i}.png", Image.fromarray((inst.mask * 255).astype(np.uint8)))
        self._save_entry(key, entry)
        return instances

    # ---- internals ----------------------------------------------------------------------------------

    def _image_call(self, endpoint: str, body: dict[str, Any], summary: dict[str, Any], kind: str) -> Image.Image:
        key = request_key(summary)
        entry = self._load(key)
        if entry is not None:
            self.log.add(IMAGE_MODEL, kind, key, cached=True)
            return Image.open(self.cache_dir / entry["image"]).convert("RGB")
        if self.mode == "replay":
            raise ReplayMiss(f"no cached {endpoint} response for key {key}")
        started = time.monotonic()
        resp = self._post(endpoint, body, model=IMAGE_MODEL, kind=kind, key=key)
        try:
            require_completed(endpoint, resp, required=False)
        except ApiError:
            self.log.add(IMAGE_MODEL, kind, key, cached=False)  # answered, so it may be billed
            raise
        data = resp.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise ApiError(f"{endpoint}: response has no data[0]")
        b64 = data[0].get("b64_json")
        if not isinstance(b64, str):
            raise ApiError(f"{endpoint}: data[0] has no b64_json")
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        self._save_png(f"{key}.png", img)
        entry = {
            "request": summary,
            "seconds": round(time.monotonic() - started, 2),
            "image": f"{key}.png",
            "image_size": list(img.size),
            "revised_prompt": data[0].get("revised_prompt"),
            "usage": resp.get("usage"),
        }
        self._save_entry(key, entry)
        self.log.add(IMAGE_MODEL, kind, key, cached=False)
        return img

    def _post(self, endpoint: str, body: dict[str, Any], *, model: str, kind: str, key: str,
              usage_cost_usd: Callable[[Any], float] | None = None) -> dict[str, Any]:
        """POST to the Model API under the retry policy and the shared limiter (`herogen.pipeline.retry`),
        each attempt a line in the ledger and each retry a logged warning. Raises `ApiError` when the
        attempts or the deadline run out, or at once on a status that is never retried.

        ``usage_cost_usd``: what an attempt of unknown outcome is counted at, given the usage it reported (None
        when no usage came back); required for a token-priced model, which has no per-image price. Other models
        are counted at their per-image price."""
        if model in PRICE_PER_MTOK_USD and usage_cost_usd is None:
            raise ClientError(f"a {model} call needs the cost to count for an attempt of unknown outcome")

        def unknown_outcome(usage: Any = None) -> None:  # it may have reached the model: logged, counted in the cost
            cost = usage_cost_usd(usage) if usage_cost_usd is not None else None
            self.log.add(model, kind, key, cached=False, outcome="unknown", cost_usd=cost)

        api_key = os.environ.get(ENV_KEY, "").strip()
        if not api_key:
            raise ClientError(f"{ENV_KEY} is not set; live mode needs it in the environment")
        data = json.dumps(body).encode("utf-8")
        policy = self.policy
        limiter = self.limiter or retry.SharedLimiter.for_key(api_key)
        deadline_s = DEADLINE_S.get(model, DEADLINE_S[SAM_MODEL])
        deadline = policy.clock() + deadline_s
        paid_edit = model == IMAGE_MODEL  # each answered Muse call is a new, billed generation
        last = "no attempt made"
        for attempt in range(1, policy.max_attempts + 1):
            slot, slot_wait = limiter.acquire(policy.wall() + (deadline - policy.clock()))
            if slot is None:
                raise ApiError(f"{endpoint} failed: no call slot came free before the {deadline_s:.0f} s deadline "
                               f"(attempt {attempt}; last: {last})")
            started = policy.clock()
            status: int | None = None
            headers: dict[str, str] = {}
            outcome = "failed"
            parsed: Any = None
            try:
                req = urllib.request.Request(f"{API_BASE}/{endpoint}", data=data, method="POST", headers={
                    "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                    "Accept": "application/json"})
                # The deadline holds while an answer is read too: no socket wait outlasts it, and the body is
                # read in pieces with the deadline checked after each (`_read_body`).
                timeout = min(TIMEOUT_S, max(0.01, deadline - policy.clock()))
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    status = r.status
                    raw = _read_body(r, policy, deadline, deadline_s).decode("utf-8", "replace")
                try:
                    parsed = json.loads(raw)
                except ValueError:
                    parsed = None
                outcome = "ok" if isinstance(parsed, dict) else "unreadable"
                last = f"HTTP {status}" if outcome == "ok" else f"HTTP {status}: the response is not a JSON object"
                if outcome == "unreadable":  # answered, so it may be billed
                    unknown_outcome()
                as_http = retry.failed_answer_status(parsed) if outcome == "ok" else None
                if as_http is not None:  # answered "failed" for a transient reason: retried like that status
                    last = f"HTTP {status} with status 'failed' ({_scrub(json.dumps(parsed.get('error')))[:200]})"
                    status, outcome = as_http, "failed"
                    unknown_outcome(parsed.get("usage"))  # answered: may be billed, priced from its usage
            except urllib.error.HTTPError as e:
                status, headers = e.code, dict(e.headers.items()) if e.headers else {}
                last = f"HTTP {e.code}: {_error_body(e)}"
                if retry.maybe_billed(status, paid_edit=paid_edit):  # a gateway error after generating, maybe
                    outcome = "unknown"
                    unknown_outcome()
            except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
                sending = isinstance(e, urllib.error.URLError)
                reason = e.reason if sending else e
                status = None  # no whole answer came back, whatever status line came first
                # urllib wraps only errors while connecting or sending in URLError. Any other error came while
                # waiting for or reading the answer, so the request went out and may have reached the model;
                # so may one that timed out or was cut off while sending.
                if not sending or isinstance(reason, (TimeoutError, ConnectionResetError, ConnectionAbortedError)):
                    outcome = "unknown"
                    unknown_outcome()
                last = _scrub(f"{type(reason).__name__}: {reason}")[:300]
            seconds = policy.clock() - started
            final = outcome in ("ok", "unreadable") or not retry.retryable(status, paid_edit=paid_edit) \
                or attempt == policy.max_attempts
            wait, why = (0.0, "") if final else policy.wait(attempt - 1, headers)
            left = max(0.0, deadline - policy.clock())
            gives_up = not final and (wait > left or left <= 0.0)
            # Others on the key are held off only for a wait this call really starts (the limiter caps it again).
            limiter.release(slot, ok=outcome == "ok", rate_limited=status == 429,
                            hold_s=0.0 if final or gives_up else wait)
            self._ledger({"model": model, "endpoint": endpoint, "key": key, "attempt": attempt, "status": status,
                          "outcome": outcome, "seconds": round(seconds, 3), "slot_wait_s": round(slot_wait, 3),
                          "wait_s": None if final or gives_up else round(wait, 3), "wait_reason": why or None,
                          "error": None if outcome == "ok" else last})
            if outcome == "ok":
                return parsed
            if outcome == "unreadable":
                raise ApiError(f"{endpoint}: {last}")
            if final or gives_up:
                break
            log.warning("%s %s attempt %d/%d failed (%s); retrying in %.1f s (%s)", model, endpoint, attempt,
                        policy.max_attempts, last[:120], wait, why)
            policy.sleep(wait)
        if not retry.retryable(status, paid_edit=paid_edit):
            raise ApiError(f"{endpoint} failed: {last} (not retried)")
        if attempt == policy.max_attempts:
            raise ApiError(f"{endpoint} failed after {attempt} attempts: {last}")
        if left <= 0.0:
            raise ApiError(f"{endpoint} failed after {attempt} attempt(s): {last}; the {deadline_s:g} s deadline "
                           f"has passed")
        raise ApiError(f"{endpoint} failed after {attempt} attempt(s): {last}; the next wait would pass the "
                       f"{deadline_s:.0f} s deadline")

    def _ledger(self, row: dict[str, Any]) -> None:
        """Append one attempt to the cache's ledger (under an exclusive lock, so parallel runs never interleave)."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"time": datetime.now().astimezone().isoformat(timespec="milliseconds"), **row},
                          sort_keys=True) + "\n"
        with open(self.cache_dir / LEDGER, "a") as f, filelock.exclusive(f.fileno()):
            f.write(line)
            f.flush()

    def _load(self, key: str) -> dict[str, Any] | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _save_entry(self, key: str, entry: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")

    def _save_png(self, name: str, img: Image.Image) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        img.save(self.cache_dir / name, format="PNG")

    def _load_instances(self, key: str, entry: dict[str, Any], size: tuple[int, int]) -> list[SamInstance]:
        out = []
        for inst in entry.get("instances", []):
            m = np.array(Image.open(self.cache_dir / inst["mask"]).convert("L")) > 127
            if m.shape != (size[1], size[0]):
                raise ClientError(f"cached mask {inst['mask']} has the wrong size")
            out.append(SamInstance(mask=m, box=tuple(inst["box"]), confidence=inst.get("confidence")))
        return out


def require_completed(endpoint: str, resp: dict[str, Any], *, required: bool) -> None:
    """Raise `ApiError` unless the response says it completed: its ``status`` is "completed" (the responses
    endpoint always sends one, so ``required``; Muse Image sends none, so only a status it does send is
    checked) and it carries no ``error``. Only a completed response is a result."""
    status = resp.get("status")
    if ((required or status is not None) and status != "completed") or resp.get("error"):
        detail = resp.get("error") or resp.get("incomplete_details")
        raise ApiError(f"{endpoint}: the response status is {status!r}, not 'completed'"
                       + (f" ({_scrub(json.dumps(detail))[:200]})" if detail else ""))


def _read_body(r: Any, policy: retry.RetryPolicy, deadline: float, deadline_s: float) -> bytes:
    """A response body, read one socket read at a time (each waits at most the socket timeout), with the call's
    deadline checked after each: a server that trickles bytes can't hold the call past it. Past the deadline
    the read stops with `TimeoutError`, like a socket timeout."""
    pieces: list[bytes] = []
    while True:
        piece = r.read1(65536)
        if not piece:
            return b"".join(pieces)
        pieces.append(piece)
        if policy.clock() > deadline:
            raise TimeoutError(f"the answer was still arriving at the {deadline_s:g} s deadline")


def _error_body(e: urllib.error.HTTPError) -> str:
    try:
        return _scrub(e.read().decode("utf-8", "replace")[:300])
    except (OSError, http.client.HTTPException) as err:  # the body itself can be cut off
        return f"(body unreadable: {type(err).__name__})"


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,), "array": (list,), "string": (str,), "boolean": (bool,), "integer": (int,),
    "number": (int, float), "null": (type(None),),
}
_SCHEMA_KEYWORDS = {"type", "properties", "required", "additionalProperties", "items", "enum", "description"}


def schema_errors(value: Any, schema: dict[str, Any], path: str = "answer") -> list[str]:
    """Where ``value`` breaks ``schema``: the JSON Schema subset the vision schemas use (type, properties,
    required, additionalProperties false, items, enum). A keyword outside that subset is an error, so a schema
    is never checked only in part."""
    unknown = sorted(set(schema) - _SCHEMA_KEYWORDS)
    if unknown:
        return [f"{path}: schema keyword(s) {', '.join(unknown)} can't be checked"]
    kind = schema.get("type")
    if kind is not None:
        if kind not in _JSON_TYPES:
            return [f"{path}: schema type {kind!r} can't be checked"]
        ok = isinstance(value, _JSON_TYPES[kind]) and not (kind in ("integer", "number") and isinstance(value, bool))
        if not ok:
            return [f"{path} should be {kind}, got {type(value).__name__}"]
    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} is {value!r}, not one of {schema['enum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        errors += [f"{path}.{k} is missing" for k in schema.get("required", []) if k not in value]
        if schema.get("additionalProperties") is False:
            errors += [f"{path}.{k} is not in the schema" for k in value if k not in props]
        for k, sub in props.items():
            if k in value:
                errors += schema_errors(value[k], sub, f"{path}.{k}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += schema_errors(item, schema["items"], f"{path}[{i}]")
    return errors


def _conforming(answer: dict[str, Any], schema: dict[str, Any], key: str) -> dict[str, Any]:
    errors = schema_errors(answer, schema["schema"])
    if errors:
        raise ClientError(f"responses entry {key} does not match schema {schema['name']!r}: " + "; ".join(errors))
    return answer


def _parse_json_object(text: str, key: str) -> dict[str, Any]:
    if not text.strip():
        raise ClientError(f"responses entry {key} has no output text")
    try:
        out = json.loads(text)
    except ValueError as e:
        raise ClientError(f"responses entry {key} is not JSON: {e}") from e
    if not isinstance(out, dict):
        raise ClientError(f"responses entry {key} is not a JSON object")
    return out


def _output_text(resp: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in resp.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "".join(chunks)


def decode_sam_output(text: str, size: tuple[int, int]) -> list[SamInstance]:
    """Decode SAM 3.1 output text into full-image masks. Needs ``meta_sam_parser`` (live runs only).

    Each mask raster spans its box, so it is resized to the box and pasted into a full-size canvas.
    """
    if not text.strip():
        return []
    try:
        import meta_sam_parser as msp  # noqa: PLC0415  (optional; only live runs decode)
    except ImportError as e:
        raise ClientError("decoding SAM masks needs meta_sam_parser on the path (live mode only)") from e
    try:
        parser = msp.image_segmentation_format().create_parser()
        parser.push(text if text.endswith("\n") else text + "\n")
        result = parser.finish(msp.CompletedOutcome()).result
    except Exception as e:  # noqa: BLE001  (an unreadable response is the API's failure, kept in the cache)
        raise ApiError(f"SAM output could not be decoded: {type(e).__name__}: {e}"[:300]) from e
    latest: dict[str, Any] = {}
    for rec in result.records:
        if rec.kind == "mask":
            latest[rec.object_id] = rec
    width, height = size
    out = []
    for rec in latest.values():
        mw, mh = int(rec.mask.width), int(rec.mask.height)
        raster = np.frombuffer(msp.decode_mask_to_raster(rec.mask), dtype=np.uint8).reshape(mh, mw)
        b = rec.bounds
        left, top, right, bottom = int(b.left), int(b.top), int(b.right), int(b.bottom)
        bw, bh = max(1, right - left), max(1, bottom - top)
        small = Image.fromarray((raster > 0).astype(np.uint8) * 255).resize((bw, bh), Image.NEAREST)
        full = np.zeros((height, width), dtype=bool)
        full[top : top + bh, left : left + bw] = np.array(small)[: height - top, : width - left] > 127
        out.append(SamInstance(mask=full, box=(left, top, right, bottom), confidence=rec.confidence))
    return out
