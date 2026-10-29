"""The one model client for herogen/pipeline: Muse Image and SAM 3.1, with record and replay.

Modes:
- ``live``: call Meta's Model API, then save a request summary and the response to the cache.
- ``replay``: read the cache only. A missing entry raises ``ReplayMiss``; nothing touches the network.

The cache key is a hash of the request (model, endpoint, prompt, parameters and the SHA-256 of every input
image), so a replay returns exactly what the live call returned for the same inputs. Images are stored as
PNG files next to a JSON entry; SAM masks are stored decoded (one PNG per instance) so replay needs no mask
decoder. The API key is read from ``MODEL_API_KEY`` at call time, sent only in the Authorization header,
and never stored, printed or logged.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

API_BASE = "https://api.meta.ai/v1"
IMAGE_MODEL = "muse-image-1.0"
SAM_MODEL = "sam-3.1"

# List-price estimates in USD used to report a cost per stage. Estimates, not billed actuals.
PRICE_PER_IMAGE_USD = {IMAGE_MODEL: 0.01, SAM_MODEL: 2.50 / 1000.0}
COST_BASIS = "estimate: call count x configured list price (muse-image $0.01/image, sam-3.1 $0.0025/image)"

ENV_KEY = "MODEL_API_KEY"
TIMEOUT_S = 240.0
RETRIES = 2


class ClientError(RuntimeError):
    """A model call failed (HTTP error, bad response, or no key in live mode)."""


class ReplayMiss(ClientError):
    """Replay mode found no cache entry for a request."""


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

    def add(self, model: str, kind: str, key: str, cached: bool) -> None:
        self.calls.append({"model": model, "kind": kind, "key": key, "cached": cached})

    def cost_usd(self, since: int = 0) -> float:
        return round(sum(PRICE_PER_IMAGE_USD.get(c["model"], 0.0) for c in self.calls[since:] if not c["cached"]), 6)

    def models(self, since: int = 0) -> list[str]:
        seen: list[str] = []
        for c in self.calls[since:]:
            if c["model"] not in seen:
                seen.append(c["model"])
        return seen


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
    """Muse Image generate/edit and SAM 3.1 text-prompt segmentation, through one record/replay cache."""

    def __init__(self, cache_dir: Path | str, mode: str = "replay", log: CallLog | None = None) -> None:
        if mode not in ("live", "replay"):
            raise ValueError(f"mode must be 'live' or 'replay', got {mode!r}")
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.log = log if log is not None else CallLog()

    # ---- public calls -------------------------------------------------------------------------------

    def generate_image(self, prompt: str, *, size: str = "1024x1024") -> Image.Image:
        body = {"model": IMAGE_MODEL, "prompt": prompt, "n": 1, "size": size, "output_format": "png"}
        summary = {"endpoint": "images/generations", **body}
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
        if entry is not None:
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
        resp = self._post("responses", body)
        # The call is billed now, so log it and keep the raw response before decoding can fail.
        self.log.add(SAM_MODEL, "segment", key, cached=False)
        entry = {"request": summary, "seconds": round(time.monotonic() - started, 2),
                 "output_text": _output_text(resp)}
        self._save_entry(key, entry)
        return self._decode_and_save(key, entry, rgb.size)

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
        resp = self._post(endpoint, body)
        data = resp.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise ClientError(f"{endpoint}: response has no data[0]")
        b64 = data[0].get("b64_json")
        if not isinstance(b64, str):
            raise ClientError(f"{endpoint}: data[0] has no b64_json")
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

    def _post(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        key = os.environ.get(ENV_KEY, "").strip()
        if not key:
            raise ClientError(f"{ENV_KEY} is not set; live mode needs it in the environment")
        data = json.dumps(body).encode("utf-8")
        last_err = ""
        for attempt in range(RETRIES + 1):
            req = urllib.request.Request(
                f"{API_BASE}/{endpoint}",
                data=data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                    parsed = json.loads(r.read().decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise ClientError(f"{endpoint}: response is not a JSON object")
                return parsed
            except urllib.error.HTTPError as e:
                detail = _scrub(e.read().decode("utf-8", "replace")[:400])
                last_err = f"HTTP {e.code}: {detail}"
                if e.code != 429 and e.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = _scrub(f"{type(e).__name__}: {e}")[:300]
            if attempt < RETRIES:
                time.sleep(2.0 * (attempt + 1))
        raise ClientError(f"{endpoint} failed: {last_err}")

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
    parser = msp.image_segmentation_format().create_parser()
    parser.push(text if text.endswith("\n") else text + "\n")
    result = parser.finish(msp.CompletedOutcome()).result
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
