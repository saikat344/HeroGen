"""Sapiens 2 pose keypoints for stage 2h, through a record/replay cache like the Model API's.

Modes:
- ``live``: run `herogen/pipeline/sapiens_app.py` on Modal (an ephemeral app, opened on the first uncached call
  and kept for the client's life, so the model loads once per stage-2h run; `close` stops it), then save the
  result to the cache. A failed or timed-out try (CALL_TIMEOUT_S) is tried again after RETRY_WAIT_S; when every
  try fails, `PoseError`.
- ``replay``: read the cache only. A missing entry raises `ReplayMiss`; nothing touches the network.

The cache key hashes the request: the model's pins (Sapiens 2 code commit, weight revision and SHA-256), the
SHA-256 of `sapiens_app.py`, the image's PNG bytes and the person box. Each entry is `<cache>/sapiens-<key>.json`.
Only the keypoints are cached, never the image.
"""
from __future__ import annotations

import ast
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from herogen.pipeline.client import ReplayMiss, png_bytes

HERE = Path(__file__).resolve().parent
APP = HERE / "sapiens_app.py"
POSE_MODEL = "sapiens2-pose-1b"
# Modal list price of the app's GPU (L40S) per second, for estimates only; the bill has the real figure.
PRICE_PER_SECOND = 1.95 / 3600
COST_BASIS = "estimate: GPU seconds of each live call x the L40S list price (model load not counted)"
RETRY_WAIT_S = (10.0, 30.0)  # waits before the second and third try of a live call
# How long one try may take: a cold start (image, weights, model load) measured 61 s, a warm call 1.5 s. A try
# that passes it (a container that never starts, say) is abandoned and tried again.
CALL_TIMEOUT_S = 300.0


class PoseError(RuntimeError):
    """A live Sapiens 2 call failed on every try (Modal missing, the app failing, a timeout)."""


def app_pins() -> dict[str, Any]:
    """What a result depends on besides its inputs: the app's pinned code commit and weights, and its source."""
    consts = {}
    for node in ast.parse(APP.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return {"model": POSE_MODEL, "code": consts["SAPIENS2_COMMIT"], "weights": consts["WEIGHTS"],
            "app_sha256": hashlib.sha256(APP.read_bytes()).hexdigest()[:16]}


def request_key(summary: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


@dataclass
class PoseLog:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def cost_usd(self, since: int = 0) -> float:
        return round(sum(c.get("cost_usd", 0.0) for c in self.calls[since:]), 6)


class PoseClient:
    def __init__(self, cache_dir: Path | str, mode: str = "replay", log: PoseLog | None = None) -> None:
        if mode not in ("live", "replay"):
            raise ValueError(f"mode must be 'live' or 'replay', got {mode!r}")
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.log = log if log is not None else PoseLog()
        self._pins = app_pins()
        self._app = None
        self._pose = None

    def keypoints(self, image: Image.Image, box: list[float]) -> dict[str, tuple[float, float, float]]:
        """name -> (x, y, score) for the person in ``box`` [x0, y0, x1, y1] (image px, exclusive edges)."""
        png = png_bytes(image.convert("RGB"))
        box = [round(float(v), 3) for v in box]
        summary = {**self._pins, "image_sha256": hashlib.sha256(png).hexdigest(), "box": box}
        key = request_key(summary)
        path = self.cache_dir / f"sapiens-{key}.json"
        if path.exists():
            out = json.loads(path.read_text())["response"]
            self.log.calls.append({"model": POSE_MODEL, "key": key, "cached": True, "cost_usd": 0.0})
        elif self.mode == "replay":
            raise ReplayMiss(f"no cached {POSE_MODEL} keypoints for key {key}")
        else:
            out = self._live(png, box)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".part")
            tmp.write_text(json.dumps({"request": summary, "response": out}, sort_keys=True) + "\n")
            tmp.replace(path)
            self.log.calls.append({"model": POSE_MODEL, "key": key, "cached": False,
                                   "cost_usd": round(out.get("seconds", 0.0) * PRICE_PER_SECOND, 6)})
        return {n: (float(xy[0]), float(xy[1]), float(s))
                for n, xy, s in zip(out["names"], out["keypoints"], out["scores"], strict=True)}

    def _live(self, png: bytes, box: list[float]) -> dict[str, Any]:
        errors = []
        for wait in (0.0, *RETRY_WAIT_S):
            time.sleep(wait)
            try:
                if self._pose is None:
                    self._open()
                return self._pose.keypoints.spawn(png, box).get(timeout=CALL_TIMEOUT_S)
            except ImportError as e:
                raise PoseError(f"{POSE_MODEL} needs the modal package (pip install herogen[remote]): {e}") from e
            except Exception as e:  # noqa: BLE001  (Modal raises many types; each try is recorded)
                errors.append(f"{type(e).__name__}: {e}"[:200])
                self.close()
        raise PoseError(f"{POSE_MODEL} failed on every try: {' | '.join(errors)}")

    def _open(self) -> None:
        from herogen.pipeline import sapiens_app

        self._app = sapiens_app.app.run()
        self._app.__enter__()
        self._pose = sapiens_app.Pose()

    def close(self) -> None:
        """Stop the live app, if one is open."""
        app, self._app, self._pose = self._app, None, None
        if app is not None:
            try:
                app.__exit__(None, None, None)
            except Exception:  # noqa: BLE001  (stopping an ephemeral app is best effort)
                pass
