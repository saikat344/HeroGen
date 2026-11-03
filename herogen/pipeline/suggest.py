"""Refine suggestions: one suggested instruction per Refine step for a finished hero, written after stage 3f.

``python -m herogen.pipeline.suggest <run folder> [--cache DIR] [--replay]`` runs right after stage 3f, in the
background and in parallel with stage 4 on the same run folder. It writes ``<run>/refine-suggestions.json``, which
the website reads to set the Refine box's placeholder for the chosen step (``texture``: Texture only, same shape;
``turnaround``: From the views).

What it does:

1. Renders a flat front image of the finished 3f hero: the refine ``render`` job (`herogen.blender.refine.backend`)
   on ``blender-work/texture-refine/prep/work.blend`` with the final atlas ``blender-work/texture-refine/texture.png``,
   through the ``body+0`` camera that 3f records in ``stage-3f.json`` (``views["body+0"].camera``). When 3f recorded
   no ``body+0`` camera (that view was not accepted), the camera is rebuilt the way 3f builds it, from
   ``texture-refine/character.json``. When the work.blend render fails (not a timeout), the refine ``probe`` job
   renders ``03-refined.glb`` through the same front orthographic camera instead. work.blend is opened, never
   saved. The image is ``blender-work/suggest/front.png`` (composited on light grey and cropped to the figure).
2. Asks Muse Spark (``muse-spark-1.3``) one image question through `herogen.pipeline.client`: the cover
   (``01-cover.png``) and the front render, the hero's description (the used rewrite in ``reword.json``, else
   ``stage-p.json`` ``details.user_text``; quoted data in the prompt) and what each step can and cannot change. The
   prompt is the frozen version ``refine-suggest-v1`` (``herogen/prompts``, kind ``refine_suggest``). The answer is
   strict JSON ``{"texture": {"text", "basis"}, "turnaround": {"text", "basis"}}`` with basis ``mismatch`` (fixes
   the most visible difference between cover and render the step can deliver) or ``variation`` (nothing is
   visibly off). The call goes through the record/replay cache (``--cache``), so a rerun replays at no cost.
3. Checks each step's text (`check_text`): non-empty, at most 100 characters, one sentence, and none of the step's
   blocked words (`BLOCKED`): the texture step can't change shape, gear, pose or face; the turnaround step can't
   change the pose, put a prop in a hand, add wind-blown cloth or change the camera, background or view. A step
   that fails gets its static text (`STATIC`) with basis ``static``.
4. Writes the JSON atomically (a temporary file in the run folder, then a rename)::

     {"version": 1, "source": "spark" | "static",
      "texture": {"text", "basis"}, "turnaround": {"text", "basis"},
      "model": "muse-spark-1.3" | null, "prompt": "refine-suggest-v1", "prompt_sha256": "...",
      "cost_usd": <float>, "calls": [<call records>], "error": null | "<type: message>", "created": "<ISO time>"}

   ``cost_usd`` and ``calls`` are the client's own estimate and records (a Muse Spark call priced by its reported
   token usage, with an unreported count at its bound); a Spark call cut off by the deadline is added as one
   record of outcome ``unknown`` priced at its output-token cap. ``model`` is null when no call was made.
   ``blender-work/suggest/check.json`` keeps the model's raw answer and the reason any step was refused.

CLI contract (exit codes):

- 0: the file was written with source ``spark``. An individual step may still be ``static`` after failing its check.
- 3: a model error, timeout or unparseable reply (or the render timed out). The static file is still written, with
  ``error`` filled in, and one line goes to stderr. It is never silently swallowed.
- 1: no file was written, because of missing inputs (run folder, cover, final texture, work.blend or a front
  camera) or an internal error. The reason goes to stderr. A partial file is never written.
- 2: usage error.

Cost lands only in the file's ``cost_usd`` and ``calls``; the website's runner adds it to the job. Nothing here
touches the service database.

Safe alongside stage 4: it reads only 3f outputs (``blender-work/texture-refine/``, ``stage-3f.json``,
``03-refined.glb``), the cover, ``stage-p.json`` and ``reword.json``, and writes only ``blender-work/suggest/``
and ``refine-suggestions.json``. Stage 4 neither reads nor writes either path.

Time: the whole run is bounded. The Blender render has ``RENDER_TIMEOUT_S`` (about 120 s) in total, the glb
fallback included, and the Spark question a total deadline of ``SPARK_DEADLINE_S`` (about 60 s); either running out
gives exit 3 with the static file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from herogen import prompts

FILE = "refine-suggestions.json"
WORK = Path("blender-work") / "suggest"
REFINE = Path("blender-work") / "texture-refine"
COVER = "01-cover.png"
GLB = "03-refined.glb"
STEPS = ("texture", "turnaround")
STATIC = {
    "texture": "Make the jacket deep red leather with brass buttons.",
    "turnaround": "Add a hooded cloak and make the boots dark brown.",
}
MAX_CHARS = 100
SPARK_DEADLINE_S = 60.0
RENDER_TIMEOUT_S = 120.0
VERSION = 1
FRONT = "body+0"
BACKGROUND = (230, 230, 230)
EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_MODEL = 0, 1, 2, 3
NO_DESCRIPTION = "none given; the hero was drawn from an uploaded image"

# Words a step's text may not use, matched as whole words with their plain inflections (`_forms`).
BLOCKED = {
    # Texture only keeps the shape, gear, pose and face.
    "texture": ("add", "remove", "longer", "shorter", "taller", "lengthen", "shorten", "pose", "hold", "held",
                "expression", "smile", "face shape"),
    # From the views keeps the A-pose, moves held props to the back or belt, has no wind and a fixed camera.
    "turnaround": ("pose", "hold", "held", "hand", "in-hand", "wind", "wind-blown", "windswept", "flowing",
                   "billowing", "camera", "background", "view"),
}


class MissingInput(RuntimeError):
    """An input the run needs is absent: exit 1, no file."""


class SuggestTimeout(RuntimeError):
    """The render or the Spark question ran out of time: exit 3, static file."""


def _forms(word: str) -> list[str]:
    forms = {word, word + "s", word + "es", word + "ed", word + "d", word + "ing"}
    if word.endswith("e"):
        forms.add(word[:-1] + "ing")
    return sorted(forms, key=len, reverse=True)


def _blocked_pattern(words: tuple[str, ...]) -> re.Pattern[str]:
    alts = [re.escape(f).replace(r"\ ", r"\s+") for w in words for f in _forms(w)]
    return re.compile(r"(?<![\w-])(?:" + "|".join(alts) + r")(?![\w-])", re.IGNORECASE)


_BLOCKED_RE = {step: _blocked_pattern(words) for step, words in BLOCKED.items()}


def check_text(step: str, text: Any) -> tuple[str | None, str | None]:
    """(cleaned text, None) when ``text`` can be the step's suggestion, else (None, reason)."""
    if not isinstance(text, str) or not text.strip():
        return None, "empty"
    clean = " ".join(text.split())
    if len(clean) > MAX_CHARS:
        return None, f"{len(clean)} characters (at most {MAX_CHARS})"
    if re.search(r"[.!?]\s+\S", clean):
        return None, "more than one sentence"
    hit = _BLOCKED_RE[step].search(clean)
    if hit:
        return None, f"uses {hit.group(0).lower()!r}, which the {step} step can't deliver"
    return clean, None


def static_step(step: str) -> dict[str, str]:
    return {"text": STATIC[step], "basis": "static"}


# ---- inputs -------------------------------------------------------------------------------------------------------

def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def description(run: Path) -> str:
    """The hero's description as quoted data: the used rewrite (``reword.json``), else stage p's user text."""
    from herogen.pipeline import reword

    text = reword.brief(run)
    if not text:
        details = (_read_json(run / "stage-p.json") or {}).get("details") or {}
        text = details.get("user_text") if isinstance(details.get("user_text"), str) else None
    version = prompts.selected("refine_suggest")
    marks = version.data["user_text"]
    text = " ".join((text or "").replace(marks["open"], " ").replace(marks["close"], " ").split())[:400]
    return text or NO_DESCRIPTION


def front_camera(run: Path) -> tuple[dict[str, Any], str]:
    """(the ``body+0`` camera, where it came from): stage-3f.json's view record, else rebuilt from character.json."""
    rec = _read_json(run / "stage-3f.json") or {}
    view = (rec.get("views") or {}).get(FRONT)
    cam = view.get("camera") if isinstance(view, dict) else None
    if isinstance(cam, dict) and {"centre", "angle", "ortho", "res"} <= set(cam):
        return {**cam, "name": FRONT}, "stage-3f.json"
    info = _read_json(run / REFINE / "character.json")
    if info and isinstance(info.get("body_centre"), list) and isinstance(info.get("height"), (int, float)):
        from herogen.blender.refine.cameras import camera
        from herogen.blender.refine.character import BODY_FRAME, BODY_RES

        return camera(FRONT, info["body_centre"], 0, info["height"] * BODY_FRAME, BODY_RES), "character.json"
    raise MissingInput(f"no {FRONT} camera in stage-3f.json or {REFINE}/character.json")


def required_inputs(run: Path) -> None:
    if not run.is_dir():
        raise MissingInput(f"run folder {run} does not exist")
    for rel in (COVER, REFINE / "texture.png", REFINE / "prep" / "work.blend"):
        if not (run / rel).is_file():
            raise MissingInput(f"missing {rel}")


# ---- render -------------------------------------------------------------------------------------------------------

def _timed_out(exc: BaseException) -> bool:
    return isinstance(exc.__cause__, subprocess.TimeoutExpired)


def render_front(run: Path, cam: dict[str, Any], timeout: float = RENDER_TIMEOUT_S,
                 blender: Callable[..., Any] | None = None) -> tuple[Path, dict[str, Any]]:
    """Render the front of the finished hero to ``blender-work/suggest/front.png``; (path, how it was made)."""
    from PIL import Image

    from herogen.blender.bl import BlenderError

    if blender is None:
        from herogen.blender.refine.backend import blender
    work = run / WORK
    work.mkdir(parents=True, exist_ok=True)
    log = work / "blender.log"
    end = time.monotonic() + timeout
    how: dict[str, Any] = {"camera": cam}
    try:
        blender("render", {"blend": str(run / REFINE / "prep" / "work.blend"),
                           "texture": str(run / REFINE / "texture.png"), "cams": [cam], "out": str(work)},
                log, timeout=timeout)
        how["source"] = "work.blend"
    except BlenderError as e:
        if _timed_out(e):
            raise SuggestTimeout(f"the front render timed out after {timeout:g} s") from e
        if not (run / GLB).is_file():
            raise
        left = end - time.monotonic()
        if left <= 1.0:
            raise SuggestTimeout(f"the front render ran out of its {timeout:g} s") from e
        how["work_blend_error"] = str(e)
        try:
            blender("probe", {"glb": str(run / GLB), "cams": [cam], "out": str(work)}, log, timeout=left)
        except BlenderError as e2:
            if _timed_out(e2):
                raise SuggestTimeout(f"the front render timed out after {timeout:g} s") from e2
            raise
        how["source"] = GLB
    raw = work / f"{cam['name']}.png"
    with Image.open(raw) as im:
        rgba = im.convert("RGBA")
    box = rgba.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
    if box is None:
        raise RuntimeError("the front render is empty")
    pad = round(0.04 * max(box[2] - box[0], box[3] - box[1]))
    box = (max(0, box[0] - pad), max(0, box[1] - pad), min(rgba.width, box[2] + pad), min(rgba.height, box[3] + pad))
    flat = Image.new("RGBA", rgba.size, BACKGROUND + (255,))
    flat.alpha_composite(rgba)
    out = work / "front.png"
    flat.convert("RGB").crop(box).save(out)
    return out, how


# ---- model --------------------------------------------------------------------------------------------------------

def request_prompt(version: prompts.PromptVersion, text: str) -> str:
    slot = version.data["user_text"]["slot"]
    template = version.step("suggest")["prompt"]
    if template.count(slot) != 1:
        raise prompts.RegistryError(f"{version.name}: the prompt must hold exactly one {slot}")
    return template.replace(slot, text)


def ask(client, version: prompts.PromptVersion, images: list, text: str,
        deadline_s: float = SPARK_DEADLINE_S) -> dict[str, Any]:
    """The model's answer to the one image question, within ``deadline_s``. A call still running at the deadline
    raises `SuggestTimeout` and is left to finish (or die with the process) on its own thread."""
    step = version.step("suggest")
    p = step["params"]
    client.purpose = "suggest refine"   # names the call in the build log and the call records
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["answer"] = client.compare_images(
                request_prompt(version, text), images, schema=step["schema"], model=step["model"],
                reasoning_effort=p["reasoning_effort"], max_output_tokens=p["max_output_tokens"],
                image_max_side=p["image_max_side"])
        except BaseException as e:  # noqa: BLE001  (handed back to the caller's thread)
            box["error"] = e

    thread = threading.Thread(target=work, name="refine-suggest", daemon=True)
    thread.start()
    thread.join(deadline_s)
    if thread.is_alive():
        raise SuggestTimeout(f"the {step['model']} question passed its {deadline_s:g} s deadline")
    if "error" in box:
        raise box["error"]
    return box["answer"]


def _cost(calls: list[dict[str, Any]]) -> float:
    from herogen.pipeline.client import call_cost

    return round(sum(call_cost(c) for c in calls if not c.get("cached")), 6)


# ---- output -------------------------------------------------------------------------------------------------------

def write_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as JSON to ``path`` through a temporary file in the same folder and a rename, so a reader
    sees the old file or the whole new one, never part of it."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _error(e: BaseException) -> str:
    msg = " ".join(str(e).split())[:300]
    return f"{type(e).__name__}: {msg}" if msg else type(e).__name__


def record(version: prompts.PromptVersion, *, source: str, steps: dict[str, dict[str, str]], model: str | None,
           calls: list[dict[str, Any]], error: str | None) -> dict[str, Any]:
    return {"version": VERSION, "source": source, "texture": steps["texture"], "turnaround": steps["turnaround"],
            "model": model, "prompt": version.name, "prompt_sha256": version.sha256, "cost_usd": _cost(calls),
            "calls": calls, "error": error,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")}


def suggest(run: Path, client, *, render: Callable[..., tuple[Path, dict[str, Any]]] | None = None,
            spark_deadline_s: float = SPARK_DEADLINE_S, render_timeout_s: float = RENDER_TIMEOUT_S) -> int:
    """Make and write ``refine-suggestions.json`` for ``run``; returns the exit code. Raises `MissingInput` (or any
    internal error) without writing the file; a model failure or timeout writes the static file and returns 3."""
    from PIL import Image

    from herogen.pipeline.client import ClientError, answer_cost_usd

    run = Path(run)
    version = prompts.selected("refine_suggest")
    step = version.step("suggest")
    required_inputs(run)
    cam, cam_source = front_camera(run)
    text = description(run)
    statics = {s: static_step(s) for s in STEPS}

    def fallback(e: BaseException, calls: list[dict[str, Any]], model: str | None) -> int:
        err = _error(e)
        write_atomic(run / FILE, record(version, source="static", steps=statics, model=model, calls=calls,
                                        error=err))
        print(f"refine-suggest: {err}; wrote static suggestions", file=sys.stderr)
        return EXIT_MODEL

    try:
        front, how = (render or render_front)(run, cam, timeout=render_timeout_s)
    except SuggestTimeout as e:
        return fallback(e, [], None)
    how["camera_source"] = cam_source
    with Image.open(run / COVER) as im:
        cover = im.convert("RGB")
    with Image.open(front) as im:
        rendered = im.convert("RGB")
    calls0 = len(client.log.calls)
    try:
        answer = ask(client, version, [cover, rendered], text, deadline_s=spark_deadline_s)
    except SuggestTimeout as e:
        calls = [dict(c) for c in client.log.calls[calls0:]]
        p = step["params"]
        calls.append({"model": step["model"], "kind": "compare", "key": None, "cached": False, "outcome": "unknown",
                      "purpose": "suggest refine",
                      "cost_usd": round(answer_cost_usd(step["model"], None, p["max_output_tokens"]), 6)})
        return fallback(e, calls, step["model"])
    except ClientError as e:   # includes ApiError (the API failed) and ReplayMiss
        return fallback(e, [dict(c) for c in client.log.calls[calls0:]], step["model"])
    calls = [dict(c) for c in client.log.calls[calls0:]]
    steps: dict[str, dict[str, str]] = {}
    refused: dict[str, str] = {}
    for s in STEPS:
        item = answer.get(s) if isinstance(answer.get(s), dict) else {}
        clean, why = check_text(s, item.get("text"))
        if clean is None or item.get("basis") not in ("mismatch", "variation"):
            refused[s] = why or f"basis {item.get('basis')!r}"
            steps[s] = static_step(s)
        else:
            steps[s] = {"text": clean, "basis": item["basis"]}
    write_atomic(run / WORK / "check.json", {"answer": answer, "refused": refused, "render": how,
                                              "description": text})
    write_atomic(run / FILE, record(version, source="spark", steps=steps, model=step["model"], calls=calls,
                                    error=None))
    return EXIT_OK


def default_cache(run: Path) -> Path:
    """The record/replay cache when ``--cache`` is not given: the user's ``cache`` folder in the service layout
    (``users/<id>/runs/<job>`` beside ``users/<id>/cache``), else ``_cache-model`` beside the run, as stage 3f."""
    run = Path(run).resolve()
    return run.parent.parent / "cache" if run.parent.name == "runs" else run.parent / "_cache-model"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m herogen.pipeline.suggest",
                                 description="Write refine-suggestions.json for a run finished by stage 3f.")
    ap.add_argument("run", type=Path, help="the run folder")
    ap.add_argument("--cache", type=Path, default=None,
                    help="record/replay cache (default: the user's cache in the service layout, else "
                         "<runs>/_cache-model)")
    ap.add_argument("--replay", action="store_true", help="read the cache only; never call the Model API")
    a = ap.parse_args(argv)   # a usage error exits 2
    try:
        from herogen.pipeline.client import ModelClient

        client = ModelClient(a.cache or default_cache(a.run), mode="replay" if a.replay else "live", fast=True)
        code = suggest(a.run, client)
    except MissingInput as e:
        print(f"refine-suggest: missing input: {e}; no file written", file=sys.stderr)
        return EXIT_FAILED
    except Exception as e:  # noqa: BLE001  (an internal error: reported, no file written)
        print(f"refine-suggest: internal error: {_error(e)}; no file written", file=sys.stderr)
        return EXIT_FAILED
    if code == EXIT_OK:
        print(f"refine-suggest: ok ({FILE})")
    return code


if __name__ == "__main__":
    sys.exit(main())
