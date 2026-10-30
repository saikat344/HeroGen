"""Part masks of the front view: SAM 3.1 `face` and `head`, written by stage 2 for the stages after it. No stage
reads them in this version; they are recorded so later stages can use them without another model call.

Each part keeps the most confident instance that lies mostly (at least half) on the figure mask grown by 4 px,
clipped to the figure. It is written as `masks/front-<part>.png` (white on black, the view's size); a part SAM
does not find on the figure gets no file. `masks/parts.json` records, per part, the file (or null), the method
(`sam-3.1:<text>`), the number of instances SAM returned, the kept instance's confidence and its pixel count. A
Model API failure (`ApiError`, after the client's retries) is recorded with `api_error` and no file, so a reader
can tell a failed call from SAM finding nothing, and the stage goes on; anything else (a replay miss, a corrupt
cache entry, a bug) is raised. A failed call is not cached, so replaying such a run raises a replay miss for that
part.

The record has its own file, not a key in `views/meta.json`: the held-out views are tied to the turnaround by a
hash of the stage 2 files they are made from (`heldout.IDENTITY_FILES`, which include `views/meta.json`). The
part masks are derived from the front view and never read by the held-out stage, so writing them (stage 2, or
the `parts` backfill on an older run) changes none of those files and every held-out view stays valid.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from herogen.pipeline import imaging
from herogen.pipeline.client import SAM_MODEL, ApiError, ModelClient
from herogen.pipeline.inputs import FACE_TEXT, HEAD_TEXT, best_instance
from herogen.pipeline.runfolder import RunFolder

PARTS = {"face": FACE_TEXT, "head": HEAD_TEXT}


RECORD = "masks/parts.json"


def part_file(part: str) -> str:
    return f"masks/front-{part}.png"


def front_part_masks(client: ModelClient, image: Image.Image, figure: np.ndarray
                     ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """SAM 3.1 part masks of the front view. Returns ({part: mask} for the parts found, {part: record})."""
    near = imaging.dilate(figure, 4)
    masks: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {}
    for part, text in PARTS.items():
        rec: dict[str, Any] = {"file": None, "method": f"{SAM_MODEL}:{text}"}
        try:
            insts = client.segment(image, text)
        except ApiError as e:
            rec.update({"api_error": True, "error": f"{type(e).__name__}: {e}"[:300]})
            info[part] = rec
            continue
        rec["instances"] = len(insts)
        best = best_instance(insts, near)
        if best is not None:
            m = best.mask & figure
            if m.any():
                masks[part] = m
                rec.update({"file": part_file(part), "confidence": best.confidence, "pixels": int(m.sum())})
        info[part] = rec
    return masks, info


def write_part_masks(run: RunFolder, client: ModelClient) -> dict[str, Any]:
    """Segment the run's front view, write `masks/front-<part>.png` and the record `masks/parts.json`. Returns
    the record. Earlier part masks the new call does not find are removed, so a file on disk always matches the
    record. No other file of the run is touched."""
    for rel in ("views/front.png", "masks/front.png"):
        if not run.exists(rel):
            raise FileNotFoundError(f"{rel} is missing from the run folder; run stage 2 first")
    with Image.open(run.path / "views/front.png") as im:
        image = im.convert("RGB")
    with Image.open(run.path / "masks/front.png") as im:
        figure = np.asarray(im.convert("L")) > 127
    masks, info = front_part_masks(client, image, figure)
    for part in PARTS:
        path = run.path / part_file(part)
        if part in masks:
            imaging.to_mask_image(masks[part]).save(path)
        elif path.exists():
            path.unlink()
    run.write_json(RECORD, info)
    return info
