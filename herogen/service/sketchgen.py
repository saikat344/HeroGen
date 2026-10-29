"""Stage p: a pencil sketch drawn by the image model from a typed description, for jobs that start from a prompt.

`python3 -m herogen.service.sketchgen --run <run folder> --out <sketch.png> --cache <cache>` reads the description
on stdin, wraps it in a fixed sketch prompt (the same framing the pipeline's stand-in sketch uses), makes one
`muse-image-1.0` generation through herogen.pipeline's model client (live, recorded in the per-user cache, so a
retried job replays it at no cost) and writes the image to `--out` and its record to `<run>/stage-p.json`. The
pipeline then starts at stage 0 from that sketch exactly as from an upload.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from herogen.service.runner import write_record

SKETCH_SIZE = "1024x1536"


def sketch_prompt(description: str) -> str:
    return ("A rough hand-drawn pencil sketch on plain white paper of a video game hero character, full body from "
            "head to feet, standing, front view, arms held a little away from the body. The character: "
            f"{description}. Graphite pencil lines only, light construction lines, no colour, no shading fill, "
            "no background, no text.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m herogen.service.sketchgen")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    a = ap.parse_args(argv)
    from herogen.pipeline.client import COST_BASIS, IMAGE_MODEL, ModelClient
    from herogen.pipeline.runfolder import now_iso

    description = sys.stdin.read().strip()
    if not description:
        print("sketchgen: no description on stdin", file=sys.stderr)
        return 2
    started, t0 = now_iso(), time.monotonic()
    client = ModelClient(a.cache, mode="live")
    img = client.generate_image(sketch_prompt(description), size=SKETCH_SIZE).convert("RGB")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    part = a.out.with_name(a.out.name + ".part")
    img.save(part, format="PNG")
    part.replace(a.out)
    ok = min(img.size) >= 256
    write_record(a.run, {
        "stage": "p", "name": "sketch_from_prompt", "ok": ok, "started": started,
        "seconds": round(time.monotonic() - t0, 2), "cost_usd": client.log.cost_usd(),
        "cost_basis": COST_BASIS + "; replayed (cached) calls cost 0", "models": [IMAGE_MODEL],
        "outputs": [], "checks": [{"name": "min_side_px", "value": min(img.size), "threshold": 256, "op": ">=",
                                   "pass": ok, "gate": True}],
        "notes": "The sketch was drawn by the image model from the user's description.",
        "calls": client.log.calls, "details": {"description": description, "size": list(img.size)},
    })
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
