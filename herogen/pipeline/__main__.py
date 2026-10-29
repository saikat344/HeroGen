"""CLI: `python3 -m herogen.pipeline run <sketch.png> --run-id <id> [--stages 0-2,5] [--live]`.

Stage 2 is followed by the held-out views (`2h`) when it passes. `--stages 2h` makes only the held-out views
for a run whose stage 2 already passed, leaving every earlier output untouched; it prints each view's status
(accepted, needs review or rejected), found yaw and file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from herogen.pipeline.client import ModelClient
from herogen.pipeline.heldout import stage2h_heldout
from herogen.pipeline.runfolder import RunFolder, runs_root
from herogen.pipeline.stages import GateFailed, stage0_input, stage1_cover, stage2_turnaround
from herogen.pipeline.results import stage5_results

OWN_STAGES = {0, 1, 2, 5}
HELDOUT = "2h"
Stage = int | str  # 0, 1, 2, 5 or "2h"


def _order(s: Stage) -> float:
    return 2.5 if s == HELDOUT else float(s)

STAND_IN_PROMPT = (
    "A rough hand-drawn pencil sketch on plain white paper of a video game hero character, full body, standing, "
    "front view: a helmet, chunky armour plates on chest and shoulders, gauntlets, belt, armoured boots. "
    "Graphite pencil lines only, light construction lines, no colour, no shading fill, no background, no text."
)


def parse_stages(spec: str) -> list[Stage]:
    """'0-2,5' -> [0, 1, 2, 5]; '2h' -> ['2h']. Only stages this package owns are accepted."""
    out: set[Stage] = set()
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == HELDOUT:
            out.add(HELDOUT)
        elif "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            if a > b:
                raise ValueError(f"bad stage range {part!r}")
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    foreign = sorted(s for s in out if s != HELDOUT and s not in OWN_STAGES)
    if foreign:
        raise ValueError(f"stages {foreign} are not run by herogen.pipeline (3-4 are herogen/blender)")
    return sorted(out, key=_order)


def _print_heldout(rec: dict) -> None:
    for name, v in (rec.get("details") or {}).get("views", {}).items():
        yaw = v.get("found_yaw_deg")
        found = f"found yaw {yaw:+.0f}" if isinstance(yaw, (int, float)) else "no yaw found"
        tries = len(v["attempts"])
        print(f"  {name}: {v['status']} after {tries} generation(s), {found}: {v['files']['image']}")


def run_pipeline(
    sketch: Path | None,
    run_id: str,
    stages: list[Stage],
    *,
    mode: str = "replay",
    runs: str | None = None,
    cache: str | None = None,
    label: str = "",
) -> int:
    root = runs_root(runs)
    run = RunFolder(root, run_id)
    client = ModelClient(Path(cache) if cache else root / "_cache", mode=mode)
    pipeline = [s for s in stages if s != 5]
    if 2 in pipeline and HELDOUT not in pipeline:
        pipeline.append(HELDOUT)  # the held-out view follows a passing stage 2
    pipeline.sort(key=_order)
    status = 0
    for n in pipeline:
        try:
            if n == 0:
                if sketch is None:
                    raise ValueError("stage 0 needs a sketch path")
                rec = stage0_input(run, sketch, label=label)
            elif n == 1:
                rec = stage1_cover(run, client)
            elif n == 2:
                rec = stage2_turnaround(run, client)
            else:
                rec = stage2h_heldout(run, client)
            print(f"stage {n} {rec['name']}: ok ({rec['seconds']} s, ${rec['cost_usd']:.4f} est)")
            if n == HELDOUT:
                _print_heldout(rec)
        except GateFailed as e:
            print(f"stage {n}: {e}; stopping", file=sys.stderr)
            if n == HELDOUT:
                _print_heldout(e.record)
            status = 2
            break
        except Exception as e:  # noqa: BLE001  (report, stop, still write results)
            print(f"stage {n}: error: {type(e).__name__}: {e}; stopping", file=sys.stderr)
            status = 1
            break
    if 5 in stages:
        data = stage5_results(run)
        print(f"stage 5 results: {run.path / 'results.html'} (total ${data['total_cost_usd']:.4f} est)")
    return status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m herogen.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run pipeline stages on a sketch")
    r.add_argument("sketch", nargs="?", type=Path)
    r.add_argument("--run-id", required=True)
    r.add_argument("--stages", default="0-2,5",
                   help="e.g. 0-2,5; 2 is followed by the held-out views 2h; 2h alone")
    r.add_argument("--live", action="store_true", help="call the Model API (default: replay from the cache)")
    r.add_argument("--runs", default=None, help="runs root (default $HEROGEN_RUNS or ./runs)")
    r.add_argument("--cache", default=None, help="record/replay cache (default <runs>/_cache)")
    r.add_argument("--label", default="", help="label for the input, e.g. 'stand-in sketch'")
    s = sub.add_parser("stand-in", help="draw a stand-in pencil sketch with muse-image-1.0")
    s.add_argument("out", type=Path)
    s.add_argument("--live", action="store_true")
    s.add_argument("--runs", default=None)
    s.add_argument("--cache", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        try:
            stages = parse_stages(a.stages)
        except ValueError as e:
            ap.error(str(e))
        return run_pipeline(
            a.sketch, a.run_id, stages, mode="live" if a.live else "replay",
            runs=a.runs, cache=a.cache, label=a.label,
        )
    root = runs_root(a.runs)
    client = ModelClient(Path(a.cache) if a.cache else root / "_cache", mode="live" if a.live else "replay")
    img = client.generate_image(STAND_IN_PROMPT, size="1024x1536")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(a.out, format="PNG")
    print(f"stand-in sketch: {a.out} {img.size[0]}x{img.size[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
