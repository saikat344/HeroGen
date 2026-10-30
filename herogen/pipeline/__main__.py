"""CLI: `python3 -m herogen.pipeline run <sketch.png> --run-id <id> [--stages 0-2,5] [--live]`.

Stage 2 is followed by the held-out views (`2h`) and the face source (`2f`, `herogen.pipeline.facesource`: sharp
head close-ups for stage 4's Head texture) when it passes. `--stages 2f` makes only the close-ups, for a run whose
stage 2 already passed; stage 2f's own failures never stop the run (the failure is recorded, printed to stderr, and
stage 4 bakes the plain views), but a replay miss does, as in every other stage.
`--stages 2h` makes only the held-out views
for a run whose stage 2 already passed, leaving every earlier output untouched; it prints each view's status
(accepted, needs review or rejected), found yaw and file. `--views front-left-60,front-right-60` makes only
those views and keeps the others from the earlier stage 2h record.

`python3 -m herogen.pipeline review --run-id <id> --view <view> --accept|--reject [--note TEXT] [--by NAME]`
records a person's decision on one held-out view: the view's files move to `heldout/` (accepted) or
`heldout/rejected/`, and the decision is appended to `stage-2h.json`. `recheck --run-id <id>` judges every
recorded held-out view again under the current checks, with no model calls.

`run` exits 0 when every stage passed, 2 when a gate failed, 3 when an input check rejected the sketch (stages 0
and 1, before any image is generated) and 1 on an error, such as the Model API failing after its retries; the
stage record then holds the error and every call made. An input check that only warns is printed and the run goes
on.

`python3 -m herogen.pipeline parts --run-id <id> [--live]` writes the front view's part masks
(`masks/front-face.png`, `masks/front-head.png` and their record `masks/parts.json`; `herogen.pipeline.parts`)
for a run whose stage 2 already passed. It writes no other file, so the held-out views stay valid. Stage 2
writes them itself; this adds them to an older run.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from herogen.pipeline import inputs
from herogen.pipeline.client import ApiError, ClientError, ModelClient
from herogen.pipeline.facesource import stage2f_face_source
from herogen.pipeline.heldout import DECISIONS, recheck_heldout, review_view, stage2h_heldout
from herogen.pipeline.runfolder import RunFolder, now_iso, runs_root
from herogen.pipeline.stages import GateFailed, InputRejected, stage0_input, stage1_cover, stage2_turnaround
from herogen.pipeline.results import stage5_results

OWN_STAGES = {0, 1, 2, 5}
HELDOUT = "2h"
FACE_SOURCE = "2f"
Stage = int | str  # 0, 1, 2, 5, "2h" or "2f"


def _order(s: Stage) -> float:
    return {HELDOUT: 2.5, FACE_SOURCE: 2.6}.get(s, s)  # type: ignore[arg-type,return-value]

STAND_IN_PROMPT = (
    "A rough hand-drawn pencil sketch on plain white paper of a video game hero character, full body, standing, "
    "front view: a helmet, chunky armour plates on chest and shoulders, gauntlets, belt, armoured boots. "
    "Graphite pencil lines only, light construction lines, no colour, no shading fill, no background, no text."
)


def parse_stages(spec: str) -> list[Stage]:
    """'0-2,5' -> [0, 1, 2, 5]; '2h' -> ['2h']; '2f' -> ['2f']. Only stages this package owns are accepted."""
    out: set[Stage] = set()
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part in (HELDOUT, FACE_SOURCE):
            out.add(part)
        elif "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            if a > b:
                raise ValueError(f"bad stage range {part!r}")
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    foreign = sorted(s for s in out if s not in (HELDOUT, FACE_SOURCE) and s not in OWN_STAGES)
    if foreign:
        raise ValueError(f"stages {foreign} are not run by herogen.pipeline (3-4 are herogen/blender)")
    return sorted(out, key=_order)


def _print_heldout(rec: dict) -> None:
    for name, v in (rec.get("details") or {}).get("views", {}).items():
        yaw = v.get("found_yaw_deg")
        found = f"found yaw {yaw:+.0f}" if isinstance(yaw, (int, float)) else "no yaw found"
        tries = len(v["attempts"])
        review = v.get("review") or {}
        by = f" (on review by {review['by']})" if review.get("by") else ""
        print(f"  {name}: {v['status']}{by} after {tries} generation(s), {found}: {v['files']['image']}")


def run_pipeline(
    sketch: Path | None,
    run_id: str,
    stages: list[Stage],
    *,
    mode: str = "replay",
    runs: str | None = None,
    cache: str | None = None,
    label: str = "",
    views: list[str] | None = None,
) -> int:
    root = runs_root(runs)
    run = RunFolder(root, run_id)
    client = ModelClient(Path(cache) if cache else root / "_cache", mode=mode)
    pipeline = [s for s in stages if s != 5]
    if 2 in pipeline:  # the held-out views and the face source follow a passing stage 2
        pipeline += [s for s in (HELDOUT, FACE_SOURCE) if s not in pipeline]
    pipeline.sort(key=_order)
    status = 0
    for n in pipeline:
        began = now_iso()
        try:
            if n == 0:
                if sketch is None:
                    raise ValueError("stage 0 needs a sketch path")
                rec = stage0_input(run, sketch, label=label)
            elif n == 1:
                rec = stage1_cover(run, client)
            elif n == 2:
                rec = stage2_turnaround(run, client)
            elif n == FACE_SOURCE:
                rec = stage2f_face_source(run, client)
                fallback = rec["details"].get("fallback")
                if fallback:
                    print(f"stage {n} {rec['name']}: fell back to the plain views ({fallback})")
                    print(f"warning: stage {n} {rec['name']} fell back: {fallback}", file=sys.stderr)
                else:
                    used = rec["details"].get("used_views") or []
                    print(f"stage {n} {rec['name']}: close-ups used for {', '.join(used) or 'no view'}")
                continue
            else:
                rec = stage2h_heldout(run, client, views=views)
            print(f"stage {n} {rec['name']}: ok ({rec['seconds']} s, ${rec['cost_usd']:.4f} est)")
            for c in inputs.warnings(rec.get("checks") or []):
                print(f"stage {n}: warning: {c['message']}", file=sys.stderr)
            if n == HELDOUT:
                _print_heldout(rec)
        except GateFailed as e:
            print(f"stage {n}: {e}; stopping", file=sys.stderr)
            if n == HELDOUT:
                _print_heldout(e.record)
            status = 2
            break
        except InputRejected as e:
            print(f"stage {n}: input rejected: {e}; stopping", file=sys.stderr)
            status = 3
            break
        except Exception as e:  # noqa: BLE001  (report, stop, still write results)
            what = "the Model API failed" if isinstance(e, ApiError) else f"error: {type(e).__name__}"
            print(f"stage {n}: {what}: {e}; stopping", file=sys.stderr)
            rec = run.read_stage(n)  # type: ignore[arg-type]
            if rec and rec.get("status") == "error" and str(rec.get("started", "")) >= began:  # this run's
                print(f"  the error and the calls made are in {run.path / f'stage-{n}.json'}", file=sys.stderr)
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
                   help="e.g. 0-2,5; 2 is followed by the held-out views 2h and the face source 2f; 2h or 2f "
                        "alone")
    r.add_argument("--live", action="store_true", help="call the Model API (default: replay from the cache)")
    r.add_argument("--runs", default=None, help="runs root (default $HEROGEN_RUNS or ./runs)")
    r.add_argument("--cache", default=None, help="record/replay cache (default <runs>/_cache)")
    r.add_argument("--label", default="", help="label for the input, e.g. 'stand-in sketch'")
    r.add_argument("--views", default=None,
                   help="held-out views to make, e.g. front-left-60,front-right-60 (default all; the others "
                        "are kept from the earlier stage 2h record)")
    c = sub.add_parser("recheck", help="judge the recorded held-out views again under the current checks")
    c.add_argument("--run-id", "--run", dest="run_id", required=True)
    c.add_argument("--runs", default=None, help="runs root (default $HEROGEN_RUNS or ./runs)")
    v = sub.add_parser("review", help="record a person's decision on one held-out view")
    v.add_argument("--run-id", "--run", dest="run_id", required=True)
    v.add_argument("--view", required=True, help="e.g. front-left-60")
    d = v.add_mutually_exclusive_group(required=True)
    d.add_argument("--accept", dest="decision", action="store_const", const="accepted")
    d.add_argument("--reject", dest="decision", action="store_const", const="rejected")
    v.add_argument("--note", default="", help="why, e.g. 'arms not in A-pose'")
    v.add_argument("--by", default=None, help="who decided (default: the login name)")
    v.add_argument("--runs", default=None, help="runs root (default $HEROGEN_RUNS or ./runs)")
    p = sub.add_parser("parts", help="write the front view's part masks (face, head) for an existing run")
    p.add_argument("--run-id", "--run", dest="run_id", required=True)
    p.add_argument("--live", action="store_true", help="call the Model API (default: replay from the cache)")
    p.add_argument("--runs", default=None, help="runs root (default $HEROGEN_RUNS or ./runs)")
    p.add_argument("--cache", default=None, help="record/replay cache (default <runs>/_cache)")
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
        views = [x.strip() for x in a.views.split(",") if x.strip()] if a.views else None
        if views and HELDOUT not in stages and 2 not in stages:
            ap.error("--views needs stage 2h")
        return run_pipeline(
            a.sketch, a.run_id, stages, mode="live" if a.live else "replay",
            runs=a.runs, cache=a.cache, label=a.label, views=views,
        )
    if a.cmd == "recheck":
        try:
            rec = recheck_heldout(RunFolder(runs_root(a.runs), a.run_id))
        except (OSError, ValueError) as e:  # a missing file, no permission, a bad request: one line
            print(f"recheck: {e}", file=sys.stderr)
            return 1
        print(f"stage 2h checked again: {'passes' if rec['ok'] else 'does not pass'}")
        _print_heldout(rec)
        return 0
    if a.cmd == "review":
        assert a.decision in DECISIONS
        try:
            rec = review_view(RunFolder(runs_root(a.runs), a.run_id), a.view, a.decision, note=a.note,
                              by=a.by if a.by is not None else getpass.getuser())
        except (OSError, ValueError) as e:  # a missing file, no permission, a bad request: one line
            print(f"review: {e}", file=sys.stderr)
            return 1
        print(f"{a.view}: {a.decision} ({rec['details']['views'][a.view]['files']['image']}); stage 2h "
              f"{'passes' if rec['ok'] else 'does not pass'}")
        _print_heldout(rec)
        return 0
    root = runs_root(a.runs)
    client = ModelClient(Path(a.cache) if a.cache else root / "_cache", mode="live" if a.live else "replay")
    if a.cmd == "parts":
        from herogen.pipeline.parts import write_part_masks

        try:
            info = write_part_masks(RunFolder(root, a.run_id), client)
        except (OSError, ValueError, ClientError) as e:  # a missing file, a replay miss, no key: one line
            print(f"parts: {e}", file=sys.stderr)
            return 1
        for part, rec in info.items():
            found = rec["file"] or ("model error" if rec.get("api_error") else "not found")
            print(f"  {part}: {found}")
        return 0
    img = client.generate_image(STAND_IN_PROMPT, size="1024x1536")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(a.out, format="PNG")
    print(f"stand-in sketch: {a.out} {img.size[0]}x{img.size[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
