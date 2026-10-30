"""CLI:

    python -m herogen.bench render --asset <glb/gltf/vrm/fbx/blend> --out <run folder> [options]
    python -m herogen.bench dataset --manifest <manifest.json> --out <dataset folder> [options]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import DEFAULT_SIZE, HEIGHT_STUDS


def _pairs(items: list[str], flag: str) -> dict[str, str]:
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise SystemExit(f"{flag} expects MATERIAL=VALUE, got {item!r}")
        out[key] = value
    return out


def _alpha_modes(items: list[str]) -> dict[str, str]:
    modes = _pairs(items, "--alpha")
    bad = {k: v for k, v in modes.items() if v not in ("clip", "opaque")}
    if bad:
        raise SystemExit(f"--alpha expects MATERIAL=clip or MATERIAL=opaque, got {bad}")
    return modes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m herogen.bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render a known 3D character into a ground-truth run folder")
    r.add_argument("--asset", required=True, type=Path,
                   help="the character: .glb, .gltf, .vrm, .fbx or .blend")
    r.add_argument("--out", required=True, type=Path, help="the run folder to write")
    r.add_argument("--pose", default="auto", choices=("auto", "a-pose", "rest", "as-shipped"),
                   help="auto: A-pose when the rig's arms are found, else the rest pose "
                        "(unrigged: as shipped)")
    r.add_argument("--yaw", type=float, default=None,
                   help="degrees about +Z that turn the figure to face -Y; read from the rig when left out "
                        "(required for a figure without a rig)")
    r.add_argument("--include", help="regex: render the meshes whose names match (hidden ones too)")
    r.add_argument("--exclude", help="regex: leave out the meshes whose names match")
    r.add_argument("--image", action="append", metavar="MATERIAL=IMAGE[@UVMAP]",
                   help="use this image (file or image name) as the material's base colour, "
                        "optionally through the named UV map")
    r.add_argument("--alpha", action="append", metavar="MATERIAL=clip|opaque",
                   help="clip the material's --image texture at alpha 0.5, or keep it opaque (the default "
                        "for --image overrides and materials without nodes)")
    r.add_argument("--vcol", action="append", metavar="MATERIAL=ATTRIBUTE",
                   help="use this colour attribute as the material's base colour")
    r.add_argument("--image-dir", action="append", type=Path, default=[],
                   help="folder to find missing texture files in, by file name")
    r.add_argument("--bind-unrigged", action="store_true",
                   help="bind meshes the rig does not deform to it, with the nearest deformed mesh's weights")
    r.add_argument("--size", type=int, default=DEFAULT_SIZE, help="square image size in pixels")
    r.add_argument("--height", type=float, default=HEIGHT_STUDS, help="figure height in studs")
    r.add_argument("--samples-flat", type=int, default=16)
    r.add_argument("--samples-lit", type=int, default=128)
    r.add_argument("--device", choices=("auto", "cpu"), default="auto")
    r.add_argument("--timeout", type=float, default=3600.0, help="seconds allowed for the Blender call")
    r.add_argument("--overwrite", action="store_true",
                   help="replace an existing run folder at --out (only one this tool wrote)")
    d = sub.add_parser("dataset", help="render the characters of a manifest into a training dataset "
                                       "(resumable: finished characters are skipped)")
    d.add_argument("--manifest", required=True, type=Path, help="JSON list of characters (see dataset.py)")
    d.add_argument("--out", required=True, type=Path, help="the dataset folder, one folder per character")
    d.add_argument("--workers", type=int, default=2, help="Blender processes at once (they share the GPU)")
    d.add_argument("--split", choices=("train", "heldout"), help="only the characters of this split")
    d.add_argument("--id", action="append", dest="ids", help="only this character (repeatable)")
    d.add_argument("--limit", type=int, help="at most this many characters this time")
    d.add_argument("--retry-failed", action="store_true", help="try characters that failed before again")
    d.add_argument("--size", type=int, default=DEFAULT_SIZE, help="whole-figure views, square pixels")
    d.add_argument("--head-size", type=int, default=1024, help="head close-ups, square pixels")
    d.add_argument("--samples-flat", type=int, default=16)
    d.add_argument("--samples-lit", type=int, default=64)
    d.add_argument("--device", choices=("auto", "cpu"), default="auto")
    d.add_argument("--timeout", type=float, default=1800.0, help="seconds allowed per character")
    a = ap.parse_args(argv)

    if a.cmd == "dataset":
        return dataset_main(a)

    from herogen.blender.bl import BlenderError

    from .render import render_asset

    try:
        rec = render_asset(
            a.asset, a.out, include=a.include, exclude=a.exclude, yaw_deg=a.yaw, pose=a.pose,
            material_images=_pairs(a.image, "--image"), material_vcols=_pairs(a.vcol, "--vcol"),
            material_alpha=_alpha_modes(a.alpha),
            image_dirs=a.image_dir, bind_unrigged=a.bind_unrigged, size=a.size, height=a.height,
            samples_flat=a.samples_flat, samples_lit=a.samples_lit, device=a.device, timeout=a.timeout,
            overwrite=a.overwrite,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BlenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"stage": 2, "ok": rec["ok"], "seconds": rec["seconds"], "pose": rec["details"]["pose"],
                      "checks": {c["name"]: c["value"] for c in rec["checks"]}}))
    if not rec["ok"]:
        print("a ground-truth check failed; see stage-2.json", file=sys.stderr)
    return 0 if rec["ok"] else 1


def dataset_main(a) -> int:
    from .dataset import run_batch
    from .render import MIN_SAMPLES_FLAT, SIZE_RANGE

    for flag, n in (("--size", a.size), ("--head-size", a.head_size)):
        if not SIZE_RANGE[0] <= n <= SIZE_RANGE[1]:
            print(f"error: {flag} must be from {SIZE_RANGE[0]} to {SIZE_RANGE[1]}", file=sys.stderr)
            return 2
    if a.samples_flat < MIN_SAMPLES_FLAT or a.samples_lit < 1:
        print(f"error: --samples-flat must be at least {MIN_SAMPLES_FLAT}, --samples-lit at least 1",
              file=sys.stderr)
        return 2
    try:
        counts = run_batch(a.manifest, a.out, workers=a.workers, split=a.split, ids=a.ids, limit=a.limit,
                           retry_failed=a.retry_failed, log=lambda m: print(m, flush=True), size=a.size,
                           head_size=a.head_size, samples_flat=a.samples_flat, samples_lit=a.samples_lit,
                           device=a.device, timeout=a.timeout)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(counts))
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
