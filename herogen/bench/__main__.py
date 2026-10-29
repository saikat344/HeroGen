"""CLI: `python -m herogen.bench render --asset <glb/gltf/fbx/blend> --out <run folder> [options]`."""
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m herogen.bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render a known 3D character into a ground-truth run folder")
    r.add_argument("--asset", required=True, type=Path, help="the character: .glb, .gltf, .fbx or .blend")
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
    a = ap.parse_args(argv)

    from herogen.blender.bl import BlenderError

    from .render import render_asset

    try:
        rec = render_asset(
            a.asset, a.out, include=a.include, exclude=a.exclude, yaw_deg=a.yaw, pose=a.pose,
            material_images=_pairs(a.image, "--image"), material_vcols=_pairs(a.vcol, "--vcol"),
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


if __name__ == "__main__":
    raise SystemExit(main())
