"""Stand-ins for dev mode and tests: a stage runner that writes small synthetic outputs instantly.

`FakeRunner` never calls a model or Blender. It writes the files and records each real stage would (same names,
same record format), with images drawn by code, a box-shaped GLB and placeholder Roblox files whose README says
they are placeholders, so the whole service can be exercised end to end on one machine. Production refuses to
start with it (see `__main__`). `fail` makes a stage reject or fail, for tests.
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path

from herogen.service.runner import Outcome, StageContext, StageSpec, write_record

SECTIONS = ("Head", "UpperTorso", "LowerTorso", "LeftUpperArm", "LeftLowerArm", "LeftHand", "RightUpperArm",
            "RightLowerArm", "RightHand", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "RightUpperLeg",
            "RightLowerLeg", "RightFoot")
ROBLOX_FILES = ("HeroGen-R15.rbxmx", "bind_meshes.luau", "rig_test.server.luau", "README.txt")


def box_glb(size: tuple[float, float, float] = (1.0, 2.0, 0.5)) -> bytes:
    """A valid glTF 2.0 binary holding one box mesh."""
    sx, sy, sz = (s / 2 for s in size)
    v = [(x, y, z) for x in (-sx, sx) for y in (0.0, 2 * sy) for z in (-sz, sz)]
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6),
             (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    pos = b"".join(struct.pack("<3f", *p) for p in v)
    idx = b"".join(struct.pack("<3H", *f) for f in faces)
    idx += b"\x00" * ((4 - len(idx) % 4) % 4)
    binchunk = pos + idx
    doc = {
        "asset": {"version": "2.0", "generator": "herogen-service fake"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"name": "Body", "mesh": 0}],
        "meshes": [{"name": "Body", "primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(binchunk)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos)},
                        {"buffer": 0, "byteOffset": len(pos), "byteLength": len(faces) * 6}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(v), "type": "VEC3",
                       "min": [-sx, 0.0, -sz], "max": [sx, 2 * sy, sz]},
                      {"bufferView": 1, "componentType": 5123, "count": len(faces) * 3, "type": "SCALAR"}],
    }
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(binchunk)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(binchunk), 0x004E4942) + binchunk)


def _figure(size: int, colour: tuple[int, int, int], width: float = 1.0):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    c, u = size // 2, size / 16
    d.ellipse([c - u, u, c + u, 3 * u], fill=colour)
    d.rectangle([c - 2 * u * width, 3 * u, c + 2 * u * width, 9 * u], fill=colour)
    d.line([c - 2 * u, 3.5 * u, c - 5 * u, 8 * u], fill=colour, width=int(u))
    d.line([c + 2 * u, 3.5 * u, c + 5 * u, 8 * u], fill=colour, width=int(u))
    d.rectangle([c - 1.8 * u, 9 * u, c - 0.4 * u, 15 * u], fill=colour)
    d.rectangle([c + 0.4 * u, 9 * u, c + 1.8 * u, 15 * u], fill=colour)
    return img


def _record(stage, name, outputs, ok=True, cost=0.0, checks=None, details=None):
    return {"stage": stage, "name": name, "ok": ok, "started": time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                                                            time.gmtime()),
            "seconds": 0.01, "cost_usd": cost, "cost_basis": "fake runner: nothing was paid for",
            "models": [], "outputs": outputs, "checks": checks or [], "notes": "Made by the fake runner.",
            "details": details or {}}


class FakeRunner:
    def __init__(self, fail: dict[str, str] | None = None, delay_s: float = 0.0, roblox: bool = True) -> None:
        self.fail = dict(fail or {})
        self.delay_s = delay_s
        self.roblox = roblox
        self.calls: list[str] = []

    def available(self, spec: StageSpec) -> str | None:
        if spec.key == "6" and not self.roblox:
            return "Roblox export is not part of this version of HeroGen yet."
        return None

    def run(self, spec: StageSpec, ctx: StageContext) -> Outcome:
        self.calls.append(spec.key)
        why = self.available(spec)
        if why:
            return Outcome("skipped", message=why, code="not_available")
        if self.delay_s:
            time.sleep(self.delay_s)
        mode = self.fail.get(spec.key)
        if mode == "error":
            return Outcome("failed", 0.01, 0.0, f"The {spec.label} stage stopped with an internal error.",
                           "stage_error", False, "fake error")
        if mode == "transient":
            return Outcome("failed", 0.01, 0.0, "The image model service did not answer properly.",
                           "model_unavailable", True, "fake transient error")
        if mode == "raise":
            raise RuntimeError("fake crash")
        run = ctx.run_dir
        run.mkdir(parents=True, exist_ok=True)
        getattr(self, f"_stage_{spec.key}")(ctx, run)
        if mode == "reject":
            rec = json.loads((run / spec.record).read_text())
            rec["ok"] = False
            rec["checks"] = [{"name": "height_agreement_4_views", "value": 0.81, "threshold": 0.95, "op": ">=",
                              "pass": False, "gate": True}]
            write_record(run, rec)
        from herogen.service.runner import CliRunner

        return CliRunner.outcome(spec, ctx, 0.01, 0, "", "", False)

    def _stage_p(self, ctx: StageContext, run: Path) -> None:
        from herogen import prompts
        from herogen.pipeline.prompt_sketch import DESIGN, SKETCH

        for rel in (DESIGN, SKETCH):
            _figure(512, (60, 60, 60)).save(run / rel, format="PNG")
        write_record(run, _record("p", "prompt_sketch", [DESIGN, SKETCH], cost=0.01,
                                  details={"user_text": ctx.prompt,
                                           "prompt_version": prompts.selected("prompt_sketch").ref()}))

    def _stage_0(self, ctx: StageContext, run: Path) -> None:
        from PIL import Image

        from herogen.pipeline.prompt_sketch import SKETCH

        src = run / SKETCH if ctx.kind == "prompt" else ctx.input_path
        with Image.open(src) as image:
            image.convert("RGB").save(run / "00-sketch.png", format="PNG")
        write_record(run, _record(0, "input", ["00-sketch.png"], details={
            "source_name": f"{ctx.name or ctx.job_id}.png", "label": ctx.kind}))

    def _stage_1(self, ctx: StageContext, run: Path) -> None:
        _figure(512, (40, 90, 200)).save(run / "01-cover.png", format="PNG")
        write_record(run, _record(1, "cover_art", ["01-cover.png"], cost=0.0125))

    def _stage_2(self, ctx: StageContext, run: Path) -> None:
        from PIL import Image

        (run / "views").mkdir(exist_ok=True)
        (run / "masks").mkdir(exist_ok=True)
        sheet = Image.new("RGB", (1024, 256), (255, 255, 255))
        for i, v in enumerate(("front", "right", "back", "left", "top")):
            img = _figure(256, (40, 90, 200), 1.0 if v in ("front", "back") else 0.5)
            img.save(run / "views" / f"{v}.png", format="PNG")
            img.convert("L").point(lambda p: 255 if p < 250 else 0).save(run / "masks" / f"{v}.png")
            if i < 4:
                sheet.paste(img, (i * 256, 0))
        sheet.save(run / "02-turnaround.png", format="PNG")
        (run / "views" / "meta.json").write_text(json.dumps({"pixels_per_unit": 40.0, "height_units": 5.5,
                                                             "image_size": 256}))
        write_record(run, _record(2, "turnaround", ["02-turnaround.png"], cost=0.025))

    def _stage_2h(self, ctx: StageContext, run: Path) -> None:
        (run / "heldout").mkdir(exist_ok=True)
        _figure(256, (40, 90, 200), 0.8).save(run / "heldout" / "front-left-34.png", format="PNG")
        write_record(run, _record("2h", "heldout", ["heldout/front-left-34.png"], cost=0.0125))

    def _stage_3(self, ctx: StageContext, run: Path) -> None:
        (run / "03-mesh.glb").write_bytes(box_glb())
        write_record(run, _record(3, "base_mesh", ["03-mesh.glb"]))

    def _stage_4(self, ctx: StageContext, run: Path) -> None:
        body = run / "04-body"
        body.mkdir(exist_ok=True)
        (body / "high.glb").write_bytes(box_glb())
        (body / "sections.json").write_text(json.dumps({"units": "studs", "height": 5.5, "profile": ctx.profile,
                                                        "sections": [{"name": n} for n in SECTIONS]}))
        rec = _record(4, "game_body", ["04-body/high.glb", "04-body/sections.json"])
        rec["profile"] = {"name": ctx.profile}
        write_record(run, rec)

    def _stage_6(self, ctx: StageContext, run: Path) -> None:
        out = run / "06-roblox"
        out.mkdir(exist_ok=True)
        (out / "HeroGen-R15.rbxmx").write_text('<roblox version="4"><!-- placeholder from the fake runner --></roblox>')
        (out / "bind_meshes.luau").write_text("-- placeholder from the fake runner\n")
        (out / "rig_test.server.luau").write_text("-- placeholder from the fake runner\n")
        (out / "README.txt").write_text("Placeholder files from HeroGen's fake runner (dev mode). Not a real rig.\n")
        write_record(run, _record(6, "roblox export", [f"06-roblox/{f}" for f in ROBLOX_FILES]))

    def _stage_5(self, ctx: StageContext, run: Path) -> None:
        (run / "results.json").write_text(json.dumps({"fake": True}))
        write_record(run, _record(5, "results", ["results.json"]))


class FakeSender:
    def __init__(self) -> None:
        self.sent = []

    def send(self, message) -> None:
        self.sent.append(message)
