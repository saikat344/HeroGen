"""Stand-ins for dev mode and tests: a stage runner that writes small synthetic outputs instantly.

`FakeRunner` never calls a model or Blender. It writes the files and records each real stage would (same names,
same record format), with images drawn by code, a box-shaped GLB and placeholder Roblox files whose README says
they are placeholders, so the whole service can be exercised end to end on one machine. Production refuses to
start with it (see `__main__`). `fail` makes a stage reject or fail, for tests.

It also writes the build log a real fast-lane run writes (`herogen.pipeline.events`, `FakeLog`): every model and
GPU call with its cost estimate, parallel calls, a recovery sheet in the turnaround, a texture view whose repaints
fail their outline gate, a hedged repaint try cancelled because the other try passed, and the step's checks,
paced over ``delay_s`` per stage so the website shows it arriving live. The numbers are illustrative.

Content-policy refusals (fast lane only, as in `herogen.service.runner.reword_and_retry`): ``fail`` mode "policy"
refuses the step once, the fake rewrite succeeds and the retry passes; "policy_twice" refuses the retry too;
"policy_reword_fails" makes the rewrite fail. Without ``fail``, a description holding a marker word picks the same
scenario on stage p, so the website can be checked with no spend: ``fakerefuse``, ``fakerefusetwice``,
``fakerewordfail``, and ``fakerefusecover`` (stage 1 refuses once; the sketch is redrawn before the retry). On the
fast lane every described hero's brief is also checked up front (`_precheck`): ``fakebrand`` is found and reworded
before stage p, ``fakeprecheckfail`` makes the check fail.
"""

from __future__ import annotations

import json
import re
import struct
import time
from pathlib import Path

from herogen.pipeline import events
from herogen.service.runner import Outcome, StageContext, StageSpec, write_record

SECTIONS = ("Head", "UpperTorso", "LowerTorso", "LeftUpperArm", "LeftLowerArm", "LeftHand", "RightUpperArm",
            "RightLowerArm", "RightHand", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "RightUpperLeg",
            "RightLowerLeg", "RightFoot")
ROBLOX_FILES = ("HeroGen-R15.rbxmx", "HeroGen-Starter.rbxlx", "bind_meshes.luau", "rig_test.server.luau",
                "README.txt")


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


class FakeLog:
    """Writes one stage's build-log events into its run folder, sleeping ``delay_s * share`` between them."""

    def __init__(self, ctx: StageContext, step: str, delay_s: float) -> None:
        self.file, self.step, self.a, self.delay_s = ctx.run_dir / events.FILE, step, ctx.attempt, delay_s

    def wait(self, share: float) -> None:
        if self.delay_s:
            time.sleep(self.delay_s * share)

    def emit(self, kind: str, **fields) -> None:
        events.emit(kind, file=self.file, step=self.step, a=self.a, **fields)

    def start(self, who: str, what: str, gpu: bool = False, **fields) -> events.Call:
        call = events.Call(who, what, gpu=gpu, step=self.step, file=str(self.file), a=self.a, **fields)
        call.start()
        return call

    @staticmethod
    def end(call: events.Call, usd: float = 0.0, outcome: str | None = None, code: str | None = None, **extra):
        call.usd, call.outcome, call.code = usd, outcome, code
        m = extra.pop("m", None)
        if m:
            call.m.update(m)
        call.extra.update(extra)
        call.end(None)

    def calls(self, specs: list[tuple], share: float, usd: float) -> list[events.Call]:
        """Start (who, what, fields) calls in parallel, wait, end them all."""
        started = [self.start(who, what, **fields) for who, what, fields in specs]
        self.wait(share)
        for call in started:
            self.end(call, usd)
        return started


POLICY_MODES = ("policy", "policy_twice", "policy_reword_fails")
# Marker words in a description (dev mode, whole words): (word, stage, mode).
POLICY_MARKERS = (("fakerefusetwice", "p", "policy_twice"), ("fakerewordfail", "p", "policy_reword_fails"),
                  ("fakerefusecover", "1", "policy"), ("fakerefuse", "p", "policy"))
FAKE_REWORD = ("An original armoured robot hero in red and gold plating, with a plain glowing chest light "
               "(a fake-runner rewrite)")
SAM, MUSE = "sam-3.1", "muse-image"
SAM_USD, MUSE_USD = 0.0025, 0.01
REPAINT_VIEWS = ("head+0", "body+0", "body+90", "body-90", "body+180")


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
        mode = self.fail.get(spec.key) or self._marker(spec, ctx)
        if spec.key not in ("p", "1", "2"):
            return self._run(spec, ctx, mode)
        from herogen.service.runner import redraw_sketch, run_with_rewording

        policy = mode in POLICY_MODES
        first = (lambda: (self._refuse(spec, ctx), 4)) if policy else (lambda: (self._run(spec, ctx, mode), 0))
        retry = ((lambda: (self._refuse(spec, ctx), 4)) if mode == "policy_twice"
                 else (lambda: (self._run(spec, ctx, None), 0)))
        return run_with_rewording(spec, ctx, first, retry,
                                  lambda: self._reword(spec, ctx, mode != "policy_reword_fails"),
                                  lambda: redraw_sketch(lambda s: self._step_again(s, ctx)),
                                  lambda: self._precheck(spec, ctx))

    def _precheck(self, spec: StageSpec, ctx: StageContext) -> None:
        """The fake up-front check, as `pipestage precheck` writes it: ``fakebrand`` in a description is the brand
        it finds (reworded to `FAKE_REWORD`), ``fakeprecheckfail`` makes the check fail, anything else is clean."""
        from herogen import prompts
        from herogen.pipeline import reword

        words = set(re.findall(r"[a-z]+", (ctx.prompt or "").casefold()))
        flog = FakeLog(ctx, spec.key, self.delay_s)
        call = flog.start("muse-spark", "precheck_brief")
        flog.wait(0.2)
        if "fakeprecheckfail" in words:
            flog.end(call, outcome="failed", code="http_5xx")
            reword.write(ctx.run_dir, {"status": "failed", "error": "ApiError", "http_status": 503,
                                       "cost_usd": 0.0}, reword.PRECHECK_FILE)
            return
        flog.end(call, 0.0011)
        names = ["fakebrand"] if "fakebrand" in words else []
        rec = {"status": "found" if names else "clean", "names_found": names, "model": "muse-spark-1.3",
               "prompt_version": prompts.selected("brief_precheck").ref(), "cost_usd": 0.0011}
        if names:
            call = flog.start("muse-spark", "reword_brief")
            flog.wait(0.2)
            flog.end(call, 0.0012)
            reword.write(ctx.run_dir, {"status": reword.USED, "reason": reword.PRECHECK, "original": ctx.prompt,
                                       "reworded": FAKE_REWORD, "removed_names": names, "names_found": names,
                                       "model": "muse-spark-1.3", "cost_usd": 0.0012,
                                       "prompt_version": prompts.selected("brief_reword").ref()})
            rec["rewrite"] = {"status": reword.USED, "cost_usd": 0.0012}
        reword.write(ctx.run_dir, rec, reword.PRECHECK_FILE)

    def _step_again(self, spec: StageSpec, ctx: StageContext) -> tuple[Outcome, int]:
        """An earlier step run again inside a retry (the sketch redrawn from a reworded brief)."""
        self.calls.append(spec.key)
        return self._run(spec, ctx, None), 0

    @staticmethod
    def _marker(spec: StageSpec, ctx: StageContext) -> str | None:
        words = set(re.findall(r"[a-z]+", (ctx.prompt or "").casefold())) if ctx.kind == "prompt" else set()
        return next((mode for word, stage, mode in POLICY_MARKERS if word in words and stage == spec.key), None)

    def _refuse(self, spec: StageSpec, ctx: StageContext) -> Outcome:
        """The image model refuses the step's request under its content policy (exit code 4)."""
        from herogen.service.runner import CliRunner

        run = ctx.run_dir
        run.mkdir(parents=True, exist_ok=True)
        flog = FakeLog(ctx, spec.key, self.delay_s)
        what = {"p": "sketch_design", "1": "cover_art", "2": "turnaround"}[spec.key]
        call = flog.start(MUSE, what)
        flog.wait(0.3)
        flog.end(call, outcome="failed", code="http_4xx")
        rec = _record(spec.key if spec.key == "p" else int(spec.key), spec.label, [], ok=False,
                      details={"brief_slot_used": spec.key == "1"})
        rec.update(status="error", error="refused by the Model API's content policy (fake runner)")
        write_record(run, rec)
        return CliRunner.outcome(spec, ctx, 0.01, 4, "", "", False)

    def _reword(self, spec: StageSpec, ctx: StageContext, ok: bool) -> None:
        """The fake rewrite: one Muse Spark call in the build log and reword.json, as `pipestage reword` writes."""
        from herogen import prompts
        from herogen.pipeline import reword

        flog = FakeLog(ctx, spec.key, self.delay_s)
        call = flog.start("muse-spark", "reword_brief")
        flog.wait(0.3)
        if ok:
            flog.end(call, 0.0012)
            reword.write(ctx.run_dir, {"status": reword.USED, "reason": reword.REASON, "original": ctx.prompt,
                                       "reworded": FAKE_REWORD, "removed_names": ["fake brand"],
                                       "model": "muse-spark-1.3",
                                       "prompt_version": prompts.selected("brief_reword").ref(), "cost_usd": 0.0012})
        else:
            flog.end(call, outcome="failed", code="http_5xx")
            reword.write(ctx.run_dir, {"status": reword.FAILED, "reason": reword.REASON, "original": ctx.prompt,
                                       "error": "ApiError", "cost_usd": 0.0})

    def _run(self, spec: StageSpec, ctx: StageContext, mode: str | None) -> Outcome:
        flog = FakeLog(ctx, spec.key, self.delay_s)
        t0 = time.monotonic()
        what = {"p": "sketch_design", "1": "cover_art", "2": "turnaround"}.get(spec.key)
        if mode == "transient" and what:
            ctx.run_dir.mkdir(parents=True, exist_ok=True)
            call = flog.start(MUSE, what)
            flog.wait(1.0)
            flog.end(call, outcome="failed", code="http_5xx")
        elif mode not in ("error", "transient", "raise"):
            ctx.run_dir.mkdir(parents=True, exist_ok=True)
            script = getattr(self, f"_log_{spec.key}", None)
            if script is not None:   # texture refine logs from `_stage_3f`, as it publishes its versions
                script(ctx, flog)
        elif self.delay_s:
            time.sleep(self.delay_s)
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
        code = 0
        if mode == "reject":
            rec = json.loads((run / spec.record).read_text())
            rec["ok"] = False
            rec["checks"] = [{"name": "height_agreement_4_views", "value": 0.81, "threshold": 0.95, "op": ">=",
                              "pass": False, "gate": True}]
            if spec.key in ("p", "1", "2"):
                rec["status"] = "gate_failed"
                code = 6 if spec.key == "p" else 2
            write_record(run, rec)
        from herogen.service.runner import CliRunner

        return CliRunner.outcome(spec, ctx, round(max(0.01, time.monotonic() - t0), 2), code, "", "", False)

    # ---- build-log scripts (one per stage; shares of delay_s add up to about 1) ----------------------------

    def _log_p(self, ctx: StageContext, log: FakeLog) -> None:
        log.calls([(MUSE, "sketch_design", {})], 0.4, MUSE_USD)
        log.emit("image", what="design")
        log.calls([(MUSE, "full_body_sketch", {})], 0.4, MUSE_USD)
        log.emit("image", what="full_body")
        log.emit("skip", what="identity_checks", why="fast_lane")
        log.wait(0.2)

    def _log_0(self, ctx: StageContext, log: FakeLog) -> None:
        log.wait(0.5)
        log.emit("check", name="min_side_px", ok=True, gate=True, op=">=", m={"value": 512, "limit": 256})
        log.emit("check", name="line_contrast", ok=True, gate=True, op=">=", m={"value": 203, "limit": 32})
        log.emit("local", what="input_checked", s=round(self.delay_s * 0.5, 3), m={"min_side_px": 512,
                                                                                   "contrast": 203})
        log.emit("image", what="sketch")
        log.emit("checks", n=2, passed=2, gated=2, failed=[], advisory=[])
        log.wait(0.5)

    def _log_1(self, ctx: StageContext, log: FakeLog) -> None:
        log.calls([(SAM, "cover_art", {"target": "character"}), (SAM, "cover_art", {"target": "feet"})], 0.2,
                  SAM_USD)
        log.calls([(MUSE, "cover_art", {})], 0.5, MUSE_USD)
        log.emit("image", what="cover")
        log.emit("checks", n=5, passed=5, gated=2, failed=[], advisory=[])
        log.wait(0.3)

    def _log_2(self, ctx: StageContext, log: FakeLog) -> None:
        sheet, top = log.start(MUSE, "turnaround"), log.start(MUSE, "top_view")
        log.wait(0.2)
        log.end(sheet, MUSE_USD)
        log.calls([(SAM, "turnaround", {"target": "character"})], 0.05, SAM_USD)
        # The first sheet's figures overlap: the fast lane asks for one more sheet, once.
        log.emit("decision", what="recovery_sheet", why="too_few_figures", m={"figures": 3})
        again = log.start(MUSE, "turnaround")
        log.wait(0.1)
        log.end(top, MUSE_USD)
        log.wait(0.1)
        log.end(again, MUSE_USD)
        log.calls([(SAM, "turnaround", {"target": "character"})], 0.05, SAM_USD)
        log.emit("local", what="sheet_split", m={"figures": 4, "height_agreement": 0.99})
        log.calls([(SAM, "top_view", {"target": "character"})], 0.05, SAM_USD)
        log.calls([(SAM, "turnaround", {"target": "face"}), (SAM, "turnaround", {"target": "head"})], 0.05, SAM_USD)
        for view in ("turnaround", "front", "right", "back", "left", "top"):
            log.emit("image", what=view)
        log.emit("checks", n=13, passed=10, gated=4, failed=[],
                 advisory=["top_depth_ratio", "carve_iou", "top_mask_iou"])
        log.wait(0.4)

    def _log_3(self, ctx: StageContext, log: FakeLog) -> None:
        shape = log.start("sam-3d-objects", "3d_shape", gpu=True)
        log.wait(0.3)
        shape.emit("gpu_wait", what="3d_shape", s=round(self.delay_s * 0.3, 3))   # "still waiting"
        log.wait(0.2)
        log.end(shape, 0.101, gpu="a100-40gb", m={"load_s": 60.4, "infer_s": 96.1})
        log.emit("local", what="mesh_placed", m={"triangles": 217836, "rotations_tested": 4})
        body = log.start("sam-3d-body", "body_joints", gpu=True)
        log.wait(0.3)
        log.end(body, 0.012, gpu="a10g", m={"load_s": 18.9, "infer_s": 4.2})
        log.emit("local", what="joints_mapped", m={"joints": 16, "from_model": 12, "flagged": 7})
        log.emit("skip", what="acceptance_judges", why="fast_lane")
        log.emit("checks", n=5, passed=5, gated=3, failed=[], advisory=[])
        log.wait(0.2)

    def _log_3f_wave(self, log: FakeLog, wave: int) -> None:
        if wave == 1:
            log.emit("local", what="views_rendered", m={"cameras": 5})
            log.calls([(SAM, "texture_refine", {"target": "face"}), (SAM, "texture_refine", {"target": "head"})],
                      0.05, SAM_USD)
            views = REPAINT_VIEWS
        else:
            views = ("hair-upper+180",)
        log.emit("decision", what="wave", n=2 * len(views), m={"wave": wave, "views": len(views), "tries": 2})
        tries = {(v, t): log.start(MUSE, "repaint", view=v, try_=t) for v in views for t in (1, 2)}
        log.wait(0.3)
        for (view, t), call in tries.items():
            if (view, t) == ("body-90", 2):
                continue
            log.end(call, MUSE_USD)
        if wave == 1:
            # Try 1 of the -90 view passed first, so try 2 is cancelled (its request was already paid for).
            log.end(tries[("body-90", 2)], MUSE_USD, outcome="cancelled")
        log.calls([(SAM, "texture_refine", {"target": "face", "view": v}) for v in views], 0.05, SAM_USD)
        for view in views:
            if view in ("head+0", "body+0"):
                for t, (median, p90) in ((1, (4.7, 13.2)), (2, (5.4, 18.0))):
                    limit = (6, 16) if view == "body+0" else (10, 24)
                    if view == "body+0" and t == 2:
                        log.emit("check", name="outline_p90_px", ok=False, gate=True, op="<=",
                                 m={"value": p90, "limit": limit[1]})
                    log.emit("decision", what="view_result", view=view, try_=t, ok=False, why="outline_moved",
                             m={"median_px": median, "median_limit": limit[0], "p90_px": p90,
                                "p90_limit": limit[1]})
                log.emit("decision", what="view_kept", view=view, ok=False, why="outline_moved", m={"tries": 2})
            else:
                log.emit("decision", what="view_result", view=view, try_=1, ok=True,
                         m={"overlap": 0.93, "median_px": 2.1})
                log.emit("decision", what="view_selected", view=view, try_=1, ok=True, m={"tries": 2})
                if view == "body-90":
                    log.emit("decision", what="view_result", view=view, try_=2, ok=False, why="lower_try_accepted")
        log.wait(0.2)

    def _log_4(self, ctx: StageContext, log: FakeLog) -> None:
        log.emit("local", what="landmarks_loaded", m={"joints": 16, "from_model": 12})
        log.wait(0.3)
        log.emit("local", what="decimated", s=round(self.delay_s * 0.3, 3),
                 m={"source_triangles": 217836, "high": 14400, "low": 2192})
        log.wait(0.3)
        log.emit("local", what="cuts_searched", s=round(self.delay_s * 0.3, 3),
                 m={"neck_radius": -0.5, "waist_radius": 0.5})
        log.emit("local", what="parts_split", m={"parts": 15, "high_triangles": 23754, "low_triangles": 5982})
        log.emit("checks", n=49, passed=45, gated=8, failed=[],
                 advisory=["max_joint_gap_posed", "pivots_in_joint_region", "joint_gap_left_elbow",
                           "joint_gap_right_elbow"])
        log.wait(0.1)

    def _log_6(self, ctx: StageContext, log: FakeLog) -> None:
        log.emit("local", what="rig_written", m={"parts": 15, "joints": 15, "attachments": 19, "hip_height": 1.719})
        log.wait(0.5)
        log.emit("checks", n=43, passed=42, gated=42, failed=[], advisory=["meshes_bound"])
        log.wait(0.5)

    def _log_5(self, ctx: StageContext, log: FakeLog) -> None:
        log.wait(0.5)
        log.emit("local", what="results_written")
        log.wait(0.5)

    def _stage_p(self, ctx: StageContext, run: Path) -> None:
        from herogen import prompts
        from herogen.pipeline.prompt_sketch import DESIGN, SKETCH

        for rel in (DESIGN, SKETCH):
            _figure(512, (60, 60, 60)).save(run / rel, format="PNG")
        from herogen.pipeline import reword

        write_record(run, _record("p", "prompt_sketch", [DESIGN, SKETCH], cost=0.01,
                                  details={"user_text": reword.brief(run) or ctx.prompt,
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


    def _stage_3(self, ctx: StageContext, run: Path) -> None:
        (run / "03-mesh.glb").write_bytes(box_glb())
        (run / "03-source.glb").write_bytes(box_glb())
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

    def _stage_3f(self, ctx: StageContext, run: Path) -> None:
        versions = []
        flog = FakeLog(ctx, "3f", self.delay_s)
        for n, accepted in ((1, ['front', 'right']), (2, ['front', 'right', 'back', 'left'])):
            self._log_3f_wave(flog, n)
            self.publish_version(run, versions, n, accepted)
            flog.emit("version", n=n, passed=3 if n == 1 else 4, of=6)
            if self.delay_s:
                time.sleep(self.delay_s * 0.2)
        (run / "03-refined.glb").write_bytes((run / versions[-1]['glb']).read_bytes())
        self.publish_progress(run, versions, done=True)
        write_record(run, {**_record("3f", "texture refine", ["03-refined.glb"]), "status": "complete",
                           "profile": ctx.profile})

    def publish_version(self, run: Path, versions: list[dict], n: int, accepted: list[str]) -> None:
        """Publish a synthetic version; browser fixtures may pause at each atomic publication."""
        from datetime import datetime, timezone

        name = f'03-refined-v{n}.glb'
        temporary = run / f'{name}.tmp'
        temporary.write_bytes(box_glb())
        temporary.replace(run / name)
        versions.append({'n': n, 'glb': name, 'accepted': accepted, 'rejected': [],
                         'pending': [v for v in ('front', 'right', 'back', 'left') if v not in accepted],
                         'time': datetime.now(timezone.utc).isoformat()})
        self.publish_progress(run, versions)

    def publish_progress(self, run: Path, versions: list[dict], done: bool = False) -> None:
        temporary = run / 'stage-3f-progress.json.tmp'
        temporary.write_text(json.dumps({'versions': versions, 'done': done}))
        temporary.replace(run / 'stage-3f-progress.json')

    def _stage_6(self, ctx: StageContext, run: Path) -> None:
        out = run / "06-roblox"
        out.mkdir(exist_ok=True)
        (out / "HeroGen-R15.rbxmx").write_text('<roblox version="4"><!-- placeholder from the fake runner --></roblox>')
        (out / "HeroGen-Starter.rbxlx").write_text('<roblox version="4"><!-- placeholder from the fake runner --></roblox>')
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
