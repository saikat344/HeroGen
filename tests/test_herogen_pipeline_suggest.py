"""Refine suggestions (herogen/pipeline/suggest.py): text checks, static fallbacks for every failure, the atomic
write, the exit codes and the file's shape, with a fake client and a fake renderer (no Blender, no Model API)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from PIL import Image  # noqa: E402

from herogen import prompts  # noqa: E402
from herogen.blender.bl import BlenderError  # noqa: E402
from herogen.pipeline import suggest as S  # noqa: E402
from herogen.pipeline.client import ApiError, CallLog, ClientError, ModelClient, ReplayMiss  # noqa: E402

CAM = {"name": "body+0", "centre": [0.0, 0.0, 0.0], "angle": 0.0, "ortho": 1.06, "res": [64, 96]}
GOOD = {"texture": {"text": "Make the goggle lenses amber like the artwork.", "basis": "mismatch"},
        "turnaround": {"text": "Add a brown leather tool belt with brass buckles.", "basis": "variation"}}
CAP_USD = prompts.load("refine-suggest-v1").step("suggest")["params"]["max_output_tokens"] * 4.25 / 1e6
KEYS = {"version", "source", "texture", "turnaround", "model", "prompt", "prompt_sha256", "cost_usd", "calls",
        "error", "created"}


def make_run(root: Path, *, cover=True, texture=True, blend=True, camera=True, text="A tiny dragon tamer") -> Path:
    run = root / "runs" / "j_test"
    (run / S.REFINE / "prep").mkdir(parents=True)
    if cover:
        Image.new("RGB", (64, 96), (200, 150, 40)).save(run / S.COVER)
    if texture:
        Image.new("RGB", (16, 16), (90, 90, 90)).save(run / S.REFINE / "texture.png")
    if blend:
        (run / S.REFINE / "prep" / "work.blend").write_bytes(b"BLENDER")
    views = {"body+0": {"accepted": True, "camera": CAM}} if camera else {"body+0": {"accepted": False}}
    (run / "stage-3f.json").write_text(json.dumps({"views": views}))
    if text is not None:
        (run / "stage-p.json").write_text(json.dumps({"details": {"user_text": text}}))
    return run


def fake_render(run, cam, timeout=S.RENDER_TIMEOUT_S):
    work = run / S.WORK
    work.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 96), (120, 120, 120)).save(work / "front.png")
    return work / "front.png", {"source": "work.blend", "camera": cam}


def timed_out_render(run, cam, timeout=S.RENDER_TIMEOUT_S):
    raise S.SuggestTimeout(f"the front render timed out after {timeout:g} s")


class FakeClient:
    """Answers like `ModelClient.compare_images`: logs one call record per paid call, then answers or raises."""

    def __init__(self, answer=None, error=None, sleep=None, cost=0.0042):
        self.log = CallLog()
        self.answer, self.error, self.sleep, self.cost = answer, error, sleep, cost
        self.prompts: list[str] = []
        self.kwargs: dict = {}

    def compare_images(self, prompt, images, **kwargs):
        self.prompts.append(prompt)
        self.kwargs = kwargs
        if self.sleep is not None:
            self.sleep.wait(5)
            return self.answer
        self.log.add(kwargs["model"], "compare", "k1", cached=False, cost_usd=self.cost)
        if self.error is not None:
            raise self.error
        return self.answer


def read(run: Path) -> dict:
    return json.loads((run / S.FILE).read_text())


class CheckText(unittest.TestCase):
    def test_accepts_a_plain_instruction(self):
        self.assertEqual(S.check_text("texture", "  Make the goggle lenses   amber like the artwork. "),
                         ("Make the goggle lenses amber like the artwork.", None))
        self.assertEqual(S.check_text("turnaround", "Add a hooded cloak and make the boots dark brown.")[1], None)

    def test_static_texts_pass_their_own_checks(self):
        for step in S.STEPS:
            self.assertEqual(S.check_text(step, S.STATIC[step]), (S.STATIC[step], None))

    def test_empty_long_and_two_sentences(self):
        for step in S.STEPS:
            self.assertEqual(S.check_text(step, ""), (None, "empty"))
            self.assertEqual(S.check_text(step, None), (None, "empty"))
            self.assertIn("characters", S.check_text(step, "Make the coat " + "very " * 20 + "red.")[1])
            self.assertEqual(S.check_text(step, "Make the coat red. Make the boots black.")[1],
                             "more than one sentence")
        self.assertEqual(S.check_text("texture", "x" * 100)[1], None)
        self.assertIsNotNone(S.check_text("texture", "x" * 101)[1])

    def test_texture_refuses_shape_gear_pose_and_face(self):
        for text in ("Add a red scarf.", "Remove the belt.", "Make the coat longer.", "Make her shorter and red.",
                     "Make the hero taller.", "Change the pose to a wave.", "Have her hold a lantern.",
                     "Give her a fierce expression.", "Add a smile.", "Make her smiling.",
                     "Soften the face shape.", "Removing the hat helps."):
            with self.subTest(text=text):
                self.assertIsNone(S.check_text("texture", text)[0])

    def test_texture_allows_colour_and_material_words(self):
        for text in ("Make the padded jacket deep red leather.", "Give the apron a worn, dusty finish.",
                     "Paint a gold emblem on the chest plate."):
            with self.subTest(text=text):
                self.assertIsNotNone(S.check_text("texture", text)[0])

    def test_turnaround_refuses_pose_props_wind_and_camera(self):
        for text in ("Change the pose to a battle stance.", "Have him hold a lantern.",
                     "Give her a sword held in hand.", "Put a staff in her hands.", "Add an in-hand torch.",
                     "Make the cape wind-blown.", "Give him a flowing scarf.", "Show a side view.",
                     "Use a lower camera angle.", "Add a forest background."):
            with self.subTest(text=text):
                self.assertIsNone(S.check_text("turnaround", text)[0])

    def test_turnaround_allows_gear_changes(self):
        for text in ("Add a hooded cloak and make the boots dark brown.", "Remove the helmet and add a red bandana.",
                     "Give him hand-painted leather armour."):
            with self.subTest(text=text):
                self.assertIsNotNone(S.check_text("turnaround", text)[0])


class Inputs(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_description_prefers_the_used_rewrite_and_strips_quote_marks(self):
        run = make_run(self.tmp, text="A pirate «captain» with a parrot")
        self.assertEqual(S.description(run), "A pirate captain with a parrot")
        (run / "reword.json").write_text(json.dumps({"status": "used", "reworded": "An original sea captain"}))
        self.assertEqual(S.description(run), "An original sea captain")

    def test_description_without_a_record(self):
        self.assertEqual(S.description(make_run(self.tmp, text=None)), S.NO_DESCRIPTION)

    def test_camera_from_stage_3f_else_character_json(self):
        run = make_run(self.tmp)
        self.assertEqual(S.front_camera(run), (CAM, "stage-3f.json"))
        (run / "stage-3f.json").write_text(json.dumps({"views": {"body+0": {"accepted": False}}}))
        with self.assertRaises(S.MissingInput):
            S.front_camera(run)
        (run / S.REFINE / "character.json").write_text(json.dumps({"body_centre": [0.1, 0.2, 0.3], "height": 1.0}))
        cam, src = S.front_camera(run)
        self.assertEqual(src, "character.json")
        self.assertEqual((cam["name"], cam["angle"], cam["res"]), ("body+0", 0.0, [1024, 1536]))
        self.assertAlmostEqual(cam["ortho"], 1.06)


class RenderFront(unittest.TestCase):
    def setUp(self):
        self.run = make_run(Path(tempfile.mkdtemp()))
        (self.run / S.GLB).write_bytes(b"glTF")
        self.jobs: list[str] = []

    def blender(self, fail=None):
        def call(mode, args, log, timeout=900.0):
            self.jobs.append(mode)
            self.assertTrue(str(log).startswith(str(self.run / S.WORK)))
            self.assertTrue(args["out"].startswith(str(self.run / S.WORK)))
            if fail and mode in fail:
                raise fail[mode]
            im = Image.new("RGBA", (64, 96), (0, 0, 0, 0))
            im.paste((200, 30, 30, 255), (20, 10, 40, 90))
            im.save(Path(args["out"]) / f"{args['cams'][0]['name']}.png")
            return {}
        return call

    def test_renders_work_blend_and_crops_to_the_figure(self):
        out, how = S.render_front(self.run, CAM, blender=self.blender())
        self.assertEqual((self.jobs, how["source"], out), (["render"], "work.blend", self.run / S.WORK / "front.png"))
        with Image.open(out) as im:
            self.assertLess(im.width, 64)
            self.assertEqual(im.getpixel((0, 0)), S.BACKGROUND)

    def test_falls_back_to_the_glb_when_work_blend_fails(self):
        _, how = S.render_front(self.run, CAM, blender=self.blender({"render": BlenderError("failed (exit 1)")}))
        self.assertEqual((self.jobs, how["source"]), (["render", "probe"], S.GLB))

    def test_a_timeout_is_not_retried(self):
        err = BlenderError("timed out")
        err.__cause__ = subprocess.TimeoutExpired("blender", 120)
        with self.assertRaises(S.SuggestTimeout):
            S.render_front(self.run, CAM, blender=self.blender({"render": err}))
        self.assertEqual(self.jobs, ["render"])


class Suggest(unittest.TestCase):
    def setUp(self):
        self.run = make_run(Path(tempfile.mkdtemp()))
        self.version = prompts.selected("refine_suggest")

    def go(self, client, **kw):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = S.suggest(self.run, client, render=kw.pop("render", fake_render), **kw)
        return code, err.getvalue()

    def assert_shape(self, data):
        self.assertEqual(set(data), KEYS)
        self.assertEqual((data["version"], data["prompt"], data["prompt_sha256"]),
                         (1, "refine-suggest-v1", self.version.sha256))
        for step in S.STEPS:
            self.assertEqual(set(data[step]), {"text", "basis"})
            self.assertLessEqual(len(data[step]["text"]), 100)
        self.assertTrue(data["created"].endswith("Z"))

    def test_spark_answer_is_written(self):
        client = FakeClient(GOOD)
        code, err = self.go(client)
        data = read(self.run)
        self.assertEqual((code, err), (0, ""))
        self.assert_shape(data)
        self.assertEqual((data["source"], data["model"], data["error"]), ("spark", "muse-spark-1.3", None))
        self.assertEqual(data["texture"], GOOD["texture"])
        self.assertEqual(data["turnaround"], GOOD["turnaround"])
        self.assertEqual((data["cost_usd"], len(data["calls"])), (0.0042, 1))
        self.assertIn("«A tiny dragon tamer»", client.prompts[0])
        self.assertEqual(client.kwargs["image_max_side"], 768)
        self.assertEqual(json.loads((self.run / S.WORK / "check.json").read_text())["refused"], {})

    def test_a_refused_step_gets_its_static_text(self):
        answer = {"texture": {"text": "Add a long red scarf.", "basis": "variation"},
                  "turnaround": GOOD["turnaround"]}
        code, _ = self.go(FakeClient(answer))
        data = read(self.run)
        self.assertEqual((code, data["source"]), (0, "spark"))
        self.assertEqual(data["texture"], {"text": S.STATIC["texture"], "basis": "static"})
        self.assertEqual(data["turnaround"], GOOD["turnaround"])
        self.assertIn("texture", json.loads((self.run / S.WORK / "check.json").read_text())["refused"])

    def assert_static(self, code, err, error_type):
        data = read(self.run)
        self.assertEqual(code, 3)
        self.assert_shape(data)
        self.assertEqual(data["source"], "static")
        for step in S.STEPS:
            self.assertEqual(data[step], {"text": S.STATIC[step], "basis": "static"})
        self.assertTrue(data["error"].startswith(error_type + ":"), data["error"])
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertIn(error_type, err)
        return data

    def test_api_error_writes_the_static_file_with_its_cost(self):
        code, err = self.go(FakeClient(error=ApiError("responses failed: HTTP 500", http_status=500)))
        data = self.assert_static(code, err, "ApiError")
        self.assertEqual((data["model"], data["cost_usd"], len(data["calls"])), ("muse-spark-1.3", 0.0042, 1))

    def test_unparseable_reply_writes_the_static_file(self):
        self.assert_static(*self.go(FakeClient(error=ClientError("responses entry k1 is not JSON"))), "ClientError")

    def test_replay_miss_writes_the_static_file(self):
        self.assert_static(*self.go(FakeClient(error=ReplayMiss("no cached responses entry"))), "ReplayMiss")

    def test_spark_timeout_counts_the_call_at_its_cap(self):
        release = threading.Event()
        try:
            code, err = self.go(FakeClient(GOOD, sleep=release), spark_deadline_s=0.2)
        finally:
            release.set()
        data = self.assert_static(code, err, "SuggestTimeout")
        self.assertEqual(data["calls"][-1]["outcome"], "unknown")
        self.assertAlmostEqual(data["cost_usd"], CAP_USD)

    def test_render_timeout_writes_the_static_file_without_a_call(self):
        client = FakeClient(GOOD)
        data = self.assert_static(*self.go(client, render=timed_out_render), "SuggestTimeout")
        self.assertEqual((data["model"], data["calls"], data["cost_usd"], client.prompts), (None, [], 0.0, []))

    def test_missing_inputs_write_nothing(self):
        for missing in ("cover", "texture", "blend", "camera"):
            with self.subTest(missing=missing):
                run = make_run(Path(tempfile.mkdtemp()), **{missing: False})
                with self.assertRaises(S.MissingInput):
                    S.suggest(run, FakeClient(GOOD), render=fake_render)
                self.assertFalse((run / S.FILE).exists())


class AtomicWrite(unittest.TestCase):
    def test_replaces_whole_or_leaves_the_old_file(self):
        d = Path(tempfile.mkdtemp())
        path = d / S.FILE
        S.write_atomic(path, {"a": 1})
        self.assertEqual(json.loads(path.read_text()), {"a": 1})
        with self.assertRaises(TypeError):
            S.write_atomic(path, {"a": object()})
        self.assertEqual(json.loads(path.read_text()), {"a": 1})
        self.assertEqual(sorted(p.name for p in d.iterdir()), [S.FILE])

    def test_a_failed_rename_leaves_no_file(self):
        d = Path(tempfile.mkdtemp())
        with mock.patch.object(S.os, "replace", side_effect=OSError("disk")), self.assertRaises(OSError):
            S.write_atomic(d / S.FILE, {"a": 1})
        self.assertEqual(list(d.iterdir()), [])


def response(answer, usage=None):
    return {"status": "completed", "usage": usage if usage is not None else {"input_tokens": 2000, "output_tokens": 300},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(answer)}]}]}


class Cli(unittest.TestCase):
    """`main` with the real client: its `_post` is faked, so pricing, call records and the cache are the client's."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.run = make_run(self.tmp)
        self.cache = self.tmp / "cache"
        env = mock.patch.dict(os.environ, {"MODEL_API_KEY": "test-key-not-real"})
        env.start()
        self.addCleanup(env.stop)
        render = mock.patch.object(S, "render_front", fake_render)
        render.start()
        self.addCleanup(render.stop)

    def main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = S.main([str(a) for a in argv])
            except SystemExit as e:
                code = e.code
        return code, err.getvalue()

    def test_exit_0_prices_like_the_client_and_a_rerun_replays(self):
        with mock.patch.object(ModelClient, "_post", return_value=response(GOOD)) as post:
            code, err = self.main(self.run, "--cache", self.cache)
        self.assertEqual((code, err, post.call_count), (0, "", 1))
        data = read(self.run)
        self.assertEqual(data["source"], "spark")
        self.assertAlmostEqual(data["cost_usd"], (2000 * 1.25 + 300 * 4.25) / 1e6)
        code, _ = self.main(self.run, "--cache", self.cache, "--replay")
        again = read(self.run)
        self.assertEqual((code, again["cost_usd"], again["calls"][0]["cached"]), (0, 0.0, True))
        self.assertEqual(again["texture"], data["texture"])

    def test_missing_usage_is_priced_at_the_output_cap(self):
        with mock.patch.object(ModelClient, "_post", return_value=response(GOOD, usage={})):
            code, _ = self.main(self.run, "--cache", self.cache)
        self.assertEqual(code, 0)
        self.assertAlmostEqual(read(self.run)["cost_usd"], CAP_USD)

    def test_exit_3_on_an_api_error(self):
        with mock.patch.object(ModelClient, "_post", side_effect=ApiError("responses failed: HTTP 503",
                                                                          http_status=503)):
            code, err = self.main(self.run, "--cache", self.cache)
        self.assertEqual((code, read(self.run)["source"]), (3, "static"))
        self.assertIn("ApiError", err)

    def test_exit_3_on_a_reply_that_breaks_the_schema(self):
        with mock.patch.object(ModelClient, "_post", return_value=response({"texture": "red"})):
            code, err = self.main(self.run, "--cache", self.cache)
        self.assertEqual((code, read(self.run)["error"].split(":")[0]), (3, "ClientError"))

    def test_exit_1_on_missing_inputs(self):
        (self.run / S.COVER).unlink()
        code, err = self.main(self.run, "--cache", self.cache, "--replay")
        self.assertEqual(code, 1)
        self.assertIn("01-cover.png", err)
        self.assertFalse((self.run / S.FILE).exists())
        code, _ = self.main(self.tmp / "no-such-run", "--replay")
        self.assertEqual(code, 1)

    def test_exit_1_on_an_internal_error(self):
        with mock.patch.object(S, "description", side_effect=RuntimeError("boom")):
            code, err = self.main(self.run, "--cache", self.cache, "--replay")
        self.assertEqual(code, 1)
        self.assertIn("internal error: RuntimeError: boom", err)
        self.assertFalse((self.run / S.FILE).exists())

    def test_exit_2_on_usage(self):
        self.assertEqual(self.main()[0], 2)
        self.assertEqual(self.main(self.run, "--bogus")[0], 2)

    def test_default_cache_follows_the_service_layout(self):
        self.assertEqual(S.default_cache(self.run), (self.tmp / "cache").resolve())
        other = self.tmp / "elsewhere" / "j_x"
        other.mkdir(parents=True)
        self.assertEqual(S.default_cache(other), (self.tmp / "elsewhere" / "_cache-model").resolve())


if __name__ == "__main__":
    unittest.main()
