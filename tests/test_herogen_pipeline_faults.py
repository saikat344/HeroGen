"""Fault injection through a local fake Model API server: HTTP 429 and 500 errors and empty or partial SAM
results never leave a stage ok on a mask SAM 3.1 did not make, and a failed call is never cached."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.pipeline import checks as C  # noqa: E402
from herogen.pipeline import client as client_mod  # noqa: E402
from herogen.pipeline import heldout, retry, stages  # noqa: E402
from herogen.pipeline.__main__ import run_pipeline  # noqa: E402
from herogen.pipeline.client import ModelClient, ReplayMiss, SamInstance  # noqa: E402
from herogen.pipeline.results import collect  # noqa: E402
from herogen.pipeline.runfolder import RunFolder  # noqa: E402
from fake_model_server import (  # noqa: E402
    FakeModelServer, data_url_image, error_body, pipeline_without_waits, pixels_key, sam_response,
)
from test_herogen_pipeline_flow import DUMMY_KEY, FakeAPI  # noqa: E402
from test_herogen_pipeline_heldout_synth import REQUEST as HELDOUT_REQUEST  # noqa: E402
from test_herogen_pipeline_synth import figure_mask, paint, place  # noqa: E402

KINDS = {stages.COVER_PROMPT: "cover", stages.TURNAROUND_PROMPT: "sheet", stages.TOP_PROMPT: "top"}


class FaultyAPI(FakeAPI):
    """The flow tests' fake API behind a real HTTP server, with faults per kind of SAM input image.

    ``faults`` maps an image kind ("sketch", "cover", "sheet", "top", "heldout") to what a SAM call on it
    returns: an HTTP status (429, 500), "empty" (no instances), "partial" (the first figure's mask loses all
    but its top 30%), or "incomplete" / "failed" (HTTP 200 whose status says the answer did not complete, with
    no output). ``texts`` limits the faults to these SAM prompts (default: all). ``muse_faults`` maps a Muse
    request kind ("cover", "sheet", ...) to "failed": an HTTP 200 with status failed and no image.
    """

    def __init__(self, faults: dict[str, object], texts: tuple[str, ...] | None = None,
                 muse_faults: dict[str, str] | None = None, **kw) -> None:
        super().__init__(**kw)
        self.faults = faults
        self.muse_faults = muse_faults or {}
        self.texts = texts
        self.kind_of: dict[str, str] = {}
        self.sam_kinds: list[str] = []
        self.sam_texts: list[tuple[str, str]] = []
        self.partial = False

    def respond(self, endpoint: str, body: dict):
        if endpoint != "responses":
            kind = KINDS.get(body.get("prompt"), "heldout" if HELDOUT_REQUEST.search(body["prompt"]) else "other")
            if self.muse_faults.get(kind) == "failed":
                return 200, {}, {"status": "failed", "error": {"code": "server_error", "message": "fake"}, "data": []}
            out = self.post(None, endpoint, body)
            img = data_url_image("data:," + out["data"][0]["b64_json"])
            self.kind_of[pixels_key(img)] = kind
            return 200, {}, out
        content = body["input"][0]["content"]
        text, img = content[0]["text"], data_url_image(content[1]["image_url"])
        kind = self.kind_of.get(pixels_key(img), "sketch")
        self.sam_kinds.append(kind)
        self.sam_texts.append((kind, text))
        fault = self.faults.get(kind) if self.texts is None or text in self.texts else None
        if isinstance(fault, int):
            return fault, {}, error_body(fault)
        if fault == "empty":
            return 200, {}, sam_response("")
        if fault in ("incomplete", "failed"):
            out = {**sam_response(""), "status": fault, "output": []}
            if fault == "incomplete":
                out["incomplete_details"] = {"reason": "max_output_tokens"}
            else:
                out["error"] = {"code": "server_error", "message": "fake failure"}
            return 200, {}, out
        self.post(None, endpoint, body)  # sets the image the decoder thresholds
        self.partial = fault == "partial"
        return 200, {}, sam_response("fake")

    def decode(self, text: str, size) -> list[SamInstance]:
        if not text.strip():
            return []
        out = super().decode(text, size)
        if self.partial and out:
            m = out[0].mask.copy()
            ys = np.flatnonzero(m.any(axis=1))
            m[ys[0] + int(0.3 * (ys[-1] - ys[0])):] = False
            out[0] = SamInstance(mask=m, box=out[0].box, confidence=0.9)
        return out


def ok_records_on_non_sam_masks(root: Path) -> list[str]:
    """Every ok stage record under ``root`` whose masks include one SAM 3.1 did not make (the top view's
    visual hull, built from the side views' SAM masks, is the one other method allowed)."""
    bad = []
    for p in sorted(root.glob("*/stage-*.json")):
        rec = json.loads(p.read_text())
        if not rec.get("ok"):
            continue
        det = rec.get("details") or {}
        methods = dict(det.get("mask_method_detail") or det.get("mask_method") or {})
        for name, v in (det.get("views") or {}).items():  # stage 2h: each view's kept generation
            methods[name] = v["attempts"][v["kept_attempt"]]["mask_method"]
        attempts = det.get("attempts") or []
        if attempts:  # stage 2: the kept sheet
            methods["split"] = attempts[det.get("kept_attempt", len(attempts) - 1)]["split_source"]
        for what, m in methods.items():
            if not (str(m).startswith(client_mod.SAM_MODEL) and m != stages.SAM_NOTHING_FOUND) \
                    and not (what == "top" and m == "visual_hull"):
                bad.append(f"{p.parent.name}/{p.name} {what}: {m}")
    return bad


class FaultInjectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "runs"
        self.sketch = Path(self.tmp.name) / "sketch.png"
        paint(place(figure_mask(200, "front"), (300, 280), (50, 40)), colour=(90, 90, 90)).save(self.sketch)
        self.err = ""

    def tearDown(self):
        # Whatever the fault, no stage is ever ok on a mask SAM did not make.
        self.assertEqual(ok_records_on_non_sam_masks(self.root), [])
        self.tmp.cleanup()

    def _live(self, api: FaultyAPI, run_id: str = "r", stage_list=(0, 1, 2, 5)) -> int:
        server = FakeModelServer(api.respond, DUMMY_KEY)
        err = io.StringIO()
        with server.serving(self.tmp.name + "/limiter"), mock.patch.dict(os.environ, {"MODEL_API_KEY": DUMMY_KEY}), \
                pipeline_without_waits(), mock.patch.object(client_mod, "decode_sam_output", api.decode), \
                contextlib.redirect_stderr(err), self.assertLogs("herogen.pipeline.client", "WARNING") as logs:
            client_mod.log.warning("test run starts")  # assertLogs needs one record even when nothing retries
            code = run_pipeline(self.sketch, run_id, list(stage_list), mode="live", runs=str(self.root))
        self.server, self.err, self.logs = server, err.getvalue(), logs.output
        self.assertFalse([line for line in logs.output if DUMMY_KEY in line or "earer" in line])
        self.assertTrue(all(h["auth_ok"] == "True" for _, h in server.requests))
        return code

    def _replay(self, run_id: str, stage_list=(0, 1, 2, 5)) -> int:
        env = {k: v for k, v in os.environ.items() if k != "MODEL_API_KEY"}
        err = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stderr(err):
            code = run_pipeline(self.sketch, run_id, list(stage_list), mode="replay", runs=str(self.root))
        self.err = err.getvalue()
        return code

    def _stage(self, n, run_id: str = "r") -> dict:
        return json.loads((self.root / run_id / f"stage-{n}.json").read_text())

    def _no_failure_cached(self) -> None:
        for p in (self.root / "_cache").glob("*.json"):
            self.assertNotIn("api_error", json.loads(p.read_text()), p.name)

    def test_sam_429_stops_stage1_before_any_paid_edit(self):
        self.assertEqual(self._live(FaultyAPI({"sketch": 429})), 1)
        s1 = self._stage(1)
        self.assertEqual((s1["ok"], s1["status"]), (False, "error"))
        self.assertIn("the Model API failed", s1["error"])
        self.assertIn("HTTP 429", s1["error"])
        self.assertIn("the Model API failed", self.err)
        self.assertEqual(self.server.count("images/edits"), 0)  # no cover made for an unmeasurable sketch
        self.assertEqual(self.server.count("responses"), retry.MAX_ATTEMPTS)
        self.assertFalse((self.root / "r" / "01-masks").exists())
        self._no_failure_cached()
        status = {s["stage"]: s["status"] for s in collect(RunFolder(self.root, "r"))["stages"]}
        self.assertEqual(status[1], "error")
        # The failure is not replayed as a result: the replay has nothing for that call and stops.
        self.assertEqual(self._replay("r2"), 1)
        self.assertIn("ReplayMiss", self.err)

    def test_sam_500_on_the_sheet_stops_stage2(self):
        self.assertEqual(self._live(FaultyAPI({"sheet": 500})), 1)
        self.assertTrue(self._stage(1)["ok"])
        s2 = self._stage(2)
        self.assertEqual((s2["ok"], s2["status"]), (False, "error"))
        self.assertIn("HTTP 500", s2["error"])
        self.assertEqual(s2["cost_usd"], client_mod.PRICE_PER_IMAGE_USD[client_mod.IMAGE_MODEL])  # the sheet
        self.assertFalse((self.root / "r" / "views").exists())
        self._no_failure_cached()

    def test_sam_500_on_one_prompt_still_stops(self):
        # Earlier versions moved on to the next prompt; a failed call is never read as "found nothing".
        api = FaultyAPI({"sketch": 500}, texts=(stages.SAM_TEXTS[0],))
        self.assertEqual(self._live(api), 1)
        self.assertEqual(self._stage(1)["status"], "error")
        self.assertEqual(api.sam_kinds, ["sketch"] * (retry.MAX_ATTEMPTS))

    def test_an_incomplete_sam_answer_is_not_nothing_found(self):
        # An incomplete answer carries no output, which would read as zero instances; only its status tells.
        api = FaultyAPI({"sketch": "incomplete"})
        self.assertEqual(self._live(api), 1)
        s1 = self._stage(1)
        self.assertEqual((s1["ok"], s1["status"]), (False, "error"))
        self.assertIn("status is 'incomplete'", s1["error"])
        self.assertIn("max_output_tokens", s1["error"])
        self.assertEqual(api.sam_kinds, ["sketch"])  # stopped at once, not taken as "try the next prompt"
        self.assertEqual(self.server.count("images/edits"), 0)
        self.assertEqual(list((self.root / "_cache").glob("*.json")), [])  # not a result, never cached
        self.assertEqual(s1["cost_usd"], client_mod.PRICE_PER_IMAGE_USD[client_mod.SAM_MODEL])  # answered: billed
        self.assertEqual(self._replay("r2"), 1)

    def test_a_failed_sam_answer_on_the_sheet_stops_stage2(self):
        self.assertEqual(self._live(FaultyAPI({"sheet": "failed"})), 1)
        s2 = self._stage(2)
        self.assertEqual(s2["status"], "error")
        self.assertIn(f"failed after {retry.MAX_ATTEMPTS} attempts: HTTP 200 with status 'failed'", s2["error"])
        self.assertIn("server_error", s2["error"])  # a transient failure: retried like a 500, then stopped

    def test_a_failed_muse_answer_stops_stage1(self):
        self.assertEqual(self._live(FaultyAPI({}, muse_faults={"cover": "failed"})), 1)
        s1 = self._stage(1)
        self.assertEqual(s1["status"], "error")
        self.assertIn(f"images/edits failed after {retry.MAX_ATTEMPTS} attempts: HTTP 200 with status 'failed'",
                      s1["error"])
        self.assertFalse((self.root / "r" / "01-cover.png").exists())

    def test_no_character_in_the_sketch_rejects_the_input(self):
        api = FaultyAPI({"sketch": "empty"})
        self.assertEqual(self._live(api), 3)
        s1 = self._stage(1)
        self.assertEqual((s1["ok"], s1["status"]), (False, "input_rejected"))
        self.assertIn("no character found in the sketch", s1["error"])
        self.assertIn("input rejected: no character found in the sketch", self.err)
        self.assertEqual(s1["details"]["mask_method"], {"sketch": stages.SAM_NOTHING_FOUND})
        self.assertEqual(api.sam_kinds, ["sketch"] * len(stages.SAM_TEXTS))  # every prompt tried
        self.assertEqual(self.server.count("images/edits"), 0)
        self.assertFalse((self.root / "r" / "stage-2.json").exists())
        status = {s["stage"]: s["status"] for s in collect(RunFolder(self.root, "r"))["stages"]}
        self.assertEqual(status[1], "input rejected")
        self.assertEqual(self._replay("r2"), 3)  # an empty result is a result, replayed the same

    def test_nothing_found_in_the_cover_fails_its_gate(self):
        self.assertEqual(self._live(FaultyAPI({"cover": "empty"})), 2)
        s1 = self._stage(1)
        self.assertFalse(s1["ok"])
        self.assertEqual(s1["details"]["mask_method"]["cover"], stages.SAM_NOTHING_FOUND)
        failed = [c["name"] for c in s1["checks"] if not c["pass"]]
        self.assertIn("cover_mask_area_frac", failed)

    def test_nothing_found_in_the_sheet_fails_every_attempt(self):
        self.assertEqual(self._live(FaultyAPI({"sheet": "empty"})), 2)
        s2 = self._stage(2)
        self.assertEqual(len(s2["details"]["attempts"]), 4)
        for a in s2["details"]["attempts"]:
            self.assertEqual(a["split_source"], stages.SAM_NOTHING_FOUND)
            self.assertEqual([(c["name"], c["value"]) for c in a["checks"]], [("figures_found", 0.0)])

    def test_a_figure_sam_half_missed_fails_the_coverage_gate(self):
        self.assertEqual(self._live(FaultyAPI({"sheet": "partial"})), 2)
        a = self._stage(2)["details"]["attempts"][0]
        cov = next(c for c in a["checks"] if c["name"] == "view_mask_sam_coverage_min")
        self.assertFalse(cov["pass"])
        self.assertLess(cov["value"], C.SAM_COVERAGE_MIN)
        self.assertEqual(min(a["sam_coverage"].values()), cov["value"])

    def test_nothing_found_in_the_drawn_top_view_uses_the_hull(self):
        self.assertEqual(self._live(FaultyAPI({"top": "empty"}), stage_list=(0, 1, 2)), 0)
        s2 = self._stage(2)
        self.assertEqual(s2["details"]["top_method"], "visual_hull")
        self.assertNotIn("top_drawn_width_vs_front_width", [c["name"] for c in s2["checks"]])

    def test_sam_500_on_a_heldout_view_stops_stage2h(self):
        self.assertEqual(self._live(FaultyAPI({"heldout": 500}), stage_list=(0, 1, 2)), 1)
        rec = self._stage("2h")
        self.assertEqual((rec["ok"], rec["status"]), (False, "error"))
        self.assertIn("HTTP 500", rec["error"])
        self.assertTrue(self._stage(2)["ok"])
        self._no_failure_cached()

    def test_nothing_found_in_a_heldout_view_rejects_it_unmeasured(self):
        api = FaultyAPI({"heldout": "empty"}, texts=stages.SAM_TEXTS)
        self.assertEqual(self._live(api, stage_list=(0, 1, 2)), 2)
        rec = self._stage("2h")
        self.assertEqual(rec["details"]["accepted_views"], [])
        for view in heldout.VIEWS:
            d = rec["details"]["views"][view.name]
            self.assertEqual(d["status"], "rejected")
            for a in d["attempts"]:
                self.assertEqual(a["mask_method"], stages.SAM_NOTHING_FOUND)
                self.assertEqual([c["name"] for c in a["checks"]], ["figure_area_frac", "figure_mask_sam_coverage"])
                self.assertEqual(a["measured"], {"mask_sam_coverage": 0.0})
        # Every figure prompt was tried, and no face or head call was made on a figure SAM did not find.
        self.assertEqual({t for k, t in api.sam_texts if k == "heldout"}, set(stages.SAM_TEXTS))


class LegacyRecordTest(unittest.TestCase):
    """Run folders and caches written by earlier versions, which fell back to threshold masks."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_cached_failure_replays_as_a_miss_not_an_api_error(self):
        from PIL import Image  # noqa: PLC0415

        img = Image.new("RGB", (16, 16), "white")
        png = client_mod.png_bytes(img)
        summary = {"endpoint": "responses", "model": client_mod.SAM_MODEL, "text": "character",
                   "image_sha256": client_mod._sha(png), "metadata": {"include_confidence": "true"}}
        key = client_mod.request_key(summary)
        cache = self.root / "cache"
        cache.mkdir()
        (cache / f"{key}.json").write_text(json.dumps({"request": summary, "api_error": "HTTP 429: busy"}))
        with self.assertRaises(ReplayMiss) as cm:
            ModelClient(cache).segment(img, "character")
        self.assertIn("only a failed call", str(cm.exception))
        with self.assertRaises(ReplayMiss):
            stages.sam_figure_mask(ModelClient(cache), img)

    def test_threshold_records_never_count_as_passed(self):
        run = RunFolder(self.root, "old")
        run.write_stage({"stage": 1, "name": "cover_art", "ok": True, "checks": [], "details": {
            "mask_method": {"sketch": "threshold (sam error: ApiError, ApiError)", "cover": "sam-3.1:character"}}})
        run.write_stage({"stage": 2, "name": "turnaround", "ok": True, "checks": [], "details": {
            "attempts": [{"attempt": 0, "split_source": "threshold"}],
            "mask_method_detail": {v: "sam-3.1:character" for v in stages.SIDE_VIEWS} | {"top": "visual_hull"}}})
        self.assertEqual(stages.degraded_masks(run.read_stage(1)), ["sketch: threshold (sam error: ApiError, ApiError)"])
        self.assertEqual(stages.degraded_masks(run.read_stage(2)), ["split: threshold"])
        status = {s["stage"]: (s["status"], s.get("ok")) for s in collect(run)["stages"]}
        self.assertEqual(status[1], ("degraded: threshold masks", False))
        self.assertEqual(status[2], ("degraded: threshold masks", False))
        with self.assertRaises(ValueError) as cm:
            heldout.stage2h_heldout(run, ModelClient(self.root / "cache"))
        self.assertIn("run stage 2 again", str(cm.exception))

    def test_new_records_are_not_degraded(self):
        rec = {"stage": 2, "ok": True, "details": {
            "attempts": [{"attempt": 0, "split_source": "sam-3.1:character"}],
            "mask_method_detail": {v: "sam-3.1:person" for v in stages.SIDE_VIEWS} | {"top": "visual_hull"}}}
        self.assertEqual(stages.degraded_masks(rec), [])
        self.assertEqual(stages.degraded_masks({"stage": 1, "details": {"mask_method": {
            "sketch": "sam-3.1:character", "cover": stages.SAM_NOTHING_FOUND}}}), [])


if __name__ == "__main__":
    unittest.main()
