"""Fast turnaround recovery with synthetic sheets and a fake model transport."""

from contextlib import nullcontext
import hashlib
import json
import os
import sys
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from herogen.pipeline import client as client_module, stages
from herogen.pipeline.client import ApiError, ModelClient, ReplayMiss, SamInstance
from herogen.pipeline.runfolder import RunFolder
import test_herogen_pipeline_flow as flow


MESSAGE = ("We could not draw clean character views from this sketch. "
           "Try again, or try a clearer full-body drawing.")


class FastTurnaroundTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = RunFolder(self.root, "hero")
        Image.new("RGB", (80, 80), "white").save(self.run.file("01-cover.png"))
        self.good = flow.sheet()[0]
        self.bad = flow.sheet(order=("front", "right", "front"))[0]
        self.client = ModelClient(self.root / "cache", mode="live", fast=True)
        self.requests = []

    def execute(self, sheets, *, attempts=9, instances=None, top=None, barrier=None, inspect_edits=True):
        self.api = flow.SheetsAPI(sheets)
        def decode(text, size):
            if instances is not None and size == sheets[0].size and self.api.last_sam_text == "character":
                return instances
            return self.api.decode(text, size)
        def parts(*_):
            self.run.write_json("masks/parts.json", {})
            return {}
        def post(client, endpoint, body, **_):
            self.requests.append([endpoint, body])
            if endpoint == "images/edits" and body["prompt"] in (stages.TOP_PROMPT, stages.TURNAROUND_PROMPT):
                if barrier is not None and body["prompt"] == stages.TOP_PROMPT:
                    barrier.wait(timeout=5)
                elif barrier is not None and self.api.n == 0:
                    barrier.wait(timeout=5)
                if top is not None and body["prompt"] == stages.TOP_PROMPT:
                    self.api.calls.append(endpoint)
                    self.api.prompts.append(body["prompt"])
                    return {"data": [{"b64_json": flow._b64png(top)}]}
            return self.api.post(client, endpoint, body)
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": flow.DUMMY_KEY}), \
                mock.patch.object(ModelClient, "_post", post), \
                mock.patch.object(client_module, "decode_sam_output", decode), \
                mock.patch("herogen.pipeline.parts.write_part_masks", side_effect=parts), \
                (mock.patch.object(self.client, "edit_image", wraps=self.client.edit_image)
                 if inspect_edits else nullcontext()) as edits:
            self.edits = edits
            return stages.stage2_turnaround(self.run, self.client, attempts=attempts)

    def sheet_calls(self):
        return [c for c in self.edits.call_args_list if c.args[0] == stages.TURNAROUND_PROMPT]

    def overlapping_sheet(self, count=4):
        # The tails and bodies overlap in x but have disjoint pixels, with four equal-height figures.
        masks = []
        rgb = np.full((200, 600, 3), 255, np.uint8)
        for i in range(count):
            x = 80 + 120 * i
            mask = np.zeros(rgb.shape[:2], bool)
            mask[10:190, x:x+35] = True
            mask[30+i*30:35+i*30, x-70:x+35] = True
            mask[150+i*8:155+i*8, x+35:x+105] = True
            rgb[mask] = (40 + i*40, 60, 100)
            masks.append(SamInstance(mask, flow.imaging.bbox(mask), .9))
        return Image.fromarray(rgb), masks

    def test_overlapping_columns_use_four_isolated_instances_in_sheet_order(self):
        sheet, instances = self.overlapping_sheet()
        self.assertEqual(len(flow.imaging.split_columns(np.logical_or.reduce([i.mask for i in instances]))), 1)
        # Segmentation order is arbitrary; view order must come from the box centres.
        record = self.execute([sheet], instances=instances[::-1])
        self.assertTrue(record["ok"])
        self.assertEqual(len(self.sheet_calls()), 1)
        details = record["details"]
        attempt = details["attempts"][0]
        self.assertEqual(attempt["split_source"], "sam-3.1:instances")
        self.assertEqual(len(attempt["column_runs"]), 1)
        self.assertEqual(len(attempt["instance_split"]["boxes"]), 4)
        self.assertEqual(attempt["assignment"]["panel_for_view"],
                         dict(zip(stages.SIDE_VIEWS, [f"panel{i}" for i in range(4)])))
        for i, name in enumerate(stages.SIDE_VIEWS):
            with Image.open(self.run.path / f"views/{name}.png") as image:
                rgb = np.asarray(image)
                own = (40 + i*40, 60, 100)
                foreground = np.any(rgb != (255, 255, 255), axis=2)
                self.assertTrue(foreground.any())
                self.assertTrue(np.all(rgb[foreground] == own))
            with Image.open(self.run.path / f"masks/{name}.png") as image:
                self.assertEqual(np.count_nonzero(np.asarray(image)), int(instances[i].mask.sum()))
        self.assertEqual(set(details["mask_method_detail"][name] for name in stages.SIDE_VIEWS),
                         {"sam-3.1:instances"})

    def test_three_overlapping_instances_remain_unusable(self):
        sheet, instances = self.overlapping_sheet(3)
        with self.assertRaisesRegex(ValueError, "We could not draw clean character views"):
            self.execute([sheet, sheet], instances=instances)
        attempts = self.run.read_stage(2)["details"]["attempts"]
        self.assertEqual([a["extracted"] for a in attempts], [False, False])
        self.assertTrue(all(a["split_source"] == "sam-3.1:character" for a in attempts))
        self.assertEqual(len(self.sheet_calls()), 2)

    def test_instance_fallback_rejects_parts_and_duplicate_centres(self):
        sheet, figures = self.overlapping_sheet()
        columns = {"runs": [(5, 495)], "views": {}}
        for fault in ("short", "small", "duplicate"):
            with self.subTest(fault=fault):
                instances = list(figures)
                mask = figures[-1].mask.copy()
                if fault == "short":
                    mask[:50] = False
                elif fault == "small":
                    mask[:, :] = False
                    mask[10:190, 390:392] = True
                else:
                    mask = figures[0].mask.copy()
                instances[-1] = SamInstance(mask, flow.imaging.bbox(mask), .9)
                out = stages._fast_instance_split(sheet, instances, flow.imaging.threshold_mask(sheet), columns)
                self.assertIs(out, columns)

    def test_instance_fallback_uses_largest_four_and_ignores_small_extra(self):
        sheet, figures = self.overlapping_sheet()
        noise = np.zeros((sheet.height, sheet.width), bool)
        noise[1, 1] = True
        record = self.execute([sheet], instances=[SamInstance(noise, (1, 1, 2, 2)), *figures])
        attempt = record["details"]["attempts"][0]
        self.assertTrue(attempt["extracted"])
        self.assertEqual(attempt["sam_instances"], 5)
        self.assertEqual(attempt["split_source"], "sam-3.1:instances")

    def test_column_split_stays_first_with_four_separate_figures(self):
        with mock.patch.object(stages, "_fast_instance_split", side_effect=AssertionError("unexpected fallback")):
            record = self.execute([self.good])
        self.assertTrue(record["ok"])
        self.assertEqual(record["details"]["attempts"][0]["split_source"], "sam-3.1:character")

    def test_non_fast_overlapping_columns_do_not_use_instances(self):
        sheet, instances = self.overlapping_sheet()
        self.client = ModelClient(self.root / "full-overlap", mode="live", fast=False)
        with mock.patch.object(stages, "_fast_instance_split", side_effect=AssertionError("unexpected fallback")), \
                self.assertRaises(stages.GateFailed):
            self.execute([sheet], instances=instances, attempts=1)
        record = self.run.read_stage(2)
        self.assertFalse(record["ok"])
        self.assertEqual(record["details"]["attempts"][0]["split_source"], "sam-3.1:character")
        self.assertFalse(self.run.exists("views/front.png"))

    def drafted(self, sheets):
        """Run stage 2, noting each draft as it is written: (path, mode, files in drafts/ then)."""
        seen, save = [], stages.save_draft
        def spy(run, rel, img):
            save(run, rel, img)
            seen.append((rel, img.mode, sorted(p.relative_to(Path(run) / "drafts").as_posix()
                                               for p in (Path(run) / "drafts").rglob("*.png"))))
        with mock.patch.object(stages, "save_draft", spy):
            try:
                return self.execute(sheets), seen
            except ValueError:
                return None, seen

    def test_drafts_show_each_sheet_and_its_cut_out_views_before_the_stage_ends(self):
        record, seen = self.drafted([self.bad, self.good])
        self.assertTrue(record["ok"])
        rels = [rel for rel, _, _ in seen]
        self.assertEqual(rels.count("02-turnaround.png"), 2, "the recovery sheet replaces the first draft")
        self.assertIn("02-top-drawn.png", rels)
        self.assertEqual([r for r in rels if r.startswith("views/")], [f"views/{n}.png" for n in stages.ALL_VIEWS])
        self.assertLess(rels.index("02-turnaround.png"), rels.index("views/front.png"))
        self.assertEqual({mode for rel, mode, _ in seen if rel.startswith("views/")}, {"RGBA"})
        self.assertIn("views/front.png", seen[rels.index("views/left.png")][2], "drafts are on disk while it runs")
        self.assertFalse((self.run.path / "drafts").exists(), "the stage clears its drafts when it ends")
        self.assertEqual(sorted(record["outputs"]), sorted(o for o in record["outputs"] if not o.startswith("drafts")))

    def test_a_failed_sheet_leaves_no_drafts(self):
        (self.run.path / "drafts").mkdir()
        (self.run.path / "drafts" / "02-turnaround.png").write_bytes(b"stale")
        record, seen = self.drafted([self.bad, self.bad])
        self.assertIsNone(record)
        self.assertEqual([rel for rel, _, _ in seen if rel == "02-turnaround.png"], ["02-turnaround.png"] * 2)
        self.assertFalse((self.run.path / "drafts").exists())

    def test_unusable_first_sheet_recovers_with_same_prompt_and_new_sample(self):
        record = self.execute([self.bad, self.good])
        self.assertTrue(record["ok"])
        self.assertEqual(self.api.n, 2)
        calls = self.sheet_calls()
        self.assertEqual([c.kwargs["attempt"] for c in calls], [0, 1])
        self.assertEqual(calls[0].args, calls[1].args)
        self.assertEqual(calls[0].kwargs["size"], calls[1].kwargs["size"])
        self.assertCountEqual(self.api.prompts, [stages.TURNAROUND_PROMPT] * 2 + [stages.TOP_PROMPT])
        details = self.run.read_stage(2)["details"]
        self.assertEqual([a["extracted"] for a in details["attempts"]], [False, True])
        self.assertEqual(details["kept_attempt"], 1)
        with Image.open(self.run.path / "02-turnaround.png") as saved:
            self.assertTrue(np.array_equal(saved, self.good))
        for name in stages.ALL_VIEWS:
            self.assertTrue(self.run.exists(f"views/{name}.png"))

    def test_two_unusable_sheets_stop_with_plain_message_and_both_attempts(self):
        with self.assertRaisesRegex(ValueError, "^" + MESSAGE.replace(".", r"\.") + "$"):
            self.execute([Image.new("RGB", self.bad.size, "white"), self.bad])
        self.assertEqual([c.kwargs["attempt"] for c in self.sheet_calls()], [0, 1])
        record = self.run.read_stage(2)
        self.assertEqual((record["ok"], record["status"], record["error"]), (False, "error", MESSAGE))
        self.assertEqual([a["extracted"] for a in record["details"]["attempts"]], [False, False])
        self.assertEqual([a["checks"][0]["value"] for a in record["details"]["attempts"]], [0, 3])
        self.assertCountEqual(self.api.prompts, [stages.TURNAROUND_PROMPT] * 2 + [stages.TOP_PROMPT])
        self.assertFalse(self.run.exists("views/meta.json"))

    def test_usable_sheet_is_requested_once_even_when_quality_fails(self):
        failed = {"name": stages.CARVE_GATE, "pass": False, "gate": True, "value": 0, "threshold": .9}
        with mock.patch.object(stages, "carve_check", return_value=(failed, {})):
            record = self.execute([self.good])
        self.assertTrue(record["ok"])
        self.assertEqual(len(self.sheet_calls()), 1)
        self.assertEqual(len(record["details"]["attempts"]), 1)
        self.assertCountEqual(self.api.prompts, [stages.TURNAROUND_PROMPT, stages.TOP_PROMPT])
        self.assertTrue(any(not c["pass"] for c in record["checks"]))
        self.assertTrue(all(not c["gate"] for c in record["checks"]))

    def test_local_extraction_error_allows_one_recovery(self):
        original = stages.square_views
        count = 0
        def square(*args):
            nonlocal count
            count += 1
            if count == 1:
                raise ValueError("unusable panel geometry")
            return original(*args)
        with mock.patch.object(stages, "square_views", side_effect=square):
            record = self.execute([self.good, self.good])
        self.assertTrue(record["ok"])
        self.assertEqual(self.api.n, 2)
        self.assertEqual(record["details"]["attempts"][0]["extraction_error"], "unusable panel geometry")

    def test_missing_or_empty_panel_allows_one_recovery(self):
        original = stages.split_sheet
        for fault in ("missing", "empty"):
            with self.subTest(fault=fault):
                count = 0
                def split(*args):
                    nonlocal count
                    result = original(*args)
                    count += 1
                    if count == 1:
                        if fault == "missing":
                            result["views"].pop("panel3")
                        else:
                            result["views"]["panel3"]["mask"][:] = False
                    return result
                with mock.patch.object(stages, "split_sheet", side_effect=split):
                    record = self.execute([self.good, self.good])
                self.assertTrue(record["ok"])
                self.assertEqual([c.kwargs["attempt"] for c in self.sheet_calls()], [0, 1])
                self.assertEqual([a["extracted"] for a in record["details"]["attempts"]], [False, True])

    def test_model_failures_never_trigger_sheet_recovery(self):
        for operation in ("edit_image", "segment"):
            for error in (ApiError("HTTP 503"), ReplayMiss("missing response")):
                with self.subTest(operation=operation, error=type(error).__name__), \
                        mock.patch.object(self.client, operation, side_effect=error) as calls:
                    with self.assertRaises(type(error)):
                        self.execute([self.good])
                    self.assertEqual(calls.call_count, 2 if operation == "edit_image" else 1)
                    self.assertEqual(len(self.sheet_calls()), 1)
                    self.assertIn(str(error), self.run.read_stage(2)["error"])

    def test_model_failure_on_recovery_preserves_first_attempt(self):
        original = self.client.edit_image
        def edit(*args, **kwargs):
            if kwargs.get("attempt") == 1:
                raise ReplayMiss("missing recovery sample")
            return original(*args, **kwargs)
        with mock.patch.object(self.client, "edit_image", side_effect=edit):
            with self.assertRaisesRegex(ReplayMiss, "missing recovery sample"):
                self.execute([self.bad])
        record = self.run.read_stage(2)
        self.assertEqual([a["attempt"] for a in record["details"]["attempts"]], [0, 1])
        self.assertFalse(record["details"]["attempts"][0]["extracted"])
        self.assertIn("the model client failed: missing recovery sample", record["error"])

    def test_top_request_overlaps_sheet_and_is_not_repeated_for_recovery(self):
        record = self.execute([self.bad, self.good], barrier=threading.Barrier(2), inspect_edits=False)
        self.assertTrue(record["ok"])
        self.assertEqual(self.api.prompts.count(stages.TOP_PROMPT), 1)
        self.assertEqual(self.api.prompts.count(stages.TURNAROUND_PROMPT), 2)
        images = [c for c in record["calls"] if c["kind"] == "edit"]
        self.assertCountEqual([c["purpose"] for c in images], ["top view", "turnaround", "turnaround"])

    def test_empty_top_saves_four_views_metadata_parts_and_skips_top_in_stage3(self):
        from herogen.blender import stage3, stage4, texture, joints
        from herogen.blender.views import load_views
        # A previous successful top must not leak into an explicit Retry.
        for rel in ("views/top.png", "masks/top.png"):
            Image.new("L", (32, 32), 255).save(self.run.file(rel))
        blank = Image.new("RGB", (300, 280), "white")
        record = self.execute([self.good], top=blank)
        self.assertTrue(record["ok"])
        self.assertEqual(self.api.prompts.count(stages.TOP_PROMPT), 1)
        self.assertEqual(len(self.sheet_calls()), 1)
        self.assertEqual(record["details"]["top_method"], "missing")
        self.assertEqual(record["details"]["missing_views"], ["top"])
        self.assertEqual(record["details"]["top_drawn_mask_method"], stages.SAM_NOTHING_FOUND)
        self.assertIn("No usable character mask", record["details"]["top_method_detail"])
        self.assertIn("Continuing with the four side views", record["notes"])
        self.assertTrue(self.run.exists("02-top-drawn.png"))
        self.assertTrue(self.run.exists("masks/parts.json"))
        self.assertFalse(self.run.exists("views/top.png"))
        self.assertFalse(self.run.exists("masks/top.png"))
        meta = self.run.read_json("views/meta.json")
        self.assertEqual(meta["views"], list(stages.SIDE_VIEWS))
        self.assertEqual(meta["missing_views"], ["top"])
        self.assertEqual(set(meta["mask_method"]), set(stages.SIDE_VIEWS))
        def render(job, *_, **kwargs):
            for camera in job["tasks"][0]["cameras"]:
                name = camera["view"]
                self.assertNotEqual(name, "top")
                Image.new("L", (meta["image_size"],) * 2).save(
                    self.run.path / "03-renders" / f"{name}.png")
            return {"blender": "synthetic"}
        mesh = (np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1]], float), np.array([[0, 1, 2]]))
        info = {"detached_side_prior_voxels": 0, "arm_depth_prior_columns": 0}
        with mock.patch.object(stage3.carve, "build_mesh", return_value=(*mesh, info)), \
                mock.patch.object(stage3.bl, "run_job", side_effect=render):
            s3 = stage3.run_stage3(self.run.path, mesh_source="carve", joints_source="silhouette", fast=True)
        self.assertTrue(s3["ok"])
        self.assertIn("missing views skipped: top", s3["notes"])
        # Stage 4's provenance, joints, colour inputs and cameras use only the available views.
        vs = load_views(self.run.path, stages.HEIGHT_UNITS)
        source, shape_views, check = stage4._shape_views(self.run.path, vs)
        self.assertEqual(source, "carve")
        self.assertEqual(shape_views, set(stages.SIDE_VIEWS))
        self.assertTrue(check["pass"])
        self.assertIsNotNone(joints.estimate(vs, stages.HEIGHT_UNITS))
        self.assertEqual(set(texture.observed_colour_views(vs, self.run.path)), set(stages.SIDE_VIEWS))
        self.assertEqual({v.camera()["view"] for v in vs.views.values()}, set(stages.SIDE_VIEWS))

    def test_top_model_errors_stay_loud_without_recovery(self):
        original = self.client.edit_image
        for error in (ApiError("top failed"), ReplayMiss("top missing")):
            def edit(prompt, *args, **kwargs):
                if prompt == stages.TOP_PROMPT:
                    raise error
                return original(prompt, *args, **kwargs)
            with self.subTest(error=type(error).__name__), \
                    mock.patch.object(self.client, "edit_image", side_effect=edit):
                with self.assertRaises(type(error)):
                    self.execute([self.good])
                self.assertEqual(len(self.sheet_calls()), 1)
                self.assertIn(str(error), self.run.read_stage(2)["error"])

    def test_top_fit_keeps_usable_pixels_and_scales_down_only_when_needed(self):
        for width, height in ((60, 40), (20, 180)):
            with self.subTest(width=width, height=height):
                mask = np.zeros((200, 200), bool)
                mask[10:10+height, 10:10+width] = True
                raw = flow.paint(mask)
                front = np.zeros((100, 100), bool)
                front[10:90, 10:90] = True
                sq = {"front": {"mask": front}}
                inst = [SamInstance(mask, flow.imaging.bbox(mask), .9)]
                with mock.patch.object(self.client, "segment", return_value=inst):
                    top = stages.model_top_view(self.client, raw, sq, 100, (255, 255, 255), raw=raw)
                expected_scale = min(80 / width, 100 / height)
                self.assertEqual(top["scale"], round(expected_scale, 4))
                self.assertEqual(flow.imaging.box_size(flow.imaging.bbox(top["mask"])),
                                 (round(width * expected_scale), round(height * expected_scale)))
                if height == 40:
                    self.client.fast = False
                    with mock.patch.object(self.client, "edit_image", return_value=raw), \
                            mock.patch.object(self.client, "segment", return_value=inst):
                        full = stages.model_top_view(self.client, raw, sq, 100, (255, 255, 255))
                    np.testing.assert_array_equal(top["mask"], full["mask"])
                    np.testing.assert_array_equal(top["image"], full["image"])
                    self.client.fast = True

    def test_non_fast_recorded_call_sequence_and_outputs_unchanged(self):
        self.client = ModelClient(self.root / "cache", mode="live", fast=False)
        record = self.execute([self.bad, self.good], attempts=4)
        stable = {k: v for k, v in record.items() if k not in ("started", "seconds", "calls")}
        if ENCODED_PINS:
            # Recorded on macOS from the original normal-mode flow; request keys and output bytes hash
            # encoded PNGs, which depend on the platform's PNG encoder.
            self.assertEqual([(c["kind"], c["key"]) for c in record["calls"]], NORMAL_CALLS)
            encoded = {**stable, "output_bytes": {p: hashlib.sha256((self.run.path / p).read_bytes()).hexdigest()
                                                  for p in record["outputs"]}}
            self.assertEqual(hashlib.sha256(json.dumps(encoded, sort_keys=True).encode()).hexdigest(), NORMAL_DIGEST)
        # Platform-independent: request content with decoded pixels, call kinds, the record and decoded outputs.
        # Derived on macOS where the encoded pins above also hold, so it pins the same original flow.
        content = flow.stable_content({"requests": self.requests, "calls": [c["kind"] for c in record["calls"]],
                                       "record": stable})
        content["outputs"] = {p: flow.file_digest(self.run.path / p) for p in record["outputs"]}
        digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, NORMAL_CONTENT_DIGEST)


# Request keys and output bytes hash encoded PNGs; these pins were recorded with the macOS encoder. The two
# sheet request keys and both digests include the default sheet prompt (turnaround-v3); output bytes do not.
ENCODED_PINS = sys.platform == "darwin"
NORMAL_CALLS = [
    ("edit", "804398e827dc012db9f1b940"),
    ("segment", "9302b952f81373828870cd26"),
    ("edit", "04cf7d09a3b79d7201325e73"),
    ("segment", "2dcb9fb7700a2af50b009578"),
    ("edit", "41740058fdaa10cf327b133a"),
    ("segment", "d38ec3d8617112edca5d1d5b"),
]
NORMAL_DIGEST = "3d18075af0971c21d49463a2831e72fe5bb755d988c68b7c28588b5390c719e1"
NORMAL_CONTENT_DIGEST = "02e621aaff89b56ecad019e35bfd23e52bb462bfd74ea37163b8926dd5f239fe"


def _profile(toes: int) -> dict:
    """A side-view figure whose toes point image right (+1) or image left (-1)."""
    mask = np.zeros((100, 100), bool)
    mask[0:97, 45:55] = True  # body and shin
    mask[97:100, 45:80] = True  # sole reaching forward
    if toes < 0:
        mask = np.fliplr(mask)
    image = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).convert("RGB")
    return {"image": image, "mask": mask}


class FastOrientSidesTests(unittest.TestCase):
    def views(self, right: int, left: int) -> tuple[dict, dict]:
        sq = {"right": _profile(right), "left": _profile(left)}
        return sq, {"panel_for_view": {"right": "panel1", "left": "panel3"}}

    def test_correct_sides_are_left_as_drawn(self):
        sq, assignment = self.views(1, -1)
        before = {n: v["mask"].copy() for n, v in sq.items()}
        _, notes, fixes = stages.fast_orient_sides(sq, assignment)
        self.assertEqual((notes, fixes), ([], {}))
        for n in sq:
            np.testing.assert_array_equal(sq[n]["mask"], before[n])

    def test_exchanged_sides_are_swapped_not_mirrored(self):
        sq, assignment = self.views(-1, 1)
        right_drawing, left_drawing = sq["left"]["image"], sq["right"]["image"]
        _, notes, fixes = stages.fast_orient_sides(sq, assignment)
        self.assertEqual(fixes, {"swapped_sides": True})
        self.assertIs(sq["right"]["image"], right_drawing)
        self.assertIs(sq["left"]["image"], left_drawing)
        self.assertEqual(assignment["panel_for_view"], {"right": "panel3", "left": "panel1"})
        self.assertIn("exchanged", notes[0])

    def test_two_views_facing_one_way_mirror_the_wrong_one_as_the_full_lane_does(self):
        sq, assignment = self.views(1, 1)
        _, notes, fixes = stages.fast_orient_sides(sq, assignment)
        self.assertEqual(fixes, {"mirrored_left": True})
        self.assertLess(stages.imaging.facing_offset(sq["left"]["mask"]), 0)
        self.assertGreater(stages.imaging.facing_offset(sq["right"]["mask"]), 0)
        self.assertIn("left view faced the wrong way", notes[0])
