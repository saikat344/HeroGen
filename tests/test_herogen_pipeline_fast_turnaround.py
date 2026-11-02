"""Fast turnaround recovery with synthetic sheets and a fake model transport."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
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

    def execute(self, sheets, *, attempts=9, instances=None):
        self.api = flow.SheetsAPI(sheets)
        def decode(text, size):
            if instances is not None and size == sheets[0].size and self.api.last_sam_text == "character":
                return instances
            return self.api.decode(text, size)
        def parts(*_):
            self.run.write_json("masks/parts.json", {})
            return {}
        with mock.patch.dict(os.environ, {"MODEL_API_KEY": flow.DUMMY_KEY}), \
                mock.patch.object(ModelClient, "_post", lambda c, e, b, **_: self.api.post(c, e, b)), \
                mock.patch.object(client_module, "decode_sam_output", decode), \
                mock.patch("herogen.pipeline.parts.write_part_masks", side_effect=parts), \
                mock.patch.object(self.client, "edit_image", wraps=self.client.edit_image) as edits:
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

    def test_unusable_first_sheet_recovers_with_same_prompt_and_new_sample(self):
        record = self.execute([self.bad, self.good])
        self.assertTrue(record["ok"])
        self.assertEqual(self.api.n, 2)
        calls = self.sheet_calls()
        self.assertEqual([c.kwargs["attempt"] for c in calls], [0, 1])
        self.assertEqual(calls[0].args, calls[1].args)
        self.assertEqual(calls[0].kwargs["size"], calls[1].kwargs["size"])
        self.assertEqual(self.api.prompts, [stages.TURNAROUND_PROMPT] * 2 + [stages.TOP_PROMPT])
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
        self.assertEqual(self.api.prompts, [stages.TURNAROUND_PROMPT] * 2)
        self.assertFalse(self.run.exists("views/meta.json"))

    def test_usable_sheet_is_requested_once_even_when_quality_fails(self):
        failed = {"name": stages.CARVE_GATE, "pass": False, "gate": True, "value": 0, "threshold": .9}
        with mock.patch.object(stages, "carve_check", return_value=(failed, {})):
            record = self.execute([self.good])
        self.assertTrue(record["ok"])
        self.assertEqual(len(self.sheet_calls()), 1)
        self.assertEqual(len(record["details"]["attempts"]), 1)
        self.assertEqual(self.api.prompts, [stages.TURNAROUND_PROMPT, stages.TOP_PROMPT])
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
                    calls.assert_called_once()
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

    def test_non_fast_recorded_call_sequence_and_outputs_unchanged(self):
        self.client = ModelClient(self.root / "cache", mode="live", fast=False)
        record = self.execute([self.bad, self.good], attempts=4)
        # Recorded from the original normal-mode flow, including request/cache identities.
        sequence = [(c["kind"], c["key"]) for c in record["calls"]]
        self.assertEqual(sequence, NORMAL_CALLS)
        stable = {k: v for k, v in record.items() if k not in ("started", "seconds", "calls")}
        stable["output_bytes"] = {p: hashlib.sha256((self.run.path / p).read_bytes()).hexdigest()
                                  for p in record["outputs"]}
        digest = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, NORMAL_DIGEST)


NORMAL_CALLS = [
    ("edit", "0867a0385b7feb0c63468c66"),
    ("segment", "9302b952f81373828870cd26"),
    ("edit", "735c9c6f449b82c89401d21b"),
    ("segment", "2dcb9fb7700a2af50b009578"),
    ("edit", "41740058fdaa10cf327b133a"),
    ("segment", "d38ec3d8617112edca5d1d5b"),
]
NORMAL_DIGEST = "05a7116567d2b3a588b02b83a66d6ad69859c682aefea708fa0038e820f7d058"
