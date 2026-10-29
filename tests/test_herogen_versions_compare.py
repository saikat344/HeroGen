"""Comparing versions: per-view scores with seen and held-out views apart, across the hand-collected
`05-scores-<series>.json` and the scorer's `05-scores.json`, with missing numbers named per side."""
from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.versions import compare as C  # noqa: E402
from herogen.versions import store as S  # noqa: E402

from test_herogen_versions import (POSES, VIEWS, TempStore, check, gaps, hand_frozen,  # noqa: E402
                                   make_run, scores_entry, sha, write_json)


def old_scores(store: Path, sketch="sk", series="v0", bodies=None) -> Path:
    """A scores file in the hand-collected layout: bodies and their views, no scorer stamp, no views list."""
    path = store / sketch / f"05-scores-{series}.json"
    write_json(path, {"method": "by hand", "yaw_range_deg": [20.0, 70.0, 1.0],
                      "heldout": {"image": "heldout/front-left-34.png", "present": True},
                      "bodies": bodies if bodies is not None else [
                          scores_entry("carve", "04-body-carve/high.glb", "carve", base=0.85, triangles=23000),
                          scores_entry("hybrid", "04-body/high.glb", "sam3d-objects", base=0.8),
                          scores_entry("hybrid-variant", "04-body/high.glb", "sam3d-objects", base=0.7,
                                       variant_of="hybrid")]})
    return path


class Compare(TempStore):
    def two_new(self, edit_b=None, **b_kw) -> tuple[Path, Path]:
        """Two versions of one sketch; `edit_b(run)` changes B's run folder before it is frozen."""
        a = Path(self.freeze(make_run(self.runs, "a"), label="m1-a")["path"])
        b_run = make_run(self.runs, "b", **{"base": 0.92, "gap_scale": 0.5, "triangles": 22000, **b_kw})
        if edit_b:
            edit_b(b_run)
        b = Path(self.freeze(b_run, label="m1-b")["path"])
        return a, b

    @staticmethod
    def edit_json(path: Path, change) -> None:
        data = json.loads(path.read_text())
        change(data)
        write_json(path, data)

    def test_new_against_new(self):
        a, b = self.two_new(poses=POSES + ("crouch",))
        r = C.compare(a, b)
        self.assertEqual((r["a"]["label"], r["b"]["label"]), ("m1-a", "m1-b"))
        self.assertEqual([v["view"] for v in r["views"]], [*VIEWS, "front-left-34"])
        front = r["views"][0]
        self.assertFalse(front["heldout"])
        self.assertEqual(front["iou"], {"a": 0.9, "b": 0.92, "delta": 0.02})
        self.assertEqual(front["seen_a"], "shape+colour")
        held = r["views"][-1]
        self.assertTrue(held["heldout"])
        self.assertEqual(held["seen_a"], "unseen")
        self.assertEqual(held["iou"]["delta"], 0.02)
        self.assertEqual(r["means"]["seen"]["iou"]["views"], list(VIEWS))
        self.assertAlmostEqual(r["means"]["seen"]["iou"]["a"], 0.915)
        self.assertEqual(r["means"]["heldout"]["ssim"]["delta"], 0.0)
        self.assertEqual(r["triangles"]["high"], {"a": 20000, "b": 22000, "delta": 2000})
        self.assertEqual(r["triangles"]["low"]["delta"], 0)
        self.assertEqual(r["gates"]["stages"], {"3": {"a": True, "b": True}, "4": {"a": True, "b": True}})
        names = [c["name"] for c in r["gates"]["checks"]]
        self.assertIn("max_joint_gap_high", names)
        self.assertNotIn("rest_iou_with_covers_top", names)             # not a gate on either side
        g = r["joint_gaps"]["gaps_high"]
        self.assertEqual(g["common_poses"], ["rest", "walk_a", "arms_up"])  # twist_turn is ungated
        self.assertEqual((g["poses_a"], g["poses_b"], g["only_b"]), (3, 4, ["crouch"]))
        self.assertEqual(g["joints"]["LeftHip"], {"a": 0.09, "b": 0.045, "delta": -0.045})
        self.assertEqual(g["worst_common"]["a"], 0.09)
        self.assertEqual(r["inputs"]["sketch"], "same")
        self.assertEqual(set(r["inputs"]["views"].values()), {"same"})
        self.assertEqual(r["missing"], [])
        self.assertEqual((r["a"]["scores_file"], r["a"]["scores_body"]), ("05-scores.json", "carve"))
        self.assertTrue(r["a"]["scores_stamped"])
        text = C.render_text(r)
        self.assertIn("held-out (unseen)", text)
        self.assertIn("front-left-34", text)
        self.assertIn("Missing: nothing", text)
        json.loads(C.to_json(r))

    def test_hand_collected_scores_against_the_scorers(self):
        old_scores(self.store)
        a = hand_frozen(self.store, label="v0-carve", mesh="carve")
        b = Path(self.freeze(make_run(self.runs, "b", base=0.9), label="m1-carve")["path"])
        r = C.compare(a, b)
        self.assertEqual(r["a"]["format"], 0)
        self.assertEqual((r["a"]["scores_file"], r["a"]["scores_body"]), ("sk/05-scores-v0.json", "carve"))
        self.assertFalse(r["a"]["scores_stamped"])
        self.assertEqual(r["views"][0]["iou"], {"a": 0.85, "b": 0.9, "delta": 0.05})
        self.assertTrue(r["views"][-1]["heldout"])                     # by name: no views list in old scores
        self.assertEqual(r["triangles"]["high"]["a"], 23000)
        self.assertEqual(r["joint_gaps"]["gaps_high"]["joints"]["Neck"]["a"], 0.06)
        self.assertEqual(r["inputs"], {"recorded_a": False, "recorded_b": True, "sketch": "unknown"})
        self.assertEqual(r["a"]["joints_source"], "silhouette")       # format 0 calls it `joints`
        self.assertIn("missing in v0-carve", "; ".join(r["missing"]))
        text = C.render_text(r)
        self.assertIn("inputs: not recorded in A", text)
        self.assertIn("not stamped: an unstamped build", text)
        # the hybrid hand-frozen version finds its own entry by mesh source, variants aside
        h = hand_frozen(self.store, label="v0-hybrid", mesh="sam3d-objects")
        self.assertEqual(C.compare(h, b)["a"]["scores_body"], "hybrid")

    def test_a_metric_missing_on_one_side_is_named(self):
        def change(sc):
            body = sc["bodies"][0]
            body["views"]["front"]["ssim"] = None
            del body["views"]["front-left-34"]
            body["views"]["front-right-30"] = body["views"]["left"] | {"seen": False, "seen_shape": False,
                                                                      "seen_colour": False}

        a, b = self.two_new(lambda run: self.edit_json(run / "05-scores.json", change))
        r = C.compare(a, b)
        self.assertIn("front ssim: missing in m1-b", r["missing"])
        self.assertIn("front-left-34: not scored in m1-b", r["missing"])
        self.assertIn("front-right-30: not scored in m1-a", r["missing"])
        row = {v["view"]: v for v in r["views"]}
        self.assertEqual(row["front"]["ssim"], {"a": 0.8, "b": None, "delta": None})
        self.assertTrue(row["front-right-30"]["heldout"])
        self.assertEqual(r["means"]["seen"]["ssim"]["views"], ["back", "left", "right"])
        self.assertNotIn("iou", r["means"]["heldout"])                 # no held-out view on both sides
        text = C.render_text(r)
        self.assertIn("missing", text)

    def test_a_lod_the_profile_does_not_build_is_not_called_missing(self):
        a, b = self.two_new(profile="best", lods=("high",))
        r = C.compare(a, b)
        self.assertEqual(r["triangles"]["low"]["why_b"], "not built")
        self.assertEqual(r["joint_gaps"]["gaps_low"], {"absent_in": "m1-b", "why": "not built"})
        low = next(c for c in r["gates"]["checks"] if c["name"] == "triangles_low")
        self.assertEqual(low["why_b"], "not built")
        self.assertEqual(r["missing"], [])
        self.assertIn("not built", C.render_text(r))

    def test_ambiguous_or_absent_scores_are_never_guessed(self):
        old_scores(self.store, bodies=[scores_entry("x", "a/high.glb", "carve"),
                                       scores_entry("y", "b/high.glb", "carve")])
        a = hand_frozen(self.store, label="v0-carve", mesh="carve")
        b = hand_frozen(self.store, label="v1-carve", mesh="carve")
        r = C.compare(a, b)
        self.assertIsNone(r["a"]["scores_body"])
        self.assertIn("2 entries that could be this body", r["missing"][0])
        self.assertIn("no 05-scores.json in the version and no 05-scores-v1.json", r["missing"][1])
        self.assertEqual(r["views"], [])
        self.assertIn("no view scores on either side", C.render_text(r))

    def test_different_inputs_are_flagged(self):
        a = Path(self.freeze(make_run(self.runs, "a"), label="m1-a")["path"])
        run = make_run(self.runs, "b")
        (run / "views" / "front.png").write_bytes(b"another front")
        b = Path(self.freeze(run, label="m1-b")["path"])
        r = C.compare(a, b)
        self.assertEqual(r["inputs"]["views"]["front"], "different")
        self.assertIn("not the same: view front", C.render_text(r))

    def test_gates_and_joints_on_one_side_only_are_missing(self):
        def change(run):
            def st4(r):
                r["checks"] = [c for c in r["checks"] if c["name"] != "max_joint_gap_high"]
                r["checks"].append(check("new_gate", 1.0, 2.0, True, rule="<="))

            def sections(r):
                for pose in r["pose_test"]["gaps_high"].values():
                    pose.pop("Neck")

            self.edit_json(run / "stage-4.json", st4)
            self.edit_json(run / "04-body" / "sections.json", sections)

        a, b = self.two_new(change)
        r = C.compare(a, b)
        self.assertIn("gate 4 max_joint_gap_high: missing in m1-b", r["missing"])
        self.assertIn("gate 4 new_gate: missing in m1-a", r["missing"])
        self.assertIn("joint gaps gaps_high Neck: missing in m1-b", r["missing"])
        self.assertEqual(r["joint_gaps"]["gaps_high"]["joints"]["Neck"]["b"], None)

    def test_the_scorers_views_list_overrides_the_name_rule(self):
        def change(sc):
            sc["views"].append({"view": "extra-cam", "heldout": False})
            sc["bodies"][0]["views"]["extra-cam"] = sc["bodies"][0]["views"]["front"]

        a, b = self.two_new(lambda run: self.edit_json(run / "05-scores.json", change))
        rows = {v["view"]: v for v in C.compare(a, b)["views"]}
        self.assertFalse(rows["extra-cam"]["heldout"])                 # a name outside the five, yet seen
        self.assertTrue(rows["front-left-34"]["heldout"])

    def test_triangles_come_from_stage_4_then_the_scores_then_sections(self):
        def sections_lods(run, high):
            self.edit_json(run / "04-body" / "sections.json",
                           lambda r: r.update(lods={"high": {"triangles": high}, "low": {"triangles": 1}}))

        def scores_high(run, high):
            self.edit_json(run / "05-scores.json", lambda r: r["bodies"][0].update(triangles=high))

        def no_stage4_lods(run):
            self.edit_json(run / "stage-4.json", lambda r: r.pop("body"))

        def no_scores_triangles(run):
            self.edit_json(run / "05-scores.json", lambda r: r["bodies"][0].update(triangles=None))

        cases = [((), 22000), ((no_stage4_lods,), 33333), ((no_stage4_lods, no_scores_triangles), 44444)]
        for i, (edits, want) in enumerate(cases):
            run = make_run(self.runs, f"t{i}", triangles=22000)
            sections_lods(run, 44444)
            scores_high(run, 33333)
            for e in edits:
                e(run)
            v = Path(self.freeze(run, label=f"m1-t{i}")["path"])
            self.assertEqual(C._triangles(C._side(v), "high"), want, edits)

    def test_a_file_not_listed_in_the_version_is_never_read(self):
        a, _ = self.two_new()
        meta = S.load_version(a)
        self.assertIsNotNone(C._record(a, meta, "stage-4.json"))
        del meta["files_sha256"]["stage-4.json"]
        self.assertIsNone(C._record(a, meta, "stage-4.json"))

    def test_a_format_1_version_never_borrows_the_hand_collected_scores(self):
        old_scores(self.store)                                         # 05-scores-v0.json beside it
        run = make_run(self.runs, "b", scores=False)
        rebuilt = Path(self.freeze(run, label="v0-rebuild")["path"])
        a = hand_frozen(self.store, label="v0-carve", mesh="carve")
        r = C.compare(a, rebuilt)
        self.assertEqual(r["a"]["scores_body"], "carve")
        self.assertIsNone(r["b"]["scores_body"])
        self.assertIsNone(r["b"]["scores_file"])
        self.assertIn("scores: missing in v0-rebuild (no 05-scores.json in the version)", r["missing"])
        self.assertEqual(r["a"]["scores_file_sha256"], sha(self.store / "sk" / "05-scores-v0.json"))
        self.assertIn("SHA-256 " + r["a"]["scores_file_sha256"], C.render_text(r))

    def test_hand_collected_scores_behind_a_link_are_not_read(self):
        real = old_scores(self.store, sketch="elsewhere")
        a = hand_frozen(self.store, label="v0-carve", mesh="carve")
        os.symlink(real, self.store / "sk" / "05-scores-v0.json")
        b = Path(self.freeze(make_run(self.runs, "b"), label="m1-b")["path"])
        r = C.compare(a, b)
        self.assertIsNone(r["a"]["scores_body"])
        self.assertIn("scores: missing in v0-carve (05-scores-v0.json beside it is a symbolic link and is not "
                      "read)", r["missing"])

    def test_a_version_that_fails_verify_is_not_compared(self):
        a, b = self.two_new()
        S._make_writable(b)
        (b / "05-scores.json").write_text("{}")
        with self.assertRaisesRegex(S.VersionError, r"sk/m1-b fails verify \(mismatched\)"):
            C.compare(a, b)
        code, _, err = self.cli("compare", a, b)
        self.assertEqual(code, 2)
        self.assertIn("fails verify", err)

    def test_an_unfinished_freeze_is_never_compared(self):
        a, _ = self.two_new()
        partial = a.parent / f"{S.PARTIAL_PREFIX}m1-x-abc"
        S._make_writable(a)
        shutil.copytree(a, partial)
        with self.assertRaisesRegex(S.VersionError, "hidden folder, not a version"):
            C.compare(a, partial)
        self.assertEqual(self.cli("compare", a, partial)[0], 2)

    def test_cli_compare(self):
        a, b = self.two_new()
        code, out, err = self.cli("compare", "sk/m1-a", "m1-b", "--store", self.store)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("A  sk/m1-a", out)
        self.assertIn("Per view (delta = B - A", out)
        code, out, _ = self.cli("compare", a, b, "--json")
        self.assertEqual(json.loads(out)["triangles"]["high"]["delta"], 2000)

    def test_gap_tables_from_the_hand_frozen_sections(self):
        d = hand_frozen(self.store, files={"high.glb": b"h", "sections.json": json.dumps(
            {"pose_test": {"ungated_poses": [], "gaps_high": gaps(1.0, ("rest",))}}).encode()})
        b = Path(self.freeze(make_run(self.runs, "b"), label="m1-b")["path"])
        r = C.compare(d, b)
        self.assertEqual(r["joint_gaps"]["gaps_high"]["common_poses"], ["rest"])
        self.assertIn("stage-4.json: missing in v0-carve", r["missing"])


if __name__ == "__main__":
    unittest.main()
