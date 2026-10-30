"""Lab record readers: every stage-record, body, score and held-out format the pipeline has written, and the
quality summary built from them. All inputs are synthetic and written by the tests."""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.lab import heldout, quality, runindex  # noqa: E402
from test_herogen_lab import record, tiny_png, write  # noqa: E402

SECTIONS = runindex.SECTIONS


def mesh_glb(tris_per_section: dict[str, int], extra_points: bool = False, version: int = 2) -> bytes:
    """A GLB whose JSON chunk declares one indexed triangle mesh per named node (no binary chunk is needed to
    count). With `extra_points`, every mesh also gets a points primitive and an unindexed triangle primitive."""
    accessors, meshes, nodes = [], [], []
    for i, (name, tris) in enumerate(tris_per_section.items()):
        accessors.append({"count": tris * 3, "componentType": 5125, "type": "SCALAR"})
        prims = [{"attributes": {"POSITION": len(accessors) - 1}, "indices": len(accessors) - 1}]
        if extra_points:
            accessors.append({"count": 9, "componentType": 5126, "type": "VEC3"})
            prims += [{"attributes": {"POSITION": len(accessors) - 1}, "mode": 0},
                      {"attributes": {"POSITION": len(accessors) - 1}}]
        meshes.append({"name": name, "primitives": prims})
        nodes.append({"name": name, "mesh": i, "translation": [0.0, float(i), 0.0]})
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}],
           "nodes": nodes, "meshes": meshes, "accessors": accessors}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    return struct.pack("<III", 0x46546C67, version, 12 + 8 + len(js)) + struct.pack("<II", len(js), 0x4E4F534A) + js


def sections_json(profile=None, budgets=None, lods=None, extra=None) -> dict:
    data = {"units": "studs", "height": 5.5, "mesh_source": "sam3d-objects", "joints_source": "sam3d-body",
            "sections": [{"name": n, "parent": None if n == "LowerTorso" else "UpperTorso", "joint": "J",
                          "pivot": [0.0, 0.0, float(i)], "triangles_high": 100 + i, "triangles_low": 20}
                         for i, n in enumerate(SECTIONS)],
            "joints": {"LeftShoulder": [0.6, 0.2, 4.0], "Bad": [1, 2]}, "ends": {"LeftToe": [0.6, -0.5, 0.1]}}
    if profile is not None:
        data["profile"] = profile
    if budgets is not None:
        data["budgets"] = budgets
    if lods is not None:
        data["lods"] = lods
    data.update(extra or {})
    return data


class Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run = Path(self._tmp.name) / "run"
        self.run.mkdir()

    def tearDown(self):
        self._tmp.cleanup()


class StageRecordTest(Tmp):
    def test_old_record_without_stamps(self):
        write(self.run / "stage-0.json", {"stage": 0, "name": "input", "ok": True, "seconds": 0.1, "checks": []})
        s = runindex.stage_summary(self.run, "0", "input")
        self.assertEqual((s["started"], s["code"], s["calls"]), (None, None, None))

    def test_new_record_stamps(self):
        rec = record(3, "base mesh")
        rec.update(code={"commit": "07bfe07", "dirty": True}, mesh_source="sam3d-objects", joints_source=7,
                   calls=[{"model": "sam-3.1", "cached": True}, {"model": "muse-image-1.0", "cached": False}, "x"])
        write(self.run / "stage-3.json", rec)
        s = runindex.stage_summary(self.run, "3", "base mesh")
        self.assertEqual(s["started"], "2026-10-28T19:00:00-07:00")
        self.assertEqual(s["code"], {"commit": "07bfe07", "dirty": True})
        self.assertEqual(s["calls"], {"count": 2, "replayed": 1, "models": ["muse-image-1.0", "sam-3.1"]})
        self.assertEqual((s["mesh_source"], s["joints_source"]), ("sam3d-objects", None))
        write(self.run / "stage-3.json", {**rec, "code": {"commit": ""}})
        self.assertIsNone(runindex.stage_summary(self.run, "3", "base mesh")["code"])

    def test_check_extras(self):
        c = runindex.normalise_check({"name": "max_joint_gap_high", "value": 0.2, "threshold": 0.15, "rule": "<=",
                                      "pass": False, "poses": ["crouch", 3, "walk_a"], "review": True})
        self.assertEqual((c["poses"], c["sections"], c["review"]), (["crouch", "walk_a"], None, True))
        c = runindex.normalise_check({"name": "triangles_Head", "value": 1, "sections": ["Head"], "poses": "x"})
        self.assertEqual((c["poses"], c["sections"], c["review"]), (None, ["Head"], False))


class GlbTest(Tmp):
    def test_counts_triangles_per_section(self):
        tris = {n: 10 + i for i, n in enumerate(SECTIONS)}
        write(self.run / "high.glb", mesh_glb(tris, extra_points=True))
        g = runindex.glb_summary(self.run / "high.glb")
        # The points primitive is not counted; the unindexed triangle list adds 9 / 3 = 3 per section.
        self.assertEqual(g["sections"], {n: t + 3 for n, t in tris.items()})
        self.assertEqual(g["triangles"], sum(tris.values()) + 3 * 15)
        self.assertEqual((g["nodes"], g["bytes"]), (15, (self.run / "high.glb").stat().st_size))

    def test_not_a_glb(self):
        write(self.run / "a.glb", b"glTF")
        write(self.run / "b.glb", b"x" * 40)
        write(self.run / "c.glb", mesh_glb({"Head": 1}, version=1))
        for name in ("a.glb", "b.glb", "c.glb"):
            self.assertIn("error", runindex.glb_summary(self.run / name), name)
        self.assertIsNone(runindex.glb_summary(self.run / "missing.glb")["bytes"])
        bad_json = struct.pack("<IIIII", 0x46546C67, 2, 24, 4, 0x4E4F534A) + b"[1,]"
        write(self.run / "d.glb", bad_json)
        self.assertIn("error", runindex.glb_summary(self.run / "d.glb"))

    def test_png_size(self):
        write(self.run / "a.png", tiny_png(size=3))
        write(self.run / "b.png", b"not a png at all, not at all")
        self.assertEqual(runindex.png_size(self.run / "a.png"), (3, 3))
        self.assertIsNone(runindex.png_size(self.run / "b.png"))
        self.assertEqual(runindex.file_entry(self.run, "a.png")["px"], [3, 3])
        self.assertIsNone(runindex.file_entry(self.run, "missing.png"))


class ScoresTest(Tmp):
    VIEW = {"iou": 0.9, "colour_error": 7.0, "ssim": 0.8, "seen": True, "seen_shape": False, "pixels": 100}

    def test_new_format_lists_its_views(self):
        write(self.run / "05-scores.json", {
            "scored": "2026-10-29T04:20:04-07:00", "code": {"commit": "9c6e61b", "dirty": False},
            "views": [{"view": "front", "heldout": False, "image": "views/front.png", "yaw_deg": 0.0},
                      {"view": "front-right-30", "heldout": True, "image": "heldout/front-right-30.png",
                       "mask": "heldout/front-right-30-mask.png"}, {"view": "../x"}],
            "bodies": [{"name": "hybrid", "glb": "04-body/high.glb", "mesh_source": "sam3d-objects",
                        "triangles": 9686, "budget": 10742, "within_budget": True,
                        "profile": {"name": "roblox-avatar", "checks": []},
                        "views": {"front": self.VIEW, "front-right-30": {**self.VIEW, "seen": False, "yaw_deg": -26}}}]})
        write(self.run / "heldout" / "front-right-30.png", tiny_png())
        write(self.run / "05-renders" / "hybrid" / "front-right-30.png", tiny_png())
        sc = runindex.scores(self.run)
        self.assertEqual(sc["views"], ["front", "front-right-30"])
        self.assertEqual(sc["heldout_views"], ["front-right-30"])
        self.assertEqual(sc["heldout_image"], "heldout/front-right-30.png")
        self.assertEqual(sc["code"], {"commit": "9c6e61b", "dirty": False})
        b = sc["bodies"][0]
        self.assertEqual((b["profile"], b["budget"], b["within_budget"]), ("roblox-avatar", 10742, True))
        self.assertEqual(b["renders"], {"front-right-30": "05-renders/hybrid/front-right-30.png"})
        self.assertEqual(b["heldout_render"], "05-renders/hybrid/front-right-30.png")
        self.assertEqual(b["views"]["front-right-30"]["yaw_deg"], -26)
        self.assertEqual((b["views"]["front"]["seen_shape"], b["views"]["front"]["seen_colour"]), (False, None))

    def test_old_format_without_a_view_list(self):
        held = {**self.VIEW, "seen": False}
        write(self.run / "05-scores.json", {"bodies": [
            {"name": "carve", "views": {"front": self.VIEW, "back": self.VIEW, "front-left-34": held}},
            {"name": "hybrid", "views": {"front": self.VIEW, "front-left-34": held, "zz-extra": self.VIEW}}]})
        sc = runindex.scores(self.run)
        self.assertEqual(sc["views"], ["front", "back", "front-left-34", "zz-extra"])
        self.assertEqual(sc["heldout_views"], ["front-left-34"])
        self.assertIsNone(sc["heldout_image"])  # no heldout/front-left-34.png in this run

    def test_frozen_format_with_a_heldout_block(self):
        f = self.run / "05-scores-v0.json"
        write(f, {"heldout": {"image": "heldout/front-left-34.png", "present": True},
                  "bodies": [{"name": "carve", "mesh_source": "carve", "triangles": 23436,
                              "views": {"front": self.VIEW, "front-left-34": {**self.VIEW, "seen": False}}}]})
        sc = runindex.read_scores(f, self.run)
        self.assertEqual((sc["file"], sc["heldout_views"]), ("05-scores-v0.json", ["front-left-34"]))
        self.assertEqual(sc["view_info"]["front-left-34"]["image"], "heldout/front-left-34.png")


class BudgetAndDeliverablesTest(Tmp):
    def body(self, sections, files=("high.glb",), stage4=None):
        d = self.run / "04-body"
        write(d / "sections.json", sections)
        for f in files:
            write(d / f, mesh_glb({n: 10 for n in SECTIONS}) if f.endswith(".glb") else tiny_png(size=4))
        if stage4 is not None:
            write(self.run / "stage-4.json", stage4)
        return runindex.run_detail(self.run)

    def test_no_budget(self):
        d = self.body(sections_json("best", None, {"high": {"triangles": 173288, "atlas_size": 2048}}))
        self.assertEqual(d["budget_meters"], {"kind": "none", "items": [
            {"key": "high", "label": "High", "value": 173288, "limit": None}]})

    def test_lod_budgets(self):
        d = self.body(sections_json(None, {"triangles_high_max": 24000, "triangles_low_max": 6000},
                                    {"high": {"triangles": 22768}}), files=("high.glb", "low.glb"))
        m = d["budget_meters"]
        self.assertEqual(m["kind"], "lods")
        # The high count is the record's own total; the low one is summed from the sections (15 x 20).
        self.assertEqual([(i["key"], i["value"], i["limit"]) for i in m["items"]],
                         [("high", 22768, 24000), ("low", 300.0, 6000)])

    def test_group_budgets(self):
        groups = {"Head": {"sections": ["Head"], "triangles_max": 4000},
                  "Torso": {"sections": ["UpperTorso", "LowerTorso"], "triangles_max": 1750}, "Bad": {"x": 1}}
        sec = sections_json("roblox-avatar", {"triangles_total_max": 10742, "groups": groups},
                            {"high": {"triangles": 9686, "triangles_by_group": {"Head": 3652}}})
        m = self.body(sec)["budget_meters"]
        self.assertEqual(m["kind"], "groups")
        self.assertEqual([(i["key"], i["value"], i["limit"]) for i in m["items"]],
                         [("total", 9686, 10742), ("Head", 3652, 4000), ("Torso", 101.0 + 102.0, 1750)])

    def test_unreadable_sections(self):
        write(self.run / "04-body" / "high.glb", mesh_glb({"Head": 1}))
        (self.run / "04-body" / "sections.json").write_text("[1,")
        m = runindex.run_detail(self.run)["budget_meters"]
        self.assertEqual((m["kind"], m["error"]), ("unreadable", "sections.json could not be read (JSONDecodeError)"))

    def test_deliverables(self):
        stage4 = record(4, "game body")
        stage4["profile"] = {"name": "roblox-avatar", "lods": ["high"]}
        budgets = {"triangles_total_max": 10742, "not_built": ["R15 bone rig with skinning", ""],
                   "cages": "not built: out of scope"}
        d = self.body(sections_json("roblox-avatar", budgets), files=("high.glb", "atlas-high.png"),
                      stage4=stage4)
        by = {x["kind"]: x for x in d["deliverables"]}
        self.assertEqual(by["GLB high"]["status"], "built")
        self.assertEqual(by["GLB high"]["triangles"], 150)
        self.assertEqual(by["GLB high"]["path"], "04-body/high.glb")
        self.assertEqual(by["Texture atlas (high)"]["px"], [4, 4])
        self.assertEqual((by["GLB low"]["status"], by["GLB low"]["reason"]),
                         ("not in profile", "profile roblox-avatar builds high only"))
        self.assertEqual(by["R15 model for Roblox"]["status"], "not built")
        self.assertEqual(by["R15 bone rig with skinning"]["status"], "not built")
        self.assertEqual(by["Cages"]["reason"], "not built: out of scope")
        self.assertNotIn("Stage report", by)

    def test_no_body(self):
        write(self.run / "stage-0.json", record(0, "input"))
        d = runindex.run_detail(self.run)
        self.assertEqual(d["deliverables"][0]["status"], "not built")
        self.assertEqual(d["budget_meters"], {"kind": "unknown", "items": []})

    def test_sections_extras(self):
        pose_test = {"gate_studs": 0.15, "ungated_poses": ["twist_turn", 4],
                     "gaps_high": {"rest": {"Neck": 0.0, "Waist": "x"}, "crouch": {"Waist": 0.16}, "bad": 3},
                     "gaps_low": None}
        d = self.body(sections_json("best", None, None, {"pose_test": pose_test}))
        sec = d["bodies"][0]["sections"]
        self.assertEqual(sec["joints"], {"LeftShoulder": [0.6, 0.2, 4.0]})
        self.assertEqual(sec["ends"], {"LeftToe": [0.6, -0.5, 0.1]})
        pt = sec["pose_test"]
        self.assertEqual((pt["gate_studs"], pt["ungated_poses"]), (0.15, ["twist_turn"]))
        self.assertEqual(pt["gaps_high"], {"rest": {"Neck": 0.0}, "crouch": {"Waist": 0.16}})
        self.assertIsNone(pt["gaps_low"])


def view_entry(status, image, review=None, machine="needs review", yaw=30.0):
    return {"status": status, "machine_status": machine, "review": review, "accepted": status == "accepted",
            "kept_attempt": 1, "found_yaw_deg": yaw, "target_yaw_deg": 30, "yaw_window_deg": [25.0, 33.0],
            "facing": "image left", "files": {"image": image, "mask": image.replace(".png", "-mask.png")},
            "attempts": [{"attempt": 0}, {"attempt": 1}]}


class HeldoutTest(Tmp):
    def test_views_format_with_decisions(self):
        review = {"view": "front-left-60", "decision": "rejected", "by": "Reviewer A", "at": "2026-10-29T05:45:53-07:00",
                  "note": "arms not in A-pose", "from_status": "needs review"}
        views = {"front-right-30": view_entry("accepted", "heldout/front-right-30.png", machine="accepted", yaw=-26),
                 "front-left-30": view_entry("needs review", "heldout/review/front-left-30.png"),
                 "front-left-60": view_entry("rejected", "heldout/rejected/front-left-60.png", review=review),
                 "front-right-60": view_entry("odd", "../outside.png"), "../x": view_entry("accepted", "x.png")}
        rec = record("2h", "heldout", ok=False, checks=[
            {"name": "front-left-60.status_accepted", "value": 0.0, "threshold": 1.0, "op": "==", "pass": False,
             "note": "rejected on human review by Reviewer A: arms not in A-pose"},
            {"name": "front-right-30.figure_area_frac", "value": 0.25, "threshold": 0.01, "op": ">=", "pass": True}],
            details={"views": views, "decisions": [review, "junk"], "reference_note": "generated images",
                     "read_by": "scoring only"})
        write(self.run / "stage-2h.json", rec)
        for rel in ("heldout/front-right-30.png", "heldout/review/front-left-30.png",
                    "heldout/rejected/front-left-60.png", "heldout/rejected/front-left-60-mask.png"):
            write(self.run / rel, tiny_png())
        h = heldout.heldout(self.run)
        self.assertEqual(h["format"], "views")
        by = {v["name"]: v for v in h["views"]}
        self.assertEqual(set(by), {"front-right-30", "front-left-30", "front-left-60", "front-right-60"})
        self.assertEqual((by["front-right-30"]["group"], by["front-right-30"]["found_yaw_deg"]), ("accepted", -26))
        self.assertEqual(by["front-left-30"]["group"], "needs review")
        self.assertEqual(by["front-left-30"]["image"], "heldout/review/front-left-30.png")
        rej = by["front-left-60"]
        self.assertEqual((rej["group"], rej["machine_status"]), ("rejected", "needs review"))
        self.assertEqual(rej["review"], {"decision": "rejected", "by": "Reviewer A", "at": "2026-10-29T05:45:53-07:00",
                                         "note": "arms not in A-pose", "from_status": "needs review"})
        self.assertEqual(rej["mask"], "heldout/rejected/front-left-60-mask.png")
        self.assertEqual(([c["name"] for c in rej["checks"]], rej["checks_failed_gate"]),
                         (["front-left-60.status_accepted"], 1))
        self.assertEqual((by["front-left-30"]["attempts"], by["front-left-30"]["kept_attempt"]), (2, 1))
        # An unknown status is not guessed at, and a file outside heldout/ is never offered.
        self.assertEqual((by["front-right-60"]["status"], by["front-right-60"]["image"]), ("not recorded", None))
        self.assertEqual(by["front-left-60"]["label"], "Front-left 60°")
        self.assertEqual(len(h["decisions"]), 1)
        self.assertEqual((h["reference_note"], h["read_by"]), ("generated images", "scoring only"))
        self.assertEqual(h["record"]["status"], "gate failed")

    def test_single_view_format(self):
        for ok, status in ((True, "passed checks"), (False, "failed checks")):
            write(self.run / "stage-2h.json", record("2h", "heldout", ok=ok, checks=[
                {"name": "heldout_figure_area_frac", "value": 0.2, "threshold": 0.01, "pass": True}],
                details={"view": "front-left-34", "attempts": [{}, {}]}))
            write(self.run / "heldout" / "front-left-34.png", tiny_png())
            h = heldout.heldout(self.run)
            self.assertEqual(h["format"], "single")
            v = h["views"][0]
            self.assertEqual((v["name"], v["status"], v["image"], v["attempts"]),
                             ("front-left-34", status, "heldout/front-left-34.png", 2))
            self.assertEqual(len(v["checks"]), 1)
            self.assertIsNone(v["mask"])

    def test_failed_checks_are_not_rejections(self):
        write(self.run / "stage-2h.json", record("2h", "heldout", ok=False, details={"view": "front-left-34"}))
        v = heldout.heldout(self.run)["views"][0]
        self.assertEqual((v["status"], v["group"]), ("failed checks", "failed checks"))

    def test_a_record_that_is_not_an_object(self):
        write(self.run / "stage-2h.json", ["views"])
        write(self.run / "heldout" / "front-left-34.png", tiny_png())
        h = heldout.heldout(self.run)
        self.assertEqual((h["format"], h["error"]), ("unreadable", "stage-2h.json is not a JSON object"))
        write(self.run / "stage-0.json", 7)
        s = runindex.stage_summary(self.run, "0", "input")
        self.assertEqual((s["status"], s["error"]), ("unreadable", "stage-0.json is not a JSON object"))
        self.assertEqual(runindex._run_entry(self.run)["stages"]["0"], "unreadable")

    def test_images_without_a_record(self):
        self.assertIsNone(heldout.heldout(self.run))
        for name in ("front-left-34.png", "front-left-34-mask.png", "front-left-34-raw.png", "notes.png"):
            write(self.run / "heldout" / name, tiny_png())
        h = heldout.heldout(self.run)
        self.assertEqual(h["format"], "files")
        self.assertEqual([(v["name"], v["status"], v["mask"]) for v in h["views"]],
                         [("front-left-34", "not recorded", "heldout/front-left-34-mask.png")])

    def test_unreadable_record(self):
        (self.run / "stage-2h.json").write_text("{nope")
        h = heldout.heldout(self.run)
        self.assertEqual((h["format"], h["views"], h["record"]["status"]), ("unreadable", [], "unreadable"))


def chk(name, value, threshold, ok, op=">=", gate=True, **extra):
    return runindex.normalise_check({"name": name, "value": value, "threshold": threshold, "op": op, "pass": ok,
                                     "gate": gate, **extra})


def stage(key, checks, status="pass"):
    return {"stage": key, "status": status, "checks": checks}


class QualityTest(unittest.TestCase):
    GAPS = {"rest": {"Waist": 0.0}, "crouch": {"Waist": 0.16, "LeftKnee": 0.05}, "twist_turn": {"Waist": 0.33},
            "kick_R": {"RightHip": 0.4}}

    def test_labels(self):
        self.assertEqual(quality.label("min_side_px"), "Sketch size, shortest side (px)")
        self.assertEqual(quality.label("texture_ssim_left"), "Texture structure, left view (SSIM)")
        self.assertEqual(quality.label("triangles_LeftArm"), "Triangles, left arm part group")
        self.assertEqual(quality.label("front-left-30.found_yaw_in_window"),
                         "Front-left 30°: angle inside the requested window (° off)")
        self.assertEqual(quality.label("some_new_check"), "some new check")

    def test_groups_and_counts(self):
        stages = [stage("0", [chk("min_side_px", 1024, 256, True)]),
                  stage("2", [chk("figures_found", 4, 4, True, "==")]),
                  stage("3", [chk("silhouette_iou_left", 0.69, 0.9, False)], "gate failed"),
                  stage("4", [chk("texture_ssim_left", 0.6, 0.7, False), chk("texture_max_px", 1024, 2048, True, "<="),
                              chk("cap_points_outside_neighbour_high", 27, 0, False, "==", gate=False)]),
                  stage("5", [chk("stages_reported", 6, 6, True)])]
        q = quality.quality(stages, None, None, [], True)
        groups = {g["name"]: g for g in q["groups"]}
        self.assertEqual((q["gates"], q["passed"]), (5, 3))
        self.assertEqual((groups["Body"]["gates"], groups["Body"]["failed"]), (2, 1))  # texture_max_px is Body
        self.assertEqual((groups["Texture"]["gates"], groups["Held-out"]["gates"]), (1, 0))
        self.assertEqual(q["reported_below"], 1)
        self.assertNotIn("stages_reported", [c["name"] for c in q["checks"]])
        self.assertEqual(q["status"]["word"], "Needs attention")

    def test_joint_failure_uses_the_gap_table(self):
        c = chk("max_joint_gap_high", 0.16, 0.15, False, "<=", poses=["rest", "crouch"])
        q = quality.quality([stage("4", [c])], None, {"gaps_high": self.GAPS, "ungated_poses": ["twist_turn"]}, [],
                            True)
        f = q["failures"][0]
        self.assertEqual(f["act"], {"type": "joint", "joint": "Waist", "pose": "crouch", "lod": "high", "gap": 0.16,
                                    "limit": 0.15, "pose_in_viewer": True})
        self.assertEqual(f["text"], "Joint gap 0.160 studs at the waist, crouch pose, high body. Limit 0.15.")
        # Without a pose list the ungated poses are skipped: the worst is kick_R, a pose the viewer cannot show.
        c = chk("max_joint_gap_high", 0.4, 0.15, False, "<=")
        f = quality.quality([stage("4", [c])], None, {"gaps_high": self.GAPS, "ungated_poses": ["twist_turn"]}, [],
                            True)["failures"][0]
        self.assertEqual((f["act"]["pose"], f["act"]["joint"], f["act"]["pose_in_viewer"]), ("kick_R", "RightHip",
                                                                                            False))
        # No gap table: the check's own numbers, and nothing to show.
        f = quality.quality([stage("4", [c])], None, None, [], True)["failures"][0]
        self.assertIsNone(f["act"])
        self.assertIn("0.4", f["text"])

    def test_heldout_pose_table(self):
        c = chk("max_joint_gap_high_heldout", 0.2, 0.15, False, "<=", poses=["bow"])
        pt = {"gaps_high_heldout": {"bow": {"Neck": 0.2}}, "gaps_high": {"bow": {"Neck": 9.0}}}
        f = quality.quality([stage("4", [c])], None, pt, [], True)["failures"][0]
        self.assertEqual((f["act"]["joint"], f["act"]["gap"]), ("Neck", 0.2))
        self.assertIn("a held-out pose", f["text"])

    def test_texture_items_merge_per_view(self):
        checks = [chk("texture_colour_error_left", 13.56, 12.0, False, "<="), chk("texture_ssim_left", 0.6386, 0.7, False),
                  chk("texture_ssim_right", 0.69, 0.7, False)]
        fails = quality.quality([stage("4", checks)], None, None, [], True)["failures"]
        self.assertEqual([f["act"] for f in fails], [{"type": "view", "view": "left"}, {"type": "view", "view": "right"}])
        self.assertEqual(fails[0]["text"], "Texture on the left view doesn't match it: colour error 13.56 (needs ≤ 12), "
                                           "structure 0.6386 (needs ≥ 0.7).")

    def test_heldout_record_from_elsewhere_and_its_reasons(self):
        own = [stage("2h", [chk("front-left-60.status_accepted", 0, 1, False, "==")])]
        other = stage("2h", [chk("front-left-60.status_accepted", 0.0, 1.0, False, "==",
                                 note="rejected on human review by Reviewer A: arms not in A-pose")])
        q = quality.quality(own, other, None, [{"name": "front-left-60", "status": "rejected"}], True)
        f = q["failures"][0]
        self.assertEqual((f["kind"], f["group"], f["act"]), ("warn", "Held-out", {"type": "heldout",
                                                                                 "view": "front-left-60"}))
        self.assertEqual(f["text"], "Front-left 60°: rejected on human review by Reviewer A: arms not in A-pose.")
        self.assertEqual(q["gates"], 1)  # the other record replaced the run's own stage 2h

    def test_statuses(self):
        ok = [stage(k, [chk("a", 1, 1, True)]) for k in ("0", "1", "2", "3", "4")]
        self.assertEqual(quality.quality(ok, None, None, [], True)["status"]["word"], "Ready")
        self.assertEqual(quality.quality(ok, None, None, [], False)["status"]["word"], "No body yet")
        self.assertEqual(quality.quality(ok[3:], None, None, [], True)["status"]["word"], "Partial run")
        self.assertEqual(quality.quality([], None, None, [], False)["status"]["word"], "Not run")
        bad = ok + [{"stage": "2h", "status": "unreadable", "checks": []}]
        st = quality.quality(bad, None, None, [], True)["status"]
        self.assertEqual((st["word"], st["detail"]), ("Can't read", "unreadable: stage-2h.json"))


if __name__ == "__main__":
    unittest.main()
