"""Lab workspace: several runs roots, characters grouped by sketch, profiles, held-out records shared between runs
of one turnaround, frozen versions, their path safety, the API payloads and the CLI. Synthetic folders only."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.lab import __main__ as cli  # noqa: E402
from herogen.lab import versions  # noqa: E402
from herogen.lab.server import make_server  # noqa: E402
from herogen.lab.workspace import Workspace  # noqa: E402
from test_herogen_lab import record, tiny_png, write  # noqa: E402
from test_herogen_lab_records import mesh_glb, sections_json, view_entry  # noqa: E402

SECTIONS = [s["name"] for s in sections_json()["sections"]]
VIEWS = ["front", "back", "left", "right"]


def make_run(root: Path, name: str, sketch=(10, 10, 10), views=(20, 20, 20), profile="best", stage4_profile=None,
             body=True, heldout_views=None, started="2026-10-29T03:00:00-07:00", source_name="pencil-sketch.png",
             mesh_source="sam3d-objects") -> Path:
    """A run folder: sketch and turnaround images drawn in the given colours (the colour decides the bytes, so two
    runs with the same colour share a sketch or a turnaround), records for stages 0 to 4, and optionally a body
    and a stage-2h record with held-out views."""
    run = root / name
    if sketch is not None:
        write(run / "00-sketch.png", tiny_png(sketch))
    write(run / "stage-0.json", {**record(0, "input", details={"source_name": source_name,
                                                                 "label": "pencil sketch"}), "started": started})
    write(run / "stage-1.json", record(1, "cover_art"))
    write(run / "stage-2.json", record(2, "turnaround"))
    if views is not None:
        for v in VIEWS:
            write(run / "views" / f"{v}.png", tiny_png(views))
    write(run / "stage-3.json", record(3, "base mesh"))
    if body:
        s4 = {**record(4, "game body", checks=[{"name": "max_joint_gap_high", "value": 0.16, "threshold": 0.15,
                                                "rule": "<=", "pass": False, "poses": ["crouch"]}]),
              "started": started}
        if stage4_profile is not None:
            s4["profile"] = {"name": stage4_profile, "lods": ["high"]}
        write(run / "stage-4.json", s4)
        write(run / "04-body" / "high.glb", mesh_glb({n: 10 for n in SECTIONS}))
        extra = {"pose_test": {"gate_studs": 0.15, "gaps_high": {"crouch": {"Waist": 0.16}}},
                 "mesh_source": mesh_source}
        write(run / "04-body" / "sections.json", sections_json(profile, None, {"high": {"triangles": 150}}, extra))
    if heldout_views is not None:
        views_rec = {n: view_entry(st, f"heldout/{n}.png", machine=st) for n, st in heldout_views.items()}
        write(run / "stage-2h.json", {**record("2h", "heldout", ok=False, checks=[
            {"name": f"{n}.status_accepted", "value": 0.0, "threshold": 1.0, "op": "==", "pass": st == "accepted",
             "note": f"{st} by the checks"} for n, st in heldout_views.items()], details={"views": views_rec}),
            "started": started})
        for n in heldout_views:
            write(run / "heldout" / f"{n}.png", tiny_png())
    return run


def make_version(root: Path, vid: str, label: str, sketch_run="pencil-2", mesh_source="carve", tris=10,
                 files_sha=None) -> Path:
    folder = root / vid
    write(folder / "high.glb", mesh_glb({n: tris for n in SECTIONS}))
    write(folder / "sections.json", sections_json("roblox-ingame"))
    sha = versions.file_sha256(folder / "high.glb")
    write(folder / "version.json", {
        "label": label, "milestone": "base", "mesh_source": mesh_source, "joints": "silhouette",
        "profile": "roblox-ingame budgets 24000/6000", "built": "2026-10-28T21:58:35-07:00", "code": "commit e79a2d9",
        "provenance": "reconstructed", "sketch_run": sketch_run, "frozen": "2026-10-29", "immutable": True,
        "files_sha256": files_sha if files_sha is not None else {"high.glb": sha, "low.glb": "0" * 64}})
    return folder


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.a, self.b, self.v = base / "a" / "runs", base / "b" / "runs", base / "versions"
        # Root a: three profiles of one sketch, plus a run of another sketch and one with no sketch at all.
        make_run(self.a, "pencil-2", profile=None)
        make_run(self.a, "pencil-2-best", profile="best", started="2026-10-29T04:00:00-07:00")
        make_run(self.a, "pencil-2-avatar", profile=None, stage4_profile="roblox-avatar")
        make_run(self.a, "other", sketch=(90, 10, 10), views=(90, 90, 90), profile="roblox-ingame")
        make_run(self.a, "no-sketch", sketch=None, profile="roblox-ingame")
        # Root b: a same-named run of the same sketch that only made the held-out views, and one whose turnaround
        # differs (so it does not share them).
        make_run(self.b, "pencil-2", body=False, heldout_views={"front-right-30": "accepted",
                                                                 "front-left-30": "needs review",
                                                                 "front-left-60": "rejected"})
        make_run(self.b, "pencil-2-redrawn", views=(30, 30, 30), profile="roblox-ingame")
        make_version(self.v, "pencil-2/v0-carve", "v0-carve")
        make_version(self.v, "pencil-2/v0-hybrid", "v0-hybrid", mesh_source="sam3d-objects", tris=12)
        make_version(self.v, "lonely", "lonely", sketch_run="nowhere")
        write(self.v / "pencil-2" / "05-scores-v0.json", {"bodies": [
            {"name": "carve", "mesh_source": "carve", "triangles": 150, "views": {"front": {"iou": 0.96}}},
            {"name": "hybrid", "mesh_source": "sam3d-objects", "triangles": 180, "views": {"front": {"iou": 0.97}}},
            {"name": "hybrid-variant", "mesh_source": "sam3d-objects", "triangles": 180, "variant_of": "hybrid",
             "views": {}}]})
        self.ws = Workspace([self.a, self.b], self.v)

    def tearDown(self):
        self.tmp.cleanup()

    def character(self, data, run_id):
        return next(c for c in data["characters"] if run_id in c["runs"])

    def test_runs_from_several_roots(self):
        runs = {r["id"]: r for r in self.ws.runs()}
        self.assertEqual(set(runs), {"r1/pencil-2", "r1/pencil-2-best", "r1/pencil-2-avatar", "r1/other",
                                     "r1/no-sketch", "r2/pencil-2", "r2/pencil-2-redrawn"})
        self.assertEqual((runs["r2/pencil-2"]["root"], runs["r2/pencil-2"]["name"]), ("r2", "pencil-2"))
        self.assertIsNone(self.ws.run_path("r3/pencil-2"))
        self.assertIsNone(self.ws.run_path("r1/../b"))
        self.assertIsNone(self.ws.run_path("pencil-2"))
        self.assertEqual(self.ws.run_path("r2/pencil-2"), self.b / "pencil-2")
        self.assertEqual(runs["r1/pencil-2-best"]["triangles"], 150)  # counted from the GLB

    def test_grouped_by_sketch(self):
        data = self.ws.workspace()
        pencil = self.character(data, "r1/pencil-2")
        self.assertEqual(set(pencil["runs"]), {"r1/pencil-2", "r1/pencil-2-best", "r1/pencil-2-avatar",
                                               "r2/pencil-2", "r2/pencil-2-redrawn"})
        self.assertEqual((pencil["title"], pencil["label"]), ("pencil-sketch", "pencil sketch"))
        self.assertEqual(pencil["runs"][0], "r1/pencil-2-best")  # newest first
        self.assertEqual(self.character(data, "r1/other")["runs"], ["r1/other"])
        lone = self.character(data, "r1/no-sketch")
        self.assertEqual((lone["key"], lone["sketch_sha"]), ("run-r1-no-sketch", None))
        self.assertEqual(data["characters"][0]["key"], pencil["key"])
        self.assertEqual([r["key"] for r in data["roots"]], ["r1", "r2"])

    def test_profiles(self):
        pencil = self.character(self.ws.workspace(), "r1/pencil-2")
        self.assertEqual(pencil["profiles"]["best"], ["r1/pencil-2-best"])
        self.assertEqual(pencil["profiles"]["roblox-avatar"], ["r1/pencil-2-avatar"])  # from stage-4.json
        self.assertEqual(pencil["profiles"]["roblox-ingame"], ["r2/pencil-2-redrawn"])
        self.assertEqual(sorted(pencil["profiles"][None]), ["r1/pencil-2", "r2/pencil-2"])  # none recorded

    def test_heldout_shared_by_an_identical_turnaround(self):
        d = self.ws.run_detail("r1/pencil-2-best")
        self.assertEqual(d["heldout_source"], "r2/pencil-2")
        self.assertEqual(d["heldout"]["format"], "views")
        self.assertEqual({v["name"]: v["group"] for v in d["heldout"]["views"]},
                         {"front-right-30": "accepted", "front-left-30": "needs review", "front-left-60": "rejected"})
        held = next(g for g in d["quality"]["groups"] if g["name"] == "Held-out")
        self.assertEqual((held["gates"], held["failed"]), (3, 2))
        # A different turnaround of the same sketch does not borrow them; nor does another sketch.
        self.assertIsNone(self.ws.run_detail("r2/pencil-2-redrawn")["heldout_source"])
        self.assertIsNone(self.ws.run_detail("r1/other")["heldout"])
        # A run's own record wins.
        own = self.ws.run_detail("r2/pencil-2")
        self.assertEqual(own["heldout_source"], "r2/pencil-2")

    def test_own_record_wins_over_images_only(self):
        write(self.a / "pencil-2-best" / "heldout" / "front-left-34.png", tiny_png())
        d = self.ws.run_detail("r1/pencil-2-best")
        self.assertEqual((d["heldout_source"], d["heldout"]["format"]), ("r2/pencil-2", "views"))
        d = self.ws.run_detail("r2/pencil-2-redrawn")
        self.assertIsNone(d["heldout"])
        write(self.b / "pencil-2-redrawn" / "heldout" / "front-left-34.png", tiny_png())
        d = self.ws.run_detail("r2/pencil-2-redrawn")
        self.assertEqual((d["heldout_source"], d["heldout"]["format"]), ("r2/pencil-2-redrawn", "files"))

    def test_run_detail_payload(self):
        d = self.ws.run_detail("r1/pencil-2-avatar")
        json.dumps(d, allow_nan=False)
        self.assertEqual((d["id"], d["root"], d["name"], d["profile"]),
                         ("r1/pencil-2-avatar", "r1", "pencil-2-avatar", "roblox-avatar"))
        self.assertEqual(d["path"], f"{self.a}/pencil-2-avatar")
        self.assertEqual(d["images"]["sketch"]["px"], [2, 2])
        self.assertEqual(d["bodies"][0]["glb"]["high"]["triangles"], 150)
        self.assertEqual(d["quality"]["failures"][0]["act"]["joint"], "Waist")
        self.assertIsNone(self.ws.run_detail("r1/missing"))

    def test_versions(self):
        data = self.ws.workspace()
        pencil = self.character(data, "r1/pencil-2")
        self.assertEqual([v["id"] for v in pencil["versions"]], ["pencil-2/v0-carve", "pencil-2/v0-hybrid"])
        self.assertEqual([v["id"] for v in data["unattached_versions"]], ["lonely"])
        carve = pencil["versions"][0]
        self.assertEqual((carve["label"], carve["triangles"], carve["frozen"], carve["immutable"]),
                         ("v0-carve", 150, "2026-10-29", True))
        self.assertEqual(carve["integrity"], [{"file": "high.glb", "present": True, "matches": True},
                                              {"file": "low.glb", "present": False, "matches": None}])
        self.assertEqual((carve["scores"]["file"], carve["scores"]["body"]["name"]), ("05-scores-v0.json", "carve"))
        # v0-hybrid has 180 triangles and mesh source sam3d-objects: one non-variant body matches.
        self.assertEqual(pencil["versions"][1]["scores"]["body"]["name"], "hybrid")
        write(self.v / "pencil-2" / "v0-carve" / "high.glb", mesh_glb({n: 11 for n in SECTIONS}))
        carve = self.ws.version_detail("pencil-2/v0-carve")
        self.assertFalse(carve["integrity"][0]["matches"])  # the file changed since it was frozen
        self.assertIsNone(carve["scores"])  # and no longer matches the scored body's triangle count
        self.assertIsNone(self.ws.version_detail("pencil-2/missing"))
        self.assertIsNone(self.ws.version_detail("../versions"))

    def test_ambiguous_scores_are_not_matched(self):
        write(self.v / "pencil-2" / "05-scores-v1.json", {"bodies": [
            {"name": "a", "mesh_source": "carve", "triangles": 150, "views": {}},
            {"name": "b", "mesh_source": "carve", "triangles": 150, "views": {}}]})
        write(self.v / "pencil-2" / "05-scores-v0.json", {"bodies": []})
        self.assertIsNone(self.ws.version_detail("pencil-2/v0-carve")["scores"])

    def test_versions_attach_by_sketch_not_by_name(self):
        # A run named like the version's sketch_run, in another root, holding another sketch: the name alone no
        # longer says which character the version belongs to, so it attaches to neither.
        make_run(self.b, "pencil-2-twin", sketch=(70, 70, 70))
        (self.b / "pencil-2-twin").rename(self.b / "pencil-2-other")
        make_run(self.b / ".." / "c", "pencil-2", sketch=(70, 70, 70))
        ws = Workspace([self.a, self.b, self.b.parent / "c"], self.v)
        data = ws.workspace()
        self.assertEqual(sum(len(c["versions"]) for c in data["characters"]), 0)
        why = {v["id"]: v["unattached_reason"] for v in data["unattached_versions"]}
        self.assertEqual(why["pencil-2/v0-carve"], "runs named pencil-2 hold 2 different sketches")
        self.assertEqual(why["lonely"], "no run named nowhere in the runs folders")
        # A run of that name without a sketch cannot vouch for one either.
        ws = Workspace([self.a, self.b], self.v)
        (self.b / "pencil-2" / "00-sketch.png").unlink()
        why = {v["id"]: v["unattached_reason"] for v in ws.workspace()["unattached_versions"]}
        self.assertEqual(why["pencil-2/v0-carve"], "a run named pencil-2 has no 00-sketch.png to match")

    def test_unreadable_version_is_listed_with_its_reason(self):
        (self.v / "pencil-2" / "v0-hybrid" / "version.json").write_text("{nope")
        why = {v["id"]: v["unattached_reason"] for v in self.ws.workspace()["unattached_versions"]}
        self.assertEqual(why["pencil-2/v0-hybrid"], "version.json could not be read (JSONDecodeError)")

    def test_runs_without_a_sketch_never_borrow(self):
        # Two runs with no sketch and identical views: the one without a held-out record does not take the other's.
        make_run(self.b, "nosketch-held", sketch=None, body=False, heldout_views={"front-right-30": "accepted"})
        d = self.ws.run_detail("r1/no-sketch")
        self.assertEqual((d["heldout"], d["heldout_source"]), (None, None))
        self.assertEqual(self.ws.run_detail("r2/nosketch-held")["heldout_source"], "r2/nosketch-held")

    def test_symlinked_views_do_not_make_twins(self):
        twin = make_run(self.a, "linked", views=None)
        try:
            os.symlink(self.b / "pencil-2" / "views", twin / "views")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        self.assertIsNone(self.ws.run_detail("r1/linked")["heldout_source"])

    def test_generation_keeps_a_borrowed_stage_out_of_the_total(self):
        g = self.ws.run_detail("r1/pencil-2-best")["generation"]
        rows = {r["stage"]: r for r in g["rows"]}
        self.assertEqual(rows["2h"]["from_run"], "r2/pencil-2")
        self.assertEqual(g["total"]["seconds"], 5 * 1.5)  # stages 0 to 4 of this run, 1.5 s each
        self.assertAlmostEqual(g["total"]["cost_usd"], 0.05)
        self.assertEqual(g["with_borrowed"]["seconds"], 6 * 1.5)
        own = self.ws.run_detail("r2/pencil-2")["generation"]
        self.assertIsNone(own["with_borrowed"])
        self.assertIsNone(own["rows"][[r["stage"] for r in own["rows"]].index("2h")]["from_run"])

    def test_unreadable_run_folder_stays_listed(self):
        if os.name != "posix" or os.geteuid() == 0:
            self.skipTest("needs a non-root POSIX user")
        locked = make_run(self.a, "locked")
        locked.chmod(0)
        try:
            runs = {r["id"]: r for r in self.ws.runs()}
            self.assertEqual((runs["r1/locked"]["readable"], runs["r1/locked"]["error"]),
                             (False, "folder could not be read (PermissionError)"))
            data = self.ws.workspace()
            json.dumps(data, allow_nan=False)
            self.assertTrue(any(c["runs"] == ["r1/locked"] for c in data["characters"]))
            d = self.ws.run_detail("r1/locked")
            self.assertEqual(d["error"], "folder could not be read (PermissionError)")
            server = make_server(self.ws, port=0, quiet=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_address[1]}"
                for path in ["/api/runs/r1/locked", "/api/workspace", "/api/runs"]:
                    with urllib.request.urlopen(base + path, timeout=5) as r:
                        self.assertEqual(r.status, 200, path)
                        body = json.loads(r.read())
                self.assertIn("r1/locked", {x["id"] for x in body["runs"]})
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
        finally:
            locked.chmod(0o755)

    def test_no_versions_root(self):
        ws = Workspace([self.a])
        data = ws.workspace()
        self.assertIsNone(data["versions_root"])
        self.assertEqual(sum(len(c["versions"]) for c in data["characters"]), 0)
        self.assertIsNone(ws.version_detail("pencil-2/v0-carve"))

    def test_malformed_runs_and_versions(self):
        (self.a / "pencil-2" / "stage-1.json").write_text("{broken")
        (self.a / "pencil-2" / "04-body" / "sections.json").write_text("[1, 2")
        (self.a / "pencil-2" / "05-scores.json").write_text('{"bodies": 3}')
        write(self.a / "pencil-2" / "stage-2h.json", ["not", "a", "record"])
        (self.v / "lonely" / "version.json").write_text("{nope")
        data = self.ws.workspace()
        json.dumps(data, allow_nan=False)
        self.assertFalse(data["runs"]["r1/pencil-2"]["readable"])
        self.assertEqual(data["unattached_versions"][0]["error"], "version.json could not be read (JSONDecodeError)")
        d = self.ws.run_detail("r1/pencil-2")
        json.dumps(d, allow_nan=False)
        self.assertEqual(d["quality"]["status"]["word"], "Can't read")
        self.assertEqual(d["bodies"][0]["sections"]["error"], "sections.json could not be read (JSONDecodeError)")
        self.assertEqual(d["scores"]["error"], "05-scores.json has no list of bodies")


class VersionPathSafetyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "versions"
        self.folder = make_version(self.root, "pencil-2/v0-carve", "v0-carve")
        write(self.root / "pencil-2" / "05-scores-v0.json", {"bodies": []})
        write(self.folder / "notes.txt", b"x")
        write(self.folder / ".hidden.json", {})
        write(self.folder / "sub" / "x.png", tiny_png())
        write(base / "outside.glb", b"glTF")
        self.links = True
        try:
            os.symlink(base / "outside.glb", self.folder / "link.glb")
        except (OSError, NotImplementedError):
            self.links = False

    def tearDown(self):
        self.tmp.cleanup()

    def test_allowed(self):
        self.assertEqual(versions.safe_version_file(self.root, "pencil-2/v0-carve/high.glb"),
                         (self.folder / "high.glb").resolve())
        self.assertIsNotNone(versions.safe_version_file(self.root, "pencil-2/v0-carve/version.json"))

    def test_refused(self):
        for rel in ["pencil-2/v0-carve/../v0-carve/high.glb", "pencil-2/v0-carve/./high.glb", "../outside.glb",
                    "pencil-2/v0-carve/notes.txt", "pencil-2/v0-carve/.hidden.json", "pencil-2/v0-carve/sub/x.png",
                    "pencil-2/05-scores-v0.json", "pencil-2/v0-carve", "pencil-2/v0-carve/", "pencil-2/v1/high.glb",
                    "pencil-2\\v0-carve\\high.glb", "pencil-2/v0-carve/high.glb\x00.json", "/pencil-2/v0-carve/high.glb",
                    "pencil-2/v0-carve/missing.glb", ""]:
            self.assertIsNone(versions.safe_version_file(self.root, rel), rel)
        self.assertIsNone(versions.safe_version_file(None, "pencil-2/v0-carve/high.glb"))

    def test_symlink_out_refused(self):
        if not self.links:
            self.skipTest("symlinks unavailable")
        self.assertIsNone(versions.safe_version_file(self.root, "pencil-2/v0-carve/link.glb"))


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.a, cls.b, cls.v = base / "a", base / "b", base / "v"
        make_run(cls.a, "pencil-2", heldout_views={"front-right-30": "accepted"})
        make_run(cls.b, "pencil-2", profile="roblox-ingame")
        make_version(cls.v, "pencil-2/v0-carve", "v0-carve")
        write(base / "outside.png", tiny_png())
        cls.server = make_server(Workspace([cls.a, cls.b], cls.v, labels=["runs-a", "runs-b"]), port=0, quiet=True)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=5) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def test_workspace(self):
        status, headers, body = self.get("/api/workspace")
        self.assertEqual((status, headers["Content-Type"]), (200, "application/json"))
        d = json.loads(body)
        self.assertEqual(d["roots"], [{"key": "r1", "label": "runs-a", "exists": True},
                                      {"key": "r2", "label": "runs-b", "exists": True}])
        self.assertEqual(d["versions_root"]["exists"], True)
        self.assertEqual(d["profiles"], ["best", "roblox-ingame", "roblox-avatar"])
        self.assertEqual(len(d["characters"]), 1)
        self.assertEqual(sorted(d["characters"][0]["runs"]), ["r1/pencil-2", "r2/pencil-2"])
        self.assertEqual(d["characters"][0]["versions"][0]["id"], "pencil-2/v0-carve")

    def test_run_and_version_detail(self):
        status, _, body = self.get("/api/runs/r2/pencil-2")
        d = json.loads(body)
        self.assertEqual((status, d["heldout_source"], d["path"]), (200, "r1/pencil-2", "runs-b/pencil-2"))
        status, _, body = self.get("/api/versions/pencil-2/v0-carve")
        self.assertEqual((status, json.loads(body)["label"]), (200, "v0-carve"))
        for bad in ["/api/versions/pencil-2", "/api/versions/..%2Fv", "/api/versions/", "/api/runs/r3/pencil-2",
                    "/api/runs/r1/..", "/api/runs/r1%2F..%2Fb"]:
            self.assertEqual(self.get(bad)[0], 404, bad)

    def test_files(self):
        status, headers, body = self.get("/runs/r2/pencil-2/04-body/high.glb")
        self.assertEqual((status, headers["Content-Type"], body[:4]), (200, "model/gltf-binary", b"glTF"))
        status, headers, body = self.get("/versions/pencil-2/v0-carve/high.glb")
        self.assertEqual((status, headers["Content-Type"], body[:4]), (200, "model/gltf-binary", b"glTF"))
        self.assertEqual(headers["Content-Security-Policy"], "default-src 'none'; sandbox")
        for bad in ["/runs/r2/pencil-2/../../outside.png", "/runs/r2/%2e%2e/outside.png", "/runs/r3/pencil-2/00-sketch.png",
                    "/runs/r2/pencil-2/%2e%2e/%2e%2e/outside.png", "/runs/r2", "/runs/r2/", "/runs/pencil-2/00-sketch.png",
                    "/versions/pencil-2/v0-carve/../../../outside.png", "/versions/pencil-2/v0-carve/%2e%2e/x.glb",
                    "/versions/pencil-2/high.glb", "/versions/../a/pencil-2/00-sketch.png"]:
            self.assertEqual(self.get(bad)[0], 404, bad)

    def test_versions_route_without_a_versions_root(self):
        server = make_server(Workspace([self.a]), port=0, quiet=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            for path in ["/versions/pencil-2/v0-carve/high.glb", "/api/versions/pencil-2/v0-carve"]:
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(base + path, timeout=5)
                self.assertEqual(cm.exception.code, 404, path)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


class CliTest(unittest.TestCase):
    def run_cli(self, *args):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = cli.main(["serve", *args])
        return code, err.getvalue()

    def test_refuses_bad_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.run_cli("--runs", f"{tmp}/missing"), (2, f"runs folder not found: {tmp}/missing\n"))
            self.assertEqual(self.run_cli("--runs", tmp, "--runs", tmp + "/")[0], 2)
            code, err = self.run_cli("--runs", tmp, "--versions", f"{tmp}/nope")
            self.assertEqual((code, err), (2, f"versions folder not found: {tmp}/nope\n"))


if __name__ == "__main__":
    unittest.main()
