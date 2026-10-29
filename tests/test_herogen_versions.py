"""Versioned bodies: freeze, list and verify on synthetic run folders and stores in temporary folders."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from herogen.versions import store as S  # noqa: E402
from herogen.versions.__main__ import main  # noqa: E402

COMMIT = "c0ffee" + "0" * 34
VIEWS = ("front", "back", "left", "right")
POSES = ("rest", "walk_a", "arms_up", "twist_turn")
JOINTS = ("Neck", "LeftShoulder", "LeftHip")
IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def check(name, value, threshold, passed, gate=True, rule=None):
    c = {"name": name, "value": value, "threshold": threshold, "pass": passed, "gate": gate}
    if rule:
        c["rule"] = rule
    return c


def gaps(scale: float, poses=POSES) -> dict:
    return {p: {j: round(scale * (i + 1) * (k + 1) / 100, 4) for k, j in enumerate(JOINTS)}
            for i, p in enumerate(poses)}


def view_row(iou, colour, ssim, seen=True, shape=True):
    return {"iou": iou, "colour_error": colour, "ssim": ssim, "pixels": 1000, "seen": seen,
            "seen_shape": seen and shape, "seen_colour": seen}


def scores_entry(name="carve", glb="04-body/high.glb", mesh="carve", base=0.9, triangles=20000,
                 heldout=("front-left-34",), variant_of=None, triangles_low=5000):
    views = {v: view_row(round(base + i / 100, 4), 5.0 + i, 0.8 + i / 100) for i, v in enumerate(VIEWS)}
    for h in heldout:
        views[h] = view_row(round(base - 0.3, 4), 40.0, 0.12, seen=False)
    e = {"name": name, "glb": glb, "mesh_source": mesh, "joints_source": "silhouette", "triangles": triangles,
         "triangles_low": triangles_low, "atlas": 1024, "views": views}
    if variant_of:
        e["variant_of"] = variant_of
    return e


def make_run(root: Path, name: str = "r1", *, code: dict | None = None, stamp: bool = True, mesh: str = "carve",
             profile: str | None = "roblox-ingame", lods=("high", "low"), sketch: bytes = b"sketch one",
             base: float = 0.9, gap_scale: float = 1.0, poses=POSES, triangles: int = 20000, scores: bool = True,
             old: bool = False, heldout: bool = True) -> Path:
    """A run folder shaped like the pipeline's, with made-up bytes for every image and mesh. `old` writes
    records as runs did before commit stamping, profiles, mesh sources, held-out decisions and scores."""
    run = root / name
    run.mkdir(parents=True)
    (run / "00-sketch.png").write_bytes(sketch)
    for v in (*VIEWS, "top"):
        (run / "views").mkdir(exist_ok=True)
        (run / "masks").mkdir(exist_ok=True)
        (run / "views" / f"{v}.png").write_bytes(b"view " + v.encode() + sketch)
        (run / "masks" / f"{v}.png").write_bytes(b"mask " + v.encode() + sketch)
    if heldout:
        (run / "heldout" / "review").mkdir(parents=True)
        for suffix in ("", "-mask", "-raw"):
            (run / "heldout" / f"front-left-34{suffix}.png").write_bytes(b"held" + suffix.encode() + sketch)
        (run / "heldout" / "review" / "front-left-34.png").write_bytes(b"not a scored view")
    stamp_code = None if old or not stamp else (code if code is not None else {"commit": COMMIT, "dirty": False})

    def record(n, name, ok=True, checks=(), **extra):
        r = {"stage": n, "name": name, "ok": ok, "started": "2026-10-28T19:00:00-07:00", "seconds": 1.0,
             "cost_usd": 0.0, "outputs": [], "checks": list(checks), "notes": ""}
        if stamp_code is not None and n in (3, 4):
            r["code"] = stamp_code
        r.update(extra)
        write_json(run / f"stage-{n}.json", r)

    record(0, "input")
    record(1, "cover_art")
    record(2, "turnaround")
    if not old:
        record("2h", "heldout", details={"views": {"front-left-34": {"status": "accepted"}}})
    record(3, "base mesh", checks=[check("silhouette_iou_front", base + 0.05, 0.9, True)],
           **({} if old else {"mesh_source": mesh, "joints_source": "silhouette"}))
    st4_checks = [check("triangles_high", triangles, 24000, triangles <= 24000, rule="<="),
                  check("max_joint_gap_high", round(0.03 * gap_scale * len(poses), 4), 0.15, True, rule="<="),
                  check("rest_iou_with_covers_top", 0.6, 0.9, False, gate=False)]
    if "low" in lods:
        st4_checks.append(check("triangles_low", 5000, 6000, True, rule="<="))
    extra4 = {"body": {"lods": {lod: {"triangles": triangles if lod == "high" else 5000} for lod in lods}}}
    if not old:
        extra4.update(mesh_source=mesh, joints_source="silhouette",
                      profile={"name": profile, "lods": list(lods), "budgets": None})
    record(4, "game body", checks=st4_checks, **extra4)
    if not old:
        write_json(run / "03-joints.json", {"joints": {}})
    if mesh != "carve":
        (run / "03-source.glb").write_bytes(b"source mesh")
    (run / "03-mesh.glb").write_bytes(b"base mesh")
    body = run / "04-body"
    body.mkdir()
    (body / "high.glb").write_bytes(b"high body " + sketch + str(base).encode())
    (body / "atlas-high.png").write_bytes(b"atlas high")
    (body / "atlas-high-side-trust.png").write_bytes(b"atlas variant")
    if "low" in lods:
        (body / "low.glb").write_bytes(b"low body")
        (body / "atlas-low.png").write_bytes(b"atlas low")
    (body / "pose-renders").mkdir()
    (body / "pose-renders" / "rest.png").write_bytes(b"render")
    (body / "textured-vs-views.png").write_bytes(b"sheet")
    sections = {"units": "studs", "height": 5.5, "sections": [],
                "pose_test": {"ungated_poses": ["twist_turn"], "gaps_high": gaps(gap_scale, poses),
                              "gaps_low": gaps(gap_scale * 1.1, poses) if "low" in lods else None}}
    if not old:
        sections.update(mesh_source=mesh, joints_source="silhouette", profile=profile, code=stamp_code,
                        budgets=None)
    write_json(body / "sections.json", sections)
    if scores and not old:
        entry = scores_entry(name="carve" if mesh == "carve" else "hybrid", mesh=mesh, base=base,
                             triangles=triangles, triangles_low=5000 if "low" in lods else None)
        variant = scores_entry(name=entry["name"] + "-side-trust", mesh=mesh, base=base, variant_of=entry["name"])
        write_json(run / "05-scores.json", {
            "scored": "2026-10-28T20:00:00-07:00", "scorer": "herogen.blender.score", "code": stamp_code,
            "views": [{"view": v, "heldout": False} for v in VIEWS] + [{"view": "front-left-34", "heldout": True}],
            "bodies": [entry, variant]})
    return run


def hand_frozen(store: Path, sketch="sk", label="v0-carve", mesh="carve", files=None) -> Path:
    """A version frozen by hand before the tool existed (format 0), as the first base versions were."""
    d = store / sketch / label
    d.mkdir(parents=True)
    files = files or {"high.glb": b"hand high", "sections.json": json.dumps(
        {"pose_test": {"ungated_poses": ["twist_turn"], "gaps_high": gaps(2.0)}}).encode(),
        "stage-4.json": json.dumps({"stage": 4, "ok": True, "checks": [
            check("triangles_high", 23000, 24000, True), check("max_joint_gap_high", 0.12, 0.15, True)],
            "body": {"lods": {"high": {"triangles": 23000}, "low": {"triangles": 5600}}}}).encode()}
    for name, data in files.items():
        (d / name).write_bytes(data)
    write_json(d / "version.json", {
        "label": label, "milestone": "base", "mesh_source": mesh, "joints": "silhouette",
        "profile": "roblox-ingame budgets 24000/6000", "built": "2026-10-28T21:58:35-07:00",
        "code": "an unstamped build", "provenance": "reconstructed: stage records predate commit stamping",
        "sketch_run": sketch, "files_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in files.items()},
        "frozen": "2026-10-29", "immutable": True})
    return d


class TempStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="herogen-versions-")
        self.tmp = Path(self._tmp.name)
        self.store = self.tmp / "store"
        self.store.mkdir()
        self.runs = self.tmp / "runs"
        self.runs.mkdir()
        env = mock.patch.dict(os.environ, {S.ENV_STORE: ""})
        env.start()
        self.addCleanup(env.stop)

    def tearDown(self):
        S.remove_tree(self.tmp)
        self._tmp.cleanup()

    def freeze(self, run, label="m1-test", **kw):
        kw.setdefault("sketch", "sk")
        kw.setdefault("by", "tester")
        return S.freeze(run, label, self.store, **kw)

    def cli(self, *args) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main([str(a) for a in args])
        return code, out.getvalue(), err.getvalue()


class Freeze(TempStore):
    def test_freeze_holds_the_body_records_and_hashes(self):
        run = make_run(self.runs, mesh="sam3d-objects")
        meta = self.freeze(run, milestone="m1", note="first")
        d = self.store / "sk" / "m1-test"
        self.assertEqual(Path(meta["path"]), d)
        names = sorted(p.name for p in d.iterdir())
        self.assertEqual(names, sorted([
            "high.glb", "low.glb", "sections.json", "atlas-high.png", "atlas-high-side-trust.png", "atlas-low.png",
            "stage-0.json", "stage-1.json", "stage-2.json", "stage-2h.json", "stage-3.json", "stage-4.json",
            "03-joints.json", "03-source.glb", "05-scores.json", "version.json"]))
        v = json.loads((d / "version.json").read_text())
        origin = {n: (run / "04-body" / n if (run / "04-body" / n).is_file() else run / n)
                  for n in names if n != "version.json"}
        self.assertEqual(v["files_sha256"], {n: sha(p) for n, p in origin.items()})
        self.assertEqual(v["files_bytes"], {n: p.stat().st_size for n, p in origin.items()})
        self.assertEqual((v["format"], v["label"], v["sketch"], v["milestone"], v["note"], v["frozen_by"]),
                         (1, "m1-test", "sk", "m1", "first", "tester"))
        self.assertEqual((v["profile"], v["mesh_source"], v["joints_source"]),
                         ("roblox-ingame", "sam3d-objects", "silhouette"))
        self.assertEqual(v["code"], {"commit": COMMIT, "dirty": False, "consistent": True})
        self.assertEqual(set(v["code_records"]), {"stage-3.json", "stage-4.json", "sections.json", "05-scores.json"})
        self.assertFalse(v["allow_dirty"])
        self.assertEqual(v["built"], "2026-10-28T19:00:00-07:00")
        self.assertIsNone(v["source"]["run"])                       # the run is outside the store
        self.assertEqual((v["source"]["run_name"], v["source"]["body"]), ("r1", "04-body"))
        self.assertEqual(v["source"]["run_records_sha256"],
                         {p.name: sha(p) for p in sorted(run.glob("stage-*.json"))})
        self.assertEqual(v["inputs"]["sketch"], sha(run / "00-sketch.png"))
        self.assertEqual(v["inputs"]["views"], {x: sha(run / "views" / f"{x}.png") for x in (*VIEWS, "top")})
        self.assertEqual(v["inputs"]["masks"], {x: sha(run / "masks" / f"{x}.png") for x in (*VIEWS, "top")})
        self.assertEqual(v["inputs"]["heldout"], {"front-left-34": sha(run / "heldout" / "front-left-34.png")})
        self.assertEqual(v["inputs"]["heldout_masks"],
                         {"front-left-34": sha(run / "heldout" / "front-left-34-mask.png")})
        self.assertEqual(v["scores"], {"file": "05-scores.json", "body": "hybrid", "variants": ["hybrid-side-trust"]})
        self.assertEqual(v["missing"], {"files": [], "fields": []})
        self.assertTrue(v["immutable"])
        self.assertTrue(S.verify(d)["ok"])

    @unittest.skipIf(IS_ROOT, "permissions do not bind the superuser")
    def test_a_version_is_read_only(self):
        d = Path(self.freeze(make_run(self.runs))["path"])
        with self.assertRaises(PermissionError):
            (d / "high.glb").write_bytes(b"changed")
        with self.assertRaises(PermissionError):
            (d / "extra.txt").write_text("x")
        self.assertEqual(d.stat().st_mode & 0o777, 0o555)

    def test_an_existing_label_is_refused_and_kept(self):
        run = make_run(self.runs)
        d = Path(self.freeze(run)["path"])
        before = {p.name: p.read_bytes() for p in d.iterdir()}
        other = make_run(self.runs, "r2", base=0.5)
        with self.assertRaisesRegex(S.VersionError, "already exists; labels are write-once"):
            self.freeze(other)
        self.assertEqual({p.name: p.read_bytes() for p in d.iterdir()}, before)
        code, _, err = self.cli("freeze", "--run", other, "--label", "m1-test", "--store", self.store,
                                "--sketch", "sk")
        self.assertEqual(code, 2)
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertIn("already exists", err)

    def test_a_dirty_run_is_refused_unless_allowed(self):
        run = make_run(self.runs, code={"commit": COMMIT, "dirty": True})
        with self.assertRaisesRegex(S.VersionError, r"dirty: true in .*stage-4\.json.*--allow-dirty"):
            self.freeze(run)
        self.assertEqual(list((self.store / "sk").iterdir()) if (self.store / "sk").exists() else [], [])
        v = self.freeze(run, allow_dirty=True)
        self.assertTrue(v["allow_dirty"])
        self.assertEqual(v["code"]["dirty"], True)
        self.assertTrue(json.loads((Path(v["path"]) / "version.json").read_text())["allow_dirty"])

    def test_dirty_scores_alone_also_refuse(self):
        run = make_run(self.runs)
        s = json.loads((run / "05-scores.json").read_text())
        s["code"] = {"commit": "d" * 40, "dirty": True}
        write_json(run / "05-scores.json", s)
        with self.assertRaisesRegex(S.VersionError, r"dirty: true in 05-scores\.json"):
            self.freeze(run)
        v = self.freeze(run, allow_dirty=True)
        self.assertEqual(v["code"], {"commit": COMMIT, "dirty": True, "consistent": False})

    def test_unknown_dirty_state_is_recorded_not_refused(self):
        v = self.freeze(make_run(self.runs, code={"commit": COMMIT, "dirty": None}))
        self.assertEqual(v["code"], {"commit": COMMIT, "dirty": None, "consistent": True})

    def test_an_older_run_freezes_what_exists_and_names_what_is_missing(self):
        run = make_run(self.runs, old=True)
        with self.assertRaisesRegex(S.VersionError, "names the commit that built the body.*--allow-unknown-code"):
            self.freeze(run)
        self.assertFalse((self.store / "sk").exists())                # no empty sketch folder left behind
        v = self.freeze(run, allow_unknown_code=True)
        self.assertTrue(v["allow_unknown_code"])
        self.assertFalse(v["allow_dirty"])
        self.assertIsNone(v["code"])
        self.assertEqual(v["code_records"], {"sections.json": None, "stage-3.json": None, "stage-4.json": None})
        self.assertEqual(v["missing"]["files"], ["stage-2h.json", "03-joints.json", "05-scores.json"])
        self.assertEqual(v["missing"]["fields"], ["code", "profile", "mesh_source", "joints_source"])
        self.assertIsNone(v["scores"])
        self.assertTrue(S.verify(Path(v["path"]))["ok"])
        bare = make_run(self.runs, "r2", old=True, heldout=False, sketch=b"other")
        (bare / "00-sketch.png").unlink()
        v = self.freeze(bare, label="m1-bare", sketch="bare", allow_unknown_code=True)
        self.assertIn("inputs.sketch", v["missing"]["fields"])
        self.assertIn("inputs.heldout", v["missing"]["fields"])

    def test_only_scored_held_out_views_are_hashed(self):
        run = make_run(self.runs)
        (run / "heldout" / "front-left-34-raw-mask.png").write_bytes(b"a raw image's mask")
        (run / "heldout" / "front-right-30.png").write_bytes(b"no mask: not a scored view")
        v = self.freeze(run)
        self.assertEqual(set(v["inputs"]["heldout"]), {"front-left-34"})
        self.assertEqual(set(v["inputs"]["heldout_masks"]), {"front-left-34"})

    def test_a_profile_without_a_low_lod_does_not_miss_one(self):
        v = self.freeze(make_run(self.runs, profile="best", lods=("high",)))
        self.assertEqual(v["missing"], {"files": [], "fields": []})
        self.assertNotIn("low.glb", v["files_sha256"])

    def test_scores_without_this_body_are_reported(self):
        run = make_run(self.runs)
        s = json.loads((run / "05-scores.json").read_text())
        s["bodies"] = [b for b in s["bodies"] if b["glb"] != "04-body/high.glb"]
        write_json(run / "05-scores.json", s)
        v = self.freeze(run)
        self.assertIsNone(v["scores"])
        self.assertEqual(v["missing"]["fields"], ["scores"])

    def test_an_alternative_body_keeps_its_own_records(self):
        run = make_run(self.runs, mesh="sam3d-objects")
        alt = run / "04-body-carve"
        alt.mkdir()
        (alt / "high.glb").write_bytes(b"carve body")
        stamp = {"commit": "b" * 40, "dirty": False}
        write_json(alt / "stage-3.json", {"stage": 3, "ok": True, "checks": [], "mesh_source": "carve", "code": stamp})
        write_json(alt / "stage-4.json", {"stage": 4, "ok": True, "checks": [], "mesh_source": "carve", "code": stamp})
        write_json(alt / "stage-5.json", {"stage": 5, "ok": True, "checks": []})   # only in the body folder
        v = self.freeze(run, body="04-body-carve")
        self.assertEqual(v["files_sha256"]["stage-3.json"], sha(alt / "stage-3.json"))
        self.assertEqual(v["files_sha256"]["stage-4.json"], sha(alt / "stage-4.json"))
        self.assertNotIn("03-joints.json", v["files_sha256"])          # the run's other body's stage 3
        self.assertNotIn("03-source.glb", v["files_sha256"])
        self.assertNotIn("03-joints.json", v["missing"]["files"])     # its stage 3 kept none: not expected
        self.assertEqual(v["source"]["run_records_sha256"],             # the run's own records, not the body's
                         {p.name: sha(p) for p in sorted(run.glob("stage-*.json"))})
        self.assertEqual(v["source"]["from_body_folder"], ["stage-3.json", "stage-4.json", "stage-5.json"])
        self.assertEqual(v["mesh_source"], "carve")
        self.assertEqual(v["code"]["commit"], "b" * 40)               # the body's own records name its code
        self.assertFalse(v["code"]["consistent"])                    # the run's 05-scores.json names another
        code, out, err = self.cli("freeze", "--run", run, "--body", "04-body-carve", "--label", "m1-cli",
                                  "--store", self.store, "--by", "tester")    # sketch found by its hash
        self.assertEqual((code, err), (0, ""))
        cli = json.loads((self.store / "sk" / "m1-cli" / "version.json").read_text())
        self.assertEqual((cli["source"]["body"], cli["files_sha256"]["high.glb"]),
                         ("04-body-carve", sha(alt / "high.glb")))

    def test_refusals(self):
        run = make_run(self.runs)
        for kw, msg in ((dict(label="../up"), "label"), (dict(label="/abs"), "label"), (dict(label=".hidden"), "label"),
                        (dict(body="../04-body"), "body"), (dict(body="views"), "body"),
                        (dict(sketch="a/b"), "sketch")):
            label = kw.pop("label", "m1-x")
            with self.assertRaisesRegex(S.VersionError, msg):
                self.freeze(run, label=label, **kw)
        with self.assertRaisesRegex(S.VersionError, "not found"):
            self.freeze(self.runs / "nope")
        (run / "04-body" / "high.glb").unlink()
        with self.assertRaisesRegex(S.VersionError, "no body to freeze"):
            self.freeze(run)
        broken = make_run(self.runs, "r2")
        (broken / "stage-4.json").write_text("{not json")
        with self.assertRaisesRegex(S.VersionError, "stage-4.json could not be read"):
            self.freeze(broken)
        self.assertEqual([p.name for p in self.store.rglob("*")], [])

    def test_the_sketch_folder_is_found_by_the_sketch_hash(self):
        a = make_run(self.runs, "a")
        with self.assertRaisesRegex(S.VersionError, "name its folder with --sketch"):
            S.freeze(a, "m1-a", self.store, by="t")
        self.freeze(a, label="m1-a", sketch="hero")
        b = make_run(self.runs, "b", base=0.8)                         # same sketch, another build
        self.assertEqual(S.freeze(b, "m1-b", self.store, by="t")["sketch"], "hero")
        other = make_run(self.runs, "c", sketch=b"another sketch")
        with self.assertRaisesRegex(S.VersionError, "different sketch"):
            self.freeze(other, label="m1-c", sketch="hero")
        with self.assertRaisesRegex(S.VersionError, "name its folder with --sketch"):
            S.freeze(other, "m1-c", self.store, by="t")

    def test_the_store_comes_from_the_flag_or_the_environment(self):
        run = make_run(self.runs)
        with self.assertRaisesRegex(S.VersionError, "HEROGEN_VERSIONS"):
            S.freeze(run, "m1-x", None, sketch="sk")
        with mock.patch.dict(os.environ, {S.ENV_STORE: str(self.store)}):
            self.assertEqual(Path(S.freeze(run, "m1-x", None, sketch="sk", by="t")["path"]).parent.parent,
                             self.store)
            code, out, _ = self.cli("list")
            self.assertEqual(code, 0)
            self.assertIn("sk/m1-x", out)
        with self.assertRaisesRegex(S.VersionError, "is not a folder"):
            S.store_root(self.tmp / "missing")


class CodeStamps(TempStore):
    def test_a_stamp_of_the_wrong_shape_is_an_error_not_unknown(self):
        for i, (code, msg) in enumerate(((
                {"commit": COMMIT, "dirty": "true"}, r"code\.dirty is 'true'"),
                ({"commit": COMMIT, "dirty": 1}, r"code\.dirty is 1"),
                ({"commit": 123, "dirty": False}, r"code\.commit is 123"),
                ("abc", "code is str"))):
            run = make_run(self.runs, f"r{i}", code=code)
            with self.assertRaisesRegex(S.VersionError, msg):
                self.freeze(run, allow_dirty=True, allow_unknown_code=True)
        self.assertFalse((self.store / "sk").exists())

    def test_dirty_is_read_across_every_stamp(self):
        run = make_run(self.runs)                                     # stage 3, 4, sections, scores: clean
        r = json.loads((run / "stage-0.json").read_text())
        r["code"] = {"commit": COMMIT, "dirty": None}                  # one stamp that does not know
        write_json(run / "stage-0.json", r)
        self.assertEqual(self.freeze(run)["code"]["dirty"], None)      # not all false: unknown, not clean
        run = make_run(self.runs, "r2")
        self.assertEqual(self.freeze(run, label="m1-clean")["code"]["dirty"], False)

    def test_dirty_blocks_whatever_the_commit(self):
        run = make_run(self.runs, code={"commit": None, "dirty": True})
        with self.assertRaisesRegex(S.VersionError, "dirty: true in"):
            self.freeze(run, allow_unknown_code=True)
        v = self.freeze(run, allow_dirty=True, allow_unknown_code=True)
        self.assertEqual(v["code"], {"commit": None, "dirty": True, "consistent": True})
        self.assertIn("code.commit", v["missing"]["fields"])

    def test_a_dirty_stamp_in_any_record_blocks(self):
        run = make_run(self.runs)
        r = json.loads((run / "stage-0.json").read_text())
        r["code"] = {"commit": COMMIT, "dirty": True}
        write_json(run / "stage-0.json", r)
        with self.assertRaisesRegex(S.VersionError, r"dirty: true in stage-0\.json"):
            self.freeze(run)

    def test_unknown_code_needs_the_flag_even_when_only_the_scores_are_stamped(self):
        run = make_run(self.runs, stamp=False)
        s = json.loads((run / "05-scores.json").read_text())
        s["code"] = {"commit": COMMIT, "dirty": False}
        write_json(run / "05-scores.json", s)
        with self.assertRaisesRegex(S.VersionError, "--allow-unknown-code"):
            self.freeze(run)
        v = self.freeze(run, allow_unknown_code=True)
        self.assertEqual(v["code"], {"commit": None, "dirty": False, "consistent": True})
        self.assertTrue(v["allow_unknown_code"])

    def test_records_are_read_from_the_copies(self):
        run = make_run(self.runs)                                     # clean in the run
        real = S._copy_file

        def dirtying(src, dst):
            out = real(src, dst)
            if dst.name == "stage-4.json":                            # the copy is what the version holds
                rec = json.loads(dst.read_text())
                rec["code"] = {"commit": COMMIT, "dirty": True}
                dst.write_text(json.dumps(rec))
            return out

        with mock.patch.object(S, "_copy_file", dirtying):
            with self.assertRaisesRegex(S.VersionError, r"dirty: true in stage-4\.json"):
                self.freeze(run)


class Links(TempStore):
    def test_symbolic_links_in_the_run_are_refused(self):
        outside = self.tmp / "elsewhere"
        outside.mkdir()
        for i, (rel, msg) in enumerate((("04-body/high.glb", "04-body/high.glb"),
                                         ("stage-4.json", "stage-4.json"),
                                         ("views/front.png", "views/front.png"),
                                         ("heldout/front-left-34.png", "heldout/front-left-34.png"),
                                         ("04-body/atlas-high.png", "04-body/atlas-high.png"),
                                         ("04-body", "04-body"))):
            run = make_run(self.runs, f"r{i}")
            target = outside / f"{i}-{Path(rel).name}"
            os.rename(run / rel, target)
            os.symlink(target, run / rel)
            with self.assertRaisesRegex(S.VersionError, f"{re.escape(msg)} is or goes through a symbolic link|"
                                                         f"{re.escape(msg)} is a symbolic link"):
                self.freeze(run)
        self.assertFalse((self.store / "sk").exists())

    def test_copy_and_hash_never_follow_a_link(self):
        real = self.tmp / "real.bin"
        real.write_bytes(b"x")
        link = self.tmp / "link.bin"
        os.symlink(real, link)
        with self.assertRaises(OSError):
            S.sha256_file(link)
        with self.assertRaises(OSError):
            S._copy_file(link, self.tmp / "copy.bin")


class StoreSide(TempStore):
    def test_an_empty_claim_is_named_as_one(self):
        run = make_run(self.runs)
        (self.store / "sk" / "m1-test").mkdir(parents=True)            # a freeze killed after claiming
        with self.assertRaisesRegex(S.VersionError, "empty folder left by an interrupted freeze.*remove"):
            self.freeze(run)
        self.assertEqual(S.list_versions(self.store)["empty_claims"], ["sk/m1-test"])
        code, out, _ = self.cli("list", "--store", self.store)
        self.assertIn("empty claim from an interrupted freeze (not a version; safe to remove): sk/m1-test", out)
        (self.store / "sk" / "m1-test").rmdir()
        self.assertTrue(S.verify(Path(self.freeze(run)["path"]))["ok"])

    def test_store_folders_that_are_links_are_refused(self):
        run = make_run(self.runs)
        outside = self.tmp / "outside"
        outside.mkdir()
        os.symlink(outside, self.store / "sk")
        with self.assertRaisesRegex(S.VersionError, "sketch folder sk in the store is a symbolic link"):
            self.freeze(run)
        self.assertEqual(list(outside.iterdir()), [])                 # nothing written outside the store
        self.assertEqual(S.list_versions(self.store)["not_versions"], ["sk (a symbolic link, not read)"])
        (self.store / "sk").unlink()
        (self.store / "sk").mkdir()
        os.symlink(outside, self.store / "sk" / "m1-test")
        with self.assertRaisesRegex(S.VersionError, "sk/m1-test in the store is a symbolic link"):
            self.freeze(run)
        real = hand_frozen(self.store, sketch="real")
        os.symlink(real, self.store / "sk" / "v0-linked")
        with self.assertRaisesRegex(S.VersionError, "goes through a symbolic link"):
            S.resolve_version("sk/v0-linked", self.store)
        with self.assertRaisesRegex(S.VersionError, "not found"):
            S.resolve_version("v0-linked", self.store)

    def test_a_link_to_a_hidden_folder_is_still_hidden(self):
        partial = self.store / "sk" / f"{S.PARTIAL_PREFIX}m1-x-abc"
        partial.mkdir(parents=True)
        write_json(partial / "version.json", {"label": "m1-x", "files_sha256": {}})
        link = self.tmp / "looks-fine"
        os.symlink(partial, link)
        for call in (lambda: S.load_version(link), lambda: S.verify(link), lambda: S.resolve_version(str(link))):
            with self.assertRaisesRegex(S.VersionError, "hidden folder, not a version"):
                call()


class Text(TempStore):
    def test_text_that_names_a_path_is_refused(self):
        run = make_run(self.runs)
        for kw in (dict(note="see /srv/runs/hero"), dict(note="from ~/runs"), dict(note="by ~someone"),
                   dict(milestone="../m1"), dict(note="up a/../b"), dict(note=r"C:\\runs\\a"),
                   dict(note=r"share \\\\host\\x"), dict(note="(/opt/x)"), dict(note="path=/srv/x"),
                   dict(note="x=a/Users/me/y"), dict(note="under b/home/me"), dict(by="/root")):
            with self.assertRaisesRegex(S.VersionError, "names a path"):
                self.freeze(run, **kw)
        self.assertFalse((self.store / "sk").exists())
        plain = ["front / back ratio", "budgets 24000/6000, see m0-carve...", "A/B test", "and/or", "x / y / z",
                 "~ 5 studs", "a ~ b", "1:2 scale", "IoU 0.96 vs 0.97"]
        for text in plain:
            self.assertFalse(S.names_a_path(text), text)
        v = self.freeze(run, note="front / back ratio, budgets 24000/6000", milestone="m1 (profiles)")
        self.assertEqual(v["note"], "front / back ratio, budgets 24000/6000")

    def test_record_text_that_names_a_path_is_refused(self):
        run = make_run(self.runs)
        st4 = json.loads((run / "stage-4.json").read_text())
        st4["profile"]["name"] = "/opt/profiles/best"
        write_json(run / "stage-4.json", st4)
        with self.assertRaisesRegex(S.VersionError, "profile would record a path"):
            self.freeze(run)
        self.assertFalse((self.store / "sk").exists())


class Atomic(TempStore):
    def test_a_label_claimed_during_the_freeze_is_never_replaced(self):
        run = make_run(self.runs)
        dest = self.store / "sk" / "m1-test"
        real = S._fsync_dir

        def racing(path):
            if path.name.startswith(S.PARTIAL_PREFIX):                # just before the rename
                dest.mkdir(exist_ok=True)                             # another freeze takes the label
            return real(path)

        with mock.patch.object(S, "_fsync_dir", racing):
            with self.assertRaisesRegex(S.VersionError, "claimed by another freeze while this one ran"):
                self.freeze(run)
        self.assertEqual(list(dest.iterdir()), [])                    # theirs, untouched and not replaced
        self.assertEqual([p.name for p in (self.store / "sk").iterdir()], ["m1-test"])

    def test_a_filesystem_that_ignores_permissions_is_reported(self):
        run = make_run(self.runs)
        with mock.patch.object(S.Path, "chmod", lambda self, mode: None):
            v = self.freeze(run)
        self.assertEqual(len(v["warnings"]), 2)
        self.assertIn("still writable", v["warnings"][0])
        with mock.patch.object(S.Path, "chmod", lambda self, mode: None):
            code, _, err = self.cli("freeze", "--run", run, "--label", "m1-cli", "--store", self.store,
                                    "--by", "t")
        self.assertEqual(code, 0)
        self.assertIn("warning: the version's files are still writable", err)

    def test_an_interrupted_freeze_leaves_nothing_that_looks_complete(self):
        run = make_run(self.runs)
        real = S._copy_file
        calls = []

        def failing(src, dst):
            calls.append(src)
            if len(calls) == 3:
                raise KeyboardInterrupt
            return real(src, dst)

        with mock.patch.object(S, "_copy_file", failing):
            with self.assertRaises(KeyboardInterrupt):
                self.freeze(run)
        self.assertFalse((self.store / "sk").exists())
        self.assertEqual(S.list_versions(self.store)["versions"], [])
        self.assertTrue(S.verify(Path(self.freeze(run)["path"]))["ok"])   # the label is still free

    def test_a_failed_rename_leaves_no_version(self):
        run = make_run(self.runs)
        with mock.patch.object(S.os, "rename", side_effect=OSError(28, "No space left on device")):
            with self.assertRaises(OSError):
                self.freeze(run)
        self.assertFalse((self.store / "sk").exists())
        hand_frozen(self.store)                                       # a sketch folder that was there stays
        with mock.patch.object(S.os, "rename", side_effect=OSError(28, "No space left on device")):
            with self.assertRaises(OSError):
                self.freeze(run)
        self.assertEqual([p.name for p in (self.store / "sk").iterdir()], ["v0-carve"])

    def test_a_killed_freeze_is_ignored_and_reported(self):
        run = make_run(self.runs)
        partial = self.store / "sk" / f"{S.PARTIAL_PREFIX}m1-test-abc123"
        partial.mkdir(parents=True)
        (partial / "high.glb").write_bytes(b"half")
        (partial / "version.json").write_text("{}")
        listed = S.list_versions(self.store)
        self.assertEqual(listed["versions"], [])
        self.assertEqual(listed["unfinished"], [f"sk/{partial.name}"])
        with self.assertRaisesRegex(S.VersionError, "not found"):
            S.resolve_version("m1-test", self.store)
        with self.assertRaisesRegex(S.VersionError, "not found"):
            S.resolve_version(f"sk/{partial.name}", self.store)
        self.freeze(run)
        code, out, _ = self.cli("list", "--store", self.store)
        self.assertIn("sk/m1-test", out)
        self.assertIn("unfinished freeze", out)

    def test_copy_detects_a_changed_source(self):
        src = self.tmp / "src.bin"
        src.write_bytes(b"abc")
        real_open = open

        def opening(path, mode="r", *a, **kw):
            f = real_open(path, mode, *a, **kw)
            if Path(path) == src and "r" in mode:
                with real_open(src, "ab") as g:                   # another writer appends mid-copy
                    g.write(b"def")
            return f

        with mock.patch("builtins.open", opening):
            with self.assertRaisesRegex(S.VersionError, "changed while it was being frozen"):
                S._copy_file(src, self.tmp / "dst.bin")


class PathSafety(TempStore):
    def strings(self, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from self.strings(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from self.strings(v)
        elif isinstance(obj, str):
            yield obj

    def assert_no_path_leaves(self, v: dict, run: Path):
        text = json.dumps(v)
        for s in self.strings(v):
            self.assertFalse(os.path.isabs(s) or s.startswith(("/", "\\")) or re_drive(s), s)
            self.assertNotIn("..", s.replace("\\", "/").split("/"), s)
        for folder in (self.tmp, self.tmp.resolve(), run.resolve(), Path.home()):
            self.assertNotIn(str(folder), text)
        for name in v["files_sha256"]:
            self.assertTrue(S.safe_file_name(name), name)

    def test_a_run_outside_the_store_leaves_no_path_in_the_record(self):
        run = make_run(self.runs, mesh="sam3d-objects")
        d = Path(self.freeze(run)["path"])
        v = json.loads((d / "version.json").read_text())
        self.assertIsNone(v["source"]["run"])
        self.assertEqual(v["source"]["run_name"], "r1")
        self.assertEqual(set(v["source"]["run_records_sha256"]),
                         {"stage-0.json", "stage-1.json", "stage-2.json", "stage-2h.json", "stage-3.json",
                          "stage-4.json"})
        self.assertNotIn("source.run", v["missing"]["fields"])     # left out on purpose, not missing
        self.assert_no_path_leaves(v, run)

    def test_a_run_inside_the_store_is_recorded_relative_to_it(self):
        run = make_run(self.store / "runs-here", mesh="sam3d-objects")
        v = json.loads((Path(self.freeze(run)["path"]) / "version.json").read_text())
        self.assertEqual(v["source"]["run"], "runs-here/r1")
        self.assertEqual((self.store / v["source"]["run"]).resolve(), run.resolve())
        self.assert_no_path_leaves(v, run)

    def test_verify_never_follows_an_unsafe_name(self):
        d = hand_frozen(self.store)
        meta = json.loads((d / "version.json").read_text())
        outside = self.tmp / "outside.txt"
        outside.write_text("secret")
        meta["files_sha256"]["../../outside.txt"] = sha(outside)
        meta["files_sha256"][str(outside)] = sha(outside)
        write_json(d / "version.json", meta)
        rep = S.verify(d)
        self.assertFalse(rep["ok"])
        self.assertEqual(sorted(rep["unsafe"]), sorted(["../../outside.txt", str(outside)]))
        self.assertEqual(rep["checked"], 3)

    def test_version_references_stay_inside_the_store(self):
        hand_frozen(self.store)
        for ref in ("../store/sk/v0-carve", "sk/../sk/v0-carve", "sk/v0-carve/..", "a/b/c"):
            with self.assertRaises(S.VersionError):
                S.resolve_version(ref, self.store)
        self.assertEqual(S.resolve_version("sk/v0-carve", self.store), self.store / "sk" / "v0-carve")
        self.assertEqual(S.resolve_version("v0-carve", self.store), self.store / "sk" / "v0-carve")


def re_drive(s: str) -> bool:
    return len(s) > 2 and s[1] == ":" and s[2] in "\\/" and s[0].isalpha()


class Verify(TempStore):
    def test_verify_catches_a_changed_missing_or_extra_file(self):
        d = Path(self.freeze(make_run(self.runs))["path"])
        self.assertEqual(self.cli("verify", "--version", "sk/m1-test", "--store", self.store)[0], 0)
        S._make_writable(d)
        (d / "high.glb").write_bytes(b"tampered")
        (d / "atlas-low.png").unlink()
        (d / "stray.png").write_bytes(b"x")
        rep = S.verify(d)
        self.assertFalse(rep["ok"])
        self.assertEqual([m["file"] for m in rep["mismatched"]], ["high.glb"])
        self.assertEqual(rep["missing"], ["atlas-low.png"])
        self.assertEqual(rep["unlisted"], ["stray.png"])
        code, out, _ = self.cli("verify", "--version", d)
        self.assertEqual(code, 1)
        self.assertIn("changed: high.glb", out)
        self.assertIn("missing: atlas-low.png", out)
        self.assertIn("FAILED", out)
        code, out, _ = self.cli("verify", "--version", d, "--json")
        self.assertFalse(json.loads(out)["ok"])

    def test_an_extra_file_alone_fails(self):
        d = Path(self.freeze(make_run(self.runs))["path"])
        S._make_writable(d)
        (d / "stray.png").write_bytes(b"x")
        rep = S.verify(d)
        self.assertEqual((rep["ok"], rep["unlisted"], rep["checked"]), (False, ["stray.png"], 14))

    def test_a_linked_or_unreadable_file_is_reported_and_the_check_goes_on(self):
        d = Path(self.freeze(make_run(self.runs))["path"])
        S._make_writable(d)
        real = self.tmp / "high-copy.glb"
        real.write_bytes((d / "high.glb").read_bytes())
        (d / "high.glb").unlink()
        os.symlink(real, d / "high.glb")                              # same bytes, but not a regular file
        rep = S.verify(d)
        self.assertEqual(rep["not_regular"], ["high.glb"])
        self.assertFalse(rep["ok"])
        if not IS_ROOT:
            (d / "low.glb").chmod(0)
            rep = S.verify(d)
            self.assertEqual(rep["unreadable"], [{"file": "low.glb", "error": "Permission denied"}])
            self.assertEqual(rep["checked"], 12)                      # every other file still checked
            code, out, _ = self.cli("verify", "--version", d)
            self.assertEqual(code, 1)
            self.assertIn("unreadable: low.glb (Permission denied)", out)
            self.assertIn("not a regular file: high.glb", out)

    def test_a_malformed_hash_or_a_wrong_label_is_reported(self):
        d = hand_frozen(self.store)
        meta = json.loads((d / "version.json").read_text())
        meta["files_sha256"]["high.glb"] = 12
        write_json(d / "version.json", meta)
        with self.assertRaisesRegex(S.VersionError, r"files_sha256\['high.glb'\] is 12, not a SHA-256"):
            S.verify(d)
        self.assertEqual(self.cli("verify", "--version", d)[0], 2)
        e = hand_frozen(self.store, label="v0-other")
        meta = json.loads((e / "version.json").read_text())
        meta["label"] = "v0-carve"
        write_json(e / "version.json", meta)
        rep = S.verify(e)
        self.assertEqual((rep["ok"], rep["label_recorded"]), (False, "v0-carve"))
        listed = {v["label"]: v for v in S.list_versions(self.store)["versions"]}
        self.assertEqual(listed["v0-other"]["label_recorded"], "v0-carve")
        code, out, _ = self.cli("list", "--store", self.store)
        self.assertIn("label mismatch: sk/v0-other has version.json label 'v0-carve'", out)

    def test_an_unfinished_freeze_is_never_a_version(self):
        partial = self.store / "sk" / f"{S.PARTIAL_PREFIX}m1-x-abc"
        partial.mkdir(parents=True)
        (partial / "high.glb").write_bytes(b"h")
        write_json(partial / "version.json", {"label": "m1-x", "files_sha256": {"high.glb": sha(partial /
                                                                                                "high.glb")}})
        for call in (lambda: S.verify(partial), lambda: S.load_version(partial),
                     lambda: S.resolve_version(str(partial), self.store)):
            with self.assertRaisesRegex(S.VersionError, "hidden folder, not a version"):
                call()
        code, _, err = self.cli("verify", "--version", partial)
        self.assertEqual(code, 2)
        self.assertIn("hidden folder", err)

    def test_verify_reads_a_hand_frozen_version(self):
        d = hand_frozen(self.store)
        self.assertTrue(S.verify(d)["ok"])
        S._make_writable(d)
        (d / "stage-4.json").write_text("{}")
        self.assertEqual([m["file"] for m in S.verify(d)["mismatched"]], ["stage-4.json"])

    def test_list_reads_both_formats(self):
        hand_frozen(self.store)
        self.freeze(make_run(self.runs))
        (self.store / "sk" / "empty").mkdir()
        (self.store / "sk" / "loose").mkdir()
        (self.store / "sk" / "loose" / "high.glb").write_bytes(b"x")
        res = S.list_versions(self.store)
        by = {v["label"]: v for v in res["versions"]}
        self.assertEqual(set(by), {"v0-carve", "m1-test"})
        self.assertEqual(by["v0-carve"]["format"], 0)
        self.assertEqual(by["v0-carve"]["joints_source"], "silhouette")
        self.assertIsNone(by["v0-carve"]["commit"])
        self.assertEqual(by["m1-test"]["commit"], COMMIT)
        self.assertEqual(res["not_versions"], ["sk/loose"])
        self.assertEqual(res["empty_claims"], ["sk/empty"])
        self.assertEqual(S.list_versions(self.store, sketch="other")["versions"], [])
        code, out, _ = self.cli("list", "--store", self.store, "--json")
        self.assertEqual(len(json.loads(out)["versions"]), 2)


class Cli(TempStore):
    def test_errors_are_one_line_without_a_traceback(self):
        cases = [("freeze", "--run", self.runs / "none", "--label", "x", "--store", self.store),
                 ("freeze", "--run", self.runs),
                 ("verify", "--version", "nothing", "--store", self.store),
                 ("compare", "a", "b"),
                 ("list", "--store", self.tmp / "absent"),
                 ("bogus",),
                 ()]
        for args in cases:
            code, out, err = self.cli(*args)
            self.assertEqual(code, 2, args)
            self.assertEqual(len(err.strip().splitlines()), 1, (args, err))
            self.assertNotIn("Traceback", err)
            self.assertTrue(err.startswith("python3 -m herogen.versions: "), err)

    def test_unexpected_failures_are_one_line_too(self):
        with mock.patch.object(S, "list_versions", side_effect=RuntimeError("boom")):
            with mock.patch("herogen.versions.__main__.list_versions", side_effect=RuntimeError("boom")):
                code, _, err = self.cli("list", "--store", self.store)
        self.assertEqual(code, 2)
        self.assertEqual(err.strip(), "python3 -m herogen.versions: unexpected RuntimeError: boom")

    def test_freeze_prints_one_summary_line(self):
        run = make_run(self.runs, old=True)
        code, out, err = self.cli("freeze", "--run", run, "--label", "v0-old", "--store", self.store, "--sketch",
                                  "sk", "--by", "tester", "--milestone", "base", "--allow-unknown-code")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(len(out.strip().splitlines()), 1)
        self.assertIn("frozen sk/v0-old:", out)
        self.assertIn("code unknown (allowed)", out)
        self.assertIn("missing: stage-2h.json, 03-joints.json, 05-scores.json, code, profile", out)


if __name__ == "__main__":
    unittest.main()
