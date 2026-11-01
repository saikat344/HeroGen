"""The prompt registry: frozen versions load, never change, and are selected by config rather than call sites."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from herogen import prompts  # noqa: E402

REPO = Path(_HERE).parent
INSTALLED_CHECK = """
import sys
from pathlib import Path
from herogen import prompts
site = Path(sys.argv[1]).resolve()
assert Path(prompts.__file__).resolve().is_relative_to(site), prompts.__file__
names = prompts.available()
assert names, "no prompt versions installed"
for n in names:
    prompts.load(n)
prompts.selected("prompt_sketch")
cal = prompts.calibration(prompts.selected("same_character"))
assert cal is not None, "no calibration installed for the default same-character version"
print("versions found:", ", ".join(names))
"""

# Every frozen version and its content hash. A frozen version never changes: new text needs a new name, added here.
FROZEN = {
    "hero-sketch-v1": "ca48ac85c697a97b61865c45cf5886adaffa88f9e71851d0649923e568620624",
    "hero-sketch-v2": "baea6e678b205dc10dccefa48c1c19d5da394f01870fd1bd9dac8d075b27294c",
    "same-character-v1": "1842a33a49508ef5fc037a717da20b48c887fb03cde836d19929f556dc821cc9",
    "same-character-v2": "8f9332dedcf4a674d592dca9f59804eddae9239414ba94045995276ad1865aef",
    "turnaround-v1": "e91c9ed957eadec81d24a0bee2c3bbd0c94eaac97dd1afd09b36ec54d6b3f7d1",
    "turnaround-v2": "8b9bb3ebf0d533efe7a80036e266e92bfd996dc3f916cff9361d8de1166fc44b",
    "cover-redo-v1": "75a790168a42d41ec46216be0193efb3af6f60966a0a87381484d378264be570",
    "turnaround-redo-v1": "b9f6a99f4ea46a88e7e1957beebb538c9305a28ae2bb09bd4fe5a643744c6ebd",
}


class FrozenVersions(unittest.TestCase):
    def test_every_version_is_pinned(self):
        self.assertEqual(sorted(FROZEN), prompts.available())

    def test_frozen_hashes_unchanged(self):
        for name, sha in FROZEN.items():
            with self.subTest(name=name):
                v = prompts.load(name)
                self.assertEqual(v.sha256, sha, f"{name} changed; frozen versions need a new name")
                raw = json.loads((prompts.VERSIONS / f"{name}.json").read_text(encoding="utf-8"))
                self.assertEqual(raw["sha256"], sha)

    def test_edited_text_is_refused(self):
        src = json.loads((prompts.VERSIONS / "hero-sketch-v1.json").read_text(encoding="utf-8"))
        src["steps"][1]["prompt"] += " Extra words."
        with tempfile.TemporaryDirectory() as d:
            Path(d, "hero-sketch-v1.json").write_text(json.dumps(src), encoding="utf-8")
            with mock.patch.object(prompts, "VERSIONS", Path(d)):
                with self.assertRaisesRegex(prompts.RegistryError, "changed after it was frozen"):
                    prompts.load("hero-sketch-v1")

    def test_hash_covers_model_and_params(self):
        src = json.loads((prompts.VERSIONS / "hero-sketch-v1.json").read_text(encoding="utf-8"))
        base = prompts.content_hash(src)
        for path, value in ((("steps", 0, "model"), "other-model"), (("steps", 0, "params", "size"), "1024x1024"),
                            (("user_text", "max_chars"), 301)):
            d = json.loads(json.dumps(src))
            node = d
            for k in path[:-1]:
                node = node[k]
            node[path[-1]] = value
            self.assertNotEqual(prompts.content_hash(d), base, path)

    def test_hero_sketch_v1_shape(self):
        v = prompts.load("hero-sketch-v1")
        design, full = v.step("design"), v.step("full_body")
        self.assertEqual(design["model"], "muse-image-1.0")
        self.assertEqual(full["model"], "muse-image-1.0")
        slot = v.data["user_text"]
        self.assertEqual(design["template"].count(slot["slot"]), 1)
        self.assertIn(slot["open"] + slot["slot"] + slot["close"], design["template"])
        self.assertNotIn(slot["slot"], full["prompt"])  # the full-body edit never sees the user's text
        self.assertTrue(design["template"].endswith("Monochrome graphite only, no colour, no digital rendering, no text."))
        self.assertTrue(full["prompt"].startswith("Redraw this exact character as a full-body front view."))

    def test_hero_sketch_v2_only_adds_head_negatives(self):
        v1 = json.loads(json.dumps(prompts.load("hero-sketch-v1").data))
        v2 = json.loads(json.dumps(prompts.load("hero-sketch-v2").data))
        negative = " Constraints: no halo, no glow, no rings, circles or symbols around or behind the head."
        for before, after in zip(v1["steps"], v2["steps"]):
            key = "template" if "template" in before else "prompt"
            self.assertEqual(after[key], before[key] + negative)
            after[key] = before[key]
        v2["name"], v2["sha256"] = v1["name"], v1["sha256"]
        self.assertEqual(v2, v1)

    def test_turnaround_preserves_legacy_and_only_adds_gear_continuity(self):
        from herogen.pipeline.stages import SHEET_SIZE, TURNAROUND_PROMPT

        before = prompts.load("turnaround-v1").step("sheet")
        after = prompts.load("turnaround-v2").step("sheet")
        self.assertEqual(before["prompt"], TURNAROUND_PROMPT)
        self.assertEqual(before["params"], {"size": SHEET_SIZE, "n": 1, "output_format": "png"})
        extra = (" Keep every carried item and accessory, including bags and their straps, attached to the same "
                 "place on the body in all four views, including both profiles; show each item's visible "
                 "parts with natural occlusion.")
        self.assertEqual(after, {**before, "prompt": before["prompt"] + extra})


class Calibration(unittest.TestCase):
    def test_every_calibration_matches_its_frozen_version(self):
        files = sorted(prompts.CALIBRATION.glob("*.json"))
        self.assertTrue(files)
        for f in files:
            with self.subTest(file=f.name):
                v = prompts.load(f.stem)
                cal = prompts.calibration(v)
                t = cal["rating_set"]["true_pairs"]
                self.assertLessEqual(t["added_prop_listed"], t["answers"])
                self.assertLessEqual(t["same"], t["answers"])

    def test_v2_numbers(self):
        cal = prompts.calibration(prompts.load("same-character-v2"))
        self.assertEqual(cal["rating_set"]["true_pairs"]["added_prop_listed"], 15)
        self.assertEqual(cal["rating_set"]["true_pairs"]["answers"], 32)

    def test_none_without_a_file_and_refused_for_another_hash(self):
        self.assertIsNone(prompts.calibration(prompts.load("hero-sketch-v1")))
        src = json.loads((prompts.CALIBRATION / "same-character-v2.json").read_text(encoding="utf-8"))
        src["version_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as d:
            Path(d, "same-character-v2.json").write_text(json.dumps(src), encoding="utf-8")
            with mock.patch.object(prompts, "CALIBRATION", Path(d)):
                with self.assertRaisesRegex(prompts.RegistryError, "another version"):
                    prompts.calibration(prompts.load("same-character-v2"))


class Selection(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(prompts.selected("prompt_sketch").name, "hero-sketch-v2")
        self.assertEqual(prompts.selected("same_character").name, "same-character-v2")
        self.assertEqual(prompts.selected("turnaround").name, "turnaround-v1")
        with mock.patch.dict(os.environ, {"HEROGEN_TURNAROUND_VERSION": "turnaround-v2"}):
            self.assertEqual(prompts.selected("turnaround").name, "turnaround-v2")

    def test_env_override_and_explicit(self):
        with mock.patch.dict(os.environ, {"HEROGEN_PROMPT_SKETCH_VERSION": "no-such-v9"}):
            with self.assertRaisesRegex(prompts.RegistryError, "unknown prompt version"):
                prompts.selected("prompt_sketch")
            self.assertEqual(prompts.selected("prompt_sketch", "hero-sketch-v1").name, "hero-sketch-v1")

    def test_kind_mismatch(self):
        with self.assertRaisesRegex(prompts.RegistryError, "not 'prompt_sketch'"):
            prompts.selected("prompt_sketch", "same-character-v1")

    def test_bad_names(self):
        for bad in ("", "../x-v1", "Hero-v1", "hero"):
            with self.assertRaises(prompts.RegistryError):
                prompts.load(bad)

    def test_ref(self):
        v = prompts.load("hero-sketch-v1")
        self.assertEqual(v.ref(), {"name": "hero-sketch-v1", "sha256": FROZEN["hero-sketch-v1"]})


class InstalledPackageTest(unittest.TestCase):
    """`pip install` of the project ships the prompt versions, defaults and calibrations (package data)."""

    def test_installed_package_finds_its_prompt_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, site = Path(tmp) / "src", Path(tmp) / "site"
            shutil.copytree(REPO / "herogen", src / "herogen", ignore=shutil.ignore_patterns("__pycache__", "runs"))
            for name in ("pyproject.toml", "README.md", "LICENSE"):
                shutil.copy2(REPO / name, src / name)
            p = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                                "--disable-pip-version-check", "--target", str(site), str(src)],
                               capture_output=True, text=True, timeout=300, cwd=tmp)
            if p.returncode != 0 and re.search(r"No matching distribution|Could not find a version|"
                                               r"Failed to establish a new connection", p.stderr):
                self.skipTest("pip could not fetch the build backend (offline)")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            q = subprocess.run([sys.executable, "-c", INSTALLED_CHECK, str(site)], capture_output=True, text=True,
                               timeout=120, cwd=tmp, env={**os.environ, "PYTHONPATH": str(site)})
            self.assertEqual(q.returncode, 0, q.stderr[-2000:])
            self.assertIn("hero-sketch-v1", q.stdout)


if __name__ == "__main__":
    unittest.main()
