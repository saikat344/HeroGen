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
prompts.selected("same_character")
print("versions found:", ", ".join(names))
"""

# Every frozen version and its content hash. A frozen version never changes: new text needs a new name, added here.
FROZEN = {
    "hero-sketch-v1": "ca48ac85c697a97b61865c45cf5886adaffa88f9e71851d0649923e568620624",
    "same-character-v1": "1842a33a49508ef5fc037a717da20b48c887fb03cde836d19929f556dc821cc9",
    "same-character-v2": "8f9332dedcf4a674d592dca9f59804eddae9239414ba94045995276ad1865aef",
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


class Selection(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(prompts.selected("prompt_sketch").name, "hero-sketch-v1")
        self.assertEqual(prompts.selected("same_character").name, "same-character-v2")

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
    """`pip install` of the project ships the prompt versions and defaults (package data)."""

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
            self.assertIn("versions found: hero-sketch-v1", q.stdout)


if __name__ == "__main__":
    unittest.main()
