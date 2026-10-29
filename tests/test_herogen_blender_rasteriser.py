"""The pytorch3d stand-in for utils3d's rasteriser (herogen/blender/remote/objects.py), on one known triangle.

It needs torch and pytorch3d, which only the SAM 3D Objects image installs, so it skips elsewhere; the image's
own report (`modal run herogen/blender/remote/objects.py::build`) runs the same self-test inside the image.
"""
from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

OBJECTS = Path(__file__).resolve().parents[1] / "herogen" / "blender" / "remote" / "objects.py"
HAVE = all(importlib.util.find_spec(m) is not None for m in ("torch", "pytorch3d"))


def runner_functions() -> dict:
    """The rasteriser and its self-test, taken from the Modal app's source without importing modal."""
    tree = ast.parse(OBJECTS.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("pytorch3d_rasteriser", "rasteriser_selftest")]
    ns: dict = {}
    exec(compile(ast.Module(keep, []), str(OBJECTS), "exec"), ns)
    return ns


class TestRasteriserSource(unittest.TestCase):
    def test_the_self_test_covers_the_conventions_post_processing_reads(self):
        ns = runner_functions()
        self.assertIn("pytorch3d_rasteriser", ns)
        src = ast.get_source_segment(OBJECTS.read_text(), next(
            n for n in ast.parse(OBJECTS.read_text()).body if isinstance(n, ast.FunctionDef)
            and n.name == "rasteriser_selftest"))
        for prop in ("mask_row0_top", "face_id_one_based", "depth", "uv_row0_bottom", "uv_dr_zero"):
            self.assertIn(prop, src)
        self.assertIn('"rasteriser": rasteriser_selftest()', OBJECTS.read_text())   # the image report runs it


@unittest.skipUnless(HAVE, "torch and pytorch3d are installed only in the SAM 3D Objects image")
class TestRasteriser(unittest.TestCase):
    def test_conventions(self):
        result = runner_functions()["rasteriser_selftest"]()
        self.assertTrue(result["ok"], result)


if __name__ == "__main__":
    unittest.main()
