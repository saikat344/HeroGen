"""Front-view part masks (herogen/pipeline/parts.py): which SAM 3.1 instance is kept, how failures are recorded,
the files and record written, the `parts` command, and that stage 4's colour bake does not depend on them.

SAM is never called: a stand-in client answers with synthetic masks. Nothing generated here is written inside the
repository.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from herogen.blender import views

SIZE, PPS, CMID, VREF = 256, 160.0, 128.0, 250.0   # z = 0 at row 250
CENTRE, RADIUS = np.array([0.0, 0.0, 1.0]), 0.5


def disk() -> np.ndarray:
    """A round figure in a view: the silhouette of a sphere at CENTRE."""
    cols = (np.arange(SIZE) + 0.5 - CMID) / PPS
    rows = (VREF - (np.arange(SIZE) + 0.5)) / PPS
    u, v = np.meshgrid(cols, rows)
    return u ** 2 + (v - CENTRE[2]) ** 2 <= RADIUS ** 2


def face_mask() -> np.ndarray:
    cols = (np.arange(SIZE) + 0.5 - CMID) / PPS
    rows = (VREF - (np.arange(SIZE) + 0.5)) / PPS
    u, v = np.meshgrid(cols, rows)
    return disk() & (np.abs(u) < 0.3) & (v >= 0.8) & (v <= 1.2)


def sphere(n_lat: int = 12, n_lon: int = 24):
    """A closed UV sphere (outward winding) and one atlas cell per triangle."""
    verts = [CENTRE + [0, 0, RADIUS], CENTRE - [0, 0, RADIUS]]
    for i in range(1, n_lat):
        th = np.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * np.pi * j / n_lon
            verts.append(CENTRE + RADIUS * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]))
    verts = np.array(verts)

    def ring(i, j):
        return 2 + (i - 1) * n_lon + j % n_lon

    faces = []
    for j in range(n_lon):
        faces += [[0, ring(1, j), ring(1, j + 1)], [1, ring(n_lat - 1, j + 1), ring(n_lat - 1, j)]]
    for i in range(1, n_lat - 1):
        for j in range(n_lon):
            a, b, c, d = ring(i, j), ring(i, j + 1), ring(i + 1, j), ring(i + 1, j + 1)
            faces += [[a, c, d], [a, d, b]]
    faces = np.array(faces)
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum("ij,ij->i", n, tri.mean(axis=1) - CENTRE) < 0
    faces[flip] = faces[flip][:, ::-1]
    g = int(np.ceil(np.sqrt(len(faces))))
    cell, k = 1.0 / g, np.arange(len(faces))
    u0, v0, m = (k % g) * cell, (k // g) * cell, 0.15 * cell
    corner = np.stack([np.stack([u0 + m, v0 + m], 1), np.stack([u0 + cell - m, v0 + m], 1),
                       np.stack([u0 + m, v0 + cell - m], 1)], axis=1)
    return verts, faces, corner


class StandInClient:
    """Answers `segment(image, text)` from a dict; an exception in it is raised."""

    def __init__(self, answers):
        self.answers = answers

    def segment(self, image, text):
        a = self.answers[text]
        if isinstance(a, Exception):
            raise a
        return a


class TestPartMasks(unittest.TestCase):
    def setUp(self):
        from herogen.pipeline.client import ApiError, SamInstance
        self.SamInstance, self.ApiError = SamInstance, ApiError
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.figure = disk()

    def tearDown(self):
        self.tmp.cleanup()

    def inst(self, mask, conf):
        rows, cols = np.nonzero(mask)
        return self.SamInstance(mask=mask, box=(cols.min(), rows.min(), cols.max() + 1, rows.max() + 1),
                                confidence=conf)

    def run_folder(self, name="r"):
        from herogen.pipeline.runfolder import RunFolder
        run = RunFolder(self.root, name)
        (run.path / "views").mkdir(parents=True, exist_ok=True)
        (run.path / "masks").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (SIZE, SIZE), (200, 200, 200)).save(run.path / "views" / "front.png")
        views.save_mask(run.path / "masks" / "front.png", self.figure)
        run.write_json("views/meta.json", {"image_size": SIZE})
        return run

    def test_most_confident_instance_on_the_figure(self):
        from herogen.pipeline.parts import front_part_masks
        off = np.zeros((SIZE, SIZE), bool)
        off[0:10, 0:10] = True                       # off the figure: dropped whatever its confidence
        weak = face_mask() & (np.arange(SIZE)[None, :] < CMID)
        masks, info = front_part_masks(StandInClient({"face": [self.inst(off, 0.99), self.inst(weak, 0.5),
                                                               self.inst(face_mask(), 0.8)], "head": []}),
                                       Image.new("RGB", (SIZE, SIZE)), self.figure)
        np.testing.assert_array_equal(masks["face"], face_mask() & self.figure)
        self.assertEqual((info["face"]["confidence"], info["face"]["instances"]), (0.8, 3))
        self.assertNotIn("head", masks)
        self.assertIsNone(info["head"]["file"])

    def test_api_error_is_recorded_not_raised(self):
        from herogen.pipeline.parts import front_part_masks
        masks, info = front_part_masks(StandInClient({"face": self.ApiError("HTTP 500"), "head": []}),
                                       Image.new("RGB", (SIZE, SIZE)), self.figure)
        self.assertTrue(info["face"]["api_error"])
        self.assertIsNone(info["face"]["file"])
        self.assertEqual(masks, {})

    def test_replay_miss_is_raised(self):
        from herogen.pipeline.client import ReplayMiss
        from herogen.pipeline.parts import front_part_masks
        with self.assertRaises(ReplayMiss):
            front_part_masks(StandInClient({"face": ReplayMiss("no entry"), "head": []}),
                             Image.new("RGB", (SIZE, SIZE)), self.figure)

    def test_write_part_masks_files_and_record(self):
        from herogen.pipeline.parts import RECORD, write_part_masks
        run = self.run_folder()
        views.save_mask(run.path / "masks" / "front-head.png", self.figure)      # stale, from an earlier call
        info = write_part_masks(run, StandInClient({"face": [self.inst(face_mask(), 0.9)], "head": []}))
        self.assertEqual(RECORD, "masks/parts.json")
        self.assertEqual(run.read_json("views/meta.json"), {"image_size": SIZE})    # untouched
        self.assertEqual(run.read_json("masks/parts.json"), info)
        self.assertEqual(info["face"]["file"], "masks/front-face.png")
        np.testing.assert_array_equal(views.load_mask(run.path / "masks" / "front-face.png"), face_mask())
        self.assertFalse((run.path / "masks" / "front-head.png").exists())

    def test_write_part_masks_needs_stage_2(self):
        from herogen.pipeline.parts import write_part_masks
        from herogen.pipeline.runfolder import RunFolder
        with self.assertRaises(FileNotFoundError):
            write_part_masks(RunFolder(self.root, "empty"), StandInClient({}))

    def test_parts_command_reports_a_replay_miss_in_one_line(self):
        from herogen.pipeline.__main__ import main
        self.run_folder()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main(["parts", "--run-id", "r", "--runs", str(self.root)])
        self.assertEqual(code, 1)
        self.assertTrue(err.getvalue().startswith("parts: no cached SAM response"))
        self.assertEqual(len(err.getvalue().strip().splitlines()), 1)


class TestStage4IgnoresPartMasks(unittest.TestCase):
    """Stage 4's colour projection reads no part mask: the atlas is byte-identical with and without them."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name) / "run"
        (self.run / "views").mkdir(parents=True)
        (self.run / "masks").mkdir()
        colours = {"front": (230, 20, 20), "back": (20, 20, 230), "left": (20, 200, 20), "right": (200, 200, 20)}
        for name, colour in colours.items():
            img = np.zeros((SIZE, SIZE, 3), np.uint8)
            img[disk()] = colour
            img[face_mask()] = (250, 220, 180) if name == "front" else colour
            Image.fromarray(img).save(self.run / "views" / f"{name}.png")
            views.save_mask(self.run / "masks" / f"{name}.png", disk())
        (self.run / "views" / "meta.json").write_text(json.dumps({"pixels_per_unit": PPS, "height_units": 1.0,
                                                                   "image_size": SIZE}))

    def tearDown(self):
        self.tmp.cleanup()

    def bake(self):
        from herogen.blender import texture
        vs = views.load_views(self.run, 1.0)
        verts, faces, corner = sphere()
        atlas, stats, _ = texture.bake({"Head": SimpleNamespace(verts=verts, faces=faces)}, {"Head": corner},
                                       texture.observed_colour_views(vs, self.run), 64)
        return atlas.tobytes(), json.dumps(stats, sort_keys=True), sorted(vs.views), vs.notes

    def test_atlas_is_the_same_with_part_masks(self):
        before = self.bake()
        views.save_mask(self.run / "masks" / "front-face.png", face_mask())
        views.save_mask(self.run / "masks" / "front-head.png", disk())
        (self.run / "masks" / "parts.json").write_text(json.dumps(
            {"face": {"file": "masks/front-face.png"}, "head": {"file": "masks/front-head.png"}}))
        self.assertEqual(self.bake(), before)

    def test_stage4_code_names_no_part_mask(self):
        root = Path(__file__).resolve().parent.parent / "herogen" / "blender"
        for name in ("texture.py", "stage4.py"):
            text = (root / name).read_text()
            for ref in ("parts.json", "front-face", "front-head", "part_masks"):
                self.assertNotIn(ref, text, f"{name} reads {ref}")


if __name__ == "__main__":
    unittest.main()
