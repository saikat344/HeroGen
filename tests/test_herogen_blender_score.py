"""herogen.blender.score: silhouette IoU, colour error and SSIM, the held-out yaw search over any number of
held-out views, mesh sanity, body discovery and variants, and 05-scores.json, on synthetic box figures whose
silhouettes and colours are known exactly. No Blender, no model calls; everything is written to a temporary
folder.
"""
from __future__ import annotations

import io
import json
import math
import shutil
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from herogen.blender import glb as G
from herogen.blender import score as S
from herogen.blender.views import PROJ, save_mask
from herogen.pipeline.results import collect_scores, render_scores
from herogen.pipeline.runfolder import RunFolder

# Two triangles per side of a box's 8 corners, corner i = bits (x, y, z); `box` turns each one to face outward.
BOX_FACES = np.array([
    [0, 2, 3], [0, 3, 1],   # x = -1
    [4, 5, 7], [4, 7, 6],   # x = +1
    [0, 1, 5], [0, 5, 4],   # y = -1
    [2, 6, 7], [2, 7, 3],   # y = +1
    [0, 4, 6], [0, 6, 2],   # z = -1
    [1, 3, 7], [1, 7, 5],   # z = +1
])

# A figure that looks different from every side: a body and an "arm" sticking out to the character's left
# (+X) and forwards (-Y), higher up. In the front view the arm is beside the body, not in front of it.
BODY = ((0.0, 0.0, 1.0), (0.5, 0.3, 1.0))
ARM = ((0.95, -0.35, 1.45), (0.45, 0.12, 0.12))
RED, GREEN, BLUE = (220, 30, 30), (30, 200, 40), (30, 40, 220)


def box(centre, half):
    c, h = np.array(centre, float), np.array(half, float)
    corners = np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)], float) * 2 - 1
    faces = BOX_FACES.copy()
    p = corners[faces]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    inward = np.einsum("ij,ij->i", normal, p.mean(axis=1)) < 0
    faces[inward] = faces[inward][:, ::-1]
    return c + corners * h, faces


def analytic_mask(boxes, yaw_deg, size=256, pps=70.0, centre_col=120.0, feet_row=230.0):
    """Exact silhouette of axis-aligned boxes at ``yaw_deg``: each projects to a rectangle."""
    return analytic_layers(boxes, yaw_deg, size, pps, centre_col, feet_row).any(axis=0)


def analytic_layers(boxes, yaw_deg, size=256, pps=70.0, centre_col=120.0, feet_row=230.0):
    a = math.radians(yaw_deg)
    cols = np.arange(size) + 0.5
    rows = np.arange(size) + 0.5
    uu, vv = np.meshgrid((cols - centre_col) / pps, (feet_row - rows) / pps)
    out = []
    for c, h in boxes:
        uc = math.cos(a) * c[0] + math.sin(a) * c[1]
        hu = abs(math.cos(a)) * h[0] + abs(math.sin(a)) * h[1]
        out.append((np.abs(uu - uc) <= hu) & (np.abs(vv - c[2]) <= h[2]))
    return np.array(out)


def front_image(boxes=(BODY, ARM), colours=(RED, GREEN), size=256):
    """The front view in colour: each box flat in its colour (they do not overlap there), white elsewhere."""
    img = np.full((size, size, 3), 255, np.uint8)
    for layer, col in zip(analytic_layers(boxes, 0.0, size), colours):
        img[layer] = col
    return img


def png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def stripe_texture(colours=(RED, GREEN)) -> np.ndarray:
    tex = np.zeros((8, 8, 3), np.uint8)
    tex[:, :4] = colours[0]
    tex[:, 4:] = colours[1]
    return tex


def write_coloured_body(path: Path, boxes=(BODY, ARM), texture=None) -> Path:
    """Boxes textured flat: box i samples column (i + 0.5) / len(boxes) of a striped texture."""
    texture = stripe_texture() if texture is None else texture
    meshes = []
    for i, (c, h) in enumerate(boxes):
        v, f = box(c, h)
        meshes.append((f"part{i}", v, f, np.tile([(i + 0.5) / len(boxes), 0.5], (len(v), 1))))
    return G.write_glb(path, meshes, texture_png=png(texture))


def make_run(root: Path, yaw_heldout: float = 40.0, boxes=(BODY, ARM), extra_heldout: dict | None = None) -> Path:
    run = root / "run"
    (run / "masks").mkdir(parents=True)
    (run / "views").mkdir()
    (run / "heldout").mkdir()
    for v, yaw in S.CONTRACT_YAW.items():
        save_mask(run / "masks" / f"{v}.png", analytic_mask(boxes, yaw))
    Image.fromarray(front_image(boxes)).save(run / "views" / "front.png")
    held = {S.HELDOUT_VIEW: yaw_heldout, **(extra_heldout or {})}
    for name, yaw in held.items():
        m = analytic_mask(boxes, yaw, centre_col=140.0, feet_row=240.0)
        save_mask(run / "heldout" / f"{name}-mask.png", m)
        Image.fromarray(np.where(m[..., None], 128, 255).astype(np.uint8).repeat(3, axis=2)).save(
            run / "heldout" / f"{name}.png")
    return run


def accept_heldout(run: Path) -> None:
    """A stage 2h record that accepts `HELDOUT_VIEW` (its single-view form, which the results page reads)."""
    (run / "stage-2h.json").write_text(json.dumps({"stage": "2h", "ok": True,
                                                   "outputs": [f"heldout/{S.HELDOUT_VIEW}.png"],
                                                   "details": {"view": S.HELDOUT_VIEW}}))


def write_body(path: Path, boxes) -> Path:
    return G.write_glb(path, [(f"part{i}", *box(c, h)) for i, (c, h) in enumerate(boxes)])


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="herogen-score-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class Projection(unittest.TestCase):
    def test_contract_yaws_match_the_stage3_cameras(self):
        pts = np.random.default_rng(0).normal(size=(50, 3))
        for v, yaw in S.CONTRACT_YAW.items():
            np.testing.assert_allclose(S.project(pts, yaw), pts @ PROJ[v].T, atol=1e-12, err_msg=v)

    def test_front_left_view_turns_the_figure_toward_image_left(self):
        # Turned 45 degrees toward the image's left edge: the face (-Y) points image left, and the character's
        # left (+X) stays on the image right as in the front view.
        self.assertLess(S.project(np.array([[0.0, -1.0, 0.0]]), 45.0)[0, 0], 0)
        self.assertGreater(S.project(np.array([[1.0, 0.0, 0.0]]), 45.0)[0, 0], 0)

    def test_depth_is_nearest_toward_the_camera(self):
        # front camera on -Y: the face (-Y) is nearer than the back; left camera on +X: +X is nearer
        self.assertLess(S.view_depth(np.array([[0.0, -1.0, 0.0]]), 0.0)[0], 0)
        self.assertLess(S.view_depth(np.array([[1.0, 0.0, 0.0]]), 90.0)[0], 0)

    def test_heldout_ranges_follow_the_name(self):
        self.assertEqual(S.heldout_yaw("front-left-34"), ((15.0, 75.0), 45.0))       # the older 3/4 view
        self.assertEqual(S.heldout_yaw("front-right-34"), ((-75.0, -15.0), -45.0))
        self.assertEqual(S.heldout_yaw("front-left-30"), ((5.0, 55.0), 30.0))        # named by target yaw
        self.assertEqual(S.heldout_yaw("front-left-60"), ((35.0, 85.0), 60.0))
        self.assertEqual(S.heldout_yaw("front-right-60"), ((-85.0, -35.0), -60.0))
        self.assertEqual(S.heldout_yaw("front-left-80"), ((55.0, 89.0), 80.0))
        self.assertEqual(S.heldout_yaw("three-quarter-back"), (S.OTHER_YAW_RANGE, None))
        self.assertEqual(S.heldout_yaw("front-left-120"), (S.OTHER_YAW_RANGE, None))


class Rasterize(unittest.TestCase):
    def test_large_triangle_area(self):
        xy = np.array([[10.0, 10.0], [190.0, 10.0], [10.0, 150.0]])
        m = S.rasterize(xy, np.array([[0, 1, 2]]), (200, 200))
        self.assertAlmostEqual(m.sum() / (0.5 * 180 * 140), 1.0, delta=0.02)

    def test_dense_sub_pixel_mesh_leaves_no_holes(self):
        n = 200   # a 40 x 40 px square cut into 80,000 triangles a fifth of a pixel across
        g = np.linspace(0, 40, n + 1)
        xs, ys = np.meshgrid(g + 5, g + 5)
        xy = np.stack([xs.ravel(), ys.ravel()], axis=1)
        i = np.arange(n)[:, None] * (n + 1) + np.arange(n)[None, :]
        q = np.stack([i, i + 1, i + n + 2, i + n + 1], axis=-1).reshape(-1, 4)
        faces = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        m = S.rasterize(xy, faces, (60, 60))
        self.assertEqual(int(m.sum()), 40 * 40)
        self.assertTrue(m[5:45, 5:45].all())

    def test_both_windings_fill(self):
        xy = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]])
        a = S.rasterize(xy, np.array([[0, 1, 2]]), (30, 30))
        b = S.rasterize(xy, np.array([[0, 2, 1]]), (30, 30))
        self.assertTrue((a == b).all() and a.any())


class HeldoutYaw(Tmp):
    def test_finds_the_drawn_angle_and_scores_the_right_body_higher(self):
        run = make_run(self.tmp, yaw_heldout=40.0)
        right = S.score_body(run, "right", write_body(self.tmp / "right.glb", (BODY, ARM)),
                             seen=set(S.TURNAROUND), budget=1000)
        h = right["views"][S.HELDOUT_VIEW]
        self.assertAlmostEqual(h["yaw_deg"], 40.0, delta=1.0)
        self.assertGreater(h["iou"], 0.97)
        self.assertAlmostEqual(h["scale"], 1.0, delta=0.02)
        self.assertFalse(h["seen"])
        self.assertEqual(h["yaw_range"], [15.0, 75.0])
        self.assertFalse(h["yaw_at_range_edge"])
        searched = [y for y, _ in h["yaw_search"]]
        self.assertEqual(len(searched), len(set(searched)))      # each yaw rendered once
        self.assertIn([h["yaw_deg"], h["iou"]], h["yaw_search"])
        self.assertIn("45", h["at_fixed_yaw"])
        self.assertLess(h["at_fixed_yaw"]["45"]["iou"], h["iou"])
        for v in S.TURNAROUND:
            self.assertGreater(right["views"][v]["iou"], 0.97, v)
            self.assertTrue(right["views"][v]["seen"])
        # A body without the arm, and one with the arm on the wrong side, fit the held-out view worse.
        no_arm = S.score_body(run, "no-arm", write_body(self.tmp / "a.glb", (BODY,)), seen=set(), budget=1000)
        mirrored = ((-ARM[0][0], ARM[0][1], ARM[0][2]), ARM[1])
        wrong = S.score_body(run, "wrong-side", write_body(self.tmp / "b.glb", (BODY, mirrored)), seen=set(),
                             budget=1000)
        for other in (no_arm, wrong):
            self.assertLess(other["views"][S.HELDOUT_VIEW]["iou"], h["iou"] - 0.05, other["name"])

    def test_every_heldout_view_is_found_and_searched_on_its_own_side(self):
        run = make_run(self.tmp, yaw_heldout=40.0, extra_heldout={"front-right-34": -35.0})
        (run / "heldout" / "front-left-34-raw.png").write_bytes((run / "heldout" / "front-left-34.png").read_bytes())
        views = S.load_score_views(run)
        self.assertEqual([v.name for v in views if v.heldout], ["front-left-34", "front-right-34"])  # no -raw
        e = S.score_body(run, "b", write_body(self.tmp / "b.glb", (BODY, ARM)), views=views)
        right = e["views"]["front-right-34"]
        self.assertEqual(right["yaw_range"], [-75.0, -15.0])
        self.assertAlmostEqual(right["yaw_deg"], -35.0, delta=1.0)
        self.assertGreater(right["iou"], 0.97)
        self.assertIn("-45", right["at_fixed_yaw"])
        override = S.load_score_views(run, {"front-right-34": (-40.0, -38.0)})
        e2 = S.score_body(run, "c", write_body(self.tmp / "c.glb", (BODY, ARM)), views=override)
        self.assertTrue(e2["views"]["front-right-34"]["yaw_at_range_edge"])

    def test_generated_views_by_target_angle_skip_review_and_rejected(self):
        run = make_run(self.tmp, yaw_heldout=40.0,
                       extra_heldout={"front-left-30": 31.0, "front-left-60": 58.0, "front-right-30": -29.0})
        held = run / "heldout"
        for sub in ("review", "rejected"):            # views that must never be scored
            (held / sub).mkdir()
            for f in ("front-right-60.png", "front-right-60-mask.png"):
                (held / sub / f).write_bytes((held / "front-left-30.png").read_bytes())
        (held / "pose-reference.png").write_bytes((held / "front-left-30.png").read_bytes())   # no mask: input
        (run / "stage-2h.json").write_text(json.dumps({"stage": "2h", "ok": True, "details": {
            "reference": "generated", "views": {"front-left-30": {"status": "accepted", "found_yaw_deg": 31.0,
                                                                  "yaw_window_deg": [25.0, 35.0]}}}}))
        views = S.load_score_views(run)
        self.assertEqual([v.name for v in views if v.heldout],
                         ["front-left-30", "front-left-34", "front-left-60", "front-right-30"])
        e = S.score_body(run, "b", write_body(self.tmp / "b.glb", (BODY, ARM)), views=views)
        for name, target, drawn in (("front-left-30", 30, 31.0), ("front-left-60", 60, 58.0),
                                    ("front-right-30", -30, -29.0)):
            row = e["views"][name]
            self.assertAlmostEqual(row["yaw_deg"], drawn, delta=1.0, msg=name)
            self.assertGreater(row["iou"], 0.97, name)
            self.assertIn(f"{target:g}", row["at_fixed_yaw"])
            self.assertEqual((row["reference"], row["measures"]), ("generated", "consistency"))
            self.assertFalse(row["seen"])
        self.assertEqual(e["views"]["front-left-30"]["stage2h"]["found_yaw_deg"], 31.0)
        self.assertNotIn("stage2h", e["views"]["front-left-60"])
        path = S.write_scores(run, [e], views)
        listed = {v["view"]: v for v in json.loads(path.read_text())["views"]}
        self.assertEqual(listed["front-left-30"]["measures"], "consistency")
        self.assertNotIn("front-right-60", listed)

    def test_heldout_without_a_record_has_no_reference_label(self):
        run = make_run(self.tmp)
        row = S.score_body(run, "b", write_body(self.tmp / "b.glb", (BODY, ARM)))["views"][S.HELDOUT_VIEW]
        self.assertIsNone(row["reference"])
        self.assertIsNone(row["measures"])

    def test_writes_renders_silhouettes_and_diffs(self):
        run = make_run(self.tmp)
        S.score_body(run, "b", write_body(self.tmp / "b.glb", (BODY, ARM)), seen=set(), budget=1000)
        for v in S.TURNAROUND + (S.HELDOUT_VIEW,):
            for suffix in ("", "-silhouette", "-diff"):
                self.assertTrue((run / "05-renders" / "b" / f"{v}{suffix}.png").exists(), v + suffix)

    def test_refuses_unsafe_names_and_a_seen_heldout(self):
        run = make_run(self.tmp)
        body = write_body(self.tmp / "b.glb", (BODY,))
        with self.assertRaises(ValueError):
            S.score_body(run, "../x", body, seen=set(), budget=1)
        with self.assertRaises(ValueError):
            S.score_body(run, "b", body, seen={S.HELDOUT_VIEW}, budget=1)
        with self.assertRaises(ValueError):
            S.score_body(run, "b", body, frame="other")


class Frames(Tmp):
    def test_contract_frame_keeps_the_origin_and_fit_uses_the_bounding_box(self):
        run = make_run(self.tmp)
        shifted = ((BODY[0][0] + 0.3, BODY[0][1], BODY[0][2]), BODY[1]), ((ARM[0][0] + 0.3,) + ARM[0][1:], ARM[1])
        p = write_body(self.tmp / "shifted.glb", shifted)
        fit = S.score_body(run, "fit", p, frame="fit")
        self.assertGreater(fit["views"]["front"]["iou"], 0.97)
        self.assertEqual(fit["frame"], "fit")
        # In the contract frame the origin is the figure's: a body built 0.3 units off centre scores lower...
        off = S.score_body(run, "off", p, frame="contract", height=2.0)
        self.assertLess(off["views"]["front"]["iou"], 0.8)
        # ...and one built on it scores the same as the fit (X/Y centred on the figure's box, feet at 0).
        centred = tuple(((c[0] - 0.45, c[1] + 0.085, c[2]), h) for c, h in (BODY, ARM))
        on = S.score_body(run, "on", write_body(self.tmp / "on.glb", centred), frame="contract", height=2.0)
        for v in S.TURNAROUND:
            self.assertGreater(on["views"][v]["iou"], 0.97, v)
        self.assertAlmostEqual(on["pixels_per_unit"], 70.0, delta=0.5)


class ExactRegistration(Tmp):
    """Exact silhouettes of an exact body: any misregistration, even one pixel, shows as IoU below 1."""

    def test_turnaround_views_register_to_the_pixel(self):
        run = make_run(self.tmp)
        fit = S.score_body(run, "fit", write_body(self.tmp / "fit.glb", (BODY, ARM)), frame="fit")
        centred = tuple(((c[0] - 0.45, c[1] + 0.085, c[2]), h) for c, h in (BODY, ARM))
        on = S.score_body(run, "on", write_body(self.tmp / "on.glb", centred), frame="contract", height=2.0)
        for v in S.TURNAROUND:
            self.assertEqual(fit["views"][v]["iou"], 1.0, v)
            self.assertEqual(on["views"][v]["iou"], 1.0, v)

    def test_a_transparent_card_does_not_misplace_the_body(self):
        """The same exact body with a fully transparent alphaMode MASK card above its head still registers to
        the pixel: the scorer places a body by its visible surface, not by every vertex."""
        tex = np.zeros((8, 8, 4), np.uint8)
        tex[..., :3] = 128
        tex[:, 4:, 3] = 255
        meshes = [(f"part{i}", *box(c, h), np.tile([0.75, 0.5], (8, 1))) for i, (c, h) in enumerate((BODY, ARM))]
        top = max(c[2] + h[2] for c, h in (BODY, ARM))
        cv = np.array([[-0.6, -0.3, top + 0.1], [0.6, -0.3, top + 0.1], [0.6, -0.3, top + 0.9], [-0.6, -0.3, top + 0.9]])
        meshes.append(("card", cv, np.array([[0, 1, 2], [0, 2, 3]]), np.tile([0.25, 0.5], (4, 1))))
        p = G.write_glb(self.tmp / "carded.glb", meshes, texture_png=png(tex), alpha_mode="MASK")
        run = make_run(self.tmp)
        r = S.score_body(run, "carded", p, frame="fit")
        for v in (*S.TURNAROUND, S.HELDOUT_VIEW):
            self.assertEqual(r["views"][v]["iou"], 1.0, v)

    def test_heldout_view_registers_and_the_fine_search_finds_half_degrees(self):
        # 40.5 degrees is off the 2-degree coarse grid (15, 17, ... 75): only the 0.5-degree search reaches it
        run = make_run(self.tmp, yaw_heldout=40.5)
        h = S.score_body(run, "b", write_body(self.tmp / "b.glb", (BODY, ARM)))["views"][S.HELDOUT_VIEW]
        self.assertEqual(h["yaw_deg"], 40.5)
        self.assertEqual(h["iou"], 1.0)


def card(x0, x1, z0, z1, y=-0.6):
    """A flat quad facing the front camera (-Y), with texture coordinates across it (u left to right)."""
    v = np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    return v, f, uv


def half_alpha_texture(right_alpha: int = 64) -> np.ndarray:
    """Red, fully opaque on its left half; on its right half alpha `right_alpha` (below a 0.5 cutoff)."""
    tex = np.zeros((8, 8, 4), np.uint8)
    tex[..., :3] = RED
    tex[:, :4, 3] = 255
    tex[:, 4:, 3] = right_alpha
    return tex


class AlphaMask(Tmp):
    """glTF alphaMode MASK: the shape is the alpha-clipped surface; OPAQUE ignores texture alpha."""

    def render(self, path):
        glb = G.load_glb(path)
        verts, faces = glb.merged()
        xy = np.stack([(verts[:, 0] + 1.0) * 50.0, (2.5 - verts[:, 2]) * 50.0], axis=1)      # 50 px per unit
        clip = S.alpha_clip(glb, faces)
        cover = S.coverage(xy, faces, (128, 128), clip)
        colour = S.body_colour(glb, faces)
        img, m = S.colour_render(colour, xy, S.view_depth(verts, 0.0), faces, (128, 128), clip)
        return clip, cover, img, m

    def test_a_masked_card_keeps_only_its_opaque_half(self):
        v, f, uv = card(-0.5, 0.5, 1.0, 2.0)
        p = G.write_glb(self.tmp / "card.glb", [("card", v, f, uv)], texture_png=png(half_alpha_texture()),
                        alpha_mode="MASK", alpha_cutoff=0.5)
        clip, cover, img, m = self.render(p)
        self.assertIsNotNone(clip)
        self.assertEqual(int(clip.mask_faces.sum()), 2)
        cols = np.nonzero(cover.any(axis=0))[0]
        # x -0.5 .. about 0: the left (opaque) half; the edge falls where the bilinearly sampled alpha crosses
        # the cutoff, within one texel (6.25 px) of the middle
        self.assertEqual(cols.min(), 25)
        self.assertLessEqual(abs(int(cols.max()) - 50), 4)
        self.assertLessEqual(abs(int(cover.sum()) - 25 * 50), 4 * 50)
        self.assertTrue(np.array_equal(cover, m))                       # the colour render covers the same pixels
        self.assertTrue(np.all(img[m][:, 0] > 200))                     # and shows the card's red

    def test_a_transparent_card_does_not_move_or_scale_the_body(self):
        """An opaque 2 x 2 box with a fully transparent card above it is placed by the box alone: in a held-out
        fit, a turnaround registration (both frames) and the scorer, the silhouette matches the box's mask."""
        import io

        from PIL import Image

        def rgba_png(a):
            b = io.BytesIO()
            Image.fromarray(a.astype(np.uint8), "RGBA").save(b, format="PNG")
            return b.getvalue()

        quad = np.array([[0, 1, 2], [0, 2, 3]])
        box = np.array([[-1, 0, 0], [1, 0, 0], [1, 0, 2], [-1, 0, 2]], float)
        card_v = np.array([[-1, -0.01, 2], [1, -0.01, 2], [1, -0.01, 3], [-1, -0.01, 3]], float)
        tex = np.zeros((8, 8, 4))
        tex[..., :3] = 128
        tex[:, 4:, 3] = 255                                       # right half opaque, left half alpha 0
        p = self.tmp / "carded.glb"
        G.write_glb(p, [("body", box, quad, np.full((4, 2), [0.75, 0.5])),
                        ("hair_card", card_v, quad, np.full((4, 2), [0.25, 0.5]))],
                    texture_png=rgba_png(tex), alpha_mode="MASK")
        g = G.load_glb(p)
        verts, faces = g.merged()
        clip = S.alpha_clip(g, faces)
        mask = np.zeros((64, 64), bool)
        mask[8:56, 8:56] = True
        sil, _ = S.silhouette(verts, faces, 0.0, mask, clip)
        self.assertGreater(S.iou(sil, mask), 0.98)
        ext = S.visible_points(verts, faces, clip)
        self.assertAlmostEqual(float(ext[:, 2].max()), 2.0)       # the card adds nothing to the extent
        view = types.SimpleNamespace(name="front", mask=mask, heldout=False, yaw=0.0)
        for frame in S.FRAMES:
            reg = S.registration(verts, [view], frame, None, ext)
            self.assertAlmostEqual(reg["height"], 2.0, msg=frame)
            cover = S.coverage(S.turnaround_xy(verts, view, reg), faces, mask.shape, clip)
            self.assertGreater(S.iou(cover, mask), 0.98, frame)
        # without the clip the card counts, as it renders under OPAQUE
        self.assertAlmostEqual(S.registration(verts, [view], "fit", None)["height"], 3.0)

    def test_a_clipped_point_shows_the_surface_behind_it(self):
        v, f, uv = card(-0.5, 0.5, 1.0, 2.0)
        bv, bf = box((0.0, 0.0, 1.5), (0.5, 0.2, 0.5))                  # white (no material), behind the card
        p = G.write_glb(self.tmp / "both.glb", [("box", bv, bf), ("card", v, f, uv)],
                        texture_png=png(half_alpha_texture()), alpha_mode="MASK")
        _, cover, img, m = self.render(p)
        left, right = img[50, 35], img[50, 65]                          # same row, the card's two halves
        self.assertGreater(left[0], 200)
        self.assertLess(left[1], 80)                                     # red: the card
        self.assertTrue(np.all(right > 240))                             # white: the box seen through the clip
        self.assertTrue(m[50, 65] and cover[50, 65])

    def test_an_opaque_material_ignores_texture_alpha(self):
        v, f, uv = card(-0.5, 0.5, 1.0, 2.0)
        p = G.write_glb(self.tmp / "opaque.glb", [("card", v, f, uv)], texture_png=png(half_alpha_texture(0)))
        clip, cover, img, m = self.render(p)
        self.assertIsNone(clip)
        self.assertEqual(int(cover.sum()), 50 * 50)                     # the whole card, alpha 0 or not
        self.assertEqual(G.load_glb(p).materials[0].alpha_mode, "OPAQUE")

    def test_scores_use_the_clipped_shape(self):
        run = make_run(self.tmp)
        v, f, uv = card(-0.5, 0.5, 0.0, 2.0, y=0.0)
        masked = G.write_glb(self.tmp / "m.glb", [("card", v, f, uv)], texture_png=png(half_alpha_texture()),
                             alpha_mode="MASK")
        opaque = G.write_glb(self.tmp / "o.glb", [("card", v, f, uv)], texture_png=png(half_alpha_texture()))
        a = S.score_body(run, "m", masked, frame="contract", height=2.0)
        b = S.score_body(run, "o", opaque, frame="contract", height=2.0)
        self.assertEqual(a["alpha_clip"]["mask_faces"], 2)
        self.assertIsNone(b["alpha_clip"])
        self.assertNotEqual(a["views"]["front"]["iou"], b["views"]["front"]["iou"])
        sil = np.asarray(Image.open(run / "05-renders" / "m" / "front-silhouette.png").convert("L")) > 127
        full = np.asarray(Image.open(run / "05-renders" / "o" / "front-silhouette.png").convert("L")) > 127
        self.assertAlmostEqual(sil.sum() / full.sum(), 0.5, delta=0.03)

    def test_a_ground_truth_reference_measures_accuracy(self):
        run = make_run(self.tmp)
        (run / "stage-2h.json").write_text(json.dumps({"stage": "2h", "ok": True,
                                                       "details": {"reference": "ground-truth"}}))
        row = S.score_body(run, "b", write_body(self.tmp / "b.glb", (BODY, ARM)))["views"][S.HELDOUT_VIEW]
        self.assertEqual((row["reference"], row["measures"]), ("ground-truth", "accuracy"))
        self.assertEqual(S.MEASURES, {"generated": "consistency", "ground-truth": "accuracy"})


class Colour(Tmp):
    def test_flat_render_matches_the_view_colours(self):
        run = make_run(self.tmp)
        good = S.score_body(run, "good", write_coloured_body(self.tmp / "good.glb"))
        f = good["views"]["front"]
        self.assertEqual(good["colour_source"], "texture")
        self.assertLess(f["colour_error"], 1.0)
        # SSIM windows reach past the core into the backgrounds (black render, white view), as in stage 4
        self.assertGreater(f["ssim"], 0.75)
        self.assertGreater(f["pixels"], 1000)
        # no colour image for the other turnaround views: IoU only
        self.assertIsNone(good["views"]["back"]["colour_error"])
        swapped = S.score_body(run, "swapped", write_coloured_body(self.tmp / "s.glb",
                                                                   texture=stripe_texture((GREEN, RED))))
        self.assertGreater(swapped["views"]["front"]["colour_error"], 50.0)
        img = np.asarray(Image.open(run / "05-renders" / "good" / "front.png").convert("RGB"))
        self.assertEqual(tuple(img[230 - 70, 120]), RED)         # the render is the body's own colour

    def test_atlas_variant_replaces_the_textures(self):
        run = make_run(self.tmp)
        body = write_coloured_body(self.tmp / "b.glb")
        alt = self.tmp / "alt.png"
        Image.fromarray(np.full((8, 8, 3), BLUE, np.uint8)).save(alt)
        cache = {}
        base = S.score_body(run, "base", body, geometry_cache=cache)
        var = S.score_body(run, "base-blue", body, atlas=alt, variant_of="base", geometry_cache=cache)
        self.assertEqual(len(cache), 1)                         # the variant reused the mesh's geometry
        self.assertEqual(var["views"]["front"]["iou"], base["views"]["front"]["iou"])
        self.assertGreater(var["views"]["front"]["colour_error"], base["views"]["front"]["colour_error"] + 50)
        self.assertEqual((var["variant_of"], var["atlas_image"], var["atlas"]), ("base", "alt.png", 8))

    def test_untextured_body_has_no_colour_scores(self):
        run = make_run(self.tmp)
        e = S.score_body(run, "plain", write_body(self.tmp / "p.glb", (BODY, ARM)))
        self.assertIsNone(e["colour_source"])
        self.assertIsNone(e["views"]["front"]["colour_error"])

    def test_srgb_round_trip(self):
        c = np.linspace(0, 1, 11)
        np.testing.assert_allclose(S._to_srgb(S._to_linear(c)), c, atol=1e-9)


class Sanity(unittest.TestCase):
    def check(self, verts, faces):
        return S.mesh_sanity(np.asarray(verts, float), np.asarray(faces), 1e-6)

    def test_closed_box(self):
        s = self.check(*box((0, 0, 0), (1, 1, 1)))
        self.assertTrue(s["watertight"])
        self.assertEqual((s["boundary_edges"], s["nonmanifold_edges"], s["inconsistent_edges"]), (0, 0, 0))
        self.assertEqual(s["components"], 1)
        self.assertFalse(s["inside_out"])

    def test_split_vertices_are_welded(self):
        v, f = box((0, 0, 0), (1, 1, 1))
        s = self.check(v[f.ravel()], np.arange(36).reshape(12, 3))   # every face with its own vertices
        self.assertTrue(s["watertight"])

    def test_open_flipped_inside_out_duplicate_and_degenerate(self):
        v, f = box((0, 0, 0), (1, 1, 1))
        self.assertEqual(self.check(v, f[1:])["boundary_edges"], 3)
        flipped = f.copy()
        flipped[0] = flipped[0, ::-1]
        self.assertEqual(self.check(v, flipped)["inconsistent_edges"], 3)
        self.assertTrue(self.check(v, f[:, ::-1])["inside_out"])
        self.assertEqual(self.check(v, np.concatenate([f, f[:1]]))["nonmanifold_edges"], 3)
        self.assertEqual(self.check(v, np.concatenate([f, [[0, 0, 1], [0, 1, 3]]]))["degenerate"], 1)


class Discovery(Tmp):
    def stage4_body(self, run: Path, folder: str, mesh_source: str, variants=()) -> None:
        d = run / folder
        d.mkdir()
        write_coloured_body(d / "high.glb")
        write_body(d / "low.glb", (BODY,))
        (d / "sections.json").write_text(json.dumps({
            "height": 2.0, "mesh_source": mesh_source, "joints_source": "silhouette",
            "shape_views": ["front"] if mesh_source == "sam3d-objects" else ["back", "front", "left", "right"],
            "texture": {"colour_views": ["back", "front", "left", "right"]}, "lods": {"low": {"triangles": 12}}}))
        for name in variants:
            Image.fromarray(stripe_texture()).save(d / f"atlas-high-{name}.png")

    def test_bodies_variants_seen_flags_and_comparison_sheets(self):
        run = make_run(self.tmp, extra_heldout={"front-right-34": -30.0})
        self.stage4_body(run, "04-body-carve", "carve")
        self.stage4_body(run, "04-body", "sam3d-objects", variants=("side-trust",))
        names = [(b.name, b.variant_of, b.frame) for b in S.discover_bodies(run)]
        self.assertEqual(names, [("carve", None, "contract"), ("hybrid", None, "contract"),
                                 ("hybrid-side-trust", "hybrid", "contract")])
        self.assertEqual(S.main(["--run", str(run)]), 0)
        data = json.loads((run / "05-scores.json").read_text())
        by = {b["name"]: b for b in data["bodies"]}
        self.assertEqual(set(by), {"carve", "hybrid", "hybrid-side-trust"})
        hyb = by["hybrid"]
        self.assertEqual((hyb["mesh_source"], hyb["glb"], hyb["triangles_low"]), ("sam3d-objects",
                                                                                 "04-body/high.glb", 12))
        self.assertEqual(hyb["views"]["left"], {**hyb["views"]["left"], "seen": True, "seen_shape": False,
                                                "seen_colour": True})
        self.assertTrue(by["carve"]["views"]["left"]["seen_shape"])
        for b in by.values():
            for v in ("front-left-34", "front-right-34"):
                self.assertFalse(b["views"][v]["seen"])
        self.assertEqual(by["hybrid-side-trust"]["variant_of"], "hybrid")
        self.assertEqual([v["view"] for v in data["views"] if v["heldout"]], ["front-left-34", "front-right-34"])
        for name in ("heldout-comparison.png", "heldout-comparison-front-left-34.png",
                     "heldout-comparison-front-right-34.png"):
            self.assertTrue((run / "05-renders" / name).exists(), name)
        # the results page reads the same file
        accept_heldout(run)                                      # the page shows views the record accepts
        page = collect_scores(RunFolder(run.parent, run.name))
        shown = {b["name"]: b for b in page["bodies"]}
        self.assertEqual(shown["hybrid"]["heldout_renders"],
                         {S.HELDOUT_VIEW: f"05-renders/hybrid/{S.HELDOUT_VIEW}.png"})
        self.assertIsNotNone(shown["hybrid"]["views"]["front"]["colour_error"])

    def test_old_single_heldout_view_only(self):
        run = make_run(self.tmp)
        self.stage4_body(run, "04-body", "carve")
        self.assertEqual(S.main(["--run", str(run)]), 0)
        b = json.loads((run / "05-scores.json").read_text())["bodies"][0]
        self.assertEqual(b["name"], "carve")
        self.assertEqual([v for v in b["views"] if v not in S.TURNAROUND], [S.HELDOUT_VIEW])

    def test_no_body_is_a_clear_error(self):
        run = make_run(self.tmp)
        self.assertEqual(S.main(["--run", str(run)]), 1)


class ScoresFile(Tmp):
    def test_cli_writes_scores_the_results_page_reads(self):
        run = make_run(self.tmp)
        ours = write_body(run / "body.glb", (BODY, ARM))
        theirs = write_body(self.tmp / "vendor.glb", (BODY,))
        code = S.main(["--run", str(run), "--body", f"ours={ours}", "--body", f"vendor={theirs}",
                       "--seen", "ours=front,back,left,right", "--budget", "vendor=10"])
        self.assertEqual(code, 0)
        data = json.loads((run / "05-scores.json").read_text())
        by = {b["name"]: b for b in data["bodies"]}
        self.assertEqual(by["ours"]["glb"], "body.glb")          # run-relative
        self.assertEqual(by["vendor"]["glb"], "vendor.glb")      # outside the run: file name only
        self.assertIsNone(by["ours"]["budget"])                  # no budget unless asked for
        self.assertIsNone(by["ours"]["within_budget"])
        self.assertNotIn("profile", by["ours"])
        self.assertFalse(by["vendor"]["within_budget"])          # 12 triangles over a budget of 10
        self.assertEqual(by["ours"]["sanity"]["meshes"], 2)
        self.assertTrue(by["ours"]["sanity"]["watertight"])
        self.assertTrue(by["vendor"]["views"]["front"]["seen"])  # default: every observed view
        accept_heldout(run)                                      # the page shows views the record accepts
        page = collect_scores(RunFolder(run.parent, run.name))
        shown = {b["name"]: b for b in page["bodies"]}
        self.assertEqual(shown["ours"]["heldout_renders"],
                         {S.HELDOUT_VIEW: f"05-renders/ours/{S.HELDOUT_VIEW}.png"})
        self.assertIn("held out; yaw", "".join(render_scores(page)))

    def test_seen_shape_differs_from_seen_colour(self):
        run = make_run(self.tmp)
        b = write_body(self.tmp / "b.glb", (BODY, ARM))
        self.assertEqual(S.main(["--run", str(run), "--body", f"b={b}", "--seen", "b=front,back,left,right",
                                 "--seen-shape", "b=front"]), 0)
        v = json.loads((run / "05-scores.json").read_text())["bodies"][0]["views"]
        self.assertEqual((v["left"]["seen_shape"], v["left"]["seen_colour"], v["left"]["seen"]), (False, True, True))
        self.assertTrue(v["front"]["seen_shape"])

    def test_roblox_profile_checks_part_groups_by_mesh_name(self):
        run = make_run(self.tmp)
        n = 46   # an open 46 x 46 grid: 4,232 triangles, over the 4,000 head limit
        g = np.linspace(-0.5, 0.5, n + 1)
        xs, zs = np.meshgrid(g, g + 2.0)
        grid_v = np.stack([xs.ravel(), np.zeros(xs.size), zs.ravel()], axis=1)
        i = np.arange(n)[:, None] * (n + 1) + np.arange(n)[None, :]
        q = np.stack([i, i + 1, i + n + 2, i + n + 1], axis=-1).reshape(-1, 4)
        grid_f = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        sections = G.write_glb(self.tmp / "r15.glb", [("UpperTorso", *box(*BODY)), ("LowerTorso.001", *box(*ARM)),
                                                      ("Head", grid_v, grid_f)])
        plain = write_body(self.tmp / "plain.glb", (BODY, ARM))
        self.assertEqual(S.main(["--run", str(run), "--profile", "roblox-avatar", "--body", f"r15={sections}",
                                 "--body", f"plain={plain}"]), 0)
        by = {b["name"]: b for b in json.loads((run / "05-scores.json").read_text())["bodies"]}
        checks = {c["name"]: c for c in by["r15"]["profile"]["checks"]}
        self.assertEqual(by["r15"]["budget"], 10742)
        self.assertEqual(checks["triangles_Torso"]["value"], 24)        # ".001" suffix still counts
        self.assertFalse(checks["triangles_Head"]["pass"])
        self.assertIsNone(checks["triangles_LeftArm"]["value"])          # no such meshes: not checked
        self.assertFalse(checks["watertight_parts"]["pass"])             # the head grid is open
        self.assertIsNone(checks["not_built"]["pass"])                   # listed, not checked
        self.assertFalse(checks["not_built"]["required"])
        self.assertIn("DynamicHead", checks["not_built"]["note"])
        self.assertEqual((by["r15"]["profile"]["pass"], by["r15"]["profile"]["status"]), (False, "fail"))
        plain_checks = {c["name"]: c for c in by["plain"]["profile"]["checks"]}
        self.assertTrue(plain_checks["triangles_total"]["pass"])
        # nothing failed, but the part groups (unnamed meshes) and the texture (none) could not be measured
        self.assertIsNone(plain_checks["texture_max_px"]["pass"])
        self.assertEqual((by["plain"]["profile"]["pass"], by["plain"]["profile"]["status"]), (None, "incomplete"))
        with self.assertRaises(ValueError):
            S.score_body(run, "x", plain, seen=set(), profile="nope")
        with self.assertRaises(SystemExit):
            S.main(["--run", str(run), "--profile", "roblox", "--body", f"plain={plain}"])   # old name

    def test_an_avatar_body_within_every_limit_passes(self):
        """Fifteen closed parts named after the sections, a small texture: every required check is measured and
        met, so the profile passes; what it does not build is listed but never counts."""
        from herogen.blender import SECTIONS

        run = make_run(self.tmp)
        meshes = []
        for i, name in enumerate(SECTIONS):
            v, f = box((0.0, 0.0, 0.2 + 0.12 * i), (0.3, 0.2, 0.05))
            meshes.append((name, v, f, np.tile([0.5, 0.5], (len(v), 1))))
        p = G.write_glb(self.tmp / "avatar.glb", meshes, texture_png=png(stripe_texture()))
        entry = S.score_body(run, "avatar", p, profile="roblox-avatar")
        checks = {c["name"]: c for c in entry["profile"]["checks"]}
        for name, c in checks.items():
            if name != "not_built":
                self.assertTrue(c["pass"], c)
        self.assertIsNone(checks["not_built"]["pass"])
        self.assertEqual((entry["profile"]["pass"], entry["profile"]["status"]), (True, "pass"))

    def test_ingame_and_best_profiles(self):
        run = make_run(self.tmp)
        d = run / "04-body"
        d.mkdir()
        high = write_body(d / "high.glb", (BODY, ARM))
        write_body(d / "low.glb", (BODY,))
        lone = write_body(self.tmp / "lone.glb", (BODY,))
        self.assertEqual(S.main(["--run", str(run), "--profile", "roblox-ingame", "--body", f"h={high}",
                                 "--body", f"lone={lone}"]), 0)
        by = {b["name"]: b for b in json.loads((run / "05-scores.json").read_text())["bodies"]}
        checks = {c["name"]: c for c in by["h"]["profile"]["checks"]}
        self.assertEqual(by["h"]["budget"], 24000)
        self.assertEqual((checks["triangles_high"]["value"], checks["triangles_high"]["limit"]), (24, 24000))
        self.assertEqual((checks["triangles_low"]["value"], checks["triangles_low"]["pass"]), (12, True))
        self.assertEqual((by["h"]["profile"]["pass"], by["h"]["profile"]["status"]), (True, "pass"))
        lone_low = {c["name"]: c for c in by["lone"]["profile"]["checks"]}["triangles_low"]
        self.assertIsNone(lone_low["pass"])                              # no low LOD: reported, not failed
        self.assertEqual((by["lone"]["profile"]["pass"], by["lone"]["profile"]["status"]), (None, "incomplete"))
        self.assertEqual(S.main(["--run", str(run), "--profile", "best", "--body", f"h={high}"]), 0)
        best = json.loads((run / "05-scores.json").read_text())["bodies"]
        best = next(b for b in best if b["name"] == "h")
        self.assertEqual(best["profile"], {"name": "best", "checks": [], "pass": True, "status": "pass"})
        self.assertIsNone(best["budget"])

    def test_rescoring_replaces_a_body_and_keeps_the_rest(self):
        run = make_run(self.tmp)
        a = write_body(self.tmp / "a.glb", (BODY, ARM))
        S.main(["--run", str(run), "--body", f"a={a}", "--body", f"b={a}"])
        S.main(["--run", str(run), "--body", f"a={write_body(self.tmp / 'c.glb', (BODY,))}"])
        names = [b["name"] for b in json.loads((run / "05-scores.json").read_text())["bodies"]]
        self.assertEqual(sorted(names), ["a", "b"])

    def test_cli_rejects_bad_arguments(self):
        run = make_run(self.tmp)
        a = write_body(self.tmp / "a.glb", (BODY,))
        for argv in (["--body", f"a={a}", "--seen", f"a={S.HELDOUT_VIEW}"],
                     ["--body", f"a={a}", "--budget", "b=5"],
                     ["--body", f"a={a}", "--frame", "a=sideways"],
                     ["--body", f"a={a}", "--yaw-range", "nope=1,2"],
                     ["--body", f"a={a}", "--yaw-range", f"{S.HELDOUT_VIEW}=12"],
                     ["--body", "noequals"]):
            with self.assertRaises(SystemExit, msg=argv):
                S.main(["--run", str(run), *argv])
        self.assertEqual(S.main(["--run", str(run), "--body", f"a={self.tmp / 'missing.glb'}"]), 1)

    def test_package_cli_forwards_to_the_scorer(self):
        from herogen.blender.__main__ import main as blender_main

        run = make_run(self.tmp)
        a = write_body(self.tmp / "a.glb", (BODY, ARM))
        self.assertEqual(blender_main(["score", "--run", str(run), "--body", f"a={a}"]), 0)
        self.assertTrue((run / "05-scores.json").exists())


if __name__ == "__main__":
    unittest.main()
