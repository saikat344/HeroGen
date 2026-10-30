"""Target profiles (herogen.blender.PROFILES): best (the default: no triangle budget, full detail, no low LOD),
roblox-ingame (24,000 / 6,000) and roblox-avatar (the triangle, watertight-part and texture
limits of Roblox's character body specification), as stage 4 builds them.

The selection rule runs everywhere numpy is installed; the end-to-end tests need Blender and skip without it.
Nothing generated here is written inside the repository.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from test_herogen_body_quality import quality_client

import numpy as np

from herogen.blender import AVATAR_GROUPS, DEFAULT_PROFILE, PROFILES, SECTIONS, bl, synth
from herogen.blender import stage4 as S4
from herogen.blender.split import Section


def candidate(per_section: dict) -> dict:
    """A stand-in decimation candidate: each section a mesh of the given triangle count."""
    covered = {n: Section(n, np.zeros((3, 3)), np.zeros((per_section.get(n, 10), 3), dtype=np.int64),
                          np.zeros(3), [], 0) for n in SECTIONS}
    return {"covered": covered, "total": sum(len(s.faces) for s in covered.values())}


class Selection(unittest.TestCase):
    def test_profiles_and_default(self):
        self.assertEqual(DEFAULT_PROFILE, "best")
        self.assertEqual(set(PROFILES), {"best", "roblox-ingame", "roblox-avatar"})
        self.assertEqual(PROFILES["roblox-ingame"]["lods"], {"high": 24000, "low": 6000})
        self.assertEqual(PROFILES["roblox-avatar"]["triangles_total"], sum(n for _, n in AVATAR_GROUPS.values()))
        self.assertEqual(sorted(s for secs, _ in AVATAR_GROUPS.values() for s in secs), sorted(SECTIONS))
        self.assertEqual(set(S4.LOD_BUILD["best"]), {"high"})
        self.assertIsNone(S4.LOD_BUILD["best"]["high"]["budget"])
        self.assertEqual(set(S4.LOD_BUILD["roblox-avatar"]), {"high"})
        self.assertIsNone(S4._budgets_record("best"))
        # atlas sizes per profile: in-game 1024 / 512; best and the avatar from the source's pixel density, at
        # most 4096 and the avatar's 2048 px limit
        self.assertEqual({lod: b["atlas"] for lod, b in S4.LOD_BUILD["roblox-ingame"].items()},
                         {"high": 1024, "low": 512})
        self.assertEqual(S4.LOD_BUILD["roblox-avatar"]["high"]["atlas"],
                         ("source", PROFILES["roblox-avatar"]["texture_max_px"]))
        self.assertEqual(S4.LOD_BUILD["best"]["high"]["atlas"], ("source", 4096))
        self.assertEqual(S4.UV_GUTTER_PX, 2)
        not_built = " ".join(S4._budgets_record("roblox-avatar")["not_built"])
        for part in ("rig", "skinning", "attachments", "DynamicHead", "cages"):
            self.assertIn(part, not_built)

    def test_one_budget_takes_the_largest_candidate_that_fits(self):
        cands = [candidate({n: k for n in SECTIONS}) for k in (100, 300, 500)]     # totals 1500, 4500, 7500
        build = {"budget": 5000}
        self.assertEqual(set(S4._pick(cands, build, "roblox-ingame").values()), {1})
        self.assertEqual(set(S4._pick(cands, {"budget": 1000}, "roblox-ingame").values()), {0})   # none fits
        self.assertEqual(set(S4._pick(cands[:1], {"budget": None}, "best").values()), {0})

    def test_part_groups_choose_their_own_candidate(self):
        # candidate k has 400 * (k + 1) triangles in every section: the head (one section, 4,000) can take the
        # densest that fits, the torso (two sections, 1,750) a sparser one, the limbs (three, 1,248) sparser still
        cands = [candidate({n: 400 * (k + 1) for n in SECTIONS}) for k in range(10)]
        pick = S4._pick(cands, {"budget": "groups"}, "roblox-avatar")
        self.assertEqual(pick["Head"], 9)                 # 4,000
        self.assertEqual(pick["UpperTorso"], pick["LowerTorso"])
        self.assertEqual(pick["UpperTorso"], 1)           # 2 x 800 = 1,600
        self.assertEqual(pick["LeftHand"], 0)             # 3 x 400 = 1,200
        self.assertEqual(len({pick[n] for n in AVATAR_GROUPS["LeftArm"][0]}), 1)
        too_big = [candidate({n: 2000 for n in SECTIONS})]
        self.assertEqual(set(S4._pick(too_big, {"budget": "groups"}, "roblox-avatar").values()), {0})

    def test_unknown_profile_is_refused(self):
        with self.assertRaises(ValueError):
            S4.run_stage4(Path("."), profile="roblox")


@unittest.skipUnless(bl.find_blender() is not None, "Blender not installed")
class ProfilesEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from herogen.blender.stage3 import run_stage3

        cls.tmp = Path(tempfile.mkdtemp(prefix="herogen-profiles-"))
        cls.rundir = cls.tmp / "run"
        synth.write_run(cls.rundir, synth.humanoid(), size=384, ppu=60.0)
        assert run_stage3(cls.rundir, nz=96, timeout=300, mesh_source="carve", joints_source="silhouette", quality_client=quality_client())["ok"]
        cls.stage3_triangles = int(np.load(cls.rundir / "blender-work" / "03-mesh.npz")["faces"].shape[0])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def records(self):
        body = self.rundir / "04-body"
        stage4 = json.loads((self.rundir / "stage-4.json").read_text())
        sections = json.loads((body / "sections.json").read_text())
        self.assertEqual(sections["texture"]["head_texture_sizes"], {"high": 1024})
        self.assertEqual(stage4["body"]["texture"], sections["texture"])
        from PIL import Image
        with Image.open(body / "atlas-head-high.png") as im:
            self.assertEqual(im.size, (1024, 1024))
        return stage4, sections, body

    def test_best_keeps_full_detail_and_builds_no_low_lod(self):
        from herogen.blender.stage4 import run_stage4

        # a low LOD from an earlier in-game build must not survive
        run_stage4(self.rundir, timeout=600, profile="roblox-ingame")
        self.assertTrue((self.rundir / "04-body" / "low.glb").exists())
        rec = run_stage4(self.rundir, timeout=900)                      # the default profile
        stage4, sections, body = self.records()
        self.assertEqual(rec["profile"], {"name": "best", "lods": ["high"], "budgets": None})
        self.assertEqual((sections["profile"], sections["budgets"]), ("best", None))
        self.assertIn("code", stage4)                                # the code version that built it
        self.assertEqual(sections["code"], stage4["code"])
        self.assertFalse((body / "low.glb").exists())
        self.assertFalse((body / "atlas-low.png").exists())
        self.assertTrue((body / "high.glb").exists())
        lods = sections["lods"]
        self.assertEqual(set(lods), {"high"})
        self.assertEqual(lods["high"]["body_triangles_after_decimation"], self.stage3_triangles)   # undecimated
        checks = {c["name"]: c for c in stage4["checks"]}
        self.assertFalse(checks["triangles_high"]["gate"])           # no budget under best: reported only
        self.assertIsNone(checks["triangles_high"]["threshold"])
        self.assertNotIn("triangles_low", checks)
        self.assertNotIn("max_joint_gap_low", checks)
        for name in ("sections_non_empty", "open_edges", "max_joint_gap_high", "rest_iou_with_covers_front",
                     "texture_colour_error_front"):
            self.assertTrue(checks[name]["gate"], name)               # every other gate unchanged
        self.assertEqual(checks["open_edges"]["value"], 0)
        self.assertTrue(checks["watertight_parts"]["gate"])           # gated under every profile
        self.assertFalse(checks["max_joint_gap_high_heldout"]["gate"])  # held-out poses: reported only
        # the atlas: the smallest size with a texel per view pixel, or the largest allowed
        choice = sections["texture"]["atlas_choice"]["high"]
        size = sections["texture"]["atlas_sizes"]["high"]
        self.assertEqual(choice["chosen"], size)
        reach = [int(n) for n, d in choice["texels_per_stud"].items() if d >= choice["view_px_per_stud"]]
        self.assertEqual(size, min(reach) if reach else 4096)
        self.assertIsNone(sections["sections"][0]["triangles_low"])

    def test_avatar_fits_every_part_group(self):
        from herogen.blender.stage4 import run_stage4

        rec = run_stage4(self.rundir, timeout=900, profile="roblox-avatar")
        stage4, sections, body = self.records()
        checks = {c["name"]: c for c in stage4["checks"]}
        self.assertLessEqual(checks["triangles_total"]["value"], 10742)
        self.assertTrue(checks["triangles_total"]["gate"])
        for group, (secs, limit) in AVATAR_GROUPS.items():
            c = checks[f"triangles_{group}"]
            self.assertEqual((c["threshold"], c["sections"]), (limit, list(secs)))
            self.assertTrue(c["pass"], c)
            self.assertEqual(c["value"], sum(s["triangles_high"] for s in sections["sections"] if s["name"] in secs))
        self.assertEqual(checks["watertight_parts"]["value"], 0)
        self.assertTrue(checks["texture_max_px"]["pass"])
        self.assertFalse((body / "low.glb").exists())
        self.assertEqual(sections["budgets"]["triangles_total_max"], 10742)
        self.assertIn("not built: R15 bone rig with skinning, attachments, facial animation (DynamicHead), cages",
                      rec["notes"])
        self.assertEqual(set(rec["body"]["chosen_targets"]["high"]), set(AVATAR_GROUPS))
        raw = (body / "high.glb").read_bytes()
        n = int.from_bytes(raw[12:16], "little")
        self.assertEqual(len(json.loads(raw[20:20 + n])["meshes"]), 15)


if __name__ == "__main__":
    unittest.main()
