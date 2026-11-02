"""Final section budgets include geometry added after whole-body decimation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from herogen.blender import SECTIONS, bl, stage4
from test_herogen_blender_profiles import candidate


class FinalBudgets(unittest.TestCase):
    def test_fitting_ladder_keeps_objects_and_selection_unchanged(self):
        cands = [dict(candidate({n: count for n in SECTIONS}), target=target)
                 for count, target in ((100, 1000), (300, 2000), (500, 3000))]
        before = list(cands)
        build = {"budget": 5000}
        pick = stage4._pick(cands, build, "roblox-ingame")
        make = mock.Mock(side_effect=AssertionError("a fitting ladder must not be extended"))
        stage4._fit_budget(cands, build, "roblox-ingame", make)
        self.assertEqual([id(c) for c in cands], [id(c) for c in before])
        self.assertEqual(stage4._pick(cands, build, "roblox-ingame"), pick)
        make.assert_not_called()

    def test_added_caps_and_covers_extend_below_the_old_floor(self):
        # A 2700-triangle collapse fits its own target, but not the final 6000 budget.
        make = mock.Mock(side_effect=lambda target: {"target": target, "total": target * 2 + 1700})
        cands = [make(t) for t in range(2700, 5101, 300)]
        stage4._fit_budget(cands, {"budget": 6000}, "roblox-ingame", make)
        pick = stage4._pick(cands, {"budget": 6000}, "roblox-ingame")
        chosen = cands[pick[SECTIONS[0]]]
        self.assertLess(chosen["target"], 2700)
        self.assertLessEqual(chosen["total"], 6000)
        self.assertLessEqual(6000 - chosen["total"], 600)

    def test_impossible_budget_fails_loudly_at_a_finite_floor(self):
        make = mock.Mock(side_effect=lambda target: {"target": target, "total": target + 7000})
        cands = [make(2700)]
        with self.assertRaisesRegex(ValueError, "body 7001 > 6000"):
            stage4._fit_budget(cands, {"budget": 6000}, "roblox-ingame", make)
        self.assertEqual(cands[-1]["target"], 1)
        self.assertLessEqual(make.call_count, 13)

    def test_avatar_group_shortfalls_extend_the_ladder_too(self):
        def make(target):
            return dict(candidate({n: target for n in SECTIONS}), target=target)

        cands = [make(2000), make(3000)]
        stage4._fit_budget(cands, {"budget": "groups"}, "roblox-avatar", make)
        pick = stage4._pick(cands, {"budget": "groups"}, "roblox-avatar")
        for secs, limit in stage4.PROFILES["roblox-avatar"]["groups"].values():
            self.assertLessEqual(sum(len(cands[pick[n]]["covered"][n].faces) for n in secs), limit)

    def test_unbudgeted_profile_never_decimates_again(self):
        cands = [{"total": 100000}]
        make = mock.Mock(side_effect=AssertionError("no triangle budget"))
        stage4._fit_budget(cands, {"budget": None}, "best", make)
        make.assert_not_called()


@unittest.skipUnless(bl.find_blender(), "Blender is required")
class BlenderBudgetSearch(unittest.TestCase):
    def test_real_collapse_and_final_geometry_are_both_counted(self):
        # Synthetic octahedron subdivisions: no external mesh fixtures.
        verts = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
        faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                          [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
        for _ in range(4):
            points, mids, refined = verts.tolist(), {}, []
            for a, b, c in faces:
                ids = []
                for i, j in ((a, b), (b, c), (c, a)):
                    edge = tuple(sorted((int(i), int(j))))
                    if edge not in mids:
                        mids[edge] = len(points)
                        p = verts[i] + verts[j]
                        points.append((p / np.linalg.norm(p)).tolist())
                    ids.append(mids[edge])
                ab, bc, ca = ids
                refined.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
            verts, faces = np.asarray(points), np.asarray(refined)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            mesh = work / "mesh.npz"
            np.savez(mesh, verts=verts, faces=faces)
            calls = []

            def make(target):
                out = work / f"dec-{target}.npz"
                result = bl.run_job({"tasks": [{"task": "decimate", "mesh": str(mesh),
                                              "targets": {"body": target}, "out": {"body": str(out)}}]},
                                    work, f"decimate-{target}")
                count = len(np.load(out)["faces"])
                self.assertLessEqual(count, target)
                self.assertEqual(result["results"][0]["body"]["triangles_out"], count)
                calls.append(target)
                # Closed cap/cover geometry consumes a fixed share of the final budget.
                return {"target": target, "total": count + 400}

            cands = [make(800), make(900)]
            stage4._fit_budget(cands, {"budget": 1000}, "roblox-ingame", make)
            pick = stage4._pick(cands, {"budget": 1000}, "roblox-ingame")
            self.assertLess(min(calls), 800)
            self.assertLessEqual(cands[pick[SECTIONS[0]]]["total"], 1000)
