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

    def test_fast_exhausted_ladder_selects_smallest_complete_candidate(self):
        def make(target):
            return dict(candidate({n: 500 + target for n in SECTIONS}), target=target)

        cands = [make(2700)]
        stage4._fit_budget(cands, {"budget": 6000}, "roblox-ingame", make, fast=True)
        self.assertEqual(cands[-1]["target"], 1)
        pick = stage4._pick(cands, {"budget": 6000}, "roblox-ingame", fast=True)
        self.assertEqual(set(pick.values()), {len(cands) - 1})
        self.assertEqual(cands[pick[SECTIONS[0]]]["total"], 7515)
        with self.assertRaises(ValueError) as exc:
            stage4._pick(cands, {"budget": 6000}, "roblox-ingame")
        self.assertEqual(str(exc.exception).encode(),
                         b"stage 4 cannot meet triangle budgets after cuts, caps and covers: body 7515 > 6000")

    def test_fast_fallback_skips_smaller_candidates_that_break_hard_limits(self):
        cands = [candidate({n: 600 for n in SECTIONS}),
                 candidate({n: 500 for n in SECTIONS}),
                 candidate({"Head": 20001}), candidate({"RightHand": 0}),
                 candidate({"LeftHand": 0})]
        del cands[-1]["covered"]["LeftHand"]
        cands.append(candidate({}))
        cands[-1]["covered"]["Head"].verts = np.empty((0, 3))
        self.assertEqual(set(stage4._pick(cands, {"budget": 1}, "roblox-ingame", fast=True).values()), {1})
        with self.assertRaisesRegex(ValueError, "cannot meet hard Roblox limits"):
            stage4._pick(cands[2:], {"budget": 1}, "roblox-ingame", fast=True)

    def test_fast_fallback_accepts_exact_hard_triangle_limit(self):
        cands = [candidate({"Head": 20000})]
        self.assertEqual(set(stage4._pick(cands, {"budget": 6000}, "roblox-ingame", fast=True).values()), {0})

    def test_fast_exhausted_ladder_still_raises_if_hard_limits_cannot_fit(self):
        def make(target):
            return dict(candidate({"Head": 20001}), target=target)

        cands = [make(2)]
        with self.assertRaisesRegex(ValueError, "cannot meet hard Roblox limits"):
            stage4._fit_budget(cands, {"budget": 6000}, "roblox-ingame", make, fast=True)
        self.assertEqual(cands[-1]["target"], 1)

    def test_fast_fitting_ladder_keeps_the_full_lane_selection(self):
        cands = [candidate({n: k for n in SECTIONS}) for k in (100, 300, 500)]
        build = {"budget": 5000}
        self.assertEqual(stage4._pick(cands, build, "roblox-ingame", fast=True),
                         stage4._pick(cands, build, "roblox-ingame"))

    def test_fast_budget_miss_exports_and_records_the_real_triangle_count(self):
        from herogen.blender import glb
        from herogen.blender.record import StageRecord, check

        cands = [candidate({n: count for n in SECTIONS}) for count in (800, 600, 700)]
        pick = stage4._pick(cands, {"budget": 6000}, "roblox-ingame", fast=True)
        selected = {n: cands[pick[n]]["covered"][n] for n in SECTIONS}
        with tempfile.TemporaryDirectory() as tmp:
            rec = StageRecord(Path(tmp), 4, "game body")
            body = rec.run / "04-body"
            body.mkdir()
            glb.write_glb(body / "low.glb", [(n, s.verts, s.faces) for n, s in selected.items()])
            rec.data["outputs"] = ["04-body/low.glb"]
            rec.add(check("sections_non_empty", len(selected), 15))
            rec.add(check("triangles_low", 9000, 6000, passed=False, rule="<="))
            stage4._record_fast_outputs(rec, ("low",))
            result = rec.finish()
            self.assertTrue(result["ok"])
            checks = {c["name"]: c for c in result["checks"]}
            self.assertFalse(checks["triangles_low"]["pass"])
            self.assertFalse(checks["triangles_low"]["gate"])
            self.assertEqual(checks["exported_parts_low"]["value"], 15)
            self.assertEqual(checks["exported_triangles_per_mesh_low"]["value"], 600)
            self.assertIn("low-detail body uses 9,000 triangles, above the 6,000 target", result["notes"])

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
