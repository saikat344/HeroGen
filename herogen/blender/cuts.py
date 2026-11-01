"""Where the rigid sections are cut: a joint's cut moved along its own normal when the gate poses show it gaps
less there, by more than the measurement's own noise.

The cut planes start at the estimated joints (`split.build_rig`). A costume can have a ledge there, a belt or a
hem where the body above is narrower than below; a cut on the ledge leaves the wider side's rim standing proud
when the joint bends. The search is built not to chase the gate poses:

- Only the gate poses choose (`pose.pose_library`, the ungated twist left out); the held-out poses
  (`pose.heldout_poses`) never do, so stage 4's held-out gap shows whether a move holds for other poses.
- Joints move in groups, the two joints of a group together and left and right alike: neck and waist, elbows and
  wrists, knees and ankles (`GROUPS`). A candidate is scored by the worst gate gap of the group's own joints, and
  refused if it opens a neighbouring joint whose sections it changes (a waist move changes the shoulders' and
  hips' sections) by more than `NOISE_STUDS`.
- Only a joint whose unmoved gap is within `MARGIN_STUDS` of the gate may move; the others stay at the joint.
- A move must lower its group's worst gap by more than `NOISE_STUDS`, the measurement's noise; among the moves that
  do, the one with the lowest gap plus `NOISE_STUDS` per bone radius moved wins (the distance from the anatomical
  joint costs), and with none the cuts stay at the joints.

`NOISE_STUDS` and `MARGIN_STUDS` are measured: the worst gate gap of each joint, at unmoved cuts, over nine
decimations of one stage 3 mesh (10.8k to 20.4k triangles) varies by 0.0046 / 0.0065 (median over joints, hybrid
/ carve), 0.018 / 0.031 at the 90th percentile and at most 0.020 / 0.044 (pencil-sketch figure). NOISE_STUDS is
the 90th percentile (pooled, rounded up), MARGIN_STUDS the largest. Between two recordings of the same SAM 3D
Objects call the source mesh itself changes, and with it single joints' gaps by much more (the left hip, 0.053 and
0.284 studs on the 20k reference); the search runs on the one recorded mesh, so that variation is reported, not
absorbed.

How the rule holds on poses it never sees was checked afterwards, as a report that chose nothing, on the
pencil-sketch figure's 20k reference bodies: the moves the gate poses favour most (hybrid waist +0.5 radius, carve
waist +0.5, carve knee +0.25 with ankle -0.5) lower the held-out worst gap of their joints too (0.191 to 0.132,
0.107 to 0.101, 0.108 to 0.085). The rule's two constants come from the decimation measurement above, not from
those poses.

The search runs once on a reference decimation (`REFERENCE_TRIANGLES`) and every LOD of every profile is cut at the
same places. Moving a cut at most half a bone radius keeps its pivot (the cut's centre) within the pivot check's
limit; the shoulder and hip cuts keep their own measured offsets.
"""
from __future__ import annotations

import itertools
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import PARENT, PIVOT_JOINT, covers, pose, split

# group -> its joint kinds; a kind is moved by one fraction on both sides
GROUPS = {"torso": ("Neck", "Waist"), "arms": ("Elbow", "Wrist"), "legs": ("Knee", "Ankle")}
SIDED = {"Elbow", "Wrist", "Knee", "Ankle"}
CHILD = {"Neck": "Head", "Waist": "UpperTorso", "LeftElbow": "LeftLowerArm", "RightElbow": "RightLowerArm",
         "LeftWrist": "LeftHand", "RightWrist": "RightHand", "LeftKnee": "LeftLowerLeg", "RightKnee": "RightLowerLeg",
         "LeftAnkle": "LeftFoot", "RightAnkle": "RightFoot"}
JOINTS = tuple(CHILD)
FRACTIONS = (-0.5, -0.25, 0.0, 0.25, 0.5)
NOISE_STUDS = 0.025
MARGIN_STUDS = 0.045
REFERENCE_TRIANGLES = 20000


class UnmeasurableJoints(ValueError):
    """A cut has no shared rim between two nonempty sections; its gap is unknown."""

    def __init__(self, reasons: dict[str, str]):
        self.reasons = reasons
        detail = "; ".join(f"{j}: {reason}" for j, reason in sorted(reasons.items()))
        super().__init__(f"cannot measure cut gaps ({detail}); check the stage 3 mesh and joint placement")


def joints_of(kind: str) -> tuple:
    return (f"Left{kind}", f"Right{kind}") if kind in SIDED else (kind,)


def affected(kinds) -> set:
    """Every joint whose rim lies on a section that moving these joints' cuts changes."""
    secs = set()
    for kind in kinds:
        for j in joints_of(kind):
            child = CHILD[j]
            secs |= {child, PARENT[child]}
    return {PIVOT_JOINT[s] for s, p in PARENT.items() if p is not None and (s in secs or p in secs)}


def unmeasurable_joints(secs: dict, joints: set | None = None) -> dict[str, str]:
    """Missing parent/child sections or shared rims, never evidence of a zero gap."""
    expected = {PIVOT_JOINT[s] for s, p in PARENT.items() if p is not None}
    if joints is not None:
        expected &= joints
    missing = {}
    for child, parent in PARENT.items():
        j = PIVOT_JOINT[child]
        if parent is None or j not in expected:
            continue
        empty = [n for n in (child, parent) if not len(secs[n].faces)]
        if empty:
            missing[j] = "empty section " + ", ".join(empty)
        elif not any(r["neighbour"] == parent and len(r["points"]) for r in secs[child].rims):
            missing[j] = f"no shared rim between {child} and {parent}"
    return missing


def worst_by_joint(verts: np.ndarray, faces: np.ndarray, sk, offsets: dict, pose_sets: dict, cover_mesh: tuple,
                   cover: dict, dome: tuple[int, int], joints: set | None = None,
                   unverified: dict | None = None) -> dict:
    """Per pose set, each joint's worst visible gap (studs), covers on (only `joints`, when given).

    By default missing measurements raise. The fast lane can supply `unverified` to collect their reasons;
    those joints are omitted from the measurements, never assigned a zero gap.
    """
    secs, _ = split.build_sections(verts, faces, sk, cap="dome", rings=dome[0], ring_points=dome[1],
                                   cut_offsets=offsets)
    expected = {PIVOT_JOINT[s] for s, p in PARENT.items() if p is not None}
    if joints is not None:
        expected &= joints
    missing = unmeasurable_joints(secs, joints)
    if missing:
        if unverified is None:
            raise UnmeasurableJoints(missing)
        unverified.update(missing)
        expected -= missing.keys()
        joints = expected
    covered, _ = covers.add_covers(secs, cover_mesh[0], cover_mesh[1], **cover)
    out = {}
    for name, poses in pose_sets.items():
        g = pose.joint_gaps(covered, poses, visible=True, joints=joints)
        out[name] = {j: max(row[j] for row in g.values()) for j in sorted(expected)}
    return out


def validate_workers(workers: int) -> None:
    """Reject invalid candidate concurrency before doing geometry work."""
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 12:
        raise ValueError("cut workers must be an integer from 1 through 12")


def choose(verts: np.ndarray, faces: np.ndarray, sk, poses: dict, cover_mesh: tuple, cover: dict,
           dome: tuple[int, int], gate: float, workers: int = 3, fast: bool = False) -> tuple[dict, dict]:
    """Cut offsets (joint -> studs along the cut normal) for the reference mesh, and the record of the search.

    `poses` are the gate poses (the ungated ones left out); `cover_mesh` (verts, faces) and `cover` (add_covers
    keyword arguments) shape the shoulder covers as stage 4 does; `dome` is (rings, ring_points) of the caps;
    `gate` the joint-gap gate the margin is measured from. `workers` (1 through 12) bounds concurrent candidate
    evaluations; results are consumed in the original candidate order, including equal-score ties. `fast`
    reports unmeasurable baseline joints and leaves their cut kinds unmoved; candidates must still preserve
    every measurement that exists at the baseline."""
    validate_workers(workers)
    t0 = time.monotonic()
    rig = split.build_rig(sk)
    radius = {j: float(rig.bones[CHILD[j]][2]) for j in JOINTS}

    def worst_gaps(fractions: dict, joints: set | None = None, unverified: dict | None = None) -> dict:
        offsets = {j: fractions.get(kind, 0.0) * radius[j] for kind in fractions for j in joints_of(kind)}
        if unverified is not None:
            return worst_by_joint(verts, faces, sk, offsets, {"gate": poses}, cover_mesh, cover, dome, joints,
                                  unverified=unverified)["gate"]
        return worst_by_joint(verts, faces, sk, offsets, {"gate": poses}, cover_mesh, cover, dome, joints)["gate"]

    def evaluate(fr: dict):
        try:
            return fr, worst_gaps(fr, requested), None
        except UnmeasurableJoints as exc:
            return fr, None, exc.reasons

    unverified = {} if fast else None
    zero = worst_gaps({}, unverified=unverified)
    eligible = sorted(k for kinds in GROUPS.values() for k in kinds
                      if all(j in zero for j in joints_of(k))
                      if max(zero[j] for j in joints_of(k)) >= gate - MARGIN_STUDS)
    chosen = {k: 0.0 for kinds in GROUPS.values() for k in kinds}
    trials = {}
    for group, kinds in GROUPS.items():
        movable = [k for k in kinds if k in eligible]
        if not movable:
            continue
        own = [j for k in kinds for j in joints_of(k) if j in zero]
        others = sorted((affected(kinds) - set(own)) & zero.keys())
        unmoved = max(zero[j] for j in own)
        best_score = float("inf")
        best = {}
        tried = [{"fractions": {}, "worst_gap": round(unmoved, 4), "score": round(unmoved, 4)}]
        candidates = [dict(zip(movable, combo)) for combo in itertools.product(FRACTIONS, repeat=len(movable))
                      if any(combo)]
        requested = set(own) | set(others)     # only the joints this group's cuts can change
        # map preserves product order even when a later candidate finishes first. The serial path also
        # keeps profiling straightforward and avoids thread overhead for a single worker.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = map(evaluate, candidates) if workers == 1 else pool.map(evaluate, candidates)
            for fr, g, missing in results:
                if missing is not None:
                    tried.append({"fractions": fr, "unmeasurable_joints": missing})
                    continue
                worst = max(g[j] for j in own)
                score = worst + NOISE_STUDS * sum(abs(f) for f in fr.values())
                # a move may not open a neighbouring joint by more than the noise
                worse = [j for j in others if g[j] > zero[j] + NOISE_STUDS]
                tried.append({"fractions": fr, "worst_gap": round(worst, 4), "score": round(score, 4),
                              "neighbours_opened": worse})
                if worst < unmoved - NOISE_STUDS and not worse and score < best_score:
                    best_score, best = score, fr
        chosen.update(best)
        trials[group] = {"joints": own, "neighbours": others, "tried": tried}
    offsets = {j: chosen[k] * radius[j] for kinds in GROUPS.values() for k in kinds for j in joints_of(k)}
    record = {"method": __doc__.strip().split("\n\n")[1].strip(),
              "noise_studs": NOISE_STUDS, "margin_studs": MARGIN_STUDS,
              "reference_triangles": int(len(faces)), "fractions": list(FRACTIONS),
              "radius_studs": {j: round(r, 4) for j, r in radius.items()},
              "unmoved_worst_gap": {j: round(v, 4) for j, v in zero.items()},
              "eligible": eligible, "trials": trials, "chosen_fraction": chosen,
              "offsets_studs": {j: round(o, 4) for j, o in offsets.items()},
              "seconds": round(time.monotonic() - t0, 1)}
    if fast:
        record["unverified_joints"] = unverified
    return offsets, record
