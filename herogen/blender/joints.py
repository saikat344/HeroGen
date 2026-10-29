"""Joint estimation from the orthographic masks of an A-posed figure.

Front (or mirrored back) gives X and Z, the side view (left, or mirrored right) gives Y and Z, the top view
gives Y for the arm joints. Every joint comes from a silhouette feature:

- HeadTop: top of the figure. Head width: widest central row in the top 12%. Shoulder line: scanning down,
  the first row whose extent reaches 1.5x the head width (a scarf, collar or hair stays below that).
  Neck: the narrowest central band between the shoulder line and the widest head row when it is a real
  narrowing (<= 0.8x head width), capped by the chin from head symmetry (2 * z_widest - z_top); with no
  narrowing (scarf, collar), the chin estimate alone.
- Every joint height is bounded by an anthropometric range (neck 0.78-0.92 H, shoulders 0.68-0.88 H, waist
  0.50-0.66 H, crotch 0.38-0.56 H, ankles 0.02-0.12 H); `Skeleton.evidence` records what decided each joint.
- Shoulders: the arm is the part of the figure outside the torso's width below the armpit (found by scanning
  up from mid-body to the last row where the arm is still clear of the torso). A principal axis through those
  pixels gives the arm's line; the shoulder sits on it one arm radius below the shoulder line. Elbow and wrist sit at anthropometric fractions of
  the shoulder-to-fingertip length (0.42 and 0.75, from upper arm : forearm : hand = 0.186 : 0.146 : 0.108 of
  stature), and the wrist moves to a clear narrowing of the arm width if there is one.
- Hips: legs are the two runs either side of the body axis below the crotch (the highest row with a gap at
  the axis). Hip X is the leg centre just below the crotch, hip Z half a thigh radius above the crotch.
- Ankles: in the side view, the top of the foot (the highest low row clearly wider than the shin).
  Knees: halfway between hip and ankle (thigh and shank are about equal, 0.245 vs 0.246 of stature).
- Waist: centre of the narrowest band of the torso between crotch and armpit (fallback: 45% of the way up).
  Root (the LowerTorso pivot, R15's root attachment): halfway from the midpoint of the hips to the waist.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .views import ViewSet, label


@dataclass
class Plane:
    """A mask resampled so that axis 0 runs up in world Z (or Y for the top view) and axis 1 runs +world."""
    m: np.ndarray       # bool [n_rows, n_cols]
    h: np.ndarray       # world coordinate of each column centre (increasing)
    v: np.ndarray       # world coordinate of each row centre (increasing)
    px: float           # pixel size in studs

    def col(self, x: float) -> int:
        return int(np.clip(np.round((x - self.h[0]) / self.px), 0, len(self.h) - 1))

    def row(self, z: float) -> int:
        return int(np.clip(np.round((z - self.v[0]) / self.px), 0, len(self.v) - 1))

    def runs(self, r: int) -> list[tuple[float, float]]:
        """Runs of figure pixels in row r as (lo, hi) world edges."""
        line = self.m[r].astype(np.int8)
        d = np.diff(np.concatenate([[0], line, [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        return [(self.h[s] - self.px / 2, self.h[e - 1] + self.px / 2) for s, e in zip(starts, ends)]

    def run_at(self, r: int, x: float) -> tuple[float, float] | None:
        """The run of row r containing x, else the nearest run."""
        rs = self.runs(r)
        if not rs:
            return None
        for lo, hi in rs:
            if lo <= x <= hi:
                return lo, hi
        return min(rs, key=lambda s: min(abs(s[0] - x), abs(s[1] - x)))


def _plane(vs: ViewSet, primary: str, mirror: str) -> Plane | None:
    name = primary if primary in vs.views else (mirror if mirror in vs.views else None)
    if name is None:
        return None
    view = vs.views[name]
    u, v = view.pixel_uv()
    m = view.mask
    hu, vv = u[0], v[:, 0]
    if name == mirror:           # back/right: u = -X / -Y
        m = m[:, ::-1]
        hu = -hu[::-1]
    return Plane(m[::-1], hu, vv[::-1], 1.0 / view.pps)


@dataclass
class Skeleton:
    joints: dict[str, np.ndarray]
    ends: dict[str, np.ndarray]           # HandTip / Toe end points for the bones
    radii: dict[str, float]               # rough radius per body part, for the section split
    notes: list[str] = field(default_factory=list)
    arms: dict = field(default_factory=dict)   # side -> {"mask": front-plane bool, "dir": (dx, dz)}
    front: Plane | None = None
    evidence: dict = field(default_factory=dict)  # joint -> what decided it

    def as_json(self) -> dict:
        return {k: [round(float(c), 4) for c in v] for k, v in self.joints.items()}


def _band_centre(w: np.ndarray, z: np.ndarray, lo: int, hi: int, tol: float) -> tuple[float, float]:
    """Centre z and width of the contiguous band of near-minimal width around the minimum of w[lo:hi]."""
    seg = w[lo:hi]
    i = int(np.argmin(seg))
    wmin = seg[i]
    a = b = i
    while a > 0 and seg[a - 1] <= wmin * (1 + tol):
        a -= 1
    while b < len(seg) - 1 and seg[b + 1] <= wmin * (1 + tol):
        b += 1
    return float((z[lo + a] + z[lo + b]) / 2), float(wmin)


def estimate(vs: ViewSet, height: float) -> Skeleton:
    notes: list[str] = []
    F = _plane(vs, "front", "back")
    S = _plane(vs, "left", "right")
    T = vs.views.get("top")
    Tp = None
    if T is not None:
        u, v = T.pixel_uv()
        Tp = Plane(T.mask[::-1].T.copy(), v[::-1, 0], u[0], 1.0 / T.pps)  # rows = X, cols = Y
    if F is None:
        raise ValueError("joint estimation needs the front or back mask")
    top_derived = "visual_hull" in str((vs.meta.get("mask_method") or {}).get("top", ""))
    px = F.px
    rows = np.flatnonzero(F.m.any(axis=1))
    z_top = F.v[rows[-1]] + px / 2
    top_run = F.runs(rows[-1])
    x0 = float(np.mean([(a + b) / 2 for a, b in top_run]))

    def width_profile(P: Plane, centre: float) -> np.ndarray:
        w = np.zeros(len(P.v))
        for r in range(len(P.v)):
            run = P.run_at(r, centre)
            if run is not None and run[0] <= centre + 4 * P.px and run[1] >= centre - 4 * P.px:
                w[r] = run[1] - run[0]
        return w

    w = width_profile(F, x0)
    zr = F.v
    r_top = rows[-1]
    evidence: dict[str, str] = {}

    def clamp(name: str, z: float, lo: float, hi: float) -> float:
        """Bound a joint height to its anthropometric range (fractions of H); record when the bound decides."""
        zc = float(np.clip(z, lo * height, hi * height))
        if abs(zc - z) > 1e-9:
            evidence[name] = evidence.get(name, "") + f"; clamped from {z / height:.3f} H to [{lo}, {hi}] H"
            notes.append(f"{name} clamped to its range ({z / height:.3f} H -> {zc / height:.3f} H)")
        return zc

    # Head width: the widest central row in the top 12% of the figure, which is head (or hair) on any humanoid.
    r_head_lo = F.row(z_top - 0.12 * height)
    r_hw = r_head_lo + int(np.argmax(w[r_head_lo:r_top + 1]))
    head_w = float(w[r_hw])
    head_half_w = head_w / 2
    # Shoulder line: scanning down from the widest head row, the first row whose full extent reaches 1.5x the
    # head width (shoulders are about 2x head width; a scarf, collar or hair stays under 1.5x), within
    # 0.70-0.92 H. Scanning against a fixed head width, not a running maximum, so sloping shoulders and
    # collars that widen row by row cannot drag the line down.
    extent = np.array([(F.runs(r)[-1][1] - F.runs(r)[0][0]) if F.runs(r) else 0.0 for r in range(len(zr))])
    r_st = None
    for r in range(min(r_hw, F.row(0.92 * height)), F.row(0.70 * height), -1):
        if extent[r] >= 1.5 * head_w:
            r_st = r
            break
    if r_st is None:
        r_st = F.row(0.82 * height)
        evidence["ShoulderLine"] = "prior 0.82 H (no row reached 1.5x head width in 0.70-0.92 H)"
        notes.append("no shoulder line found; shoulder line at 0.82 H")
    else:
        evidence["ShoulderLine"] = (f"front extent reaches 1.5x head width ({head_w:.2f}) at "
                                    f"{zr[r_st] / height:.3f} H")
    shoulder_top = float(zr[r_st])
    # Neck: the narrowest central band between the shoulder line and the widest head row, if it is a real
    # narrowing (at most 0.8x head width); capped by the chin estimated from head symmetry (the widest row sits
    # about mid-head, so chin ~ 2 * z_widest - z_top), which a collar or scarf cannot hide. Range 0.78-0.92 H.
    chin = 2 * float(zr[r_hw]) - z_top
    band_z, neck_w = _band_centre(w, zr, r_st, r_hw, 0.04) if r_hw > r_st + 1 else (chin, head_w)
    if neck_w <= 0.8 * head_w:
        neck_z = min(band_z, chin) if chin > shoulder_top else band_z
        evidence["Neck"] = (f"narrowest front band at {band_z / height:.3f} H (width {neck_w / head_w:.2f}x head)"
                            + (f", capped by the chin from head symmetry at {chin / height:.3f} H"
                               if neck_z < band_z else ""))
    else:
        neck_z = max(chin, shoulder_top + 0.02 * height)
        evidence["Neck"] = (f"no narrowing (band {neck_w / head_w:.2f}x head width): chin from head symmetry "
                            f"at {chin / height:.3f} H")
    neck_z = clamp("Neck", neck_z, 0.78, 0.92)
    neck_w = float(w[F.row(neck_z)]) or neck_w
    # body axis: the neck's centre
    nr = F.run_at(F.row(neck_z), x0)
    x0 = (nr[0] + nr[1]) / 2

    def side_y(z: float, y_hint: float = 0.0) -> float:
        if S is None:
            return 0.0
        run = S.run_at(S.row(z), y_hint)
        return 0.0 if run is None else (run[0] + run[1]) / 2

    # Armpits and arms, per side.
    joints: dict[str, np.ndarray] = {}
    ends: dict[str, np.ndarray] = {}
    radii: dict[str, float] = {"head": head_half_w, "neck": neck_w / 2}
    armpit = {}
    torso_edge = {}
    r_neck = F.row(neck_z)
    for side, sgn in (("Left", 1), ("Right", -1)):
        def separated(r: int) -> bool:
            central = F.run_at(r, x0)
            if central is None:
                return False
            return any((a > central[1]) if sgn > 0 else (b < central[0]) for a, b in F.runs(r))

        # Scan up from mid-body, where an A-posed arm hangs clear of the torso, to the last separated row: the
        # armpit. (Scanning down from the neck would stop at any notch between a shoulder and the chest.)
        found = None
        for start in (0.55, 0.60, 0.50, 0.65, 0.45):
            r = F.row(start * height)
            if separated(r):
                while r + 1 < r_neck and separated(r + 1):
                    r += 1
                found = r
                break
        if found is None:
            notes.append(f"{side} arm never separates from the torso; armpit at 0.72 H")
            found = F.row(0.72 * height)
        armpit[side] = F.v[found]
        below = F.run_at(max(found - max(2, int(round(0.02 * height / px))), 0), x0)
        torso_edge[side] = below[1] if sgn > 0 else below[0]
    torso_half = (torso_edge["Left"] - torso_edge["Right"]) / 2
    radii["torso"] = torso_half

    hh, vv = np.meshgrid(F.h, F.v)
    arms: dict = {}
    for side, sgn in (("Left", 1), ("Right", -1)):
        edge = torso_edge[side]
        # the arm's line from the part hanging clear of the torso (below the armpit)
        sel = F.m & ((hh - edge) * sgn > 0) & (vv < armpit[side]) & (vv > 0.15 * height)
        # keep the piece hanging from the armpit (a wide thigh or boot can also reach past the torso edge)
        lab, n_lab = label(sel)
        if n_lab > 1:
            top = lab[(vv > armpit[side] - 0.05 * height) & sel]
            if len(top):
                sel = lab == int(np.bincount(top).argmax())
        pts = np.stack([hh[sel], vv[sel]], axis=1)
        if len(pts) < 20:
            notes.append(f"{side} arm: too few pixels ({len(pts)})")
            pts = np.array([[edge + sgn * 0.3, 0.7 * height], [edge + sgn * 1.0, 0.5 * height]])
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        d = vt[0]
        if d @ np.array([sgn, -1.0]) < 0:      # from the shoulder outwards and down
            d = -d
        nrm = np.array([-d[1], d[0]])
        # Refit the line through slice centres from 20% to 70% along the arm: the slices near the armpit are
        # cut by the selection and the hand may be offset, both of which tilt a plain principal axis.
        s = (pts - c) @ d
        t = (pts - c) @ nrm
        edges_s = np.linspace(s.min(), s.max(), 21)
        mids = []
        for i in range(4, 14):
            k = (s >= edges_s[i]) & (s < edges_s[i + 1])
            if k.sum() > 2:
                mids.append(c + (edges_s[i] + edges_s[i + 1]) / 2 * d + (t[k].max() + t[k].min()) / 2 * nrm)
        if len(mids) >= 3:
            mids = np.array(mids)
            c = mids.mean(axis=0)
            _, _, vt = np.linalg.svd(mids - c, full_matrices=False)
            d = vt[0] if vt[0] @ d > 0 else -vt[0]
            nrm = np.array([-d[1], d[0]])
            s = (pts - c) @ d
            t = (pts - c) @ nrm
        bins = np.linspace(s.min(), s.max(), 41)
        half = np.zeros(40)
        for i in range(40):
            k = (s >= bins[i]) & (s < bins[i + 1])
            half[i] = (t[k].max() - t[k].min() + px) / 2 if k.sum() > 1 else 0.0
        arm_r = float(np.median(half[8:24][half[8:24] > 0])) if (half[8:24] > 0).any() else 0.15
        # shoulder: on the arm line, one arm radius below the shoulder line; for a near-horizontal arm, where
        # the line is half an arm radius outside the torso edge
        if abs(d[1]) > 0.3:
            z_sh = clamp(side + "Shoulder", shoulder_top - arm_r, 0.68, 0.88)
            k_a = (z_sh - c[1]) / d[1]
            # Second estimate for sloping shoulders (pads, trapezius): walk up the arm axis from the armpit
            # while the silhouette above the axis stays an arm's thickness; the shoulder is where it grows
            # past 1.3 arm radii (the body takes over). The more outward of the two estimates wins.
            up = nrm if nrm[1] > 0 else -nrm
            k_ap = float((np.array([edge, armpit[side]]) - c) @ d)
            k_b = None
            for k in np.arange(k_ap, k_ap - 4 * arm_r - (k_ap - k_a if k_a < k_ap else 0), -px):
                p0 = c + k * d
                t = 0.0
                while t < 3 * arm_r:
                    q = p0 + (t + px) * up
                    if not F.m[F.row(q[1]), F.col(q[0])]:
                        break
                    t += px
                if t > 1.3 * arm_r:
                    k_b = k
                    break
            if k_b is not None and k_b > k_a:
                k_sh = k_b
                evidence[side + "Shoulder"] = ("arm axis where the silhouette above it grows past 1.3 arm radii "
                                               "(sloping shoulder)")
            else:
                k_sh = k_a
                evidence[side + "Shoulder"] = ("arm axis (front, below armpit) one arm radius below the shoulder "
                                               "line" + evidence.get(side + "Shoulder", ""))
        else:
            k_sh = (edge + sgn * 0.5 * arm_r - c[0]) / d[0]
            evidence[side + "Shoulder"] = "near-horizontal arm: axis at half an arm radius past the torso edge"
        sh2 = c + k_sh * d
        tip2 = c + s.max() * d
        length = s.max() - k_sh
        el2 = sh2 + 0.42 * length * d
        wr_frac = 0.75
        # wrist refinement: a clear narrowing between 60% and 88% of the arm
        prof_s = (bins[:-1] + bins[1:]) / 2 - k_sh
        win = (prof_s > 0.60 * length) & (prof_s < 0.88 * length) & (half > 0)
        if win.sum() >= 3:
            i_min = np.flatnonzero(win)[np.argmin(half[win])]
            left_max = half[(prof_s > 0.4 * length) & (prof_s < prof_s[i_min])].max(initial=0)
            right_max = half[(prof_s > prof_s[i_min]) & (half > 0)].max(initial=0)
            if half[i_min] < 0.85 * min(left_max, right_max):
                wr_frac = float(prof_s[i_min] / length)
                notes.append(f"{side} wrist from arm narrowing at {wr_frac:.2f} of the arm")
        wr2 = sh2 + wr_frac * length * d
        evidence[side + "Elbow"] = "0.42 of shoulder-to-fingertip along the arm axis (anthropometric)"
        evidence[side + "Wrist"] = (f"arm narrowing at {wr_frac:.2f} of the arm" if wr_frac != 0.75
                                    else "0.75 of shoulder-to-fingertip (anthropometric)")

        # Arm depth: from an observed top view at each joint's x; otherwise (no top view, or one stage 2
        # derived from the other views) one value for the whole arm, the side view's main body run at the
        # shoulder (an A-posed arm hangs in the body's plane).
        y_arm = side_y(float(sh2[1]))

        def arm_y(x: float, z: float) -> float:
            if Tp is not None and not top_derived:
                run = Tp.run_at(Tp.row(x), 0.0)
                if run is not None:
                    return (run[0] + run[1]) / 2
            return y_arm

        for name, p in (("Shoulder", sh2), ("Elbow", el2), ("Wrist", wr2)):
            joints[side + name] = np.array([p[0], arm_y(p[0], p[1]), p[1]])
        ends[side + "HandTip"] = np.array([tip2[0], arm_y(tip2[0], tip2[1]), tip2[1]])
        radii[side + "Arm"] = arm_r
        arms[side] = {"mask": sel, "dir": (float(d[0]), float(d[1])), "y": float(np.median(
            [joints[side + n][1] for n in ("Shoulder", "Elbow", "Wrist")]))}

    # Crotch and legs.
    r_ap = F.row(min(armpit.values()))
    c0 = F.col(x0)
    crotch_r = None
    for r in range(r_ap, F.row(0.15 * height), -1):
        if not F.m[r, c0]:
            rs = F.runs(r)
            if any(b < x0 for a, b in rs) and any(a > x0 for a, b in rs):
                crotch_r = r
                break
    if crotch_r is None:
        notes.append("no gap between the legs; crotch from proportions (0.47 H)")
        crotch_z = 0.47 * height
        evidence["Crotch"] = "prior 0.47 H (no gap between the legs in the front view)"
    else:
        crotch_z = F.v[crotch_r] + px / 2
        evidence["Crotch"] = f"highest front row with a gap at the body axis, {crotch_z / height:.3f} H"
    crotch_z = clamp("Crotch", crotch_z, 0.38, 0.56)
    crotch_r = F.row(crotch_z) - 1

    def leg_run(r: int, sgn: int):
        rs = [s for s in F.runs(r) if ((s[0] + s[1]) / 2 - x0) * sgn > 0]
        # not an arm (hands can hang beside the thighs), and not absurdly far out (spread boots are allowed)
        rs = [s for s in rs if abs((s[0] + s[1]) / 2 - x0) < torso_half * 2.5
              and not any(a["mask"][r, F.col((s[0] + s[1]) / 2)] for a in arms.values())]
        if not rs:
            return None
        return min(rs, key=lambda s: abs((s[0] + s[1]) / 2 - x0))

    # Ankle height from the side view: the top of the foot.
    if S is not None:
        ws = np.array([(max(b for a, b in S.runs(r)) - min(a for a, b in S.runs(r))) if S.runs(r) else 0.0
                       for r in range(len(S.v))])
        shin_w = float(np.median(ws[S.row(0.08 * height):S.row(0.18 * height)]))
        foot_top = None
        for r in range(S.row(0.15 * height), -1, -1):
            if ws[r] > 1.35 * shin_w:
                foot_top = S.v[r] + S.px / 2
                break
        if foot_top is None:
            notes.append("no foot visible in the side view; ankle at 0.06 H")
            foot_top = 0.06 * height
        r_foot = S.row(foot_top / 2)
        toe_y = min(a for a, b in S.runs(r_foot)) if S.runs(r_foot) else -0.3
        heel_y = max(b for a, b in S.runs(r_foot)) if S.runs(r_foot) else 0.3
    else:
        shin_w, foot_top, toe_y, heel_y = 0.3, 0.06 * height, -0.3, 0.3
        notes.append("no side view; ankle at 0.06 H")
    evidence["Ankle"] = (f"top of the foot in the side view, {foot_top / height:.3f} H"
                         if not any("ankle at 0.06 H" in n for n in notes) else "prior 0.06 H")
    ankle_z = clamp("Ankle", foot_top, 0.02, 0.12)
    for side, sgn in (("Left", 1), ("Right", -1)):
        runs = [leg_run(r, sgn) for r in range(F.row(crotch_z - 0.08 * height), crotch_r + 1)]
        runs = [s for s in runs if s is not None]
        if runs:
            hip_x = float(np.mean([(a + b) / 2 for a, b in runs]))
            thigh_r = float(np.median([(b - a) / 2 for a, b in runs]))
        else:
            hip_x, thigh_r = x0 + sgn * 0.5 * torso_half, 0.2
        hip_z = crotch_z + 0.5 * thigh_r

        def leg_x(z: float, fallback: float) -> float:
            run = leg_run(F.row(z), sgn)
            return fallback if run is None else (run[0] + run[1]) / 2

        ank_x = leg_x(ankle_z + 0.5 * shin_w, hip_x)
        knee_z = (hip_z + ankle_z) / 2
        knee_x = leg_x(knee_z, (hip_x + ank_x) / 2)
        evidence[side + "Hip"] = "leg-run centre below the crotch; half a thigh radius above the crotch"
        evidence[side + "Knee"] = "halfway from hip to ankle (anthropometric); x/y from the leg runs"
        evidence[side + "Ankle"] = evidence["Ankle"] + "; x from the front leg run"
        joints[side + "Hip"] = np.array([hip_x, side_y(hip_z), hip_z])
        joints[side + "Knee"] = np.array([knee_x, side_y(knee_z), knee_z])
        joints[side + "Ankle"] = np.array([ank_x, side_y(ankle_z + shin_w), ankle_z])
        ends[side + "Toe"] = np.array([ank_x, toe_y + 0.1 * (heel_y - toe_y), 0.4 * ankle_z])
        radii[side + "Leg"] = thigh_r
    radii["shin"] = shin_w / 2

    # Waist: narrowest torso band between crotch and armpit.
    ap = min(armpit.values())
    lo_r, hi_r = F.row(crotch_z + 0.15 * (ap - crotch_z)), F.row(ap - 0.1 * (ap - crotch_z))
    wz, wmin = _band_centre(w, zr, lo_r, hi_r, 0.03)
    r_w = F.row(wz)
    above, below = w[r_w:hi_r + 1].max(initial=0), w[lo_r:r_w + 1].max(initial=0)
    if wmin < 0.92 * min(above, below):
        waist_z = wz
        evidence["Waist"] = (f"narrowest torso band at {wz / height:.3f} H, "
                             f"{wmin / min(above, below):.2f}x the wider parts above and below")
    else:
        waist_z = crotch_z + 0.45 * (ap - crotch_z)
        evidence["Waist"] = "no narrowing with wider torso above and below: 45% from crotch to armpit (prior)"
        notes.append("no clear waist; waist at 45% from crotch to armpit")
    waist_z = clamp("Waist", waist_z, 0.50, 0.66)
    joints["HeadTop"] = np.array([x0, side_y(z_top - 0.1), z_top])
    joints["Neck"] = np.array([x0, side_y(neck_z), neck_z])
    # Waist and root depth: the median side-view centre over the torso band, so forearms that overlap the
    # torso in the side view (bent elbows) do not drag it forward.
    band = np.linspace(crotch_z, ap, 25)
    torso_y = float(np.median([side_y(z) for z in band]))
    joints["Waist"] = np.array([x0, torso_y, waist_z])
    hip_mid = (joints["LeftHip"] + joints["RightHip"]) / 2
    ends["HipCentre"] = hip_mid
    # R15 hangs the LowerTorso from the root at the LowerTorso's centre: halfway from the hips to the waist
    joints["Root"] = (hip_mid + joints["Waist"]) / 2
    joints["Root"][1] = torso_y
    radii["crotch_z"] = float(crotch_z)
    evidence["HeadTop"] = "top of the front silhouette"
    evidence["Root"] = "halfway from the hip midpoint to the waist"
    for side in ("Left", "Right"):
        evidence.setdefault(side + "Shoulder", "arm axis")
    return Skeleton(joints, ends, radii, notes, arms, F,
                    {k: v.lstrip("; ") for k, v in evidence.items()})


def joint_errors(est: dict, truth: dict) -> dict[str, float]:
    return {k: float(np.linalg.norm(np.asarray(est[k]) - np.asarray(truth[k]))) for k in truth if k in est}
