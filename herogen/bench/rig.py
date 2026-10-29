"""Finding a humanoid's arms and legs from bone names, and the A-pose targets (pure Python + numpy).

Rigs name bones many ways: `upperarm.l` (KayKit), `arm-left` (Kenney), `upper_arm.L` / `DEF-upper_arm.L.001`
(Rigify), `mixamorig:LeftArm` (Mixamo), `Arm_L`, `arm_upper_L`. A name is split into a side (the character's
left or right, Blender's `.L` convention) and a base with prefixes, the side token and numeric suffixes
removed.
"""
from __future__ import annotations

import math
import re

import numpy as np

A_POSE_ARM_DEG = 45.0   # arms straight out and 45 degrees down
A_POSE_LEG_DEG = 5.0    # each leg 5 degrees out from vertical seen from the front ("slightly apart")

_PREFIX = re.compile(r"^(?:[a-z0-9_]*:|def[-_.]|org[-_.]|mch[-_.]|bip0?1[ _]?|b_|bn_)+")
_NUMBERED = re.compile(r"([._]\d{1,3})+$")
_SIDE_SUFFIX = re.compile(r"(?:[._\- ])(l|r|left|right)$")
_SIDE_PREFIX = re.compile(r"^(l|r|left|right)(?:[._\- ])")
_SIDE_WORD = re.compile(r"(left|right)")

# Base names of the bone that starts each limb, best first. `arm` / `leg` alone are last because Mixamo calls
# the shin `LeftLeg` (its thigh is `LeftUpLeg`), so a plain `leg` only counts when nothing better exists.
LIMB_ROOTS = {
    "arm": ("upperarm", "upper_arm", "arm_upper", "uparm", "up_arm", "arm"),
    "leg": ("upperleg", "upper_leg", "leg_upper", "upleg", "up_leg", "thigh", "leg"),
}
# Words that mark a bone as part of a limb (any rig layer: deform, mechanism, control), and words that keep a
# bone out of it: the shoulder and hip stay where they are.
LIMB_PARTS = {
    "arm": ("arm", "hand", "wrist", "palm", "elbow", "finger", "thumb", "index", "middle", "ring", "pinky",
            "handslot", "wep"),
    "leg": ("thigh", "leg", "shin", "calf", "knee", "foot", "ankle", "toe", "heel"),
}
LIMB_EXCLUDE = {"arm": ("shoulder", "clavicle", "collar", "armor", "armour"), "leg": ("hip", "pelvis")}
# Base names of the joint at the far end of each limb (its position gives the limb direction).
LIMB_ENDS = {
    "arm": ("wrist", "hand", "palm"),
    "leg": ("ankle", "foot"),
}


def split_name(name: str) -> tuple[str | None, str]:
    """(side, base) with side 'L', 'R' or None, e.g. 'DEF-upper_arm.L.001' -> ('L', 'upper_arm')."""
    s = _NUMBERED.sub("", name.strip().lower())
    s = _PREFIX.sub("", s)
    side = None
    m = _SIDE_SUFFIX.search(s)
    if m:
        side, s = m.group(1)[0].upper(), s[: m.start()]
    else:
        m = _SIDE_PREFIX.match(s)
        if m:
            side, s = m.group(1)[0].upper(), s[m.end():]
        else:
            m = _SIDE_WORD.search(s)
            if m:
                side, s = m.group(1)[0].upper(), (s[: m.start()] + s[m.end():])
    return side, _NUMBERED.sub("", s.strip("._- ")).strip("._- ")


def find_limb_root(names: list[str], limb: str, side: str) -> str | None:
    """The bone that starts `limb` ('arm' or 'leg') on `side` ('L' or 'R'): the best-ranked base name, then
    the shortest name (so `upper_arm.L` wins over `upper_arm.L.001`)."""
    ranked = []
    for n in names:
        s, base = split_name(n)
        if s == side and base in LIMB_ROOTS[limb]:
            ranked.append((LIMB_ROOTS[limb].index(base), len(n), n))
    return min(ranked)[2] if ranked else None


def limb_bones(names: list[str], limb: str, side: str) -> list[str]:
    """Every bone on `side` whose base name marks it as part of `limb`. Turning these as one rigid group turns
    the limb even in rigs whose deform bones hang from helper bones rather than from each other."""
    out = []
    for n in names:
        s, base = split_name(n)
        if s != side or any(w in base for w in LIMB_EXCLUDE[limb]):
            continue
        if any(w in base for w in LIMB_PARTS[limb]):
            out.append(n)
    return out


def find_limb_end(names: list[str], limb: str, side: str) -> str | None:
    ranked = []
    for n in names:
        s, base = split_name(n)
        if s == side and base in LIMB_ENDS[limb]:
            ranked.append((LIMB_ENDS[limb].index(base), len(n), n))
    return min(ranked)[2] if ranked else None


def target_direction(limb: str, side: str) -> np.ndarray:
    """Unit direction from the limb's root joint to its end in the contract frame (facing -Y, left at +X)."""
    out = 1.0 if side == "L" else -1.0
    if limb == "arm":
        a = math.radians(A_POSE_ARM_DEG)
        return np.array([out * math.cos(a), 0.0, -math.sin(a)])
    a = math.radians(A_POSE_LEG_DEG)
    return np.array([out * math.sin(a), 0.0, -math.cos(a)])


def frontal(v) -> np.ndarray:
    """The direction seen from the front: its forward/backward (Y) part dropped."""
    v = np.asarray(v, dtype=np.float64)
    return np.array([v[0], 0.0, v[2]])


def sagittal_lean(v) -> float:
    """Degrees a direction leans forward (-Y, positive) or backward from the frontal plane."""
    v = np.asarray(v, dtype=np.float64)
    return math.degrees(math.atan2(-v[1], math.hypot(v[0], v[2])))


def pose_aim(limb: str):
    """What the A-pose turns toward its target: an arm's whole direction (it ends in the frontal plane), but
    only a leg's frontal part (a turn about the forward axis), so a leg keeps the forward lean a large foot or
    the modelling gave it."""
    return (lambda v: np.asarray(v, dtype=np.float64)) if limb == "arm" else frontal


def angle_below_horizontal(v) -> float:
    """Degrees a direction points below the horizontal plane (90 = straight down)."""
    v = np.asarray(v, dtype=np.float64)
    return math.degrees(math.atan2(-v[2], math.hypot(v[0], v[1])))


def angle_from_vertical(v) -> float:
    """Degrees between a direction and straight down."""
    v = np.asarray(v, dtype=np.float64)
    return math.degrees(math.acos(np.clip(-v[2] / np.linalg.norm(v), -1.0, 1.0)))


def angle_between(a, b) -> float:
    """Degrees between two directions."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


def rotation_between(a, b) -> np.ndarray:
    """3x3 rotation taking direction `a` onto direction `b` about their common perpendicular (no twist)."""
    a = np.asarray(a, dtype=np.float64) / np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # opposite: half turn about any axis perpendicular to a
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        k = perp / np.linalg.norm(perp)
        return 2.0 * np.outer(k, k) - np.eye(3)
    k = v / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * kx + (1.0 - c) * (kx @ kx)
