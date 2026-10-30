"""Finding a humanoid's arms and legs from bone names, and the A-pose targets (pure Python + numpy).

Rigs name bones many ways: `upperarm.l` (KayKit), `arm-left` (Kenney), `upper_arm.L` / `DEF-upper_arm.L.001`
(Rigify), `mixamorig:LeftArm` (Mixamo), `J_Bip_L_UpperArm` (VRM), `Bip001 L UpperArm` (3ds Max),
`CC_Base_L_Thigh` (Character Creator), `Character1_LeftUpLeg` (HumanIK), `Upper Arm L`, `Arm_L`,
`arm_upper_L`; glTF exports may add a node number (`Head_08`) or the armature's name (`Head_Armature`). A name
is split into a side (the character's left or right, Blender's `.L` convention) and a base with prefixes,
the side token and numeric suffixes removed.
"""
from __future__ import annotations

import math
import re

import numpy as np

A_POSE_ARM_DEG = 45.0   # arms straight out and 45 degrees down
POSE_CONVERGED_DEG = 0.01  # a limb counts as posed when its direction is this close to the target
POSE_STEPS = 8             # turns allowed to get there
A_POSE_LEG_DEG = 5.0    # each leg 5 degrees out from vertical seen from the front ("slightly apart")

_PREFIX = re.compile(r"^(?:[a-z0-9_]*:|def[-_.]|org[-_.]|mch[-_.]|bip0*1_?|cc_base_|mixamorig\d*_|"
                     r"(?:quickrig)?character\d*_|b_|bn_)+")
_OBJECT_SUFFIX = re.compile(r"_armature$")    # glTF exports may append the armature object's name
_NUMBERED = re.compile(r"([._]\d{1,4})+$")    # `.001`, and glTF exports' node numbers (`Head_08`)
_SIDE_SUFFIX = re.compile(r"(?:[._\- ])(l|r|left|right)$")
_SIDE_PREFIX = re.compile(r"^(l|r|left|right)(?:[._\- ])")
_SIDE_WORD = re.compile(r"(left|right)")
_VRM = re.compile(r"^j_(?:bip|adj|sec)_([lcr])_")    # VRoid / VRM: J_Bip_L_UpperArm, J_Bip_C_Head

# Base names of the bone that starts each limb, best first. `arm` / `leg` alone are last because Mixamo calls
# the shin `LeftLeg` (its thigh is `LeftUpLeg`), so a plain `leg` only counts when nothing better exists.
LIMB_ROOTS = {
    "arm": ("upperarm", "upper_arm", "arm_upper", "uparm", "up_arm", "arm"),
    "leg": ("upperleg", "upper_leg", "leg_upper", "upleg", "up_leg", "thigh", "leg"),
}
# Name tokens (whole words) that mark a bone as part of a limb (any rig layer: deform, mechanism, control),
# and tokens that keep a bone out of it: the shoulder and hip stay where they are.
LIMB_PARTS = {
    "arm": frozenset(("arm", "upperarm", "lowerarm", "forearm", "uparm", "hand", "hands", "handslot",
                      "wrist", "palm", "elbow", "finger", "fingers", "thumb", "index", "middle", "ring",
                      "pinky", "wep")),
    "leg": frozenset(("leg", "upleg", "upperleg", "lowerleg", "thigh", "shin", "calf", "knee", "foot",
                      "ankle", "toe", "toes", "toebase", "heel")),
}
LIMB_EXCLUDE = {"arm": frozenset(("shoulder", "clavicle", "collar", "armor", "armour")),
                "leg": frozenset(("hip", "hips", "pelvis"))}
_TOKEN = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
# Base names of the joint at the far end of each limb (its position gives the limb direction).
LIMB_ENDS = {
    "arm": ("wrist", "hand", "palm"),
    "leg": ("ankle", "foot"),
}


def split_name(name: str) -> tuple[str | None, str]:
    """(side, base) with side 'L', 'R' or None, e.g. 'DEF-upper_arm.L.001' -> ('L', 'upper_arm')."""
    s = _NUMBERED.sub("", re.sub(r"\s+", "_", name.strip().lower()))
    s = _NUMBERED.sub("", _OBJECT_SUFFIX.sub("", s))
    s = _PREFIX.sub("", s)
    side = None
    m = _VRM.match(s)
    if m:
        return (None if m.group(1) == "c" else m.group(1).upper()), s[m.end():].strip("._- ")
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


def name_tokens(name: str) -> set[str]:
    """The words of a bone name, split at separators, digits and camelCase, lower-cased:
    'mixamorig:LeftHandIndex1' -> {'mixamorig', 'left', 'hand', 'index', '1'}."""
    return {t.lower() for t in _TOKEN.findall(name)}


def limb_bones(names: list[str], limb: str, side: str) -> list[str]:
    """Every bone on `side` with a name word that marks it as part of `limb` (whole words, so 'earring' is
    not a ring finger and 'handle' not a hand). Turning these as one rigid group turns the limb even in rigs
    whose deform bones hang from helper bones rather than from each other."""
    out = []
    for n in names:
        s, _ = split_name(n)
        words = name_tokens(n)
        if s != side or words & LIMB_EXCLUDE[limb]:
            continue
        if words & LIMB_PARTS[limb]:
            out.append(n)
    return out


def independent_roots(parents: dict[str, str | None], roots: list[str]) -> list[str]:
    """The roots with no other root above them: turning a bone turns everything below it, so a root that
    hangs (through other bones) under another root must not be turned again."""
    chosen = set(roots)
    out = []
    for r in roots:
        p, above = parents.get(r), False
        while p is not None:
            if p in chosen:
                above = True
                break
            p = parents.get(p)
        if not above:
            out.append(r)
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


# The head: the topmost bone with the word "head" in its name, or Rigify's head (`spine.006`) where nothing is
# so named; it and every bone below it (jaw, eyes, hair, hat) make the head region.
HEAD_WORD = "head"
RIGIFY_HEAD = re.compile(r"(?:^|[-_.:])spine\.006$", re.IGNORECASE)
# Material name words that mark a face material.
FACE_WORDS = frozenset(("face", "facial", "eye", "eyes", "eyeball", "iris", "pupil", "sclera", "cornea",
                        "brow", "brows", "eyebrow", "eyebrows", "lash", "lashes", "eyelash", "eyelashes",
                        "mouth", "lip", "lips", "teeth", "tooth", "tongue"))


def find_head(names: list[str], parents: dict[str, str | None], deform: set[str] | None = None) -> str | None:
    """The bone that carries the head: the first of `head_candidates`, or None."""
    found = head_candidates(names, parents, deform)
    return found[0] if found else None


def head_candidates(names: list[str], parents: dict[str, str | None],
                    deform: set[str] | None = None) -> list[str]:
    """Bones that may carry the head, best first: those named with the whole word `head` (`Head`,
    `mixamorig:Head`, `J_Bip_C_Head`; not `forehead`) and not hanging below another such bone, deform bones
    first, then shorter names. Rigify's `DEF-spine.006` when no bone is so named, or when none of those
    deforms the skin and it does. A rig may also carry a helper (`head_ref`) beside the real head; the caller
    takes the first candidate the skin actually follows."""
    named = [n for n in names if HEAD_WORD in name_tokens(n) and split_name(n)[0] is None]
    rigify = [n for n in names if RIGIFY_HEAD.search(n)]
    if not named or (deform is not None and not set(named) & deform and set(rigify) & deform):
        # Rigify names its head control `head` but deforms the skin with `DEF-spine.006`
        named = [n for n in rigify if deform is None or n in deform] or rigify
    chosen = set(named)
    top = []
    for n in named:
        p = parents.get(n)
        while p is not None and p not in chosen:
            p = parents.get(p)
        if p is None:
            top.append(n)
    return sorted(top, key=lambda n: (deform is not None and n not in deform, len(n), n))


def below(names: list[str], parents: dict[str, str | None], root: str) -> list[str]:
    """`root` and every bone that hangs below it."""
    out = []
    for n in names:
        p = n
        while p is not None and p != root:
            p = parents.get(p)
        if p == root:
            out.append(n)
    return out


def is_face_material(name: str) -> bool:
    """True when a material's name has a word that marks it as part of the face ('Face', 'MTL_Eyes',
    'EyeBrow'; not 'Surface' or 'Interface')."""
    return bool(name_tokens(name) & FACE_WORDS)
