"""herogen stages 3 and 4: silhouette-carved base mesh, then a sectioned R15 game body.

Stage 3 carves a voxel grid with the orthographic masks of `docs/ARCHITECTURE.md`, surfaces it with surface
nets (or builds it with SAM 3D Objects from the front view), and checks silhouette IoU against the masks. Stage 4
takes the joints (from the masks, or SAM 3D Body's from `03-joints.json`), cuts the mesh into the 15 R15 sections
with planes at the joints, closes each cut with a dome centred on the pivot, builds the LODs a target profile
asks for (`PROFILES`) and runs a pose test that measures the largest gap at each joint.

Heavy geometry runs in numpy in this process; Blender (headless, with a timeout on every call) decimates,
writes the GLB files and renders.
"""
from __future__ import annotations

# Contract: R15-style rigid sections, in contract order.
SECTIONS = (
    "Head", "UpperTorso", "LowerTorso",
    "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
    "RightUpperLeg", "RightLowerLeg", "RightFoot",
)

# Each section's parent section and the joint its pivot sits on (R15 naming; LowerTorso hangs from the root).
PARENT = {
    "Head": "UpperTorso", "UpperTorso": "LowerTorso", "LowerTorso": None,
    "LeftUpperArm": "UpperTorso", "LeftLowerArm": "LeftUpperArm", "LeftHand": "LeftLowerArm",
    "RightUpperArm": "UpperTorso", "RightLowerArm": "RightUpperArm", "RightHand": "RightLowerArm",
    "LeftUpperLeg": "LowerTorso", "LeftLowerLeg": "LeftUpperLeg", "LeftFoot": "LeftLowerLeg",
    "RightUpperLeg": "LowerTorso", "RightLowerLeg": "RightUpperLeg", "RightFoot": "RightLowerLeg",
}
PIVOT_JOINT = {
    "Head": "Neck", "UpperTorso": "Waist", "LowerTorso": "Root",
    "LeftUpperArm": "LeftShoulder", "LeftLowerArm": "LeftElbow", "LeftHand": "LeftWrist",
    "RightUpperArm": "RightShoulder", "RightLowerArm": "RightElbow", "RightHand": "RightWrist",
    "LeftUpperLeg": "LeftHip", "LeftLowerLeg": "LeftKnee", "LeftFoot": "LeftAnkle",
    "RightUpperLeg": "RightHip", "RightLowerLeg": "RightKnee", "RightFoot": "RightAnkle",
}

# Contract defaults. The in-game budgets match the contract example; for reference, ZBurst's current stand-in
# hero (HeroV6, also 15 rigid R15 sections) ships at about 23.6k high / 5.7k low triangles, and ZBurst keeps every
# MeshPart under 20k triangles, which a single section here never approaches.
HEIGHT_STUDS = 5.5
TRIANGLES_HIGH_MAX = 24000
TRIANGLES_LOW_MAX = 6000

# Target profiles: what stage 4 builds and what the scorer checks. The default target is the best-quality body,
# with no triangle budget; Roblox's in-game and avatar budgets are opt-in.
# - best: the high body keeps the stage 3 mesh's full detail (no decimation, no triangle budget); no low LOD.
# - roblox-ingame: a high and a low LOD within TRIANGLES_HIGH_MAX / TRIANGLES_LOW_MAX.
# - roblox-avatar: one body that meets the triangle, watertight-part and texture limits of the Roblox character
#   body specification for the Marketplace (create.roblox.com/docs/art/characters/specifications): 10,742
#   triangles in total, per part group Head 4,000, Torso 1,750 and each arm and leg 1,248; every part watertight;
#   textures at most 2048 px. One LOD (Roblox makes its own). The rest of the specification is not built
#   (AVATAR_NOT_BUILT), so the body is not a complete avatar body.
AVATAR_GROUPS = {
    "Head": (("Head",), 4000),
    "Torso": (("UpperTorso", "LowerTorso"), 1750),
    "LeftArm": (("LeftUpperArm", "LeftLowerArm", "LeftHand"), 1248),
    "RightArm": (("RightUpperArm", "RightLowerArm", "RightHand"), 1248),
    "LeftLeg": (("LeftUpperLeg", "LeftLowerLeg", "LeftFoot"), 1248),
    "RightLeg": (("RightUpperLeg", "RightLowerLeg", "RightFoot"), 1248),
}
AVATAR_NOT_BUILT = ("R15 bone rig with skinning", "attachments", "facial animation (DynamicHead)",
                    "cages (inner and outer, for layered clothing)")
PROFILES = {
    "best": {"lods": {"high": None}},
    "roblox-ingame": {"lods": {"high": TRIANGLES_HIGH_MAX, "low": TRIANGLES_LOW_MAX}},
    "roblox-avatar": {"lods": {"high": "groups"}, "triangles_total": 10742, "groups": AVATAR_GROUPS,
                      "texture_max_px": 2048, "watertight_parts": True,
                      "not_built": AVATAR_NOT_BUILT},
}
DEFAULT_PROFILE = "best"

VIEWS = ("front", "back", "left", "right", "top")
