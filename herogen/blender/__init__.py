"""herogen stages 3 and 4: silhouette-carved base mesh, then a sectioned R15 game body.

Stage 3 carves a voxel grid with the orthographic masks of `docs/ARCHITECTURE.md`, surfaces it with surface
nets, and checks silhouette IoU against the masks. Stage 4 estimates the joints from the masks, cuts the mesh
into the 15 R15 sections with planes at the joints, closes each cut with a dome centred on the pivot, builds
high and low LODs and runs a pose test that measures the largest gap at each joint.

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

# Contract defaults. The budgets match the contract example; for reference, ZBurst's current stand-in hero
# (HeroV6, also 15 rigid R15 sections) ships at about 23.6k high / 5.7k low triangles, and ZBurst keeps every
# MeshPart under 20k triangles, which a single section here never approaches.
HEIGHT_STUDS = 5.5
TRIANGLES_HIGH_MAX = 24000
TRIANGLES_LOW_MAX = 6000

VIEWS = ("front", "back", "left", "right", "top")
