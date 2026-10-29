"""herogen stage 6: a Roblox R15 rig model (`.rbxmx`) for a stage-4 body, plus the Studio scripts that bind
its meshes, checked against the body it came from.

Frames (all in studs):
  Contract (Blender) frame B: +Z up, faces -Y, the character's left is +X, feet at z = 0.
  GLB frame G (glTF):         +Y up, faces +Z; G = (Bx, Bz, -By).
  Roblox frame R:             +Y up, faces -Z (a part's LookVector), the character's left is -X.
                              R = (-Bx, Bz, By) = (-Gx, Gy, -Gz): a 180 degree turn about Y from G,
                              not a mirror.

A MeshPart in a model file can only reference an uploaded mesh asset; it cannot carry the geometry. The rig
is therefore written with every MeshId empty, and `bind_meshes.luau` binds the meshes Studio's importer makes
from `high.glb` onto it. See docs/ARCHITECTURE.md, "Roblox export".
"""
from __future__ import annotations

import numpy as np

# The R15 tree, root first: (part, parent part, joint). The joint's Motor6D lives in the child part and is
# named after the joint; both parts carry an Attachment named <joint>RigAttachment (as
# Humanoid:BuildRigFromAttachments expects). The checker restates this table on purpose (see check.py).
R15 = (
    ("LowerTorso", "HumanoidRootPart", "Root"),
    ("UpperTorso", "LowerTorso", "Waist"),
    ("Head", "UpperTorso", "Neck"),
    ("LeftUpperArm", "UpperTorso", "LeftShoulder"),
    ("LeftLowerArm", "LeftUpperArm", "LeftElbow"),
    ("LeftHand", "LeftLowerArm", "LeftWrist"),
    ("RightUpperArm", "UpperTorso", "RightShoulder"),
    ("RightLowerArm", "RightUpperArm", "RightElbow"),
    ("RightHand", "RightLowerArm", "RightWrist"),
    ("LeftUpperLeg", "LowerTorso", "LeftHip"),
    ("LeftLowerLeg", "LeftUpperLeg", "LeftKnee"),
    ("LeftFoot", "LeftLowerLeg", "LeftAnkle"),
    ("RightUpperLeg", "LowerTorso", "RightHip"),
    ("RightLowerLeg", "RightUpperLeg", "RightKnee"),
    ("RightFoot", "RightLowerLeg", "RightAnkle"),
)

# create.roblox.com/docs/art/modeling/specifications: a mesh "can not exceed 20,000 triangles".
MESH_TRIANGLE_LIMIT = 20000
# The HumanoidRootPart size Roblox's own R15 Package Importer sample uses.
HRP_SIZE = (2.0, 2.0, 1.0)

RIG_NAME = "HeroGen-R15"
RIG_FILE = RIG_NAME + ".rbxmx"
BIND_FILE = "bind_meshes.luau"
RIG_TEST_FILE = "rig_test.server.luau"
README_FILE = "README.txt"
RECORD_FILE = "stage-6.json"
OUTPUT_FILES = (RIG_FILE, BIND_FILE, RIG_TEST_FILE, README_FILE, RECORD_FILE)


class RobloxExportError(Exception):
    """A body that can't be exported, or a broken input file. The message is one line, meant for the user."""


def b_to_r(p) -> list[float]:
    """Contract (Blender) frame to Roblox frame: (-Bx, Bz, By)."""
    return [-float(p[0]), float(p[2]), float(p[1])]


def g_to_r(p) -> list[float]:
    """glTF frame to Roblox frame: (-Gx, Gy, -Gz)."""
    return [-float(p[0]), float(p[1]), -float(p[2])]


def g_to_r_points(pts: np.ndarray) -> np.ndarray:
    """g_to_r for every row of an (n, 3) array."""
    return np.asarray(pts, dtype=np.float64) * np.array([-1.0, 1.0, -1.0])
