"""Write the starter place (`HeroGen-Starter.rbxlx`): a Roblox XML place file (`<roblox version="4">`, the
format of the rig's `.rbxmx`, with the place's services as its top-level items) that holds the rig ready to play.

Tree:
  Workspace
    Part Baseplate (anchored, 512 x 20 x 512, its top at Y = 0)
    SpawnLocation (anchored, 12 x 1 x 12, on the baseplate at the origin, no spawn force field)
    Model HeroGen-R15 (the rig of rbxmx.build_model), turned to face the spawn, its feet on the baseplate
  Lighting (daylight: the ambient, brightness and time of day of Studio's Baseplate template)
  StarterPlayer (no children; LoadCharacterAppearance off, so a StarterCharacter keeps its own look)
  ServerScriptService
  ServerStorage
    BoolValue HeroGenStarterPlace (true): tells bind_meshes.luau it runs in this place

Studio adds every service and property the file leaves out with its default, as it does for places written by
rbx-dom. The place holds no meshes: bind_meshes.luau binds the meshes Studio imports from `high.glb`, then, seeing
the marker, installs a copy of the bound rig as StarterPlayer.StarterCharacter.
"""
from __future__ import annotations

from . import STARTER_MARKER
from .rbxmx import CFrame, Item, angles, build_model, write_items

BASEPLATE_SIZE = (512.0, 20.0, 512.0)
BASEPLATE_CENTRE = (0.0, -10.0, 0.0)  # top face at Y = 0, the floor the rig's HipHeight is measured from
BASEPLATE_COLOUR = (91, 93, 105)
SPAWN_SIZE = (12.0, 1.0, 12.0)
SPAWN_CENTRE = (0.0, 0.5, 0.0)
# The floor point under the rig's root part: 16 studs ahead of the spawn's centre along -Z, the way a player
# spawned there faces, so the player sees the hero's front.
RIG_SPOT = (0.0, 0.0, -16.0)
RIG_YAW_DEG = 180.0  # the rig faces -Z; turned about Y it faces +Z, towards the spawn
LIGHTING = {"ambient": (70, 70, 70), "outdoor_ambient": (70, 70, 70), "brightness": 3.0, "clock_time": 14.5}
TECHNOLOGY_SHADOWMAP = 3  # Enum.Technology.ShadowMap
MATERIAL_PLASTIC = 256  # Enum.Material.Plastic
SURFACE_SMOOTH = 0  # Enum.SurfaceType.Smooth


def rig_placement(spec: dict) -> CFrame:
    """The rigid transform from the rig's own frame into the place: a turn of RIG_YAW_DEG about the vertical
    through the root part, the root part over RIG_SPOT, and the lowest foot vertex on the baseplate's top."""
    rot = angles((0.0, RIG_YAW_DEG, 0.0))
    hx, _, hz = spec["hrp"]["center"]
    lift = BASEPLATE_CENTRE[1] + BASEPLATE_SIZE[1] / 2 - spec["feet_min_y"]
    target = CFrame((RIG_SPOT[0], RIG_SPOT[1] + lift, RIG_SPOT[2]))
    return target * CFrame((0.0, 0.0, 0.0), rot) * CFrame((-hx, 0.0, -hz))


def _move(model: Item, placement: CFrame) -> None:
    """Move every part of a model by `placement`. Attachments and Motor6D C0/C1 are part-relative, so they
    stay as written."""
    for child in model.children:
        child.props = [(kind, name, placement * value if name == "CFrame" else value)
                       for kind, name, value in child.props]


def _part(cls: str, name: str, centre, size, colour=None) -> Item:
    p = Item(cls, name)
    p.prop("bool", "Anchored", True).prop("CoordinateFrame", "CFrame", CFrame(centre))
    p.prop("Vector3", "size", size)
    p.prop("token", "TopSurface", SURFACE_SMOOTH).prop("token", "BottomSurface", SURFACE_SMOOTH)
    if colour is not None:
        p.prop("Color3uint8", "Color3uint8", colour)
    return p


def build_place(spec: dict) -> list[Item]:
    """The place's services, in file order."""
    workspace = Item("Workspace", "Workspace")
    base = workspace.add(_part("Part", "Baseplate", BASEPLATE_CENTRE, BASEPLATE_SIZE, BASEPLATE_COLOUR))
    base.prop("bool", "Locked", True).prop("token", "Material", MATERIAL_PLASTIC)
    spawn = workspace.add(_part("SpawnLocation", "SpawnLocation", SPAWN_CENTRE, SPAWN_SIZE))
    spawn.prop("bool", "Neutral", True).prop("int", "Duration", 0)  # no force field bubble on the hero
    rig = workspace.add(build_model(spec))
    _move(rig, rig_placement(spec))

    lighting = Item("Lighting", "Lighting")
    lighting.prop("Color3", "Ambient", [c / 255 for c in LIGHTING["ambient"]])
    lighting.prop("Color3", "OutdoorAmbient", [c / 255 for c in LIGHTING["outdoor_ambient"]])
    lighting.prop("float", "Brightness", LIGHTING["brightness"]).prop("float", "ClockTime", LIGHTING["clock_time"])
    lighting.prop("bool", "GlobalShadows", True).prop("token", "Technology", TECHNOLOGY_SHADOWMAP)
    lighting.prop("float", "EnvironmentDiffuseScale", 1).prop("float", "EnvironmentSpecularScale", 1)

    starter_player = Item("StarterPlayer", "StarterPlayer")
    starter_player.prop("bool", "LoadCharacterAppearance", False)

    server_storage = Item("ServerStorage", "ServerStorage")
    server_storage.add(Item("BoolValue", STARTER_MARKER)).prop("bool", "Value", True)
    return [workspace, lighting, starter_player, Item("ServerScriptService", "ServerScriptService"), server_storage]


def write_rbxlx(spec: dict) -> str:
    """The `.rbxlx` text of the starter place for a rig spec."""
    return write_items(build_place(spec))


def place_summary(spec: dict) -> dict:
    """What the stage record says about the place."""
    t = rig_placement(spec)
    return {
        "rig_root_part": [round(float(v), 6) for v in (t * CFrame(spec["hrp"]["center"])).pos],
        "rig_yaw_deg": RIG_YAW_DEG,
        "baseplate": {"centre": list(BASEPLATE_CENTRE), "size": list(BASEPLATE_SIZE)},
        "spawn": {"centre": list(SPAWN_CENTRE), "size": list(SPAWN_SIZE)},
        "marker": f"ServerStorage.{STARTER_MARKER}",
        "meshes": "none; bind_meshes.luau binds them and installs StarterPlayer.StarterCharacter",
    }
