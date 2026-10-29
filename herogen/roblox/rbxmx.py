"""Write a rig spec (rig.py) as a Roblox XML model file (`.rbxmx`, `<roblox version="4">`).

The file holds only the properties the rig sets; Studio gives every other property its class default, as it
does for files written by rbx-dom. Floats are written as float32 in their shortest round-trip decimal form, so
the same body always gives the same bytes.

Tree:
  Model (PrimaryPart = HumanoidRootPart)
    Humanoid (RigType R15, HipHeight)  > 6 NumberValue scale values
    Part HumanoidRootPart (2 x 2 x 1, Transparency 1)  > RootRigAttachment, RootAttachment
    MeshPart <section> x 15, each with
      StringValue AvatarPartScaleType, Attachment <joint>RigAttachment, Motor6D <joint>,
      the <child joint>RigAttachments of its child sections, and its accessory attachments
Every MeshId is left empty (see bind_meshes.luau).
"""
from __future__ import annotations

import math
from xml.sax.saxutils import escape

import numpy as np

R15_RIG_TYPE = 1  # Enum.HumanoidRigType.R15
IDENTITY = np.eye(3)


def _num32(x: float) -> str:
    s = np.format_float_positional(np.float32(x), unique=True, trim="-")
    return "0" if s == "-0" else s


def _num64(x: float) -> str:
    s = np.format_float_positional(np.float64(x), unique=True, trim="-")
    return "0" if s == "-0" else s


def angles(deg) -> np.ndarray:
    """CFrame.Angles(rx, ry, rz) in degrees, i.e. Rx * Ry * Rz, with exact zeros for right angles."""
    def cs(a: float) -> tuple[float, float]:
        c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
        return (0.0 if abs(c) < 1e-12 else c), (0.0 if abs(s) < 1e-12 else s)
    (cx, sx), (cy, sy), (cz, sz) = cs(deg[0]), cs(deg[1]), cs(deg[2])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rx @ ry @ rz


class CFrame:
    """A rigid transform: position and a 3 x 3 rotation (columns are the right, up and back vectors)."""

    def __init__(self, pos, rot=None):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.rot = IDENTITY.copy() if rot is None else np.asarray(rot, dtype=np.float64)

    def inverse(self) -> "CFrame":
        rt = self.rot.T
        return CFrame(-(rt @ self.pos), rt)

    def __mul__(self, other: "CFrame") -> "CFrame":
        return CFrame(self.pos + self.rot @ other.pos, self.rot @ other.rot)


class Item:
    def __init__(self, cls: str, name: str):
        self.cls = cls
        self.props: list[tuple[str, str, object]] = [("string", "Name", name)]
        self.children: list[Item] = []
        self.ref = ""

    def prop(self, kind: str, name: str, value) -> "Item":
        self.props.append((kind, name, value))
        return self

    def add(self, child: "Item") -> "Item":
        self.children.append(child)
        return child


def _value_xml(kind: str, name: str, value, pad: str) -> list[str]:
    if kind == "string":
        return [f'{pad}<string name="{name}">{escape(str(value))}</string>']
    if kind == "bool":
        return [f'{pad}<bool name="{name}">{"true" if value else "false"}</bool>']
    if kind == "float":
        return [f'{pad}<float name="{name}">{_num32(value)}</float>']
    if kind == "double":
        return [f'{pad}<double name="{name}">{_num64(value)}</double>']
    if kind == "token":
        return [f'{pad}<token name="{name}">{int(value)}</token>']
    if kind == "Ref":
        return [f'{pad}<Ref name="{name}">{value.ref if value is not None else "null"}</Ref>']
    if kind == "Vector3":
        x, y, z = value
        return [f'{pad}<Vector3 name="{name}">', f"{pad}  <X>{_num32(x)}</X>", f"{pad}  <Y>{_num32(y)}</Y>",
                f"{pad}  <Z>{_num32(z)}</Z>", f"{pad}</Vector3>"]
    if kind == "CoordinateFrame":
        out = [f'{pad}<CoordinateFrame name="{name}">']
        for tag, v in zip("XYZ", value.pos):
            out.append(f"{pad}  <{tag}>{_num32(v)}</{tag}>")
        for i in range(3):
            for j in range(3):
                out.append(f"{pad}  <R{i}{j}>{_num32(value.rot[i, j])}</R{i}{j}>")
        out.append(f"{pad}</CoordinateFrame>")
        return out
    raise ValueError(f"unknown property type {kind}")


def _assign_refs(item: Item, counter: list[int]) -> None:
    item.ref = str(counter[0])
    counter[0] += 1
    for c in item.children:
        _assign_refs(c, counter)


def _item_xml(item: Item, depth: int, out: list[str]) -> None:
    pad = "  " * depth
    out.append(f'{pad}<Item class="{item.cls}" referent="{item.ref}">')
    out.append(f"{pad}  <Properties>")
    for kind, name, value in item.props:
        out.extend(_value_xml(kind, name, value, pad + "    "))
    out.append(f"{pad}  </Properties>")
    for c in item.children:
        _item_xml(c, depth + 1, out)
    out.append(f"{pad}</Item>")


def build_model(spec: dict) -> Item:
    """The instance tree for a rig spec."""
    model = Item("Model", spec["name"])
    hum = model.add(Item("Humanoid", "Humanoid"))
    hum.prop("float", "HipHeight", spec["hip_height"]).prop("token", "RigType", R15_RIG_TYPE)
    for key, value in spec["scale_values"].items():
        hum.add(Item("NumberValue", key)).prop("double", "Value", value)

    hrp_cf = CFrame(spec["hrp"]["center"])
    hrp = model.add(Item("Part", "HumanoidRootPart"))
    hrp.prop("CoordinateFrame", "CFrame", hrp_cf).prop("Vector3", "size", spec["hrp"]["size"])
    hrp.prop("float", "Transparency", 1)
    model.prop("Ref", "PrimaryPart", hrp)

    items, frames = {"HumanoidRootPart": hrp}, {"HumanoidRootPart": hrp_cf}
    for p in spec["parts"]:
        cf = CFrame(p["center"])
        part = model.add(Item("MeshPart", p["name"]))
        part.prop("bool", "Anchored", False).prop("CoordinateFrame", "CFrame", cf)
        part.prop("Vector3", "size", p["size"])
        scale_type = part.add(Item("StringValue", "AvatarPartScaleType"))
        scale_type.prop("string", "Value", spec["avatar_part_scale_type"])
        items[p["name"]], frames[p["name"]] = part, cf

    def attach(owner: str, name: str, world: CFrame) -> CFrame:
        local = frames[owner].inverse() * world
        items[owner].add(Item("Attachment", name)).prop("CoordinateFrame", "CFrame", local)
        return local

    # Rig attachments: one per joint in each of its two parts, both at the joint's pivot, world-aligned at
    # rest. Motor6D C0 and C1 are the two attachments' CFrames, as Humanoid:BuildRigFromAttachments sets them.
    child_atts = []
    for p in spec["parts"]:
        world = CFrame(p["pivot"])
        att = p["joint"] + "RigAttachment"
        c1 = attach(p["name"], att, world)
        c0 = frames[p["parent"]].inverse() * world
        m = items[p["name"]].add(Item("Motor6D", p["joint"]))
        m.prop("CoordinateFrame", "C0", c0).prop("CoordinateFrame", "C1", c1)
        m.prop("Ref", "Part0", items[p["parent"]]).prop("Ref", "Part1", items[p["name"]])
        child_atts.append((p["parent"], att, world))
    for owner, att, world in child_atts:
        attach(owner, att, world)
    for e in spec["extra_attachments"]:
        rot = angles(e["orientation_deg"]) if e["orientation_deg"] else None
        attach(e["part"], e["name"], CFrame(e["position"], rot))
    return model


def write_rbxmx(spec: dict) -> str:
    """The `.rbxmx` text for a rig spec."""
    model = build_model(spec)
    _assign_refs(model, [0])
    out = ['<roblox version="4">']
    _item_xml(model, 1, out)
    out.append("</roblox>")
    return "\n".join(out) + "\n"
