"""The rig spec: every part's size and placement, the joints and attachments, in the Roblox frame.

Each section's MeshPart is the axis-aligned box of its exact vertex bounds in `high.glb` (converted to R), so
the mesh Studio imports from the same GLB fills the part at scale 1. The joints sit at the `sections.json`
pivots. The HumanoidRootPart is Roblox's standard 2 x 2 x 1 box centred on the Root joint, and HipHeight puts
the lowest foot vertex on the floor (Humanoid.HipHeight: height above the floor = 0.5 * RootPart.Size.Y +
HipHeight).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import HRP_SIZE, MESH_TRIANGLE_LIMIT, R15, RIG_NAME, RobloxExportError, b_to_r, g_to_r, g_to_r_points
from .glb import mesh_nodes

FEET = ("LeftFoot", "RightFoot")
# Humanoid scale values. BodyTypeScale and BodyProportionScale 0 leave Classic parts at their own size.
SCALE_VALUES = (("BodyHeightScale", 1.0), ("BodyWidthScale", 1.0), ("BodyDepthScale", 1.0),
                ("HeadScale", 1.0), ("BodyTypeScale", 0.0), ("BodyProportionScale", 0.0))
AVATAR_PART_SCALE_TYPE = "Classic"
NOTE_STUDS = 1e-3  # GLB-versus-sections.json disagreements above this are noted in the record
PROFILE_HINT = "use the Roblox in-game or avatar profile"


def _rnd(v, n: int = 6) -> list[float]:
    return [round(float(x), n) for x in v]


def load_sections(body: Path) -> dict:
    path = Path(body) / "sections.json"
    try:
        sj = json.loads(path.read_text())
    except FileNotFoundError:
        raise RobloxExportError(f"missing input: {path}") from None
    except (OSError, ValueError) as exc:
        raise RobloxExportError(f"{path}: unreadable ({exc.__class__.__name__})") from None
    if not isinstance(sj, dict) or not isinstance(sj.get("sections"), list):
        raise RobloxExportError(f"{path}: no 'sections' list")
    return sj


def build_rig(body: Path, glb_name: str = "high.glb", name: str = RIG_NAME) -> dict:
    """Read a stage-4 body folder and return the rig spec (plain JSON-able data, in R, studs)."""
    body = Path(body)
    glb_path = body / glb_name
    sj = load_sections(body)
    if not glb_path.is_file():
        raise RobloxExportError(f"missing input: {glb_path}")
    if sj.get("units") != "studs":
        raise RobloxExportError(f"sections.json units are {sj.get('units')!r}, expected 'studs'")
    frame = str(sj.get("frame", ""))
    if "faces -Y" not in frame or "+Z up" not in frame:
        raise RobloxExportError(f"sections.json frame is {sj.get('frame')!r}; expected the contract frame "
                                "(+Z up, faces -Y)")
    nodes = mesh_nodes(glb_path)

    by_name = {}
    for s in sj["sections"]:
        if not isinstance(s, dict) or "name" not in s:
            raise RobloxExportError("sections.json has a section without a name")
        by_name[s["name"]] = s
    names = [p for p, _, _ in R15]
    missing, extra = sorted(set(names) - set(by_name)), sorted(set(by_name) - set(names))
    if missing or extra:
        raise RobloxExportError(f"sections.json is not the 15 R15 sections: missing {missing}, "
                                f"unexpected {extra}")
    missing = sorted(set(names) - set(nodes))
    if missing:
        raise RobloxExportError(f"{glb_name} has no mesh nodes named {missing}")
    over = [(p, nodes[p]["triangles"]) for p in names if nodes[p]["triangles"] > MESH_TRIANGLE_LIMIT]
    if over:
        worst = ", ".join(f"{p} has {n}" for p, n in over)
        raise RobloxExportError(f"{glb_name}: {worst} triangles, over Roblox's {MESH_TRIANGLE_LIMIT} per "
                                f"mesh; {PROFILE_HINT}")

    parts, notes, lows = [], [], {}
    for pname, parent, joint in R15:
        s = by_name[pname]
        want_parent = None if parent == "HumanoidRootPart" else parent
        if s.get("parent") != want_parent or s.get("joint") != joint:
            raise RobloxExportError(f"{pname}: sections.json has parent {s.get('parent')!r} and joint "
                                    f"{s.get('joint')!r}; R15 needs {want_parent!r} and {joint!r}")
        try:
            pivot = b_to_r(s["pivot"])
            sj_box = np.array([b_to_r(s["bounds_min"]), b_to_r(s["bounds_max"])])
        except (KeyError, IndexError, TypeError, ValueError):
            raise RobloxExportError(f"{pname}: sections.json needs a 3-number pivot, bounds_min and "
                                    "bounds_max") from None
        node = nodes[pname]
        pts = g_to_r_points(node["positions"])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        size = hi - lo
        if not np.all(np.isfinite(pts)) or float(size.min()) <= 0:
            raise RobloxExportError(f"{pname}: degenerate vertex bounds {size.tolist()}")
        lows[pname] = float(lo[1])
        d_node = max(abs(a - b) for a, b in zip(g_to_r(node["translation"]), pivot))
        d_bounds = float(max(np.abs(lo - sj_box.min(0)).max(), np.abs(hi - sj_box.max(0)).max()))
        if d_node > NOTE_STUDS:
            notes.append(f"{pname}: GLB node origin is {d_node:.4f} studs from the sections.json pivot")
        if d_bounds > NOTE_STUDS:
            notes.append(f"{pname}: GLB vertex bounds differ from sections.json bounds by {d_bounds:.4f} "
                         "studs")
        parts.append({
            "name": pname, "parent": parent, "joint": joint,
            "center": _rnd((lo + hi) / 2), "size": _rnd(size), "pivot": _rnd(pivot),
            "triangles": node["triangles"],
            "mesh_id": "",  # placeholder: bind_meshes.luau binds the imported mesh of GLB node `name`
            "glb_node_vs_pivot_studs": round(d_node, 6),
            "glb_vs_sections_bounds_studs": round(d_bounds, 6),
        })

    pc = {p["name"]: p for p in parts}
    feet_min_y = min(lows[f] for f in FEET)
    hrp_center = list(pc["LowerTorso"]["pivot"])
    hip_height = hrp_center[1] - HRP_SIZE[1] / 2 - feet_min_y
    if not math.isfinite(hip_height) or hip_height <= 0:
        raise RobloxExportError(f"HipHeight would be {hip_height:.3f} studs; the Root joint is too low for a "
                                "2-stud root part")

    def top(p):
        return p["center"][1] + p["size"][1] / 2

    def front(p):
        return p["center"][2] - p["size"][2] / 2

    def back(p):
        return p["center"][2] + p["size"][2] / 2

    head, ut, lt = pc["Head"], pc["UpperTorso"], pc["LowerTorso"]
    ls, rs = pc["LeftUpperArm"]["pivot"], pc["RightUpperArm"]["pivot"]
    hc = head["center"]
    # Accessory attachments (create.roblox.com/docs/avatar/character-bodies/specifications#attachments), each
    # in the region the specification names; world-aligned except the grips, which point forward (-90 deg
    # about X).
    extra = [
        ("HumanoidRootPart", "RootAttachment", hrp_center, None),
        ("Head", "FaceCenterAttachment", hc, None),
        ("Head", "FaceFrontAttachment", [hc[0], hc[1], front(head)], None),
        ("Head", "HatAttachment", [hc[0], top(head), hc[2]], None),
        ("Head", "HairAttachment", [hc[0], top(head), hc[2]], None),
        ("UpperTorso", "NeckAttachment", list(head["pivot"]), None),
        ("UpperTorso", "BodyFrontAttachment", [ut["center"][0], ut["center"][1], front(ut)], None),
        ("UpperTorso", "BodyBackAttachment", [ut["center"][0], ut["center"][1], back(ut)], None),
        ("UpperTorso", "LeftCollarAttachment", [ls[0], (ls[1] + top(ut)) / 2, ls[2]], None),
        ("UpperTorso", "RightCollarAttachment", [rs[0], (rs[1] + top(ut)) / 2, rs[2]], None),
        ("LowerTorso", "WaistCenterAttachment", list(lt["pivot"]), None),
        ("LowerTorso", "WaistFrontAttachment", [lt["pivot"][0], lt["pivot"][1], front(lt)], None),
        ("LowerTorso", "WaistBackAttachment", [lt["pivot"][0], lt["pivot"][1], back(lt)], None),
    ]
    for side in ("Left", "Right"):
        ua, hand, foot = pc[side + "UpperArm"], pc[side + "Hand"], pc[side + "Foot"]
        extra += [
            (side + "UpperArm", side + "ShoulderAttachment", [ua["center"][0], top(ua), ua["center"][2]],
             None),
            (side + "Hand", side + "GripAttachment", list(hand["center"]), [-90.0, 0.0, 0.0]),
            (side + "Foot", side + "FootAttachment",
             [foot["center"][0], foot["center"][1] - foot["size"][1] / 2, foot["center"][2]], None),
        ]
    return {
        "name": name,
        "frame": "Roblox: +Y up, faces -Z, character's left is -X; R = (-Bx, Bz, By) = (-Gx, Gy, -Gz); studs",
        "joints_kind": "Motor6D",
        "height": sj.get("height"),
        "feet_min_y": round(feet_min_y, 6),
        "lowest_min_y": round(min(lows.values()), 6),
        "hip_height": round(hip_height, 6),
        "hrp": {"center": _rnd(hrp_center), "size": list(HRP_SIZE)},
        "parts": parts,
        "extra_attachments": [{"part": a, "name": n, "position": _rnd(p), "orientation_deg": o}
                              for a, n, p, o in extra],
        "scale_values": dict(SCALE_VALUES),
        "avatar_part_scale_type": AVATAR_PART_SCALE_TYPE,
        "notes": notes,
    }
