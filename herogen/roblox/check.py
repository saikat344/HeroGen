"""Check a `.rbxmx` R15 rig against the stage-4 body it was made from, without Studio.

The checker shares no code with the writer, so a writer bug can't hide itself:
- the file is read back with Python's XML parser, and CFrames are multiplied here;
- the R15 tree and the accessory attachment list are restated below rather than imported;
- the frame conversion is derived again (Blender -> glTF -> Roblox);
- part extents come from the GLB's POSITION accessor min/max (which glTF requires), not from the vertices the
  writer reads, and pivots and the toe points come from `sections.json`.

`check_rig` returns stage-record checks: {"name", "value", "threshold", "rule", "pass", "gate", "detail"}.
`rule` is the comparison the check passes on: tolerance checks pass when value <= threshold, count checks when
value == threshold, the orientation checks on a strict `<` or `>`; a check without a threshold has no rule.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from . import RobloxExportError
from .glb import read_chunks

POS_TOL = 1e-3     # studs: GLB min/max and the file are float32; sections.json pivots are rounded to 1e-4
REST_TOL = 1e-4    # studs: Part0 * C0 * C1^-1 against Part1
ROT_TOL = 1e-5     # rotation matrix entries
JOINT_SLACK = 0.05  # studs a joint may sit outside a part's box
TRI_LIMIT = 20000  # create.roblox.com/docs/art/modeling/specifications

# child: (parent, joint). Restated, not imported, so the checker can't inherit a writer typo.
R15 = {
    "LowerTorso": ("HumanoidRootPart", "Root"), "UpperTorso": ("LowerTorso", "Waist"),
    "Head": ("UpperTorso", "Neck"),
    "LeftUpperArm": ("UpperTorso", "LeftShoulder"), "LeftLowerArm": ("LeftUpperArm", "LeftElbow"),
    "LeftHand": ("LeftLowerArm", "LeftWrist"),
    "RightUpperArm": ("UpperTorso", "RightShoulder"), "RightLowerArm": ("RightUpperArm", "RightElbow"),
    "RightHand": ("RightLowerArm", "RightWrist"),
    "LeftUpperLeg": ("LowerTorso", "LeftHip"), "LeftLowerLeg": ("LeftUpperLeg", "LeftKnee"),
    "LeftFoot": ("LeftLowerLeg", "LeftAnkle"),
    "RightUpperLeg": ("LowerTorso", "RightHip"), "RightLowerLeg": ("RightUpperLeg", "RightKnee"),
    "RightFoot": ("RightLowerLeg", "RightAnkle"),
}
ACCESSORY = {
    "HumanoidRootPart": ("RootAttachment",),
    "Head": ("FaceCenterAttachment", "FaceFrontAttachment", "HatAttachment", "HairAttachment"),
    "UpperTorso": ("NeckAttachment", "BodyFrontAttachment", "BodyBackAttachment", "LeftCollarAttachment",
                   "RightCollarAttachment"),
    "LowerTorso": ("WaistCenterAttachment", "WaistFrontAttachment", "WaistBackAttachment"),
    "LeftUpperArm": ("LeftShoulderAttachment",), "RightUpperArm": ("RightShoulderAttachment",),
    "LeftHand": ("LeftGripAttachment",), "RightHand": ("RightGripAttachment",),
    "LeftFoot": ("LeftFootAttachment",), "RightFoot": ("RightFootAttachment",),
}
SCALE_VALUES = {"BodyHeightScale": 1.0, "BodyWidthScale": 1.0, "BodyDepthScale": 1.0, "HeadScale": 1.0,
                "BodyTypeScale": 0.0, "BodyProportionScale": 0.0}
CF_KEYS = ("X", "Y", "Z", "R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22")
BASEPARTS = {"Part", "MeshPart", "UnionOperation", "WedgePart", "CornerWedgePart", "TrussPart"}


# --- frames -------------------------------------------------------------------------------------------------
# Contract B: +Z up, faces -Y, left +X.  glTF G = (Bx, Bz, -By).  Roblox R: +Y up, faces -Z, left -X.
# B -> R: up Bz -> +Ry; front -By -> -Rz, so Rz = By; left +Bx -> -Rx, so Rx = -Bx.

def _b_to_r(p) -> np.ndarray:
    return np.array([-p[0], p[2], p[1]], dtype=np.float64)


def _g_to_r(p) -> np.ndarray:
    return _b_to_r((p[0], -p[2], p[1]))  # G -> B is (Gx, -Gz, Gy)


# --- CFrames as 4 x 4 matrices ------------------------------------------------------------------------------

def _cf(values) -> np.ndarray:
    x, y, z, *r = values
    m = np.eye(4)
    m[:3, :3] = np.array(r, dtype=np.float64).reshape(3, 3)
    m[:3, 3] = (x, y, z)
    return m


def _inv(m: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    rt = m[:3, :3].T
    out[:3, :3] = rt
    out[:3, 3] = -rt @ m[:3, 3]
    return out


def _cf_err(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(np.abs(a[:3, :3] - b[:3, :3]).max())


# --- reading the file ---------------------------------------------------------------------------------------

class Inst:
    def __init__(self, el: ET.Element, parent: "Inst | None"):
        self.cls = el.get("class", "")
        self.ref = el.get("referent", "")
        self.parent = parent
        self.props: dict[str, object] = {}
        props = el.find("Properties")
        for p in (props if props is not None else ()):
            self.props[p.get("name", "")] = _value(p)
        self.name = str(self.props.get("Name") or "")
        self.children = [Inst(c, self) for c in el.findall("Item")]

    def child(self, name: str, cls: str | None = None) -> "Inst | None":
        for c in self.children:
            if c.name == name and (cls is None or c.cls == cls):
                return c
        return None

    def cframe(self, prop: str = "CFrame") -> np.ndarray:
        v = self.props.get(prop)
        return _cf(v) if isinstance(v, tuple) and len(v) == 12 else np.eye(4)  # unwritten CFrame = identity

    def size(self) -> np.ndarray | None:
        v = self.props.get("size", self.props.get("Size"))
        return np.array(v, dtype=np.float64) if isinstance(v, tuple) and len(v) == 3 else None

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


def _value(p: ET.Element):
    tag, text = p.tag, (p.text or "").strip()
    try:
        if tag in ("float", "double", "int", "int64", "token"):
            return float(text)
        if tag == "bool":
            return text == "true"
        if tag == "CoordinateFrame":
            return tuple(float(p.find(k).text) for k in CF_KEYS)
        if tag == "Vector3":
            return tuple(float(p.find(k).text) for k in "XYZ")
        if tag == "Ref":
            return None if text in ("", "null") else ("ref", text)
        if tag == "Content":
            url = p.find("url")
            return (url.text or "").strip() if url is not None else ""
    except (AttributeError, TypeError, ValueError):
        return ("malformed", tag)
    return text


# --- the body -----------------------------------------------------------------------------------------------

def _glb_boxes(path: Path) -> dict[str, dict]:
    """Per mesh node: lo/hi in R from accessor min/max plus the node translation, and the triangle count."""
    doc, _ = read_chunks(path)
    out = {}
    for node in doc.get("nodes", []):
        if "mesh" not in node:
            continue
        t = node.get("translation", (0.0, 0.0, 0.0))
        lo, hi, tris = np.full(3, np.inf), np.full(3, -np.inf), 0
        for prim in doc["meshes"][node["mesh"]]["primitives"]:
            acc = doc["accessors"][prim["attributes"]["POSITION"]]
            lo = np.minimum(lo, np.add(acc["min"], t))
            hi = np.maximum(hi, np.add(acc["max"], t))
            count = doc["accessors"][prim["indices"]]["count"] if "indices" in prim else acc["count"]
            tris += count // 3
        a, b = _g_to_r(lo), _g_to_r(hi)
        out[node.get("name", "")] = {"lo": np.minimum(a, b), "hi": np.maximum(a, b), "tris": int(tris)}
    return out


def _abs_extent(cf: np.ndarray, size: np.ndarray) -> np.ndarray:
    return np.abs(cf[:3, :3]) @ size


def _inside(cf: np.ndarray, size: np.ndarray, world_point: np.ndarray, slack: float) -> bool:
    local = _inv(cf) @ np.append(world_point, 1.0)
    return bool(np.all(np.abs(local[:3]) <= size / 2 + slack))


# --- the checks ---------------------------------------------------------------------------------------------

def check_rig(body: Path, rig_path: Path, glb_name: str = "high.glb") -> list[dict]:
    """Every check of the rig file at `rig_path` against the body folder `body`."""
    try:
        return _check_rig(Path(body), Path(rig_path), glb_name)
    except OSError as exc:
        raise RobloxExportError(f"cannot read {exc.filename or 'an input'}: {exc.strerror or exc}") from None
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RobloxExportError(f"{body}: sections.json or {glb_name} is malformed "
                                f"({exc.__class__.__name__}: {exc})") from None


def _check_rig(body: Path, rig_path: Path, glb_name: str) -> list[dict]:
    checks: list[dict] = []

    def add(name, value, threshold, passed, detail="", gate=True, rule="=="):
        checks.append({"name": name, "value": value, "threshold": threshold,
                       "rule": rule if threshold is not None else None, "pass": bool(passed),
                       "gate": gate, "detail": detail})

    def tol(name, value, threshold, detail=""):
        value = float(value)
        add(name, round(value, 9), threshold, np.isfinite(value) and value <= threshold, detail, rule="<=")

    def count(name, value, want, detail=""):
        add(name, value, want, value == want, detail)

    sj = json.loads((body / "sections.json").read_text())
    sections = {s["name"]: s for s in sj["sections"]}
    boxes = _glb_boxes(body / glb_name)

    try:
        root = ET.parse(rig_path).getroot()
    except ET.ParseError as exc:
        add("xml_well_formed", 0, 1, False, str(exc))
        return checks
    roots = root.findall("Item")
    add("xml_well_formed", 1, 1, root.tag == "roblox" and root.get("version") == "4",
        f"root <{root.tag} version={root.get('version')!r}>")
    count("one_root_model", int(len(roots) == 1 and roots[0].get("class") == "Model"), 1,
          ",".join(r.get("class", "?") for r in roots))
    if not roots:
        return checks
    model = Inst(roots[0], None)
    bad = [f"{i.name or i.cls}.{k}" for i in model.walk() for k, v in i.props.items()
           if isinstance(v, tuple) and v[:1] == ("malformed",)]
    if bad:
        add("xml_values_valid", len(bad), 0, False, f"unreadable property values: {', '.join(bad[:5])}")
        return checks
    by_ref = {i.ref: i for i in model.walk() if i.ref}
    if len(by_ref) != sum(1 for _ in model.walk()):
        add("referents_unique", 0, 1, False, "missing or repeated referent")

    def deref(v):
        return by_ref.get(v[1]) if isinstance(v, tuple) and v and v[0] == "ref" else None

    # structure
    hums = [c for c in model.children if c.cls == "Humanoid"]
    hum = hums[0] if len(hums) == 1 else None
    count("humanoid_r15", int(hum is not None and hum.props.get("RigType") == 1.0), 1,
          f"{len(hums)} Humanoid, RigType {hum.props.get('RigType') if hum else None}")
    bad_scale = [k for k, v in SCALE_VALUES.items()
                 if not (hum and (nv := hum.child(k, "NumberValue")) and nv.props.get("Value") == v)]
    count("humanoid_scale_values", len(SCALE_VALUES) - len(bad_scale), len(SCALE_VALUES), ",".join(bad_scale))
    hrp = model.child("HumanoidRootPart")
    hrp_ok = (hrp is not None and hrp.cls == "Part" and deref(model.props.get("PrimaryPart")) is hrp
              and hrp.props.get("Transparency") == 1.0 and hrp.size() is not None)
    count("root_part", int(hrp_ok), 1, "HumanoidRootPart: a Part, the PrimaryPart, invisible")

    parts: dict[str, Inst] = {"HumanoidRootPart": hrp} if hrp else {}
    extra, dup = [], []
    for c in model.children:
        if c.cls in BASEPARTS and c is not hrp:
            if c.name in R15 and c.cls == "MeshPart":
                if c.name in parts:
                    dup.append(c.name)
                parts[c.name] = c
            else:
                extra.append(f"{c.name}:{c.cls}")
    missing = [n for n in R15 if n not in parts]
    problems = [f"missing {missing}" if missing else "", f"unexpected {extra}" if extra else "",
                f"repeated {dup}" if dup else ""]
    add("sections_r15_names", 15 - len(missing), 15, not (missing or extra or dup),
        "; ".join(p for p in problems if p) or "HumanoidRootPart and the 15 R15 MeshParts, nothing else")

    # joints and attachments
    worst = {"pivot": 0.0, "pair": 0.0, "c": 0.0, "rest": 0.0, "rot": 0.0}
    good_joints, att_pairs, outside, joint_bad = 0, 0, [], []
    for name, (pname, jname) in R15.items():
        part, parent = parts.get(name), parts.get(pname)
        if part is None or parent is None:
            joint_bad.append(f"{name}: part missing")
            continue
        joints = [c for c in part.children if c.cls in ("Motor6D", "AnimationConstraint")]
        att = jname + "RigAttachment"
        a0, a1 = parent.child(att, "Attachment"), part.child(att, "Attachment")
        if a0 is not None and a1 is not None:
            att_pairs += 1
        if len(joints) != 1 or joints[0].name != jname or joints[0].cls != "Motor6D":
            joint_bad.append(f"{name}: {[j.name + ':' + j.cls for j in joints]}")
            continue
        j = joints[0]
        if deref(j.props.get("Part0")) is not parent or deref(j.props.get("Part1")) is not part:
            joint_bad.append(f"{jname}: Part0/Part1 are not {pname}/{name}")
            continue
        good_joints += 1
        if a0 is None or a1 is None:
            continue
        c0, c1 = j.cframe("C0"), j.cframe("C1")
        worst["c"] = max(worst["c"], *(_cf_err(c0, a0.cframe())), *(_cf_err(c1, a1.cframe())))
        p0, p1 = parent.cframe(), part.cframe()
        pivot = _b_to_r(sections[name]["pivot"])
        w0, w1 = (p0 @ a0.cframe())[:3, 3], (p1 @ a1.cframe())[:3, 3]
        worst["pivot"] = max(worst["pivot"], float(np.linalg.norm(w0 - pivot)),
                             float(np.linalg.norm(w1 - pivot)))
        worst["pair"] = max(worst["pair"], float(np.linalg.norm(w0 - w1)))
        d, r = _cf_err(p0 @ c0 @ _inv(c1), p1)
        worst["rest"], worst["rot"] = max(worst["rest"], d), max(worst["rot"], r)
        for q in (parent, part):
            if q is hrp or q.size() is None:
                continue
            if not _inside(q.cframe(), q.size(), pivot, JOINT_SLACK):
                outside.append(f"{jname} outside {q.name}")
    count("joints_named_and_linked", good_joints, 15, "; ".join(joint_bad))
    count("rig_attachment_pairs", att_pairs, 15)
    tol("attachments_at_pivots", worst["pivot"], POS_TOL, "studs from the sections.json pivots")
    tol("attachment_pairs_coincide", worst["pair"], POS_TOL, "studs")
    tol("motor_c0_c1_equal_attachments", worst["c"], ROT_TOL)
    tol("rest_pose_position", worst["rest"], REST_TOL, "studs, Part0 * C0 * C1^-1 against Part1")
    tol("rest_pose_rotation", worst["rot"], ROT_TOL)
    count("joints_inside_parts", len(outside), 0, "; ".join(outside))

    # reachability from the root through the joints
    seen, frontier = {"HumanoidRootPart"}, ["HumanoidRootPart"]
    while frontier:
        cur = frontier.pop()
        for child, (pname, jname) in R15.items():
            if pname == cur and child not in seen and child in parts:
                j = parts[child].child(jname, "Motor6D")
                if j is not None and deref(j.props.get("Part0")) is parts.get(cur):
                    seen.add(child)
                    frontier.append(child)
    count("joint_chain_reachable", len(seen) - 1, 15, "sections reachable from HumanoidRootPart")

    # geometry against the GLB
    w_size, w_centre, w_rot, max_tris, no_node = 0.0, 0.0, 0.0, 0, []
    for name in R15:
        part, box = parts.get(name), boxes.get(name)
        if box is None:
            no_node.append(name)
            continue
        max_tris = max(max_tris, box["tris"])
        if part is None or part.size() is None:
            continue
        cf = part.cframe()
        w_size = max(w_size, float(np.abs(_abs_extent(cf, part.size()) - (box["hi"] - box["lo"])).max()))
        w_centre = max(w_centre, float(np.abs(cf[:3, 3] - (box["hi"] + box["lo"]) / 2).max()))
    for inst in model.walk():
        for prop in ("CFrame", "C0", "C1"):
            if isinstance(inst.props.get(prop), tuple):
                rot = inst.cframe(prop)[:3, :3]
                w_rot = max(w_rot, float(np.abs(rot @ rot.T - np.eye(3)).max()),
                            abs(float(np.linalg.det(rot)) - 1.0))
    if no_node:
        add("glb_has_sections", 15 - len(no_node), 15, False, f"GLB lacks {no_node}")
    tol("section_extents_match_glb", w_size, POS_TOL, "studs")
    tol("section_centres_match_glb", w_centre, POS_TOL, "studs")
    tol("rotations_proper", w_rot, ROT_TOL, "every CFrame orthonormal with determinant +1")
    add("triangles_per_mesh", max_tris, TRI_LIMIT, max_tris <= TRI_LIMIT, "most triangles in one section", rule="<=")

    # HipHeight: height above the floor = 0.5 * RootPart.Size.Y + HipHeight, floor = lowest foot vertex
    feet = [boxes[f]["lo"][1] for f in ("LeftFoot", "RightFoot") if f in boxes]
    if hum is not None and hrp is not None and hrp.size() is not None and len(feet) == 2:
        want = hrp.cframe()[1, 3] - hrp.size()[1] / 2 - min(feet)
        hh = float(hum.props.get("HipHeight", float("nan")))
        tol("hip_height", abs(hh - want), POS_TOL, f"HipHeight {hh:.4f}, feet on the floor at {want:.4f}")
        lowest = min(b["lo"][1] for b in boxes.values())
        tol("feet_lowest", min(feet) - lowest, POS_TOL, "studs any section reaches below the feet")
    else:
        add("hip_height", None, POS_TOL, False, "needs the Humanoid, HumanoidRootPart and both feet", rule="<=")

    # facing and handedness, from the body itself (toes ahead of the ankles; up is +Y)
    try:
        toe = (_b_to_r(sj["ends"]["LeftToe"]) + _b_to_r(sj["ends"]["RightToe"])) / 2
        ankle = (_b_to_r(sections["LeftFoot"]["pivot"]) + _b_to_r(sections["RightFoot"]["pivot"])) / 2
        fwd = np.array([toe[0] - ankle[0], 0.0, toe[2] - ankle[2]])
        fwd /= np.linalg.norm(fwd)
    except (KeyError, ZeroDivisionError, FloatingPointError):
        fwd = None
    add("faces_minus_z", None if fwd is None else round(float(fwd[2]), 4), -0.9,
        fwd is not None and fwd[2] < -0.9, "forward Z, from the toes ahead of the ankles", rule="<")
    lu, ru = parts.get("LeftUpperArm"), parts.get("RightUpperArm")
    dot = None
    if fwd is not None and lu is not None and ru is not None:
        right = np.cross(fwd, [0.0, 1.0, 0.0])
        rl = ru.cframe()[:3, 3] - lu.cframe()[:3, 3]
        dot = float(rl @ right / np.linalg.norm(rl))
    add("not_mirrored", None if dot is None else round(dot, 4), 0.9, dot is not None and dot > 0.9,
        "Right* parts on the character's right: cos(RightUpperArm - LeftUpperArm, right)", rule=">")
    head = parts.get("Head")
    ff = head.child("FaceFrontAttachment", "Attachment") if head else None
    dz = None if ff is None else round(float((head.cframe() @ ff.cframe())[2, 3] - head.cframe()[2, 3]), 4)
    add("face_front_on_front", dz, 0.0, dz is not None and dz < 0,
        "FaceFrontAttachment Z minus the head centre's Z (negative: on the front)", rule="<")

    # accessory attachments and scale type
    acc_missing, acc_outside, no_type = [], [], []
    for pname, names in ACCESSORY.items():
        part = parts.get(pname)
        for an in names:
            a = part.child(an, "Attachment") if part else None
            if a is None:
                acc_missing.append(an)
            elif pname != "HumanoidRootPart" and part.size() is not None:
                if not np.all(np.abs(a.cframe()[:3, 3]) <= part.size() / 2 + 1e-3):
                    acc_outside.append(an)
    for name in R15:
        st = parts[name].child("AvatarPartScaleType", "StringValue") if name in parts else None
        if st is None or st.props.get("Value") != "Classic":
            no_type.append(name)
    count("accessory_attachments", len(acc_missing) + len(acc_outside), 0,
          "; ".join(x for x in (f"missing {acc_missing}" if acc_missing else "",
                                f"outside their part {acc_outside}" if acc_outside else "") if x))
    count("avatar_part_scale_type", 15 - len(no_type), 15, ",".join(no_type))
    meshes = sum(1 for n in R15 if n in parts and parts[n].props.get("MeshId"))
    add("meshes_bound", meshes, None, True,
        "sections with a MeshId (0 until bind_meshes.luau runs in Studio)", gate=False)
    return checks


# --- the starter place --------------------------------------------------------------------------------------
# Restated, not imported from place.py, for the same reason as R15 above.
PLACE_SERVICES = ("Workspace", "Lighting", "StarterPlayer")
STARTER_MARKER = "HeroGenStarterPlace"


def check_place(place_path: Path, rig_path: Path) -> list[dict]:
    """Every check of the starter place at `place_path` against the rig file at `rig_path`: the place parses,
    holds the services, a baseplate, a spawn, the marker and exactly that rig, moved rigidly (turned about the
    vertical only) to stand on the baseplate clear of the spawn, facing it."""
    try:
        return _check_place(Path(place_path), Path(rig_path))
    except OSError as exc:
        raise RobloxExportError(f"cannot read {exc.filename or 'an input'}: {exc.strerror or exc}") from None


def _check_place(place_path: Path, rig_path: Path) -> list[dict]:
    checks: list[dict] = []

    def add(name, value, threshold, passed, detail="", rule="=="):
        checks.append({"name": "place_" + name, "value": value, "threshold": threshold,
                       "rule": rule if threshold is not None else None, "pass": bool(passed), "gate": True,
                       "detail": detail})

    def tol(name, value, threshold, detail=""):
        value = float(value)
        add(name, round(value, 9), threshold, np.isfinite(value) and value <= threshold, detail, rule="<=")

    def count(name, value, want, detail=""):
        add(name, value, want, value == want, detail)

    try:
        root = ET.parse(place_path).getroot()
    except ET.ParseError as exc:
        add("xml_well_formed", 0, 1, False, str(exc))
        return checks
    add("xml_well_formed", 1, 1, root.tag == "roblox" and root.get("version") == "4",
        f"root <{root.tag} version={root.get('version')!r}>")
    tops = [Inst(el, None) for el in root.findall("Item")]
    every = [i for t in tops for i in t.walk()]
    bad = [f"{i.name or i.cls}.{k}" for i in every for k, v in i.props.items()
           if isinstance(v, tuple) and v[:1] == ("malformed",)]
    if bad:
        add("xml_values_valid", len(bad), 0, False, f"unreadable property values: {', '.join(bad[:5])}")
        return checks
    by_ref = {i.ref: i for i in every if i.ref}
    dangling = [f"{i.name}.{k}" for i in every for k, v in i.props.items()
                if isinstance(v, tuple) and v[:1] == ("ref",) and v[1] not in by_ref]
    count("referents_resolve", int(len(by_ref) == len(every) and not dangling), 1,
          "; ".join(x for x in ("missing or repeated referent" if len(by_ref) != len(every) else "",
                                f"dangling {dangling[:5]}" if dangling else "") if x))

    services = {}
    for t in tops:
        services.setdefault(t.cls, []).append(t)
    have = [s for s in PLACE_SERVICES if len(services.get(s, ())) == 1 and services[s][0].name == s]
    count("services", len(have), len(PLACE_SERVICES),
          ", ".join(s for s in PLACE_SERVICES if s not in have) or "Workspace, Lighting, StarterPlayer, once each")
    ws = services["Workspace"][0] if "Workspace" in have else None
    if ws is None:
        return checks

    base = ws.child("Baseplate", "Part")
    base_ok = base is not None and base.props.get("Anchored") is True and base.size() is not None
    count("baseplate", int(base_ok), 1, "an anchored Part named Baseplate in Workspace")
    floor = (base.cframe()[1, 3] + _abs_extent(base.cframe(), base.size())[1] / 2) if base_ok else None

    spawns = [c for c in ws.walk() if c.cls == "SpawnLocation"]
    sp = spawns[0] if len(spawns) == 1 else None
    sp_ok = sp is not None and sp.props.get("Anchored") is True and sp.size() is not None
    count("spawn", int(sp_ok), 1, f"{len(spawns)} SpawnLocation; want one, anchored")
    if sp_ok and floor is not None:
        bottom = sp.cframe()[1, 3] - _abs_extent(sp.cframe(), sp.size())[1] / 2
        tol("spawn_on_baseplate", abs(bottom - floor), POS_TOL, "studs between the spawn's bottom and the floor")

    marker = [c for s in services.get("ServerStorage", ()) for c in s.children if c.name == STARTER_MARKER]
    count("marker", int(len(marker) == 1 and marker[0].cls == "BoolValue" and marker[0].props.get("Value") is True),
          1, f"ServerStorage.{STARTER_MARKER}: a BoolValue set to true")
    sps = services.get("StarterPlayer", [])
    count("starter_player_empty", sum(len(s.children) for s in sps), 0,
          "StarterCharacter is installed by bind_meshes.luau after the meshes are bound, never by the file")

    # the rig: the same instances as the .rbxmx, its parts moved by one rigid transform
    try:
        rig_root = ET.parse(rig_path).getroot().find("Item")
    except ET.ParseError:
        rig_root = None
    if rig_root is None:
        count("rig_present", 0, 1, f"{rig_path.name} holds no model")
        return checks
    want = Inst(rig_root, None)
    rigs = [c for c in ws.children if c.cls == "Model" and c.name == want.name]
    count("rig_present", len(rigs), 1, f"Model {want.name} directly in Workspace")
    if len(rigs) != 1:
        return checks
    rig = rigs[0]
    got_all, want_all = list(rig.walk()), list(want.walk())
    same_tree = [i.cls + ":" + i.name for i in got_all] == [i.cls + ":" + i.name for i in want_all]
    if not same_tree:
        count("rig_matches_file", 0, 1, f"the place's {rig.name} differs from {rig_path.name} in its instances")
        return checks
    # a Ref matches when it points at the same position in the two trees (referent numbers differ)
    g_ref = {i.ref: n for n, i in enumerate(got_all)}
    w_ref = {i.ref: n for n, i in enumerate(want_all)}
    g_hrp, w_hrp = rig.child("HumanoidRootPart"), want.child("HumanoidRootPart")
    if g_hrp is None or w_hrp is None:
        count("rig_matches_file", 0, 1, "no HumanoidRootPart")
        return checks
    t = g_hrp.cframe() @ _inv(w_hrp.cframe())
    worst_pos, worst_rot, differ = 0.0, 0.0, []
    for g, w in zip(got_all, want_all):
        for k in set(g.props) | set(w.props):
            gv, wv = g.props.get(k), w.props.get(k)
            if g.parent is rig and k == "CFrame" and g.cls in BASEPARTS:
                d, r = _cf_err(g.cframe(), t @ w.cframe())
                worst_pos, worst_rot = max(worst_pos, d), max(worst_rot, r)
            elif isinstance(gv, tuple) and gv[:1] == ("ref",) or isinstance(wv, tuple) and wv[:1] == ("ref",):
                gt = g_ref.get(gv[1]) if isinstance(gv, tuple) else None
                wt = w_ref.get(wv[1]) if isinstance(wv, tuple) else None
                if gt is None or wt is None or gt != wt:
                    differ.append(f"{g.name}.{k}")
            elif gv != wv:
                differ.append(f"{g.name}.{k}")
    count("rig_matches_file", int(not differ), 1,
          f"every value as in {rig_path.name} except the parts' CFrames" + (f"; differ: {differ[:5]}" if differ else ""))
    tol("rig_rigid_move_position", worst_pos, REST_TOL, "studs, each part against one transform of the file's rig")
    tol("rig_rigid_move_rotation", worst_rot, ROT_TOL)
    up = t[:3, :3] @ np.array([0.0, 1.0, 0.0])
    tol("rig_upright", float(np.abs(up - [0.0, 1.0, 0.0]).max()), ROT_TOL, "the move turns about the vertical only")

    parts = [c for c in rig.children if c.cls in BASEPARTS and c.size() is not None]
    feet = [c for c in parts if c.name in ("LeftFoot", "RightFoot")]
    if floor is not None and len(feet) == 2:
        lowest = min(f.cframe()[1, 3] - _abs_extent(f.cframe(), f.size())[1] / 2 for f in feet)
        tol("feet_on_baseplate", abs(lowest - floor), POS_TOL, "studs between the lowest foot and the floor")
    else:
        add("feet_on_baseplate", None, POS_TOL, False, "needs the baseplate and both feet", rule="<=")
    if sp_ok and parts:
        lo = np.min([p.cframe()[:3, 3] - _abs_extent(p.cframe(), p.size()) / 2 for p in parts], axis=0)
        hi = np.max([p.cframe()[:3, 3] + _abs_extent(p.cframe(), p.size()) / 2 for p in parts], axis=0)
        s_c, s_h = sp.cframe()[:3, 3], _abs_extent(sp.cframe(), sp.size()) / 2
        gap = max(float(s_c[0] - s_h[0] - hi[0]), float(lo[0] - s_c[0] - s_h[0]),
                  float(s_c[2] - s_h[2] - hi[2]), float(lo[2] - s_c[2] - s_h[2]))
        add("rig_clear_of_spawn", round(gap, 6), 0.0, gap > 0.0, "studs between the rig's and the spawn's footprints",
            rule=">")
        fwd = t[:3, :3] @ np.array([0.0, 0.0, -1.0])  # the rig faces -Z in its own file
        to_spawn = np.array([s_c[0] - g_hrp.cframe()[0, 3], 0.0, s_c[2] - g_hrp.cframe()[2, 3]])
        cos = float(fwd @ to_spawn / max(np.linalg.norm(to_spawn), 1e-9))
        add("rig_faces_spawn", round(cos, 4), 0.9, cos > 0.9, "cos(the rig's forward, the way to the spawn)", rule=">")
    return checks
