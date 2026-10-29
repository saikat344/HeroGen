"""Runs inside Blender:

    blender --background --factory-startup --disable-autoexec --python bench_job.py -- job.json

One job renders one asset:

1. Load the asset (.glb/.gltf, .fbx or .blend) and choose its mesh objects: the visible ones, or those whose
   names match `include`, minus those matching `exclude`.
2. Rigs: clear actions, mute drivers and bone constraints, return every bone to its rest pose. With
   `bind_unrigged`, meshes the rig does not deform get the weights of the nearest deformed mesh.
3. Pose: "a-pose" turns each arm so its root-to-end direction points out and 45 degrees down, and each leg
   about the forward axis so it points 5 degrees out from vertical seen from the front (its forward or
   backward lean stays as modelled). Each limb turns as one piece about its root joint; elbows and knees keep
   their rest bend. "rest" keeps the rest pose; "as-shipped" keeps the file's own pose and animation frame.
   Before posing, the figure's facing is read from the rig: the right-to-left hip (else shoulder) vector must
   point along +X, which with +Z up means the figure faces -Y. With no `yaw_deg` the turn that achieves this
   is found from the rig (to the nearest 90 degrees); a given `yaw_deg` that disagrees with the rig is an
   error, and an unrigged figure needs `yaw_deg`, since nothing in it says which way it faces.
4. Freeze the posed meshes into the contract frame (turned by the yaw to face -Y), scale to `height`, and
   export `gt_glb`.
5. For every camera: a mask and a depth map by casting one orthographic ray through each pixel centre against
   the frozen mesh, then a flat (unlit base colour) and a lit colour render with Cycles, on a transparent
   background.

Writes `<job>.result.json` next to the job file, and `<work>/<view>.npz` (mask, depth) and
`<work>/<view>-{flat,lit}.png` for the caller to turn into the run folder.
"""
import json
import math
import os
import re
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

JOB = json.loads(open(sys.argv[sys.argv.index("--") + 1]).read())
sys.path.insert(0, JOB["repo_root"])

from herogen.bench import cameras as CAM  # noqa: E402
from herogen.bench import rig as RIG  # noqa: E402
from herogen.bench import texture as TEX  # noqa: E402

IMAGE_EXTS = (".png", ".tga", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".exr")
FACING_TOLERANCE_DEG = 45.0  # a given yaw must put the rig's left side within this of +X
DISTAL_FRACTION = 0.85


# --------------------------------------------------------------------------------------------- loading

def load_asset(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".blend":
        bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=path)
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=path)
        else:
            raise ValueError(f"unsupported asset type {ext!r} (use .glb, .gltf, .fbx or .blend)")
    return bpy.context.scene


SKIPPED_TYPES = ("CURVE", "SURFACE", "META", "FONT", "CURVES", "POINTCLOUD", "VOLUME", "GPENCIL",
                 "GREASEPENCIL")


def skipped_geometry(scene, chosen):
    """Visible geometry the bench does not render and the true mesh does not contain: non-mesh objects,
    particle systems (hair) on the chosen meshes, and instancers."""
    out = []
    for o in scene.objects:
        if not o.visible_get() or o.hide_render:
            continue
        if o.type in SKIPPED_TYPES:
            out.append(f"{o.name} ({o.type.lower()})")
        if o.instance_type != "NONE" and (o.instance_type != "COLLECTION" or o.instance_collection):
            out.append(f"{o.name} (instances: {o.instance_type.lower()})")
    for o in chosen:
        for ps in o.particle_systems:
            if ps.settings.render_type != "NONE":
                out.append(f"{o.name} (particles: {ps.name}, {ps.settings.type.lower()})")
    return out


def choose_meshes(scene, include, exclude):
    inc = re.compile(include) if include else None
    exc = re.compile(exclude) if exclude else None
    chosen = []
    for o in scene.objects:
        if o.type != "MESH" or len(o.data.polygons) == 0:
            continue
        keep = bool(inc.search(o.name)) if inc else (o.visible_get() and not o.hide_render)
        if keep and exc and exc.search(o.name):
            keep = False
        if keep:
            chosen.append(o)
    if not chosen:
        raise ValueError("no mesh objects selected (check include/exclude)")
    return chosen


def rigs_of(meshes):
    arms = []
    for o in meshes:
        for m in o.modifiers:
            if m.type == "ARMATURE" and m.object is not None and m.object not in arms:
                arms.append(m.object)
        p = o.parent
        if p is not None and p.type == "ARMATURE" and p not in arms and o.parent_type == "BONE":
            arms.append(p)
    return arms


def deforming_rig(o):
    """The armature that deforms or carries this mesh, or None."""
    for m in o.modifiers:
        if m.type == "ARMATURE" and m.object is not None and m.show_render:
            return m.object
    if o.parent is not None and o.parent.type == "ARMATURE" and o.parent_type == "BONE":
        return o.parent
    return None


# --------------------------------------------------------------------------------------------- rig reset

def clear_animation(scene, report):
    """Stop object and shape-key animation from moving anything (rigs are reset by `reset_rig`): actions
    unassigned, NLA tracks and drivers muted. Values stay as the file left them at its current frame."""
    targets = [(o.name, "object", o) for o in scene.objects if o.type != "ARMATURE"]
    targets += [(o.name, "shape-key", o.data.shape_keys) for o in scene.objects
                if o.type == "MESH" and o.data.shape_keys is not None]
    for name, label, idb in targets:
        ad = idb.animation_data
        if ad is None:
            continue
        if ad.action is not None:
            report.append(f"{name}: cleared {label} action {ad.action.name!r}")
            ad.action = None
        for track in ad.nla_tracks:
            track.mute = True
        for fc in ad.drivers:
            fc.mute = True


def reset_rig(arm, report):
    ad = arm.animation_data
    if ad is not None:
        if ad.action is not None:
            report.append(f"{arm.name}: cleared action {ad.action.name!r}")
            ad.action = None
        for track in ad.nla_tracks:
            track.mute = True
        for fc in ad.drivers:
            fc.mute = True
    muted = 0
    for pb in arm.pose.bones:
        for c in pb.constraints:
            if c.enabled:
                c.enabled = False
                muted += 1
        pb.matrix_basis = Matrix.Identity(4)
    if muted:
        report.append(f"{arm.name}: muted {muted} bone constraints")
    arm.data.pose_position = "POSE"


def weight_source(arm):
    """One temporary mesh joining every mesh in the file that `arm` deforms (hidden ones too), at rest, with
    their vertex groups: the skin to copy weights from."""
    parts = [o for o in bpy.context.scene.objects
             if o.type == "MESH" and o.data.polygons and o.vertex_groups and deforming_rig(o) is arm]
    if not parts:
        return None, []
    copies = []
    for o in parts:
        c = o.copy()
        c.data = o.data.copy()
        c.modifiers.clear()
        c.parent = None
        c.matrix_world = o.matrix_world.copy()
        bpy.context.scene.collection.objects.link(c)
        copies.append(c)
    with bpy.context.temp_override(active_object=copies[0], selected_editable_objects=copies,
                                   selected_objects=copies):
        bpy.ops.object.join()
    return copies[0], [o.name for o in parts]


def bind_unrigged(meshes, report):
    """Bind meshes no rig deforms to the rig that deforms the others, with the weights of its nearest skin
    (at rest)."""
    rigs = [r for r in (deforming_rig(o) for o in meshes) if r is not None and r.type == "ARMATURE"]
    if not rigs:
        return
    arm = max(set(rigs), key=rigs.count)
    loose = [o for o in meshes if deforming_rig(o) is None and o.parent_type != "BONE"]
    if not loose:
        return
    source, names = weight_source(arm)
    if source is None:
        report.append(f"bind_unrigged: {arm.name} deforms no mesh with vertex groups; nothing bound")
        return
    for o in loose:
        for g in source.vertex_groups:
            if o.vertex_groups.get(g.name) is None:
                o.vertex_groups.new(name=g.name)
        dt = o.modifiers.new("BenchWeights", "DATA_TRANSFER")
        dt.object = source
        dt.use_object_transform = True
        dt.use_vert_data = True
        dt.data_types_verts = {"VGROUP_WEIGHTS"}
        dt.vert_mapping = "POLYINTERP_NEAREST"
        dt.layers_vgroup_select_src = "ALL"
        dt.layers_vgroup_select_dst = "NAME"
        with bpy.context.temp_override(object=o, active_object=o):
            bpy.ops.object.modifier_apply(modifier=dt.name)
        mod = o.modifiers.new("BenchArmature", "ARMATURE")
        mod.object = arm
        report.append(f"{o.name}: bound to {arm.name} with weights from the nearest of {', '.join(names)}")
    bpy.data.objects.remove(source, do_unlink=True)


# --------------------------------------------------------------------------------------------- posing

def bone_world_head(arm, name):
    return np.array(arm.matrix_world @ arm.pose.bones[name].head)


def limb_group(arm, limb, side, root_name):
    """The limb's bones: those named as its parts on its side, plus everything below its root bone."""
    out = set(RIG.limb_bones([b.name for b in arm.data.bones], limb, side))

    def walk(b):
        out.add(b.name)
        for c in b.children:
            walk(c)
    walk(arm.data.bones[root_name])
    return sorted(out)


def group_roots(arm, group):
    """Bones of the group whose parent is outside it and that hang under no other such bone: turning these
    turns the whole group exactly once."""
    g = set(group)
    roots = [n for n in group if arm.data.bones[n].parent is None or arm.data.bones[n].parent.name not in g]
    parents = {b.name: (b.parent.name if b.parent else None) for b in arm.data.bones}
    return RIG.independent_roots(parents, roots)


def chain_tail(arm, root):
    """The tail of the last bone reached by following first children down from `root`, and that bone."""
    b = arm.data.bones[root]
    while b.children:
        b = sorted(b.children, key=lambda c: c.name)[0]
    return np.array(arm.matrix_world @ arm.pose.bones[b.name].tail), b.name


def limb_end_point(arm, limb, side, root, group, meshes):
    """Where the limb ends: its wrist/hand (ankle/foot) joint if the rig has one below the root; else the tail
    of the limb's bone chain when it has two or more bones; else (a one-bone limb, whose bone axis need not
    follow the geometry) the centre of the limb's far end: the mean of its weighted vertices that lie at least
    `DISTAL_FRACTION` of the furthest one's distance from the root joint (the hand or foot block, not a
    single corner of it)."""
    deform = [n for n in group if arm.data.bones[n].use_deform]
    end = RIG.find_limb_end(deform, limb, side) or RIG.find_limb_end(group, limb, side)
    if end is not None:
        return bone_world_head(arm, end), f"joint {end}"
    if len(group) >= 2:
        tail, last = chain_tail(arm, root)
        return tail, f"tail of {last}"
    pivot = bone_world_head(arm, root)
    pts = []
    gset = set(group)
    dg = bpy.context.evaluated_depsgraph_get()
    for o in meshes:
        if deforming_rig(o) is not arm:
            continue
        idx = {g.index: g.name for g in o.vertex_groups}
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        mw = np.array(ev.matrix_world)
        for v_src, v_ev in zip(o.data.vertices, me.vertices):
            if not v_src.groups:
                continue
            g = max(v_src.groups, key=lambda g: g.weight)
            if idx.get(g.group) not in gset:
                continue
            pts.append(mw[:3, :3] @ np.array(v_ev.co) + mw[:3, 3])
        ev.to_mesh_clear()
    if not pts:
        return None, "no vertices weighted to the limb"
    pts = np.array(pts)
    d = np.linalg.norm(pts - pivot, axis=1)
    far = pts[d >= DISTAL_FRACTION * d.max()]
    return far.mean(axis=0), f"centre of the {len(far)} furthest weighted vertices"


def rotate_bones_world(arm, names, rot3, pivot_world):
    """Turn pose bones (and with them everything parented below) by one world rotation about a world point.
    The bones must not be parents of one another."""
    mw = arm.matrix_world
    r = Matrix([list(row) for row in rot3]).to_4x4()
    t = Matrix.Translation(Vector(pivot_world))
    turn = mw.inverted() @ t @ r @ t.inverted() @ mw
    targets = {n: turn @ arm.pose.bones[n].matrix for n in names}
    for n, m in targets.items():
        arm.pose.bones[n].matrix = m
    bpy.context.view_layer.update()


def rig_left_axis(arms):
    """World vector from the right to the left hip (else shoulder) joint of the first rig that has both, and
    how it was found; (None, reason) when no rig names a left and a right limb. The hips come first: they
    carry the body's facing, while a rest pose may twist the shoulders."""
    for arm in arms:
        deform = [b.name for b in arm.data.bones if b.use_deform] or [b.name for b in arm.data.bones]
        for limb in ("leg", "arm"):
            left, right = RIG.find_limb_root(deform, limb, "L"), RIG.find_limb_root(deform, limb, "R")
            if left and right:
                v = bone_world_head(arm, left) - bone_world_head(arm, right)
                if math.hypot(v[0], v[1]) > 1e-9:
                    return v, f"{arm.name}: {right} -> {left}"
    return None, "no rig names a left and a right arm or leg"


def resolve_yaw(arms, given):
    """The yaw (degrees) that turns the figure to face -Y, and a record of how it was decided."""
    axis, how = rig_left_axis(arms)
    if axis is None:
        if given is None:
            raise ValueError("the figure has no rig with left and right limbs, so its facing cannot be read; "
                             "look at it and pass --yaw (0 when it already faces -Y)")
        return float(given), {"source": "given", "checked": False,
                              "note": f"unrigged or unnamed limbs ({how})"}
    measured = -math.degrees(math.atan2(axis[1], axis[0]))     # turns the left axis onto +X
    snapped = float((round(measured / 90.0) * 90) % 360)
    record = {"rig_axis": how, "rig_yaw_deg": round(measured % 360, 3), "checked": True}
    if given is None:
        yaw, record = snapped, dict(record, source="rig")
    else:
        off = CAM.angular_gap(measured, float(given))
        if off > FACING_TOLERANCE_DEG:
            raise ValueError(f"--yaw {given:g} disagrees with the rig: it would put the figure's left side "
                             f"{off:.0f} degrees away from +X (the rig says --yaw {snapped:g}; or leave "
                             f"--yaw out)")
        yaw, record = float(given), dict(record, source="given")
    turned = CAM.yaw_matrix(yaw)[:3, :3] @ axis
    record["left_axis_after_yaw"] = (turned[:2] / math.hypot(turned[0], turned[1])).round(6).tolist()
    return yaw, record


LIMBS = (("arm", "L"), ("arm", "R"), ("leg", "L"), ("leg", "R"))


def find_limbs(arms):
    """{(limb, side): (rig, root bone)} for every limb a rig names (the first rig that has it)."""
    found = {}
    for arm in arms:
        deform = [b.name for b in arm.data.bones if b.use_deform] or [b.name for b in arm.data.bones]
        for limb, side in LIMBS:
            root = RIG.find_limb_root(deform, limb, side)
            if root is not None and (limb, side) not in found:
                found[(limb, side)] = (arm, root)
    return found


def a_pose(found, meshes, yaw, report):
    """Pose the four limbs; returns the per-limb record (measured after posing)."""
    yaw3 = np.array(yaw)[:3, :3]
    limbs = []
    for (limb, side), (arm, root) in found.items():
        group = limb_group(arm, limb, side, root)
        roots = group_roots(arm, group)
        end, how = limb_end_point(arm, limb, side, root, group, meshes)
        if end is None:
            report.append(f"{arm.name} {limb} {side}: {how}; not posed")
            continue
        pivot = bone_world_head(arm, root)
        cur = yaw3 @ (end - pivot)        # in the contract orientation
        tgt = RIG.target_direction(limb, side)
        aim = RIG.pose_aim(limb)
        # One turn is exact when every bone below inherits its parent's rotation; repeat the
        # correction for rigs where some do not, or where the far-end vertices change.
        after, steps = cur, 0
        while steps < RIG.POSE_STEPS and RIG.angle_between(aim(after), tgt) > RIG.POSE_CONVERGED_DEG:
            rot_world = yaw3.T @ RIG.rotation_between(aim(after), tgt) @ yaw3
            rotate_bones_world(arm, roots, rot_world, bone_world_head(arm, root))
            end2, _ = limb_end_point(arm, limb, side, root, group, meshes)
            after = yaw3 @ (end2 - bone_world_head(arm, root))
            steps += 1
        limbs.append({
            "rig": arm.name, "limb": limb, "side": side, "root_bone": root, "end": how,
            "bones_turned": roots,
            "before_deg_below_horizontal": round(RIG.angle_below_horizontal(cur), 2),
            "after_deg_below_horizontal": round(RIG.angle_below_horizontal(after), 2),
            "after_deg_from_vertical": round(RIG.angle_from_vertical(after), 2),
            "after_frontal_deg_from_vertical": round(RIG.angle_from_vertical(RIG.frontal(after)), 2),
            "after_sagittal_lean_deg": round(RIG.sagittal_lean(after), 2),
            "after_outward": bool(after[0] * (1 if side == "L" else -1) > 0),
            "error_deg": round(RIG.angle_between(aim(after), tgt), 4),
            "error_basis": "3D direction" if limb == "arm" else "direction seen from the front",
            "converged": bool(RIG.angle_between(aim(after), tgt) <= RIG.POSE_CONVERGED_DEG),
            "steps": steps,
        })
    return limbs


# --------------------------------------------------------------------------------------------- materials

IMAGE_INDEX = {}  # lower-case file name and file stem -> path, over the job's image_dirs


def index_images(dirs):
    for d in dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(IMAGE_EXTS):
                    IMAGE_INDEX.setdefault(f.lower(), os.path.join(root, f))
                    IMAGE_INDEX.setdefault(os.path.splitext(f)[0].lower(), os.path.join(root, f))


def resolve_images(dirs, report):
    """Reload images whose data is missing from a file of the same name in one of `dirs`."""
    index_images(dirs)
    index = IMAGE_INDEX
    missing = []
    for img in bpy.data.images:
        if img.type != "IMAGE" or has_pixels(img):
            continue
        keys = [os.path.basename(bpy.path.abspath(img.filepath)).lower(), img.name.lower(),
                os.path.splitext(img.name)[0].lower()]
        hit = next((index[k] for k in keys if k in index), None)
        if hit:
            img.filepath = hit
            img.reload()
            report.append(f"image {img.name!r} relinked to {os.path.basename(hit)}")
        else:
            missing.append(img.name)
    return missing


def has_pixels(img):
    """True when the image's pixels can be read: loaded, packed in the file (packed images load lazily), or
    on disk."""
    return bool(img.has_data or img.packed_file is not None
                or (img.filepath and os.path.exists(bpy.path.abspath(img.filepath))))


def _image(spec):
    """An image by datablock name, by file name or stem in the image folders, or by path."""
    if spec in bpy.data.images and has_pixels(bpy.data.images[spec]):
        return bpy.data.images[spec]
    path = IMAGE_INDEX.get(spec.lower()) or IMAGE_INDEX.get(os.path.basename(spec).lower())
    if path is None and not os.path.exists(spec):
        raise ValueError(f"image {spec!r} is not in the file, the image folders or on disk")
    return bpy.data.images.load(path or spec, check_existing=True)


def base_colour_source(mat, image_override, vcol_override):
    """(socket or None, rgba default) giving the material's base colour, and how it was found."""
    nt = mat.node_tree
    if image_override:
        spec, _, uv_name = image_override.partition("@")
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = _image(spec)
        how = f"image {os.path.basename(spec)} (override)"
        if uv_name:
            uv = nt.nodes.new("ShaderNodeUVMap")
            uv.uv_map = uv_name
            nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
            how += f" on UV map {uv_name}"
        return tex.outputs["Color"], None, how
    if vcol_override:
        attr = nt.nodes.new("ShaderNodeVertexColor")
        attr.layer_name = vcol_override
        return attr.outputs["Color"], None, f"colour attribute {vcol_override} (override)"
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is not None:
        inp = bsdf.inputs["Base Color"]
        if inp.is_linked:
            return inp.links[0].from_socket, None, "principled base colour (linked)"
        return None, tuple(inp.default_value), "principled base colour (value)"
    for n in nt.nodes:
        if n.type in ("BSDF_DIFFUSE", "EMISSION", "BSDF_GLOSSY") and "Color" in n.inputs:
            inp = n.inputs["Color"]
            if inp.is_linked:
                return inp.links[0].from_socket, None, f"{n.type.lower()} colour (linked)"
            return None, tuple(inp.default_value), f"{n.type.lower()} colour (value)"
    return None, tuple(mat.diffuse_color), "viewport colour"


# --------------------------------------------------------------------------------------------- alpha

ALPHA_CUTOFF = 0.5          # texture alpha below this is a hole, as a game engine clips alphaMode MASK
ALPHA_LAYERS = 64           # transparent surfaces a ray (and Cycles) may pass through before it stops
RAY_STEP = 1e-5             # studs a ray moves past a clipped texel before it looks again


def _from(sock):
    """(node, output socket) feeding an input socket, or (None, None)."""
    if not sock.is_linked:
        return None, None
    link = sock.links[0]
    return link.from_node, link.from_socket


def _alpha_image_node(sock):
    """The image node whose Alpha output feeds `sock` directly, else None."""
    node, out = _from(sock)
    if node is not None and node.type == "TEX_IMAGE" and out.name == "Alpha" and node.image is not None:
        return node
    return None


def read_alpha(mat, bsdf, override_node, flag, legacy):
    """How a material's alpha is decided: {"mode": "opaque"|"clip", "cutoff", "source", "node"}.

    With --alpha (needed for an --image override or an old material without nodes, where nothing else
    says), the flag decides and the alpha comes from the override image. Otherwise the material's own
    Principled Alpha input decides, read the way the glTF importer writes alphaMode: unlinked is OPAQUE (a
    constant alpha below the cutoff hides the whole material); `1 - (alpha < c)` is MASK with cutoff c;
    `round(alpha)` is MASK at 0.5; a texture alpha linked straight in (BLEND) is clipped at 0.5 as well. Any
    other alpha source cannot be sampled the same way in the masks, and is refused."""
    name = mat.name
    if flag is not None:
        if flag == "opaque":
            return {"mode": "opaque", "cutoff": None, "source": "--alpha opaque", "node": None}
        if override_node is None:
            raise ValueError(f"--alpha {name}=clip needs an --image for {name}: the alpha comes from that "
                             f"image")
        return {"mode": "clip", "cutoff": ALPHA_CUTOFF, "source": "--alpha clip on the --image override",
                "node": override_node}
    if override_node is not None or legacy:
        why = "an --image override" if override_node is not None else "an old material without nodes"
        return {"mode": "opaque", "cutoff": None, "source": f"{why}; opaque unless --alpha {name}=clip",
                "node": None}
    if bsdf is None:
        return {"mode": "opaque", "cutoff": None, "source": "no principled shader", "node": None}
    inp = bsdf.inputs["Alpha"]
    if not inp.is_linked:
        v = float(inp.default_value)
        if v >= ALPHA_CUTOFF:
            return {"mode": "opaque", "cutoff": None, "source": f"alpha {v:g} (glTF OPAQUE)", "node": None}
        return {"mode": "clip", "cutoff": ALPHA_CUTOFF, "node": None, "constant": v,
                "source": f"constant alpha {v:g}: the whole material is a hole"}
    node, _ = _from(inp)
    img = _alpha_image_node(inp)
    if img is not None:
        return {"mode": "clip", "cutoff": ALPHA_CUTOFF, "node": img,
                "source": "texture alpha linked straight in (glTF BLEND), clipped at 0.5"}
    if node.type == "MATH" and node.operation == "ROUND":
        img = _alpha_image_node(node.inputs[0])
        if img is not None:
            return {"mode": "clip", "cutoff": ALPHA_CUTOFF, "source": "round(texture alpha) (glTF MASK)",
                    "node": img}
    if node.type == "MATH" and node.operation == "SUBTRACT" and not node.inputs[0].is_linked \
            and abs(node.inputs[0].default_value - 1.0) < 1e-9:
        lt, _ = _from(node.inputs[1])
        if lt is not None and lt.type == "MATH" and lt.operation == "LESS_THAN" and \
                not lt.inputs[1].is_linked:
            img = _alpha_image_node(lt.inputs[0])
            if img is not None:
                c = float(lt.inputs[1].default_value)
                return {"mode": "clip", "cutoff": c, "source": f"1 - (texture alpha < {c:g}) (glTF MASK)",
                        "node": img}
    raise ValueError(f"material {name}: its alpha comes from a {node.type.lower()} node the bench cannot "
                     f"sample in the masks (only an image's alpha, glTF MASK/BLEND/OPAQUE, or --alpha); "
                     f"pass --alpha {name}=opaque to ignore it")


def check_alpha_node(name, node, colour_sock):
    """The clipped alpha must be the base-colour texture's own alpha (glTF keeps alpha there), sampled the way
    the bench's masks can reproduce."""
    cnode = colour_sock.node if colour_sock is not None else None
    if cnode is None or cnode.type != "TEX_IMAGE" or cnode.image != node.image:
        raise ValueError(f"material {name}: the alpha comes from {node.image.name}, which is not its "
                         f"base-colour texture; glTF keeps alpha in the base-colour texture, so pack it "
                         f"there first")
    if node.interpolation not in TEX.INTERPOLATIONS:
        raise ValueError(f"material {name}: texture interpolation {node.interpolation} cannot be matched "
                         f"in the masks (use Linear or Closest)")
    if node.extension not in TEX.EXTENSIONS:
        raise ValueError(f"material {name}: texture extension {node.extension} is not supported")
    vec, _ = _from(node.inputs["Vector"])
    if vec is not None and vec.type != "UVMAP":
        raise ValueError(f"material {name}: the alpha texture's coordinates come from a "
                         f"{vec.type.lower()} node; only a plain UV map can be matched in the masks")


def alpha_sampler(node):
    """What the masks need to read this texture's alpha as Cycles does."""
    img = node.image
    px = np.empty(len(img.pixels), dtype=np.float32)
    img.pixels.foreach_get(px)
    w, h = img.size
    vec, _ = _from(node.inputs["Vector"])
    return {"alpha": px[3::4].reshape(h, w), "interpolation": node.interpolation, "extension": node.extension,
            "uv_map": (vec.uv_map or None) if vec is not None else None, "image": img.name}


def normalise_materials(meshes, images, vcols, alphas, report):
    """Rebuild every used material around its base colour and its alpha: a lit principled shader and an unlit
    emission shader fed by the same colour, each cut by the same alpha step (1 where the texture alpha is at
    least the cutoff, else 0) when the material clips. `set_shading` switches the output between them; the lit
    one is what the GLB exporter reads, so a clipped material exports as alphaMode MASK. Overrides must name
    materials the chosen meshes use. Returns {material: alpha record} and {material: sampler} for the
    masks."""
    used = sorted({m.name for o in meshes for m in o.data.materials if m is not None})
    unknown = sorted((set(images) | set(vcols) | set(alphas)) - set(used))
    if unknown:
        raise ValueError(f"--image/--vcol/--alpha name materials the chosen meshes do not use: "
                         f"{', '.join(unknown)} (they use: {', '.join(used) or 'no materials'})")
    done, records, samplers = {}, {}, {}
    for o in meshes:
        if not o.data.materials:
            mat = bpy.data.materials.new(o.name + "_bench")
            mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
            o.data.materials.append(mat)
        for slot_mat in o.data.materials:
            if slot_mat is None or slot_mat.name in done:
                continue
            mat = slot_mat
            legacy = None if mat.use_nodes else tuple(mat.diffuse_color)
            mat.use_nodes = True
            nt = mat.node_tree
            bsdf = None if legacy is not None else \
                next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            sock, value, how = base_colour_source(mat, images.get(mat.name), vcols.get(mat.name))
            if legacy is not None and sock is None and not images.get(mat.name) and not vcols.get(mat.name):
                value, how = legacy, "legacy diffuse colour"
            override = sock.node if images.get(mat.name) else None
            alpha = read_alpha(mat, bsdf, override, alphas.get(mat.name), legacy is not None)
            if alpha["node"] is not None:
                check_alpha_node(mat.name, alpha["node"], sock)
            out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
            if out is None:
                out = nt.nodes.new("ShaderNodeOutputMaterial")
            lit = nt.nodes.new("ShaderNodeBsdfPrincipled")
            lit.name = "BenchLit"
            lit.inputs["Roughness"].default_value = 0.6
            lit.inputs["Metallic"].default_value = 0.0
            flat = nt.nodes.new("ShaderNodeEmission")
            flat.name = "BenchFlat"
            flat.inputs["Strength"].default_value = 1.0
            hole = nt.nodes.new("ShaderNodeBsdfTransparent")
            cut = nt.nodes.new("ShaderNodeMixShader")      # Fac 0: the hole, 1: the flat colour
            cut.name = "BenchFlatCut"
            nt.links.new(hole.outputs["BSDF"], cut.inputs[1])
            nt.links.new(flat.outputs["Emission"], cut.inputs[2])
            if sock is not None:
                nt.links.new(sock, lit.inputs["Base Color"])
                nt.links.new(sock, flat.inputs["Color"])
            else:
                lit.inputs["Base Color"].default_value = value
                flat.inputs["Color"].default_value = value
            if alpha["mode"] == "clip" and alpha["node"] is not None:
                below = nt.nodes.new("ShaderNodeMath")
                below.operation = "LESS_THAN"
                below.inputs[1].default_value = alpha["cutoff"]
                step = nt.nodes.new("ShaderNodeMath")
                step.operation = "SUBTRACT"
                step.inputs[0].default_value = 1.0
                nt.links.new(alpha["node"].outputs["Alpha"], below.inputs[0])
                nt.links.new(below.outputs["Value"], step.inputs[1])
                nt.links.new(step.outputs["Value"], lit.inputs["Alpha"])
                nt.links.new(step.outputs["Value"], cut.inputs["Fac"])
                samplers[mat.name] = dict(alpha_sampler(alpha["node"]), cutoff=alpha["cutoff"])
            else:
                keep = 0.0 if alpha["mode"] == "clip" else 1.0   # a constant alpha below the cutoff: all hole
                lit.inputs["Alpha"].default_value = keep
                cut.inputs["Fac"].default_value = keep
                if alpha["mode"] == "clip":
                    samplers[mat.name] = {"constant": 0.0, "cutoff": alpha["cutoff"]}
            for link in list(out.inputs["Surface"].links):
                nt.links.remove(link)
            for link in list(out.inputs["Displacement"].links):
                nt.links.remove(link)
            nt.links.new(lit.outputs["BSDF"], out.inputs["Surface"])
            records[mat.name] = {
                "mode": alpha["mode"], "cutoff": alpha["cutoff"], "source": alpha["source"],
                "image": alpha["node"].image.name if alpha["node"] is not None else None,
                "clipped_texel_share": (round(image_alpha(alpha["node"].image, alpha["cutoff"]) or 0.0, 5)
                                        if alpha["node"] is not None else None),
            }
            done[mat.name] = how
    report.update(done)
    return records, samplers


def images_feeding_colour(meshes):
    """Images the base colour of the given meshes depends on (upstream of each material's flat shader)."""
    found = set()
    for o in meshes:
        for mat in o.data.materials:
            if mat is None or "BenchFlat" not in mat.node_tree.nodes:
                continue
            stack, seen = [mat.node_tree.nodes["BenchFlat"]], set()
            while stack:
                node = stack.pop()
                if node.name in seen:
                    continue
                seen.add(node.name)
                if node.type == "TEX_IMAGE" and node.image is not None:
                    found.add(node.image)
                for inp in node.inputs:
                    stack.extend(link.from_node for link in inp.links)
    return found


def image_alpha(img, cutoff=0.5):
    """Share of the image's texels with alpha below `cutoff`, or None when the image has no alpha."""
    if img.channels < 4 or img.alpha_mode == "NONE" or not has_pixels(img):
        return None
    px = np.empty(len(img.pixels), dtype=np.float32)
    img.pixels.foreach_get(px)
    return float((px[3::4] < cutoff).mean())


def set_shading(meshes, mode):
    seen = set()
    for o in meshes:
        for mat in o.data.materials:
            if mat is None or mat.name in seen:
                continue
            seen.add(mat.name)
            nt = mat.node_tree
            out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output)
            if mode == "flat":
                src = nt.nodes["BenchFlatCut"].outputs["Shader"]
            else:
                src = nt.nodes["BenchLit"].outputs["BSDF"]
            for link in list(out.inputs["Surface"].links):
                nt.links.remove(link)
            nt.links.new(src, out.inputs["Surface"])


# --------------------------------------------------------------------------------------------- freezing

def freeze(meshes, pre, report):
    """Replace the chosen objects by static copies of their evaluated (posed) meshes, transformed by `pre`
    (world -> contract orientation); delete everything else. Returns the new objects."""
    dg = bpy.context.evaluated_depsgraph_get()
    pre_m = Matrix([list(r) for r in pre])
    baked = []
    for o in meshes:
        ev = o.evaluated_get(dg)
        me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
        me.transform(pre_m @ ev.matrix_world)
        baked.append((o.name, me))
    scene = bpy.context.scene
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    frozen = []
    for name, me in baked:
        new = bpy.data.objects.new(name, me)  # the source names, free again, carry into the GLB
        scene.collection.objects.link(new)
        frozen.append(new)
    report.append(f"froze {len(frozen)} meshes")
    return frozen


def all_triangles(objs):
    verts, tris, base = [], [], 0
    for o in objs:
        me = o.data
        me.calc_loop_triangles()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        t = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
        me.loop_triangles.foreach_get("vertices", t)
        mw = np.array(o.matrix_world)
        co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        verts.append(co)
        tris.append(t.reshape(-1, 3) + base)
        base += len(co)
    return np.concatenate(verts), np.concatenate(tris)


def apply_matrix(objs, m):
    mm = Matrix([list(r) for r in m])
    for o in objs:
        o.data.transform(mm)
        o.data.update()


# --------------------------------------------------------------------------------------------- rendering

def triangle_alpha(objs, samplers):
    """For every triangle (in `all_triangles` order): the index of its material's alpha sampler (-1 when the
    material is opaque) and its three corners' UVs in the UV map that sampler reads."""
    index = {name: i for i, name in enumerate(samplers)}
    tri_s, tri_uv = [], []
    for o in objs:
        me = o.data
        me.calc_loop_triangles()
        n = len(me.loop_triangles)
        mat_idx = np.empty(n, dtype=np.int64)
        me.loop_triangles.foreach_get("material_index", mat_idx)
        loops = np.empty(n * 3, dtype=np.int64)
        me.loop_triangles.foreach_get("loops", loops)
        names = [m.name if m is not None else None for m in me.materials]
        s_of_slot = np.array([index.get(nm, -1) for nm in names] or [-1], dtype=np.int64)
        s = s_of_slot[np.clip(mat_idx, 0, len(s_of_slot) - 1)]
        uv = np.zeros((n, 3, 2), dtype=np.float64)
        for si in sorted(set(s.tolist()) - {-1}):
            sampler = list(samplers.values())[si]
            if "constant" in sampler:
                continue
            layer = (me.uv_layers.get(sampler["uv_map"]) if sampler["uv_map"]
                     else next((u for u in me.uv_layers if u.active_render), None))
            if layer is None:
                raise ValueError(f"{o.name}: no UV map {sampler['uv_map'] or '(render)'} for its alpha "
                                 f"texture")
            flat = np.empty(len(me.loops) * 2, dtype=np.float64)
            layer.data.foreach_get("uv", flat)
            per_loop = flat.reshape(-1, 2)
            pick = s == si
            uv[pick] = per_loop[loops.reshape(-1, 3)[pick]]
        tri_s.append(s)
        tri_uv.append(uv)
    return np.concatenate(tri_s), np.concatenate(tri_uv), list(samplers.values())


def hit_alpha(tri, point, verts, tris, tri_uv, sampler):
    """Texture alpha at a point on triangle `tri`, sampled as Cycles samples the texture."""
    if "constant" in sampler:
        return sampler["constant"]
    a, b, c = verts[tris[tri]]
    uv = TEX.barycentric_uv(a, b, c, point, *tri_uv[tri])
    return TEX.sample(sampler["alpha"], float(uv[0]), float(uv[1]), sampler["interpolation"],
                      sampler["extension"])


def raycast_view(bvh, cam, verts, tris, alpha, clip=True):
    """Mask and depth from one orthographic ray per pixel centre. A ray that meets a clipped texel (alpha
    below its material's cutoff) goes on through it; the first surface it does not pass is the hit."""
    tri_s, tri_uv, samplers = alpha
    size = cam["resolution"]
    cr = CAM.project(verts, cam)
    c0 = max(0, int(math.floor(cr[:, 0].min())) - 2)
    c1 = min(size - 1, int(math.ceil(cr[:, 0].max())) + 2)
    r0 = max(0, int(math.floor(cr[:, 1].min())) - 2)
    r1 = min(size - 1, int(math.ceil(cr[:, 1].max())) + 2)
    mask = np.zeros((size, size), dtype=bool)
    depth = np.full((size, size), np.nan, dtype=np.float32)
    rows, cols = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    rows, cols = rows.ravel(), cols.ravel()
    origins, fwd = CAM.pixel_rays(cam, rows, cols)
    d = Vector(fwd.tolist())
    far = 2.0 * cam["distance"]
    centre = np.asarray(cam["image_centre_world"])
    clipping = clip and bool(samplers)
    for r, c, o in zip(rows.tolist(), cols.tolist(), origins.tolist()):
        origin, left = Vector(o), far
        for _ in range(ALPHA_LAYERS):
            hit, _, tri, dist = bvh.ray_cast(origin, d, left)
            if hit is None:
                break
            s = int(tri_s[tri]) if clipping else -1
            if s < 0 or hit_alpha(tri, hit, verts, tris, tri_uv, samplers[s]) >= samplers_cutoff(samplers[s]):
                mask[r, c] = True
                depth[r, c] = float(np.dot(np.array(hit) - centre, fwd))
                break
            origin = hit + d * RAY_STEP          # past the clipped texel, on along the same ray
            left -= dist + RAY_STEP
    return mask, depth


def samplers_cutoff(sampler):
    return sampler.get("cutoff", ALPHA_CUTOFF)


def setup_cycles(scene, size, report):
    scene.render.engine = "CYCLES"
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.display_settings.display_device = "sRGB"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.view_settings.use_curve_mapping = False
    scene.cycles.film_exposure = 1.0
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    scene.render.use_motion_blur = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.transparent_max_bounces = ALPHA_LAYERS   # layered alpha cards stay see-through
    scene.cycles.seed = 0
    device = "CPU"
    if JOB.get("device", "auto") != "cpu":
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for kind in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
                try:
                    prefs.compute_device_type = kind
                except TypeError:
                    continue
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type == kind]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type == kind
                    scene.cycles.device = "GPU"
                    device = f"{kind}: " + ", ".join(sorted({d.name for d in gpus}))
                    break
        except Exception as exc:  # no usable GPU backend: render on the CPU
            report.append(f"GPU setup failed ({exc.__class__.__name__}); rendering on the CPU")
    return device


def bench_world(scene):
    """A world of the bench's own (never the file's world or HDRI): one plain Background node. Returns it."""
    world = bpy.data.worlds.new("BenchWorld")
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    scene.world = world
    return bg


def add_render_camera(cam):
    data = bpy.data.cameras.new("BenchCam")
    data.type = "ORTHO"
    data.ortho_scale = cam["ortho_scale"]
    data.clip_start = 0.01
    data.clip_end = 2.0 * cam["distance"]
    data.sensor_fit = "HORIZONTAL"
    obj = bpy.data.objects.new("BenchCam", data)
    bpy.context.scene.collection.objects.link(obj)
    r, u, f = (Vector(cam[k]) for k in ("right", "up", "forward"))
    m = Matrix.Identity(4)
    for i in range(3):
        m[i][0], m[i][1], m[i][2], m[i][3] = r[i], u[i], -f[i], cam["location"][i]
    obj.matrix_world = m
    return obj


def lights_for(cam):
    """A key and a fill sun fixed to the camera (every view lit alike), plus the world as soft ambient."""
    r, u, f = (np.asarray(cam[k], dtype=np.float64) for k in ("right", "up", "forward"))
    made = []
    for name, travel, strength in (("Key", f - 0.9 * u + 0.6 * r, 3.0), ("Fill", f - 0.3 * u - 0.8 * r, 1.0)):
        d = bpy.data.lights.new(name, "SUN")
        d.energy = strength
        d.angle = math.radians(10.0)
        o = bpy.data.objects.new(name, d)
        travel = travel / np.linalg.norm(travel)
        o.rotation_euler = Vector((0.0, 0.0, -1.0)).rotation_difference(Vector(travel.tolist())).to_euler()
        bpy.context.scene.collection.objects.link(o)
        made.append(o)
    return made


def main():
    t0 = time.time()
    report, mat_report, timings = [], {}, {}
    scene = load_asset(JOB["asset"])
    meshes = choose_meshes(scene, JOB.get("include"), JOB.get("exclude"))
    rigs = rigs_of(meshes)
    missing_images = resolve_images(JOB.get("image_dirs", []), report)
    skipped = skipped_geometry(scene, meshes)
    pose_mode = JOB.get("pose", "auto")
    limbs = []
    if pose_mode != "as-shipped":
        clear_animation(scene, report)
        for arm in rigs:
            reset_rig(arm, report)
        bpy.context.view_layer.update()
    yaw_deg, facing = resolve_yaw(rigs, JOB.get("yaw_deg"))
    yaw = CAM.yaw_matrix(yaw_deg)
    if JOB.get("bind_unrigged") and rigs:
        bind_unrigged(meshes, report)
    found = find_limbs(rigs)
    missing_limbs = [f"{limb} {side}" for limb, side in LIMBS if (limb, side) not in found]
    if pose_mode == "a-pose" and missing_limbs:
        raise ValueError(f"--pose a-pose needs both arms and both legs in the rig; not found: "
                         f"{', '.join(missing_limbs)}")
    if pose_mode in ("auto", "a-pose") and rigs and not missing_limbs:
        limbs = a_pose(found, meshes, yaw, report)
        pose_used = "a-pose"
    elif rigs and pose_mode in ("auto", "rest"):
        pose_used = "rest"
        if pose_mode == "auto":
            report.append(f"rigged, but not found: {', '.join(missing_limbs)}; kept the rest pose")
    else:
        pose_used = "as-shipped"
    timings["load_pose"] = round(time.time() - t0, 2)

    alpha_records, samplers = normalise_materials(
        meshes, JOB.get("material_images", {}), JOB.get("material_vcols", {}), JOB.get("material_alpha", {}),
        mat_report)
    used = images_feeding_colour(meshes)
    unused_missing = sorted(set(missing_images) - {im.name for im in used})
    missing_images = sorted(im.name for im in used if not has_pixels(im))
    if unused_missing:
        report.append(f"missing images no rendered colour uses: {', '.join(unused_missing)}")
    object_names, rig_names = [o.name for o in meshes], [a.name for a in rigs]
    frozen = freeze(meshes, yaw, report)
    verts, tris = all_triangles(frozen)
    contract = CAM.contract_matrix(verts.min(0), verts.max(0), JOB["height"])
    apply_matrix(frozen, contract)
    verts, tris = all_triangles(frozen)
    alpha = triangle_alpha(frozen, samplers)

    gt = JOB["gt_glb"]
    os.makedirs(os.path.dirname(gt), exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in frozen:
        o.select_set(True)
    bpy.context.view_layer.objects.active = frozen[0]
    bpy.ops.export_scene.gltf(filepath=gt, export_format="GLB", use_selection=True, export_apply=True,
                              export_yup=True, export_animations=False, export_skins=False)

    size = int(JOB["size"])
    ppu = CAM.choose_ppu(verts, size, JOB["height"])
    cams = CAM.all_cameras(ppu, size, JOB["height"], CAM.camera_distance(verts, JOB["height"]))
    work = JOB["work"]
    os.makedirs(work, exist_ok=True)

    t1 = time.time()
    bvh = BVHTree.FromPolygons([tuple(v) for v in verts.tolist()], [tuple(t) for t in tris.tolist()],
                               epsilon=0.0)
    for cam in cams:
        mask, depth = raycast_view(bvh, cam, verts, tris, alpha)
        extra = {}
        if cam["view"] == "front":
            # the camera-scale check measures the geometry, which alpha clipping can make look shorter
            extra["geometry_mask"] = raycast_view(bvh, cam, verts, tris, alpha, clip=False)[0] \
                if alpha[2] else mask
        np.savez_compressed(os.path.join(work, cam["view"] + ".npz"), mask=mask, depth=depth, **extra)
    timings["raycast"] = round(time.time() - t1, 2)

    t2 = time.time()
    device = setup_cycles(scene, size, report)
    bg = bench_world(scene)
    for cam in cams:
        cobj = add_render_camera(cam)
        scene.camera = cobj
        # flat: emission only, no lights, black world
        set_shading(frozen, "flat")
        bg.inputs["Strength"].default_value = 0.0
        scene.cycles.samples = int(JOB.get("samples_flat", 16))
        scene.cycles.use_denoising = False
        scene.render.filepath = os.path.join(work, cam["view"] + "-flat.png")
        bpy.ops.render.render(write_still=True)
        # lit: principled base colour, camera-fixed suns, soft ambient world
        set_shading(frozen, "lit")
        bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs["Strength"].default_value = 0.35
        lights = lights_for(cam)
        scene.cycles.samples = int(JOB.get("samples_lit", 128))
        scene.cycles.use_denoising = True
        scene.render.filepath = os.path.join(work, cam["view"] + "-lit.png")
        bpy.ops.render.render(write_still=True)
        for o in lights + [cobj]:
            bpy.data.objects.remove(o, do_unlink=True)
    timings["render"] = round(time.time() - t2, 2)
    timings["total"] = round(time.time() - t0, 2)

    result = {
        "ok": True,
        "blender": bpy.app.version_string,
        "objects": object_names,
        "rigs": rig_names,
        "pose_requested": pose_mode,
        "pose": pose_used,
        "limbs": limbs,
        "yaw_deg": yaw_deg,
        "facing": facing,
        "skipped_geometry": skipped,
        "alpha": alpha_records,
        "contract_scale": float(contract[0, 0]),
        "triangles": int(len(tris)),
        "vertices": int(len(verts)),
        "bounds_min": verts.min(0).round(6).tolist(),
        "bounds_max": verts.max(0).round(6).tolist(),
        "pixels_per_unit": ppu,
        "cameras": cams,
        "materials": mat_report,
        "missing_images": missing_images,
        "device": device,
        "timings": timings,
        "report": report,
    }
    path = sys.argv[sys.argv.index("--") + 1]
    stem = path[: -len(".json")] if path.endswith(".json") else path
    with open(stem + ".result.json", "w") as fh:
        json.dump(result, fh, indent=2)


def run():
    """Input problems (a ValueError) go back to the caller as {"ok": false, "error": ...}, not a traceback."""
    path = sys.argv[sys.argv.index("--") + 1]
    stem = path[: -len(".json")] if path.endswith(".json") else path
    try:
        main()
    except ValueError as exc:
        with open(stem + ".result.json", "w") as fh:
            json.dump({"ok": False, "error": str(exc)}, fh, indent=2)


run()
