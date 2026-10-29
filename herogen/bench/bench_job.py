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
4. Freeze the posed meshes into the contract frame (`yaw_deg` turns the figure to face -Y first), scale to
   `height`, and export `gt_glb`.
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

IMAGE_EXTS = (".png", ".tga", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".exr")
POSE_STEPS = 8
POSE_CONVERGED_DEG = 0.01
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
    """Bones of the group whose parent is outside it: turning these turns the whole group once."""
    g = set(group)
    return [n for n in group if arm.data.bones[n].parent is None or arm.data.bones[n].parent.name not in g]


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


def a_pose(arms, meshes, yaw, report):
    """Pose every limb that can be found; returns the per-limb record (angles measured after posing)."""
    yaw3 = np.array(yaw)[:3, :3]
    limbs = []
    for arm in arms:
        deform = [b.name for b in arm.data.bones if b.use_deform] or [b.name for b in arm.data.bones]
        for limb in ("arm", "leg"):
            for side in ("L", "R"):
                root = RIG.find_limb_root(deform, limb, side)
                if root is None:
                    continue
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
                while steps < POSE_STEPS and RIG.angle_between(aim(after), tgt) > POSE_CONVERGED_DEG:
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
        if img.type != "IMAGE" or img.packed_file is not None:
            continue
        if img.has_data or (img.filepath and os.path.exists(bpy.path.abspath(img.filepath))):
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


def _image(spec):
    """An image by datablock name, by file name or stem in the image folders, or by path."""
    if spec in bpy.data.images and bpy.data.images[spec].has_data:
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


def normalise_materials(meshes, images, vcols, report):
    """Rebuild every used material around its base colour: a lit principled shader and an unlit emission
    shader fed by the same source. `set_shading` switches the material output between them."""
    done = {}
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
            sock, value, how = base_colour_source(mat, images.get(mat.name), vcols.get(mat.name))
            if legacy is not None and sock is None and not images.get(mat.name) and not vcols.get(mat.name):
                value, how = legacy, "legacy diffuse colour"
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
            if sock is not None:
                nt.links.new(sock, lit.inputs["Base Color"])
                nt.links.new(sock, flat.inputs["Color"])
            else:
                lit.inputs["Base Color"].default_value = value
                flat.inputs["Color"].default_value = value
            for link in list(out.inputs["Surface"].links):
                nt.links.remove(link)
            for link in list(out.inputs["Displacement"].links):
                nt.links.remove(link)
            nt.links.new(lit.outputs["BSDF"], out.inputs["Surface"])
            done[mat.name] = how
    report.update(done)


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
                src = nt.nodes["BenchFlat"].outputs["Emission"]
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

def raycast_view(bvh, cam, verts):
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
    far = 2.0 * CAM.CAMERA_DISTANCE
    centre = np.asarray(cam["image_centre_world"])
    for r, c, o in zip(rows.tolist(), cols.tolist(), origins.tolist()):
        hit = bvh.ray_cast(Vector(o), d, far)[0]
        if hit is not None:
            mask[r, c] = True
            depth[r, c] = float(np.dot(np.array(hit) - centre, fwd))
    return mask, depth


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
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.cycles.use_adaptive_sampling = False
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


def add_render_camera(cam):
    data = bpy.data.cameras.new("BenchCam")
    data.type = "ORTHO"
    data.ortho_scale = cam["ortho_scale"]
    data.clip_start = 0.01
    data.clip_end = 500.0
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
    pose_mode = JOB.get("pose", "auto")
    yaw = CAM.yaw_matrix(float(JOB.get("yaw_deg", 0.0)))
    limbs = []
    if pose_mode != "as-shipped":
        for arm in rigs:
            reset_rig(arm, report)
        bpy.context.view_layer.update()
    if JOB.get("bind_unrigged") and rigs:
        bind_unrigged(meshes, report)
    if pose_mode in ("auto", "a-pose") and rigs:
        limbs = a_pose(rigs, meshes, yaw, report)
    arms_posed = sum(1 for x in limbs if x["limb"] == "arm")
    if pose_mode == "a-pose" and arms_posed < 2:
        raise ValueError(f"a-pose asked for but only {arms_posed} arms were found and posed")
    if arms_posed == 2:
        pose_used = "a-pose"
    elif rigs and pose_mode in ("auto", "rest"):
        pose_used = "rest"
    else:
        pose_used = "as-shipped"
    if pose_mode == "auto" and rigs and arms_posed < 2:
        report.append(f"rigged, but {arms_posed} arms found; kept the rest pose")
    timings["load_pose"] = round(time.time() - t0, 2)

    normalise_materials(meshes, JOB.get("material_images", {}), JOB.get("material_vcols", {}), mat_report)
    used = images_feeding_colour(meshes)
    unused_missing = sorted(set(missing_images) - {im.name for im in used})
    missing_images = sorted(im.name for im in used if not (im.has_data or im.packed_file is not None
                                                          or os.path.exists(bpy.path.abspath(im.filepath))))
    if unused_missing:
        report.append(f"missing images no rendered colour uses: {', '.join(unused_missing)}")
    object_names, rig_names = [o.name for o in meshes], [a.name for a in rigs]
    frozen = freeze(meshes, yaw, report)
    verts, tris = all_triangles(frozen)
    contract = CAM.contract_matrix(verts.min(0), verts.max(0), JOB["height"])
    apply_matrix(frozen, contract)
    verts, tris = all_triangles(frozen)

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
    cams = CAM.all_cameras(ppu, size, JOB["height"])
    work = JOB["work"]
    os.makedirs(work, exist_ok=True)

    t1 = time.time()
    bvh = BVHTree.FromPolygons([tuple(v) for v in verts.tolist()], [tuple(t) for t in tris.tolist()],
                               epsilon=0.0)
    for cam in cams:
        mask, depth = raycast_view(bvh, cam, verts)
        np.savez_compressed(os.path.join(work, cam["view"] + ".npz"), mask=mask, depth=depth)
    timings["raycast"] = round(time.time() - t1, 2)

    t2 = time.time()
    device = setup_cycles(scene, size, report)
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
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
        "yaw_deg": float(JOB.get("yaw_deg", 0.0)),
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


main()
