"""Runs inside Blender: `Blender --background --factory-startup --python blender_job.py -- <job.json>`.

Jobs (one JSON file, list of tasks run in order):
  {"task": "silhouettes", "mesh": npz, "glb": path|null, "cameras": [...], "out_dir": dir}
  {"task": "decimate", "mesh": npz, "targets": {"high": n, "low": n}, "out": {"high": npz, "low": npz}}
  {"task": "body", "lods": {"high": npz, "low": npz}, "glb": {"high": path, "low": path},
   "colors": {section: [r, g, b]}, "poses": [{"name", "matrices": {section: 4x4}}],
   "pose_cameras": [...], "pose_dir": dir}
Mesh npz files hold `verts` [N, 3] and `faces` [M, 3]; a body LOD npz holds `<section>__verts`,
`<section>__faces` and `<section>__pivot`. The job writes `<job>.result.json`.
"""
import json
import math
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _MATERIALS.clear()
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    return scene


def make_object(name, verts, faces, origin=(0.0, 0.0, 0.0), color=None, smooth=True, uv=None, image=None,
                material="BodyAtlas"):
    me = bpy.data.meshes.new(name)
    o = np.asarray(origin, dtype=np.float64)
    me.from_pydata((np.asarray(verts) - o).tolist(), [], np.asarray(faces, dtype=np.int64).tolist())
    if uv is not None:
        layer = me.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", np.asarray(uv, dtype=np.float32).reshape(-1))
    if smooth:
        me.shade_smooth()
        if hasattr(me, "set_sharp_from_angle"):
            me.set_sharp_from_angle(angle=math.radians(45.0))
    obj = bpy.data.objects.new(name, me)
    obj.location = Vector(o.tolist())
    bpy.context.scene.collection.objects.link(obj)
    if image is not None:
        me.materials.append(atlas_material(image, material))
    elif color is not None:
        mat = bpy.data.materials.new(name + "Mat")
        mat.diffuse_color = (*color, 1.0)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (*color, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.7
        me.materials.append(mat)
    if color is not None:
        obj.color = (*color, 1.0)
    return obj


_MATERIALS = {}


def atlas_material(path, name="BodyAtlas"):
    """One shared material per texture image: the image into Base Color, rough and non-metallic (reads as
    unlit). ``name`` names the material (the body atlas or the Head's own texture)."""
    if path in _MATERIALS and _MATERIALS[path].name in bpy.data.materials:
        return _MATERIALS[path]
    img = bpy.data.images.load(path, check_existing=True)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Linear"
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Metallic"].default_value = 0.0
    nodes.active = tex
    _MATERIALS[path] = mat
    return mat


def add_camera(cam):
    data = bpy.data.cameras.new("Cam")
    data.type = "ORTHO"
    data.ortho_scale = cam["ortho_scale"]
    data.clip_start = 0.01
    data.clip_end = 500.0
    obj = bpy.data.objects.new("Cam", data)
    bpy.context.scene.collection.objects.link(obj)
    r, u, f = (Vector(cam[k]) for k in ("right", "up", "forward"))
    m = Matrix.Identity(4)
    for i in range(3):
        m[i][0], m[i][1], m[i][2], m[i][3] = r[i], u[i], -f[i], cam["location"][i]
    obj.matrix_world = m
    return obj


def setup_render(scene, res, silhouette, texture=False):
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    sh = scene.display.shading
    sh.show_object_outline = False
    sh.show_cavity = False
    sh.show_shadows = False
    sh.show_specular_highlight = not silhouette
    if silhouette:
        scene.display.render_aa = "OFF"
        scene.render.image_settings.color_mode = "BW"
        sh.light = "FLAT"
        sh.color_type = "SINGLE"
        sh.single_color = (1.0, 1.0, 1.0)
        scene.world.color = (0.0, 0.0, 0.0)
    elif texture == "flat":
        # the textured body exactly as painted: no lighting, no antialiasing, black background
        scene.display.render_aa = "OFF"
        scene.render.image_settings.color_mode = "RGB"
        sh.light = "FLAT"
        sh.color_type = "TEXTURE"
        scene.world.color = (0.0, 0.0, 0.0)
    else:
        scene.display.render_aa = "8"
        scene.render.image_settings.color_mode = "RGB"
        sh.light = "STUDIO"
        sh.color_type = "TEXTURE" if texture else "OBJECT"
        scene.world.color = (0.82, 0.84, 0.88)


def render_to(scene, cam_obj, path):
    scene.camera = cam_obj
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def export_glb(path, objects):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True,
                              export_apply=True, export_yup=True)


def load_mesh(path):
    d = np.load(path)
    return d["verts"], d["faces"]


def task_silhouettes(t):
    reset_scene()
    verts, faces = load_any_mesh(t["mesh"])
    scene = reset_scene()
    obj = make_object("BaseMesh", verts, faces, color=(0.7, 0.7, 0.72))
    out = {"glb": None, "renders": {}}
    if t.get("glb"):
        export_glb(t["glb"], [obj])
        out["glb"] = t["glb"]
    for cam in t["cameras"]:
        setup_render(scene, cam["resolution"], silhouette=True)
        cobj = add_camera(cam)
        path = f'{t["out_dir"]}/{cam["view"]}.png'
        render_to(scene, cobj, path)
        out["renders"][cam["view"]] = path
    return out


def load_any_mesh(path):
    """verts/faces from an npz, or from a GLB (all meshes joined, transforms applied, back in Z-up)."""
    if not path.endswith(".glb"):
        return load_mesh(path)
    bpy.ops.import_scene.gltf(filepath=path)
    verts, faces, base = [], [], 0
    for o in bpy.context.scene.objects:
        if o.type != "MESH":
            continue
        me = o.data
        me.calc_loop_triangles()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        mw = np.array(o.matrix_world)
        verts.append(co @ mw[:3, :3].T + mw[:3, 3])
        tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
        me.loop_triangles.foreach_get("vertices", tri)
        faces.append(tri.reshape(-1, 3) + base)
        base += len(co)
    verts, faces = np.concatenate(verts), np.concatenate(faces)
    # glTF splits vertices along sharp edges; weld them back so the body is one closed surface
    _, first, inverse = np.unique(np.round(verts, 6), axis=0, return_index=True, return_inverse=True)
    faces = inverse.reshape(-1)[faces]
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    return verts[first], faces[keep]


def task_decimate(t):
    out = {}
    reset_scene()
    verts, faces = load_any_mesh(t["mesh"])
    reset_scene()
    obj = make_object("Body", verts, faces, smooth=False)
    n0 = len(faces)
    for lod, target in t["targets"].items():
        mod = obj.modifiers.new("Decimate", "DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = min(1.0, float(target) / n0)
        mod.use_collapse_triangulate = True
        dg = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
        me.loop_triangles.foreach_get("vertices", tri)
        np.savez(t["out"][lod], verts=co.reshape(-1, 3), faces=tri.reshape(-1, 3))
        out[lod] = {"triangles_in": n0, "triangles_out": len(tri) // 3, "ratio": round(mod.ratio, 5)}
        ev.to_mesh_clear()
        obj.modifiers.remove(mod)
    return out


HEAD_TEXTURE_MATERIAL = "HeadTexture"


def build_body(path, colors, image=None):
    """The LOD's sections as objects. ``image`` is one texture for every section (a path), or a dict with the
    shared atlas under "body" and a section's own texture under its name (the Head's, material "HeadTexture")."""
    d = np.load(path)
    names = sorted({k.split("__")[0] for k in d.files})
    images = image if isinstance(image, dict) else {"body": image}

    def texture(name):
        own = images.get(name)
        return (own, HEAD_TEXTURE_MATERIAL if name == "Head" else name + "Texture") if own else \
            (images.get("body"), "BodyAtlas")

    out = {}
    for name in names:
        img, mat = texture(name)
        out[name] = make_object(name, d[name + "__verts"], d[name + "__faces"], origin=d[name + "__pivot"],
                                color=colors.get(name), uv=d[name + "__uv"] if (name + "__uv") in d.files else None,
                                image=img, material=mat)
    return out


def task_uv(t):
    """Smart-project the sections of each LOD into one UV atlas; save per-corner UVs. Sections named in
    ``separate`` (the Head, which has its own texture) are each unwrapped and packed alone into the whole UV
    square; the others share one atlas. Islands are unwrapped without margin and packed by concave shape
    with a gutter given as a fraction of each texture side."""
    out = {}
    separate = list(t.get("separate", []))
    for lod, path in t["lods"].items():
        margin = t["margin"][lod] if isinstance(t.get("margin"), dict) else t.get("margin", 0.002)
        reset_scene()
        objs = build_body(path, {})
        groups = [[n for n in objs if n not in separate]] + [[n] for n in separate if n in objs]
        for names in groups:
            if not names:
                continue
            if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")
            for n in names:
                objs[n].select_set(True)
            bpy.context.view_layer.objects.active = objs[names[0]]
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            group_margin = margin
            if names[0] in separate:
                group_margin = (t["separate_margin"][lod] if isinstance(t.get("separate_margin"), dict)
                                else t.get("separate_margin", margin))
            bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.0,
                                     area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
            bpy.ops.uv.select_all(action="SELECT")
            bpy.ops.uv.pack_islands(rotate=True, margin=group_margin, shape_method="CONCAVE",
                                     margin_method="FRACTION")
            bpy.ops.object.mode_set(mode="OBJECT")
        arrs = {}
        for name, o in objs.items():
            me = o.data
            uv = np.empty(len(me.loops) * 2, dtype=np.float32)
            me.uv_layers.active.data.foreach_get("uv", uv)
            arrs[name + "__uv"] = uv.reshape(-1, 3, 2)
        np.savez(t["out"][lod], **arrs)
        out[lod] = {"sections": len(objs), "margin": margin}
    return out


def task_body(t):
    colors = t["colors"]
    out = {"glb": {}, "pose_renders": []}
    # One fresh scene per LOD so both GLBs carry the bare section names; the pose LOD is built last.
    pose_lod = t.get("pose_lod", "high")
    order = [lod for lod in t["lods"] if lod != pose_lod] + [pose_lod]
    objs = {}
    atlases = t.get("atlas", {})
    textured = bool(atlases)
    for lod in order:
        scene = reset_scene()
        objs = build_body(t["lods"][lod], colors, image=atlases.get(lod))
        if t.get("glb", {}).get(lod):
            export_glb(t["glb"][lod], list(objs.values()))
            out["glb"][lod] = t["glb"][lod]
    out["rest_silhouettes"], out["rest_textured"] = {}, {}
    for cam in t.get("rest_cameras", []):
        setup_render(scene, cam["resolution"], silhouette=True)
        path = f'{t["rest_dir"]}/silhouette-{cam["view"]}.png'
        render_to(scene, add_camera(cam), path)
        out["rest_silhouettes"][cam["view"]] = path
        if textured:
            setup_render(scene, cam["resolution"], silhouette=False, texture="flat")
            path = f'{t["rest_dir"]}/textured-{cam["view"]}.png'
            render_to(scene, add_camera(cam), path)
            out["rest_textured"][cam["view"]] = path
    rest = {name: o.matrix_world.copy() for name, o in objs.items()}
    for pose in t.get("poses", []):
        for name, o in objs.items():
            m = pose["matrices"].get(name)
            o.matrix_world = Matrix(m) @ rest[name] if m is not None else rest[name]
        for cam in t["pose_cameras"]:
            setup_render(scene, cam["resolution"], silhouette=False, texture=textured)
            cobj = add_camera(cam)
            path = f'{t["pose_dir"]}/{pose["name"]}-{cam["view"]}.png'
            render_to(scene, cobj, path)
            out["pose_renders"].append(path)
    return out


def main():
    job_path = sys.argv[sys.argv.index("--") + 1]
    job = json.loads(open(job_path).read())
    results = []
    for t in job["tasks"]:
        t0 = time.time()
        fn = {"silhouettes": task_silhouettes, "decimate": task_decimate, "body": task_body,
              "uv": task_uv}[t["task"]]
        r = fn(t)
        r["task"] = t["task"]
        r["seconds"] = round(time.time() - t0, 3)
        results.append(r)
    stem = job_path[: -len(".json")] if job_path.endswith(".json") else job_path  # only the file's own suffix
    with open(stem + ".result.json", "w") as fh:
        json.dump({"ok": True, "blender": bpy.app.version_string, "results": results}, fh, indent=2)


if __name__ == "__main__":
    main()
