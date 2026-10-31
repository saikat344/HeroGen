"""Blender-side atlas preparation, flat renders and export in the imported source frame."""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

POS_LO = -2.0
POS_EXT = 4.0   # the position pass stores (p - POS_LO) / POS_EXT, so world coordinates in [-2, 2] fit [0, 1]
SEAM_GRID = 1e-6  # vertices at the same position on this grid (scene units) are one point split at a UV seam


def reset(engine="BLENDER_WORKBENCH"):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return setup(bpy.context.scene, engine)


def setup(sc, engine):
    sc.render.engine = engine
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.display.shading.light = "FLAT"
    sc.display.shading.color_type = "TEXTURE"
    sc.render.film_transparent = True
    if sc.world is None:
        sc.world = bpy.data.worlds.new("w")
    if engine == "CYCLES":
        sc.cycles.device = "CPU"
        sc.cycles.samples = 1
        sc.cycles.use_denoising = False
        sc.cycles.max_bounces = 0
        sc.cycles.pixel_filter_type = "BOX"
        sc.cycles.filter_width = 0.01
    return sc


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    # Refinement derives its normals from geometry; imported custom corner normals become invalid when
    # the bake copy's UV-seam vertices are merged. Clear them before any mesh processing or joining.
    for o in objs:
        if o.data.has_custom_normals:
            bpy.context.view_layer.objects.active = o
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
    if len(objs) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.join()
        objs = [bpy.context.view_layer.objects.active]
    return objs


def add_camera(sc, cam):
    data = bpy.data.cameras.new(cam["name"])
    data.type = "ORTHO"
    data.ortho_scale = cam["ortho"]
    data.clip_start = 0.001
    data.clip_end = 100
    obj = bpy.data.objects.new(cam["name"], data)
    sc.collection.objects.link(obj)
    a = math.radians(cam["angle"])
    d = 10.0
    c = Vector(cam["centre"])
    obj.location = c + Vector((-math.sin(a) * d, -math.cos(a) * d, 0))
    obj.rotation_euler = (math.radians(90), 0, -a)
    sc.camera = obj
    sc.render.resolution_x, sc.render.resolution_y = cam["res"]
    sc.render.resolution_percentage = 100
    return obj


def emission_material(name, build):
    """A material whose emission colour comes from build(nodes, links) -> output socket."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(build(nt.nodes, nt.links), em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return m


def position_socket(nodes, links):
    geo = nodes.new("ShaderNodeNewGeometry")
    sub = nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = (POS_LO, POS_LO, POS_LO)
    div = nodes.new("ShaderNodeVectorMath")
    div.operation = "DIVIDE"
    div.inputs[1].default_value = (POS_EXT, POS_EXT, POS_EXT)
    links.new(geo.outputs["Position"], sub.inputs[0])
    links.new(sub.outputs[0], div.inputs[0])
    return div.outputs[0]


def normal_socket(nodes, links):
    geo = nodes.new("ShaderNodeNewGeometry")
    mad = nodes.new("ShaderNodeVectorMath")
    mad.operation = "MULTIPLY_ADD"
    mad.inputs[1].default_value = (0.5, 0.5, 0.5)
    mad.inputs[2].default_value = (0.5, 0.5, 0.5)
    links.new(geo.outputs["Normal"], mad.inputs[0])
    return mad.outputs[0]


def white_socket(nodes, links):
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (1, 1, 1, 1)
    return rgb.outputs[0]


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    for p in obj.data.polygons:
        p.material_index = 0


def render_colour(sc, path):
    setup(sc, "BLENDER_WORKBENCH")
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def render_position(sc, obj, path):
    keep = list(obj.data.materials)
    assign(obj, emission_material("pos", position_socket))
    setup(sc, "CYCLES")
    sc.view_settings.view_transform = "Raw" if "Raw" in [i.identifier for i in
                                                          sc.view_settings.bl_rna.properties["view_transform"].enum_items] else "Standard"
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    obj.data.materials.clear()
    for m in keep:
        obj.data.materials.append(m)


def texture_material(img):
    m = bpy.data.materials.new("refined")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 1.0
    nt.nodes.active = tex
    return m


def bake(obj, mat, img, margin):
    assign(obj, mat)
    target = mat.node_tree.nodes.new("ShaderNodeTexImage")
    target.image = img
    for n in mat.node_tree.nodes:
        n.select = False
    target.select = True
    mat.node_tree.nodes.active = target
    sc = bpy.context.scene
    setup(sc, "CYCLES")
    sc.render.bake.margin = margin
    sc.render.bake.margin_type = "EXTEND"
    sc.render.bake.use_clear = True
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type="EMIT", margin=margin, use_clear=True)


def bake_slots(obj, mats, img, margin):
    """Like `bake`, but every material slot keeps its faces and gets its own emission material."""
    for i, mat in enumerate(mats):
        obj.data.materials[i] = mat
        target = mat.node_tree.nodes.new("ShaderNodeTexImage")
        target.image = img
        for n in mat.node_tree.nodes:
            n.select = False
        target.select = True
        mat.node_tree.nodes.active = target
    sc = bpy.context.scene
    setup(sc, "CYCLES")
    sc.render.bake.margin = margin
    sc.render.bake.margin_type = "EXTEND"
    sc.render.bake.use_clear = True
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type="EMIT", margin=margin, use_clear=True)


def prep(a):
    out = Path(a["out"])
    out.mkdir(parents=True, exist_ok=True)
    reset("CYCLES")
    obj = import_glb(a["glb"])[0]
    me = obj.data
    old_uv = me.uv_layers.active.name
    src_imgs = []   # each material slot's base colour image
    for m in me.materials:
        img = None
        for n in (m.node_tree.nodes if m and m.use_nodes else []):
            if n.type == "TEX_IMAGE" and n.image is not None:
                img = n.image
        src_imgs.append(img)
    if not src_imgs or any(i is None for i in src_imgs):
        raise SystemExit("a material of the GLB has no base colour image")
    src_img = src_imgs[0]
    source_image_size = list(src_img.size)
    mw = obj.matrix_world.copy()
    mwi = mw.inverted()
    orig = [v.co.copy() for v in me.vertices]
    head = a.get("head")
    if head:
        c, r, k = Vector(head["centre"]), head["radius"], head["gain"]
        for v in me.vertices:
            p = mw @ v.co
            d = (p - c).length
            f = k * d if d < r else d + (k - 1) * r     # monotone radial map: no folding
            if d > 1e-9:
                v.co = mwi @ (c + (p - c) * (f / d))
    uv = me.uv_layers.new(name="refine")
    me.uv_layers.active = uv
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.0015, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    for v, co in zip(me.vertices, orig):
        v.co = co
    me.update()
    for l in me.uv_layers:
        l.active_render = l.name == "refine"
    size = a["tex"]

    def src_socket_for(image):
        def src_socket(nodes, links):
            uvn = nodes.new("ShaderNodeUVMap")
            uvn.uv_map = old_uv
            t = nodes.new("ShaderNodeTexImage")
            t.image = image
            t.interpolation = "Linear"   # Cubic bleeds SAM's black atlas gutters into thin triangles
            links.new(uvn.outputs[0], t.inputs[0])
            return t.outputs["Color"]
        return src_socket

    base = bpy.data.images.new("base", size, size, alpha=False)
    if len(src_imgs) == 1:
        bake(obj, emission_material("bake_src", src_socket_for(src_img)), base, 16)
    else:
        bake_slots(obj, [emission_material(f"bake_src{i}", src_socket_for(im)) for i, im in enumerate(src_imgs)],
                   base, 16)
    base.filepath_raw = str(out / "base.png")
    base.file_format = "PNG"
    base.save()
    # Smooth normals for the facing weights come from a copy whose split vertices (SAM splits them at its UV seams)
    # are merged; the mesh itself is never merged (merging made some thin faces bake black on s003).
    import bmesh
    twin = obj.copy()
    twin.data = me.copy()
    bpy.context.scene.collection.objects.link(twin)
    bm = bmesh.new()
    bm.from_mesh(twin.data)
    merged = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    merged -= len(bm.verts)
    bm.to_mesh(twin.data)
    bm.free()
    for p in twin.data.polygons:
        p.use_smooth = True
    for name, sock, margin, target in (("pos", position_socket, 4, obj), ("nrm", normal_socket, 4, twin)):
        img = bpy.data.images.new(name, size, size, alpha=False, float_buffer=True)
        img.colorspace_settings.name = "Non-Color"
        bake(target, emission_material("bake_" + name, sock), img, margin)
        img.filepath_raw = str(out / f"{name}.exr")
        img.file_format = "OPEN_EXR"
        img.save()
    cov = bpy.data.images.new("cov", size, size, alpha=False)
    cov.colorspace_settings.name = "Non-Color"
    bake(obj, emission_material("bake_cov", white_socket), cov, 0)
    cov.filepath_raw = str(out / "cov.png")
    cov.file_format = "PNG"
    cov.save()
    bpy.data.objects.remove(twin, do_unlink=True)
    me.uv_layers.remove(me.uv_layers[old_uv])
    for im in list(bpy.data.images):          # a transfer source brings its own "texture" image: never reuse it
        if im.name.startswith("texture"):
            bpy.data.images.remove(im)
    tex = bpy.data.images.load(str(out / "base.png"))
    tex.name = "texture"
    assign(obj, texture_material(tex))
    for m in list(bpy.data.materials):
        if m.users == 0:
            bpy.data.materials.remove(m)
    obj.name = "body"
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "work.blend"))
    (out / "prep.json").write_text(json.dumps({"old_uv": old_uv, "src_image": source_image_size, "tex": size,
                                               "head": head, "merged_vertices": merged, "vertices": len(me.vertices),
                                               "faces": len(me.polygons)}))


def open_work(blend, texture):
    bpy.ops.wm.open_mainfile(filepath=blend)
    obj = bpy.data.objects["body"]
    img = bpy.data.images["texture"]
    img.filepath = texture
    img.reload()
    return bpy.context.scene, obj


def render(a):
    out = Path(a["out"])
    out.mkdir(parents=True, exist_ok=True)
    sc, obj = open_work(a["blend"], a["texture"])
    for cam in a["cams"]:
        add_camera(sc, cam)
        render_colour(sc, str(out / f"{cam['name']}.png"))
        if a.get("pos"):
            render_position(sc, obj, str(out / f"{cam['name']}.pos.exr"))


def probe(a):
    out = Path(a["out"])
    out.mkdir(parents=True, exist_ok=True)
    sc = reset()
    obj = import_glb(a["glb"])[0]
    for cam in a["cams"]:
        add_camera(sc, cam)
        render_colour(sc, str(out / f"{cam['name']}.png"))
        render_position(sc, obj, str(out / f"{cam['name']}.pos.exr"))


def smooth_seam_normals(me):
    """Give every face corner the area-weighted normal of all faces at its position.

    The source arrives split at its UV seams, so its copies of one point carry separate vertex normals; averaging
    over the position, not the vertex, smooths across those seams. Vertices and faces are left as they are."""
    n = len(me.vertices)
    co = np.empty(n * 3)
    me.vertices.foreach_get("co", co)
    _, group = np.unique(np.round(co.reshape(-1, 3) / SEAM_GRID).astype(np.int64), axis=0, return_inverse=True)
    group = group.ravel()
    pn, area = np.empty(len(me.polygons) * 3), np.empty(len(me.polygons))
    me.polygons.foreach_get("normal", pn)
    me.polygons.foreach_get("area", area)
    pn = pn.reshape(-1, 3)
    loop_vert = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", loop_vert)
    starts, totals = np.empty(len(me.polygons), np.int64), np.empty(len(me.polygons), np.int64)
    me.polygons.foreach_get("loop_start", starts)
    me.polygons.foreach_get("loop_total", totals)
    loop_poly = np.repeat(np.arange(len(me.polygons)), totals)
    acc = np.zeros((group.max() + 1 if n else 0, 3))
    np.add.at(acc, group[loop_vert], pn[loop_poly] * area[loop_poly, None])
    out = acc[group[loop_vert]]
    length = np.linalg.norm(out, axis=1, keepdims=True)
    out = np.where(length > 1e-12, out / np.maximum(length, 1e-12), pn[loop_poly])
    me.shade_smooth()
    me.normals_split_custom_set(out.tolist())


def export(a):
    sc, obj = open_work(a["blend"], a["texture"])
    img = bpy.data.images["texture"]
    img.pack()
    Path(a["out"]).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    # Without normals a viewer must light every triangle flat, and a coarse face then splits into lit and unlit
    # facets. Export smooth normals without changing the topology.
    smooth_seam_normals(obj.data)
    bpy.ops.export_scene.gltf(filepath=a["out"], export_format="GLB", export_image_format="AUTO", export_texcoords=True, export_normals=True,
                              export_materials="EXPORT", export_yup=True, use_selection=True)



def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    mode, args = argv[0], json.loads(Path(argv[1]).read_text())
    {"probe": probe, "prep": prep, "render": render, "export": export}[mode](args)
    Path(argv[1] + ".result.json").write_text(json.dumps({"blender": bpy.app.version_string}))


if __name__ == "__main__":
    main()
