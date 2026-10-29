# HeroGen architecture: stages, run folder and file formats

A pipeline that turns one hand-drawn character sketch into a game-ready, sectioned 3D body, with a measured
check after every stage. Each stage reads and writes files in one run folder, so stages can be built, run and
tested separately.

## Run folder

`$HEROGEN_RUNS/<run-id>/` (default `HEROGEN_RUNS=./runs`, git-ignored). Every stage writes its outputs and one
`stage-<n>.json` record, then stops if its gate fails.

| Stage | Owner package | Writes |
| --- | --- | --- |
| 0 input | `herogen/pipeline` | `00-sketch.png` (copied input) |
| 1 cover art | `herogen/pipeline` | `01-cover.png` |
| 2 turnaround | `herogen/pipeline` | `02-turnaround.png` (all views in one image), `views/{front,back,left,right,top}.png`, `masks/{view}.png` (white figure on black, same size as the view) |
| 3 base mesh | `herogen/blender` | `03-mesh.glb`, `03-renders/{view}.png` (rendered from the same cameras, silhouette only) |
| 4 game body | `herogen/blender` | `04-body/high.glb`, `04-body/low.glb`, `04-body/sections.json`, `04-body/pose-renders/*.png` |
| 5 results | `herogen/pipeline` | `results.html`, `results.json` |
| 6 Roblox export | `herogen/roblox` | `06-roblox/HeroGen-R15.rbxmx`, `06-roblox/bind_meshes.luau`, `06-roblox/rig_test.server.luau`, `06-roblox/README.txt`, `stage-6.json` |

Stage 6 writes the files Roblox Studio imports (see "Roblox export"); play-testing in a game engine is not part
of this pipeline.

## Stage record: `stage-<n>.json`

```json
{
  "stage": 2,
  "name": "turnaround",
  "ok": true,
  "started": "2026-10-28T19:00:00-07:00",
  "seconds": 41.2,
  "cost_usd": 0.02,
  "cost_basis": "estimate: image count x configured price",
  "models": ["muse-image-1.0", "sam-3.1"],
  "outputs": ["02-turnaround.png", "views/front.png"],
  "checks": [
    {"name": "height_agreement_front_side", "value": 0.97, "threshold": 0.95, "pass": true}
  ],
  "notes": ""
}
```

`checks` hold measured numbers only. A gate fails when any check with `"gate": true` (default true) fails.

## Views and cameras

- Orthographic views. Character faces **-Y** in Blender (front camera on -Y looking +Y), +Z up, feet at Z = 0.
- `front`: camera on -Y looking +Y. `back`: camera on +Y looking -Y. The character's left is +X, so `left`:
  camera on +X looking -X, and `right`: camera on -X looking +X. `top`: camera above looking down -Z, the
  character's front at the bottom of the image. Every view image is square, the figure centred, the same pixel
  scale across views (the split step pads, it never rescales one view alone).
- `views/meta.json`: `{"pixels_per_unit": <float>, "height_units": <float>, "image_size": <int>}`.

## Stage 4 body format: `sections.json`

R15-style rigid sections: `Head, UpperTorso, LowerTorso, LeftUpperArm, LeftLowerArm, LeftHand, RightUpperArm,
RightLowerArm, RightHand, LeftUpperLeg, LeftLowerLeg, LeftFoot, RightUpperLeg, RightLowerLeg, RightFoot`.

```json
{
  "units": "studs",
  "height": 5.5,
  "sections": [
    {"name": "Head", "pivot": [0.0, 0.0, 4.6], "triangles_high": 1200, "triangles_low": 300,
     "bounds_min": [-0.5, -0.5, 4.5], "bounds_max": [0.5, 0.5, 5.5]}
  ],
  "joints": {"Neck": [0.0, 0.0, 4.5]},
  "budgets": {"triangles_high_max": 24000, "triangles_low_max": 6000},
  "checks": []
}
```

## Held-out view, mask method and covers

**Held-out view (stage 2h).** After stage 2 passes, one extra image of the same character in the same A-pose
seen from the front-left at about 45 degrees: `heldout/front-left-34.png` and `heldout/front-left-34-mask.png`,
with its checks in `stage-2h.json`. No building stage reads `heldout/`; it exists so a body can be scored on a
view it was not built from. The angle the image model actually drew is unknown, so a scorer has to search yaw
and fit scale and offset from the mask's bounding box. When a run folder holds `05-scores.json`, the results
page shows its per-body table (see `herogen/pipeline/results.py` for the format).

**Mask method.** `views/meta.json` records how each mask was made: `"mask_method": {"front": "sam-3.1", ...}`
(`"sam-3.1"`, `"threshold"` or `"visual_hull"`), and `"top_method"`.

**Shoulder covers.** Stage 4 adds one cover per shoulder, merged into its owner section (`UpperTorso`) so each
GLB keeps exactly 15 meshes. `sections.json` lists them under `"covers"`: the owner, the shaping parameters, and
`pieces_high` / `pieces_low` with each piece's triangles. Covers count in the triangle budgets.

## Roblox export

```bash
python3 -m herogen.roblox export --body <run>/04-body --out <run>/06-roblox
python3 -m herogen.roblox check --body <run>/04-body --rig <file>.rbxmx
```

`herogen/roblox` uses Python and numpy only. `export` writes the rig, checks it and records the result; `check`
runs the same checks on any `.rbxmx` against a body. Exit codes: 0 when every gate check passes, 1 when one
fails (each failure is printed), 2 when the body can't be exported (one `error:` line, and for `export` a
`stage-6.json` with `"ok": false` and an `"error"`).

**Outputs** (in `--out`, except the record; the export first deletes any of these files left in `--out` by an
earlier run):

| File | What it is |
| --- | --- |
| `HeroGen-R15.rbxmx` | The R15 rig, a Roblox XML model (`<roblox version="4">`) |
| `bind_meshes.luau` | A Command Bar script that binds the meshes Studio's importer makes from `high.glb` onto the rig |
| `rig_test.server.luau` | An optional Script that plays stock R15 catalog animations and prints joint separation |
| `README.txt` | Step-by-step Studio instructions for the end user, with the path to `high.glb` and the HipHeight |
| `stage-6.json` | The stage record: at the run root when `--out` is a run's `06-roblox` folder (its parent holds `stage-4.json` or `04-body/`), otherwise in `--out` |

**Frames.** Contract B: +Z up, faces -Y, left +X. GLB G = (Bx, Bz, -By). Roblox R: +Y up, faces -Z, the
character's left is -X. R = (-Bx, Bz, By) = (-Gx, Gy, -Gz), a 180 degree turn about Y from G (not a mirror).
Units are studs throughout.

**The rig** (`rig.py`, `rbxmx.py`):
- A Model named `HeroGen-R15`, its PrimaryPart the HumanoidRootPart.
- A Humanoid with RigType R15, HipHeight, and NumberValues BodyHeightScale, BodyWidthScale, BodyDepthScale and
  HeadScale at 1 and BodyTypeScale and BodyProportionScale at 0.
- HumanoidRootPart: a Part of 2 x 2 x 1 studs, Transparency 1, centred on the `LowerTorso` pivot (the Root
  joint).
- 15 MeshParts with the R15 names. Each one's Size is its GLB node's exact vertex bounds in R, its CFrame the
  bounds centre with no rotation, Anchored false, and a StringValue `AvatarPartScaleType` = `Classic`.
- 15 Motor6Ds, one in each section, named after its joint, with Part0 the parent section and Part1 the section.
  Both parts hold an Attachment `<joint>RigAttachment` at the `sections.json` pivot, world-aligned; the
  Motor6D's C0 and C1 are those two attachments' CFrames, as `Humanoid:BuildRigFromAttachments` sets them.
- 19 accessory attachments from the Roblox character specification, world-aligned unless noted:
  RootAttachment at the root part's centre; on the Head, FaceCenter at its centre, FaceFront on its front face,
  Hat and Hair on top; on the UpperTorso, Neck at the neck pivot, BodyFront and BodyBack on its front and back
  faces, and Left/RightCollar above the shoulder pivots, halfway to its top; on the LowerTorso, WaistCenter at
  the Root pivot and WaistFront and WaistBack level with it on its front and back faces; Left/RightShoulder on
  top of the upper arms; Left/RightGrip at the hand centres, turned -90 degrees about X so they point forward;
  Left/RightFoot at the bottom centre of the feet.
- HipHeight = (root part centre Y - 1) - the lowest vertex of the two feet, so the feet stand on the floor
  (Roblox: height above the floor = 0.5 * RootPart.Size.Y + HipHeight).
- **Every MeshId is empty.** A MeshPart can only reference an uploaded mesh asset, so the rig carries the parts'
  sizes and placement and `bind_meshes.luau` supplies the meshes in Studio. The record says so under
  `rig.meshes`, and the `meshes_bound` info check counts sections with a MeshId (0 in an exported rig).

The file holds only the properties listed; Studio gives every other property its class default. Floats are
float32 in shortest round-trip decimals and referents are numbered in document order, so a body always gives
the same bytes.

**Refused bodies** (exit 2):
- a missing or unreadable `sections.json` or GLB, or a GLB that isn't glTF 2.0 binary with an embedded buffer;
- units other than studs, or a frame other than the contract's;
- sections that aren't exactly the 15 R15 sections with R15 parents and joints, or lack a pivot or bounds;
- a GLB without a mesh node for every section, or with two mesh nodes of one name;
- mesh nodes that are rotated, scaled, given a matrix, nested or not triangles, or positions that aren't dense
  float VEC3;
- a section whose vertex bounds are flat or not finite;
- a Root joint so low that HipHeight would be zero or less with the 2-stud root part;
- any section over 20,000 triangles, Roblox's limit per mesh (the message says to use the Roblox in-game or
  avatar profile).

**Checks** (`check.py`). The checker shares no code with the writer: it reads the file back with Python's XML
parser, restates the R15 tree and attachment list, derives the frame conversion again, and takes part extents
from the GLB's POSITION accessor min/max and pivots and toe points from `sections.json`.

| Check | Passes when |
| --- | --- |
| `xml_well_formed`, `one_root_model` | The file parses as `<roblox version="4">` holding one Model |
| `xml_values_valid` | Only written when a property value can't be read (the checks stop there) |
| `humanoid_r15`, `humanoid_scale_values` | One Humanoid with RigType R15; the six scale values as above |
| `root_part` | HumanoidRootPart is a Part, the PrimaryPart, with Transparency 1 |
| `sections_r15_names` | HumanoidRootPart and exactly the 15 R15 MeshParts, each once, no other parts |
| `joints_named_and_linked` | Each section has one Motor6D, named after its joint, Part0 its parent, Part1 itself (15) |
| `joint_chain_reachable` | All 15 sections are reached from HumanoidRootPart through the Motor6Ds |
| `rig_attachment_pairs` | Both `<joint>RigAttachment`s exist for all 15 joints |
| `attachments_at_pivots`, `attachment_pairs_coincide` | Attachments within 0.001 stud of the pivots and of each other |
| `motor_c0_c1_equal_attachments` | C0 and C1 equal the attachment CFrames within 1e-5 |
| `rest_pose_position`, `rest_pose_rotation` | Part0 * C0 * C1^-1 = Part1 within 1e-4 stud and 1e-5 |
| `joints_inside_parts` | Each pivot lies inside both its parts' boxes (the root part excepted), with 0.05 stud slack |
| `section_extents_match_glb`, `section_centres_match_glb` | Each part's box matches its GLB node's bounds within 0.001 stud |
| `rotations_proper` | Every CFrame is orthonormal with determinant +1 (within 1e-5) |
| `triangles_per_mesh` | No section over 20,000 triangles |
| `hip_height` | HipHeight within 0.001 stud of the value that puts the lowest foot vertex on the floor |
| `feet_lowest` | No section reaches more than 0.001 stud below the feet |
| `faces_minus_z` | The toes lie ahead of the ankles along -Z (forward Z below -0.9) |
| `not_mirrored` | RightUpperArm - LeftUpperArm points to the character's right (cosine above 0.9) |
| `face_front_on_front` | FaceFrontAttachment lies on the head's -Z side |
| `accessory_attachments`, `avatar_part_scale_type` | All 19 present and inside their parts; every section `Classic` |
| `meshes_bound` (info, not a gate) | The number of sections with a MeshId |

Tests (`tests/test_herogen_roblox_*.py`) build synthetic box bodies (`herogen/roblox/synth.py`) and corrupt
rigs to show each check catches its fault: a C0 offset, a renamed part, a missing joint, a wrong HipHeight, a
moved neck pivot, a mirrored rig, a resized foot, a reflected rotation, a wrong rig type, a mislinked Motor6D,
an extra part, an unreadable property value, a body facing backwards, a writer with a mirrored frame
conversion and a section below the feet.
Another test installs the project with `pip install --target` from a copy of the checkout and runs an export
from the installed package, which shows the templates ship as package data (`[tool.setuptools.package-data]`
in `pyproject.toml`); it skips only when pip can't fetch the build backend. Two optional tests run when their
tool is on `PATH`: with `lune`, rbx-dom reads the file back and every CFrame and size must agree within 1e-5
stud; with `luau-compile`, both Studio scripts must compile.

**Stage record** (`stage-6.json`, placed as in the outputs table): the stage-record fields above, with
`outputs` relative to the record's folder, plus `code` (`{"commit", "dirty"}`, or null outside a git
checkout), `body` (the body folder relative to the record's folder), `inputs` (sha256 and size of
`sections.json` and the GLB), and `rig` (name, frame, joint type, HipHeight, root part, and each
section's centre, size, pivot and triangles).

**In Studio** (`README.txt` gives the end user these steps):
1. File > Import 3D on `high.glb` with Scale Unit Stud and Rig Type No Rig. The importer uploads the meshes and
   texture to the user's account.
2. Insert `HeroGen-R15.rbxmx` from file.
3. Run `bind_meshes.luau` in the Command Bar. It finds each section's imported MeshPart by name (ignoring `_Node`
   and `_Geo` suffixes), fits the imported layout to the rig with one of the 24 axis-aligned rotations plus a
   translation, and stops, leaving the rig unchanged, if the worst centre residual is 0.01 stud or more or any
   imported part's size differs from the rig's by 0.01 stud or more. Only then does it call
   `MeshPart:ApplyMesh` on each section, copy any SurfaceAppearance, keep every attachment where it was in the
   world, reset each Motor6D's C0 and C1 to its attachments, and print the residual and the rest-pose joint
   error.
4. Optionally run `rig_test.server.luau` inside the rig, then use the rig as `StarterPlayer.StarterCharacter`.

The Studio scripts and these steps have not yet been run in a Studio session. The joints are Motor6D; an
AnimationConstraint variant is not written.

## Rules

- No secrets in code, logs or run folders. Model keys come from the environment.
- Committed tests use synthetic inputs only. No game assets, no real sketches, no personal paths.
- Model calls go through one client module per package with a replay mode for tests.
