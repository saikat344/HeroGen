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
| 2 turnaround | `herogen/pipeline` | `02-turnaround.png` (all views in one image), `views/{front,back,left,right,top}.png`, `masks/{view}.png` (white figure on black, same size as the view), `masks/front-{face,head}.png` (front-view part masks, when SAM 3.1 finds them) and `masks/parts.json` (their record) |
| 3 base mesh | `herogen/blender` | `03-mesh.glb`, `03-joints.json`, `03-renders/{view}.png` (rendered from the same cameras, silhouette only); with SAM 3D Objects also `03-source.glb` (the model's own textured mesh) |
| 4 game body | `herogen/blender` | `04-body/high.glb`, `04-body/sections.json`, `04-body/atlas-high.png`, `04-body/pose-renders/*.png`; `04-body/low.glb` and `atlas-low.png` when the profile has a low LOD (`roblox-ingame`); for a SAM 3D Objects mesh also the comparison atlases `04-body/atlas-high-{no-source-fallback,side-trust}.png` |
| scores (after 4) | `herogen/blender` | `05-scores.json`, `05-renders/<body>/<view>{,-silhouette,-diff}.png`, `05-renders/heldout-comparison*.png` |
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
Stage 3 and 4 records (and `sections.json`) also record the code that built them,
`"code": {"commit": "<git HEAD sha>", "dirty": <tracked files differ from it>}`, or `null` outside a git
checkout, so a body can be compared with the version that made it.

A stage that stops before its checks are complete writes its record with `"ok": false`, a `status` and an
`error`, plus every call made so far: `"error"` (the Model API failed after the client's retries, or any other
error) or `"input_rejected"` (an input check rejected the sketch, see below). `python3 -m herogen.pipeline run` exits 0
when every stage passed, 2 when a gate failed, 3 when the input was rejected and 1 on an error.

## Input checks (stages 0 and 1)

Before any image is generated, stages 0 and 1 check that the sketch can be used (`herogen/pipeline/inputs.py`).
Each input check is an ordinary check (name, value, threshold, op, pass, gate) plus a `verdict`:

- `"pass"`.
- `"reject"`: the check is gated. The stage writes its record with status `"input_rejected"` and every check it
  measured, the run stops with exit status 3, and the record's `error` joins the `message` of every rejecting
  check, each saying what is wrong and what to do.
- `"warn"`: not gated. The check keeps its `message`, the stage's `notes` repeat it, the CLI prints it, and the
  run goes on.

| Check | Stage | Measures | Verdict when it fails |
| --- | --- | --- | --- |
| `min_side_px` | 0 | the sketch's shorter side in pixels, `>= 256` | reject |
| `sketch_line_contrast` | 0 | how much darker supported strokes are than the paper: median image luminance minus the 1st percentile within dark stroke samples (0-255), `>= 32`. A sample is the brightest pixel in a straight three-pixel run darker than the median; at each centre use the darkest horizontal, vertical or diagonal run. Paper pixels, isolated dark pixels and pairs are excluded from the percentile; no samples means 0. No minimum fraction of the page must be ink. Paper must occupy most of the image; light lines on dark paper measure as faint. No model call | reject |
| `upload_real_person_photo_probability` | 1, before every other model call | Muse Spark's probability that the upload is a photograph of a real person, `< 0.5`, from the frozen question `real-person-photo-v1` (`inputs.PHOTO_JUDGE`: prompt, model, settings and strict schema, pinned by SHA-256 in `PHOTO_JUDGE_SHA256`; a changed question is refused until it gets a new name and hash). `details.input.photo` records the question's name and hash, the probability, the `image_kind` and the reason | reject |
| `sketch_characters_found` | 1 | SAM 3.1 figure instances (the prompt that found the figure, see Mask method) at least 25% of the largest one's area, `== 1`. None is "no character found in the sketch"; two or more are several characters | reject |
| `sketch_small_extra_figures` | 1 | the other, smaller figure instances (insets, detail drawings), `<= 0` | warn |
| `sketch_feet_on_figure` | 1 | SAM 3.1 `foot` instances on the main figure (at least half their area on its mask), `>= 1`. With none, SAM 3.1 `leg` is asked too | reject when legs are found on the figure (it is cropped); warn when none are (a long robe hides the lower body) |
| `sketch_face_offset_in_head` | 1 | how far the face is turned: the SAM 3.1 `face` centroid's offset from the `head` centroid over the head's box width (the held-out views' face cue), on the most confident face and head mostly on the main figure, `abs <= 0.23` (`details.input.face_offset` keeps the sign, + is image right). With no head or face on the figure, or a face mostly outside the head, the value is null and the verdict `"not_measured"`, which neither warns nor stops | warn |

Stage 1 asks the photo question before any other model call, so a photo of a real person is rejected before SAM 3.1
(or anything else) sees it: no face or head prompt runs on it. Stage 1 runs its input checks on the sketch before
it asks for the cover, so a rejected sketch costs no edit. The body-part checks are measured only on a sketch with
exactly one character, and their prompts are listed in `details.input.sam_prompts` (`foot`, `leg` only when no foot
is found, `face`, `head`), with the counts behind them (`feet_on_figure`, `legs_on_figure`: null when not asked).
The sketch's figure mask (`01-masks/sketch.png`, compared with the cover) is the largest instance alone, never a
union with the smaller figures. `details.input` holds the measurements behind the checks (`figure_area_fracs`: each
instance's area over the largest one's).

How the thresholds were set: on the tuning sketches s001-s045 only, with the unseen sketches s046-s058 kept for
reporting. With the supported-stroke measurement, the two faint-pencil sketches measure a line contrast of 11 and
12 and the 43 others 88-253. The contrast cut stays at 32, approximately the geometric mean of 12 and 88 (32.5).
The two
multiple-character sketches have a second instance at 0.66 and 0.77 of the largest and every other sketch has one
instance, so the tuning set bounds the character cut only to (0, 0.66]; 0.25 is kept from the diagnosis that
proposed it.

Every tuning sketch whose figure SAM found has a foot on it except five: the two cropped at the knees (s008, s041),
where `leg` finds two legs on the figure, and the two robes hiding the feet (s016, s031) and the side view s045,
where it finds none.

The two side-view sketches have a face offset of -0.30 (s040) and +0.43 (s045), and every other sketch where it
could be measured is within 0.157 (s033), three-quarter sketches included; 0.23 is the middle of that gap. Two side
views are too few to reject on, so the check warns.

The photo threshold was set on tuning images: the sketches s001-s045, 12 photo-style images of people made with
Muse Image, 5 photo-realistic and 8 stylised 3D renders of characters, 4 phone photos of drawings and 2 photos of a
toy and a statue. The photos of people measure 0.95-1.0 and everything else 0-0.10, and 0.5 lies inside that gap.
The photos of people are generated, not photos of real people; real photographs are not yet part of the measurement.

Besides the figure prompt and the photo question (about $0.002-0.004), the input checks make 3 or 4 SAM 3.1 calls on the sketch ($0.0075-0.01 at list price).
A run recorded without them has no cache entries for those calls, so replaying its stage 1 stops on a replay miss.

## Views and cameras

- Orthographic views. Character faces **-Y** in Blender (front camera on -Y looking +Y), +Z up, feet at Z = 0.
- `front`: camera on -Y looking +Y. `back`: camera on +Y looking -Y. The character's left is +X, so `left`:
  camera on +X looking -X, and `right`: camera on -X looking +X. `top`: camera above looking down -Z, the
  character's front at the bottom of the image. Every view image is square, the figure centred, the same pixel
  scale across views (the split step pads, it never rescales one view alone).
- `views/meta.json`: `{"pixels_per_unit": <float>, "height_units": <float>, "image_size": <int>}`.

## Stage 4 body format: `sections.json`

Stage 4 builds the body as R15-style rigid sections: `Head, UpperTorso, LowerTorso,
LeftUpperArm, LeftLowerArm, LeftHand, RightUpperArm, RightLowerArm, RightHand, LeftUpperLeg, LeftLowerLeg,
LeftFoot, RightUpperLeg, RightLowerLeg, RightFoot`.

```json
{
  "units": "studs",
  "height": 5.5,
  "form": "rigid",
  "sections": [
    {"name": "Head", "pivot": [0.0, 0.0, 4.6], "triangles_high": 1200, "triangles_low": 300,
     "bounds_min": [-0.5, -0.5, 4.5], "bounds_max": [0.5, 0.5, 5.5]}
  ],
  "joints": {"Neck": [0.0, 0.0, 4.5]},
  "profile": "roblox-ingame",
  "budgets": {"triangles_high_max": 24000, "triangles_low_max": 6000},
  "checks": []
}
```

**Cuts and pivots.** Each cut is a plane through a joint (`herogen/blender/split.py`); the pivot is the area
centroid of the child's largest opening there (largest by area), so it depends only on the cut's shape, not on
how densely the mesh is sampled along it (the mean of the opening's vertices, used before, moved the
pencil-sketch hybrid's waist pivot 0.06 studs between a 16k-triangle decimation and the full mesh). A cut vertex
stays at least 1e-4 studs from both ends of the edge it splits, so a plane through a vertex leaves no sliver for
welding to collapse.

Stage 4 may then move the neck, waist, elbow, wrist, knee and ankle cuts along their normals by up to half the
child bone's radius (`herogen/blender/cuts.py`): a cut on a costume ledge (a belt, a hem) leaves the wider side's
rim standing proud when the joint bends. Only the gate poses choose; the held-out poses are measured and reported
beside them and never used to choose a cut. Joints move in groups (neck with waist, elbows with wrists, knees with
ankles), left and right alike; a joint may move only if its unmoved gap is within 0.045 studs of the gate, a move
must lower its group's worst gate gap by more than 0.025 studs and may not open a neighbouring joint by more, and
among such moves the lowest gap plus 0.025 studs per bone radius moved wins. Both constants are measured: over nine
decimations of one stage 3 mesh a joint's worst gate gap varies by up to 0.044 studs, 0.018-0.031 at the 90th
percentile. The search runs once on a 20,000-triangle decimation of the stage 3 mesh and every LOD of every profile
is cut at the same places; `sections.json` and `stage-4.json` record it under `"cuts"`, with every candidate tried
and the reference body's worst gate and held-out gaps, cuts unmoved and moved. On the pencil-sketch figure it moves
the hybrid's waist up half a radius (reference body, waist: gate 0.183 to 0.150, held-out 0.191 to 0.132) and the
carve's waist up half a radius, knees a quarter radius down and ankles half a radius up (carve full detail, worst
gate gap 0.127 to 0.116). The worst gaps left are at joints it does not move: the hybrid's left hip (0.28 in
walk_a) and the carve's right shoulder (0.156 in arms_across).

The cut search requires a nonempty child and parent section and a nonempty shared rim for every joint it
measures. An unmeasurable baseline stops stage 4 with an error naming the joints and the empty sections or
missing rims; the stage 3 mesh and joint placement need correction before cuts can be chosen. A candidate move
that removes an own or neighbouring joint's measurement is rejected and recorded in the trial's
`unmeasurable_joints` map (joint to reason), without a gap or score. Missing measurements never count as zero.

**Atlases.** All sections of a LOD share one UV atlas: Blender's Smart UV Project unwraps them without margin, and
the islands are packed by their concave shapes with a 2-texel gutter. `roblox-ingame` uses 1024 for the high and
512 for the low LOD. `best` and `roblox-avatar` take the smallest of 1024, 2048 and 4096 that gives at least one
texel per pixel of the views the colours come from, at the atlas's measured UV fill and the body's surface area
(at most 4096 under `best`, 2048 under `roblox-avatar`); `sections.json` records the choice under
`"texture": {"atlas_choice"}`. A finer atlas only reproduces the views the colours were baked from more exactly:
on the pencil-sketch figure (1078 px views, 168.5 px per stud), from 1024 to 2048 to 4096 the full-detail carve's
front colour error at the view's resolution fell 5.34, 3.06, 1.93, while the held-out view's at a quarter of its
size, about as large as the body is on a screen, stayed 39.19, 39.35, 39.38 (half a degree of yaw moves it by up
to 0.09; the hybrid: 38.02, 38.26, 38.31, up to 0.26). Both chose 2048 (229 and 246 texels per stud). UV fill
(the fraction of the atlas the triangles cover, measured as the summed area of the UV triangles): hybrid full
detail 48% at 2048, in-game 41% / 35% (high / low), avatar 51%; carve full detail 58%, in-game 49% / 44%, avatar
58%, against 15% to 32% with the earlier fixed 0.004 island margin. On the hybrid the side views' texture follows its shape: at 2048 left 11.92 / 0.702, within
the gates by a small margin, and at the in-game 1024 left 13.40 / 0.653, which fails them.

**Pose test.** Each section is rotated about its pivot into the test poses (`herogen/blender/pose.py`); a rim
point of a cut that one section carries outside the other section's closed surface is an exposed cut, and its
distance to that surface is the gap. The gated gap counts only exposures a ray in one of 26 directions can see.
The gate set is ZBurst's intake set (walk, arms forward, arms up, crouch; the twist is reported, not gated) and
thirteen more poses that reach the same joints further and in other directions; eight held-out poses over the
same joint ranges are measured and reported (`max_joint_gap_<lod>_heldout`, not gated) and choose nothing. On the
pencil-sketch carve the wider gate set fails the in-game body that passed the intake set: its right shoulder in
arms_across gaps 0.156. The inside, distance and ray queries go through a uniform grid of each section's
triangles (`herogen/blender/meshindex.py`) instead of testing every point against every triangle. They give the
same answers: distances and ray hits pair by pair, and the inside test (signed ray crossings) the generalised
winding number's for every point off the surface. Measured on a pencil-sketch hybrid body, every joint's gap in
every pose is identical either way, at full detail (171,716 triangles; visible pass 503 s brute force, 7.0 s
through the grid; geometric 187 s and 3.0 s) and on the 22,768-triangle in-game body (32.7 s and 3.3 s).

## Turnaround sheet (stage 2)

Stage 2 asks Muse Image for a sheet of four views up to four times, stopping at the first sheet whose gated
checks all pass. It keeps the best sheet: fully passing sheets first, then sheets failing only the carve
check, then sheets failing any other gate. Within each group it prefers the fewest failed gated checks,
then the smallest summed shortfall (`checks.shortfall`: how far each failed value is past its threshold,
as a fraction of the threshold); a tie keeps the earlier sheet. A gate a sheet was never measured on counts
as failed with a shortfall of 1, so a sheet that could not be split into four figures ranks below every sheet
that could.
`stage-2.json` lists every attempt with its checks and `rank` (`fails_non_carve_gate`, `failed_gates`,
`shortfall`, compared in that order), and `details.kept_attempt` names the kept
one; `02-turnaround.png`, `views/`, `masks/`, the record's checks and `view_assignment` are all the kept
sheet's. A top view is requested for an attempt whose non-carve side gates pass, even if its preliminary
carve fails, so every such attempt is ranked on the masks that would be saved. Each top request makes one
Muse Image edit call and one or two SAM 3.1 calls. Four eligible attempts therefore make four top requests,
but the prompt and cover are identical: the client caches the image and its SAM masks. With an initially
empty cache, the top requests together need one paid Muse edit and one or two paid SAM calls (besides the
sheet and part-mask calls); later attempts reuse those results and scale them to their own sides. Top checks
and the final carve can change retrying and the kept sheet, and a top API error stops the stage; deferring top requests until
after selection would change those outcomes.

**Carve check (`carve_silhouette_iou_min`).** Every sheet that splits into four figures is carved exactly as
stage 3 carves it (`herogen.blender.views.views_from_masks` and `herogen.blender.stage3.carve_view_ious`: the
same joints, arm prior, voxel resolution and smoothing, with the mesh rasterised in numpy instead of rendered
by Blender; on six stage-3 runs the two agreed to 1e-4 in every view). The worst view's silhouette IoU is checked
against stage 3's carve gate, `IOU_GATE` (0.90), and the note names the worst view and lists every IoU. Views that
disagree (a staff in one view only, arms angled differently front and back, one leg hidden in the back) carve
into a body that keeps only what every view shows, which stage 3's carve gate refuses; checked here, the sheet
is retried. The MVP body is the hybrid (SAM 3D Objects plus Body), with carve only a fallback. The carve check
remains one of the six side-sheet retry/ranking gates; after attempts are exhausted, a kept sheet failing
only this check passes stage 2. Its stage record preserves the failed value and `pass: false`, sets
`gate: false`, and notes that the carve of these views would fail stage 3's gate, so a carve fallback would
be refused. Attempt records retain `gate: true` and their original ranks. The other five side gates and all
top-view gates still fail the stage as before; a carve failure alongside another gate remains a stage failure.
A drawn top view is used only if the carve with it still passes the gate (stage 3 carves with
an observed top view and gates its IoU); otherwise the visual-hull top is used and a note says why, and
`details.carve_ious_with_drawn_top` records the IoUs that decided it. The final carve includes whichever top
will be saved, including a visual-hull fallback; the derived top constrains the geometry but its own IoU is
not gated. Its carve replaces the preliminary side-only carve before ranking and acceptance, so a failure
uses the remaining sheet attempts. `details.carve_ious_with_saved_top` records the final IoUs.

**Part masks.** After the views are cut, SAM 3.1 segments `face` and `head` on `views/front.png`
(`herogen/pipeline/parts.py`). Each part keeps the most confident instance lying at least half on the front mask
grown by 4 px, clipped to the figure, written as `masks/front-face.png` / `masks/front-head.png`.
`masks/parts.json` records, per part, `file` (null when SAM found nothing on the figure), `method`, `instances`,
`confidence` and `pixels`, or `api_error` and `error` when the Model API failed; the stage goes on either way, and
`stage-2.json` holds the same record under `details.part_masks`. A replay miss stops the stage. The record is
its own file, not a key in `views/meta.json`, because `views/meta.json` is one of the files whose hash ties the
held-out views to the turnaround (`heldout.IDENTITY_FILES`): writing the part masks changes none of them.
`python3 -m herogen.pipeline parts --run-id <id> [--live]` adds the part masks to a run made before stage 2
wrote them, and writes no other file, so its accepted held-out views stay accepted. No stage reads the part
masks in this version; stages 3 and 4 build exactly as without them.

**Recorded, not gated.** `a_pose_hand_span_over_shoulders` (widest row at 35-60% of the height over the widest
at 15-25%) is still measured on every sheet, with `gate: false` and a note. Its fixed height bands read hat
brims, big chibi heads, shoulder armour and T-posed arms as shoulders: it failed 10 of 21 true turnarounds of
real game characters and 30 of 105 generated sheets three judges called usable, and next to the carve check it
stopped no sheet they called unusable. `right_profile_width_over_height` and `left_profile_width_over_height`
(profile box width over height, at most 0.33) are recorded the same way: they measure body shape, not pose,
so round bellies, robes, hat brims, chibi heads, wings, tails and backpacks exceed them. They failed 10 of the 21
true turnarounds and 67 of the 105 usable sheets, 16 of which also carve, and stopped no unusable sheet the
carve check let through.

## Stage 3: mesh source and joints

`python3 -m herogen.blender run --stages 3 [--mesh-source carve|sam3d-objects] [--joints silhouette|sam3d-body]`.

- `carve`: the visual hull of the observed masks, surfaced and smoothed. Its shape is built from every
  observed view.
- `sam3d-objects` (default, with `sam3d-body` joints): SAM 3D Objects reconstructs a textured mesh from `views/front.png` and `masks/front.png` only;
  it is kept as `03-source.glb`, put in the contract frame and remeshed into one closed manifold surface
  (`herogen/blender/sam3d_mesh.py`). The front view decides the facing, a small yaw / roll and perspective fit
  and an outline conform onto the front mask; the pitch, a tilt the front view cannot see, is registered on the
  observed left and right masks before conform and remeshing. Depth is registered after remeshing. Its shape
  is built from the front view alone.

`stage-3.json` records `mesh_source` (what was used), `mesh_source_requested`, `joints_source` and `shape_views`,
the views the shape was built from, and `pose_views`, the views the hybrid's pitch was registered on. **Every
observed view's silhouette IoU is gated, whatever the mesh source**, in stage 3 and in stage 4's rest IoU; a view
the shape was not built from carries a note saying so, and says when the pitch was registered on it. Only a
derived view (`mask_method` `visual_hull`, made by stage 2 from the others) is reported, not gated. The texture
gates apply to every observed colour view.

**The hybrid's pitch is registered on the side views.** The front silhouette barely reacts to a tilt about X, so a
fit on it alone drifted towards its limit: on one generated turnaround it chose -18 degrees (front IoU 0.875 at
pitch 0, 0.896 at -18, before the conform), the conform pulled the front outline back, and the side views showed
the body tipped over (left 0.186, right 0.185). The orientation now applies the facing and a small yaw about Z
first, then the pitch about X (the body's side-to-side axis) and the roll about Y (the front view's axis), so both
mean the same tilt of the body at every facing. The front fit (yaw, roll, perspective correction) runs with the
pitch held at 0. The pitch is then swept from -30 to 30 degrees in 2 degree steps, scored by the mean
half-resolution IoU of the left and right masks; every local peak within 0.03 of the best is refined to 1 degree
and the best is kept if it lies within +-20 degrees. The front fit is then redone with that pitch, before the
conform and the remesh. The pitch stays 0, with the reason recorded, unless both left and right are observed
(present, non-empty and not derived; back, top and the held-out views are never used), when the sweep is flat (the
side IoU varies by less than 0.02), when it is ambiguous (another peak more than 8 degrees away within 0.01 of the
best) and when the best lies beyond +-20 degrees. No Y offset is fitted or applied during pitch registration:
the frame stays centred on the bounding box until depth registration. `mesh.source.pose_fit` in `stage-3.json`
records the views used and those excluded with their
reasons, the status (`fitted` or `fallback`) and reason, `pitch_deg`, `limit_deg`, `y_offset_studs` (0) with a
note, the sweep (`sweep_half_res`) and its scores at pitch 0 and at the fitted pitch (`side_iou_half_res`: the
sweep's scores, not the final mesh's); `mesh.source.fit` holds the pitch with the rest of the fit, and
`pose_views` lists the views the pitch was registered on (none after a fallback). The shape is still built from
the front view alone (`shape_views` stays `["front"]`).

**Measured pitch registration.** NumPy replay of orientation, conform and remesh on 34 diagnosis records
(32 distinct sources) improved median worst-side IoU from 0.6663 to 0.7378; median front/back/left/right IoU
changed from 0.9688/0.8702/0.6744/0.6708 to 0.9698/0.8688/0.7436/0.7392. On 34 available of 36 listed scorecard
turnarounds, worst-side median improved from 0.6704 to 0.7356 and worst-observed median from 0.6572 to 0.7056;
two turnarounds lacked completed inputs. Every-observed-view silhouette passes stayed 0/34 in each set. These
are observed-mask agreement measurements, not held-out generalization, and the overlapping sets are not pooled.
Front minimum on the diagnosis fell from 0.9097 to 0.7864; the remesh can lose front detail after the conform.

**Depth is registered after pitch and remeshing.** Each view's mask is placed by its own bounding box, so a
mesh centred on its own Y extent can land off the side masks when a tail or pack sets different extents. After
the pitch fit, front refit, conform and remesh, the mesh and the source vertices used by the texture fallback
move together along Y. The whole-pixel offset maximises mean silhouette IoU on observed left/right masks and
the top mask only when drawn, never derived or held out; shape and pitch are unchanged. `stage-3.json` records
both `mesh.source.pose_fit` and `mesh.source.depth_register`, the latter holding offset in studs and pixels,
views, and IoU before and after. Scoring counts the complete silhouette, including pixels outside the
observation frame, so clipping cannot improve the objective. The chosen offset is re-rendered in an expanded
frame; the record reports that verified IoU.

**Measured pitch fit plus depth registration.** A paired NumPy replay on all 33 offload and 33 scorecard
source cases uses the actual side-mask pitch fit, front refit, conform and remesh, then compares the same mesh
with and without depth registration. The two sources missing from the local cache were recovered from the
stored scorecard runs and matched their original hashes. Median IoUs are:

| Set | Variant | Front | Back | Left | Right | Top | Worst observed | Silhouette/open-edge passes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Offload (33) | pitch fit | 0.96910 | 0.86860 | 0.74350 | 0.73730 | 0.58150 | 0.73330 | 0/33 |
| Offload (33) | pitch fit + depth | 0.96910 | 0.86860 | 0.76550 | 0.75300 | 0.57145 | 0.74520 | 0/33 |
| Scorecard (33) | pitch fit | 0.96180 | 0.82170 | 0.74680 | 0.75170 | 0.56055 | 0.71250 | 0/33 |
| Scorecard (33) | pitch fit + depth | 0.96180 | 0.82170 | 0.76420 | 0.76140 | 0.55400 | 0.71530 | 0/33 |
| All cases (66) | pitch fit | 0.96555 | 0.83505 | 0.74360 | 0.73925 | 0.57245 | 0.72545 | 0/66 |
| All cases (66) | pitch fit + depth | 0.96555 | 0.83505 | 0.76435 | 0.76045 | 0.55950 | 0.74090 | 0/66 |

Top medians include derived views (28 offload and 20 scorecard top masks, only two observed, both offload);
only observed masks enter the worst-view score and the unchanged >=0.90 silhouette gate. The benchmark also
requires zero open edges, met by every mesh; it does not establish full stage-3 pipeline success. Its final IoUs
use the original finite observation cameras, while `depth_register` separately records complete-silhouette IoU.
The 66 cases contain 65 distinct source meshes, with one shared across sets, and are not independent held-out
samples. Worst-view agreement improves on 50 cases, is equal on 15 and decreases on one at four-decimal input
precision (scorecard s016, 0.6566 to 0.6565); 19 improve by more than 0.02, and cases below 0.5 fall from 9 to 6.
Individual views can still lose IoU, as the top medians show. These measurements replace the earlier pitch-zero
proxy for the combined implementation.

**Causes that remain.** The pitch fit still happens in the bounding-box-centred frame, so missing or misplaced
tails/packs can bias its pitch before the later depth translation. Translation cannot correct shape: on
turnarounds rendered from exact model masks the reconstructed side views stay 0.51-0.73 whatever the pitch,
the figure too thin front to back. The remesh can drop thin parts and front detail (one source: front 0.952
after conform, 0.794 after remeshing), so the front can move with small changes of pitch.
Stage 4 can still be run on such a mesh with `--force`; its rest IoU gates fail the same way. A failed hybrid build (including either unavailable SAM 3D source or Blender export failure) retries once
with `carve` and `silhouette`. `fallback` records the constructor transition and reason,
`mesh_source_requested` and `joints_source_requested` preserve the request, and the notes and CLI say so.
Both attempts share cost accounting. A completed quality rejection never triggers this retry; `--no-fallback` stops the stage instead, with a one-line error.
Every failure of a remote source counts: the call itself, its cached output (an unreadable GLB, a missing file)
and mapping its result (a mesh that covers no pixel of the front view, say). Stage 3 removes its previous
`stage-3.json` when it starts and writes each output only once it is complete, so a stage that stops never
leaves an earlier passed record or a half-written file for stage 4.

Stage 3's gates: the silhouette IoU (at least 0.90) of every observed view, `open_edges` (0: the
mesh is closed) and, by mesh source, `independent_views` for the carve (at least two of the front-or-back pair,
the left-or-right pair and the top) or `front_view_present` for `sam3d-objects` (the one view it is built from).
`triangles` and `non_manifold_edges` (edges shared by more than two triangles) are reported, not gated.

`03-joints.json` holds the joints stage 4 cuts at:

```json
{
  "source": "sam3d-body",
  "units": "studs",
  "frame": "contract: +Z up, feet at z=0, faces -Y, x/y centred on the figure's bounding box",
  "joints": {"Neck": [0.0, 0.01, 4.6], "LeftShoulder": [0.6, 0.0, 4.2]},
  "evidence": {"Neck": "MHR skeleton joint 113 projected; X/Z through the front view mapping; Y mesh mid-depth"},
  "joint_sources": {"Neck": "sam3d-body", "LeftShoulder": "silhouette (X/Z), mesh (Y)"},
  "sources": {"sam3d-body": {}, "sam3d-body-y-mhr3d": {}, "silhouette": {}},
  "distance_to_silhouette": {"Neck": 0.05},
  "flag_threshold_studs": 0.5,
  "flagged": [],
  "y_method": "mesh", "limb_roots": "silhouette", "remote": {"cached": true, "gpu": "A10G",
    "person_box": [112, 20, 288, 380], "box_source": "front mask bounding box (stage 2)", "detector": null}
}
```

With `--joints sam3d-body`, SAM 3D Body's keypoints on the front view give X and Z through the front view's own
mapping and the mesh's mid-depth gives Y (`herogen/blender/body_joints.py`). Shoulders and hips (`limb_roots`)
are cut where the outline separates the limb, from the silhouette finder, since the model's anatomical
shoulder sits inside bulky costume pieces; each joint's source is in `joint_sources`, both finders' joints are
kept under `sources`, and joints more than 0.5 studs apart are listed in `flagged`. With `--joints silhouette`
the file holds the silhouette finder's joints only (`"source": "silhouette"`), and stage 4 finds them again
itself.

## Remote models (SAM 3D Objects, SAM 3D Body)

`herogen/blender/remote/objects.py` and `body.py` are self-contained Modal apps (`herogen-sam3d-objects`,
`herogen-sam3d-body`); `client.py` is the one client the stages use. Each call's outputs are recorded under
`<runs>/_cache-remote/<key>/` (the model's output files, `entry.json` with the request, the run log, wall seconds
and a cost estimate, and `modal.log`, scrubbed of tokens, the Modal workspace name and the home directory).
The key hashes the runner and the SHA-256 of its source file (`runner_sha`), the model code commit on GitHub
(`rev`), the Hugging Face weight revisions the runner pins (`weights`: `facebook/sam-3d-objects` and its depth
model `Ruicheng/moge-vitl`; `facebook/sam-3d-body-dinov3` and its field-of-view model
`Ruicheng/moge-2-vitl-normal`), MoGe's code commit for SAM 3D Body (`moge`) and the commit of its DINOv3 backbone
code (`dinov3`; upstream would fetch that code from GitHub's main branch at run time, the image clones it at the
pinned commit and loads it from there), the renderer, the parameters, SAM 3D Body's person box and its source and
the optional detector (`bbox`, `box_source`, `detector`) and the SHA-256 of every input image, not the GPU type (a
CPU run of SAM 3D Body keys separately). A cached entry made by a different runner (another source file, commit or
weights) is not used; replay names the fields that differ. The runners refuse a checkpoint volume holding another
revision. Errors are one line; a failed call keeps the scrubbed log tail on the exception, not in the records. A
GPU with no recorded list price gets an unknown (`null`) cost, not a guess. `--remote replay` (the default) reads
the cache
only; `--remote live` runs `modal run` in a subprocess with a timeout and records the result. Configuration is by
environment, set by the client: `HEROGEN_OBJECTS_GPU`, `HEROGEN_BODY_GPU`, `HEROGEN_OBJECTS_TIMEOUT`,
`HEROGEN_BODY_TIMEOUT`, and the checkpoint volumes `HEROGEN_OBJECTS_VOLUME` (default `herogen-sam3d-objects-ckpt`)
and `HEROGEN_BODY_VOLUME` (default `herogen-sam3d-body-ckpt`). The Hugging Face token lives only in the Modal
secret `huggingface`. Stage records count a replayed call as no new spend. The first live SAM 3D Objects call
builds its image, which compiles CUDA extensions (13 minutes measured for a rebuild from the requirements
step on, longer from scratch); build it ahead with `modal run herogen/blender/remote/objects.py::build` (CPU
only; it also reports that none of the excluded packages can be imported and runs the pytorch3d rasteriser's
self-test), or give that first call a longer `--remote-timeout`. The SAM 3D Body image's build also fails if a
package it leaves out can be imported. SAM 3D Body's person box is the bounding box of stage 2's front mask, and
by default no person detector runs; `--body-detector vitdet` runs upstream's ViTDet detector (detectron2,
weights under CC BY-SA 3.0, downloaded on first use) instead, with the mask's box used only if it finds nobody.
The detector is part of the cache key, and `03-joints.json` records it (`remote.detector`) with the box and its
source (`remote.person_box`, `remote.box_source`) and the runner's code commits (`remote.code`: `rev`, `moge`,
`dinov3`).

## Scoring: `05-scores.json`

`python3 -m herogen.blender.score --run <folder>` (or `python3 -m herogen.blender score`) scores bodies with
numpy only, so any GLB can be compared with ours on the same views (`herogen/blender/score.py`; GLBs are read by
`herogen/blender/glb.py`, the one GLB reader).

- **Bodies.** Any number of `--body NAME=GLB`, from anywhere (frozen builds of earlier milestones, other
  tools' outputs; outside the run only the file name is recorded). A `version.json` next to a GLB is recorded
  under the body's `"version"`: its `label` and plain fields (`frozen`, `sketch`, `profile`, `note`, and
  `code`: `{"commit", "dirty"}`). With no `--body`, every stage 4 body: `04-body-carve/high.glb` as `carve`, `04-body/high.glb` as
  `carve` or `hybrid` by its mesh source, `04-body-<x>/high.glb` as `<x>`, and each `atlas-high-<variant>.png`
  next to one as `<name>-<variant>`, the same mesh with that atlas (`variant_of`). `--body NAME=GLB` scores any
  file, `--atlas NAME=PNG` gives it another atlas.
- **Views.** The four turnaround views at their contract yaw, and the held-out views `stage-2h.json` admits
  (stage 2h makes `front-left-30`, `front-right-30`, `front-left-60` and `front-right-60`; any other name is
  scored the same way), each image with its mask: see Held-out views for the rule. Files in subfolders, such
  as `review/`, `rejected/` or `archive/`, are never scored.
- **Placement.** Turnaround views use stage 3's mapping: one pixel scale (the front mask's height over the
  body's height), u = 0 on each mask's box centre, feet on its bottom edge. Stage 4 bodies are placed as built
  (`"frame": "contract"`), so a stage 4 body is scored with the same mapping and metric as stage 4's gates;
  other GLBs by their own bounding box (`"frame": "fit"`). A held-out view's yaw is searched (2 degrees, then
  0.5 around the best) 25 degrees each side of the target its name gives (`front-left-30`: 5 to 55;
  `front-right-60`: -85 to -35), except `front-left-34` (the single three-quarter view an earlier stage 2h wrote,
  asked for at 45 degrees): 15 to 75; other names all the way round. Scale and offset are fitted from the mask's
  box, and the view is also measured at the target itself (`at_fixed_yaw`; 45 degrees for `front-left-34`). Each
  held-out row carries `reference` and `measures` from `stage-2h.json` (`measures` is `"consistency"` when the
  record says `"reference": "generated"` and `"accuracy"` for `"ground-truth"`, otherwise both are `null`) and stage 2h's own `status` and
  `found_yaw_deg` for the view, when it records them.
- **Measures, per view.** Silhouette IoU; colour error (mean absolute, 0-255) and grey SSIM between a flat render
  of the body's own colours (base-colour texture, factor and vertex colours; nearest surface, no lighting) and
  the view's image, inside both silhouettes as stage 4's texture gates measure them; `seen_shape` (the shape was
  built from the view), `seen_colour` (the colours were) and `seen` (either), from `sections.json`
  (`shape_views`, `texture.colour_views`) or `--seen` / `--seen-shape`. A held-out view is never seen. A body's
  shape, a ground-truth body's included, is its alpha-clipped surface: faces of an `alphaMode: MASK` material
  count, in the silhouette and the colour alike, only where alpha reaches `alphaCutoff`, and the body is placed
  (the turnaround registration and the held-out fit) by that surface's extent, so a transparent card never
  shifts or shrinks it; OPAQUE materials are never clipped (BLEND is scored as opaque).
- **Per body.** Triangles (and the low LOD's), atlas size, mesh sanity after welding (degenerate triangles,
  boundary, non-manifold and inconsistently wound edges, components, inside-out closed meshes, the meshes that
  are not closed; self-intersections are not checked), and the checks of a profile when one is asked for.

```json
{
  "scored": "2026-10-29T10:00:00-07:00",
  "scorer": "herogen.blender.score",
  "code": {"commit": "<git HEAD sha>", "dirty": false},
  "method": {"iou": "...", "colour": "...", "heldout_yaw": "..."},
  "views": [{"view": "front", "mask": "masks/front.png", "image": "views/front.png", "heldout": false,
             "yaw_deg": 0.0},
            {"view": "front-left-34", "mask": "heldout/front-left-34-mask.png", "heldout": true,
             "image": "heldout/front-left-34.png", "yaw_range": [15.0, 75.0], "fixed_yaw": 45.0}],
  "bodies": [{
    "name": "hybrid", "glb": "04-body/high.glb", "mesh_source": "sam3d-objects", "joints_source": "sam3d-body",
    "triangles": 22768, "triangles_low": 5778, "budget": null, "within_budget": null, "atlas": 1024,
    "colour_source": "texture", "frame": "contract", "pixels_per_unit": 168.5, "height": 5.5,
    "shape_views": ["front"], "colour_views": ["back", "front", "left", "right"],
    "sanity": {"watertight": true, "boundary_edges": 0, "open_meshes": []},
    "views": {
      "left": {"iou": 0.70, "colour_error": 14.0, "ssim": 0.62, "pixels": 89000, "yaw_deg": 90.0,
               "seen": true, "seen_shape": false, "seen_colour": true},
      "front-left-34": {"iou": 0.63, "colour_error": 39.7, "ssim": 0.14, "pixels": 122000, "yaw_deg": 21.5,
                        "seen": false, "seen_shape": false, "seen_colour": false, "scale": 0.98,
                        "yaw_range": [15.0, 75.0], "yaw_at_range_edge": false, "yaw_search": [[15.0, 0.626]],
                        "at_fixed_yaw": {"45": {"iou": 0.55, "colour_error": 41.5, "ssim": 0.13}},
                        "reference": null, "measures": null}
    },
    "notes": [],
    "version": null
  }]
}
```

`05-scores.json` also records the scorer's own `"code"` version. Scoring again replaces bodies of the same name
and keeps the others. The results page reads `name`, `glb`, `triangles`, `atlas` and, per view, `iou`,
`colour_error`, `ssim`, `yaw_deg`, `scale` and `seen`; the lab also reads `variant_of`, `mesh_source`,
`joints_source` and `triangles_low`; every other field is additional.

## Target profiles

The default target is the best-quality body, with no triangle budget; Roblox's budgets are opt-in. Stage 4
builds to one profile (`python3 -m herogen.blender run --stages 4 --profile ...`, `herogen.blender.PROFILES`)
and the scorer checks the same names (`--profile`):

| Profile | Stage 4 builds | Budget gates |
| --- | --- | --- |
| `best` (default) | `high.glb` from the stage 3 mesh undecimated (full detail); no low LOD | none: `triangles_high` is reported, not gated |
| `roblox-ingame` | `high.glb` and `low.glb`, each the largest whole-body decimation whose sectioned total fits | `triangles_high` <= 24,000, `triangles_low` <= 6,000 |
| `roblox-avatar` | one `high.glb` (Roblox makes its own LODs) that meets the triangle, watertight-part and texture limits of Roblox's character body specification; each part group takes, from a ladder of whole-body decimations, the densest candidate that fits its own budget (sections are separate closed meshes, so groups may come from different candidates), with the low cap and cover settings | `triangles_total` <= 10,742; per group `triangles_Head` <= 4,000, `triangles_Torso` (`UpperTorso`, `LowerTorso`) <= 1,750, each arm and leg (three sections) <= 1,248; `watertight_parts` (every part closed and manifold after welding); `texture_max_px` <= 2048 |

Budget gates apply only under a Roblox profile; every other gate (sections, open edges, watertight parts, pivots, joint gaps,
rest IoU, texture) is the same under every profile, applied to the LODs the profile builds. The rest of the
specification is not built, so a `roblox-avatar` body is not yet a complete avatar body: the R15 bone rig with
skinning, attachments, facial animation (DynamicHead) and the inner and outer cages for layered clothing; the
records list them (`not_built`).
`stage-4.json` records `"profile": {"name", "lods", "budgets"}`; `sections.json` records `"profile"` and
`"budgets"` (`null` under `best`; `{"triangles_high_max": 24000, "triangles_low_max": 6000}` under
`roblox-ingame`; the total, the groups, the texture size, `watertight_parts` and `not_built` under
`roblox-avatar`), and `"lods"` holds only the LODs built (`triangles_low` of each section is `null` without a
low LOD). A LOD the profile does not build is removed from `04-body/` if an earlier run left one.

The scorer applies no budget unless asked: `--profile best` adds no checks, `--profile roblox-ingame` checks
the body's triangles (and its low LOD's, when a `low.glb` sits next to it) against 24,000 and 6,000, and
`--profile roblox-avatar` checks the specification's triangle, watertight-part and texture limits, finding part
groups by mesh name (a group with no mesh of its names is reported with value `null`); what the profile does not
build is listed (`not_built`, `"required": false`), not checked. A profile's `"pass"` is `true` (`"status":
"pass"`) only when every required check was measured and met, `false` (`"fail"`) when one fails and `null`
(`"incomplete"`) when one could not be measured: no `low.glb` next to the body under `roblox-ingame`, meshes not
named after the sections or no texture under `roblox-avatar`. `--budget NAME=N` sets a plain total for one body.

## Held-out views, mask method and covers

**Held-out views (stage 2h).** After stage 2 passes, four extra images of the same character in the
turnaround's A-pose, each its own edit of the cover: `front-left-30`, `front-right-30`, `front-left-60` and
`front-right-60`, named by their target yaw (front-left views face image left). No building stage reads
`heldout/`; the views exist so a body can be scored on views it was not built from. Each generation sends the
cover, a pose reference (the front view, or the nearer profile for turns of 50 degrees or more) and a grey
mannequin of the turnaround's pose drawn at the requested yaw, both at one shared scale. The mannequin takes
only landmark positions from the turnaround: the torso is the run holding the front figure's centre in each
torso row (or, in a row parted at the centre, the span of the nearest runs on either side). The legs stand on
the detected feet, since a silhouette alone can't tell a leg from a tail, a staff or a cape.
Once per run, SAM 3.1 is asked for `foot`, then `shoe`, then `boot` on the turnaround's front view,
stopping at the first prompt that finds two feet. A foot is an instance with half its pixels or more on the
figure and pixels on its sole row (0.98 of the height); overlapping instances count once. Each foot's leg is
the sole-row run it overlaps most, clipped to the detected foot's own columns. That clipped span sets both
the ankle position and the starting columns for tracing up to the knee row (0.71). On each row the trace
keeps, of the runs overlapping the last one, the one overlapping its starting columns most, then the last
run most. The knee run is traced the same way to the hip row (0.52), using the knee's columns. Legs on one
run at the knee row split it at its middle. The hips span the two traced legs; legs that join below the hip
row give the run they join.

Foot columns keep a wider prop touching a foot at the sole row from winning merely because the merged
sole run contains more of the prop than the leg. This is still an outline heuristic: incorrect foot
segmentation or a prop overlapping the trace's starting columns can misplace a leg. A shield covering the
knees gives a shared run split at its middle; something still touching a leg in the hip row widens the hips.
When no prompt finds two usable feet on the sole row (a long robe can hide them), the guide uses the outline
rule: the outermost runs of the knee and sole rows, and for the hips the hip row's run holding the figure's
centre (the span of the nearest runs either side when the row is parted there). `stage-2h.json` records which
rule drew the guide (`details.guide_legs.method`: `sam-3.1:<prompt>` or `outline`), the legs' runs and what
each prompt found in `details.guide_legs.feet`; an outline fallback records why in `feet.fallback_reason`. A
SAM failure on the feet stops the stage like any other. A replay of a run recorded before feet were asked
for misses at the first foot prompt.

The prompt asks for the figure at the mannequin's size and place, so a generation's pixels map to the views'
pixels by one over the shared framing scale (times the input width over the generated width). The generated
figure is placed at that scale on the square canvas of `views/`, centred, its feet on the front view's
baseline. A figure drawn taller or wider than the canvas allows is neither cut nor shrunk: the square is padded
by the same number of pixels on every side, so it stays square and centred and the views' pixel (x, y) sits at
(x + pad, y + pad). Each attempt records the pad as `canvas_pad_px`, 0 when the figure fits. Its own height is
never used to scale it, so `height_agreement_front` (at least 0.95 of the front view's height, both ways)
measures whether the model drew the figure at the guide's size.

Each view is checked against the turnaround: silhouette landmarks are placed in 3D from the front and profile
views and fitted over yaw. The fit lets the arms be held a few degrees off the A-pose (each hand moves along
its arm's tangent to the height it is seen at), so arm drift doesn't read as a turn; the pose error is measured
at the found yaw against the turnaround's own arms. Facing is three votes (the fit, the toes, SAM 3.1 face
inside head). The stage writes into `stage-2h.json` the building azimuths a body may be built or coloured from
(`build_azimuths_deg`: 0 to 315 in 45-degree steps, the four turnaround views and the diagonals between them).
A view is judged on a yaw range: from the drift-robust fit's yaw to the plain fit's (the hands where the
turnaround has them), widened by 2 degrees on each side, the larger bound of `fit_error_deg` (found minus true
yaw, -2 to +1 degrees, measured with arms drifted up to 8 degrees on synthetic figures). The whole range must be
at least `azimuth_clearance_deg` (10) from every building azimuth. That this keeps the true yaw 10 degrees clear
holds on synthetic outlines; it is not yet measured on real images. The yaw windows are 25-33 degrees for the
30-degree views and 57-65 for the 60-degree views (the same 10 plus 2 degrees from every building azimuth); up
to 5 degrees outside needs human review.

Known limitation, measured on synthetic outlines: when a figure's arms are raised 3-8 degrees above the
turnaround's and both of its hands are also measured 0.01-0.02 of the figure height too low, both fits read
the turn too small. A 30-degree view drawn 35.5-37 degrees from the front can then pass with its true yaw
8-9.5 degrees from the 45-degree diagonal instead of 10 (24 of 980 near-diagonal test views; none at 38 or
beyond, none among the 60-degree views). It stays until the fit's error is measured on real images with known
yaws.

A view is generated up to four times, stopping at the first accepted generation. The first request is the
target; each next one is the last request plus the target's |yaw| minus the last generation's found |yaw| (more
after an under-turn, less after an overshoot), kept within 5-85 degrees, with the nearer profile as the pose
reference once it reaches 50. After a generation whose facing checks said the other way, or that had no figure
or turnaround landmarks to measure, the same request is sent again as a new sample. The generation kept is the
best by status (accepted, else needs human review because a check could not decide, else rejected because a
gated check failed), then the smaller window miss, then the lower pose error. Its image, mask and raw generation go to `heldout/<view>.png`, `-mask.png` and `-raw.png` when
accepted, and to `heldout/review/` or `heldout/rejected/` otherwise.

A person decides views with
`python3 -m herogen.pipeline review --run-id <id> --view <view> --accept|--reject [--note TEXT] [--by NAME]`,
which moves the view's three files (all or none) and appends the decision (who, when, decision, note, image
hash) to `details.decisions`. A view whose gated checks failed can be rejected but not accepted, and a view
judged under other checks must be judged again first with `recheck --run-id <id>` (no model calls).
`run --stages 2h --views <v1,v2>` remakes only those views and keeps the others with their decisions: each kept
view must come from the current turnaround (else the run stops and says to remake every view), and is judged
again if the checks changed since. The files of a remade view that a person decided on move to
`heldout/archive/`, never deleted. The gate passes when every view is accepted.

`stage-2h.json` also holds each view's status, the checks' own verdict (`machine_status`), found yaw and every
attempt with its request, inputs and yaw ranges (`details.views`); the turnaround it was made from
(`stage2_identity`, a hash of the cover, views and masks), the checks it was judged under (`checks_version`, a
hash of every constant and judging function, so any change to them is noticed) and its image's SHA-256; and
`"reference": "generated"`: a score against a generated view measures consistency with the generated
character, not accuracy. A scorer reads the accepted views from the record (`heldout.accepted_views`: status
accepted, the current turnaround and a matching image hash), never from the folder layout. New files are
written to `heldout/.staging/` and swapped in only when the stage finishes; until then the earlier record is
kept, marked as running. If the stage stops early (an error, a replay miss or the 30-minute time limit) the
earlier record, calls and files stay, the stopped run is added to `stopped_runs` with its error, calls and
cost, and `cost_usd` counts both (`run_cost_usd` is the run that made the views). When a run folder holds
`05-scores.json`, the results page shows its per-body table with a highlighted row, and an image pair beside
the body's render, for each held-out view the record accepts; other views the scores file names are listed as
not accepted (see `herogen/pipeline/results.py` for the format).

The blender scorer (`herogen.blender.score`) takes its held-out views from the record, by its form
(`heldout_sources`; the formats the lab reads too):
- a per-view record (`details.views` with statuses): the views `heldout.accepted_views` returns (status
  accepted, made from the current turnaround, image still matching its recorded SHA-256), each with the mask
  its `files` name (else `heldout/<name>-mask.png`);
- the older one-view record (`details.view`): its view, when the record passed and lists the image;
- a bench record (`"reference": "ground-truth"`, views rendered from a known body, no per-view status): every
  view it lists whose image and mask exist;
- no `stage-2h.json`: every `heldout/<name>.png` directly in `heldout/` with a matching `-mask.png`;
- a record that cannot be read, that lists no view or whose stage is still running: no held-out view.

`05-scores.json` names the form under `heldout_source` (`source`, the `rule` and `skipped`: each view file in
`heldout/` that was not admitted and why, such as "made from an earlier turnaround" or "its image changed since
stage 2h recorded it"), and each held-out entry of `views` carries its `source`. A name `front-left-<deg>` or
`front-right-<deg>` gives the target yaw the scorer searches around, and each row carries the view's `status`
from the record. A held-out row carries `reference` and `measures` from `stage-2h.json`: when the record says
`"reference": "generated"`, the row is labelled `"measures": "consistency"`, since a generated view can invent
detail and a score against it measures consistency with the generated character, not accuracy; a
`"reference": "ground-truth"` view, a render of a known body, is labelled `"measures": "accuracy"`.

**Mask method.** Every figure mask is SAM 3.1's: the first of the prompts `character` and `person` that finds
something, recorded per mask as `sam-3.1:<prompt>` (`sam-3.1:none` when neither finds anything, an empty mask).
No mask is ever made another way, and nothing is measured on a mask SAM did not make:

- A SAM or Muse Image call that fails after the client's retries stops the stage (record status `"error"`,
  with the error and every call made). So does an answer whose `status` is not `completed` (or that carries an
  `error`): an incomplete SAM answer has no output, which would otherwise read as "nothing found". The
  responses endpoint always sends a status; Muse Image sends none, so only a status it does send is checked.
  Such an answer is logged as a call (it may be billed) but never cached. The one exception is the held-out face cue: a failed `face` or `head`
  call sends that view to human review, so it can only keep the view from being accepted.
- No character found in the sketch rejects the input: stage 1 segments the sketch before it asks for the
  cover, so a sketch SAM finds nothing in costs no edit, and its record has status `"input_rejected"`, the
  error "no character found in the sketch" and `sketch_characters_found` 0 (see Input checks). Nothing found in a generated image fails that image's gate: the
  cover's `cover_mask_area_frac`, a turnaround sheet's `figures_found` (it is generated again), a held-out
  generation's `figure_mask_sam_coverage` (rejected, measured no further, generated again). Nothing found in
  the drawn top view means it is not used, so the top view is the visual hull.
- A turnaround sheet is split into its four figures on the SAM mask only; when SAM's columns don't give four
  figures, `figures_found` fails. The background-threshold mask is only a reference: the gated
  `view_mask_sam_coverage_min` (stage 2) and `figure_mask_sam_coverage` (stage 2h) checks require SAM's mask to
  cover at least half (`SAM_COVERAGE_MIN`, 0.5) of the threshold area of each figure, so a figure SAM half
  missed fails its gate instead of being measured.
- The client never caches a failed call, so a replay never repeats a failure: a call that failed live has no
  cache entry, and its replay stops with `ReplayMiss`, as does an entry for a failed call that an earlier
  version cached. A replay miss, a corrupt cache entry or any other error stops the stage.

`views/meta.json` records `"mask_method": {"front": "sam-3.1", ...}` (`"sam-3.1"` or `"visual_hull"`, the top
silhouette implied by the four side views' SAM masks, whose top-view checks gate), the detailed method per view
and `"top_method"`. Records written by earlier versions can hold `"threshold"` masks, from a fallback that no
longer exists; such a record is not a measurement and never counts as passed, whatever its `ok`: the results
page shows it as "degraded: threshold masks", stage 2h refuses to start from such a stage 2, `herogen.blender`
refuses to build on it (without `--force`), and a held-out view measured on one is never an accepted view.

**Shoulder covers.** Stage 4 adds one cover per shoulder, merged into its owner section (`UpperTorso`) so each
GLB keeps exactly 15 meshes. `sections.json` lists them under `"covers"`: the owner, the shaping parameters, and
`pieces_high` / `pieces_low` with each piece's triangles. Covers count in the triangle budgets (in the Torso
group under `roblox-avatar`).

## Versions

A version is one stage 4 body frozen with the records that built and measured it, so a later build can be
compared with it on the same sketch. `herogen/versions` writes and reads them:

```bash
python3 -m herogen.versions freeze --run runs/my-hero --label m1-change [--body 04-body] [--sketch S] \
    [--milestone M] [--note TEXT] [--by NAME] [--allow-dirty] [--allow-unknown-code] [--store DIR]
python3 -m herogen.versions list [--sketch S] [--json] [--store DIR]
python3 -m herogen.versions verify --version V [--json] [--store DIR]
python3 -m herogen.versions compare A B [--json] [--store DIR]
```

The store is `--store` or `$HEROGEN_VERSIONS`. There is no default, and the folder must already exist. A version
(`V`, `A`, `B`) is given as a version folder, as `<sketch>/<label>` in the store, or as a label that only one
sketch folder holds. A folder whose name starts with `.` (such as an unfinished `.partial-*` freeze) is never
a version and is refused however it is given. Exit status: 0 done, 1 `verify` found a problem, 2 refused or
failed. A refusal or failure prints one line on stderr and never a traceback.

**Layout.** Versions are never kept in git.

```
<store>/<sketch>/<label>/                      one version: flat files plus version.json, read-only
<store>/<sketch>/05-scores-<series>.json       scores collected by hand for format 0 versions (see compare)
<store>/<sketch>/.partial-<label>-<random>/    an unfinished freeze: not a version, safe to delete
<store>/<sketch>/<label>/ (empty)              an empty claim from an interrupted freeze: safe to remove
```

Sketch folders and version folders in the store are never symbolic links: a freeze refuses to write through
one, `<sketch>/<label>` and label references refuse to read through one, and `list` names one as not read.

Labels and sketch folder names are one plain name: `[A-Za-z0-9][A-Za-z0-9_.-]{0,79}`. File names inside a
version are plain names too (`[A-Za-z0-9][A-Za-z0-9_.-]{0,119}`), so none is a folder, `.`, `..` or an absolute
path. A freeze refuses to write a `version.json` in which any key or string value names a path. Text is split
into words at white space and `=`, with opening quotes and brackets stripped. A word names a path when it starts
at a root (`/` followed by a name, `~` followed by anything, `\\` or a drive such as `C:\`) or has a `..`
step. Text also names a path when it holds a home prefix (`/Users/`, `/home/` or `\Users\`) anywhere. A lone
`/` between words ("front / back") and a slash inside a word ("24000/6000") are plain text. `--note`,
`--milestone` and `--by` are checked the same way before anything is copied.

**What a version holds.** Every file is copied byte for byte. Nothing in a run is changed.

| Files | Taken from |
| --- | --- |
| `high.glb` (required), `low.glb`, `sections.json`, every `atlas*.png` | the body folder `<run>/<body>/` (`--body`, default `04-body`; any `04-body-<name>`) |
| `stage-0.json`, `stage-1.json`, `stage-2.json`, `stage-2h.json` (the held-out decisions), `stage-3.json`, `stage-4.json`, `stage-5.json`, `03-joints.json`, `03-source.glb`, `05-scores.json` | the body folder when it holds its own copy, else the run folder |

A body folder with its own `stage-3.json` (such as `04-body-carve/`) had its own stage 3, so the run folder's
`03-joints.json` and `03-source.glb`, which belong to the run's other body, are never taken for it. The sketch,
views, masks and held-out images are not copied. Their SHA-256 is recorded under `inputs` instead. Renders,
pose renders and `03-mesh.glb` are not part of a version. Only regular files are read: a symbolic link anywhere
in what is copied or hashed (the file or a folder on its way, such as `04-body/`, `views/` or `heldout/`) is
refused.

**`version.json`, format 1.**

| Field | Value |
| --- | --- |
| `format` | `1` |
| `label`, `sketch` | the version's folder and its sketch folder |
| `milestone`, `note` | from `--milestone` and `--note`, else null |
| `frozen` | when it was frozen, ISO 8601 with seconds and UTC offset |
| `frozen_by` | `--by`, else the login name, else null |
| `built` | `stage-4.json` `started` |
| `profile` | `stage-4.json` `profile.name` (or `profile` when it is a string), else `sections.json` `profile` |
| `budgets` | `stage-4.json` `profile.budgets` when not null, else `sections.json` `budgets` |
| `mesh_source`, `joints_source` | the first set value in `sections.json`, then `stage-4.json`, then `stage-3.json` |
| `code` | `{"commit", "dirty", "consistent"}`, or null when no frozen record carries a stamp. `commit` is the first non-null `code.commit` among `stage-4.json`, `sections.json` and `stage-3.json` (the records that built the body), else null. `dirty` is true when any stamp says true, false when every stamp says false, and null otherwise. `consistent` is true when the stamps name at most one commit |
| `code_records` | each frozen JSON record's stamp, `{"commit", "dirty"}`, keyed by file name: every record whose `code` is not null, and always `stage-3.json`, `stage-4.json` and `sections.json` when frozen (null when they carry no stamp). A stamp is an object whose `commit` is a string or null and whose `dirty` is true, false or null. Any other shape is refused |
| `allow_dirty`, `allow_unknown_code` | whether `--allow-dirty` and `--allow-unknown-code` were given |
| `source` | `run`: the run folder's path relative to the store, with `/` separators, when the run lies inside the store; null otherwise, since a record never holds a path that leaves the store. `run_name`: the run folder's name. `run_records_sha256`: the SHA-256 of every `stage-*.json` in the run folder itself (not its body folders), keyed by name, which identifies the run wherever it lives. `body`: the body folder. `from_body_folder`: the run-level files taken from the body folder |
| `inputs` | SHA-256 of the run's `00-sketch.png` (`sketch`, or null). `views` and `masks`: `views/<v>.png` and `masks/<v>.png` for front, back, left, right and top. `heldout` and `heldout_masks`: each `heldout/<name>.png` that has a `heldout/<name>-mask.png`, skipping `-mask` and `-raw` images and subfolders |
| `scores` | `{"file": "05-scores.json", "body", "variants"}`, or null. `body` is the name of this body's entry in the run's scores: the one entry whose `glb` is `<body>/high.glb` and that has no `variant_of`. `variants` are the entries whose `variant_of` names it |
| `missing` | `files`: expected files the run does not have. Expected are `high.glb`, `low.glb`, `sections.json`, `atlas-high.png`, `atlas-low.png`, stages 0 to 4 and 2h, `03-joints.json`, `03-source.glb` and `05-scores.json`. The low LOD's two files are not expected when `stage-4.json` `profile.lods` leaves out `low`. `03-source.glb` is not expected when the mesh source is `carve` or unknown. `03-joints.json` and `03-source.glb` are not expected for a body folder with its own `stage-3.json`. `stage-5.json` is never expected. `fields`: the fields above that are null although a value was looked for: `code`, `code.commit` (when `code` is set), `profile`, `mesh_source`, `joints_source`, `built`, `inputs.sketch`, `inputs.views`, `inputs.heldout`, and `scores` when `05-scores.json` is frozen but has no entry for this body |
| `files_sha256`, `files_bytes` | every frozen file's SHA-256 (64 lowercase hex digits) and size, keyed by file name. `version.json` itself is not listed |
| `immutable` | `true` |

**Format 0.** These are the versions frozen by hand before this tool existed (`v0-*`). They have no `format`
and keep only `label`, `milestone`, `mesh_source`, `joints` (free text, read as `joints_source`), `profile`
(free text), `built`, `code` (free text naming the code, not an object), `provenance`, `sketch_run`,
`files_sha256`, `frozen` (a date) and `immutable`. Readers treat every other field as missing.

**Freeze.** A freeze is refused when:

- the label already exists in the sketch folder (labels are write-once), or an empty claim from an
  interrupted freeze holds it (the error says so; remove the empty folder and freeze again);
- the sketch folder or the label in the store is a symbolic link;
- any frozen record's stamp says `"dirty": true`, whatever its commit, unless `--allow-dirty` is given (stage
  records, `sections.json` and `05-scores.json` alike);
- none of `stage-4.json`, `sections.json` and `stage-3.json` names a commit, unless `--allow-unknown-code` is
  given;
- a stamp has another shape (a `dirty` that is not true, false or null, or a `commit` that is not a string or
  null), or a frozen JSON record cannot be parsed;
- a symbolic link is on the way to anything copied or hashed;
- the run folder does not exist, or its body folder has no `high.glb`;
- a label, sketch, body or file name is not a plain name, or a text field names a path (see Layout).

The sketch folder is `--sketch` when given. That is refused when the run has a sketch and the folder already
holds versions whose `inputs.sketch` differs from its hash. Without `--sketch`, it is the one sketch folder
whose versions record this run's sketch hash. Without a sketch hash or a match, `--sketch` is required. Two
matches are refused.

Writing is atomic:

1. Each file is copied into a new `.partial-<label>-<random>/` folder in the sketch folder, never through a
   symbolic link, hashed as it is written and flushed to disk. A source whose size or modification time changes
   during its copy is refused.
2. The records are parsed from those copies, so `version.json` describes exactly the bytes the version holds.
3. `version.json` is written last.
4. The files are made read-only (0444).
5. The label is claimed with `mkdir <label>`, which fails if anything is already there, and the partial folder
   is renamed onto that empty claim in one step (a rename on POSIX replaces an empty folder, so the claim is
   what makes it never replace another freeze's). On Windows, where a rename never replaces a folder, it
   renames directly. The folder is then made read-only (0555).

Any error or interrupt removes the partial folder, the claim, and the sketch folder when this freeze created
it and it is empty. A process that is killed can leave a `.partial-*` folder, which `list` reports as
unfinished and every other command refuses. Between the claim and the rename, a kill can leave an empty
`<label>` folder, which `list` reports as an empty claim and a later freeze of that label names as one. A claim
made by another freeze while this one runs refuses this one.

After writing, the freeze reads the permission bits back. When the files or the folder are still writable, as on
a filesystem that ignores permissions (such as some SMB mounts), it completes and prints a warning.

**List** prints each version's sketch and label (the folder's name), milestone, frozen time, profile, mesh and
joints sources, code commit (or "not stamped") and file count. It also names unfinished freezes, empty claims,
folders that hold no readable `version.json`, and store folders that are symbolic links. A `version.json`
whose `label` differs from its folder's name is shown under `label_recorded`.

**Verify** re-hashes every file in `files_sha256` (either format). It is refused when a recorded hash is not a
SHA-256 hex digest. It passes only when every listed file is a regular file (not a symbolic link) that can be
read and has its recorded hash, every listed name is a plain file name (unsafe names are reported and never
opened), the folder holds nothing beyond the listed files and `version.json`, and `version.json`'s `label` (when
set) is the folder's name. Every problem is reported and none stops the check: a changed, missing, unreadable
or irregular file, an unlisted file, an unsafe name or a label mismatch.

**Compare** first verifies both versions as `verify` does and refuses one that fails. It then prints, or with
`--json` writes, the two versions side by side. Every delta is B - A.

- *Scores.* Scores come from the version's own `05-scores.json`. Only a format 0 version without one reads
  `<sketch>/05-scores-<series>.json`, where `<series>` is the label up to its first hyphen (`v0-carve` reads
  `05-scores-v0.json`). That file is not part of the version, so its SHA-256 is reported with the comparison
  (`scores_file_sha256`); when it is a symbolic link it is not read and the scores are missing. A format 1
  version without its own `05-scores.json` has no scores. The body's entry is the one `scores.body` names.
  When `version.json` names no `scores.body`, it is the entry whose `glb` is `<source.body>/high.glb` with no
  `variant_of`. When it names no `source.body` either (format 0), it is the one entry with the version's
  `mesh_source` and no `variant_of`. With no unique match, the scores are reported missing, never guessed.
- *Per view.* IoU, colour error and SSIM for each view either side scored: the views that are not held out
  first (front, back, left, right, top, in that order), then the held-out views. A view counts as held out when
  the scores file's `views` list marks it so. A scores file without that list (the hand-collected format) counts
  every view other than those five as held out. The seen column says what was built from the view:
  `shape+colour`, `shape`, `colour` or `unseen`. Each group gets a mean over the views that have the metric on
  both sides. A held-out view's `measures` (consistency or accuracy) is shown when its scores record it.
- *Triangles.* For the high and low LODs, from `stage-4.json` `body.lods`, else the scores entry's `triangles`
  and `triangles_low`, else `sections.json` `lods`.
- *Gates.* Stage 3's and stage 4's `ok`, and every check that is a gate (`gate` not false) on either side, with
  its threshold, value and pass or fail on each side. A side where the check is not a gate is marked
  "not gated". An operator (`<=`, `>=`, `==`, `<`, `>`) is shown only when the check records one.
- *Joint gaps.* From `sections.json` `pose_test`: `gaps_high`, `gaps_low`, `gaps_high_heldout` and
  `gaps_low_heldout`. Each joint's worst gap is taken over the poses both sides ran. For the gated keys, poses
  listed in either side's `ungated_poses` are left out. The table names how many poses each side ran.
- *Inputs.* The sketch, each view, mask and held-out view is `same`, `different` or `unknown`, by the hashes
  in `inputs`. A format 0 version records none.
- *Missing.* Every number present on one side and absent on the other is listed as missing in the side that
  lacks it: a view or one of its metrics, a triangle count, a gate check, a joint's gap or a whole gap table. A
  side without scores or without `stage-3.json`, `stage-4.json` or `sections.json` is listed too (its gate
  checks then are not listed one by one). A low LOD number is "not built", not missing, when `stage-4.json`
  `profile.lods` leaves out `low`.

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

## Model API calls: retries, shared limit, ledger

All Muse Image, SAM 3.1 and Muse Spark calls go through `herogen/pipeline/client.py`; the policy is in
`herogen/pipeline/retry.py`.

Muse Spark (`muse-spark-1.3`) answers questions about images (`ModelClient.compare_images`): one Responses API
call with the prompt, the images as JPEG data URLs scaled to at most 1024 px, and a strict JSON schema. The answer
is checked against the schema, live and on replay (types, required fields, enums, no extra fields; a schema keyword
outside that subset is refused), so a non-conforming answer stops the stage instead of being guessed. It is
cached like any result; an incomplete answer is never cached. Its cost is estimated from the token usage at
$1.25 / $4.25 per million input / output tokens (reasoning counts as output) and logged per call; a call whose
answer never came back is counted at 0, since its tokens are unknown.

- **What is retried:** HTTP 408, 409, 425, 429 and 5xx, except 501 and 505 (they say the request can never
  work), and network errors (no HTTP status). Never any other 4xx, and never a 200 whose body is not a JSON
  object. A 200 answer whose `status` is `failed` with error code `rate_limit_exceeded` or `server_error` is
  retried as a 429 or a 500 (and logged as a call of unknown outcome, since it may be billed); any other answer
  that did not complete stops the call at once.
- **How long to wait:** the server's `Retry-After` (seconds or an HTTP date; a non-finite or unreadable value is
  ignored) plus 0-1 s of jitter; else the
  largest rate-limit reset header (`retry-after-ms`, `ratelimit-reset`, `x-ratelimit-reset`,
  `x-ratelimit-reset-requests`, `x-ratelimit-reset-tokens`; an `x-ratelimit-reset` over 10^9 is read as an epoch
  time in seconds, over 10^11 in milliseconds) plus the same jitter; else full-jitter exponential backoff, a random time between 0 and
  min(60 s, 2 s x 2^n) before retry n + 1.
- **Limits:** at most 8 attempts within a per-call time limit of 10 minutes for SAM 3.1 and Muse Spark and 15 for Muse
  Image (`DEADLINE_S`), separate from each attempt's 240 s socket timeout. A wait that would end past it is not started;
  the call fails at once. A call that fails raises `ApiError`, which stops the stage (see "Mask method").
- **Shared limit per key:** every process calling with one API key shares one limit on calls in flight, in a
  JSON state file under `HEROGEN_LIMITER_DIR` (default `~/.cache/herogen/api-limiter`) named by a 16-hex-digit
  SHA-256 prefix of the key, never the key, and guarded by an exclusive file lock (`herogen/pipeline/filelock.py`:
  `flock` on POSIX, `LockFileEx` on Windows). The limit starts at 4. A 429
  halves it (never below 1) and, when its caller starts the wait it was given, holds every process off for that
  wait, at most 60 s (`HOLD_MAX_S`), so a huge or bad wait never blocks the key; a hold read back from the
  state file is cut to 60 s from now too. Every 20
  successful calls since the last 429 raise it by one, up to 8. A slot held by a process that has died, or
  for longer than 300 s, is freed. The start and maximum are configured, not measured.
- **Ledger and logs:** every HTTP attempt is a line in `<cache>/calls-ledger.jsonl`: time, model, endpoint,
  request key, attempt, status, outcome (`ok`, `failed`, `unknown` or `unreadable`), seconds, the wait for a
  call slot and the wait before the next attempt with its reason, and the scrubbed error. Each retry is also a
  warning on the `herogen.pipeline.client` logger. Neither holds a header or the key. An attempt that may
  have reached the model and be billed without a usable answer has outcome `unknown`, is also in the stage
  record's `calls` (`"outcome": "unknown"`) and is counted in the cost estimate: one that timed out or was cut
  off while sending, any error while waiting for or reading the answer, a 200 answer that can't be read or
  that failed, and, for a Muse Image generation (paid and not idempotent), any 5xx but 503. A Muse Image
  generation is never retried on a 409.

## Releases

Tags deploy the hosted service (`herogen/service`) to Modal through `.github/workflows/release.yml`. The
rules are `tools/release/release.py`, tested in `tests/test_tools_release.py`; the workflow only calls it.

A release deploys the service's Modal app, `herogen/service/modal_app.py`, so it needs the service code merged
into the tagged commit. A tag on a commit without that file passes the tag check and the tests, then fails at the
first step of the deploy job with `herogen/service/modal_app.py is not in this commit, so there is no service to
deploy`.

Releases are trunk plus tags: every release tag, rc or final, must point at a commit on `main`, that is, its
commit is an ancestor of (or is) the tip of `origin/main`. A tag on a commit only on another branch deploys nothing,
whatever its name; merge the change into `main` and tag a commit there with a new tag. A commit that was on a side
branch passes once it has been merged into `main`.

| Tag | Deploys to | Only when |
| --- | --- | --- |
| `vX.Y.Z-rc.N` | the `staging` GitHub environment | always |
| `vX.Y.Z` | the `production` GitHub environment | the same commit also carries a tag `vX.Y.Z-rc.N` (any N), and the `staging` environment has a successful deployment of that commit |
| any other tag starting with `v` | nothing: the workflow fails and names the format | |

X, Y, Z and N are whole numbers without leading zeros, and `rc` is lower case: `v1.4.0-rc.1`, `v1.4.0`, `v2.0.10`.
`v1.4`, `v1.4.0-rc1`, `v1.4.0-RC.1`, `v01.4.0` and `v1.4.0-beta.1` are refused. Lightweight and annotated tags
both work; an annotated tag is read as the commit it points at.

### Jobs

1. **plan** checks the tag (`release.py plan <tag>`) and passes the environment, `staging` or `production`, to
   the deploy job. It fetches `main` explicitly
   (`git fetch --no-tags origin +refs/heads/main:refs/remotes/origin/main`) and refuses any tag whose commit is not
   an ancestor of `origin/main` (`git merge-base --is-ancestor`), naming the tag and the commit; it also refuses
   when `origin/main` cannot be resolved. For a production tag it also:
   - lists the tags on the tag's commit (`git tag --points-at`, on a checkout holding every tag) and refuses unless
     one is an rc tag of the same version;
   - asks GitHub's deployments API for the `staging` deployments of that commit (`environment=staging`,
     `sha=<commit>`, following every page up to 20 of 100) and each one's statuses, and refuses unless one of
     them has a `success` status. A deployment GitHub later marked `inactive`, because a newer one to staging
     succeeded, still counts. It refuses, saying which, when staging has no deployment of the commit, when one is
     still queued, pending or in progress (or has no status yet), when none succeeded (it names each one's latest
     state), and when the API cannot be read: 401, 403, 404, 5xx or any other answer but 200, no answer, an
     answer that is not a list, a next page on another host, or more than 20 pages. This job alone has the
     `deployments: read` permission; its token goes to the check step only, and the checkout keeps none.
2. **test** is `.github/workflows/tests.yml`, called as a reusable workflow: the same unit tests and guard as on
   every branch push. Tags starting with `v` do not start `tests.yml` on their own, so a tag runs the tests once.
3. **deploy** needs both, and runs in the GitHub environment the plan chose, with that environment's secrets and
   variables. Deploys to one environment run one at a time: a running deploy is never cancelled, and a deploy
   waiting behind it is replaced by a newer one that starts waiting. Its steps:
   - before anything is installed, checks that `herogen/service/modal_app.py` is in the commit, that every
     setting below is set, and that the custom domain holds the base URL's host (`release.py check <tag>`). It
     fails the job with every problem at once: the missing file, each missing or blank setting by name, as a
     secret or a variable, and a domain or base URL that does not fit, naming both hosts. A secret's value is
     never printed;
   - installs the Modal CLI pinned with its dependencies in `tools/release/requirements.txt`;
   - `release.py deploy <tag>` checks the same again, then runs
     `python -m modal deploy -m herogen.service.modal_app --tag <tag>`, so each Modal deployment is tagged with its
     release tag. The service bakes every `HEROGEN_SERVICE_*` variable of the deploying shell into its image, so
     of those the deploy gets only the five variables below and two fixed ones; any other `HEROGEN_SERVICE_*`
     variable in the job is removed. The fixed ones: `HEROGEN_SERVICE_MODE=production` (the service refuses to
     deploy in any other mode; staging deploys use it too) and `HEROGEN_SERVICE_EMAIL_PROVIDER=resend` (production
     mode needs a real email provider, and Resend is the only one the service has).

### Staging and production

Staging and production are told apart by the Modal app name, `HEROGEN_SERVICE_APP`, set per GitHub environment
(for example `herogen-service-staging` and `herogen-service`). The service names its app, volumes (`<app>-data`,
`<app>-meta`, `<app>-legal`) and Modal secrets (`<app>-web`, `<app>-model`, `<app>-admin`) after it, so the two
share nothing. Staging is a separate app that also runs in production mode (`HEROGEN_SERVICE_MODE=production`,
real sign-in mail through Resend); only its name, settings and secrets differ. Each environment has its own Modal
token, so they can also be in different Modal workspaces. The deploy uses the token's default Modal environment.

Each environment has its own custom domain, set up before its first deploy, so `HEROGEN_SERVICE_BASE_URL` is known
from the start and the check requires it. Deploying first and then setting the base URL to the address Modal
prints for the `web` function is not supported by this flow.

`HEROGEN_SERVICE_DOMAIN` attaches the custom domain: the service passes it to its `web` function's
`custom_domains`, reading it as a comma-separated list (entries stripped, empty ones dropped). The check reads it
the same way and requires every entry to be a bare host name (no scheme, port or path), the base URL to be
`https://<host>` with no port or path, and the base URL's host to be one of the entries (compared without regard
to case). Usually it is that one host. The service accepts requests only for the base URL's host
(`HEROGEN_SERVICE_HOSTS` defaults to it and is not passed), so any other domain listed is attached but its
requests are refused.

The SAM 3D apps that stage 3 can call, `herogen-sam3d-objects` and `herogen-sam3d-body`
(`herogen/blender/remote/`), have fixed names, so staging and production share them. This flow does not deploy
them; redeploying either changes it for both environments at once. Per-environment names for them are not built
yet.

Each GitHub environment needs:

| Name | Kind | What it is |
| --- | --- | --- |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | secret | The Modal API token the deploy runs with |
| `HEROGEN_SERVICE_APP` | variable | The Modal app name, which prefixes every volume and secret of the service |
| `HEROGEN_SERVICE_BASE_URL` | variable | The service's public origin, `https://<host>` on the environment's own custom domain |
| `HEROGEN_SERVICE_DOMAIN` | variable | The custom domain the web function is served on: the base URL's host (a comma-separated list must include it) |
| `HEROGEN_SERVICE_EMAIL_FROM` | variable | The sign-in mail's From address, on your own sending domain |
| `HEROGEN_SERVICE_TERMS_VERSION` | variable | The version of the published terms users accept |

The service also needs, in Modal, before the first deploy of an app: the secrets `<app>-web`
(`HEROGEN_SESSION_KEY`, `HEROGEN_EMAIL_API_KEY`), `<app>-model` (`MODEL_API_KEY`) and `<app>-admin`
(`HEROGEN_ADMIN_KEY`), and the terms and privacy documents on the `<app>-legal` volume. The release flow does not
create them.

### Cutting a release

Tag the commit to release as the first rc of a new version and push the tag; it deploys to staging:

```bash
git tag v1.4.0-rc.1 <commit>
git push origin v1.4.0-rc.1
```

A fix for the same version goes on a new commit with the next rc, `v1.4.0-rc.2`. To promote to production, wait
until the rc's staging deploy has succeeded, then tag the same commit with the version alone and push it:

```bash
git tag v1.4.0 'v1.4.0-rc.2^{commit}'
git push origin v1.4.0
```

A production tag fails in **plan** and deploys nothing when its commit has no rc tag of its version or no
successful staging deployment. Tag that commit `v1.4.0-rc.N` if it has none, let it deploy to staging, then re-run
the final's failed release run: a re-run reads the tags and the deployments again. An rc and its final pushed
together deploy staging only: the final's check runs before the rc's staging deploy has succeeded, so it refuses;
re-run the final's release run once staging is green.

To check a tag after creating it and before pushing it, run the same check in a clone with every tag fetched. For
a production tag it reads the deployments too, so it needs the repository and a token that may read them:

```bash
python3 tools/release/release.py plan v1.4.0-rc.1
GITHUB_REPOSITORY=<owner>/<repo> GITHUB_TOKEN=<token> python3 tools/release/release.py plan v1.4.0
```

### Rolling back

Two ways, both through the same flow:

- **Redeploy an earlier tag.** Re-run the release run of the earlier tag in GitHub Actions ("Re-run all jobs"). It
  checks out that tag's commit, runs its tests and deploys it again with the environment's current settings.
  For a production tag the staging check passes again, since the earlier rc's successful staging deployment is
  still recorded. GitHub allows re-runs up to 30 days after a run.
- **Release an earlier commit as a new version.** Tag the earlier commit with a new rc, for example
  `v1.4.1-rc.1`, and push it; once it has deployed to staging, tag and push its final, `v1.4.1`, as above. The
  commit must hold the release workflow and the service code.

Neither rolls back data: the `<app>-data` and `<app>-meta` volumes are left as they are, so the earlier code runs
on whatever the newer code wrote.

## Rules

- No secrets in code, logs or run folders. Model keys come from the environment.
- Committed tests use synthetic inputs only. No game assets, no real sketches, no personal paths.
- Model calls go through one client module per package with a replay mode for tests.

## Lab

`herogen/lab` is a read-only local viewer (`python3 -m herogen.lab serve`). It never writes to a run or version
folder and binds to 127.0.0.1 only.

**Workspace.** One or more runs folders (`--runs`, repeatable) and an optional versions folder (`--versions`).
Runs are named `<root>/<run>` with root keys `r1`, `r2`, ... in the order given. `workspace.py` groups runs into
characters by the SHA-256 of `00-sketch.png`, hashed from the file because no stage record holds a sketch hash (a
run without one, or whose sketch is a symlink, is its own character), and files each run under the profile its body
records: `04-body/sections.json` `profile`, else `stage-4.json` `profile.name`, else "not recorded". A run folder
that cannot be listed stays in the workspace as a run that can't be read, with the reason.

**Record formats read** (`runindex.py`, `heldout.py`):

| Record | Formats |
| --- | --- |
| `stage-<n>.json` | with or without `started`, `code` (`commit`, `dirty`) and `calls`; comparisons in `op` or `rule` (a sentence in `rule` is a note); `poses` and `sections` on checks when listed |
| `stage-2h.json` | `details.views` with per-view `status`, `machine_status`, `review` and `files`, checks named `<view>.<check>`; or the older `details.view` with unprefixed checks (passed or failed its checks); or no record, only `heldout/<view>.png` ("not recorded") |
| `sections.json` | budgets as `triangles_high_max`/`triangles_low_max`, as `triangles_total_max` with part `groups`, or none; `lods.<lod>` totals and `triangles_by_group`; `pose_test.gaps_*` tables; `not_built` lists or `"not built: ..."` entries |
| `05-scores.json` | with a `views` list marking held-out views, or without one (a view every body records as unseen is held out); frozen `05-scores-<tag>.json` with a `heldout` block |
| `version.json` | `DIR/<group>/<version>/` or `DIR/<version>/`, with `sketch_run` and `files_sha256` |

A file that cannot be parsed, or a record that is not a JSON object, is reported by name (`"<file> could not be
read (<error>)"`) and everything else is still shown; NaN, infinities and wrongly typed fields become null. The
triangle counts in the viewer, the deliverables, the versions list and compare are counted from each GLB's JSON
chunk (indexed or unindexed triangle-list primitives per named node); the budget meters show the body record's own
totals (`lods.<lod>.triangles`, `triangles_by_group`) against its own budgets.

**Linking.** A run with a sketch and without its own `stage-2h.json` uses the newest held-out record of a run with
the same sketch and byte-identical `views/{front,back,left,right}.png` (symlinked views do not count); the API
names that run in `heldout_source`. Its gates count in the run's status, so the page names the run wherever that
status is shown (the header, the pipeline strip and the quality panel); its time and cost are listed but kept out
of the run's total (`generation.total`, with `generation.with_borrowed` beside it). A version attaches to a
character only when every run named by its `sketch_run`, in every root, holds that character's sketch; otherwise
it is listed in `unattached_versions` with the reason. Its scores are the one non-variant body of a frozen score
file whose mesh source and high-LOD triangle count match the version's own `high.glb`, else none.

**API** (`server.py`): `/api/workspace`, `/api/runs`, `/api/runs/<root>/<run>`, `/api/versions/<id>`, files at
`/runs/<root>/<run>/<path>` and `/versions/<id>/<file>`, and the page at `/` with a Content-Security-Policy that
allows the pinned three.js modules and the page's own files only. `/static/` serves CSS, JS, SVG, JSON, text, TTF
and WOFF2 files, never HTML. Every other response carries `default-src 'none'; sandbox`. Each file path is decoded
once and must consist of plain names; a run folder that is a symlink is refused, and the file must resolve inside
its own run folder (a link to a sibling run is refused), its own version folder or the static folder. An idle or
half-sent request is closed after 30 seconds.

**Page** (`static/`): `app.js` (panels), `viewer.js` (three.js views, drawn only when something changes),
`poses.js` (the seven test poses of `herogen/blender/pose.py`, mapped from Blender's +Z up to glTF's +Y up),
`dom.js` (helpers; run data goes on the page as text, icons are rebuilt from the vendored SVGs through an
allow-list) and `fonts/` (the three interface fonts and `fonts.css`, same-origin, so the CSP needs no font source
of its own). Records name poses the viewer does not define (the newer pose sets); "Show me" then isolates the joint
at rest and says so.

## Service

`herogen/service` is the hosted, invite-only version: people sign in with an emailed link, upload a sketch or
describe a hero, follow its stages, and download the GLB and a Roblox bundle. It runs on Modal; dev mode runs the
whole service on one machine with stand-ins.

```bash
pip install -r herogen/service/requirements-dev.txt
python3 -m herogen.service dev                    # http://127.0.0.1:8800, sign-in links printed and in service-data/outbox.jsonl
python3 -m herogen.service dev --runner local     # the real stage CLIs on this machine (Blender, MODEL_API_KEY)
python3 -m herogen.service admin invite --count 3 # admin commands against the dev store
modal deploy -m herogen.service.modal_app         # deploy (see "Deploying" below)
```

### Components

| Part | Where | What it does |
| --- | --- | --- |
| Web | `web.py`, Modal `web` | Starlette ASGI app: pages, JSON API, Lab API, downloads. Holds the `<app>-web` secret. |
| Metadata store | `store.py`, `snapshot.py`, Modal `StoreService` | Users, allowlist, invites, sign-in links, sessions, jobs, stages, audit log. The only writer. |
| Upload cleaner | `uploads.py`, Modal `sanitize_image` | Opens and re-encodes uploads, with no network, no secrets, no volumes and no access to other Modal resources. |
| Job orchestrator | `worker.py` `run_job`, Modal `run_job` | One per job attempt: claims the job, runs the stage groups in order, finishes the job. |
| Stage groups | `worker.py` `run_group`, `runner.py`, Modal `model_stages`, `blender_stages`, `export_stages`, `heldout_stages` (beside the job) | Run the pipeline's own CLIs on a local copy of the job's files, each group in a fresh container. |
| Sweep, retention | `worker.py` `reconcile`, `purge`, Modal `sweep` (every minute), `retention` (daily 03:17 UTC) | Requeue jobs whose worker died, start queued jobs, delete expired files and records. |
| Admin | `admin.py`, Modal `admin` entrypoint | Invites, allowlist, users, quotas, jobs, audit log. No admin web page. |

### Decisions

**Web: Starlette (BSD-3-Clause) on Modal's ASGI support.** FastAPI is built on Starlette; the extra it brings
(pydantic models, OpenAPI) buys nothing for a dozen hand-checked endpoints, and Starlette alone has the pieces
used here: request-body limits per route (`max_body_size`), the trusted-host check, and multipart parsing through
python-multipart (Apache-2.0) with limits on files, fields and part size. Dev mode serves the same app with
uvicorn on 127.0.0.1.

**Jobs: an orchestrator and three stage groups, retries owned by the store.** Modal starts an input again after its
container dies, without a limit, and has no idempotency key, so no function sets Modal `retries` and every job
function is `nonpreemptible`. Instead:

- `job_claim` is a worker's first call. It starts the next numbered attempt of a queued job and returns nothing for
  a job that is running, finished, deleted or unknown, so a duplicate or restarted orchestrator does nothing. A job
  out of attempts (`max_attempts`, default 2) fails there, before any paid call.
- Every later update names its attempt, and the store refuses updates from any attempt but the live one; a
  canceled or deleted job's worker stops at the next stage. A stage group that a restarted container starts more
  than twice for one attempt stops before running anything (`group_enter`).
- Each running stage holds a lease covering its timeout plus 10 minutes, and the orchestrator renews the job's lease
  every 2 minutes to at least 10 minutes ahead for as long as it runs (`worker.LeaseKeeper`, `store.job_renew`),
  so no copy in or back, cold start or stage costs a live job its lease, however long it takes. The sweep requeues
  a job whose lease ran out (its worker died: within 10 minutes of the last renewal, or at the end of the running
  stage's own lease if that is later) or fails it when it is out of attempts. A container that hangs is stopped
  by its function's timeout, which ends the renewals.
- When an attempt ends without its worker recording the stage it was running (a cancel, a delete, or the sweep's
  requeue or failure), the store closes that stage in the same write: "pending" when the job is queued again,
  "canceled" when it was canceled, "failed" otherwise. Only the live attempt may write a stage, so nothing else
  would, and the stage would show as running for good.
- Errors are sorted three ways. A failed gate rejects the job, with the stage's reason. A timeout, a network
  error, or the model or GPU service answering HTTP 429 or 5xx queues another attempt. Anything else (a 4xx, a bad
  response, a bug) fails the job.
- A failed job gives its quota back only if it spent nothing (the recorded cost of its stages, model calls and
  compute together, is zero), and at most twice per user in any 24 hours (`store.REFUNDS_PER_DAY`). A refund gives
  back the day's slot only; the job still counts toward the lifetime total. A stage that fails after paid calls
  therefore cannot be used to run paid work past the quota.
- A retried attempt resumes after the leading stages whose records already passed. Model calls go through the
  model client's record/replay cache, kept per user, so a repeated call is replayed at no cost.
- A group works on a local copy of the run folder, its input and the user's caches. After each stage it copies the
  results back to the data volume and commits, but only while its attempt is still the live one, so a stale worker
  never overwrites a newer attempt's files.
- At most `max_running_jobs` jobs run at once (default 4, to stay well inside the Model API's per-team rate limits);
  the rest wait queued, oldest first.

Stages run as the same commands a person types (`python3 -m herogen.pipeline run`, `python3 -m herogen.blender
run`), each in a subprocess with a timeout and an environment that holds only what the stage needs:
`MODEL_API_KEY` reaches the model stages and the held-out views only, and no stage sees the session key or the
email key. Options that
only some versions of a stage CLI have (`--profile`, `--remote-cache`) are passed only when that CLI's `--help`
lists them. A job that starts from a description first runs stage `p` (`sketchgen.py`): one `muse-image-1.0`
generation from a fixed pencil-sketch prompt around the description, recorded as `stage-p.json`; the pipeline then
starts at stage 0 from that sketch.

| Group | Stages | Worker | Secret |
| --- | --- | --- | --- |
| model | p (described heroes only), 0, 1, 2 | 2 CPU, 4 GiB, 4.5 h | `<app>-model` |
| blender | 3, 4 | 8 CPU, 32 GiB, 6.5 h, Blender 4.5.14 LTS | none |
| export | 6 (Roblox rig: `python3 -m herogen.roblox export --body <run>/04-body --out <run>/06-roblox`), 5 (results page) | 2 CPU, 4 GiB, 2.5 h | none |
| heldout (beside the job) | 2h (held-out views, for scoring) | 2 CPU, 4 GiB, 1.5 h, at most `heldout_concurrency` (2) at once service-wide | `<app>-model` |

The function timeouts follow the stage timeouts (`HEROGEN_SERVICE_STAGE_TIMEOUT`, 1 h per stage, and
`HEROGEN_SERVICE_BLENDER_TIMEOUT`, 3 h per Blender stage): each group's function allows all its stages back to back
plus 30 minutes, and the orchestrator (`run_job`, 14 h with the defaults) allows the job's own three groups plus 30
minutes; it never waits for the held-out views.
`modal_app` computes them at deploy time and refuses a configuration over Modal's 24-hour limit.

**Held-out views (stage 2h) run beside the job and can never fail or delay it.** No building stage reads
`heldout/`; the views exist only to score the body on views it was not built from. So:

- The model group runs stage 2 alone (`pipestage.py`, through the same functions the pipeline CLI uses; the CLI
  itself would follow stage 2 with 2h). Once stage 2 has passed, the orchestrator starts 2h in its own function
  (`heldout_stages`, started with `spawn` and never awaited) and goes straight on to stages 3, 4, 6 and 5. The GLB
  and the Roblox bundle are delivered as soon as those finish.
- 2h runs at most once per job (`store.side_claim`) and is never retried. Whatever happens to it (a failed gate,
  every view rejected, a Model API error, a timeout, a crash, a lost container, a rate or quota limit) is recorded
  on the 2h stage only, as passed or "unavailable" with a sentence saying the hero is not affected. Its lease is
  renewed the same way from the claim to its record (`store.side_renew`), covering the copy in of the user's cache,
  the run and the copy back; a lost worker is marked unavailable by the sweep within 10 minutes of its last
  renewal. It never sets the job's state or error, never
  requeues, retries or refunds it, and never changes whether the job counts toward the quota (its cost is left
  out of the refund decision).
- Its cost joins the job's recorded cost (`cost_detail.heldout_usd`), at the job's finish if 2h finished first,
  else when 2h finishes. A 2h that finishes after the job, including a canceled job, attaches its record and cost
  and changes nothing else.
- Files: 2h works on a local copy and writes back only `heldout/` and `stage-2h.json` (and new cache entries), only
  into a run folder that still exists, and only while its claim holds and the job is not deleted. After each
  commit it reloads and checks again: deleting a hero or an account marks it deleted in the store before its files
  are removed, so a delete that landed during the copy back is seen there, and every file the copy back wrote is
  removed and committed again. The job's own copy back does the same. So a deleted hero's or account's files may
  reappear only for the moment between those two commits, never for good. The job's own copy back neither copies
  nor deletes 2h's two paths, so the two never write the same file.
- Model API share: 2h starts only after the job's own Model API calls (stages p and 0 to 2) are done, and at most
  `heldout_concurrency` (default 2) held-out runs execute at once across the service (`max_containers` of
  `heldout_stages`; more wait in Modal's queue), so they yield the API to other jobs' own stages. The pipeline's
  shared limiter coordinates the processes of one container only (it locks a local file), so across containers
  this cap is what bounds the held-out views' share.
- The page lists 2h last as "quality check (held-out views)" and keeps polling while it runs after the job (each
  stage in the job's JSON says whether it runs beside the job, `side`; only those keep a finished job polling). The
  results page (stage 5) is written when the job's own path ends, so it shows 2h only if 2h finished first; the
  viewer always reads the current `stage-2h.json`.

Stages 6 and 5 are optional: a rejection or
error there is recorded on the stage and shown as the reason the Roblox bundle is unavailable, and the job still
delivers its GLB. When the deployed code has no `herogen.roblox` package, stage 6 is recorded as skipped with
that reason. The export writes `stage-6.json` at the run root with its `outputs` relative to it; an `error` line
in that record is shown to the user only when it names no path.

**Storage: two Modal Volumes per app, per-user folders.** `<app>-data` holds every user's files under
`users/<user-id>/` (`inputs/<job-id>.png`, `runs/<job-id>/`, `cache/`, `cache-remote/`; see `storage.py`). User and
job ids are made by the service (`u_` or `j_` and 26 random base32 characters) and checked against that pattern
before they touch a path. The only request text that becomes a path is a run-relative file path, accepted only as
plain name segments of an allowed type (PNG, JPEG, GLB, JSON) that resolve, with no link along the way, inside
that job's run folder. Each user's storage is limited (`user_storage_max_bytes`, default 2 GiB), counted from the
recorded size of each finished job. `<app>-legal` holds the published `terms.md` and `privacy.md`.

**Metadata store: SQLite, one writer, snapshots on a Volume.** The store needs transactions (a quota check and the
job it allows, a sign-in link and the account and session it opens), so a key-value store does not fit, and a
Modal Dict also drops entries after 7 days without use. SQLite cannot keep its own file on a Volume either: Volumes
have no file locking and the last commit wins. So `StoreService` runs in at most one container
(`max_containers=1`, one input at a time), keeps the live database on local disk (WAL journal, `synchronous=FULL`)
and after every write transaction copies it with SQLite's online backup into a new numbered snapshot on
`<app>-meta`, then commits, before the call returns. Two store containers can still overlap during a deploy, and a
Volume has no compare-and-swap: a commit that started before a check can land after it. So no file is ever written
by two containers, and the newest state is decided by file names (`snapshot.py`):

- Each store container has a random id. On start it adds a lease file `lease-<epoch>-<id>`, one epoch above every
  lease it sees, and checks after committing that its lease is the largest (epoch, then id). Each snapshot is a new
  file `meta-<epoch>-<id>-<generation>.db`; the newest state is the largest (epoch, id, generation) snapshot SQLite
  can read.
- Before writing, a store reloads the Volume. It stops for good if a larger lease or a snapshot from a larger epoch
  exists, and restores and retries the call if the newest snapshot is not the one it holds. After committing it
  reloads again and reports the write saved only if its lease is still the largest. A call the store cannot
  complete this way reaches the user as "busy" (below).
- Snapshots are written in SQLite's rollback-journal mode, so each is one self-contained file: reading it needs no
  `-wal` or `-shm` file beside it, and none is ever created on the Volume.

Guaranteed: if a save returns, its snapshot is on the Volume, and every store that starts afterwards restores that
snapshot or one written later by a store that restored it (directly or through such later snapshots), unless one of
those snapshots is found corrupt when it is restored. No store container, stale or new, can replace or outrank it:
a newer store's lease landed after that commit, so the newer store restored it, and everything an older store
commits afterwards ranks below the newer store's epoch. A save refused before writing is retried after restoring,
and nothing of it reached the Volume. A save that cannot be confirmed after its commit is reported as failed and not
retried; that write may still survive. A store that has been superseded never writes again.

Reading a snapshot: the store copies it to local disk and uses it only after the copy has the file's size on the
Volume and SQLite opens it and `PRAGMA integrity_check` passes. If the file cannot be read or opened right now (an
I/O error, a short read), the store does not start and logs why; it never falls back, so a passing fault cannot drop
confirmed writes. A file proven corrupt (not a database, a malformed image or a failed integrity check, which is also
how a torn or partly written file shows) is moved to quarantine on the Volume (`quarantine-<id>-<name>`, which no
container restores, prunes or counts among the three snapshots kept), logged, and the previous snapshot is restored;
the writes only that file held are then lost. The store starts from an empty database only on a Volume holding no
snapshot file, and no quarantined one either unless an operator allowed it ("If the store will not start",
below).

The schema is versioned (`PRAGMA user_version`): an older database is upgraded in place by the steps in
`store.MIGRATIONS`, and a version this code does not know is refused at start. A managed Postgres would remove the
single-writer limit, at the cost of another provider account; every query is in `store.py`, so moving is contained.
Rate-limit counters live in the store's memory and reset when it restarts.

**Sign-in: invite-only, emailed links, no passwords.**

- A sign-in link carries a 256-bit token in its URL fragment (`/signin#t=...`), which browsers never send to a
  server or in a Referer. The page shows the masked address and asks for an explicit click, then POSTs the token,
  so a mail scanner that fetches the link cannot use it up. Links expire after 15 minutes (`link_ttl_s`), work once,
  and using one cancels every other outstanding link for that address.
- Links are built from the configured `base_url`, never from the request's Host header.
- A link is sent only to an active account, an allowlisted address, or an address with a usable invite code (a code
  may be limited in uses, lifetime and to one address). Before answering, the request does only the rate-limit checks
  and, when a code is given, checks the code alone (a read that is the same for every address; an unknown, expired,
  revoked or used-up code is refused). The answer is then the same for every address. Whether the address may sign
  in, the link (a durable write) and the email all happen after the response, so neither the answer nor its timing
  says who has an account.
- Sign-in emails must go out without click tracking: turn off click and open tracking for the sending domain in
  Resend, or the link is rewritten through the provider's redirect.
- Creating an account, and signing in after the terms version changes, needs two confirmations: that the person is
  18 or older, and that they accept the Terms of Service and Privacy Policy (`terms_version`). Without a published
  version no one can sign in, and in production both documents must be on `<app>-legal`.
- Sessions are a cookie: `__Host-hg_session`, HttpOnly, Secure, SameSite=Lax, Path=/. They last 30 days at most
  and end after 7 days unused. Logging out, disabling the account or deleting it revokes the session on the server.
- Tokens, session ids and invite codes are stored only as HMAC-SHA256 under keys derived from `HEROGEN_SESSION_KEY`,
  one per purpose, so a copy of the database cannot be replayed as a link or a session.
- The audit log names accounts by id and addresses and IPs by keyed digests, never in clear.

**CSRF.** Three layers on every POST, PUT, PATCH and DELETE:

1. SameSite=Lax cookies.
2. An origin check: `Sec-Fetch-Site` must be `same-origin` when present, `Origin` must equal the service's origin when
   present, and one of the two must be present.
3. An `X-CSRF-Token` header that must equal the HMAC of the browser's own `__Host-hg_csrf` cookie, a random value set
   on the first page or session request and replaced at sign-in. The page reads its token from `/api/session`.

**Rate limits** (count per window, settable): sign-in requests 20 per hour per IP and 5 per hour per address; link
checks and uses 60 per hour per IP; new jobs 20 per hour per user; the whole API 600 per minute per IP, counted in
each web container. Behind a proxy that sets the client address, `HEROGEN_SERVICE_IP_HEADER` names its header (the
last entry of a list is used).

**Quota.** Each user may start `jobs_per_day` jobs in any 24 hours (default 5) and `jobs_total` in all (default
50); admins can set a different quota per user. The check runs in the same transaction that creates the job, before
any paid call. Every job counts toward both limits, except one canceled before it started; a failed job that spent
nothing counts toward the lifetime total but not the day's (see "Jobs": at most two such refunds per user per
day).

**Uploads.**

- An upload is accepted only as PNG, JPEG or WebP. Its type is recognised by its leading bytes; the file name and
  declared type are ignored.
- It must be within 12 MiB and 40 megapixels. The pixel count and the memory the decoded image will take (its
  mode from the header; at most 4 bytes per allowed pixel) are checked before anything is decoded, and Pillow's
  decompression-bomb warning is an error. It must have one frame and at least 256 pixels on its short side.
- 16-bit images become 8-bit in integers (the high byte) and 32-bit ones in float32 a block of rows at a time, so
  converting never makes a full-size float copy; running out of memory is a rejection, not a server error.
- Pillow opens it with only the matching decoder enabled, verifies it, decodes it, applies the EXIF orientation and
  converts it to sRGB.
- What is kept is a new RGB image made from the pixels alone, with transparency on white and at most 2048 pixels on
  its long side. It is saved as a PNG with no metadata: no EXIF, GPS, XMP, ICC profile, text or time chunks. The
  client's file name is never stored.
- Descriptions are normalised (NFC, control and direction characters removed, at most 600 characters) and passed
  to the stage on stdin, never on a command line. Hero names keep letters, digits, spaces and `' - . _` (at most 60).

**Downloads.** `GET /api/jobs/<id>/download/glb` sends `04-body/high.glb` (`?lod=low` for `low.glb` when the
profile builds it). `GET /api/jobs/<id>/download/roblox` sends a zip of exactly the files `stage-6.json` lists in its
`outputs` (`06-roblox/`: the `.rbxmx` rig, `bind_meshes.luau`, the rig test script and the README), each checked
like a run file, plus `04-body/high.glb`, which the bind script binds, and `stage-6.json` itself. The zip keeps the
run's layout because the README refers to `../04-body/high.glb` and `../stage-6.json`. It is offered only when
stage 6 passed. Both are `Content-Disposition:
attachment` with a file name made from the hero's name.

**Security headers.** Every response carries `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`,
`X-Frame-Options: DENY`, `Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Resource-Policy: same-origin`, a
restrictive `Permissions-Policy`, `Cache-Control: no-store` and, on https, `Strict-Transport-Security` (two years).
Pages get the Lab's CSP: the service's own files, the inline import map by its SHA-256, and the one pinned three.js
path (`https://cdn.jsdelivr.net/npm/three@0.160.0/`, each module with its integrity hash). Everything else,
including run files, gets `default-src 'none'; frame-ancestors 'none'; sandbox`. Requests must name an allowed host
(`HEROGEN_SERVICE_HOSTS`, default the host of `base_url`).

**Observability.** Every request and job event is one JSON log line (`event`, ids, stage, state, seconds, cost),
never an address, token, description or IP. Each job records the code commit it was started with (baked into the
image at deploy time, `-dirty` when `herogen/` had uncommitted changes), its wall time and its cost: the stage
records' model and GPU estimates plus wall seconds times the worker's list price (8 CPU and 32 GiB at
$0.047 per core-hour and $0.008 per GiB-hour for the Blender worker). Costs are estimates; Modal's and Meta's bills
are the actuals.

### API

| Method and path | What |
| --- | --- |
| `GET /api/session` | Signed in or not, the user and usage, the limits, the CSRF token |
| `POST /api/auth/request` `{email, invite_code?}` | Send a sign-in link if the address may sign in |
| `POST /api/auth/link` `{token}` | The masked address of a link and whether terms must be accepted (nothing is used up) |
| `POST /api/auth/verify` `{token, accept_terms, adult}` | Use the link: create the account if new, open a session |
| `POST /api/auth/logout` | End the session |
| `DELETE /api/account` `{confirm: "delete my account"}` | Delete the account and every file now |
| `GET /api/quota` | Jobs in the last 24 hours and in total, the limits, stored bytes |
| `GET /api/jobs` | The user's jobs, newest first |
| `POST /api/jobs` (multipart: `kind` sketch or prompt, `sketch` or `prompt`, `name`, `profile`; header `Idempotency-Key`) | Start a job; the same key returns the same job (200), the same key with a different request is refused (409) |
| `GET /api/jobs/<id>` | State, stages, the error with its stage and reason, images, downloads, the Roblox README, time, cost, commit |
| `POST /api/jobs/<id>/cancel`, `DELETE /api/jobs/<id>` | Cancel; delete with its files |
| `GET /api/jobs/<id>/download/glb`, `.../download/roblox` | Downloads |
| `GET /api/workspace`, `GET /api/runs/r1/<id>`, `GET /runs/r1/<id>/<path>` | The Lab API over the user's own runs |

Errors are `{"error": {"code", "message"}}`, with the message written for the user. When the metadata store is
restarting or cannot confirm a write, any call answers 503 with `"code": "busy"` and `Retry-After: 5`. For a new job
that answer means the job may or may not have been saved: its sketch is kept, and sending the same form again with
the same `Idempotency-Key` returns the saved job (200) or starts it (201), never a second one. The page keeps the key,
with a hash of the form, in the tab's `sessionStorage` from before it sends the request until the server answers with
the job (200 or 201) or a 409 (the key belongs to a saved job with a different form). On any other answer (503, another
5xx, a lost connection, 401, 429, a refused upload) it sends the same key with the next attempt from that tab, also
after a reload; other tabs and devices make their own keys. A sketch no job claims is deleted by retention after a day. Status polling is every 3
seconds while a job is queued or running, and after that while its quality check runs.

### Lab API and viewer

The service answers the Lab's JSON API per user (`labapi.py`): each user's runs folder is the one runs root, key
`r1`, labelled "Your heroes", so a run is `r1/<job-id>` as the Lab names it. It uses the Lab's own code when the
deployed HeroGen has it (`herogen.lab.workspace.Workspace`, else `herogen.lab.runindex`) and otherwise builds a
minimal summary with the same field names. `/viewer` serves the Studio page and `/static/` its files when
`herogen/lab/static/` holds the Studio (`index.html` and `app.js`); otherwise it serves the service's own minimal
viewer, which reads only that API. Both open `#run=r1/<job-id>`.

### Remote models from a worker

herogen.blender's remote client starts `$HEROGEN_MODAL run <runner>.py::main --job ... --out ...`. The Blender
worker image points `HEROGEN_MODAL` at `modal_bridge.py`, which takes the same arguments, calls the runner's `infer`
in its deployed app (the runner's `APP` constant) and writes the files the runner's own `main` writes, so the
client's cache, cost estimate and error handling are unchanged. Stage 3 requests the live hybrid by default through
`HEROGEN_SERVICE_STAGE3_ARGS` (`--mesh-source sam3d-objects --joints sam3d-body --remote live`); it
runs with the pipeline's own defaults.

### Retention

| What | Kept |
| --- | --- |
| Uploads, described heroes' sketches, run folders (images, meshes, records) | `retention_days` (default 30) after the job finishes, then deleted; at once when the user deletes the job or account |
| An uploaded sketch no job claims (a job create the store could not confirm and that was not saved) | A day, then deleted by retention |
| A job's description and name | Cleared with its files |
| A job's row (state, times, cost, attempts, error, commit) | Kept for quota and cost accounting, without files, description or name |
| Model caches (per user: recorded model and remote-model outputs) | `retention_days` after each entry was written; at once when the account is deleted |
| Sign-in links | Deleted a day after they expire or are used |
| Sessions | Deleted when they expire, or a day after they are revoked |
| Audit log | `audit_retention_days` (default 365) |
| Account | Until the user deletes it. The address is then replaced by `deleted:<user-id>` and removed everywhere else: sign-in links, the allowlist, invites bound to it (which are revoked) and any audit entry |

### Settings

Non-secret settings are environment variables read at start (`config.py`; on Modal, baked into the image from the
deploying shell). The main ones:

| Variable | Default | Meaning |
| --- | --- | --- |
| `HEROGEN_SERVICE_APP` | `herogen-service` | Modal app name; prefixes every volume and secret |
| `HEROGEN_SERVICE_MODE` | `dev` | `production` requires https, a real email provider and its key, your own sending domain |
| `HEROGEN_SERVICE_BASE_URL` | `http://127.0.0.1:8800` | The public origin; sign-in links and the origin check use it |
| `HEROGEN_SERVICE_DOMAIN` | | Custom domains for the web function (comma-separated) |
| `HEROGEN_SERVICE_HOSTS` | host of the base URL | Allowed Host headers |
| `HEROGEN_SERVICE_EMAIL_PROVIDER`, `HEROGEN_SERVICE_EMAIL_FROM` | `file`, `HeroGen <signin@example.com>` | `resend` in production; the From address |
| `HEROGEN_SERVICE_TERMS_VERSION` | (empty) | The published terms version users accept; empty closes sign-in |
| `HEROGEN_SERVICE_JOBS_PER_DAY`, `HEROGEN_SERVICE_JOBS_TOTAL` | 5, 50 | Default quota |
| `HEROGEN_SERVICE_MAX_RUNNING`, `HEROGEN_SERVICE_MAX_ATTEMPTS` | 4, 2 | Jobs running at once; attempts per job |
| `HEROGEN_SERVICE_HELDOUT_MAX` | 2 | Held-out view runs at once, service-wide |
| `HEROGEN_SERVICE_RETENTION_DAYS`, `HEROGEN_SERVICE_AUDIT_RETENTION_DAYS` | 30, 365 | Retention |
| `HEROGEN_SERVICE_USER_STORAGE`, `HEROGEN_SERVICE_UPLOAD_MAX_BYTES` | 2 GiB, 12 MiB | Size limits |
| `HEROGEN_SERVICE_PROFILES`, `HEROGEN_SERVICE_PROFILE` | `roblox-ingame,roblox-avatar,best`, `roblox-ingame` | Profiles offered, and the default |
| `HEROGEN_SERVICE_STAGE3_ARGS` | `--mesh-source sam3d-objects --joints sam3d-body --remote live` | Extra `herogen.blender run` options for stage 3 |
| `HEROGEN_SERVICE_IP_HEADER` | (empty) | Header holding the client address behind a proxy |

`config.ENV` lists every variable, including the link and session lifetimes, the rate limits and the stage
timeouts.

| Secret (Modal secret) | Variables | Used by |
| --- | --- | --- |
| `<app>-web` | `HEROGEN_SESSION_KEY` (at least 32 random bytes), `HEROGEN_EMAIL_API_KEY` | `web`, `admin_command` |
| `<app>-model` | `MODEL_API_KEY` | `model_stages` |
| `<app>-admin` | `HEROGEN_ADMIN_KEY` (at least 32 random bytes) | `StoreService`, `admin_command`: the store's admin methods (invites, allowlist, users, quotas, audit) accept calls only with this key, so the web and worker functions cannot reach them |
| `huggingface` | `HF_TOKEN` | the SAM 3D apps, only when stage 3 uses them |

### Deploying

From the repository root, with the secrets and the legal documents in place. Any local use of
`herogen.service.modal_app` (deploy, `admin`, `selfcheck`) refuses to run unless `HEROGEN_SERVICE_MODE=production`
and the other production settings are valid:

```bash
export HEROGEN_SERVICE_MODE=production
export HEROGEN_SERVICE_BASE_URL=https://<your host>
export HEROGEN_SERVICE_EMAIL_PROVIDER=resend
export HEROGEN_SERVICE_EMAIL_FROM="HeroGen <signin@<your sending domain>>"
export HEROGEN_SERVICE_TERMS_VERSION=<version of the published terms>
modal volume put herogen-service-legal terms.md /terms.md
modal volume put herogen-service-legal privacy.md /privacy.md
modal deploy -m herogen.service.modal_app
modal run -m herogen.service.modal_app::selfcheck
modal run -m herogen.service.modal_app::admin --args "invite --count 3"
```

**If the store will not start.** The store refuses to start when its newest snapshot cannot be read (the log names
the file and the reason), when every snapshot is quarantined, or when the Volume holds snapshot files this version
does not recognise. A passing fault clears on the next start. For a file that never becomes readable (a permanent I/O
error, or corruption SQLite reports as an open error such as "unsupported file format"):

```bash
modal run -m herogen.service.modal_app::admin --args "snapshots"
modal run -m herogen.service.modal_app::admin --args "snapshot-quarantine meta-00000004-<id>-0000000123.db"
```

`snapshot-quarantine` first takes a newer lease (a store still running stops writing), then renames the file to
`quarantine-<id>-<name>`, so the next store starts from the snapshot before it; the writes only that file held are
lost. Nothing is deleted. It refuses to quarantine the last snapshot unless `--allow-empty` is added, which lets the
next store start from an empty database (every account, invite and job record is then gone). Both commands work on
the `<app>-meta` volume directly, so they run while the store is down, and every use is logged.

Staging is the same with `HEROGEN_SERVICE_APP=herogen-service-staging` (its own secrets
`herogen-service-staging-web`, `-model` and `-admin`, and its own volumes) and its own base URL. Without a custom domain the
base URL is the address Modal prints for the `web` function; set it and deploy again.

### Body acceptance measurements

`herogen.blender.body_quality` supplies two measurements without changing stage 3's gate:

- `cross_sections` intersects the mesh at 35%, 45% and 55% of its height. Each score is the XY convex hull's
  area divided by its axis-aligned bounding rectangle's area; their mean is `rectangularity`. A rectangle
  scores 1 and a finely tessellated ellipse approaches pi/4. Missing or degenerate sections return no score.
  This descriptor measures cross-section shape independently of observed-view silhouette agreement; it
  is not a universal anatomical rule.
- `visual_score` asks Muse Spark 1.3 whether the body is a usable base for the sketch. The frozen prompt's
  SHA-256 is `8e054e5d6ba4b8efb4316878ed01c4f9c33eb3d889b5d6edc91f0781ad1ad38c`.
  A 1280x688 sheet shows the sketch followed by front, left, back, right, +30 and -30 degree grey views.
  The two oblique views are not construction or registration views. Grey renders use smooth two-sided
  Lambert shading, an orthographic scale shared across all views, 512px tiles, white background, no
  simplification and no per-view stretching. The scale fits the widest cardinal/30/60 degree direction.
  The strict schema returns integer `usable_probability` (0–100) and a nonempty `reason`. The call uses
  `ModelClient.compare_images`, low reasoning effort, 1200 maximum output tokens, 1280 maximum image
  side and JPEG quality 90. The client's cache, retry policy, attempt ledger and token prices apply.
  Model errors, missing views, invalid responses and a changed prompt hash raise; none supplies a pass.

Neither measurement alone asserts topology, rigging or export quality. Acceptance thresholds belong to
an explicitly calibrated policy, separate from these measurement functions.
