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

A stage that stops before its checks are complete writes its record with `"ok": false`, a `status` and an
`error`, plus every call made so far: `"error"` (the Model API failed after the client's retries, or any other
error) or `"input_rejected"` (stage 1 found no character in the sketch). `python3 -m herogen.pipeline run`
exits 0 when every stage passed, 2 when a gate failed, 3 when the input was rejected and 1 on an error.

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

## Held-out views, mask method and covers

**Held-out views (stage 2h).** After stage 2 passes, four extra images of the same character in the
turnaround's A-pose, each its own edit of the cover: `front-left-30`, `front-right-30`, `front-left-60` and
`front-right-60`, named by their target yaw (front-left views face image left). No building stage reads
`heldout/`; the views exist so a body can be scored on views it was not built from. Each generation sends the
cover, a pose reference (the front view, or the nearer profile for turns of 50 degrees or more) and a grey
mannequin of the turnaround's pose drawn at the requested yaw, both at one shared scale.

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
  cover, so a sketch SAM finds nothing in costs no edit, and its record has status `"input_rejected"` and the
  error "no character found in the sketch". Nothing found in a generated image fails that image's gate: the
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
and `"top_method"`.

**Shoulder covers.** Stage 4 adds one cover per shoulder, merged into its owner section (`UpperTorso`) so each
GLB keeps exactly 15 meshes. `sections.json` lists them under `"covers"`: the owner, the shaping parameters, and
`pieces_high` / `pieces_low` with each piece's triangles. Covers count in the triangle budgets.

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
