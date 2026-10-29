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

Importing the body into a game engine and play-testing it are not part of this pipeline.

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

## Versions

A version is one stage 4 body frozen with the records that built and measured it, so a later build can be
compared with it on the same sketch. `herogen/versions` writes, lists and verifies them:

```bash
python3 -m herogen.versions freeze --run runs/my-hero --label m1-change [--body 04-body] [--sketch S] \
    [--milestone M] [--note TEXT] [--by NAME] [--allow-dirty] [--allow-unknown-code] [--store DIR]
python3 -m herogen.versions list [--sketch S] [--json] [--store DIR]
python3 -m herogen.versions verify --version V [--json] [--store DIR]
```

The store is `--store` or `$HEROGEN_VERSIONS`. There is no default, and the folder must already exist. A version
(`V`) is given as a version folder, as `<sketch>/<label>` in the store, or as a label that only one
sketch folder holds. A folder whose name starts with `.` (such as an unfinished `.partial-*` freeze) is never
a version and is refused however it is given. Exit status: 0 done, 1 `verify` found a problem, 2 refused or
failed. A refusal or failure prints one line on stderr and never a traceback.

**Layout.** Versions are never kept in git.

```
<store>/<sketch>/<label>/                      one version: flat files plus version.json, read-only
<store>/<sketch>/05-scores-<series>.json       scores collected by hand for format 0 versions
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
