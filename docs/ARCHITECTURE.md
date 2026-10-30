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
