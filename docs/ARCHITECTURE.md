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
