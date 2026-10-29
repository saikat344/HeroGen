# HeroGen

HeroGen turns one hand-drawn character sketch into a game-ready hero body for Roblox, and checks every stage
with measurements.

```
sketch → cover art → turnaround (front, back, left, right) → 3D mesh + joints → 15-section R15 body → results
```

## Stages

| Stage | What it does | Models and tools |
| --- | --- | --- |
| 0 input | Copies the sketch into a run folder | |
| 1 cover art | A clean colour rendering of the same character | Meta Muse Image (`muse-image-1.0`), SAM 3.1 |
| 2 turnaround | Four orthographic A-pose views in one generation, split by code, checked for agreement; one held-out 3/4 view for scoring only | Muse Image, SAM 3.1 |
| 3 base mesh | Carves a mesh from the four view masks and checks its silhouettes against them | Blender 4.5 LTS (headless) |
| 4 game body | Finds the joints from the silhouettes, cuts the mesh into the 15 R15 sections with pivots, shoulder covers, high and low LODs within triangle budgets, turnaround colours baked into one atlas per LOD, and a pose test that measures visible joint gaps | Blender |
| 5 results | A static results page with every stage's images, checks, time and cost | code |
| versions | `python3 -m herogen.versions` freezes a run's body and its records as a write-once, hashed version, then lists, verifies and compares versions (per-view scores on seen and held-out views, triangles, gates, joint gaps) | code |

Every stage writes `stage-<n>.json` with its measured checks and stops the run when a gate fails. Stages 3 and 4
refuse to start unless the previous stage passed (`--force` overrides). The file formats, cameras and checks are
in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Requirements

- Python 3.11 or newer, with `numpy` and `pillow`.
- Blender 4.5 LTS for stages 3 and 4 (set `HEROGEN_BLENDER` if it isn't at the default macOS path or on `PATH`).
- For live model calls in stages 1 and 2: a Meta Model API key in `MODEL_API_KEY` (see `.env.example`), and
  `meta-sam-parser` to decode SAM 3.1 masks (`pip install ".[live]"`).

## Running

```bash
python3 -m herogen.pipeline run sketch.png --run-id my-hero --live
python3 -m herogen.blender run --run runs/my-hero --stages 3,4
python3 -m herogen.pipeline run sketch.png --run-id my-hero --stages 5
```

Without `--live`, model calls replay from the cache, which is how the tests run.

## Lab

```bash
python3 -m herogen.lab serve --runs runs --port 8790     # then open http://127.0.0.1:8790/
python3 -m herogen.lab serve --runs runs --runs other/runs --versions versions
```

A local, read-only page for the runs of one character at a time. `--runs` can be given more than once (default
`$HEROGEN_RUNS` or `./runs`); `--versions` names a folder of frozen versions (default `$HEROGEN_VERSIONS`, none
when unset). Runs are named `<root>/<run>`, the root being `r1`, `r2`, ... in the order given.

- **Characters and profiles.** Runs are grouped by the SHA-256 of their sketch file (`00-sketch.png`; the stage
  records do not hold a sketch hash, so the lab hashes the file), then by the profile their body records (best
  quality, Roblox in-game, Roblox avatar, or not recorded). The run picker lists a profile's runs newest first;
  the default is the newest run with a body and scores.
- **Pipeline.** Sketch, artwork, turnaround views, held-out views and the 3D body, each with its stage's status
  and gate count (a stage with no gates never shows as passed); clicking one shows it in the viewer.
- **3D viewer.** The run's `04-body/high.glb` or `low.glb`: textured, clay, wireframe or per-section colours; the
  seven test poses of `herogen/blender/pose.py`, each section turned about its pivot; click a section, or pick it
  from the list or the legend, to isolate it and see its triangles, pivot and joint; front, 3/4, side and back
  cameras, orbit, and an axis gizmo. "Side by side" puts the body beside a turnaround view through that view's own
  orthographic camera (its pixels per stud from `views/meta.json`, framed on the view's mask), or beside the
  sketch or artwork, scaled to the figure there (approximate).
- **Quality.** Every check of stages 0 to 4 in plain words, grouped into artwork, views, body, texture and
  held-out, with the failing gates first. "Show me" turns the body to the worst joint gap's pose and isolates the
  joint (from the pose test's gap table), puts a texture failure's view beside the body, or opens the held-out
  view a check names.
- **Held-out views.** Accepted, needs-review and rejected views as `stage-2h.json` records them, with the review
  decision and the yaw the checks found, and the body rendered here at that yaw; older single-view records show
  as passed or failed checks. A run without its own record shows the record of a run of the same sketch whose
  turnaround views are byte-identical; the header, the pipeline and the quality panel name that run, since its
  gates count in the status. Scores against a held-out view come from `05-scores.json` when it holds them.
- **Deliverables**, with their sizes, triangle counts (counted from the GLB) and a copy-path button, and what the
  run does not build, with the reason its files give. **Generation details**: time, estimated cost, models and
  model calls per stage, from the stage records; a held-out stage borrowed from another run is listed but kept
  out of this run's total.
- **Versions and compare.** Frozen versions (with a SHA-256 check of their files) and every body of the character;
  pick two to see them side by side with one camera and their scores per view, with the change. Versions that
  belong to no character here, or whose `version.json` cannot be read, are listed apart with the reason.
- Dark, light and automatic themes; keyboard: `1`-`4` cameras, `O` orbit, arrow keys and `+`/`-` in the 3D view,
  `Esc` to show every section.

A file the page cannot read is named in a banner and the rest of the run is still shown; a run folder that cannot
be listed stays in the list as a run that can't be read. The server binds to 127.0.0.1, checks the Host header,
uses the standard library only, and serves PNG, JPEG, GLB and JSON files from inside their own run or version
folder only. The page loads three.js 0.160.0 from jsDelivr through an import map pinned by SHA-384 and makes no
other external requests; its fonts are served by the lab itself.

The page's icons are Lucide 1.64.0 (ISC; some derive from Feather, MIT; drawn here at stroke width 1.5) and
HeroGen's own; the wordmark is outlined from Inter (SIL Open Font License 1.1). Its type is Bricolage Grotesque and
Figtree (variable TrueType from google/fonts) and JetBrains Mono (WOFF2 from its v2.304 release), all SIL Open Font
License 1.1, vendored in `herogen/lab/static/fonts/` with their licences. Sources, versions and SHA-256 sums are in
`herogen/lab/static/licenses/THIRD-PARTY-NOTICES.txt`.

## Tests

```bash
python3 -m unittest discover -s tests -v
python3 tools/guard/guard.py tree
```

Tests use synthetic inputs drawn by code and never call a model. Blender tests skip when Blender isn't installed.

## Third-party models and libraries

Muse Image and SAM 3.1 are called through Meta's Model API. `meta-sam-parser`, an optional dependency used to
decode SAM 3.1 output, is released by Meta under the SAM License. Blender is GPL-licensed and runs as a separate
program.

## Licence

HeroGen's code is licensed under the [Apache License 2.0](LICENSE). The models and libraries it calls keep their
own licences.
