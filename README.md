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
| 2 turnaround | Four orthographic A-pose views in one generation, split by code, checked for agreement; then four held-out turned views (30 and 60 degrees each side) for scoring only, each accepted, sent to human review or rejected | Muse Image, SAM 3.1 |
| 3 base mesh | Carves a mesh from the four view masks (default), or builds it with SAM 3D Objects from the front view (`--mesh-source sam3d-objects`, which today fails the side views' silhouette gate); finds the joints from the silhouettes (default) or with SAM 3D Body (`--joints sam3d-body`); checks the mesh's silhouettes against every observed view's mask | Blender 4.5 LTS (headless); optionally SAM 3D Objects and SAM 3D Body on Modal |
| 4 game body | Cuts the mesh at the joints into the 15 R15 sections with pivots and shoulder covers, at the detail a target profile asks for (default `best`: full detail, no triangle budget; `roblox-ingame`: high and low LODs within 24,000 / 6,000 triangles; `roblox-avatar`: one body within the triangle, watertight-part and texture limits of Roblox's character body specification, without its rig, attachments, facial animation or cages), bakes the turnaround colours into one atlas per LOD (a SAM 3D Objects mesh's own texture where no view faces the surface), and runs a pose test that measures visible joint gaps | Blender |
| scores (after 4) | Scores every body, ours or another tool's GLB, on the turnaround views and every held-out view: silhouette IoU, colour error, SSIM, mesh sanity, and optionally a platform's limits | numpy (`herogen.blender.score`) |
| 5 results | A static results page with every stage's images, checks, scores, time and cost | code |
| versions | `python3 -m herogen.versions` freezes a run's body and its records as a write-once, hashed version, then lists, verifies and compares versions (per-view scores on seen and held-out views, triangles, gates, joint gaps) | code |
| 6 Roblox export | An R15 rig model for Roblox Studio, the scripts that put the hero's meshes on it, and step-by-step import instructions (`python3 -m herogen.roblox export --body <run>/04-body --out <run>/06-roblox`) | code |

Every stage writes `stage-<n>.json` with its measured checks and stops the run when a gate fails. Stages 3 and 4
refuse to start unless the previous stage passed (`--force` overrides). The file formats, cameras and checks are
in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Requirements

- Python 3.11 or newer, with `numpy` and `pillow`.
- Blender 4.5 LTS for stages 3 and 4 (set `HEROGEN_BLENDER` if it isn't at the default macOS path or on `PATH`).
- For live model calls in stages 1 and 2: a Meta Model API key in `MODEL_API_KEY` (see `.env.example`), and
  `meta-sam-parser` to decode SAM 3.1 masks (`pip install ".[live]"`).
- For the SAM 3D sources of stage 3 (`--mesh-source sam3d-objects`, `--joints sam3d-body`): `pip install
  ".[remote]"` (`modal`, and `scipy`, which replaying them needs too). Live calls also need:
  - a [Modal](https://modal.com) account, set up with `modal setup`;
  - access granted on Hugging Face to the gated models `facebook/sam-3d-objects` and
    `facebook/sam-3d-body-dinov3` (request it on each model page);
  - a Modal secret named `huggingface` holding a Hugging Face token with that access:
    `modal secret create huggingface HF_TOKEN=<token>`. The token is read only inside the Modal containers;
    HeroGen never reads, prints or stores it;
  - the SAM 3D Objects checkpoints in their Modal volume, downloaded once with
    `modal run herogen/blender/remote/objects.py::download` (SAM 3D Body downloads its own on first use);
  - the SAM 3D Objects image, built once with `modal run herogen/blender/remote/objects.py::build` (on a CPU;
    it compiles CUDA extensions: 13 minutes measured when only the layers from the requirements on were
    rebuilt, longer from scratch); otherwise the first live call spends that time building it and needs a longer
    `--remote-timeout`. The SAM 3D Body image builds in about 5 minutes, within the default timeout.
  Volumes are `herogen-sam3d-objects-ckpt` and `herogen-sam3d-body-ckpt` unless `HEROGEN_OBJECTS_VOLUME` or
  `HEROGEN_BODY_VOLUME` names another. SAM 3D Objects runs on an A100-40GB by default (`--objects-gpu`; it needs
  about 23 GB), SAM 3D Body on an A10G (`--body-gpu`, `none` for CPU); `--remote-timeout` bounds each call.
  Every call is an ephemeral Modal app that stops when the call returns.

## Running

```bash
python3 -m herogen.pipeline run sketch.png --run-id my-hero --live
python3 -m herogen.blender run --run runs/my-hero --stages 3,4
python3 -m herogen.blender score --run runs/my-hero
python3 -m herogen.pipeline run sketch.png --run-id my-hero --stages 5
```

Stage 4 builds the best-quality body by default, with no triangle budget. For Roblox, choose a profile:
`--profile roblox-ingame` (a high and a low LOD within 24,000 and 6,000 triangles) or `--profile roblox-avatar`
(one body within the triangle, watertight-part and texture limits of Roblox's character body specification:
10,742 triangles split across the head, torso, arms and legs, watertight parts, textures up to 2048 px). The rest
of the specification is not built: the R15 bone rig with skinning, attachments, facial animation (DynamicHead)
and layered-clothing cages, so the body is not yet a complete avatar body. Budget gates apply only under a Roblox
profile; see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#target-profiles).

The hybrid body (SAM 3D Objects shape, SAM 3D Body joints) instead of the carve. Its shape comes from the front
view alone and does not yet match the side views: on the pencil-sketch run its silhouette IoU is 0.688 left and
0.691 right against the 0.90 gate (front 0.965, back 0.944), so stage 3 fails and stage 4 needs `--force`:

```bash
python3 -m herogen.blender run --run runs/my-hero --stages 3 --mesh-source sam3d-objects \
  --joints sam3d-body --remote live
python3 -m herogen.blender run --run runs/my-hero --stages 4 --force
```

Without `--live` (stages 1 and 2) or `--remote live` (stage 3), model calls replay from the cache
(`runs/_cache-remote/` for the SAM 3D calls), which is how the tests run; a remote source with no cached result
falls back to the carve or the silhouette joints and says so in `stage-3.json` (`--no-fallback` makes it an
error).

The scorer takes any GLB, so bodies made by other tools from the same sketch are scored on the same views,
including the held-out ones:

```bash
python3 -m herogen.blender score --run runs/my-hero --body other=other-tool.glb
```

`--body` may be given any number of times and point anywhere, so earlier milestones' frozen bodies are scored
against today's on the same views; a `version.json` next to a GLB is recorded with its score (its `label`), and
every stage record and score file records the code version (`git` commit and whether it was modified) that
made it.

With no `--body` it scores every stage 4 body in the run. It applies no triangle budget by default; add
`--profile roblox-ingame` or `--profile roblox-avatar` to check a Roblox profile's limits.

The hosted, invite-only service (sign-in by emailed link, per-user quotas, GLB and Roblox downloads) is `herogen/service`; `python3 -m herogen.service dev` runs it locally (docs/ARCHITECTURE.md, "Service").

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
pip install -r requirements-test.txt
python3 -m unittest discover -s tests -v
python3 tools/guard/guard.py tree
```

Tests use synthetic inputs drawn by code and never call a model. `requirements-test.txt` pins every pip package
the tests use. Three tools are not pip packages, and the tests that need them skip when they are missing:

- **Blender** (stages 3 and 4, and the benchmark renders), found as described under Requirements.
- **luau-compile**, an optional syntax check of the Roblox scripts, when it is on `PATH`.
- **Lune**, an optional cross-check of the Roblox model with rbx-dom, when it is on `PATH`.

CI installs `requirements-test.txt` but none of the three, so those tests skip there; run them locally before
changing the code they cover. CI prints every skipped test and its reason after the test run.

The service page's browser tests (`tests/test_herogen_service_e2e.py`) drive a headless Chrome over the DevTools
protocol and skip unless `E2E_CDP_PORT` names its remote-debugging port; the module's docstring shows how to start
one. They also need uvicorn (`herogen/service/requirements-dev.txt`). CI installs neither uvicorn nor a browser,
so they skip there; run them locally before changing the service page.

The hybrid mesh tests need `scipy`, which `requirements-test.txt` includes. The pytorch3d rasteriser's test
needs `torch` and `pytorch3d`, which only the SAM 3D Objects image installs, so it skips everywhere else.

## Releases

Tags deploy the hosted service (`herogen/service`) to Modal through `.github/workflows/release.yml`: `vX.Y.Z-rc.N`
to staging, then `vX.Y.Z` on the same commit to production, each after the tests pass. Any other tag starting with
`v` fails. Deploying needs the service code in the tagged commit. The rules, the secrets and variables each GitHub
environment needs, and how to roll back are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#releases).

## Third-party models and libraries

Muse Image and SAM 3.1 are called through Meta's Model API. `meta-sam-parser`, an optional dependency used to
decode SAM 3.1 output, is released by Meta under the SAM License. Blender is GPL-licensed and runs as a separate
program.

SAM 3D Objects and SAM 3D Body (code and weights) are released by Meta under the SAM License. HeroGen copies
none of their code: the Modal images clone each repository at a pinned commit, the weights are downloaded by
users who have been granted access, and the few changes the runs need are small self-written substitutions
made inside the container (`herogen/blender/remote/`). Libraries whose licences allow only non-commercial use are
never installed, and each image's build fails if one can be imported:
- nvdiffrast, which SAM 3D Objects uses for texture baking: the bake runs with SAM 3D's `pytorch3d` rendering
  engine and a self-written pytorch3d rasteriser;
- `diff_gaussian_rasterization` (the Inria splat rasteriser, SAM 3D's default splat backend): every splat render
  uses gsplat (Apache-2.0) instead;
- `smplx` (licensed for non-commercial scientific research), listed in SAM 3D Objects' requirements but never
  imported by its inference: it is filtered out of the install.
The weights are downloaded at pinned Hugging Face revisions, recorded with every result. MoGe (MIT), the depth and
field-of-view model both use, is installed at a pinned commit and its weights at pinned revisions. SAM 3D Body's
DINOv3 backbone code (DINOv3 License) is cloned at a pinned commit too, instead of being fetched from GitHub's main
branch at run time. Two copyleft
libraries run inside the Objects image: `pymeshfix` and `igraph` (GPL), which SAM 3D Objects' mesh
post-processing imports; they run unmodified in the Modal container and are not distributed with HeroGen.
`point-cloud-utils` (GPL) and `pdoc3` (AGPL), which its requirements list but its code never imports, are not
installed. Licence flag: SAM 3D Body's optional person detector, ViTDet (`--body-detector vitdet`, off by
default; detectron2 is Apache-2.0), downloads weights released under CC BY-SA 3.0, which allows commercial use
with attribution and requires adaptations of them to be shared under the same licence. By default stage 3
gives SAM 3D Body the bounding box of stage 2's front mask instead, and those weights are never downloaded.
The other libraries the images install keep their own licences.

## Licence

HeroGen's code is licensed under the [Apache License 2.0](LICENSE). The models and libraries it calls keep their
own licences.
