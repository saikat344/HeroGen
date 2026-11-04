# HeroGen

One sketch or one sentence becomes a finished, game-ready 3D hero.

## The problem

An indie game developer needs 3D characters that match what they imagined, and that ship at a polygon budget a
game can play. HeroGen goes from the idea to a textured, rigged Roblox R15 body within that budget, and records a
measured check at every step.

## Try it in 2 minutes

Live product (link coming):

<!-- LIVE_LINK -->

3-minute demo video (link coming):

<!-- VIDEO_LINK -->

1. Open the live link. The gallery of earlier heroes loads with no sign-in.
2. Open a gallery hero: orbit its 3D model, switch between textured, clay and wireframe, and compare it with its
   starting sketch.
3. To make your own, type a one-sentence character idea (up to 300 characters) or upload a sketch, then press
   Create. A guest workspace opens in your browser: no email, no account.
4. Watch the build log: each model call, each check with its measured value and limit, time and estimated cost.
5. When it is done, play the Idle, Walk, Wave and Jump previews and download the GLB or the Roblox bundle.

The demo stays up through judging, Nov 5–8, and the finals on Nov 9.

## Built during the hackathon, and what came before

| | When | What it is here |
| --- | --- | --- |
| HeroGen (this repository) | Built during the event, Oct 26 – Nov 4, 2026 | The whole product: pipeline, checks, website, Roblox export |
| Z/BURST, its game and its FloodYard environment | Existed before the event | Only a stage to show finished heroes in. Its hero contract supplies the targets HeroGen aims at: 15 rigid R15 parts, 24,000 / 6,000 triangles, 1024 / 512 px maps and its pose set |

## Public access, no secrets

- **Public gallery.** Earlier heroes are served read-only from a published gallery, with no sign-in, under the
  label "Previously generated with HeroGen" (`herogen/service/gallery.py`, `/api/gallery`).
- **Guest workspace.** Opening one needs no email or code. Opening it alone never starts compute
  (`GenerationStore.guest_open` in `herogen/service/generation.py`).

Live generation is limited in code (`herogen/service/store.py`, `generation.py`, `web.py`, `config.py`):

- **Checked before any model or GPU work.** The spend guard runs inside the transaction that creates the job,
  before it is queued (`Store.job_create`, `_guard_generation`). When the operator turns spend limits on
  (`HEROGEN_SERVICE_SPEND_LIMITS`), a job is refused unless
  `spent + (queued and running jobs + 1) x running estimate <= limit`; the texture refine is then also capped
  per hero (`HEROGEN_SERVICE_REFINE_CAP_USD`, default $0.80).
- **A global pause switch** refuses new heroes with "Live generation is paused right now."
- **Global concurrency.** At most `max_running` heroes run at once across all workspaces (default 5, admin range
  1 to 32; `Store.jobs_to_dispatch`). The deployed SAM 3D apps are fixed at 5 containers each
  (`herogen/blender/remote/deploy.py`).
- **Explicit states, no silent wait.** A paused service or a spent budget answers HTTP 503 with
  `generation_paused` or `generation_budget_exhausted` and a plain sentence. A hero waiting for a free worker
  shows its place in line ("2 heroes ahead of you"), from `queue_position` and `queued_ahead`.
- **Per-workspace limits.** 20 new heroes per hour per workspace and 2 GB of saved heroes. Uploads are at most
  12 MB and are resized to 2048 px on the long side.

## Meta models used at runtime

| Model | What it does | Stages | Where it runs |
| --- | --- | --- | --- |
| Muse Image (`muse-image-1.0`) | Draws the sketch from an idea, the colour cover, the four-view turnaround sheet, face close-ups, held-out turned views, and the texture repaints | p, 1, 2, 2f, 2h, 3f | Meta Model API |
| SAM 3.1 (`sam-3.1`) | Cuts out the figure in every image; finds the head, face, hair and feet for the outline and face checks and the texture work | p, 1, 2, 2f, 2h, 3f | Meta Model API |
| Muse Spark (`muse-spark-1.3`) | Text and image questions only: the brief pre-check and the content-policy reword (website), the same-character check of a drawn sketch, the real-person photo check on uploads, the Refine suggestions after 3f, and the body usability question in stage 3 (command line; the website skips it) | p, 1, 3, after 3f | Meta Model API |
| SAM 3D Objects | The 3D shape and its texture, from the front view and its mask | 3 | Modal GPU, A100-40GB |
| SAM 3D Body | The skeleton: named joints and a body mesh, from the front view | 3 | Modal GPU, A10G |
| Sapiens 2 (`sapiens2-pose-1b`) | Pose keypoints that measure the yaw of the held-out views | 2h (command line only; the website does not run 2h) | Modal GPU, L4 |

The model ids are constants in `herogen/pipeline/client.py` (`IMAGE_MODEL`, `SAM_MODEL`, `TEXT_MODEL`) and in the
frozen prompts in `herogen/prompts/versions/`. The Muse Spark calls are in `herogen/pipeline/reword.py`,
`suggest.py`, `prompt_sketch.py`, `inputs.py` and `herogen/blender/body_quality.py`.

- Every model HeroGen itself calls is a Meta model. (The SAM 3D pipelines' own sub-models, as their upstream
  repositories ship them, are listed under [Third-party models and libraries](#third-party-models-and-libraries).)
- Plain code is not a model call: OpenCV, NumPy, SciPy and Blender geometry do the splitting, cutting,
  decimation, texture baking, scoring and Roblox export.
- Browser features are not model calls: the website's 3D viewer and its animation previews run three.js on the
  pivots stage 4 wrote.
- No model is fine-tuned. Every weight is the upstream release, downloaded at a pinned revision.

## Open-weight models

SAM 3D Objects and SAM 3D Body are open-weight models (SAM License, gated on Hugging Face). HeroGen runs them on
Modal GPUs as persistent deployed apps (`python -m herogen.blender.remote.deploy`): SAM 3D Objects on an
A100-40GB, SAM 3D Body on an A10G. Each app scales to zero after 300 idle seconds and runs at most 5 containers.

| Model | Code commit | Weight revision (Hugging Face) | Runner |
| --- | --- | --- | --- |
| SAM 3D Objects | `f91db411c50efee93d8db7aeb323885650f6f722` | `facebook/sam-3d-objects` `2e73555018d2741ccd486e56c24fac41155a1dc6` | `herogen/blender/remote/objects.py` |
| SAM 3D Body | `b5c765a0d89d789985e186d396315e7590887b94` | `facebook/sam-3d-body-dinov3` `11aaa346c7204874a1cbafe3d39a979080b2c55a` | `herogen/blender/remote/body.py` |

How the pipeline uses them in stage 3: SAM 3D Objects turns the front turnaround view and its SAM 3.1 mask into a
textured mesh. SAM 3D Body reads the same view, inside the front mask's bounding box, and returns the joints.
HeroGen fits the mesh to the turnaround views, places the joints in it, and stage 4 cuts the mesh at those
joints into the 15 R15 parts. Every call is cached by its inputs, runner source and pinned revisions, so a
replay returns exactly what the live call returned. Sapiens 2 is open-weight too (`facebook/sapiens2-pose-1b`,
revision `f5fed8b97b99698d5eea1d14ff0855d0b4c3f000`, `herogen/pipeline/sapiens_app.py`).

## Game-ready evidence: one real run

Job `j_2brhf76zy375t3gy4rt5tjx7dn`, "Crimson Samurai", profile `roblox-ingame`. It is a turnaround redo of an
uploaded sketch, with the note "Make the armour lacquered crimson red and black, with a red silk sash." The
sketch and cover were reused from the earlier version; stages 2 to 6 ran in this job. Every number below is read
from its stage records.

| Measure | Value | Record |
| --- | --- | --- |
| Shape (stage 3) | SAM 3D Objects returned 9,466 triangles; HeroGen's internal watertight remesh of it has 211,112 triangles, 0 open edges and 0 non-manifold edges, and is what stage 4 cuts and decimates | `stage-3.json` (`mesh.source.remote.source_triangles`, `mesh.triangles`) |
| Texture refine | 6 of 6 repainted views accepted, 4096 px atlas | `stage-3f.json` |
| R15 cut | 15 of 15 parts, high and low | `stage-4.json` |
| Roblox high LOD | 23,798 triangles (budget 24,000), 1024 px body atlas | `stage-4.json` |
| Roblox low LOD | 5,714 triangles (budget 6,000), 512 px body atlas | `stage-4.json` |
| Face detail | The Head has its own texture, 1024 px on high and 512 px on low, separate from the body atlas, taking the refined colour | `stage-4.json`, `HEAD_TEXTURE` in `herogen/blender/stage4.py` |
| Stage 4 checks | 8 of 8 gates passed; 33 of 53 checks passed in all | `stage-4.json` |
| Pose test | 20 gate poses (rest included) plus 8 held-out poses. Gap method: each cut's shared rim is carried by both parts, and the gap is how far a rim point ends up outside the other part's surface. Limit 0.15 studs. Worst gap 0.534 studs (high) and 0.493 (low) on the gate poses, 0.516 / 0.502 on the held-out poses. **Not passed**: on the website the gap is recorded, not a gate, and the record notes visible seams at 0.53 studs at the widest. | `stage-4.json` |
| Roblox rig checks | 42 of 42 gates passed (43 of 43 checks): R15 names, Motor6D joints, attachments at the pivots, rest pose, a starter place | `stage-6.json` |
| Time | 803 s (13 min 23 s), turnaround to Roblox files | `events.jsonl` |
| Model and GPU cost | est. $0.494: stage 2 $0.0175, stage 3 $0.1458 (Modal GPU time), stage 3f $0.3275, Refine suggestions $0.0032 | stage records, `refine-suggestions.json` |

No held-out 30° / 60° view score was recorded for this run, so none is given.

## Originality

On the website, a brief is checked before anything is drawn (`herogen/pipeline/reword.py`, prompts
`brief-precheck-v2` and `brief-reword-v2`). One Muse Spark call lists any brand, logo, franchise, trademarked
character or real-person names, and any held items, pets or companions. When there are any, the brief is
rewritten once: names become a plain archetype, and held items, pets and companions are left out. Worn clothing
stays. A rewrite that still contains a removed name or item is refused, never repaired. The same rewrite runs
once if Muse Image refuses a brief under its content policy. The sketch prompts also ask for empty hands and no
pet or companion, and stage 1 rejects uploads that are photographs of real people.

## Run it locally

Two ways to run HeroGen on your own computer. The first needs no accounts and costs nothing; the second makes real
heroes and uses paid services.

**Supported systems.** macOS (Apple silicon, used for development) and Linux. Windows is untested.

### 1. Get the code and a Python environment

Python 3.11 or newer.

```bash
git clone https://github.com/saikat344/HeroGen.git
cd HeroGen
python3 -m venv .venv
source .venv/bin/activate
pip install ".[live,remote,refine]"
pip install -r herogen/service/requirements-dev.txt
```

The commands below use `python`, the virtual environment's interpreter. On Linux, also run
`pip install "opencv-python-headless==4.11.0.86"`: the Linux builds of OpenCV 5 cannot read the EXR renders texture
refine uses, while the macOS build can.

### 2. Try the website with no accounts (free)

```bash
python -m herogen.service dev --port 8793 --data service-data
```

Open `http://127.0.0.1:8793/`. This uses the default `--runner fake`: every step finishes instantly with stand-in
results and no model is called, so you can click through the whole site. `--data` is the folder the site keeps its
heroes and database in; it is created if it does not exist (default `service-data`).

### 3. Make real heroes (paid)

A real hero runs the pipeline and Blender on your computer and calls two paid services: the **Meta Model API**
(Muse Image, SAM 3.1, Muse Spark) and **Modal** GPUs for SAM 3D Objects (an A100-40GB) and SAM 3D Body (an A10G).
Expect about **$0.50 of model and GPU calls per hero** (est.; see [Game-ready evidence](#game-ready-evidence-one-real-run)),
and about 10–15 minutes.

1. **Blender 4.5 LTS** (4.5.1 used here), run headless. HeroGen looks for
   `/Applications/Blender.app/Contents/MacOS/Blender` on macOS, then `blender` on `PATH`. Otherwise set
   `HEROGEN_BLENDER` to the executable, for example `export HEROGEN_BLENDER=/opt/blender/blender` on Linux.
2. **Meta Model API key.** Export it in the shell that runs HeroGen: `export MODEL_API_KEY=<your key>`. HeroGen reads
   environment variables only; it does not load a `.env` file. `.env.example` lists the variables it reads.
3. **Hugging Face access.** Create a read token at <https://huggingface.co/settings/tokens>, and request access to the
   gated models `facebook/sam-3d-objects` and `facebook/sam-3d-body-dinov3` on their model pages. Approval is
   granted by the model owners and may not be immediate; the next steps need it.
4. **Modal.** Create an account, then sign in and store your Hugging Face token as a Modal secret:
   ```bash
   modal setup
   modal secret create huggingface HF_TOKEN=<your Hugging Face token>
   ```
5. **One time: build the SAM 3D Objects image, then download its checkpoints** (in this order; the build compiles
   CUDA extensions and takes 15 minutes or more):
   ```bash
   modal run herogen/blender/remote/objects.py::build
   modal run herogen/blender/remote/objects.py::download
   ```
6. **Deploy both GPU runners** as persistent Modal apps and point local runs at them:
   ```bash
   python -m herogen.blender.remote.deploy
   export HEROGEN_REMOTE_DEPLOYED=1
   ```
   Without `HEROGEN_REMOTE_DEPLOYED=1`, each call starts a temporary `modal run` app instead, which pays the GPU
   start-up every time. Either way the `modal` command must be on `PATH` (it is, inside the virtual environment), or
   named by `HEROGEN_MODAL`.
7. **Start the site with the real stages** and open `http://127.0.0.1:8793/`:
   ```bash
   python -m herogen.service dev --port 8793 --data service-data --runner local
   ```
   Describe a hero or upload a sketch, then press Create. The build log shows each model call, its time and its
   estimated cost. A finished hero has a textured 3D model, Idle/Walk/Wave/Jump previews and, when every Roblox
   check passes, a Roblox download.

**Or one hero from the command line**, without the website (same requirements as above):

```bash
python -m herogen.pipeline run --prompt "a knight in battered plate armour" --run-id my-hero --live
python -m herogen.blender run --run runs/my-hero --stages 3,4 --remote live --refine live --profile roblox-ingame
python -m herogen.roblox export --body runs/my-hero/04-body --out runs/my-hero/06-roblox
```

The first command writes the sketch, cover and turnaround views to `runs/my-hero/`; the second builds the 3D shape,
refines its texture and cuts the Roblox body into `runs/my-hero/04-body/`; the third writes the Roblox files to
`runs/my-hero/06-roblox/`. [Running](#running) has every option.

### 4. Run the tests

```bash
pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python tools/guard/guard.py tree
```

They use synthetic inputs and never call a model. Tests that need an optional tool (Blender, Node, a headless
Chrome for the browser tests) skip when it is missing; [Tests](#tests) lists them.

## How it works

```
(text prompt → full-body sketch) → sketch → cover art → turnaround (front, back, left, right) → 3D mesh + joints
→ 15-section rigid R15 body → results
```

### Stages

| Stage | What it does | Models and tools |
| --- | --- | --- |
| p prompt sketch (optional) | A hand-drawn graphite design image from the user's character brief, then a full-body A-pose edit of it; checked for one figure, a full figure and the same character, with up to three attempts | Muse Image, SAM 3.1, Muse Spark (`muse-spark-1.3`) |
| 0 input | Copies the sketch (or stage p's full-body image) into a run folder; input checks here and at the start of stage 1 reject an unusable sketch before any image is generated, or warn | |
| 1 cover art | A clean colour rendering of the same character | Meta Muse Image (`muse-image-1.0`), SAM 3.1, Muse Spark (`muse-spark-1.3`, the input check for photos of real people) |
| 2 turnaround | Four orthographic A-pose views in one generation, split by code, checked for agreement; then four held-out turned views (30 and 60 degrees each side) for scoring only, each accepted, sent to human review or rejected | Muse Image, SAM 3.1 |
| 3 base mesh | Builds a hybrid from SAM 3D Objects and SAM 3D Body joints (default), with an automatic, recorded carve fallback if the hybrid cannot build; `--mesh-source carve --joints silhouette` selects the local path; records observed silhouette overlaps and checks closed edges plus multiview usability with Muse Spark | Blender 4.5 LTS (headless); optionally SAM 3D Objects and SAM 3D Body on Modal |
| 3f texture refine | Under `best`, `roblox-ingame` and `roblox-avatar`, repaints fixed head/body renders of SAM's textured mesh, registers and gates every view, matches its colours and projects accepted detail sequentially into a 4096 atlas; protected hair passes refine the sides and back. Writes `03-refined.glb`; all-view rejection keeps `03-source.glb`. Default API cap $0.80 per run for every profile; estimated $0.365 at one sample per view | Muse Image, SAM 3.1, Blender |
| 4 game body | Cuts the mesh at the joints into 15 rigid R15 sections, each pivoting at the centre of its cut, with shoulder covers; the neck, waist, elbow, wrist, knee and ankle cuts move off a costume ledge only where the gate poses show a gain beyond noise; at the detail a target profile asks for (default `best`: full detail, no triangle budget; `roblox-ingame`: high and low LODs within 24,000 / 6,000 triangles; `roblox-avatar`: one body within the triangle, watertight-part and texture limits of Roblox's character body specification, without its rig, attachments, facial animation or cages), bakes validated refined colours into the shared atlas and Head texture of every LOD (otherwise the turnaround bake with raw source fallback and Head close-ups); atlas sizes (1024 / 512 under `roblox-ingame`; under `best` and `roblox-avatar` the smallest size with a texel per view pixel, at most 4096 and 2048; in the view bake, a SAM 3D Objects mesh's own texture where no view faces the surface), and runs a pose test that measures visible joint gaps on 20 gate poses (8 more held out and reported, never used to choose) | Blender |
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
- For stage 3f: `pip install ".[refine]"` (SciPy and OpenCV, including its EXR reader). Live calls also need
  `.[live]` and `MODEL_API_KEY`. The default `best` profile runs 3f between stages 3 and 4; model mode defaults
  to replay, so use `--refine live` for new repaints. Both Roblox profiles also run 3f with the same cap and
  gates. `--stages 3f` refines a saved run independently. Stage 4 transfers a complete, source-hash-matched
  refinement onto every sectioned LOD, including its Head texture, after checking the source frame. Otherwise
  it retains the turnaround bake with the raw source fallback and Head close-ups.
- For live model calls in stages p, 1 and 2: a Meta Model API key in `MODEL_API_KEY` (see `.env.example`), and
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
  By default local calls use an ephemeral Modal app. For persistent deployments, deploy both runners with
  `python -m herogen.blender.remote.deploy` (it sets each runner's capacity: up to 5 containers, none kept warm),
  then set `HEROGEN_REMOTE_DEPLOYED=1`. Both deployments scale to zero after 300 idle seconds.
  `herogen.blender.remote.prewarm()` submits model loading without waiting; the website calls it when a hero
  that will reach stage 3 is accepted, so the models load while the images are drawn (a cold SAM 3D Objects load
  takes about 90 s). `HEROGEN_PREWARM=0` disables it. Prewarming incurs startup and idle GPU costs.
  Stop finite deployments with `modal app stop` after draining their inputs. The hosted service uses these
  persistent deployments through its bridge.

## Running

```bash
python3 -m herogen.pipeline run sketch.png --run-id my-hero --live
python3 -m herogen.blender run --run runs/my-hero --stages 3,4 --remote live --refine live
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

The hybrid body (SAM 3D Objects shape, SAM 3D Body joints) is the default. To populate its remote
cache, use `--remote live`; the hosted service requests live hybrid builds by default. An unavailable
SAM 3D source or a failed hybrid build retries the local carve with silhouette joints once. The stage
record preserves the requested and used sources, the fallback reason and spend from both attempts;
the CLI prints the fallback reason. `--no-fallback` stops instead. A quality rejection does not retry.
Stage 3 keeps silhouette overlaps as diagnostics and gates usability with a fixed Muse Spark question
on the sketch, four cardinal renders and two unseen angles. It needs `MODEL_API_KEY`; the question uses
the repository's retry, cache and cost accounting. `--quality replay` requires its exact cached answer.
Model failures stop loudly with a failed record and cost, and never trigger a carve retry. Stage 4 still
gates rest IoU at 0.90 against the turnaround on shape views and the received stage-3 mesh on other observed
views, plus separate texture checks, so passing stage 3 does not guarantee a finished game body.

Stages p, 1 and 2 replay model calls from the cache unless `--live` is set. In stage 3, `--remote`
controls only SAM 3D calls: they replay from `runs/_cache-remote/` unless `--remote live` is set.
The body usability question separately defaults to `--quality live`, a paid call; `--quality replay`
requires every attempted answer in its cache. Tests use cached or mocked calls. A remote source with no
cached result falls back to the carve or the silhouette joints and says so in `stage-3.json`
(`--no-fallback` makes it an error).

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

The hosted service (public saved-result inspection, optional capped public live access, email/invite access,
private workspaces, GLB and Roblox downloads) is `herogen/service`; `python3 -m herogen.service dev` runs it locally (docs/ARCHITECTURE.md, "Service").

Starting from a text prompt instead of a sketch:

```bash
python3 -m herogen.pipeline run --prompt "a knight in battered plate armour" --run-id my-knight --stages 0-2,5 --live
python3 -m herogen.pipeline prompt "a knight in battered plate armour" --run-id my-knight --live   # stage p alone
```

Stage p writes `p-design.png`, `p-full-body.png` and every attempt under `p-attempts/`, then stage 0 copies
`p-full-body.png` as the sketch. The prompt is limited to 300 characters. The user's text sits in a bounded
slot of the design prompt, introduced as describing only the character and never how the picture is drawn, and
the full-body edit never sees it; no check measures the drawing style yet. The prompt text, models and
parameters come from a frozen, hashed prompt version in `herogen/prompts/versions/` (`hero-sketch-v3` by
default), recorded in `stage-p.json`. Both sketch steps exclude halos, glows and rings or symbols around the
head, and ask for both hands empty with no pet or companion; the original v1 remains available for replay. Stage 1 restores the recorded brief in a quoted,
300-character slot for colours, materials and gear, with the same head exclusions. It records whether the
slot was used and the cover prompt SHA-256 in `stage-1.json`. Sketch uploads keep the original cover prompt
byte for byte. Stage 2 takes colours from the cover. The checks, their thresholds and the measurements behind them are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#stage-p-prompt-to-sketch).

Redo the cover or turnaround as a new version, with an optional appearance note:

```bash
python3 -m herogen.pipeline run --parent my-hero --from-stage 1 --note "make the coat burnt orange" --run-id my-orange-hero --stop-after 1 --live
python3 -m herogen.pipeline run --parent my-hero --from-stage 2 --note "left view: keep the satchel strap across the chest" --run-id my-satchel-hero --stop-after 2 --live
python3 -m herogen.pipeline run --run-id my-orange-hero --stages 2,5 --live
```

`--parent` accepts a run id under `--runs`, or a directory path. The destination must be new. Earlier passing
stages, their outputs and their records are copied into it at zero new model cost; the parent is never changed.
A cover redo edits a copy of the parent's cover and still checks its outline against the original sketch.
A turnaround redo keeps the cover and regenerates the **whole four-view sheet**, even when the note names
just one view, so the views stay consistent. All existing gates and retries still apply.

The optional note describes changes to the character's colours, materials, gear or character style. It is
cleaned like the character brief, quoted between `«»`, and limited to 300 characters after cleaning; longer
notes are refused. The slot says the note cannot instruct the image, layout, number of views or checks.
`cover-redo-v1` and `turnaround-redo-v2` freeze the note prompts and hashes. The default sheet prompt is
`turnaround-v3`, which asks for arms clear of the body ending in visible hands, props stowed on the back or belt
and opposite-facing profiles; with one attempt per input it gave 27 of 30 complete rigged exports on a 30-input
set. `turnaround-v1` and `turnaround-v2` stay selectable with `HEROGEN_TURNAROUND_VERSION`, and
`turnaround-redo-v2` builds on the v3 sheet prompt. Without a note, existing prompt bytes and prompt
selection stay unchanged. `redo.json` and the stage records identify the parent, redo
stage and cleaned note. `--stop-after 1|2` also omits later side work; without it the CLI follows its normal
selected stages, including held-out views and face sources after stage 2.

The signed-in service exposes backend version APIs (the UI is separate):

- `POST /api/jobs/<id>/redo` with JSON `{"stage":"cover","note":"green coat"}` (or `"turnaround"`) and an
  `Idempotency-Key` creates a new owned job with `parent_job`. It stops after the requested stage.
- `POST /api/jobs/<id>/continue` continues a stopped version through the remaining stages, using the same
  job. Repeated requests while it is active return that job.
- `GET /api/jobs/<id>/versions` returns `versions`: the owned parent family, including siblings and
  descendants, with stage, note, status and `cover_thumbnail_url` (the existing cover image URL, or null).

Unsafe requests need the normal same-origin CSRF token. Storage and rate limits apply to redos and
continuations, and so does the spend guard when spend limits are on. Earlier versions must have stopped before
branching; deleted versions are omitted from lists.

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
- **3D viewer.** The run's final body GLB — `03-refined.glb`, then `03-source.glb`, then `04-body/high.glb` for
  the best profile; the sectioned `04-body/high.glb` for the Roblox profiles — or its `low.glb`: textured, clay,
  wireframe or per-section colours, and beside them a Light toggle, Lit or Art. Lit (the default) lights the body
  with three's RoomEnvironment through a PMREM environment plus one gentle key light, tone mapped with an exposure
  slider (0.5–2.0); Art shows the base colour texture unlit through a MeshBasicMaterial, exactly as painted and
  with no tone mapping, so the two views separate studio presentation from the paint itself. Clay, wireframe and
  per-section colours keep working under both. A refined or SAM-source body is one mesh with no R15 sections: it
  answers every one of these controls too (one neutral colour for the per-section view, nothing to pose or
  isolate) and its triangles are counted. The seven test poses of `herogen/blender/pose.py`, each section turned
  about its pivot; click a section, or pick it from the list or the legend, to isolate it and see its triangles,
  pivot and joint; front, 3/4, side and back cameras, orbit, and an axis gizmo. "Side by side" puts the body
  beside a turnaround view through that view's own orthographic camera (its pixels per stud from
  `views/meta.json`, framed on the view's mask), or beside the sketch or artwork, scaled to the figure there
  (approximate). The compare view lights both bodies the same way. Tone mapping is ACESFilmic: Khronos PBR
  Neutral only reached three.js in r162, after the pinned 0.160.0, and ACESFilmic is that version's nearest
  equivalent for the soft studio look.
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
folder only. The page loads three.js 0.160.0 and its GLTFLoader, OrbitControls and RoomEnvironment addons from
jsDelivr through an import map pinned by SHA-384 and makes no other external requests; its fonts are served by the
lab itself.

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

Tags on a commit on `main` deploy the hosted service (`herogen/service`) to Modal through
`.github/workflows/release.yml`: `vX.Y.Z-rc.N` to staging, then `vX.Y.Z` on the same commit to production once
its staging deploy has succeeded, each after the tests pass. Any other tag starting with `v`, and any tag on a
commit that is not on `main`, fails. Deploying needs the service code in the tagged commit. The
rules, the secrets and variables each GitHub environment needs, and how to roll back are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#releases).

## Third-party models and libraries

Muse Image, SAM 3.1 and Muse Spark are called through Meta's Model API. `meta-sam-parser`, an optional dependency used to
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

Licensed under Apache 2.0 — see [LICENSE](LICENSE). The models and libraries HeroGen calls keep their own
licences, listed under [Third-party models and libraries](#third-party-models-and-libraries).
