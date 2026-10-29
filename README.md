# HeroGen

HeroGen turns one hand-drawn character sketch into a game-ready hero body for Roblox, and checks every stage
with measurements.

```
sketch → cover art → turnaround (front, back, left, right) → 3D mesh + joints → 15-section R15 body → scores
```

## Stages

| Stage | What it does | Models and tools |
| --- | --- | --- |
| 0 input | Copies the sketch into a run folder | |
| 1 cover art | A clean colour rendering of the same character | Meta Muse Image (`muse-image-1.0`), SAM 3.1 |
| 2 turnaround | Four orthographic A-pose views in one generation, split by code, checked for agreement; one held-out 3/4 view for scoring only | Muse Image, SAM 3.1 |
| 3 base mesh | A textured mesh from the front view, or a silhouette carve from the four masks | SAM 3D Objects on Modal, or Blender |
| 3 joints | The skeleton from the front view, or from the silhouettes | SAM 3D Body on Modal, or code |
| 4 game body | Cuts the mesh at the joints into the 15 R15 sections with pivots, shoulder covers, high and low LODs within triangle budgets, turnaround colours baked into one atlas per LOD, and a pose test that measures visible joint gaps | Blender 4.5 LTS (headless) |
| 5 results | Scores every body on the four views and the held-out view, and writes a static results page | Blender, code |

Every stage writes `stage-<n>.json` with its measured checks and stops the run when a gate fails. The file
formats, cameras and checks are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Requirements

- Python 3.11 or newer, with `numpy` and `pillow`.
- Blender 4.5 LTS for stages 3 to 5 (set `BLENDER` if it isn't on `PATH`).
- A Meta Model API key in `MODEL_API_KEY` for stages 1 and 2 (see `.env.example`).
- For SAM 3D: a [Modal](https://modal.com) account (`modal setup`), and a Hugging Face read token from an account
  granted access to `facebook/sam-3d-objects` and `facebook/sam-3d-body-dinov3`, stored as the Modal secret
  `huggingface`. Modal downloads the checkpoints; this repository doesn't contain them.

## Running

```bash
python3 -m herogen.pipeline run sketch.png --run-id my-hero --live
python3 -m herogen.blender run --run runs/my-hero --stages 3,4
```

Without `--live`, model calls replay from the cache, which is how the tests run.

## Tests

```bash
python3 -m unittest discover -s tests -v
python3 tools/guard/guard.py tree
```

Tests use synthetic inputs drawn by code and never call a model or Modal. Blender tests skip when Blender isn't
installed.

## Third-party models

SAM 3D Objects and SAM 3D Body are released by Meta under the SAM License; their checkpoints are gated on
Hugging Face and are not redistributed here. Muse Image and SAM 3.1 are called through Meta's Model API.

## Licence

To be decided by the maintainer.
