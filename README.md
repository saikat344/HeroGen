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
