"""SAM 3D Objects on a Modal GPU: one image and its mask -> a textured GLB (the stage 3 source mesh).

Run through `herogen.blender.remote.client` (record/replay), which calls:

    modal run objects.py::main --job <job.json> --out <dir>

`job.json`: {"jobs": [{"name": str, "image": png path, "mask": png path}], "seed": int}. For each job the
output directory gets `<name>/mesh.glb` and `<name>/run.json` (seconds, GPU memory, mesh size, the predicted
pose); `run.json` at the top holds the GPU name, load time and peak memory.

Model: facebook/sam-3d-objects (gated on Hugging Face; code and weights under the SAM License, which travels with
any redistribution). Nothing from its repository is copied here: the image clones it at a pinned commit, and the
two changes it needs are made at run time by the self-written code below. The checkpoints live in a Modal volume
(`HEROGEN_OBJECTS_VOLUME`); the `download` entry point fills it once.

No nvdiffrast. Its licence (NVIDIA Source Code License, 1-Way Commercial) allows only non-commercial use, so the
image does not install it. SAM 3D's texture bake samples with `rendering_engine="pytorch3d"` (grid_sample), and
the rasterisation it still asks of `utils3d.torch` (`RastContext`, `rasterize_triangle_faces`, used by the
mesh hole filling and the bake's UV maps) is served by `pytorch3d_rasteriser`, a pytorch3d-based stand-in with
utils3d's conventions. The splat renders for the bake always use gsplat (Apache-2.0): upstream's
`render_utils.render_frames` is wrapped (`gsplat_backend`) so every call it receives asks for the `gsplat`
backend; the Inria `diff_gaussian_rasterization` backend (non-commercial licence) is never installed or used.

Packages left out of upstream's `requirements.txt` (`EXCLUDE`): `smplx` (its licence covers non-commercial
scientific research only; SAM 3D Objects' inference never imports it), `point-cloud-utils` (GPL) and `pdoc3`
(AGPL), which it lists but never imports, the Blender module, GPU quantisation and cloud tooling it does not use.
The checkpoints and the depth model are downloaded at the revisions in `WEIGHTS`; a volume holding another
revision is refused. The image build fails if any of `ABSENT` can be imported, and every call records
that none of them was imported (`run.json` `"absent_modules_imported"`).

The Hugging Face token is read only by huggingface_hub from the Modal secret `huggingface` inside the container;
it is never printed or written. This file imports nothing from herogen so Modal can load it on its own;
`modal run` apps are ephemeral and stop when the call returns. The first live call builds the image (CUDA
extensions are compiled: tens of minutes); `modal run objects.py::build` builds it ahead of time on a CPU
container and reports what it holds, so the first inference call fits the normal timeout.

Configuration (environment variables read when Modal loads this file):
- `HEROGEN_OBJECTS_GPU`: GPU type (default `A100-40GB`; the pipeline needs about 23 GB of GPU memory).
- `HEROGEN_OBJECTS_VOLUME`: name of the Modal volume holding the checkpoints (default
  `herogen-sam3d-objects-ckpt`; set it to reuse an existing volume).
- `HEROGEN_OBJECTS_TIMEOUT`: seconds the GPU function may run (default 1800); the client sets it to its own
  timeout so the container never outlives the caller.

Where the self-written changes apply: the image clones the repository at `REPO_SHA` and runs the repository's
own `patching/hydra` step at build time; the rasteriser and splat-render substitutions below are assigned in the
container process before the pipeline is instantiated. No file of the repository is copied into HeroGen.
"""
import json
import os
import pathlib
import time

import modal

APP = "herogen-sam3d-objects"
REPO_SHA = "f91db411c50efee93d8db7aeb323885650f6f722"
# Weight revisions on Hugging Face (pinned; recorded in every run log and in the client's cache key).
WEIGHTS = {"facebook/sam-3d-objects": "2e73555018d2741ccd486e56c24fac41155a1dc6",
           "Ruicheng/moge-vitl": "ad326bfb61facd6c52b5a825bc1e34d7c97d9672"}   # the pipeline's depth model (MIT)
CKPT_DIR = "/ckpt"
GPU = os.environ.get("HEROGEN_OBJECTS_GPU", "A100-40GB")
TIMEOUT = int(float(os.environ.get("HEROGEN_OBJECTS_TIMEOUT", "1800")))
vol = modal.Volume.from_name(os.environ.get("HEROGEN_OBJECTS_VOLUME", "herogen-sam3d-objects-ckpt"),
                             create_if_missing=True)

EXCLUDE = ["auto_gptq", "optimum", "bpy", "sagemaker", "nvidia-pyindex", "pip-system-certs", "conda-pack",
           "jupyter", "torchaudio", "smplx", "point-cloud-utils", "pdoc3"]
# Modules that must not be importable in the image: licences that forbid commercial use (smplx, nvdiffrast, the
# Inria rasteriser) and copyleft packages inference never imports (point-cloud-utils GPL, pdoc3 AGPL).
ABSENT = ["smplx", "nvdiffrast", "diff_gaussian_rasterization", "point_cloud_utils", "pdoc"]
absent_check = ("python -c \"import importlib.util, sys; found = [m for m in sys.argv[1:] if importlib.util.find_spec(m)];"
                " sys.exit('must not be installed: ' + ', '.join(found) if found else 0)\" " + " ".join(ABSENT))
filter_cmd = ("cd /opt/sam3do && grep -v -i -E '^(" + "|".join(EXCLUDE) + ")' requirements.txt > /tmp/req.txt"
              " && echo 'torch==2.5.1' > /tmp/constraints.txt && echo 'torchvision==0.20.1' >> /tmp/constraints.txt")
PIPX = "pip install -c /tmp/constraints.txt --extra-index-url https://download.pytorch.org/whl/cu121"

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0", "libegl1", "libxrender1",
                 "libxext6", "libsm6", "libx11-6", "wget", "curl")
    .env({"CUDA_HOME": "/usr/local/cuda", "CONDA_PREFIX": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.0;9.0",
          "FORCE_CUDA": "1", "MAX_JOBS": "8", "LIDRA_SKIP_INIT": "true", "PIP_NO_CACHE_DIR": "1"})
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
                 index_url="https://download.pytorch.org/whl/cu121")
    .run_commands(f"git clone https://github.com/facebookresearch/sam-3d-objects /opt/sam3do && cd /opt/sam3do"
                  f" && git checkout {REPO_SHA}")
    .run_commands(filter_cmd, f"{PIPX} -r /tmp/req.txt")
    .run_commands(f"{PIPX} https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
                  "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
    # Modal's add_python interpreter links extensions with clang++, which the image lacks: use g++.
    .env({"CC": "gcc", "CXX": "g++", "LDSHARED": "g++ -pthread -shared"})
    .run_commands("pip install wheel ninja",
                  f"{PIPX} --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git"
                  "@75ebeeaea0908c5527e7b1e305fbc7681382db47'")
    .run_commands(f"{PIPX} kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html")
    .run_commands(f"{PIPX} --no-build-isolation 'git+https://github.com/nerfstudio-project/gsplat.git"
                  "@2323de5905d5e90e035f792fe65bad0fedd413e7'")
    .run_commands(f"{PIPX} seaborn==0.13.2 gradio==5.49.0")
    # kaolin pins numpy 1.26; requirements.txt leaves numpy-2-only pandas/plyfile behind. Re-pin a consistent set.
    .run_commands("pip install numpy==1.26.4 pandas==2.2.3 plyfile==1.0.3", "pip check > /opt/pip-check.txt || true")
    .run_commands("cd /opt/sam3do && pip install --no-deps -e . && python patching/hydra",
                  "pip freeze > /opt/freeze.txt", absent_check)
)

app = modal.App(APP)


@app.function(image=modal.Image.debian_slim().pip_install("huggingface_hub[cli]<1.0"), volumes={CKPT_DIR: vol},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3600)
def fetch_checkpoints():
    from huggingface_hub import snapshot_download

    t = time.time()
    repo = "facebook/sam-3d-objects"
    p = snapshot_download(repo, revision=WEIGHTS[repo], local_dir=f"{CKPT_DIR}/hf-download", max_workers=4)
    vol.commit()
    files = [x for x in pathlib.Path(p).rglob("*") if x.is_file()]
    return {"seconds": round(time.time() - t, 1), "files": len(files), "bytes": sum(x.stat().st_size for x in files)}


class Pytorch3dRastContext:
    """Stands in for utils3d.torch.RastContext (an nvdiffrast context); pytorch3d needs no context."""

    def __init__(self, *args, **kwargs):
        self.backend = "pytorch3d"


def pytorch3d_rasteriser(ctx, vertices, faces, width, height, attr=None, uv=None, texture=None, model=None,
                         view=None, projection=None, antialiasing=True, diff_attrs=None):
    """utils3d.torch.rasterize_triangle_faces (utils3d 0.0.2) on pytorch3d, for what SAM 3D's postprocessing
    reads: `mask`, `face_id` (1-based, 0 = empty) and `depth` with row 0 at the top, and `uv` with row 0 at the
    bottom (OpenGL order, as nvdiffrast returns it). Barycentrics are made perspective-correct with the clip w.
    `uv_dr` is returned as zeros (only nvdiffrast's texture filtering reads it); `attr` and `texture` are not
    supported (SAM 3D's mesh path does not pass them)."""
    import torch
    from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
    from pytorch3d.structures import Meshes

    if attr is not None or texture is not None:
        raise NotImplementedError("pytorch3d_rasteriser: attr/texture are not supported")
    if vertices.shape[-1] == 2:
        vertices = torch.cat([vertices, torch.zeros_like(vertices[..., :1]), torch.ones_like(vertices[..., :1])], -1)
    elif vertices.shape[-1] == 3:
        vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], -1)
    mvp = projection if projection is not None else torch.eye(4).to(vertices)
    if view is not None:
        mvp = mvp @ view
    if model is not None:
        mvp = mvp @ model
    clip = vertices.float() @ mvp.float().transpose(-1, -2)            # [B, N, 4]
    w = clip[..., 3:4]
    ndc = clip[..., :3] / w
    # pytorch3d's NDC has +X to the left and keeps z > 0; OpenGL's has +X right and z in [-1, 1]
    p3 = torch.stack([-ndc[..., 0], ndc[..., 1], ndc[..., 2] * 0.5 + 0.5 + 1e-4], -1)
    f = faces.long()
    b = p3.shape[0]
    meshes = Meshes(verts=list(p3), faces=[f] * b)
    # coarse-to-fine binning exists only on the GPU; on the CPU (the image's self-test) rasterise naively
    pix, zbuf, bary, _ = rasterize_meshes(meshes, image_size=(height, width), blur_radius=0.0, faces_per_pixel=1,
                                          bin_size=64 if p3.is_cuda else 0,
                                          max_faces_per_bin=max(int(f.shape[0]), 10000),
                                          perspective_correct=False, cull_backfaces=False)
    pix, bary = pix[..., 0], bary[..., 0, :]                              # [B, H, W], [B, H, W, 3]
    offs = (torch.arange(b, device=pix.device) * f.shape[0]).view(b, 1, 1)
    valid = pix >= 0
    local = torch.where(valid, pix - offs, torch.zeros_like(pix))
    mask = valid.float()
    face_id = torch.where(valid, local + 1, torch.zeros_like(local)).float()
    depth = (zbuf[..., 0] - 1e-4) * mask + (1.0 - mask)
    ret = {"depth": depth, "mask": mask, "face_id": face_id}
    if uv is not None:
        corner = f[local]                                                 # [B, H, W, 3]
        bidx = torch.arange(b, device=pix.device).view(b, 1, 1, 1).expand_as(corner)
        winv = 1.0 / w[..., 0]
        bw = bary * winv[bidx, corner]
        bw = bw / bw.sum(-1, keepdim=True).clamp_min(1e-12)
        uvc = uv[bidx, corner]                                            # [B, H, W, 3, 2]
        uv_map = (bw[..., None] * uvc).sum(-2) * mask[..., None]
        ret["uv"] = uv_map.flip(1)
        ret["uv_dr"] = torch.zeros(uv_map.shape[:-1] + (4,), device=uv_map.device).flip(1)
    return ret


def rasteriser_selftest(device: str = "cpu") -> dict:
    """`pytorch3d_rasteriser` on one known triangle: the conventions SAM 3D's post-processing reads. The triangle
    (OpenGL clip space, w = 1) covers the lower-left half of an 8 x 8 image at z = 0, with uv (0,0), (1,0), (0,1)
    at its corners. Returns each property and whether it held."""
    import torch

    verts = torch.tensor([[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]], device=device)
    faces = torch.tensor([[0, 1, 2]], device=device)
    uv = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], device=device)
    r = pytorch3d_rasteriser(None, verts, faces, 8, 8, uv=uv)
    m, fid, depth, uvm = r["mask"][0], r["face_id"][0], r["depth"][0], r["uv"][0]
    checks = {
        # row 0 is the top: the bottom-left pixel is covered, the top-right one is not
        "mask_row0_top": bool(m[7, 0] == 1 and m[0, 7] == 0),
        "face_id_one_based": bool(fid[7, 0] == 1 and fid[0, 7] == 0),
        # OpenGL z = 0 maps to depth 0.5; empty pixels are 1
        "depth": bool(abs(float(depth[7, 0]) - 0.5) < 1e-3 and float(depth[0, 7]) == 1.0),
        # uv comes back with row 0 at the bottom: near the (0,0) corner at [0, 0], u grows to the right
        "uv_row0_bottom": bool(float(uvm[0, 0, 0]) < 0.2 and float(uvm[0, 0, 1]) < 0.2
                               and float(uvm[0, 5, 0]) > float(uvm[0, 1, 0])),
        "uv_dr_zero": bool(float(r["uv_dr"].abs().sum()) == 0.0),
    }
    return {"ok": all(checks.values()), **checks}


def gsplat_backend(render_frames):
    """Upstream's `render_frames(sample, extrinsics, intrinsics, options, ...)` unchanged, except that the options
    it receives always ask for the gsplat splat backend."""

    def render_frames_gsplat(sample, extrinsics, intrinsics, options=None, *args, **kwargs):
        return render_frames(sample, extrinsics, intrinsics, {**(options or {}), "backend": "gsplat"}, *args, **kwargs)

    return render_frames_gsplat


def checkpoint_revision(local_dir: pathlib.Path) -> str | None:
    """The Hugging Face commit a `snapshot_download(local_dir=...)` folder was downloaded at (huggingface_hub
    writes it as the first line of each file's `.cache/huggingface/download/<file>.metadata`)."""
    meta = sorted((local_dir / ".cache" / "huggingface" / "download").glob("*.metadata"))
    return meta[0].read_text().splitlines()[0].strip() if meta else None


class VramSampler:
    """Polls nvidia-smi for the device-level memory peak."""

    def __init__(self):
        import threading

        self.peak = 0
        self.stop = False
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        import subprocess

        while not self.stop:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                self.peak = max(self.peak, int(out.strip().splitlines()[0]))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)


@app.function(image=image, gpu=GPU, volumes={CKPT_DIR: vol}, timeout=TIMEOUT,
              secrets=[modal.Secret.from_name("huggingface")])
def infer(jobs: list, seed: int = 42):
    import io
    import subprocess
    import sys

    import numpy as np
    from PIL import Image

    os.environ.setdefault("HF_HOME", f"{CKPT_DIR}/hf-home")       # MoGe cache
    os.environ.setdefault("TORCH_HOME", f"{CKPT_DIR}/torch-home")  # torch.hub cache (DINOv2)
    log = {"gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                  "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
           "repo_sha": REPO_SHA, "weights": WEIGHTS, "seed": seed, "runs": {}}
    sampler = VramSampler()
    t0 = time.time()
    ck = pathlib.Path(CKPT_DIR) / "hf-download" / "checkpoints"
    have = checkpoint_revision(pathlib.Path(CKPT_DIR) / "hf-download")
    if have != WEIGHTS["facebook/sam-3d-objects"]:
        raise RuntimeError(f"the checkpoint volume holds facebook/sam-3d-objects revision {have}, not the pinned "
                           f"{WEIGHTS['facebook/sam-3d-objects']}: run objects.py::download")
    os.chdir("/opt/sam3do")
    sys.path.insert(0, "/opt/sam3do/notebook")
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    import inference as nb  # notebook/inference.py: its imports and the hydra safety check
    from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils

    import importlib.util

    # The texture bake renders views of the splat through render_utils.render_frames with upstream's default
    # backend "inria"; every call goes to gsplat instead.
    render_utils.render_frames = gsplat_backend(render_utils.render_frames)
    log["bake_backend"] = "gsplat (render_utils.render_frames wrapped to force the gsplat backend)"
    import utils3d

    # set before anything touches utils3d.torch.rasterization (which imports nvdiffrast)
    utils3d.torch.RastContext = Pytorch3dRastContext
    utils3d.torch.rasterize_triangle_faces = pytorch3d_rasteriser
    log["renderer"] = {"rendering_engine": "pytorch3d",
                       "rasteriser": "pytorch3d rasterize_meshes (stand-in for utils3d.torch.rasterize_triangle_faces)",
                       "splat_renders": log["bake_backend"],
                       "nvdiffrast_installed": importlib.util.find_spec("nvdiffrast") is not None}

    config = OmegaConf.load(str(ck / "pipeline.yaml"))
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.workspace_dir = str(ck)
    # the depth model at its pinned revision (from_pretrained passes revision on to hf_hub_download)
    depth = config.depth_model.model
    if depth.pretrained_model_name_or_path not in WEIGHTS:
        raise RuntimeError(f"unexpected depth model {depth.pretrained_model_name_or_path}; pin its revision")
    depth.revision = WEIGHTS[depth.pretrained_model_name_or_path]
    nb.check_hydra_safety(config, nb.WHITELIST_FILTERS, nb.BLACKLIST_FILTERS)
    pipe = instantiate(config)
    log["load_seconds"] = round(time.time() - t0, 1)

    files = {}
    for job in jobs:
        name = job["name"]
        img = np.asarray(Image.open(io.BytesIO(job["image"])).convert("RGB"))
        mask = np.asarray(Image.open(io.BytesIO(job["mask"])).convert("L")) > 127
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.time()
        rgba = nb.Inference.merge_mask_to_rgba(None, img, mask)
        out = pipe.run(rgba, None, seed, stage1_only=False, with_mesh_postprocess=True, with_texture_baking=True,
                       with_layout_postprocess=False, use_vertex_color=False)
        torch.cuda.synchronize()
        rec = {"seconds": round(time.time() - t, 1),
               "torch_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
               "torch_max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20)}
        glb = out.get("glb")
        d = {}
        if glb is not None:
            d["mesh.glb"] = glb.export(file_type="glb")
            rec["glb_vertices"] = int(len(glb.vertices))
            rec["glb_faces"] = int(len(glb.faces))
            try:
                rec["texture_size"] = list(glb.visual.material.baseColorTexture.size)
            except Exception as e:  # noqa: BLE001
                rec["texture_size"] = f"n/a ({type(e).__name__})"
        pose = {}
        for k in ("rotation", "translation", "scale"):
            v = out.get(k)
            if isinstance(v, torch.Tensor):
                pose[k] = v.detach().float().cpu().numpy().round(6).tolist()
        rec["pose"] = pose
        files[name] = d
        log["runs"][name] = rec
    sampler.stop = True
    log["absent_modules_imported"] = sorted(m for m in ABSENT if m in sys.modules)
    log["vram_peak_mib_nvidia_smi"] = sampler.peak
    log["total_seconds"] = round(time.time() - t0, 1)
    return log, files


@app.local_entrypoint()
def main(job: str, out: str):
    spec = json.loads(pathlib.Path(job).read_text())
    jobs = [{"name": j["name"], "image": pathlib.Path(j["image"]).read_bytes(),
             "mask": pathlib.Path(j["mask"]).read_bytes()} for j in spec["jobs"]]
    t = time.time()
    log, files = infer.remote(jobs, seed=int(spec.get("seed", 42)))
    log["wall_seconds"] = round(time.time() - t, 1)
    log["gpu_type"] = GPU
    o = pathlib.Path(out)
    for name, d in files.items():
        (o / name).mkdir(parents=True, exist_ok=True)
        for fn, b in d.items():
            (o / name / fn).write_bytes(b)
        (o / name / "run.json").write_text(json.dumps(log["runs"][name], indent=2))
    (o / "run.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({k: v for k, v in log.items() if k != "runs"}))


@app.function(image=image, timeout=600)
def image_report():
    """CPU only, in the inference image: what it holds (the image is built first if it is not cached)."""
    import importlib.util

    freeze = pathlib.Path("/opt/freeze.txt").read_text().splitlines()
    return {"repo_sha": REPO_SHA, "weights": WEIGHTS, "packages": len(freeze), "rasteriser": rasteriser_selftest(),
            "absent_importable": [m for m in ABSENT if importlib.util.find_spec(m) is not None],
            "excluded": EXCLUDE}


@app.local_entrypoint()
def download():
    print(json.dumps(fetch_checkpoints.remote()))


@app.local_entrypoint()
def build():
    print(json.dumps(image_report.remote()))
