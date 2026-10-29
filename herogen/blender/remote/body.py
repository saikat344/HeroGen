"""SAM 3D Body on Modal: one image -> named 2D/3D keypoints and the body mesh (the stage 3 joint source).

Run through `herogen.blender.remote.client` (record/replay), which calls:

    HEROGEN_BODY_GPU=A10G modal run body.py::main --job <job.json> --out <dir>

`HEROGEN_BODY_GPU` picks the GPU (default A10G; L4, T4, ...); `none` runs on 8 CPU cores (the container's copy
of the model repo is patched from cuda to cpu by the self-written `_patch_repo_for_cpu`, at the start of the
call, before the model is imported). `job.json`: {"jobs": [{"name": str, "image": png path, "bbox": [x0, y0,
x1, y1], "box_source": str}], "detector": null or "vitdet"}. The box is the person box (HeroGen passes the
bounding box of stage 2's front mask) and is required. By default no person detector runs. With `"detector":
"vitdet"`, upstream's ViTDet detector (detectron2; its weights, released under CC BY-SA 3.0, are downloaded on
that first use) finds the person instead and the caller's box is used only if it finds nobody; every run log
records the box's source. For each job the output directory gets `<name>/joints.json` (the 70 named MHR70
keypoints with image uv and camera-frame xyz, the 127 MHR skeleton joints, focal length, camera translation,
person box and its source, and the projected mesh's top and bottom), `<name>/body.npz` (mesh vertices in the
camera frame and faces) and `<name>/run.json` (seconds, memory and the box's source).

Model: facebook/sam-3d-body-dinov3 (gated on Hugging Face; SAM License). Checkpoints are cached in a Modal volume
named by `HEROGEN_BODY_VOLUME` (default `herogen-sam3d-body-ckpt`; set it to reuse an existing volume).
`HEROGEN_BODY_TIMEOUT` sets the seconds the function may run (default 1800; the client sets it to its own timeout).
The Hugging Face token is read only by huggingface_hub from the Modal secret `huggingface` inside the container; it
is never printed or written. This file imports nothing from herogen. Nothing from the model's repository is copied
here: the image clones it at a pinned commit, MoGe (its field-of-view estimator) is installed at a pinned commit
too, and the DINOv3 backbone code, which upstream would fetch from GitHub's main branch at run time, is cloned at
`DINOV3_COMMIT` and loaded from that copy (`_pin_dinov3`). The image installs the repository's listed dependencies
except those its inference never imports (`chump`, `appnope`, the PyPI `ffmpeg` wrapper, `black`, `pytest`,
`wandb`: tooling, macOS-only or training-only; `wandb` is imported only inside the training loggers), and no system
ffmpeg; the build fails if any of `ABSENT` can be imported. The weights are downloaded at the revisions in
`WEIGHTS` (the model and MoGe's), which every run log records.
"""
import io
import json
import os
import pathlib
import time

import modal

GPU = os.environ.get("HEROGEN_BODY_GPU", "A10G")
TIMEOUT = int(float(os.environ.get("HEROGEN_BODY_TIMEOUT", "1800")))
CPU_MODE = GPU.lower() == "none"
REPO_COMMIT = "b5c765a0d89d789985e186d396315e7590887b94"
MOGE_COMMIT = "74fbce054ebed49800de42d0ad0e83495065719a"    # microsoft/MoGe (MIT), the FOV estimator
# facebookresearch/dinov3 (DINOv3 License): the backbone's model code, which upstream fetches from GitHub's main
# branch through torch.hub at run time. The image clones it at this commit (`DINOV3_DIR`) and every call points
# the hub load at that copy (`pin_dinov3_source`); the build fails if upstream's hub call is not the one expected.
DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
DINOV3_DIR = "/dinov3"
DINOV3_BACKBONE = "/repo/sam_3d_body/models/backbones/dinov3.py"
DINOV3_HUB_CALL = ('"facebookresearch/dinov3",', 'source="github",')
dinov3_check = " && ".join(f"grep -qF '{t}' {DINOV3_BACKBONE}" for t in DINOV3_HUB_CALL)
# Weight revisions on Hugging Face (pinned; recorded in every run log and in the client's cache key).
WEIGHTS = {"facebook/sam-3d-body-dinov3": "11aaa346c7204874a1cbafe3d39a979080b2c55a",
           "Ruicheng/moge-2-vitl-normal": "cb0e8bbd6b1e243589717c78e750b1ba4c093acf"}   # MIT
# Modules that must not be importable in the image (licences that forbid commercial use, tools left out).
ABSENT = ["smplx", "nvdiffrast", "diff_gaussian_rasterization", "wandb", "pytest", "black", "chump"]
absent_check = ("python -c \"import importlib.util, sys; found = [m for m in sys.argv[1:] if importlib.util.find_spec(m)];"
                " sys.exit('must not be installed: ' + ', '.join(found) if found else 0)\" " + " ".join(ABSENT))
# Optional person detectors (`infer(detector=...)`); off by default, when the caller's box is used.
DETECTORS = ("vitdet",)
TORCH_PIN = "pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121"

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0",
                 "libegl1", "libgles2", "libosmesa6", "libosmesa6-dev", "freeglut3-dev")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("numpy>=2", "cython", "setuptools", "wheel")
    .pip_install(
        "pytorch-lightning", "pyrender", "opencv-python-headless", "yacs", "scikit-image", "einops",
        "timm", "dill", "pandas", "rich", "hydra-core", "hydra-submitit-launcher", "hydra-colorlog",
        "pyrootutils", "webdataset", "networkx==3.2.1", "roma", "joblib", "seaborn",
        "appdirs", "jsonlines", "xtcocotools", "loguru", "optree",
        "fvcore", "pycocotools", "tensorboard", "huggingface_hub", "trimesh", "numpy>=2",
    )
    .env({"FORCE_CUDA": "1", "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9", "MAX_JOBS": "8", "CC": "gcc", "CXX": "g++",
          "LDSHARED": "g++ -shared"})
    .run_commands(
        # detectron2 (Apache-2.0) runs only the optional ViTDet detector; its weights are not in the image
        "pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' "
        "--no-build-isolation --no-deps",
        f"pip install git+https://github.com/microsoft/MoGe.git@{MOGE_COMMIT}",
        # MoGe's requirements upgrade torch past the build detectron2 was compiled against: pin it back.
        TORCH_PIN,
        f"git clone https://github.com/facebookresearch/sam-3d-body /repo && cd /repo && git checkout {REPO_COMMIT}",
        f"git clone https://github.com/facebookresearch/dinov3 {DINOV3_DIR} && cd {DINOV3_DIR} && "
        f"git checkout {DINOV3_COMMIT}",
        dinov3_check,
        "pip install 'PyOpenGL==3.1.7'",      # pyrender pins an old PyOpenGL; osmesa needs a newer one
        "pip install 'setuptools<70'",        # detectron2's model_zoo imports pkg_resources
        "python -c 'import torch; print(torch.__version__)' > /opt/torch-version.txt",
        absent_check,
    )
    .env({"HF_HOME": "/cache/hf", "FVCORE_CACHE": "/cache/fvcore", "TORCH_HOME": "/cache/torch",
          "PYOPENGL_PLATFORM": "osmesa", "PYTHONPATH": "/repo"})
)

app = modal.App("herogen-sam3d-body")
vol = modal.Volume.from_name(os.environ.get("HEROGEN_BODY_VOLUME", "herogen-sam3d-body-ckpt"),
                             create_if_missing=True)

_fn_kwargs = dict(image=image, timeout=TIMEOUT, volumes={"/cache": vol}, secrets=[modal.Secret.from_name("huggingface")])
if CPU_MODE:
    _fn_kwargs.update(cpu=8.0, memory=32768)
else:
    _fn_kwargs.update(gpu=GPU)


def pin_dinov3_source(src: str) -> str:
    """Upstream's backbone source with its torch.hub load pointed at the pinned local copy instead of GitHub."""
    repo, source = DINOV3_HUB_CALL
    if src.count(repo) != 1 or src.count(source) != 1:
        raise RuntimeError("SAM 3D Body's DINOv3 hub call is not the one this runner pins; update DINOV3_HUB_CALL")
    return src.replace(repo, f'"{DINOV3_DIR}",').replace(source, 'source="local",')


def _pin_dinov3():
    """Container copy only: load the backbone code from the pinned clone, never from GitHub's main branch."""
    with open(DINOV3_BACKBONE) as f:
        src = f.read()
    if f'"{DINOV3_DIR}",' not in src:
        with open(DINOV3_BACKBONE, "w") as f:
            f.write(pin_dinov3_source(src))


def _patch_repo_for_cpu():
    """Container copy only: replace the repo's hard-coded CUDA placement with CPU."""
    import re

    for rel in ["sam_3d_body/sam_3d_body_estimator.py", "sam_3d_body/models/meta_arch/sam3d_body.py"]:
        f = "/repo/" + rel
        src = open(f).read()
        src = re.sub(r'recursive_to\((\w+), "cuda"\)', r'recursive_to(\1, "cpu")', src)
        open(f, "w").write(src.replace(".cuda()", ".cpu()"))
    f = "/repo/tools/build_fov_estimator.py"
    src = open(f).read()
    open(f, "w").write(src.replace("model.infer(input_image)", "model.infer(input_image, use_fp16=False)"))


@app.function(**_fn_kwargs)
def infer(jobs: list, detector: str | None = None):
    import resource
    import subprocess
    import sys
    import threading

    import torch

    _pin_dinov3()
    cpu = not torch.cuda.is_available()
    if cpu:
        _patch_repo_for_cpu()
        torch.set_num_threads(8)
    dev = torch.device("cpu" if cpu else "cuda")
    import cv2
    import numpy as np

    sys.path.insert(0, "/repo")
    os.chdir("/repo")
    t_all = time.time()
    peak = {"mib": 0, "name": None}
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,name", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout
                peak["mib"] = max(peak["mib"], int(o.split(",")[0]))
                peak["name"] = o.split(",")[1].strip()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)

    if detector is not None and detector not in DETECTORS:
        raise ValueError(f"detector must be None or one of {DETECTORS}, got {detector!r}")
    missing = [j["name"] for j in jobs if j.get("bbox") is None]
    if missing:
        raise ValueError(f"every job needs a person box: {missing}")
    if not cpu:
        threading.Thread(target=poll, daemon=True).start()
    from huggingface_hub import hf_hub_download, snapshot_download

    repo_id = "facebook/sam-3d-body-dinov3"
    local_dir = snapshot_download(repo_id=repo_id, revision=WEIGHTS[repo_id])
    # the FOV estimator's weights at their pinned revision, handed over as a local file
    moge_weights = hf_hub_download("Ruicheng/moge-2-vitl-normal", "model.pt",
                                   revision=WEIGHTS["Ruicheng/moge-2-vitl-normal"])
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from sam_3d_body.metadata.mhr70 import mhr_names

    t0 = time.time()
    model, cfg = load_sam_3d_body(os.path.join(local_dir, "model.ckpt"), device=dev,
                                  mhr_path=os.path.join(local_dir, "assets", "mhr_model.pt"))
    from tools.build_fov_estimator import FOVEstimator

    det = None
    if detector is not None:
        from tools.build_detector import HumanDetector

        det = HumanDetector(name=detector, device=dev, path="")
    fov = FOVEstimator(name="moge2", device=dev, path=moge_weights)
    t_load = time.time() - t0
    vol.commit()
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg, human_detector=det, human_segmentor=None,
                             fov_estimator=fov)
    results, files = {}, {}
    for job in jobs:
        name = job["name"]
        img_bgr = cv2.imdecode(np.frombuffer(job["png"], np.uint8), cv2.IMREAD_COLOR)
        tmp = f"/tmp/{name}.png"
        cv2.imwrite(tmp, img_bgr)
        if not cpu:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        caller_box = np.asarray([job["bbox"]], dtype=np.float32)
        box_source = job.get("box_source") or "caller"
        if det is None:
            outs = est.process_one_image(tmp, bboxes=caller_box)
        else:
            outs = est.process_one_image(tmp, bbox_thr=0.5)
            if outs:
                box_source = f"detector ({detector})"
            else:                                       # the detector found nobody: the caller's box
                outs = est.process_one_image(tmp, bboxes=caller_box)
        if not cpu:
            torch.cuda.synchronize()
        res = {"seconds": round(time.time() - t0, 2), "n_people": len(outs), "device": str(dev), "box": box_source,
               "torch_max_allocated_mib": None if cpu else round(torch.cuda.max_memory_allocated() / 2**20),
               "torch_max_reserved_mib": None if cpu else round(torch.cuda.max_memory_reserved() / 2**20)}
        results[name] = res
        files[name] = {}
        if not outs:
            continue
        areas = [(o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]) for o in outs]
        o = outs[int(np.argmax(areas))]
        cam_t = np.asarray(o["pred_cam_t"], float)
        verts = np.asarray(o["pred_vertices"], float) + cam_t
        kp3 = np.asarray(o["pred_keypoints_3d"], float) + cam_t
        kp2 = np.asarray(o["pred_keypoints_2d"], float)
        f = float(np.asarray(o["focal_length"]).reshape(-1)[0])
        # principal point from the keypoints (uv = f * xy / z + c), then the mesh projected into the image
        c = np.median(kp2 - f * kp3[:, :2] / kp3[:, 2:3], axis=0)
        proj = f * verts[:, :2] / verts[:, 2:3] + c
        joints = {
            "frame": "camera (OpenCV): x right, y down, z forward; metres; pred_cam_t added",
            "image_size": [int(img_bgr.shape[1]), int(img_bgr.shape[0])],
            "focal_length": f, "principal_point": c.tolist(), "pred_cam_t": cam_t.tolist(),
            "bbox": np.asarray(o["bbox"], float).tolist(), "box_source": box_source,
            "mhr70": {n: {"xyz": kp3[i].tolist(), "uv": kp2[i].tolist()} for i, n in enumerate(mhr_names)},
            "mhr_skeleton_joint_coords": (np.asarray(o["pred_joint_coords"], float) + cam_t).tolist(),
            "mesh_top_uv": proj[int(np.argmin(proj[:, 1]))].tolist(),
            "mesh_bottom_uv": proj[int(np.argmax(proj[:, 1]))].tolist(),
        }
        files[name]["joints.json"] = json.dumps(joints, indent=1).encode()
        buf = io.BytesIO()
        np.savez_compressed(buf, vertices=verts.astype(np.float32), faces=np.asarray(est.faces, np.int32))
        files[name]["body.npz"] = buf.getvalue()
    stop.set()
    meta = {"device": str(dev), "gpu_name": peak["name"], "vram_peak_mib_nvidia_smi": peak["mib"],
            "max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            "load_seconds": round(t_load, 1), "total_seconds": round(time.time() - t_all, 1),
            "torch": str(torch.__version__), "repo_commit": REPO_COMMIT, "moge_commit": MOGE_COMMIT,
            "dinov3_commit": DINOV3_COMMIT, "weights": WEIGHTS,
            "repo_id": repo_id, "detector": detector,
            "runs": results}
    return meta, files


@app.local_entrypoint()
def main(job: str, out: str):
    spec = json.loads(pathlib.Path(job).read_text())
    payload = [{"name": j["name"], "png": pathlib.Path(j["image"]).read_bytes(), "bbox": j["bbox"],
                "box_source": j.get("box_source")} for j in spec["jobs"]]
    t = time.time()
    meta, files = infer.remote(payload, spec.get("detector"))
    meta["wall_seconds"] = round(time.time() - t, 1)
    meta["gpu_type"] = GPU
    o = pathlib.Path(out)
    for name, d in files.items():
        (o / name).mkdir(parents=True, exist_ok=True)
        for fn, b in d.items():
            (o / name / fn).write_bytes(b)
        (o / name / "run.json").write_text(json.dumps(meta["runs"][name], indent=2))
    (o / "run.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "runs"}))
