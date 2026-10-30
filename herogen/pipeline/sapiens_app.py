"""Sapiens 2 pose (Meta) on Modal: 308 named keypoints per image, for stage 2h's checks.

Deploy once per workspace (`modal deploy herogen/pipeline/sapiens_app.py`) so every stage-2h run, local or in the
hosted service, calls the same warm app; stop it with `modal app stop herogen-sapiens-pose`. Without a deployment,
`herogen/pipeline/sapiens.py` runs it as an ephemeral app for the length of one stage-2h run.

Imported only by `herogen/pipeline/sapiens.py` in live mode; it needs the `modal` package (`pip install
herogen[remote]`). The model is Sapiens 2 pose 1B (facebookresearch/sapiens2, the Sapiens2 License, which allows
commercial use with use restrictions: no biometric processing, no surveillance), run as upstream's top-down pipeline
on the person box given: bbox padding 1.25, UDP heatmap decoding, flip test as configured. The weights are
downloaded at run time from Hugging Face at a pinned revision into the Modal volume VOLUME and checked against their
SHA-256; nothing is shipped. Results are keypoint coordinates in image pixels (pixel centres at integers) and
scores.
"""
import os
import sys
import time

import modal

SAPIENS2_REPO = "https://github.com/facebookresearch/sapiens2"
SAPIENS2_COMMIT = "7e5bae88456ac418ff0e58e74106c9fe192055d4"
WEIGHTS = {"repo": "facebook/sapiens2-pose-1b", "revision": "f5fed8b97b99698d5eea1d14ff0855d0b4c3f000",
           "file": "sapiens2_1b_pose.safetensors",
           "sha256": "2dab7014a17e99e460c18817325a71dd7a81ce48d87027f01c2ee7d7b3af969f"}
CONFIG = ("pose/configs/keypoints308/shutterstock_goliath_3po/"
          "sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py")
VOLUME = "herogen-sapiens-runner"
GPU = "L4"  # 24 GB holds the 1B model in fp32 with room for its 1024 x 768 input
SCALEDOWN_S = 300  # a stage-2h run asks every 20-90 s; the container stays loaded between its calls
MAX_CONTAINERS = 6  # several stage-2h runs share the deployed app; more calls wait in its queue
APP_NAME = "herogen-sapiens-pose"
CLASS_NAME = "Pose"

app = modal.App(APP_NAME)
weights = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "numpy==2.2.6", "opencv-python-headless==4.11.0.86",
                 "pillow==11.2.1", "safetensors==0.5.3", "scipy==1.15.3", "matplotlib==3.10.3", "iopath==0.1.10",
                 "prettytable==3.16.0", "termcolor==3.1.0", "accelerate==1.10.1", "rich==14.0.0",
                 "transformers==4.56.1", "timm==1.0.15", "huggingface_hub==0.34.4", "tqdm==4.67.1")
    .run_commands(f"git clone {SAPIENS2_REPO} /opt/sapiens2 && cd /opt/sapiens2 && git checkout {SAPIENS2_COMMIT}"
                  " && pip install --no-deps -e .")
)
WEIGHT_PATH = f"/weights/{WEIGHTS['repo']}/{WEIGHTS['revision']}/{WEIGHTS['file']}"


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_weights():
    """The pinned weights in the volume, downloaded and checked on first use."""
    import shutil

    if not os.path.exists(WEIGHT_PATH):
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(WEIGHT_PATH), exist_ok=True)
        p = hf_hub_download(WEIGHTS["repo"], WEIGHTS["file"], revision=WEIGHTS["revision"], cache_dir="/tmp/hf")
        shutil.move(os.path.realpath(p), WEIGHT_PATH)
        if _sha256(WEIGHT_PATH) != WEIGHTS["sha256"]:
            os.remove(WEIGHT_PATH)
            raise RuntimeError("downloaded Sapiens 2 weights do not match their pinned SHA-256")
        weights.commit()


@app.cls(image=image, volumes={"/weights": weights}, gpu=GPU, timeout=1800, scaledown_window=SCALEDOWN_S,
         max_containers=MAX_CONTAINERS)
class Pose:
    @modal.enter()
    def load(self):
        import torch
        from safetensors.torch import load_file

        sys.path.insert(0, "/opt/sapiens2")
        import sapiens.pose  # noqa: F401  (registers the pose modules)
        from sapiens.engine.config import Config
        from sapiens.engine.datasets import Compose
        from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
        from sapiens.registry import MODELS as REG

        t0 = time.time()
        _fetch_weights()
        if _sha256(WEIGHT_PATH) != WEIGHTS["sha256"]:
            raise RuntimeError("Sapiens 2 weights in the volume do not match their pinned SHA-256")
        root = "/opt/sapiens2/sapiens"
        cfg = Config.fromfile(f"{root}/{CONFIG}")
        cfg.model["backbone"].pop("init_cfg", None)
        model = REG.build(cfg.model)
        inc = model.load_state_dict(load_file(WEIGHT_PATH, device="cpu"), strict=False)
        if inc.missing_keys or inc.unexpected_keys:
            raise RuntimeError("Sapiens 2 checkpoint and model do not match")
        model.data_preprocessor = REG.build(cfg.data_preprocessor)
        model.pipeline = Compose(cfg.test_pipeline)
        model.to("cuda:0").eval()
        meta = parse_pose_metainfo(dict(from_file=f"{root}/pose/configs/_base_/keypoints308.py"))
        model.flip_indices = meta["flip_indices"]
        codec = dict(cfg.codec)
        if codec.pop("type") != "UDPHeatmap":
            raise RuntimeError("unexpected Sapiens 2 pose codec")
        model.codec = UDPHeatmap(**codec)
        self.names = [meta["keypoint_id2name"][i] for i in range(meta["num_keypoints"])]
        self.flip = bool(cfg.val_cfg and cfg.val_cfg.get("flip_test", False))
        self.model = model
        torch.cuda.synchronize()
        self.load_s = time.time() - t0
        self.gpu = torch.cuda.get_device_name(0)

    @modal.method()
    def app_sha256(self) -> str:
        """This source's SHA-256 prefix, which the client compares with its own before using a deployment."""
        import hashlib
        with open(__file__, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]

    @modal.method()
    def keypoints(self, png: bytes, box: list) -> dict:
        """Keypoints of the person in ``box`` [x0, y0, x1, y1] (exclusive edges) of the PNG ``png``."""
        import cv2
        import numpy as np
        import torch

        t0 = time.time()
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        m = self.model
        data = m.pipeline(dict(img=img, bbox=np.array([box], np.float32), bbox_score=np.ones(1, np.float32)))
        data = m.data_preprocessor(data)
        inputs = data["inputs"]
        with torch.no_grad():
            pred = m(inputs)
            if self.flip:
                pred = (pred + m(inputs.flip(-1)).flip(-1)[:, m.flip_indices]) / 2.0
        kps, scores = m.codec.decode(pred.cpu().numpy()[0])
        ds = data["data_samples"]["meta"]
        kps = kps / ds["input_size"] * ds["bbox_scale"] + ds["bbox_center"] - 0.5 * ds["bbox_scale"]
        return {"names": self.names, "keypoints": np.asarray(kps[0]).round(3).tolist(),
                "scores": np.asarray(scores[0]).round(4).tolist(), "box": [float(v) for v in box],
                "gpu": self.gpu, "load_s": round(self.load_s, 2), "seconds": round(time.time() - t0, 3)}
