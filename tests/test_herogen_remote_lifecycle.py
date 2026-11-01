"""Exercise the real runner initialization and request code with local model-library substitutes."""
from __future__ import annotations

import ast
import io
import os
import pathlib
import sys
import time
import types
import unittest
from functools import lru_cache
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

REMOTE = pathlib.Path(__file__).resolve().parents[1] / "herogen/blender/remote"


def runner(name, namespace):
    tree = ast.parse((REMOTE / name).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_model_state", "infer")]
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model")
    model.decorator_list = []
    for fn in model.body:
        if isinstance(fn, ast.FunctionDef):
            fn.decorator_list = []
    namespace.update(lru_cache=lru_cache, time=time, os=os, pathlib=pathlib, io=io)
    exec(compile(ast.Module(nodes + [model], []), name, "exec"), namespace)
    return namespace


def module(**kwargs):
    return types.SimpleNamespace(**kwargs)


class PersistentModelTest(unittest.TestCase):
    def test_objects_loads_at_startup_once_and_keeps_seed_and_export_parameters_per_request(self):
        class Tensor:
            pass

        torch = module(Tensor=Tensor, cuda=module(reset_peak_memory_stats=Mock(), synchronize=Mock(),
                       max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0))
        glb = module(export=Mock(return_value=b"same mesh"), vertices=[1], faces=[1],
                     visual=module(material=module(baseColorTexture=module(size=(32, 32)))))
        pipe = module(run=Mock(return_value={"glb": glb}))
        instantiate = Mock(return_value=pipe)
        config = module(depth_model=module(model=module(pretrained_model_name_or_path="depth")))
        nb = module(check_hydra_safety=Mock(), WHITELIST_FILTERS=[], BLACKLIST_FILTERS=[],
                    Inference=module(merge_mask_to_rgba=lambda _, image, mask: (image, mask)))
        render = module(render_frames=Mock())
        modules = {"torch": torch, "hydra.utils": module(instantiate=instantiate),
                   "omegaconf": module(OmegaConf=module(load=Mock(return_value=config))),
                   "inference": nb, "sam3d_objects.model.backbone.tdfy_dit.utils": module(render_utils=render),
                   "utils3d": module(torch=module()), "nvdiffrast": None}
        samplers = []

        def sampler():
            s = module(stop=False, peak=0)
            samplers.append(s)
            return s

        ns = runner("objects.py", {"CKPT_DIR": "/ckpt", "WEIGHTS": {"facebook/sam-3d-objects": "pin", "depth": "pin"},
                    "REPO_SHA": "repo", "ABSENT": [], "checkpoint_revision": lambda _: "pin",
                    "gsplat_backend": lambda f: f, "Pytorch3dRastContext": object,
                    "pytorch3d_rasteriser": object(), "VramSampler": sampler})
        png = io.BytesIO()
        Image.new("RGB", (32, 32), "white").save(png, format="PNG")
        jobs = [{"name": "job", "image": png.getvalue(), "mask": png.getvalue()}]
        original_path = list(sys.path)
        try:
            with patch.dict(sys.modules, modules), patch.dict(os.environ), patch("os.chdir"), \
                    patch("subprocess.run", return_value=module(stdout="GPU")), \
                    patch("importlib.util.find_spec", return_value=None):
                model = ns["Model"]()
                model.load()
                first = model.infer(jobs, 42)
                second = model.infer(jobs, 7)
                self.assertEqual(first[1], second[1])
                self.assertEqual(first[0]["load_seconds"], 0)
                self.assertEqual(second[0]["seed"], 7)
                self.assertEqual(instantiate.call_count, 1)
                self.assertEqual([c.args[2] for c in pipe.run.call_args_list], [42, 7])
                self.assertEqual(pipe.run.call_args.kwargs, dict(stage1_only=False, with_mesh_postprocess=True,
                                 with_texture_baking=True, with_layout_postprocess=False, use_vertex_color=False))
                pipe.run.side_effect = RuntimeError("inference failed")
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    model.infer(jobs)
                self.assertTrue(all(s.stop for s in samplers))
        finally:
            sys.path[:] = original_path

    def test_body_pins_and_loads_once_before_requests_and_reuses_the_same_box(self):
        est = module(process_one_image=Mock(return_value=[]))
        load = Mock(return_value=("model", "cfg"))
        constructor = Mock(return_value=est)
        pin = Mock()
        torch = module(cuda=module(is_available=lambda: False), set_num_threads=Mock(),
                       device=lambda x: x, __version__="test")
        modules = {"torch": torch, "cv2": module(IMREAD_COLOR=1, imdecode=lambda *a: np.zeros((32, 32, 3)),
                   imwrite=Mock()), "huggingface_hub": module(snapshot_download=Mock(return_value="/weights"),
                   hf_hub_download=Mock(return_value="/moge")),
                   "sam_3d_body": module(SAM3DBodyEstimator=constructor, load_sam_3d_body=load),
                   "sam_3d_body.metadata.mhr70": module(mhr_names=[]),
                   "tools.build_fov_estimator": module(FOVEstimator=Mock(return_value="fov"))}
        ns = runner("body.py", {"_pin_dinov3": pin, "_patch_repo_for_cpu": Mock(), "vol": module(commit=Mock()),
                    "WEIGHTS": {"facebook/sam-3d-body-dinov3": "body-pin", "Ruicheng/moge-2-vitl-normal": "moge-pin"},
                    "REPO_COMMIT": "repo", "MOGE_COMMIT": "moge", "DINOV3_COMMIT": "dino", "DETECTORS": ("vitdet",)})
        original_path = list(sys.path)
        try:
            with patch.dict(sys.modules, modules), patch("os.chdir"):
                model = ns["Model"]()
                model.load()
                self.assertEqual(load.call_count, 1)
                jobs = [{"name": "job", "png": b"test", "bbox": [1, 2, 3, 4], "box_source": "mask"}]
                first, second = model.infer(jobs), model.infer(jobs)
                self.assertEqual(first[1], second[1])
                self.assertEqual(second[0]["load_seconds"], 0)
                self.assertEqual(load.call_count, 1)
                self.assertEqual(pin.call_count, 1)
                self.assertEqual(constructor.call_count, 1)
                self.assertEqual(est.process_one_image.call_count, 2)
                for call in est.process_one_image.call_args_list:
                    np.testing.assert_array_equal(call.kwargs["bboxes"], [[1, 2, 3, 4]])
        finally:
            sys.path[:] = original_path


if __name__ == "__main__":
    unittest.main()
