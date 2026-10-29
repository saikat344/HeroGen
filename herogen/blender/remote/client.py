"""The one remote-model client for herogen/blender: SAM 3D Objects and SAM 3D Body on Modal, with record/replay.

Modes:
- ``live``: run the Modal app (`modal run <runner>.py::main`, an ephemeral app that stops when the call
  returns) in a subprocess with a timeout, then store its outputs in the cache.
- ``replay``: read the cache only. A missing entry raises ``ReplayMiss``; nothing touches the network.

The cache key hashes the request: the runner (its name and the SHA-256 of its source file, `runner_sha`), the model
code commit (`rev`), the pinned Hugging Face weight revisions (`weights`; for SAM 3D Body also MoGe's code commit),
the renderer, the parameters and the SHA-256 of every input image. A replay therefore returns exactly what the live
call returned for the same inputs from the same runner; an entry recorded by a different runner (another source,
other weights) is not used, and replay says so. Each entry is a directory
`<cache>/<key>/` holding the runner's output files and `entry.json` (request summary, run log, wall seconds
and a cost estimate). Secrets never pass through this module: the runners read the Hugging Face token from a
Modal secret inside the container, and the stored run log is scrubbed of anything that looks like a token, of
the Modal workspace name in its URLs and of the home directory.

The GPU type is not part of the key; a CPU run (SAM 3D Body with gpu none) is keyed separately, since its numerics
differ. SAM 3D Objects and SAM 3D Body are released under the SAM License; this module and the runners copy none of
their code.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNERS = {"sam3d-objects": HERE / "objects.py", "sam3d-body": HERE / "body.py"}
OBJECTS_RENDERER = "pytorch3d"       # texture bake and rasterisation without nvdiffrast (see objects.py)


def _runner_constants(path: Path) -> dict:
    """The literal module-level constants of a runner file (read, not imported: the runners need modal)."""
    import ast

    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def runner_fingerprint(runner: str) -> dict:
    """What a runner's results depend on besides its inputs: its code commit, weight revisions and source hash."""
    path = RUNNERS[runner]
    c = _runner_constants(path)
    fp = {"rev": (f"sam-3d-objects@{c['REPO_SHA']}" if runner == "sam3d-objects"
                  else f"sam-3d-body@{c['REPO_COMMIT']}"),
          "weights": dict(sorted(c["WEIGHTS"].items())),
          "runner_sha": hashlib.sha256(path.read_bytes()).hexdigest()[:16]}
    if runner == "sam3d-body":
        fp["moge"] = c["MOGE_COMMIT"]
        fp["dinov3"] = c["DINOV3_COMMIT"]
    return fp


FINGERPRINT_KEYS = ("rev", "weights", "runner_sha", "moge", "dinov3")
# Modal list prices per second (USD), for estimates only; billed actuals come from `modal billing report`.
PRICE_PER_SECOND = {"A100-40GB": 2.10 / 3600, "L4": 0.80 / 3600, "A10G": 1.10 / 3600, "T4": 0.59 / 3600,
                    "none": 8 * 0.047 / 3600 + 32 * 0.008 / 3600}
COST_BASIS = "estimate: wall seconds of the Modal call x list price of its GPU (or 8 CPU + 32 GB)"
COST_UNKNOWN = "unknown: no list price recorded for this GPU"
DEFAULT_TIMEOUT = 1800.0
DEFAULT_OBJECTS_GPU = "A100-40GB"
# A10G: measured 5.1-6.3 s per image; an L4 request was never scheduled in 25 minutes on the measured run.
DEFAULT_BODY_GPU = "A10G"


class RemoteError(RuntimeError):
    """A remote call failed (Modal missing, non-zero exit, timeout, or missing outputs). The message is one line;
    `log` holds the scrubbed tail of the run log, when there is one."""

    def __init__(self, message: str, log: str | None = None):
        super().__init__(" ".join(str(message).split()))
        self.log = log


class ReplayMiss(RemoteError):
    """Replay mode found no cache entry for a request."""


@dataclass
class RemoteResult:
    dir: Path                 # directory holding the output files
    entry: dict               # request summary, run log, seconds, cost estimate
    cached: bool

    def path(self, name: str) -> Path:
        return self.dir / name


@dataclass
class CallLog:
    calls: list = field(default_factory=list)

    def add(self, runner: str, key: str, cached: bool, cost: float | None) -> None:
        self.calls.append({"runner": runner, "key": key, "cached": cached, "cost_usd": 0.0 if cached else cost})

    def cost_usd(self) -> float | None:
        """The calls' estimated cost, or None when a call's GPU has no recorded price."""
        costs = [c["cost_usd"] for c in self.calls]
        return None if any(c is None for c in costs) else round(sum(costs), 4)


def _write_atomic(path: Path, text: str) -> None:
    """Write through a temporary file and a rename: a reader never sees a partial file."""
    tmp = path.with_name(path.name + ".part")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request_key(summary: dict) -> str:
    canon = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:24]


def _scrub(text: str) -> str:
    """The run log as it is stored: tokens, the Modal workspace (in dashboard and endpoint URLs) and the home
    directory removed."""
    text = re.sub(r"hf_[A-Za-z0-9]{16,}", "hf_redacted", text)
    text = re.sub(r"(?i)bearer\s+\S+", "bearer redacted", text)
    text = re.sub(r"(?i)(token|authorization)([\"'=: ]+)(?!bearer redacted)\S+", r"\1\2redacted", text)
    text = re.sub(r"(https?://modal\.com/(?:apps|id|logs|storage|secrets|settings|deployments)/)[^/\s]+",
                  r"\1workspace", text)
    text = re.sub(r"https?://[A-Za-z0-9-]+--[A-Za-z0-9.-]*\.modal\.run\S*", "https://workspace.modal.run", text)
    home = str(Path.home())
    return text.replace(home, "~") if len(home) > 1 else text


def find_modal() -> str | None:
    for cand in (os.environ.get("HEROGEN_MODAL"), shutil.which("modal"),
                 str(Path.home() / ".local" / "bin" / "modal")):
        if cand and Path(cand).exists():
            return cand
    return None


def objects_summary(image: Path, mask: Path, seed: int = 42) -> dict:
    """The request summary (cache key source) of a SAM 3D Objects call."""
    return {"runner": "sam3d-objects", **runner_fingerprint("sam3d-objects"), "renderer": OBJECTS_RENDERER,
            "image": sha256(image), "mask": sha256(mask), "seed": seed}


BODY_DETECTORS = (None, "vitdet")


def body_summary(image: Path, bbox: list, box_source: str, detector: str | None = None) -> dict:
    """The request summary (cache key source) of a SAM 3D Body call: the person box, where it came from and the
    optional person detector (None: off) are part of the key."""
    if bbox is None or len(bbox) != 4 or not box_source:
        raise ValueError("SAM 3D Body needs a person box [x0, y0, x1, y1] and its source")
    if detector not in BODY_DETECTORS:
        raise ValueError(f"SAM 3D Body's detector must be one of {BODY_DETECTORS}, got {detector!r}")
    return {"runner": "sam3d-body", **runner_fingerprint("sam3d-body"), "image": sha256(image),
            "bbox": [float(x) for x in bbox], "box_source": box_source, "detector": detector}


class RemoteClient:
    def __init__(self, cache_dir: Path | str, mode: str = "replay", timeout: float = DEFAULT_TIMEOUT,
                 log: CallLog | None = None, body_gpu: str = DEFAULT_BODY_GPU,
                 objects_gpu: str = DEFAULT_OBJECTS_GPU):
        if mode not in ("live", "replay"):
            raise ValueError(f"mode must be 'live' or 'replay', got {mode!r}")
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.timeout = timeout
        self.log = log if log is not None else CallLog()
        self.body_gpu = body_gpu
        self.objects_gpu = objects_gpu

    # ---- public calls ---------------------------------------------------------------------------------

    def objects(self, image: Path, mask: Path, seed: int = 42, gpu: str | None = None) -> RemoteResult:
        """SAM 3D Objects: `mesh.glb` (textured source mesh) and `run.json`. `gpu` defaults to the client's
        `objects_gpu`."""
        gpu = gpu or self.objects_gpu
        summary = objects_summary(image, mask, seed)
        return self._call("objects", summary, {"jobs": [{"name": "job", "image": str(Path(image).resolve()),
                                                         "mask": str(Path(mask).resolve())}], "seed": seed},
                          {"HEROGEN_OBJECTS_GPU": gpu, "HEROGEN_OBJECTS_TIMEOUT": str(int(self.timeout))}, gpu,
                          required=("mesh.glb",))

    def body(self, image: Path, bbox: list, box_source: str, gpu: str | None = None,
             detector: str | None = None) -> RemoteResult:
        """SAM 3D Body: `joints.json`, `body.npz` and `run.json`. `bbox` [x0, y0, x1, y1] is the person box and
        `box_source` says where it came from. `detector="vitdet"` runs upstream's person detector instead (off by
        default), with `bbox` used only if it finds nobody. The box, its source and the detector are in the cache
        key. `gpu="none"` runs on CPU (default: the client's `body_gpu`); a CPU run keys separately, the GPU type
        does not enter the key."""
        gpu = gpu or self.body_gpu
        summary = body_summary(image, bbox, box_source, detector)
        if gpu.lower() == "none":
            summary["device"] = "cpu"           # CPU numerics differ from the GPU's: never replay one for the other
        return self._call("body", summary, {"jobs": [{"name": "job", "image": str(Path(image).resolve()),
                                                      "bbox": summary["bbox"], "box_source": box_source}],
                                            "detector": detector},
                          {"HEROGEN_BODY_GPU": gpu, "HEROGEN_BODY_TIMEOUT": str(int(self.timeout))}, gpu,
                          required=("joints.json",))

    # ---- internals ------------------------------------------------------------------------------------

    def _stale(self, summary: dict) -> str | None:
        """For a miss: the fingerprint fields that differ in an entry for the same runner and inputs, if any."""
        if not self.cache_dir.is_dir():
            return None
        inputs = {k: v for k, v in summary.items() if k not in FINGERPRINT_KEYS}
        for entry_path in self.cache_dir.glob("*/entry.json"):
            try:
                req = json.loads(entry_path.read_text()).get("request", {})
            except (OSError, ValueError):
                continue
            if {k: v for k, v in req.items() if k not in FINGERPRINT_KEYS} == inputs:
                return ", ".join(k for k in FINGERPRINT_KEYS if req.get(k) != summary.get(k)) or "fields"
        return None

    def _call(self, runner: str, summary: dict, job: dict, env: dict, gpu: str, required: tuple) -> RemoteResult:
        key = request_key(summary)
        d = self.cache_dir / key
        entry_path = d / "entry.json"
        if entry_path.exists():
            entry = json.loads(entry_path.read_text())
            self.log.add(summary["runner"], key, True, 0.0)
            return RemoteResult(d, entry, cached=True)
        if self.mode == "replay":
            # the cache folder's name only: this text ends up in stage records, which must hold no local paths
            stale = self._stale(summary)
            if stale:
                raise ReplayMiss(f"the cached {summary['runner']} result for these inputs in the remote cache "
                                 f"({self.cache_dir.name}) was made by a different runner ({stale} differ); not "
                                 "used: record it again with --remote live")
            raise ReplayMiss(f"no cached {summary['runner']} result for key {key} in the remote cache "
                             f"({self.cache_dir.name})")
        modal = find_modal()
        if modal is None:
            raise RemoteError("modal CLI not found (set HEROGEN_MODAL)")
        with tempfile.TemporaryDirectory(prefix=f"herogen-{runner}-") as tmp:
            tmp = Path(tmp)
            (tmp / "job.json").write_text(json.dumps(job))
            out = tmp / "out"
            out.mkdir()
            cmd = [modal, "run", str(HERE / f"{runner}.py") + "::main", "--job", str(tmp / "job.json"),
                   "--out", str(out)]
            t0 = time.monotonic()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout,
                                      env={**os.environ, **env})
            except subprocess.TimeoutExpired as exc:
                raise RemoteError(f"{runner} timed out after {self.timeout:.0f} s") from exc
            except OSError as exc:              # modal found but not runnable (permissions, broken install)
                raise RemoteError(f"{runner}: could not start modal ({type(exc).__name__}: {exc.strerror or exc})"
                                  ) from exc
            wall = time.monotonic() - t0
            logtext = _scrub(proc.stdout[-20000:] + "\n--- stderr ---\n" + proc.stderr[-20000:])
            outputs = out / "job"
            missing = [r for r in required if not (outputs / r).exists()]
            if proc.returncode != 0 or missing:
                raise RemoteError(f"{runner} failed (exit {proc.returncode}, missing outputs {missing or 'none'}; "
                                  "the log tail is on the error)", log=logtext[-2000:])
            d.mkdir(parents=True, exist_ok=True)
            for f in outputs.iterdir():
                shutil.copy2(f, d / f.name)
            run_log = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else {}
            (d / "modal.log").write_text(logtext)
        price = PRICE_PER_SECOND.get(gpu)
        cost = None if price is None else round(wall * price, 4)
        entry = {"request": summary, "gpu": gpu, "wall_seconds": round(wall, 1), "run": run_log,
                 "cost_usd_estimate": cost, "cost_basis": COST_BASIS if price is not None else COST_UNKNOWN,
                 "files": sorted(f.name for f in d.iterdir() if f.name != "entry.json")}
        _write_atomic(entry_path, json.dumps(entry, indent=2))   # the entry is what makes the folder a cache hit
        self.log.add(summary["runner"], key, False, cost)
        return RemoteResult(d, entry, cached=False)
