"""The service's adapter to the pipeline: which stages a job runs, how each is started, and what the user is told.

Each stage runs as the same command a person would type (`python3 -m herogen.pipeline run ...`,
`python3 -m herogen.blender run ...`), in a subprocess with a timeout and an environment holding only what that
stage needs (the Model API key reaches the model stages only). Its outcome is read from the stage record it
writes (`stage-<n>.json`, docs/ARCHITECTURE.md): ok means passed, a record with a failed gate means the pipeline
rejected the sketch (shown to the user with the checks that failed), and no record means an error (logged in
full, shown as one sentence). Options that only some versions of a stage CLI have (`--profile`, `--remote-cache`)
are passed only when that CLI's `--help` lists them, so the service follows whichever pipeline it is deployed with.

The Roblox rig (stage 6) runs only when the deployed code has a `herogen.roblox` command; until then the stage is
skipped with that reason and the job still delivers its GLB.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger("herogen.service")


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    group: str            # "model" (Model API), "blender" (Blender, optional remote GPU), "export" (CPU)
    record: str           # the record that decides the outcome
    prompt_only: bool = False
    extra: tuple[str, ...] = ()   # records the stage also writes; their cost is counted, their gate is not
    optional: bool = False        # a rejection or error is recorded and shown, but the job goes on without it


PLAN = (
    StageSpec("p", "sketch from your description", "model", "stage-p.json", prompt_only=True),
    StageSpec("0", "input", "model", "stage-0.json"),
    StageSpec("1", "cover art", "model", "stage-1.json"),
    StageSpec("2", "turnaround", "model", "stage-2.json", extra=("stage-2h.json",)),
    StageSpec("3", "base mesh", "blender", "stage-3.json"),
    StageSpec("4", "game body", "blender", "stage-4.json"),
    StageSpec("6", "Roblox rig", "export", "stage-6.json", optional=True),
    StageSpec("5", "results page", "export", "stage-5.json", optional=True),
)
GROUPS = ("model", "blender", "export")
STAGE_LABELS = {s.key: s.label for s in PLAN} | {"2h": "held-out view"}
INPUT_LABELS = {"sketch": "uploaded sketch", "prompt": "sketch drawn from a description"}

TIPS = ("Sketches work best with one character drawn full body from head to feet, standing and facing forward, "
        "arms held a little away from the body, on plain paper.")
# Why a stage's gate failed, by check name prefix; the measured values follow in the message.
REASONS = (
    ("0", "min_side_px", "The sketch is too small."),
    ("1", "sketch_mask", "No single full-body figure was found in the sketch."),
    ("1", "cover_mask", "The colour version lost the figure from the sketch."),
    ("1", "silhouette_iou", "The colour version drifted too far from the sketch's outline."),
    ("2", "figures_found", "The turnaround sheet did not show four separate figures."),
    ("2", "a_pose", "In the front view the arms were not held away from the body."),
    ("2", "top", "The top view did not match the other views."),
    ("2", "", "The four turnaround views did not agree with each other."),
    ("3", "silhouette_iou", "The 3D shape did not match the views closely enough."),
    ("3", "", "The 3D shape could not be built cleanly from the views."),
    ("4", "max_joint_gap", "The body opened gaps at the joints when it was posed."),
    ("4", "triangles", "The body did not fit the triangle budget of the chosen profile."),
    ("4", "texture", "The colours on the body did not match the views closely enough."),
    ("4", "rest_iou", "The body's shape did not match the views closely enough."),
    ("4", "", "The game body did not pass its checks."),
    ("6", "", "The Roblox rig could not be built from this body."),
    ("p", "", "The sketch drawn from your description could not be used."),
)


@dataclass
class Outcome:
    state: str                    # passed, rejected, failed, skipped
    seconds: float = 0.0
    cost_usd: float = 0.0         # model and remote-GPU estimate from the stage record
    message: str = ""             # for the user
    code: str = ""                # stable error code
    transient: bool = False       # an infrastructure error worth another attempt
    detail: str = ""              # for the logs only


@dataclass
class StageContext:
    job_id: str
    kind: str                     # sketch or prompt
    profile: str
    name: str
    prompt: str
    run_dir: Path                 # runs_root / job_id
    input_path: Path
    cache_dir: Path
    remote_cache_dir: Path
    stage3_args: tuple[str, ...] = ()
    timeout_s: float = 3600.0
    blender_timeout_s: float = 3 * 3600.0
    env: dict[str, str] = field(default_factory=dict)

    @property
    def runs_root(self) -> Path:
        return self.run_dir.parent


class StageRunner(Protocol):
    def run(self, spec: StageSpec, ctx: StageContext) -> Outcome: ...

    def available(self, spec: StageSpec) -> str | None:
        """None when the stage can run here, else why not (shown to the user)."""
        ...


ROBLOX_MISSING = "Roblox export is not part of this version of HeroGen yet."


def roblox_available() -> bool:
    try:
        return importlib.util.find_spec("herogen.roblox.__main__") is not None
    except (ImportError, ValueError):
        return False


def plan_for(kind: str) -> list[StageSpec]:
    return [s for s in PLAN if kind == "prompt" or not s.prompt_only]


def read_record(run_dir: Path, name: str) -> dict[str, Any] | None:
    try:
        rec = json.loads((run_dir / name).read_text())
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def record_ok(run_dir: Path, spec: StageSpec) -> bool:
    rec = read_record(run_dir, spec.record)
    return bool(rec and rec.get("ok") is True)


def resume_index(run_dir: Path, plan: list[StageSpec]) -> int:
    """How many leading stages of `plan` already passed in this run folder (a retried job starts after them)."""
    n = 0
    for spec in plan:
        if not record_ok(run_dir, spec):
            break
        n += 1
    return n


def _fmt(v: Any) -> str:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def failed_checks(rec: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for c in rec.get("checks") or []:
        if isinstance(c, dict) and c.get("gate", True) and c.get("pass") is False:
            out.append(c)
    return out


def describe_rejection(spec: StageSpec, rec: dict[str, Any]) -> str:
    """A sentence or two for the user: what the stage found wrong, then the measured checks that failed."""
    checks = failed_checks(rec)
    reason = next((text for stage, prefix, text in REASONS
                   if stage == spec.key and any(str(c.get("name", "")).startswith(prefix) for c in checks)
                   and prefix), None)
    if reason is None:
        reason = next((text for stage, prefix, text in REASONS if stage == spec.key and not prefix),
                      "The stage did not pass its checks.")
    shown = []
    for c in checks[:4]:
        op = c.get("op", ">=")
        shown.append(f"{c.get('name')} measured {_fmt(c.get('value'))}, needs {op} {_fmt(c.get('threshold'))}")
    more = f" and {len(checks) - 4} more" if len(checks) > 4 else ""
    measured = f" Checks that failed: {'; '.join(shown)}{more}." if shown else ""
    tip = f" {TIPS}" if spec.key in ("0", "1", "2", "p") else ""
    return f"The {spec.label} stage rejected this hero. {reason}{measured}{tip}"


# Transient failures, recognised by what the model client and the remote client report: an HTTP 429 or 5xx status
# ("HTTP 503: ..."), a timeout, or a network error. Any other error (a 4xx, a bad response, a bug) is not retried.
_HTTP_STATUS = re.compile(r"\bHTTP (\d{3})\b")
_NETWORK = re.compile(r"\b(TimeoutError|timed out|URLError|ConnectionResetError|ConnectionRefusedError|"
                      r"RemoteDisconnected|IncompleteRead)\b")


def transient_failure(text: str) -> bool:
    statuses = {int(s) for s in _HTTP_STATUS.findall(text)}
    if any(s == 429 or 500 <= s <= 599 for s in statuses):
        return True
    return bool(_NETWORK.search(text))
_SCRUB = (re.compile(r"(?i)bearer\s+\S+"), re.compile(r"(?i)(api[_-]?key|token|authorization)\S*\s*[:=]\s*\S+"))


def scrub(text: str) -> str:
    for pat in _SCRUB:
        text = pat.sub("[redacted]", text)
    return text


def describe_error(spec: StageSpec, text: str, timed_out: bool) -> tuple[str, str, bool]:
    """(code, user message, transient) for a stage that stopped without a record."""
    if timed_out:
        return "timeout", f"The {spec.label} stage took too long and was stopped.", True
    if spec.group == "model" and transient_failure(text):
        return "model_unavailable", "The image model service did not answer properly.", True
    if spec.group == "blender" and "RemoteError" in text and transient_failure(text):
        return "gpu_unavailable", "The 3D reconstruction service did not answer properly.", True
    return "stage_error", f"The {spec.label} stage stopped with an internal error.", False


def _cli_flags(module: str, env: dict[str, str]) -> str:
    try:
        proc = subprocess.run([sys.executable, "-m", module, "run", "--help"], capture_output=True, text=True,
                              timeout=60, env=env)
        return proc.stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")[:40]
    return s or "sketch"


def base_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment every stage subprocess gets: no secrets, only what Python and Blender need."""
    keep = ("PATH", "LANG", "LC_ALL", "TZ", "HEROGEN_BLENDER", "HEROGEN_MODAL", "SYSTEMROOT", "TMPDIR")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = root
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("LANG", "C.UTF-8")
    env.update(extra or {})
    return env


class CliRunner:
    """Runs each stage with its own CLI in a subprocess."""

    def __init__(self, model_api_key: str = "", python: str = sys.executable) -> None:
        self._model_key = model_api_key
        self.python = python
        self._flags: dict[str, str] = {}

    def available(self, spec: StageSpec) -> str | None:
        if spec.key == "6" and not roblox_available():
            return ROBLOX_MISSING
        return None

    def flags(self, module: str, env: dict[str, str]) -> str:
        if module not in self._flags:
            self._flags[module] = _cli_flags(module, env)
        return self._flags[module]

    def command(self, spec: StageSpec, ctx: StageContext, env: dict[str, str], work: Path
                ) -> tuple[list[str], str | None]:
        """(argv, stdin text) for a stage; `work` is a private temporary folder for this call."""
        py = self.python
        pipe = [py, "-m", "herogen.pipeline", "run"]
        common = ["--run-id", ctx.job_id, "--runs", str(ctx.runs_root), "--cache", str(ctx.cache_dir)]
        if spec.key == "p":
            return ([py, "-m", "herogen.service.sketchgen", "--run", str(ctx.run_dir), "--out", str(ctx.input_path),
                     "--cache", str(ctx.cache_dir)], ctx.prompt)
        if spec.key == "0":
            src = ctx.input_path
            if ctx.name:
                named = work / "input" / f"{_slug(ctx.name)}.png"
                named.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ctx.input_path, named)
                src = named
            return pipe + [str(src), *common, "--stages", "0", "--live", "--label", INPUT_LABELS[ctx.kind]], None
        if spec.key in ("1", "2"):
            return pipe + common + ["--stages", spec.key, "--live"], None
        if spec.key == "5":
            return pipe + common + ["--stages", "5"], None
        if spec.key in ("3", "4"):
            help_text = self.flags("herogen.blender", env)
            argv = [py, "-m", "herogen.blender", "run", "--run", str(ctx.run_dir), "--stages", spec.key,
                    "--timeout", str(int(ctx.blender_timeout_s))]
            if spec.key == "3":
                argv += list(ctx.stage3_args)
                if "--remote-cache" in help_text:
                    argv += ["--remote-cache", str(ctx.remote_cache_dir)]
            if spec.key == "4" and "--profile" in help_text:
                argv += ["--profile", ctx.profile]
            return argv, None
        if spec.key == "6":
            # With --out a run's 06-roblox folder, the export writes stage-6.json at the run root, its
            # outputs relative to the run (herogen/roblox, docs/ARCHITECTURE.md "Roblox export").
            return [py, "-m", "herogen.roblox", "export", "--body", str(ctx.run_dir / "04-body"), "--out",
                    str(ctx.run_dir / "06-roblox")], None
        raise ValueError(f"unknown stage {spec.key}")

    def run(self, spec: StageSpec, ctx: StageContext) -> Outcome:
        why = self.available(spec)
        if why:
            return Outcome("skipped", message=why, code="not_available")
        extra = dict(ctx.env)
        if spec.group == "model" and self._model_key:
            extra["MODEL_API_KEY"] = self._model_key
        with tempfile.TemporaryDirectory(prefix="herogen-stage-") as home:
            env = base_env({"HOME": home, **extra})
            argv, stdin = self.command(spec, ctx, env, Path(home))
            for stale in (spec.record, *spec.extra):
                p = ctx.run_dir / stale
                if p.exists():
                    p.unlink()
            timeout = ctx.blender_timeout_s if spec.group == "blender" else ctx.timeout_s
            t0 = time.monotonic()
            timed_out = False
            try:
                proc = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=timeout, env=env,
                                      cwd=home)
                out, err, code = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired as e:
                timed_out = True
                out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
                err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
                code = -1
            seconds = round(time.monotonic() - t0, 2)
        return self.outcome(spec, ctx, seconds, code, out, err, timed_out)

    @staticmethod
    def outcome(spec: StageSpec, ctx: StageContext, seconds: float, code: int, out: str, err: str,
                timed_out: bool) -> Outcome:
        rec = read_record(ctx.run_dir, spec.record)
        tail = scrub((out[-3000:] + "\n--- stderr ---\n" + err[-6000:]).replace(str(Path.home()), "~"))
        cost = 0.0
        for r in [rec] + [read_record(ctx.run_dir, x) for x in spec.extra]:
            c = r.get("cost_usd") if r else None
            cost += float(c) if isinstance(c, (int, float)) and math.isfinite(c) else 0.0
        if rec is not None and rec.get("ok") is True and not timed_out:
            return Outcome("passed", seconds, cost, detail=tail if code else "")
        if rec is not None and rec.get("ok") is False and rec.get("error") and not timed_out:
            # The stage could not use its input at all: an error, not a rejection. The record's own error line
            # is shown only when it holds no path (it may name files on the worker).
            err = str(rec["error"])
            shown = f" {err.rstrip('.')}." if "/" not in err and "\\" not in err and len(err) < 300 else ""
            return Outcome("failed", seconds, cost, f"The {spec.label} stage could not use this body.{shown}",
                           "stage_error", False, detail=tail)
        if rec is not None and rec.get("ok") is False and not timed_out:
            return Outcome("rejected", seconds, cost, describe_rejection(spec, rec), "rejected", detail=tail)
        code_s, message, transient = describe_error(spec, tail, timed_out)
        return Outcome("failed", seconds, cost, message, code_s, transient, detail=tail)


def write_record(run_dir: Path, record: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / f"stage-{record['stage']}.json"
    part = p.with_name(p.name + ".part")
    part.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(part, p)


def stage3_args(raw: str) -> tuple[str, ...]:
    return tuple(shlex.split(raw)) if raw else ()


Callback = Callable[[StageSpec, Outcome], None]
