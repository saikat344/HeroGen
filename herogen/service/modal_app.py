"""The hosted service on Modal. Deploy from the repository root:

    modal deploy -m herogen.service.modal_app

Non-secret settings are the `HEROGEN_SERVICE_*` variables set in the deploying shell (they are baked into the
image; see docs/ARCHITECTURE.md, "Service"); `HEROGEN_SERVICE_APP` names the app and everything it owns, so
`HEROGEN_SERVICE_APP=herogen-service-staging` deploys a separate staging copy with its own volumes and secrets.

What runs where:
- `StoreService`: the metadata store, one container at most (the single writer), SQLite on local disk with a
  snapshot on the `<app>-meta` volume after every write (`snapshot.py`).
- `web`: the ASGI app (Starlette), many concurrent requests per container; holds the `<app>-web` secret
  (HEROGEN_SESSION_KEY, HEROGEN_EMAIL_API_KEY); reads and writes the `<app>-data` volume; reads `<app>-legal`.
  It can call the store's service methods only; admin methods need the `<app>-admin` key (HEROGEN_ADMIN_KEY),
  which only `StoreService` and `admin_command` hold.
- `sanitize_image`: opens uploads, with no network, no secrets and no volumes.
- `run_job`: one orchestrator per job attempt; `model_stages` (holds `<app>-model`: MODEL_API_KEY),
  `blender_stages` (Blender 4.5 LTS; SAM 3D through the deployed GPU apps by default, and
  the model secret for stage 3's usability question) and `export_stages` run the stage groups on local copies of the job's files, each in a fresh container
  (`single_use_containers`), so nothing of one job is left where another job runs.
- `sweep` (every minute): requeues jobs whose worker died and starts queued jobs; `retention` (daily) deletes
  expired files and records.
- `admin` (local entrypoint): `modal run -m herogen.service.modal_app::admin --args "invite --count 3"`. The
  `snapshots` and `snapshot-quarantine` commands work on the `<app>-meta` volume directly, so they also work
  while the store cannot start.
- `selfcheck` (local entrypoint): reports what the worker images hold (Blender version, the stage CLIs' options,
  whether the Roblox export is present).

Deploying refuses dev mode, and every function's timeout follows the stage timeouts (`TIMEOUTS`, `JOB_TIMEOUT`).
No function sets Modal `retries`: Modal starts an input again after a container dies regardless, so retries are
the store's job (`job_claim`, `group_enter`, attempts and leases), and every job function is nonpreemptible.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal

from herogen.service import config as service_config

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
APP_NAME = os.environ.get("HEROGEN_SERVICE_APP", "herogen-service")
WEB_SECRET = f"{APP_NAME}-web"
MODEL_SECRET = f"{APP_NAME}-model"
ADMIN_SECRET = f"{APP_NAME}-admin"
DATA, META, LEGAL = "/data", "/meta", "/legal"
PYTHON = "3.12"
SAM_PARSER = "meta-sam-parser==0.1.0"   # decodes SAM 3.1 output in stages 1-2 (SAM License)
BLENDER_VERSION = "4.5.14"
BLENDER_URL = (f"https://ftp.nluug.nl/pub/graphics/blender/release/Blender4.5/"
               f"blender-{BLENDER_VERSION}-linux-x64.tar.xz")   # an official blender.org mirror
BLENDER_SHA256 = "9ba871ff2ecd36526b77432745980b7e6664ecd0c7ca11c48849073dcfe06da3"
BLENDER_APT = ["xz-utils", "curl", "libx11-6", "libxi6", "libxxf86vm1", "libxfixes3", "libxrender1", "libxkbcommon0",
               "libsm6", "libgl1", "libegl1", "libgomp1", "libglib2.0-0"]
# Worker sizes (CPU cores, memory MiB) and the list prices used for compute estimates (USD per second).
CPU_CORE_USD_S = 0.047 / 3600
MEM_GIB_USD_S = 0.008 / 3600
WORKERS = {"model": (2.0, 4096), "blender": (8.0, 32768), "export": (2.0, 4096)}
HOUR = 3600
CUSTOM_DOMAINS = [d.strip() for d in os.environ.get("HEROGEN_SERVICE_DOMAIN", "").split(",") if d.strip()]

# The settings this deploy runs with, validated here: at deploy time from the deploying shell, in a container from
# the image. A deploy (or any local use of this file) refuses dev mode, which is for `python3 -m herogen.service dev`.
DEPLOY = service_config.from_env(require_key=False)
if modal.is_local() and DEPLOY.mode != "production":
    raise RuntimeError("refusing to deploy in dev mode: set HEROGEN_SERVICE_MODE=production and the other settings "
                       "in docs/ARCHITECTURE.md, \"Deploying\" (dev mode is `python3 -m herogen.service dev`)")

# Function timeouts follow the stage timeouts: each group's function outlives its stages run back to back, and the
# orchestrator outlives every group plus the copying between them.
GROUP_MARGIN_S = 1800


def group_timeout(group: str) -> int:
    from herogen.service.runner import PLAN

    per = DEPLOY.blender_timeout_s if group == "blender" else DEPLOY.stage_timeout_s
    return sum(1 for s in PLAN if s.group == group) * per + GROUP_MARGIN_S


TIMEOUTS = {g: group_timeout(g) for g in ("model", "blender", "export", "heldout")}
# The orchestrator waits for the job's own groups only; the held-out views run beside it in their own function.
JOB_TIMEOUT = sum(TIMEOUTS[g] for g in ("model", "blender", "export")) + GROUP_MARGIN_S
if JOB_TIMEOUT > 24 * HOUR:
    raise RuntimeError(f"stage timeouts add up to {JOB_TIMEOUT / HOUR:.1f} h per job, over Modal's 24 h limit")


def _commit() -> str:
    if not modal.is_local():
        return os.environ.get("HEROGEN_SERVICE_COMMIT", "")
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout.strip()
        # Everything under herogen/ goes into the image, untracked files included.
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "herogen"], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return ""


def price_per_s(group: str) -> float:
    cpu, mem = WORKERS[group]
    return cpu * CPU_CORE_USD_S + mem / 1024 * MEM_GIB_USD_S


IMAGE_ENV = {**service_config.env_for_deploy(), "HEROGEN_SERVICE_APP": APP_NAME,
             "HEROGEN_SERVICE_COMMIT": _commit(), "HEROGEN_SERVICE_DATA": DATA, "PYTHONDONTWRITEBYTECODE": "1"}
base = (modal.Image.debian_slim(python_version=PYTHON)
        .pip_install_from_requirements(str(HERE / "requirements.txt"))
        .pip_install(SAM_PARSER)
        .env(IMAGE_ENV))
web_image = base.add_local_python_source("herogen")
blender_image = (
    base.pip_install("scipy==1.18.1")   # BSD-3-Clause; hybrid orientation and outline conform
    .apt_install(*BLENDER_APT)
    .run_commands(
        f"curl -fsSL -o /tmp/blender.tar.xz {BLENDER_URL}",
        f"echo '{BLENDER_SHA256}  /tmp/blender.tar.xz' | sha256sum -c -",
        "mkdir -p /opt/blender && tar -xJf /tmp/blender.tar.xz -C /opt/blender --strip-components=1"
        " && rm /tmp/blender.tar.xz",
        "/opt/blender/blender --background --factory-startup --version",
        "printf '#!/bin/sh\\nexec python3 -m herogen.service.modal_bridge \"$@\"\\n' > /usr/local/bin/herogen-modal-bridge"
        " && chmod 755 /usr/local/bin/herogen-modal-bridge",
    )
    .env({"HEROGEN_BLENDER": "/opt/blender/blender", "HEROGEN_MODAL": "/usr/local/bin/herogen-modal-bridge"})
    .add_local_python_source("herogen")
)

app = modal.App(APP_NAME)
data_vol = modal.Volume.from_name(f"{APP_NAME}-data", create_if_missing=True)
meta_vol = modal.Volume.from_name(f"{APP_NAME}-meta", create_if_missing=True)
legal_vol = modal.Volume.from_name(f"{APP_NAME}-legal", create_if_missing=True)
web_secret = modal.Secret.from_name(WEB_SECRET)
model_secret = modal.Secret.from_name(MODEL_SECRET)
admin_secret = modal.Secret.from_name(ADMIN_SECRET)


def _settings(require_key: bool = False):
    return service_config.from_env(require_key=require_key, data_dir=Path(DATA), legal_dir=Path(LEGAL))


@app.cls(image=web_image, volumes={META: meta_vol}, secrets=[admin_secret], min_containers=1, max_containers=1,
         scaledown_window=1200, timeout=120)
class StoreService:
    """The only writer of the metadata database. It takes the snapshot lease when it starts (snapshot.py), so a
    newer store container fences off an older one during a deploy."""

    @modal.enter()
    def open(self) -> None:
        from herogen.service.snapshot import Snapshotter

        self.snap = Snapshotter(Path(META) / "meta", Path("/tmp/herogen-meta/meta.db"), meta_vol)
        self.snap.acquire()
        self._load()

    def _load(self) -> None:
        from herogen.service.store import Store

        self.snap.restore()
        self.store = Store(self.snap.local, persist=self.snap.save)

    def _run(self, name: str, args: tuple, kwargs: dict):
        from herogen.service.snapshot import SnapshotConflict, Superseded
        from herogen.service.store import StoreError

        for _ in range(3):
            try:
                return getattr(self.store, name)(*args, **kwargs)
            except Superseded:
                raise StoreError("The service is restarting; please try again.", "busy") from None
            except SnapshotConflict:
                self.store.close()
                self._load()
        raise StoreError("The service is busy; please try again.", "busy")

    @modal.method()
    def call(self, name: str, args: tuple, kwargs: dict):
        """The service's own methods (Store.PUBLIC)."""
        from herogen.service.store import Store, StoreError

        if name not in Store.PUBLIC:
            raise StoreError("unknown store method", "bad_method")
        return self._run(name, args, kwargs)

    @modal.method()
    def admin_call(self, key: str, name: str, args: tuple, kwargs: dict):
        """Admin methods (Store.ADMIN), only with the admin key, which only `admin_command` holds besides this
        container: the web and worker functions cannot reach them."""
        import hmac

        from herogen.service.store import Store, StoreError

        expected = os.environ.get("HEROGEN_ADMIN_KEY", "")
        if (len(expected) < 32 or not isinstance(key, str) or not hmac.compare_digest(key.encode(), expected.encode())
                or name not in Store.ADMIN):
            raise StoreError("not allowed", "forbidden")
        return self._run(name, args, kwargs)


class RemoteStore:
    """The store's methods, called on the StoreService container; admin methods only with the admin key."""

    def __init__(self, admin_key: str | None = None) -> None:
        self.handle = StoreService()
        self._admin_key = admin_key

    def __getattr__(self, name: str):
        from herogen.service.store import Store

        if name in Store.PUBLIC:
            return lambda *a, **k: self._remote(self.handle.call, name, a, k)
        if name in Store.ADMIN and self._admin_key:
            return lambda *a, **k: self._remote(self.handle.admin_call, self._admin_key, name, a, k)
        raise AttributeError(name)

    @staticmethod
    def _remote(method, *args):
        """Call the store. An error from Modal itself (modal.Error: the store container is starting, could not
        start, or the call did not reach it) becomes the store's "busy", which the web answers as 503 with
        Retry-After; the store's own refusals pass through unchanged."""
        import logging

        from herogen.service.store import StoreError

        try:
            return method.remote(*args)
        except modal.Error as e:
            logging.getLogger("herogen.service").error("store call failed in Modal: %s", type(e).__name__)
            raise StoreError("The service is busy; please try again in a moment.", "busy") from None


@app.function(image=web_image, block_network=True, restrict_modal_access=True, cpu=1.0, memory=2048, timeout=60,
              max_containers=20)
def sanitize_image(data: bytes, limits: dict) -> dict:
    from herogen.service.uploads import Limits, UploadRejected, sanitize

    try:
        png, info = sanitize(data, Limits(**limits))
    except UploadRejected as e:
        return {"ok": False, "message": str(e)}
    return {"ok": True, "png": png, "info": info}


def _submit(job_id: str) -> str:
    return run_job.spawn(job_id).object_id


@app.function(image=web_image, secrets=[web_secret], volumes={DATA: data_vol, LEGAL: legal_vol}, min_containers=1,
              timeout=600)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(custom_domains=CUSTOM_DOMAINS or None)
def web():
    import logging

    from herogen.service.auth import Signer
    from herogen.service.emailer import make_sender
    from herogen.service.storage import ModalVolumeView, Storage
    from herogen.service.uploads import UploadRejected
    from herogen.service.web import Services, create_app

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = _settings(require_key=True)
    limits = {"max_bytes": s.upload_max_bytes, "max_pixels": s.upload_max_pixels, "max_side": s.upload_max_side,
              "min_side": s.upload_min_side}

    def sanitize(data: bytes):
        r = sanitize_image.remote(data, limits)
        if not r["ok"]:
            raise UploadRejected(r["message"])
        return r["png"], r["info"]

    svc = Services(settings=s, store=RemoteStore(), signer=Signer(s.signing_key),
                   storage=Storage(Path(DATA), ModalVolumeView(data_vol)),
                   sender=make_sender(s.email_provider, api_key=s.email_api_key, sender=s.email_from),
                   sanitize=sanitize, submit=_submit)
    return create_app(svc)


@app.function(image=web_image, volumes={DATA: data_vol}, timeout=JOB_TIMEOUT, nonpreemptible=True,
              single_use_containers=True, max_containers=50)
def run_job(job_id: str) -> str | None:
    import logging

    from herogen.service.storage import Storage
    from herogen.service.worker import GroupResult
    from herogen.service.worker import run_job as orchestrate

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    groups = {"model": model_stages, "blender": blender_stages, "export": export_stages}

    def execute(job, attempt, group, resume):
        return GroupResult(**groups[group].remote(job, attempt, resume))

    def redispatch(jid: str) -> None:
        from herogen.service.worker import dispatch_queued

        dispatch_queued(RemoteStore(), _settings(), _submit)

    def heldout(job: dict) -> None:
        heldout_stages.spawn(job)   # started, not awaited: the job never waits for the held-out views

    return orchestrate(RemoteStore(), _settings(), Storage(Path(DATA)), job_id, modal.current_function_call_id(),
                       execute, redispatch=redispatch, volume=data_vol, heldout=heldout)


def _group(job: dict, attempt: int, resume: int, group: str) -> dict:
    import logging

    from herogen.service.runner import CliRunner
    from herogen.service.storage import Storage
    from herogen.service.worker import CopyWorkdir, make_context, run_group

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = _settings()
    store = RemoteStore()
    ctx = make_context(s, Storage(Path(DATA)), job)
    work = CopyWorkdir(ctx, Path("/tmp/herogen-work") / job["id"], data_vol,
                       lambda: store.job_current(job["id"], attempt),
                       still_wanted=lambda: (store.job_get(job["id"]) or {}).get("deleted_at") is None)
    runner = CliRunner(model_api_key=os.environ.get("MODEL_API_KEY", "") if group in ("model", "blender") else "")
    return run_group(store, s, job, attempt, group, resume, runner, work, price_per_s(group)).to_dict()


@app.function(image=web_image, secrets=[model_secret], volumes={DATA: data_vol}, cpu=WORKERS["model"][0],
              memory=WORKERS["model"][1], timeout=TIMEOUTS["model"], nonpreemptible=True, single_use_containers=True,
              max_containers=20)
def model_stages(job: dict, attempt: int, resume: int) -> dict:
    return _group(job, attempt, resume, "model")


@app.function(image=blender_image, secrets=[model_secret], volumes={DATA: data_vol}, cpu=WORKERS["blender"][0],
              memory=WORKERS["blender"][1], timeout=TIMEOUTS["blender"], nonpreemptible=True, single_use_containers=True,
              max_containers=20)
def blender_stages(job: dict, attempt: int, resume: int) -> dict:
    return _group(job, attempt, resume, "blender")


@app.function(image=web_image, volumes={DATA: data_vol}, cpu=WORKERS["export"][0], memory=WORKERS["export"][1],
              timeout=TIMEOUTS["export"], nonpreemptible=True, single_use_containers=True, max_containers=20)
def export_stages(job: dict, attempt: int, resume: int) -> dict:
    return _group(job, attempt, resume, "export")


@app.function(image=web_image, secrets=[model_secret], volumes={DATA: data_vol}, cpu=WORKERS["model"][0],
              memory=WORKERS["model"][1], timeout=TIMEOUTS["heldout"], nonpreemptible=True,
              single_use_containers=True, max_containers=DEPLOY.heldout_concurrency)
def heldout_stages(job: dict) -> str | None:
    """The held-out views (stage 2h) of one job, beside the job's own path; never fatal to it (worker.run_heldout).
    At most `heldout_concurrency` run at once service-wide, so they yield the Model API to the jobs' own stages;
    calls beyond that wait in Modal's queue."""
    import logging

    from herogen.service.runner import CliRunner
    from herogen.service.storage import Storage
    from herogen.service.worker import HeldoutWorkdir, make_context, run_heldout

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = _settings()
    store = RemoteStore()
    ctx = make_context(s, Storage(Path(DATA)), job)

    def workdir_for(claim: str):
        return HeldoutWorkdir(ctx, Path("/tmp/herogen-heldout") / job["id"], data_vol,
                              lambda: store.side_live(job["id"], "2h", claim))

    return run_heldout(store, s, job, CliRunner(model_api_key=os.environ.get("MODEL_API_KEY", "")), workdir_for,
                       price_per_s("model"))


@app.function(image=web_image, schedule=modal.Period(minutes=1), timeout=120)
def sweep() -> dict:
    from herogen.service.worker import reconcile

    return reconcile(RemoteStore(), _settings(), _submit)


@app.function(image=web_image, volumes={DATA: data_vol}, schedule=modal.Cron("17 3 * * *"), timeout=2 * HOUR)
def retention() -> dict:
    from herogen.service.storage import Storage
    from herogen.service.worker import purge

    return purge(RemoteStore(), _settings(), Storage(Path(DATA)), volume=data_vol)


@app.function(image=web_image, secrets=[web_secret, admin_secret], volumes={META: meta_vol}, timeout=120)
def admin_command(argv: list) -> str:
    import logging

    from herogen.service import admin as admin_cli
    from herogen.service.auth import Signer

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("herogen.service").warning("admin command: %s", argv[0] if argv else "")
    return admin_cli.run(list(argv), RemoteStore(admin_key=os.environ.get("HEROGEN_ADMIN_KEY", "")),
                         Signer(_settings(require_key=True).signing_key), snapshots=(Path(META) / "meta", meta_vol))


@app.function(image=blender_image, timeout=600)
def selfcheck_report() -> dict:
    from herogen.service.runner import base_env, roblox_available

    env = base_env({"HOME": "/tmp"})
    blender = subprocess.run(["/opt/blender/blender", "--background", "--factory-startup", "--version"],
                             capture_output=True, text=True, timeout=120).stdout.splitlines()[:1]
    flags = {}
    for module in ("herogen.pipeline", "herogen.blender"):
        out = subprocess.run(["python3", "-m", module, "run", "--help"], capture_output=True, text=True, env=env,
                             timeout=120).stdout
        flags[module] = sorted({w.strip("[],") for w in out.split() if w.startswith("--")})
    return {"commit": os.environ.get("HEROGEN_SERVICE_COMMIT", ""), "blender": blender, "cli_flags": flags,
            "roblox_export": roblox_available()}


@app.local_entrypoint()
def admin(args: str) -> None:
    print(admin_command.remote(shlex.split(args)))


@app.local_entrypoint()
def selfcheck() -> None:
    import json

    print(json.dumps(selfcheck_report.remote(), indent=2))
