"""Running a job: claim it, run its stage groups in order, record every stage, finish it.

`run_job` is the orchestrator (one per job attempt). It claims the job in the store, which starts a numbered
attempt and refuses duplicates, then hands each stage group (model, blender, export) to an executor: in dev mode
a function call in this process, on Modal a call to that group's function (`modal_app`). The group runs its
stages with `run_group`, which before every stage checks that its attempt is still the job's live one (a
canceled, deleted or superseded attempt stops at the next stage), records the stage as running with a lease
that covers the stage's timeout, and records the outcome.

Errors are sorted three ways. A rejection (a gate failed) finishes the job as rejected with the stage's reason.
A transient error (a timeout, the model service answering 429 or 5xx) queues the job for another attempt while
attempts remain. Anything else, and a job out of attempts, fails it; the store gives the quota back only when the
job spent nothing, at most a few times a day per user (`store.Store._refund_ok`). A worker that dies without
reporting is found by `reconcile` once its lease runs out. A retried attempt resumes after the stages that
already passed, and the per-user model cache makes repeated model calls free.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from herogen.service.config import Settings
from herogen.service.runner import (
    GROUPS,
    Outcome,
    StageContext,
    StageRunner,
    plan_for,
    record_ok,
    resume_index,
    stage3_args,
)
from herogen.service.storage import Storage

log = logging.getLogger("herogen.service")
LEASE_MARGIN_S = 600.0
CLAIM_LEASE_S = 1800.0      # from the claim to the first stage: covers a cold start of the first group
GROUP_MAX_ENTRIES = 2


def event(name: str, **fields: Any) -> None:
    """One structured log line (JSON). Never pass an email address, token, prompt or IP here."""
    log.info(json.dumps({"event": name, "at": round(time.time(), 3), **fields}, default=str, sort_keys=True))


@dataclass
class GroupResult:
    status: str                    # ok, rejected, error, stale
    stage: str | None = None
    code: str = ""
    message: str = ""
    retry: bool = False
    seconds: float = 0.0
    cost_usd: float = 0.0
    compute_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Workdir(Protocol):
    """Where a group's stages read and write the job's files."""

    def open(self) -> StageContext: ...

    def sync(self) -> None: ...

    def close(self) -> None: ...


class InPlaceWorkdir:
    """Stages work directly in the data folder (dev mode, and any machine with one shared disk)."""

    def __init__(self, ctx: StageContext) -> None:
        self.ctx = ctx

    def open(self) -> StageContext:
        self.ctx.run_dir.mkdir(parents=True, exist_ok=True)
        return self.ctx

    def sync(self) -> None:
        pass

    def close(self) -> None:
        pass


def _mirror(src: Path, dst: Path, delete: bool) -> None:
    """Make `dst` hold `src`'s files: copy new or changed ones (size or mtime), and with `delete` remove files
    `src` no longer has. Links are never followed or copied."""
    dst.mkdir(parents=True, exist_ok=True)
    seen: set[Path] = set()
    for dirpath, dirs, files in os.walk(src):
        rel_dir = Path(dirpath).relative_to(src)
        dirs[:] = [d for d in dirs if not (Path(dirpath) / d).is_symlink()]
        for f in files:
            s = Path(dirpath) / f
            if s.is_symlink() or f.endswith(".part"):
                continue
            rel = rel_dir / f
            seen.add(rel)
            d = dst / rel
            st = s.stat()
            if d.exists() and not d.is_symlink():
                dt = d.stat()
                if dt.st_size == st.st_size and int(dt.st_mtime) == int(st.st_mtime):
                    continue
            d.parent.mkdir(parents=True, exist_ok=True)
            part = d.with_name(d.name + ".part")
            shutil.copy2(s, part)
            os.replace(part, d)
    if delete:
        for dirpath, _dirs, files in os.walk(dst):
            for f in files:
                rel = Path(dirpath).relative_to(dst) / f
                if rel not in seen:
                    (dst / rel).unlink()


class CopyWorkdir:
    """Stages work on a local copy (Modal): the run folder, the input and this user's caches are copied from the
    data volume, and after each stage copied back and committed, but only while `still_current()` says this
    attempt is the job's live one, so a stale worker never overwrites the files of a newer attempt."""

    def __init__(self, ctx: StageContext, local_root: Path, volume, still_current: Callable[[], bool]) -> None:
        self.remote = ctx
        self.local_root = Path(local_root)
        self.volume = volume
        self.still_current = still_current
        self.local: StageContext | None = None

    def open(self) -> StageContext:
        r = self.remote
        self.volume.reload()
        root = self.local_root
        if root.exists():
            shutil.rmtree(root)
        local = StageContext(**{**r.__dict__, "run_dir": root / "runs" / r.job_id,
                                "input_path": root / "inputs" / r.input_path.name,
                                "cache_dir": root / "cache", "remote_cache_dir": root / "cache-remote"})
        for a, b in ((r.run_dir, local.run_dir), (r.cache_dir, local.cache_dir),
                     (r.remote_cache_dir, local.remote_cache_dir)):
            if a.is_dir():
                _mirror(a, b, delete=False)
            else:
                b.mkdir(parents=True, exist_ok=True)
        if r.input_path.is_file():
            local.input_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(r.input_path, local.input_path)
        self.local = local
        return local

    def sync(self) -> None:
        assert self.local is not None
        if not self.still_current():
            return
        r, loc = self.remote, self.local
        _mirror(loc.run_dir, r.run_dir, delete=True)
        _mirror(loc.cache_dir, r.cache_dir, delete=False)
        _mirror(loc.remote_cache_dir, r.remote_cache_dir, delete=False)
        if loc.input_path.is_file():
            r.input_path.parent.mkdir(parents=True, exist_ok=True)
            if not r.input_path.exists() or r.input_path.stat().st_size != loc.input_path.stat().st_size:
                shutil.copy2(loc.input_path, r.input_path)
        self.volume.commit()

    def close(self) -> None:
        shutil.rmtree(self.local_root, ignore_errors=True)


def make_context(settings: Settings, storage: Storage, job: dict[str, Any]) -> StageContext:
    uid, jid = job["user_id"], job["id"]
    return StageContext(job_id=jid, kind=job["kind"], profile=job["profile"], name=job["name"], prompt=job["prompt"],
                        run_dir=storage.run_dir(uid, jid), input_path=storage.input_path(uid, jid),
                        cache_dir=storage.cache_dir(uid), remote_cache_dir=storage.remote_cache_dir(uid),
                        stage3_args=stage3_args(settings.stage3_args), timeout_s=settings.stage_timeout_s,
                        blender_timeout_s=settings.blender_timeout_s)


def run_group(store, settings: Settings, job: dict[str, Any], attempt: int, group: str, resume: int,
              runner: StageRunner, workdir: Workdir, compute_price_per_s: float = 0.0) -> GroupResult:
    """Run the stages of one group for one attempt; see the module docstring."""
    jid = job["id"]
    plan = plan_for(job["kind"])
    mine = [(i, s) for i, s in enumerate(plan) if s.group == group]
    total = GroupResult("ok")
    if not mine:
        return total
    if not store.group_enter(jid, attempt, group, GROUP_MAX_ENTRIES):
        if not store.job_current(jid, attempt):
            return GroupResult("stale", mine[0][1].key)
        return GroupResult("error", mine[0][1].key, "restarts", "A worker was restarted too many times.", False)
    ctx = workdir.open()
    try:
        for index, spec in mine:
            if not store.job_current(jid, attempt):
                return GroupResult("stale", spec.key)
            if index < resume and record_ok(ctx.run_dir, spec):
                store.job_stage_set(jid, attempt, spec.key, "passed", message="kept from an earlier attempt")
                continue
            timeout = settings.blender_timeout_s if spec.group == "blender" else settings.stage_timeout_s
            if not store.job_stage_set(jid, attempt, spec.key, "running", lease_s=timeout + LEASE_MARGIN_S):
                return GroupResult("stale", spec.key)
            event("stage_started", job=jid, attempt=attempt, stage=spec.key)
            try:
                out = runner.run(spec, ctx)
            except Exception as e:
                log.exception("stage %s of %s raised", spec.key, jid)
                out = Outcome("failed", message=f"The {spec.label} stage stopped with an internal error.",
                              code="stage_error", detail=f"{type(e).__name__}")
            compute = round(out.seconds * compute_price_per_s, 6)
            workdir.sync()
            store.job_stage_set(jid, attempt, spec.key, out.state, seconds=out.seconds,
                                cost_usd=round(out.cost_usd + compute, 6), message=out.message)
            event("stage_finished", job=jid, attempt=attempt, stage=spec.key, state=out.state, code=out.code,
                  seconds=out.seconds, cost_usd=out.cost_usd, compute_usd=compute)
            if out.detail and out.state in ("failed", "rejected"):
                log.warning("stage %s of %s %s; output tail:\n%s", spec.key, jid, out.state, out.detail)
            total.seconds += out.seconds
            total.cost_usd += out.cost_usd
            total.compute_usd += compute
            if spec.optional and out.state in ("rejected", "failed"):
                continue  # recorded on the stage (the Roblox download shows why); the GLB is still delivered
            if out.state == "rejected":
                return GroupResult("rejected", spec.key, "rejected", out.message, False, total.seconds,
                                   total.cost_usd, total.compute_usd)
            if out.state == "failed":
                return GroupResult("error", spec.key, out.code, out.message, out.transient, total.seconds,
                                   total.cost_usd, total.compute_usd)
        return total
    finally:
        workdir.close()


GroupExecutor = Callable[[dict[str, Any], int, str, int], GroupResult]


def run_job(store, settings: Settings, storage: Storage, job_id: str, call_id: str, execute: GroupExecutor,
            redispatch: Callable[[str], None] | None = None, volume=None) -> str | None:
    """Orchestrate one attempt of a job; returns the job's state afterwards (None when this worker had nothing
    to do: a duplicate, or a job that is no longer queued)."""
    claim = store.job_claim(job_id, call_id, CLAIM_LEASE_S, settings.max_attempts)
    if claim is None:
        event("job_claim_skipped", job=job_id, call=call_id)
        return None
    job, attempt = claim["job"], claim["attempt"]
    event("job_started", job=job_id, attempt=attempt, kind=job["kind"], profile=job["profile"],
          commit=job["code_commit"] or settings.code_commit)
    t0 = time.monotonic()
    if volume is not None:
        volume.reload()
    run_dir = storage.run_dir(job["user_id"], job_id)
    resume = resume_index(run_dir, plan_for(job["kind"])) if attempt > 1 else 0
    if job["kind"] == "sketch" and not storage.input_path(job["user_id"], job_id).is_file():
        store.job_finish(job_id, attempt, "failed", error_code="input_missing", error_stage="0", refund=True,
                         error_message="The uploaded sketch was not saved.")
        return "failed"
    seconds = cost = compute = 0.0
    for group in GROUPS:
        try:
            res = execute(job, attempt, group, resume)
        except Exception as e:
            log.exception("group %s of %s failed", group, job_id)
            res = GroupResult("error", None, "worker_error", "A worker stopped unexpectedly.", True)
            event("group_crashed", job=job_id, attempt=attempt, group=group, error=type(e).__name__)
        seconds += res.seconds
        cost += res.cost_usd
        compute += res.compute_usd
        detail = {"model_and_gpu_usd": round(cost, 6), "compute_usd": round(compute, 6),
                  "basis": "estimate: stage records' model and GPU estimates plus wall seconds x the worker's "
                           "list price"}
        if res.status == "stale":
            event("job_stale", job=job_id, attempt=attempt, group=group)
            return (store.job_get(job_id) or {}).get("state")
        if res.status == "rejected":
            store.job_finish(job_id, attempt, "rejected", error_code="rejected", error_stage=res.stage,
                             error_message=res.message, seconds=round(time.monotonic() - t0, 2),
                             cost_usd=round(cost + compute, 6), cost_detail=detail,
                             bytes_=_bytes(storage, job, volume))
            event("job_finished", job=job_id, attempt=attempt, state="rejected", stage=res.stage)
            return "rejected"
        if res.status == "error":
            if res.retry:
                state = store.job_requeue(job_id, attempt, settings.max_attempts, res.code, res.stage,
                                          res.message + " It was tried again automatically.")
            else:
                store.job_finish(job_id, attempt, "failed", error_code=res.code, error_stage=res.stage,
                                 error_message=res.message,
                                 seconds=round(time.monotonic() - t0, 2), cost_usd=round(cost + compute, 6),
                                 cost_detail=detail, refund=True, bytes_=_bytes(storage, job, volume))
                state = "failed"
            event("job_finished" if state != "queued" else "job_requeued", job=job_id, attempt=attempt,
                  state=state, stage=res.stage, code=res.code)
            if state == "queued" and redispatch is not None:
                redispatch(job_id)
            return state
    store.job_finish(job_id, attempt, "succeeded", seconds=round(time.monotonic() - t0, 2),
                     cost_usd=round(cost + compute, 6), cost_detail=detail, bytes_=_bytes(storage, job, volume))
    event("job_finished", job=job_id, attempt=attempt, state="succeeded", seconds=round(time.monotonic() - t0, 2),
          cost_usd=round(cost + compute, 6))
    return "succeeded"


def _bytes(storage: Storage, job: dict[str, Any], volume) -> int:
    if volume is not None:
        volume.reload()
    try:
        return storage.bytes_used(job["user_id"], job["id"])
    except OSError:
        return 0


def dispatch_queued(store, settings: Settings, submit: Callable[[str], str]) -> list[str]:
    """Start workers for queued jobs while fewer than `max_running_jobs` are running; returns the job ids."""
    started = []
    for job in store.jobs_to_dispatch(settings.max_running_jobs):
        call_id = submit(job["id"])
        store.job_dispatch(job["id"], call_id)
        started.append(job["id"])
    return started


def reconcile(store, settings: Settings, submit: Callable[[str], str]) -> dict[str, Any]:
    """The periodic sweep: requeue or fail jobs whose worker died, then dispatch queued jobs."""
    stale = store.reconcile_stale(settings.max_attempts)
    started = dispatch_queued(store, settings, submit)
    if stale or started:
        event("reconcile", stale=stale, started=started)
    return {"stale": stale, "started": started}


def purge(store, settings: Settings, storage: Storage, volume=None) -> dict[str, Any]:
    """Retention: delete the files of jobs past the retention period and of deleted jobs, cache entries older than
    the retention period, uploaded sketches no job claims after a day, then old sign-in links, sessions and audit
    entries."""
    if volume is not None:
        volume.reload()
    removed = []
    for job in store.retention_due(settings.retention_days * 86400.0):
        try:
            storage.delete_job(job["user_id"], job["id"])
        except (OSError, ValueError):
            log.exception("could not delete files of %s", job["id"])
            continue
        store.job_mark_purged(job["id"])
        removed.append(job["id"])
    cache_files = storage.purge_caches(time.time() - settings.retention_days * 86400.0)
    # Sketches no job claims (a job create the store could not confirm and that never landed), after a day.
    orphans = 0
    for _uid, jid, path in storage.inputs_before(time.time() - 86400.0):
        if store.job_get(jid) is None:
            path.unlink()
            orphans += 1
    if volume is not None:
        volume.commit()
    counts = store.purge_expired(settings.audit_retention_days * 86400.0)
    event("retention", jobs=len(removed), cache_files=cache_files, orphan_inputs=orphans, **counts)
    return {"jobs": removed, "cache_files": cache_files, "orphan_inputs": orphans, **counts}


class ThreadBackend:
    """Dev mode: each job runs in a background thread of this process, its groups in-process."""

    def __init__(self, store, settings: Settings, storage: Storage, runner: StageRunner) -> None:
        import concurrent.futures

        self.store, self.settings, self.storage, self.runner = store, settings, storage, runner
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, settings.max_running_jobs),
                                                          thread_name_prefix="herogen-job")
        self.futures: dict[str, Any] = {}
        self.n = 0

    def execute(self, job: dict[str, Any], attempt: int, group: str, resume: int) -> GroupResult:
        ctx = make_context(self.settings, self.storage, job)
        return run_group(self.store, self.settings, job, attempt, group, resume, self.runner, InPlaceWorkdir(ctx))

    def submit(self, job_id: str) -> str:
        self.n += 1
        call_id = f"local-{self.n}"
        self.futures[call_id] = self.pool.submit(self._run, job_id, call_id)
        return call_id

    def _run(self, job_id: str, call_id: str) -> None:
        try:
            run_job(self.store, self.settings, self.storage, job_id, call_id, self.execute,
                    redispatch=lambda j: dispatch_queued(self.store, self.settings, self.submit))
        except Exception:
            log.exception("job %s crashed", job_id)

    def wait(self, timeout: float = 30.0) -> None:
        import concurrent.futures

        concurrent.futures.wait(list(self.futures.values()), timeout=timeout)
