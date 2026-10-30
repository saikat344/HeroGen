"""Running a job: claim it, run its stage groups in order, record every stage, finish it.

`run_job` is the orchestrator (one per job attempt). It claims the job in the store, which starts a numbered
attempt and refuses duplicates, then hands each stage group (model, blender, export) to an executor: in dev mode
a function call in this process, on Modal a call to that group's function (`modal_app`). The group runs its
stages with `run_group`, which before every stage checks that its attempt is still the job's live one (a
canceled, deleted or superseded attempt stops at the next stage), records the stage as running with a lease
that covers the stage's timeout, and records the outcome. The orchestrator and the held-out views also renew their
leases while they run (`LeaseKeeper`).

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
import secrets
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from herogen.service.config import Settings
from herogen.service.runner import (
    GROUPS,
    HELDOUT,
    Outcome,
    StageContext,
    StageRunner,
    plan_for,
    read_record,
    record_ok,
    resume_index,
    stage3_args,
)
from herogen.service.storage import Storage

log = logging.getLogger("herogen.service")
LEASE_MARGIN_S = 600.0
CLAIM_LEASE_S = 1800.0      # from the claim to the first stage: covers a cold start of the first group
GROUP_MAX_ENTRIES = 2
# Heartbeats (`LeaseKeeper`): while a worker lives it renews its lease every RENEW_EVERY_S to at least LEASE_TTL_S
# ahead, so no copy, stage or copy back outlasts it however long it takes. Renewing never shortens a longer lease:
# once the orchestrator dies, the job's lease runs out LEASE_TTL_S after the last renewal, or when the running
# stage's own lease (its timeout plus LEASE_MARGIN_S) ends if that is later. The held-out views hold no other lease,
# so theirs runs out LEASE_TTL_S after the last renewal. The TTL allows a few renewals in a row to fail (store busy).
LEASE_TTL_S = 600.0
RENEW_EVERY_S = 120.0


class LeaseKeeper:
    """Renew a lease on a background thread while a block of work runs: `with LeaseKeeper(renew, "what"):`.
    `renew()` extends the lease and returns False once the lease is no longer this worker's (the work finished,
    was canceled or was taken over), which stops the renewing. A renewal that raises is logged and tried again
    at the next beat. The thread ends with the block, or with the process when the worker dies."""

    def __init__(self, renew: Callable[[], bool], what: str, every_s: float | None = None) -> None:
        self.renew, self.what = renew, what
        self.every_s = RENEW_EVERY_S if every_s is None else every_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, name=f"lease-{what}", daemon=True)

    def _beat(self) -> None:
        while not self._stop.wait(self.every_s):
            try:
                if not self.renew():
                    return
            except Exception:
                log.warning("could not renew the lease of %s; trying again", self.what, exc_info=True)

    def __enter__(self) -> LeaseKeeper:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=60)


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


# What the held-out views write in a run folder, and only they: the job's own path never copies or deletes these, so
# the two never touch the same file.
HELDOUT_PATHS = ("heldout", "stage-2h.json")


def _excluded(rel: Path, exclude: tuple[str, ...]) -> bool:
    return bool(rel.parts) and rel.parts[0] in exclude


def _mirror(src: Path, dst: Path, delete: bool, exclude: tuple[str, ...] = ()) -> list[Path]:
    """Make `dst` hold `src`'s files: copy new or changed ones (size or mtime), and with `delete` remove files
    `src` no longer has. Paths whose first part is in `exclude` are neither copied nor deleted. Links are never
    followed or copied. Returns the files it wrote in `dst`."""
    dst.mkdir(parents=True, exist_ok=True)
    seen: set[Path] = set()
    written: list[Path] = []
    for dirpath, dirs, files in os.walk(src):
        rel_dir = Path(dirpath).relative_to(src)
        dirs[:] = [d for d in dirs if not (Path(dirpath) / d).is_symlink()]
        for f in files:
            s = Path(dirpath) / f
            rel = rel_dir / f
            if s.is_symlink() or f.endswith(".part") or _excluded(rel, exclude):
                continue
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
            written.append(d)
    if delete:
        for dirpath, _dirs, files in os.walk(dst):
            for f in files:
                rel = Path(dirpath).relative_to(dst) / f
                if rel not in seen and not _excluded(rel, exclude):
                    (dst / rel).unlink()
    return written


def _undo(written: list[Path], stop: Path) -> None:
    """Remove files a copy back wrote, then the folders it left empty, up to (not including) `stop`."""
    for p in written:
        if p.is_file() or p.is_symlink():
            p.unlink()
    for p in sorted({q.parent for q in written}, key=lambda q: len(q.parts), reverse=True):
        while p != stop and stop in p.parents and p.is_dir() and not any(p.iterdir()):
            p.rmdir()
            p = p.parent


class CopyWorkdir:
    """Stages work on a local copy (Modal): the run folder, the input and this user's caches are copied from the
    data volume, and after each stage copied back and committed, but only while `still_current()` says this
    attempt is the job's live one, so a stale worker never overwrites the files of a newer attempt. With
    `still_wanted`, a copy back whose job or account was deleted while it ran is removed again after its commit."""

    def __init__(self, ctx: StageContext, local_root: Path, volume, still_current: Callable[[], bool],
                 still_wanted: Callable[[], bool] | None = None) -> None:
        self.remote = ctx
        self.local_root = Path(local_root)
        self.volume = volume
        self.still_current = still_current
        # False once the job or its account is deleted; checked again after each commit (see `_settle`).
        self.still_wanted = still_wanted
        self.local: StageContext | None = None

    def _settle(self, written: list[Path], wanted: Callable[[], bool] | None) -> None:
        """After a commit: if the job or account was deleted meanwhile (the delete sets `deleted_at` before the
        web container removes the files, so a delete that raced this copy back is seen here), remove what this
        copy back wrote and commit again, so a deleted hero's files never come back."""
        if wanted is None or not written:
            return
        self.volume.reload()
        if wanted():
            return
        _undo(written, self.remote.run_dir.parents[2])   # stop at <data>/users
        self.volume.commit()
        event("copy_back_undone", job=self.remote.job_id, files=len(written))

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
        written = _mirror(loc.run_dir, r.run_dir, delete=True, exclude=HELDOUT_PATHS)
        written += _mirror(loc.cache_dir, r.cache_dir, delete=False)
        written += _mirror(loc.remote_cache_dir, r.remote_cache_dir, delete=False)
        if loc.input_path.is_file():
            r.input_path.parent.mkdir(parents=True, exist_ok=True)
            if not r.input_path.exists() or r.input_path.stat().st_size != loc.input_path.stat().st_size:
                shutil.copy2(loc.input_path, r.input_path)
                written.append(r.input_path)
        self.volume.commit()
        self._settle(written, self.still_wanted)

    def close(self) -> None:
        shutil.rmtree(self.local_root, ignore_errors=True)


class HeldoutWorkdir(CopyWorkdir):
    """The held-out views' work folder: a local copy like CopyWorkdir, but its copy back writes only `heldout/`
    and `stage-2h.json` (plus new cache entries), only into a run folder that still exists, and only while the
    side stage's claim holds and the job has not been deleted. So it never touches the job's own files, and never
    brings back the files of a job or account deleted while it ran: a delete that lands between its check and
    its commit is seen after the commit, and what it wrote is removed again (`_settle`)."""

    def sync(self) -> None:
        assert self.local is not None
        r, loc = self.remote, self.local
        self.volume.reload()
        if not self.still_current() or not r.run_dir.is_dir():
            return
        written: list[Path] = []
        if (loc.run_dir / "heldout").is_dir():
            written += _mirror(loc.run_dir / "heldout", r.run_dir / "heldout", delete=True)
        rec = loc.run_dir / "stage-2h.json"
        if rec.is_file():
            part = r.run_dir / "stage-2h.json.part"
            shutil.copy2(rec, part)
            os.replace(part, r.run_dir / "stage-2h.json")
            written.append(r.run_dir / "stage-2h.json")
        written += _mirror(loc.cache_dir, r.cache_dir, delete=False)
        written += _mirror(loc.remote_cache_dir, r.remote_cache_dir, delete=False)
        self.volume.commit()
        self._settle(written, self.still_wanted or self.still_current)


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
QC_UNAVAILABLE = ("The quality check is unavailable for this hero: the held-out views could not be made. "
                  "The hero is not affected.")
QC_NOT_ACCEPTED = ("The quality check did not accept the held-out views (they are kept for review). "
                   "The hero is not affected.")


def run_heldout(store, settings: Settings, job: dict[str, Any], runner: StageRunner,
                workdir_for: Callable[[str], Workdir], compute_price_per_s: float = 0.0) -> str | None:
    """The held-out views (stage 2h) for one job, beside the job's own path. They exist only for scoring, so
    whatever happens here (a failed gate, every view rejected, a Model API error, a timeout, a crash, a lost
    container, a rate limit) is recorded on the 2h stage as passed or unavailable, and the job's state, error,
    retries, refund and quota are never touched. At most one run per job (`store.side_claim`), never retried.
    Returns the stage's state, or None when this call had nothing to do."""
    jid = job["id"]
    claim = secrets.token_hex(8)
    if not store.side_claim(jid, "2h", claim, LEASE_TTL_S):
        event("heldout_skipped", job=jid)
        return None
    event("heldout_started", job=jid)
    state, message, seconds, cost = "unavailable", QC_UNAVAILABLE, 0.0, 0.0
    # The lease is renewed from the claim to the record: the copy in (the user's whole cache), the run and the copy
    # back all count, and none has a fixed bound. The function's own timeout bounds a container that hangs.
    with LeaseKeeper(lambda: store.side_renew(jid, "2h", claim, LEASE_TTL_S), f"{jid}/2h"):
        workdir = workdir_for(claim)
        try:
            ctx = workdir.open()
            out = runner.run(HELDOUT, ctx)
            seconds, cost = out.seconds, out.cost_usd
            rec = read_record(ctx.run_dir, HELDOUT.record) or {}
            if out.state == "passed":
                state, message = "passed", ""
            elif out.state == "rejected" and rec.get("status") != "error":
                message = QC_NOT_ACCEPTED
            elif out.state == "skipped":
                message = out.message
            if out.detail and state != "passed":
                log.warning("held-out views of %s: %s; output tail:\n%s", jid, out.state, out.detail)
            workdir.sync()
        except Exception:
            log.exception("held-out views of %s stopped", jid)
        finally:
            try:
                workdir.close()
            except Exception:
                log.exception("held-out work folder of %s", jid)
        total = round(cost + seconds * compute_price_per_s, 6)
        try:
            store.side_finish(jid, "2h", claim, state, seconds=seconds, cost_usd=total, message=message)
        except Exception:
            log.exception("could not record the held-out views of %s", jid)
    event("heldout_finished", job=jid, state=state, seconds=seconds, cost_usd=total)
    return state


def run_job(store, settings: Settings, storage: Storage, job_id: str, call_id: str, execute: GroupExecutor,
            redispatch: Callable[[str], None] | None = None, volume=None,
            heldout: Callable[[dict[str, Any]], None] | None = None) -> str | None:
    """Orchestrate one attempt of a job; returns the job's state afterwards (None when this worker had nothing
    to do: a duplicate, or a job that is no longer queued). Once the model group has passed stage 2, `heldout`
    is called to start the held-out views beside the rest of the job (`run_heldout`); the job never waits for
    them, and an error starting them is only logged."""
    claim = store.job_claim(job_id, call_id, CLAIM_LEASE_S, settings.max_attempts)
    if claim is None:
        event("job_claim_skipped", job=job_id, call=call_id)
        return None
    job, attempt = claim["job"], claim["attempt"]
    event("job_started", job=job_id, attempt=attempt, kind=job["kind"], profile=job["profile"],
          commit=job["code_commit"] or settings.code_commit)
    # The orchestrator lives for the whole attempt: its heartbeat keeps the job's lease through every copy,
    # stage, cold start and copy back, and stops when the orchestrator dies, so the sweep then finds the job.
    with LeaseKeeper(lambda: store.job_renew(job_id, attempt, LEASE_TTL_S), job_id):
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
            if group == "model" and res.status == "ok" and heldout is not None:
                try:
                    heldout(job)
                except Exception:
                    log.exception("could not start the held-out views of %s", job_id)
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
        took = round(time.monotonic() - t0, 2)
        store.job_finish(job_id, attempt, "succeeded", seconds=took, cost_usd=round(cost + compute, 6),
                         cost_detail=detail, bytes_=_bytes(storage, job, volume))
        event("job_finished", job=job_id, attempt=attempt, state="succeeded", seconds=took,
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
    """Dev mode: each job runs in a background thread of this process, its groups in-process, and its held-out
    views in a separate, smaller pool of threads (`heldout_concurrency`), on a local copy like on Modal."""

    def __init__(self, store, settings: Settings, storage: Storage, runner: StageRunner) -> None:
        import concurrent.futures

        self.store, self.settings, self.storage, self.runner = store, settings, storage, runner
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, settings.max_running_jobs),
                                                          thread_name_prefix="herogen-job")
        self.side_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, settings.heldout_concurrency),
                                                               thread_name_prefix="herogen-heldout")
        self.futures: dict[str, Any] = {}
        self.n = 0

    def start_heldout(self, job: dict[str, Any]) -> None:
        self.n += 1
        self.futures[f"heldout-{self.n}"] = self.side_pool.submit(self._heldout, job)

    def _heldout(self, job: dict[str, Any]) -> None:
        import tempfile

        from herogen.service.snapshot import LocalVolume

        ctx = make_context(self.settings, self.storage, job)
        root = Path(tempfile.mkdtemp(prefix="herogen-heldout-"))

        def workdir_for(claim: str) -> Workdir:
            return HeldoutWorkdir(ctx, root / "work", LocalVolume(),
                                  lambda: self.store.side_live(job["id"], "2h", claim))

        try:
            run_heldout(self.store, self.settings, job, self.runner, workdir_for)
        finally:
            shutil.rmtree(root, ignore_errors=True)

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
                    redispatch=lambda j: dispatch_queued(self.store, self.settings, self.submit),
                    heldout=self.start_heldout)
        except Exception:
            log.exception("job %s crashed", job_id)

    def wait(self, timeout: float = 30.0) -> None:
        import concurrent.futures

        deadline = time.monotonic() + timeout
        while True:   # waiting can start new futures (held-out views, redispatches): wait until none are left
            pending = [f for f in list(self.futures.values()) if not f.done()]
            if not pending or time.monotonic() > deadline:
                return
            concurrent.futures.wait(pending, timeout=max(0.0, deadline - time.monotonic()))
