"""Running a job: claim it, run its stage groups in order, record every stage, finish it.

`run_job` is the orchestrator (one per job attempt). It claims the job in the store, which starts a numbered
attempt and refuses duplicates, then hands each stage group (model, blender, export) to an executor: in dev mode
a function call in this process, on Modal a call to that group's function (`modal_app`). The group runs its
stages with `run_group`, which before every stage checks that its attempt is still the job's live one (a
canceled, deleted or superseded attempt stops at the next stage), records the stage as running with a lease
that covers the stage's timeout, and records the outcome. The orchestrator also renews its
lease while they run (`LeaseKeeper`).

Before a mesh exists, rejections stop generation, transient errors retry while attempts remain, and other
errors fail. A rejected shape with a saved mesh continues with a rough-shape note. Later stage failures become
explicit skipped stages; the website model succeeds before the game-body and export groups start. Those groups
retain an attempt lease but do not occupy a model-generation slot. Their status is recorded separately.

"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import secrets
import shutil
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from herogen.pipeline import events
from herogen.service.config import Settings
from herogen.service.runner import (
    GROUPS,
    Outcome,
    StageContext,
    StageRunner,
    plan_for,
    read_record,
    record_ok,
    resume_index,
    stage3_args,
    write_record,
)
from herogen.service.storage import Storage

log = logging.getLogger("herogen.service")
LEASE_MARGIN_S = 600.0
CLAIM_LEASE_S = 1800.0      # from the claim to the first stage: covers a cold start of the first group
GROUP_MAX_ENTRIES = 2
# Heartbeats (`LeaseKeeper`): while a worker lives it renews its lease every RENEW_EVERY_S to at least LEASE_TTL_S
# ahead, so no copy, stage or copy back outlasts it however long it takes. Renewing never shortens a longer lease:
# once the orchestrator dies, the job's lease runs out LEASE_TTL_S after the last renewal, or when the running
# stage's own lease (its timeout plus LEASE_MARGIN_S) ends if that is later.
# The TTL allows a few renewals in a row to fail (store busy).
LEASE_TTL_S = 600.0
RENEW_EVERY_S = 120.0
# While a stage runs on a local copy (Modal), its build-log file is copied to the data volume and committed this
# often, when it has grown, so the website can show the events live. Without it they would appear only when the
# stage ends and the whole run folder is copied back.
PUMP_EVERY_S = 3.0


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


class EventPump:
    """Mirror one stage's build-log file (`herogen.pipeline.events`) from the local copy to the data volume while
    the stage runs: `with EventPump(src, dst, volume, still_current):`. Every `every_s`, when the file changed,
    it checks that this attempt is still the job's live one (else it stops for good, so a stale, canceled or
    deleted attempt never writes), copies the file next to its target and renames it into place, then commits.
    It never creates the run folder. The copy back after the stage still copies everything, so a missed beat
    loses nothing."""

    def __init__(self, src: Path, dst: Path, volume, still_current: Callable[[], bool],
                 every_s: float | None = None) -> None:
        self.src, self.dst, self.volume, self.still_current = Path(src), Path(dst), volume, still_current
        self.every_s = PUMP_EVERY_S if every_s is None else every_s
        self.last: tuple[int, int] | None = None
        self.stopped = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, name="event-pump", daemon=True)

    def pump(self) -> bool:
        """One beat: True when the file was copied and committed."""
        if self.stopped:
            return False
        try:
            st = self.src.stat()
        except FileNotFoundError:
            return False
        stamp = (st.st_size, st.st_mtime_ns)
        if stamp == self.last:
            return False
        if not self.still_current():
            self.stopped = True
            return False
        if not self.dst.parent.is_dir():
            return False
        part = self.dst.with_name(self.dst.name + ".part")
        shutil.copy2(self.src, part)
        os.replace(part, self.dst)
        self.volume.commit()
        self.last = stamp
        return True

    def _beat(self) -> None:
        while not self._stop.wait(self.every_s) and not self.stopped:
            try:
                self.pump()
            except Exception:
                log.warning("could not copy the build log; trying again", exc_info=True)

    def __enter__(self) -> EventPump:
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
    detail: str = ""

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


def _mirror(src: Path, dst: Path, delete: bool) -> list[Path]:
    """Make `dst` hold `src`'s files: copy new or changed ones (size or mtime), and with `delete` remove files
    `src` no longer has. Links are never
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
            if s.is_symlink() or f.endswith(".part"):
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
                if rel not in seen:
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
        written = _mirror(loc.run_dir, r.run_dir, delete=True)
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

    def event_pump(self) -> EventPump:
        assert self.local is not None
        return EventPump(self.local.run_dir / events.FILE, self.remote.run_dir / events.FILE, self.volume,
                         self.still_current)


def make_context(settings: Settings, storage: Storage, job: dict[str, Any], *, spend_limits_enabled=False) -> StageContext:
    uid, jid = job["user_id"], job["id"]
    return StageContext(job_id=jid, kind=job["kind"], profile=job["profile"], name=job["name"], prompt=job["prompt"],
                        run_dir=storage.run_dir(uid, jid), input_path=storage.input_path(uid, jid),
                        cache_dir=storage.cache_dir(uid), remote_cache_dir=storage.remote_cache_dir(uid),
                        stage3_args=stage3_args(settings.stage3_args), timeout_s=settings.stage_timeout_s,
                        blender_timeout_s=settings.blender_timeout_s, refine_cap_usd=settings.refine_cap_usd if spend_limits_enabled else None,
                        refine_profile=settings.refine_profile)


def website_model(run_dir: Path) -> Path | None:
    from herogen.service.bundle import glb_path

    return glb_path(run_dir)


# Stage 3's structural checks. A mesh that fails one (a flat carve from too few views, open edges) is never shown
# as the hero; only a mesh rejected by the later visual acceptance checks is.
STRUCTURAL_GATES = ("independent_views", "front_view_present", "open_edges")


def structurally_sound(run_dir: Path) -> bool:
    rec = read_record(run_dir, "stage-3.json") or {}
    return not any(isinstance(c, dict) and c.get("name") in STRUCTURAL_GATES and c.get("gate", True)
                   and not c.get("pass") for c in rec.get("checks", []))


def degradation_note(spec, out):
    if spec.key == '3':
        return 'The 3D shape is rough. Showing the best mesh available.'
    reason = out.message.split('Checks that failed:')[0].strip()
    if 'stage rejected this hero.' in reason:
        reason = reason.split('stage rejected this hero.', 1)[1].strip()
    if out.code == 'stage_error':
        reason = f'the {spec.label} stage stopped unexpectedly.'
    reason = reason[:1].lower() + reason[1:]
    if spec.key == '3f':
        return 'Texture refine was skipped: ' + reason.rstrip('.') + '. Showing the unrefined texture.'
    if spec.key == '4':
        return out.message if out.message.startswith(('Roblox body unavailable', 'Roblox files unavailable')) else 'Roblox body unavailable for this hero: ' + reason
    if spec.key == '6':
        return 'Roblox files unavailable for this hero: ' + reason
    return 'The results page was skipped. Your model is available in this workbench.'


def skip_stage(store, jid, attempt, spec, ctx, message, detail=""):
    record = read_record(ctx.run_dir, spec.record)
    if detail and record is not None:
        write_record(ctx.run_dir, {**record, "ok": False, "status": "error", "error": detail})
    else:
        write_record(ctx.run_dir, {'stage': spec.key, 'name': spec.label, 'ok': False,
                              'status': 'error' if detail else 'skipped', 'reason': message,
                              **({'error': detail} if detail else {}), 'outputs': [], 'cost_usd': 0})
    store.job_stage_set(jid, attempt, spec.key, 'skipped', message=message)


def run_group(store, settings: Settings, job: dict[str, Any], attempt: int, group: str, resume: int,
              runner: StageRunner, workdir: Workdir, compute_price_per_s: float = 0.0,
              compute_timeout_s: float | None = None) -> GroupResult:
    """Run the stages of one group for one attempt; see the module docstring."""
    jid = job["id"]
    plan = plan_for(job["kind"], job["profile"])
    stop = job.get("stop_after")
    boundary = next((i for i, s in enumerate(plan) if s.key == stop), len(plan))
    mine = [(i, s) for i, s in enumerate(plan) if s.group == group and i <= boundary]
    total = GroupResult("ok")
    if not mine:
        return total
    if not store.group_enter(jid, attempt, group, GROUP_MAX_ENTRIES):
        if not store.job_current(jid, attempt):
            return GroupResult("stale", mine[0][1].key)
        return GroupResult("error", mine[0][1].key, "restarts", "A worker was restarted too many times.", False)
    prior = {r["stage"]: r for r in store.job_stages(jid)}
    ctx = workdir.open()
    ctx.attempt = attempt
    log_file = ctx.run_dir / events.FILE

    def step_event(kind: str, spec, **fields: Any) -> None:
        if kind == "step_start":
            fields.update(i=plan.index(spec) + 1, of=len(plan))
        events.emit(kind, file=log_file, step=spec.key, a=attempt, **fields)

    try:
        for index, spec in mine:
            if not store.job_current(jid, attempt):
                return GroupResult("stale", spec.key)
            if index < resume and record_ok(ctx.run_dir, spec):
                store.job_stage_set(jid, attempt, spec.key, "passed", message="kept from an earlier attempt",
                                    **({"seconds": 0.0, "cost_usd": 0.0} if spec.key not in prior else {}))
                step_event("step_start", spec)
                step_event("step_end", spec, outcome="kept", s=0, usd=0)
                continue
            rows = {r['stage']: r for r in store.job_stages(jid)}
            # The game body needs a passed stage 3 (`herogen.blender` refuses a rejected one), as refine does.
            dependency = '4' if spec.key == '6' else '3' if spec.key in ('3f', '4') else None
            if dependency and rows.get(dependency, {}).get('state') in ('skipped', 'rejected'):
                why = (rows[dependency]['message'] if spec.key == '6' else
                       'Roblox body unavailable for this hero: the 3D shape did not pass its checks.'
                       if spec.key == '4' else
                       'Texture refine was skipped because the 3D shape is rough. Showing the unrefined texture.')
                skip_stage(store, jid, attempt, spec, ctx, why)
                step_event("step_start", spec)
                step_event("step_end", spec, outcome="skipped", why="earlier_step", s=0, usd=0)
                workdir.sync()
                continue
            timeout = settings.blender_timeout_s if spec.group in ("blender", "body") else settings.stage_timeout_s
            if not store.job_stage_set(jid, attempt, spec.key, "running", lease_s=timeout + LEASE_MARGIN_S):
                return GroupResult("stale", spec.key)
            event("stage_started", job=jid, attempt=attempt, stage=spec.key)
            step_event("step_start", spec)
            stage_t0 = time.monotonic()
            pump = getattr(workdir, "event_pump", None)
            try:
                with pump() if pump else contextlib.nullcontext():
                    out = runner.run(spec, ctx)
            except Exception as e:
                from herogen.pipeline.client import ApiError, ReplayMiss

                log.exception("stage %s of %s raised", spec.key, jid)
                saved = read_record(ctx.run_dir, spec.record) or {}
                cost = saved.get("cost_usd", 0.0)
                cost = cost if isinstance(cost, (int, float)) and math.isfinite(cost) else 0.0
                model_error = spec.key in ('3f', '4', '6', '5') and isinstance(e, (ApiError, ReplayMiss))
                out = Outcome("failed", seconds=round(time.monotonic() - stage_t0, 2), cost_usd=cost, message=("The image model did not answer." if model_error else
                              f"The {spec.label} stage stopped with an internal error."),
                              code="model_unavailable" if model_error else "stage_error",
                              detail=traceback.format_exc())
            if spec.key == '3' and out.state == 'rejected' and website_model(ctx.run_dir) is None:
                out = Outcome('failed', out.seconds, out.cost_usd,
                              'The 3D shape did not pass its checks and no mesh was saved.',
                              'model_missing', detail=out.detail)
            degraded = out.state in ('failed', 'rejected') and (
                spec.key in ('3f', '4', '6', '5')
                or spec.key == '3' and out.state == 'rejected' and structurally_sound(ctx.run_dir)
            ) and website_model(ctx.run_dir) is not None
            if (spec.key in ('3f', '4', '6', '5') and out.state in ('failed', 'rejected')
                    and read_record(ctx.run_dir, spec.record) is None):
                write_record(ctx.run_dir, {'stage': spec.key, 'name': spec.label, 'ok': False,
                    'status': 'error' if out.state == 'failed' else 'rejected',
                    'error': out.detail or out.message, 'seconds': out.seconds, 'cost_usd': out.cost_usd})
            compute = round(out.seconds * compute_price_per_s, 6)
            state = "skipped" if degraded else out.state
            if spec.key == "4" and out.state == "rejected":
                state = "rejected"
            step_event("step_end", spec, outcome=state, code=out.code or None if state != "passed" else None,
                       s=out.seconds, usd=round(out.cost_usd + compute, 6))
            workdir.sync()
            store.job_stage_set(jid, attempt, spec.key, state, seconds=out.seconds,
                                cost_usd=round((prior.get(spec.key, {}).get("cost_usd") or 0.0)
                                               + out.cost_usd + compute, 6), message=degradation_note(spec, out) if degraded else out.message,
                                mesh_saved=spec.key == "3" and out.state in ("passed", "rejected")
                                           and website_model(ctx.run_dir) is not None)
            event("stage_finished", job=jid, attempt=attempt, stage=spec.key, state=out.state, code=out.code,
                  seconds=out.seconds, cost_usd=out.cost_usd, compute_usd=compute)
            if out.detail and out.state in ("failed", "rejected"):
                log.warning("stage %s of %s %s; output tail:\n%s", spec.key, jid, out.state, out.detail)
            total.seconds += out.seconds
            total.cost_usd += out.cost_usd
            total.compute_usd += compute
            if degraded or spec.optional and out.state in ("rejected", "failed"):
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


def job_event(storage: Storage, job: dict[str, Any], volume, kind: str, *, create: bool = False,
              **fields: Any) -> None:
    """A job-level build-log event, written by the orchestrator straight to the data volume. It writes only while
    none of its job's groups runs (before, between and after them), so it never races a group's copy back; on
    Modal it reloads first, so it appends to the newest file, and commits after. ``create`` makes the run folder
    (the claim, for a job that has none yet); otherwise a folder that is gone (a deleted hero) stays gone."""
    try:
        if volume is not None:
            volume.reload()
        run = storage.run_dir(job["user_id"], job["id"])
        if create:
            run.mkdir(parents=True, exist_ok=True)
        if not run.is_dir() or not events.emit(kind, file=run / events.FILE, **fields):
            return
        if volume is not None:
            volume.commit()
    except Exception:
        log.warning("could not write the %s build-log event of %s", kind, job.get("id"), exc_info=True)


def run_job(store, settings: Settings, storage: Storage, job_id: str, call_id: str, execute: GroupExecutor,
            redispatch: Callable[[str], None] | None = None, volume=None) -> str | None:
    """Claim and execute one attempt, recording its final state or retry."""
    claim = store.job_claim(job_id, call_id, CLAIM_LEASE_S, settings.max_attempts)
    if claim is None:
        event("job_claim_skipped", job=job_id, call=call_id)
        return None
    job, attempt = claim["job"], claim["attempt"]
    event("job_started", job=job_id, attempt=attempt, kind=job["kind"], profile=job["profile"],
          commit=job["code_commit"] or settings.code_commit)
    job_event(storage, job, volume, "job_start", create=True, a=attempt, of=settings.max_attempts)
    clock = getattr(store, "clock", None)   # the store's clock stamped created_at (a remote store has none here)
    waited = (clock() if callable(clock) else time.time()) - (job.get("created_at") or 0.0)
    if attempt == 1 and 0 <= waited < 86400:
        job_event(storage, job, volume, "queue", a=attempt, s=round(waited, 3))
    # The orchestrator lives for the whole attempt: its heartbeat keeps the job's lease through every copy,
    # stage, cold start and copy back, and stops when the orchestrator dies, so the sweep then finds the job.
    with LeaseKeeper(lambda: store.job_renew(job_id, attempt, LEASE_TTL_S), job_id):
        t0 = time.monotonic()
        if volume is not None:
            volume.reload()
        run_dir = storage.run_dir(job["user_id"], job_id)
        resume = (resume_index(run_dir, plan_for(job["kind"], job["profile"]))
                  if attempt > 1 or job.get("parent_job") else 0)
        if (job["kind"] == "sketch" and not job.get("parent_job")
                and not storage.input_path(job["user_id"], job_id).is_file()):
            job_event(storage, job, volume, "job_end", a=attempt, state="failed")
            store.job_finish(job_id, attempt, "failed", error_code="input_missing", error_stage="0",
                             error_message="The uploaded sketch was not saved.")
            return "failed"
        seconds = cost = compute = 0.0
        # Outcomes count new work. Stored stage rows also include earlier attempts and restarted groups.
        own_stages = {s.key for s in plan_for(job["kind"], job["profile"]) if s.group in GROUPS}
        for group in (("model",) if job.get("stop_after") else GROUPS):
            try:
                res = execute(job, attempt, group, resume)
            except Exception as e:
                log.exception("group %s of %s failed", group, job_id)
                res = GroupResult("error", None, "worker_error", "A worker stopped unexpectedly.", True,
                                  detail=traceback.format_exc())
                event("group_crashed", job=job_id, attempt=attempt, group=group, error=type(e).__name__)
            seconds += res.seconds
            cost += res.cost_usd
            compute += res.compute_usd
            recorded = sum(r["cost_usd"] or 0.0 for r in store.job_stages(job_id) if r["stage"] in own_stages)
            carried = max(0.0, recorded - cost - compute)
            detail = {"model_and_gpu_usd": round(cost, 6), "compute_usd": round(compute, 6),
                      "previous_work_usd": round(carried, 6),
                      "basis": "estimate: stage records' model and GPU estimates plus wall seconds x the worker's "
                               "list price"}
            if group == 'blender' and res.status == 'error':
                if volume is not None:
                    volume.reload()
                rows = {r['stage']: r for r in store.job_stages(job_id)}
                if rows.get('3', {}).get('state') in ('passed', 'skipped') and website_model(run_dir):
                    ctx = make_context(settings, storage, job)
                    spec = next(s for s in plan_for(job['kind'], job['profile']) if s.key == '3f')
                    skip_stage(store, job_id, attempt, spec, ctx,
                               'Texture refine was skipped: its worker stopped. Showing the unrefined texture.',
                               res.detail or res.message)
                    if volume is not None:
                        volume.commit()
                    res.status = 'ok'
            if group == 'blender' and res.status == 'ok' and not job.get('stop_after'):
                if volume is not None:
                    volume.reload()
                if website_model(run_dir) is None:
                    res = GroupResult('error', '3', 'model_missing',
                                      'The 3D shape could not be saved. Please create another hero.')
                elif store.job_model_ready(job_id, attempt, bytes_=_bytes(storage, job, volume)):
                    job_event(storage, job, volume, "model_ready", a=attempt)
                    if redispatch is not None:
                        redispatch(job_id)
            ready = (store.job_get(job_id) or {}).get('state') == 'succeeded'
            if ready and res.status in ('error', 'rejected'):
                # Worker infrastructure failures during export also preserve visitor success.
                ctx = make_context(settings, storage, job)
                for spec in plan_for(job['kind'], job['profile']):
                    rows = {r['stage']: r for r in store.job_stages(job_id)}
                    if spec.group == group and rows.get(spec.key, {}).get('state') not in ('passed', 'skipped'):
                        skip_stage(store, job_id, attempt, spec, ctx, res.message, res.detail or res.message)
                if volume is not None:
                    volume.commit()
                res.status = 'ok'
            if res.status == "stale":
                event("job_stale", job=job_id, attempt=attempt, group=group)
                now = store.job_get(job_id) or {}
                if now.get("state") == "canceled" and now.get("deleted_at") is None and now.get("attempts") == attempt:
                    job_event(storage, job, volume, "cancel", a=attempt)
                    job_event(storage, job, volume, "job_end", a=attempt, state="canceled")
                return now.get("state")
            if res.status == "rejected":
                job_event(storage, job, volume, "job_end", a=attempt, state="rejected",
                          s=round(time.monotonic() - t0, 2), usd=round(carried + cost + compute, 6))
                store.job_finish(job_id, attempt, "rejected", error_code="rejected", error_stage=res.stage,
                                 error_message=res.message, seconds=round(time.monotonic() - t0, 2),
                                 cost_usd=round(carried + cost + compute, 6), cost_detail=detail,
                                 bytes_=_bytes(storage, job, volume))
                event("job_finished", job=job_id, attempt=attempt, state="rejected", stage=res.stage)
                return "rejected"
            if res.status == "error":
                if res.retry:
                    state = store.job_requeue(job_id, attempt, settings.max_attempts, res.code, res.stage,
                                              res.message + " It was tried again automatically.")
                    if state == "queued":
                        job_event(storage, job, volume, "requeue", a=attempt, next=attempt + 1,
                                  of=settings.max_attempts, step=res.stage, code=res.code or None)
                    else:
                        job_event(storage, job, volume, "job_end", a=attempt, state="failed")
                else:
                    job_event(storage, job, volume, "job_end", a=attempt, state="failed",
                              s=round(time.monotonic() - t0, 2), usd=round(carried + cost + compute, 6))
                    store.job_finish(job_id, attempt, "failed", error_code=res.code, error_stage=res.stage,
                                     error_message=res.message,
                                     seconds=round(time.monotonic() - t0, 2), cost_usd=round(carried + cost + compute, 6),
                                     cost_detail=detail, bytes_=_bytes(storage, job, volume))
                    state = "failed"
                event("job_finished" if state != "queued" else "job_requeued", job=job_id, attempt=attempt,
                      state=state, stage=res.stage, code=res.code)
                if state == "queued" and redispatch is not None:
                    redispatch(job_id)
                return state
        took = round(time.monotonic() - t0, 2)
        final_state = "stopped" if job.get("stop_after") else "succeeded"
        if final_state == 'succeeded':
            from herogen.service.bundle import BundleUnavailable, roblox_files

            try:
                rows = {r['stage']: r for r in store.job_stages(job_id)}
                if any(rows.get(k, {}).get('state') != 'passed' for k in ('4', '6')):
                    raise BundleUnavailable('The Roblox export could not be prepared.')
                roblox_files(run_dir)
                export_state, export_message = 'ready', ''
            except BundleUnavailable as exc:
                export_state = 'unavailable'
                rows = {r['stage']: r for r in store.job_stages(job_id)}
                export_message = next((rows[k]['message'] for k in ('4', '6')
                                       if rows.get(k, {}).get('state') in ('skipped', 'rejected') and rows[k]['message']), str(exc))
            for prefix in ('Roblox body unavailable for this hero: ', 'Roblox files unavailable for this hero: '):
                export_message = export_message.removeprefix(prefix)
            job_event(storage, job, volume, "job_end", a=attempt, state="succeeded", export=export_state, s=took,
                      usd=round(carried + cost + compute, 6))
            store.job_export_finish(job_id, attempt, state=export_state, message=export_message,
                                    cost_detail=detail,
                                    bytes_=_bytes(storage, job, volume))
        else:
            job_event(storage, job, volume, "job_end", a=attempt, state=final_state, s=took,
                      usd=round(carried + cost + compute, 6))
            store.job_finish(job_id, attempt, final_state, seconds=took, cost_usd=round(carried + cost + compute, 6),
                             cost_detail=detail, bytes_=_bytes(storage, job, volume))
        event("job_finished", job=job_id, attempt=attempt, state=final_state, seconds=took,
              cost_usd=round(carried + cost + compute, 6))
        return final_state


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
    orphan_runs = 0
    for uid, jid, _path in storage.runs_before(time.time() - 86400.0):
        if store.job_get(jid) is None:
            storage.delete_job(uid, jid)
            orphan_runs += 1
    if volume is not None:
        volume.commit()
    counts = store.purge_expired(settings.audit_retention_days * 86400.0,
                                 settings.analytics_retention_days * 86400.0)
    event("retention", jobs=len(removed), cache_files=cache_files, orphan_inputs=orphans, orphan_runs=orphan_runs, **counts)
    return {"jobs": removed, "cache_files": cache_files, "orphan_inputs": orphans, "orphan_runs": orphan_runs, **counts}


class ThreadBackend:
    """Dev mode: dispatch jobs to local background threads."""

    def __init__(self, store, settings: Settings, storage: Storage, runner: StageRunner) -> None:
        import concurrent.futures

        self.store, self.settings, self.storage, self.runner = store, settings, storage, runner
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=32,
                                                          thread_name_prefix="herogen-job")
        self.futures: dict[str, Any] = {}
        self.n = 0

    def execute(self, job: dict[str, Any], attempt: int, group: str, resume: int) -> GroupResult:
        ctx = make_context(self.settings, self.storage, job,
                           spend_limits_enabled=self.store.generation_get()["spend_limits_enabled"])
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
        finally:
            dispatch_queued(self.store, self.settings, self.submit)

    def wait(self, timeout: float = 30.0) -> None:
        import concurrent.futures

        deadline = time.monotonic() + timeout
        while True:   # completion can dispatch new futures: wait until none are left
            pending = [f for f in list(self.futures.values()) if not f.done()]
            if not pending or time.monotonic() > deadline:
                return
            concurrent.futures.wait(pending, timeout=max(0.0, deadline - time.monotonic()))
