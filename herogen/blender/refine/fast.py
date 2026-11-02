"""Frozen-atlas repaint waves with isolated requests and ordered texture projection."""
from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, wait
import contextvars
import copy
import hashlib
import json
from datetime import datetime, timezone
import os
import shutil
import threading
import time
import uuid
from types import MethodType

from herogen.pipeline.client import CallLog
from herogen.refine_budget import BudgetStop

from . import buildlog

# A face close-up plus cardinal body coverage, followed by the upper back's hair pass.
WAVES = ((('head+0', False), ('body+0', False), ('body+90', False),
          ('body-90', False), ('body+180', False)), (('upper+180', True),))
CONCURRENCY = 5
ATTEMPTS = 2


def isolated_client(parent, key, persistence_lock=None):
    """Keep request files and call slices private; reuse complete root cache families on demand.

    Copying preserves subclass transport guards and durable response hooks. No two workers write the
    same cache path. Failed calls stay in their private ledger; completed families are published after drain.
    """
    child = copy.copy(parent)
    child.log = CallLog()
    child.cache_dir = parent.cache_dir / 'fast' / key
    child.cache_dir.mkdir(parents=True, exist_ok=True)
    # A nested wave must bind the original hooks to its own cache directory;
    # shallow-copied wrappers still close over the outer wave's client.
    child._isolated_load_impl = parent.__dict__.get('_isolated_load_impl', type(parent)._load)
    child._isolated_save_impl = parent.__dict__.get('_isolated_save_impl', type(parent)._save_entry)
    load = MethodType(child._isolated_load_impl, child)
    persist = MethodType(child._isolated_save_impl, child)
    persistence_lock = persistence_lock or threading.RLock()

    def durable(self, request_key, entry):
        # Subclasses may commit a shared remote Volume after saving an entry. Keep those commits serial.
        with persistence_lock:
            persist(request_key, entry)

    child._save_entry = MethodType(durable, child)

    def cached(self, request_key):
        # Nested siblings can request the same parent family concurrently. Keep the
        # entry unreadable until its complete image/mask family has been copied.
        with persistence_lock:
            entry = load(request_key)
            if entry is None and parent._load(request_key) is not None:
                for source in parent.cache_dir.glob(request_key + '*'):
                    if source.is_file():
                        shutil.copy2(source, self.cache_dir / source.name)
                entry = load(request_key)
            return entry

    child._load = MethodType(cached, child)
    return child


def publish_files(parent, child):
    """Copy a child's cache families to the parent, with image/mask families before their entry files."""
    for source in sorted(child.cache_dir.iterdir(), key=lambda p: p.suffix == '.json'):
        if source.is_file() and source.name != 'calls-ledger.jsonl':
            target = parent.cache_dir / source.name
            # Shared filesystems can refuse replacing an open, already identical cache file.
            if target.is_file() and source.read_bytes() == target.read_bytes():
                continue
            temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.part')
            try:
                shutil.copy2(source, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)


def publish(parent, child):
    """Publish after all workers drain: cache families first, then the call slice and network count."""
    publish_files(parent, child)
    parent.log.calls.extend(child.log.calls)
    # Instrumented clients may count network posts separately from their billable call log.
    if hasattr(parent, 'network_attempts'):
        parent.network_attempts += child.network_attempts - child._initial_network_attempts




def empty_progress(run, *, done=False):
    """Reset the pointer before preparation, or finish a preparation-time fallback without versions."""
    from herogen.blender.record import write_atomic
    progress = {'profile': 'fast', 'versions': [], 'done': done, 'final_n': None}
    text = json.dumps(progress, indent=2)
    write_atomic(run / 'stage-3f-progress.json', text)
    if not done:
        write_atomic(run / 'stage-3f-progress-initial.json', text)


class Pass:
    """One frozen view's independent attempts and deterministic selection frontier."""
    def __init__(self, key, name, hair, record, folder):
        self.key, self.name, self.hair, self.record, self.folder = key, name, hair, record, folder
        self.cancelled = threading.Event()
        self.tasks = []
        self.tries = []      # the build log's try number (the slot, 1-based) of each launched task
        self.completed = {}
        self.selected = None
        self.slots, self.conditional = 0, 0
        self.prepare = self.launch = None
        self.reported = False

    def submit(self, pool, client, prep, frozen, tex, cam, rendered, timeout, cover, head_found,
               body_cam, recipe, persistence_lock, user_note="", view_options=None, art=(), art_masks=None, art_framing=None,
               attempts=ATTEMPTS, conditional=0):
        """Submit `attempts` slots now; `conditional` further slots are launched only by `launch`."""
        from . import _attempt_cancelled, _view

        def prepare(index):
            # Build one slot's isolated client, record and task without submitting it.
            view_cover, options = (cover, {}) if view_options is None else view_options(self, index, cover)
            standard_side = bool(art and index == 1 and options.get('recipe_name') == 'RF6B')
            if standard_side:
                # Keep one independent ordinary artwork repaint when beauty references move the pose.
                view_cover = cover
                options = {key: value for key, value in options.items()
                           if key not in ('recipe_name', 'reference_record')}
                options['attempt_start'] = 0
            attempt_index = options.pop('attempt_start', index)
            attempt_render = options.pop('rendered', rendered)
            attempt_art = options.pop('art', art)
            attempt_masks = options.pop('art_masks', art_masks)
            before = options.pop('before', None)
            child = isolated_client(client, self.key + f'/attempt-{index}', persistence_lock)
            if hasattr(child, 'network_attempts'):
                child._initial_network_attempts = child.network_attempts
            # Every request of this try (the repaint and its SAM 3.1 checks) is logged with its view and try.
            child.purpose = 'repaint'
            child.event_fields = {'view': self.key, 'try_': index + 1}
            private = {'accepted': False, 'reason': 'incomplete', 'tries': 0, 'attempts': []}
            if standard_side:
                private['reference_style'] = 'turnaround artwork and original cover'
            if art_framing:
                private['art_framing'] = copy.deepcopy(art_framing)
            folder = self.folder / f'attempt-{index}'

            def run():
                token = _attempt_cancelled.set(self.cancelled)
                try:
                    if before is not None:
                        before()
                    return _view(child, prep, frozen, tex, cam, self.hair, folder, private,
                                 timeout, view_cover, face_exclusion=head_found, body_cam=body_cam,
                                 repaint=recipe, rendered=attempt_render, apply=lambda *args: 0,
                                 attempts=1, attempt_start=attempt_index, cancelled=self.cancelled,
                                 user_note=user_note, art=attempt_art, art_masks=attempt_masks, fast_outline=True,
                                 **options)
                finally:
                    _attempt_cancelled.reset(token)
            return run, child, private, folder

        def launch(index):
            run, child, private, folder = prepare(index)
            # Worker threads start with an empty context; carry this one (the build-log step) into each try.
            future = pool.submit(contextvars.copy_context().run, run)
            self.tasks.append((future, child, private, folder))
            self.tries.append(child.event_fields['try_'])
            return future

        self.prepare, self.launch = prepare, launch
        self.slots, self.conditional = attempts, conditional
        for index in range(attempts):
            launch(index)

    def conditional_due(self, overruled=()):
        """A conditional slot is due only when every submitted slot was judged and rejected by its gates.

        Cancelled, budget-stopped and failed slots never count as rejections. `overruled` lists
        gate-accepted slots that a later existing selection rejected; they count as rejections.
        """
        if (not self.slots <= len(self.tasks) < self.slots + self.conditional
                or len(self.completed) != len(self.tasks)):
            return False
        for index, (future, _, private, _) in enumerate(self.tasks):
            if not future.done() or future.cancelled() or future.exception() is not None:
                return False
            reason = private['attempts'][-1]['reason'] if private['attempts'] else 'incomplete'
            if ((private['accepted'] and index not in overruled) or reason == 'incomplete'
                    or reason.startswith('cancelled')):
                return False
        return True

    def collect(self, index):
        future, _, private, folder = self.tasks[index]
        try:
            future.result()
        except CancelledError:
            private['reason'] = 'cancelled: lower attempt accepted'
        except BudgetStop as exc:
            private['reason'] = str(exc)
        if private.get('reference_style'):
            for trial in private['attempts']:
                trial['reference_style'] = private['reference_style']
        self.completed[index] = private
        self.record['attempts'] = [trial for i in sorted(self.completed)
                                   for trial in self.completed[i]['attempts']]
        self.record['tries'] = len(self.record['attempts'])
        if private['reason'] == 'cancelled: lower attempt accepted':
            buildlog.try_cancelled(self.key, self.tries[index], private, self.tasks[index][1].log.calls)
        # A higher-index answer cannot win until every lower attempt is known to have failed.
        for i in range(len(self.tasks)):
            if i not in self.completed:
                break
            candidate = self.completed[i]
            if candidate['accepted']:
                if self.selected is None:
                    self.selected = i
                    self.record.update({k: v for k, v in candidate.items() if k not in ('attempts', 'tries')})
                    self.record['selected_attempt'] = i
                    self.record['winning_slot'] = i + 1
                    self.folder.mkdir(parents=True, exist_ok=True)
                    for source in self.tasks[i][3].iterdir():
                        if source.is_file():
                            shutil.copy2(source, self.folder / source.name)
                    self.cancelled.set()
                    for task, *_ in self.tasks[i + 1:]:
                        task.cancel()
                    if not self.reported:
                        self.reported = True
                        buildlog.view_selected(self.key, self.tries[i], len(self.tries))
                break
        if self.selected is None and len(self.completed) == len(self.tasks):
            self.record['reason'] = self.completed[len(self.tasks) - 1]['reason']
            if len(self.tasks) >= self.slots + self.conditional:
                self.report_kept()   # no conditional slot can follow; otherwise the wave decides

    def report_kept(self):
        """Log that the view keeps its original paint, once every launched try is known and none will follow.
        Passes with their own selection (the face close-up's beauty slots) log no outcome here."""
        if (self.reported or self.selected is not None or len(self.completed) != len(self.tasks)
                or type(self).collect is not Pass.collect):
            return
        self.reported = True
        buildlog.view_kept(self.key, self.record.get('reason'), len(self.tries))


def reproject(tex, work, records, body_cam, recovery=None):
    """Rebuild from the immutable base and saved accepted paint, in the declared pass order.

    An applied face recovery paints at head+0's position, first, only while no frontal is accepted.
    """
    import numpy as np
    from . import face_recovery, imaging as I

    views = I.load_accepted_views(work, records, tex.height, body_cam=body_cam)
    recovered = face_recovery.view(work, records, recovery, tex.height, body_cam)
    if recovered is not None:
        views.insert(0, recovered)
    tex.reset()
    gains = I.solve_exposure(views, tex.height, mode=I.EXPOSURE_MODE) if I.EXPOSURE_SOLVE else {}
    for view in views:
        gain, offset = gains.get(view['key'], (1.0, 0.0))
        figure = view['alpha'] > .5
        rgb = I.apply_exposure(view['refined'], figure, gain, offset)
        feather = view['base_feather'].copy()
        if I.CLOSEUP_FEATHER and (view['hair'] or view['closeup']):
            feather *= I.closeup_feather(figure)
        weight = view['weight'].copy()
        if view['hair']:
            current, has_current = tex.render_view(view['cam'], view['depth'], view['valid'])
            valid = has_current & figure
            if valid.any():
                sigma = I.AGREE_SIGMA * max(weight.shape)
                delta = np.abs(I.masked_blur(rgb, valid, sigma) - I.masked_blur(current, valid, sigma)).max(-1)
                lo, hi = I.AGREE_TOL
                weight *= np.where(has_current, np.clip((hi - delta) / (hi - lo), 0, 1), 1).astype(np.float32)
        texels = tex.apply(view['cam'], rgb, feather, view['depth'], view['valid'],
                                                 weight, I.HAIR_BONUS if view['hair'] else 1.0)
        if view is recovered:
            recovery['texels'] = texels
        else:
            records[view['key']]['texels'] = texels


class Publisher:
    """Publish immutable GLBs before atomic progress pointers, retaining every progress snapshot."""
    def __init__(self, run, prep, tex, work, data, body_cam, timeout, face_control=None, palette_manifest=None):
        self.run, self.prep, self.tex, self.work = run, prep, tex, work
        self.data, self.body_cam, self.timeout = data, body_cam, timeout
        self.face_control, self.palette_manifest = face_control, palette_manifest
        self.progress = {'profile': 'fast', 'versions': [], 'done': False, 'final_n': None}
        self.write()
        from herogen.blender.record import write_atomic
        write_atomic(self.run / 'stage-3f-progress-initial.json', json.dumps(self.progress, indent=2))

    def write(self):
        from herogen.blender.record import write_atomic
        write_atomic(self.run / 'stage-3f-progress.json', json.dumps(self.progress, indent=2))

    def publish(self, wave):
        from .backend import blender
        from herogen.blender.record import write_atomic
        records = self.data['views']
        accepted = [key for key, record in records.items() if record['accepted']]
        previous = self.progress['versions'][-1]['accepted'] if self.progress['versions'] else []
        recovery = self.data.get('face_recovery')
        recovered = (recovery['attempt'] if recovery and recovery.get('applied')
                     and not any(records.get(key, {}).get('accepted') for key in ('head+0', 'body+0')) else None)
        before = self.progress['versions'][-1].get('face_recovery') if self.progress['versions'] else None
        if (not set(accepted) > set(previous) and recovered == before
                and (self.progress['versions'] or self.face_control is None)):
            return None
        reproject(self.tex, self.work, records, self.body_cam, recovery)
        if self.face_control is not None:
            self.face_control.paint(self.tex, self.work, records)
        n = len(self.progress['versions']) + 1
        texture = self.tex.save(self.work / f'texture-v{n}.png')
        export_texture = texture
        if manifest := self.palette_manifest:
            from .region_palette import export_texture as grade_export
            protection = (self.face_control.default_head_weight if self.face_control is not None else None)
            export_texture, grades = grade_export(manifest, self.prep, texture, self.body_cam,
                                             self.work / f'region-palette-v{n}', timeout=self.timeout,
                                             **({'protected_head': protection} if protection is not None else {}))
            self.data['region_palette'] = grades
        name = f'03-refined-v{n}.glb'
        target = self.run / name
        temporary = target.with_name(target.stem + '.part.glb')
        try:
            blender('export', {'blend': str(self.prep / 'work.blend'), 'texture': str(export_texture),
                               'out': str(temporary)}, self.work / 'blender.log', timeout=self.timeout)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        version = {'n': n, 'glb': name, 'accepted': accepted,
                   'rejected': [key for key, r in records.items() if not r['accepted'] and r['reason'] != 'incomplete'],
                   'pending': [key for key, r in records.items() if not r['accepted'] and r['reason'] == 'incomplete'],
                   'time': datetime.now(timezone.utc).isoformat(),
                   'atlas_sha256': hashlib.sha256(export_texture.read_bytes()).hexdigest(),
                   'code_sha': (self.data.get('code') or {}).get('commit'), 'wave': wave}
        if self.face_control is not None:
            version['face_source'] = dict(self.data['face_source'])
        if recovered is not None:
            version['face_recovery'] = recovered
        self.progress['versions'].append(version)
        self.write()
        buildlog.version(n, wave, len(accepted), len(records), len(version['rejected']), len(version['pending']))
        # Immutable snapshots let recorded runners reproduce every update, including the done transition.
        write_atomic(self.run / f'stage-3f-progress-v{n}.json', json.dumps(self.progress, indent=2))
        return texture

    def finish(self):
        if self.progress['versions']:
            last = self.progress['versions'][-1]
            target = self.run / '03-refined.glb'
            temporary = target.with_name('03-refined.part.glb')
            try:
                shutil.copyfile(self.run / last['glb'], temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            self.progress['final_n'] = last['n']
        self.progress['done'] = True
        self.write()


def waves(client, prep, current, tex, cams, work, data, timeout, cover, head_found, recipe, save, state_lock,
          user_note="", beauty_context=None, art_context=None, palette_manifest=None, attempts=ATTEMPTS,
          passes=None):
    """Run the frozen waves. `passes`, when given, receives every submitted Pass for later extension."""
    from .backend import blender
    from . import face_recovery

    stopped, error, face_error = False, None, None
    recovery_changed = False
    persistence_lock = threading.RLock()
    face_control = None
    face_client = None
    recovery_client = None
    if beauty_context is not None:
        from .fast_beauty import FaceControl
        face_control = FaceControl(*beauty_context, data=data, state_lock=state_lock)
        face_control.root_client, face_control.persistence_lock = client, persistence_lock
        face_control.side_art = art_context is not None
        face_client = isolated_client(client, 'face-source', persistence_lock)
        if hasattr(face_client, 'network_attempts'):
            face_client._initial_network_attempts = face_client.network_attempts
        try:
            stopped = bool(face_control.default_fast(face_client, prep, tex, cams, work, timeout, cover,
                                                   head_found, recipe, save, user_note))
        finally:
            publish(client, face_client)
            face_client.log.calls.clear()
            if hasattr(face_client, 'network_attempts'):
                face_client._initial_network_attempts = face_client.network_attempts
    data['waves'] = []
    all_passes = []
    for sequence in WAVES:
        for name, hair in sequence:
            key = ('hair-' if hair else '') + name
            record = {'accepted': False, 'reason': 'incomplete', 'tries': 0, 'attempts': []}
            data['views'][key] = record
            if not head_found and name.startswith('head'):
                record['reason'] = 'head_not_found'
    publisher = Publisher(work.parent.parent, prep, tex, work, data, cams["body+0"], timeout,
                          face_control=face_control, palette_manifest=palette_manifest)

    def collect(job, index, wave):
        """Collect one slot; return a newly launched conditional slot's future, if one became due."""
        nonlocal stopped, error
        try:
            # Never wait while holding the budget lock: HTTP authorization needs that same lock.
            try:
                job.tasks[index][0].result()
            except (CancelledError, BudgetStop):
                pass
            with state_lock:
                job.collect(index)
            if job.tasks[index][0].done() and not job.tasks[index][0].cancelled():
                exc = job.tasks[index][0].exception()
                if isinstance(exc, BudgetStop):
                    data['stop_reason'] = str(exc)
                    stopped = True
        except BudgetStop as exc:
            job.completed[index] = job.tasks[index][2]
            data['stop_reason'] = str(exc)
            stopped = True
        except Exception as exc:
            private = job.tasks[index][2]
            private['reason'] = type(exc).__name__
            if private['attempts']:
                private['attempts'][-1]['reason'] = type(exc).__name__
            with state_lock:
                job.completed[index] = private
                job.record.update(reason=type(exc).__name__,
                                  attempts=[a for i in sorted(job.completed) for a in job.completed[i]['attempts']])
                job.record['tries'] = len(job.record['attempts'])
            if error is None:
                error = exc
        if face_control is None:
            if job.name in ('head+0', 'body+0') and error is None and not stopped:
                recover()
            return settled(job)
        if error is not None or face_error is not None:
            return settled(job)
        if index >= job.slots:
            # The default-head run paid this view's ordinary attempt 1 for the conditional slot.
            try:
                face_control.fold_default(job.key)
            except BudgetStop as exc:
                with state_lock:
                    data['stop_reason'] = str(exc)
                stopped = True
            except Exception as exc:
                # A face-recovery model error stays loud after every submitted attempt drains.
                error = exc
                return settled(job)
        if stopped or not job.conditional_due():
            return settled(job)
        return launch_conditional(job, wave)

    def recover():
        """Ordinary fast only: resolve face-only paint once both frontal passes are rejected."""
        nonlocal stopped, error, recovery_changed, recovery_client
        with state_lock:
            records = copy.deepcopy({k: data['views'][k] for k in ('head+0', 'body+0') if k in data['views']})
            previous = data.get('face_recovery')
        if previous is not None or not face_recovery.due(records):
            return
        if recovery_client is None:
            recovery_client = isolated_client(client, 'head+0/face-recovery', persistence_lock)
            if hasattr(recovery_client, 'network_attempts'):
                recovery_client._initial_network_attempts = recovery_client.network_attempts
        try:
            record = face_recovery.resolve(recovery_client, work, records, cams['head+0'], cams['body+0'])
        except BudgetStop as exc:
            record = {'applied': False, 'view': 'head+0', 'reason': str(exc)}
            with state_lock:
                data['stop_reason'] = str(exc)
            stopped = True
        except Exception as exc:
            # Model and replay failures stay loud after every submitted attempt drains.
            if error is None:
                error = exc
            return
        with state_lock:
            data['face_recovery'] = record
        recovery_changed = bool(record and record.get('applied'))

    def take_recovery():
        nonlocal recovery_changed
        changed, recovery_changed = recovery_changed, False
        return changed

    def settled(job):
        # No conditional slot follows this collection: a view with every try rejected keeps its paint.
        with state_lock:
            job.report_kept()
        return None

    def launch_conditional(job, wave):
        future = job.launch(len(job.tasks))
        remaining[future] = (job, len(job.tasks) - 1, wave)
        buildlog.extra_try(job.key, len(job.tasks))
        return future

    try:
        # Every attempt can run, while the inherited client limiter still bounds HTTP concurrency/RPM.
        with ThreadPoolExecutor(max_workers=(CONCURRENCY + 1) * ATTEMPTS) as pool:
            remaining = {}
            for wave_index, sequence in enumerate(WAVES, 1):
                jobs = []
                for name, hair in sequence:
                    key = ('hair-' if hair else '') + name
                    record = data['views'][key]
                    if record['reason'] == 'head_not_found':
                        buildlog.view_skipped(key, record['reason'])
                        continue
                    if stopped or error is not None or face_error is not None:
                        record['reason'] = ('not attempted: API cap' if stopped else
                                            'not attempted: model error' if error is not None else
                                            'not attempted: face unavailable')
                        buildlog.view_skipped(key, record['reason'])
                        continue
                    factory = face_control.pass_type(name) if face_control is not None else Pass
                    jobs.append(factory(key, name, hair, record, work / key))
                if not jobs:
                    continue
                folder = work / f'wave-{wave_index}'
                data['waves'].append({'views': [j.key for j in jobs], 'frozen_atlas': os.path.relpath(current, work),
                                      'concurrency': CONCURRENCY, 'parallel_attempts': True})
                save(ok=False, status='running')
                buildlog.wave(wave_index, len(jobs), attempts, from_version=len(publisher.progress['versions'])
                              if wave_index > 1 else 0)
                render_started = time.monotonic()
                blender('render', {'blend': str(prep / 'work.blend'), 'texture': str(current),
                                   'cams': [cams[j.name] for j in jobs], 'out': str(folder), 'pos': True},
                        folder / 'blender.log', timeout=timeout)
                buildlog.rendered(wave_index, len(jobs), time.monotonic() - render_started)
                if face_control is not None and wave_index == 1 and head_found:
                    face_control.default_render = face_control.default_work / 'wave-1'
                for job in jobs:
                    art, masks, framing = (), None, None
                    if art_context is not None:
                        from . import art_references
                        art_run, info, artwork = art_context
                        art, masks, framing = art_references(art_run, cams[job.name], cams['body+0'], info, artwork)
                    job.submit(pool, client, prep, current, tex, cams[job.name], folder, timeout, cover,
                               head_found, cams['body+0'], recipe, persistence_lock, user_note=user_note,
                               art=art, art_masks=masks, art_framing=framing, attempts=attempts,
                               **({"view_options": face_control.options, "conditional": 1}
                                  if face_control is not None else {}))
                    all_passes.append(job)
                    for index, (future, *_) in enumerate(job.tasks):
                        remaining[future] = (job, index, wave_index)
                # Only first tries contribute at this barrier, even if retries already finished.
                if wave_index == 1:
                    firsts = {job.tasks[0][0] for job in jobs}
                    if face_control is not None:
                        # Resolve every frontal candidate before choosing the immutable face source.
                        firsts.update(f for job in jobs if job.name in ('head+0', 'body+0')
                                      for f, *_ in job.tasks)
                    while firsts:
                        completed, _ = wait(firsts, return_when=FIRST_COMPLETED)
                        for future in sorted(completed, key=lambda f: all_passes.index(remaining[f][0])):
                            job, index, _ = remaining.pop(future)
                            extra = collect(job, index, wave_index)
                            firsts.remove(future)
                            if extra is not None and job.name in ('head+0', 'body+0'):
                                # The face source is chosen only after a frontal conditional slot resolves.
                                firsts.add(extra)
                        if (not firsts and face_control is not None and not stopped
                                and error is None and face_error is None):
                            # Face selection rejecting a gate-accepted beauty head counts as a
                            # failed beauty sample: with slot 2 rejected, the head pays slot 3.
                            try:
                                with state_lock:
                                    head = face_control.face_conditional(jobs, work)
                            except Exception as exc:
                                # Like face selection below: drain every submitted future, then raise.
                                face_error, head = exc, None
                            if head is not None:
                                firsts.add(launch_conditional(head, wave_index))
                    if face_control is not None and error is None and face_error is None:
                        try:
                            with state_lock:
                                face_control.choose(jobs, work)
                        except Exception as exc:
                            # Drain and collect every submitted future before reporting a
                            # missing face. A late model error must still be recorded and raised.
                            face_error = exc
                        if face_error is None:
                            try:
                                fallback_stopped = face_control.default_fast(
                                    face_client, prep, tex, cams, work, timeout, cover,
                                    head_found, recipe, save, user_note)
                                stopped = stopped or bool(fallback_stopped)
                                face_control.prepare_region(face_client, work)
                            except BudgetStop as exc:
                                with state_lock:
                                    data['stop_reason'] = str(exc)
                                stopped = True
                            except Exception as exc:
                                with state_lock:
                                    data['face_source']['region_error'] = type(exc).__name__
                                face_error = exc
                else:
                    firsts = {job.tasks[0][0] for job in jobs}
                    while firsts:
                        eligible = [f for f, (_, index, wave) in remaining.items() if wave < wave_index or index == 0]
                        completed, _ = wait(eligible, return_when=FIRST_COMPLETED)
                        for future in sorted(completed, key=lambda f: (all_passes.index(remaining[f][0]), remaining[f][1])):
                            job, index, wave = remaining.pop(future)
                            firsts.discard(future)
                            before = job.selected
                            collect(job, index, wave)
                            if (error is None and face_error is None
                                    and ((wave < wave_index and before is None and job.selected is not None)
                                         or take_recovery())):
                                publisher.publish(wave)
                if error is None and face_error is None:
                    take_recovery()
                    texture = publisher.publish(wave_index)
                    if wave_index == 1 and texture is not None:
                        current = texture  # immutable v1; later retries never change wave 2's render input
                save(ok=False, status='running')
            # Resolve remaining retries after each wave's first-try publication. All paid work drains.
            while remaining:
                completed, _ = wait(remaining, return_when=FIRST_COMPLETED)
                # Resolve simultaneous results in fixed pass/attempt order, never score order.
                for future in sorted(completed, key=lambda f: (all_passes.index(remaining[f][0]), remaining[f][1])):
                    job, index, wave = remaining.pop(future)
                    before = job.selected
                    collect(job, index, wave)
                    if error is None and face_error is None and (
                            (before is None and job.selected is not None) or take_recovery()):
                        publisher.publish(wave)
                save(ok=False, status='running')
    finally:
        if passes is not None:
            passes.extend(all_passes)
        for job in all_passes:
            for _, child, *_ in job.tasks:
                publish(client, child)
        if face_client is not None:
            publish(client, face_client)
        if recovery_client is not None:
            publish(client, recovery_client)
    save(ok=False, status='running')
    if error is not None:
        raise error
    if face_error is not None:
        raise face_error
    publisher.finish()
    if publisher.progress['versions']:
        last = publisher.progress['versions'][-1]['n']
        shutil.copyfile(work / f'texture-v{last}.png', work / 'texture.png')
        current = work / 'texture.png'
    return current, stopped
