"""Frozen-atlas repaint waves with isolated requests and ordered texture projection."""
from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import hashlib
import json
from datetime import datetime, timezone
import os
import shutil
import threading
import uuid
from types import MethodType

from herogen.pipeline.client import CallLog
from herogen.refine_budget import BudgetStop

# A face close-up plus cardinal body coverage, followed by the upper back's hair pass.
WAVES = ((('head+0', False), ('body+0', False), ('body+90', False),
          ('body-90', False), ('body+180', False)), (('upper+180', True),))
CONCURRENCY = 5


def isolated_client(parent, key, persistence_lock=None):
    """Keep request files and call slices private; reuse complete root cache families on demand.

    Copying preserves subclass transport guards and durable response hooks. No two workers write the
    same cache path. Failed calls stay in their private ledger; completed families are published after drain.
    """
    child = copy.copy(parent)
    child.log = CallLog()
    child.cache_dir = parent.cache_dir / 'fast' / key
    child.cache_dir.mkdir(parents=True, exist_ok=True)
    load = child._load
    persist = child._save_entry
    persistence_lock = persistence_lock or threading.RLock()

    def durable(self, request_key, entry):
        # Subclasses may commit a shared remote Volume after saving an entry. Keep those commits serial.
        with persistence_lock:
            persist(request_key, entry)

    child._save_entry = MethodType(durable, child)

    def cached(self, request_key):
        entry = load(request_key)
        if entry is None and (parent.cache_dir / f'{request_key}.json').is_file():
            for source in parent.cache_dir.glob(request_key + '*'):
                if source.is_file():
                    shutil.copy2(source, self.cache_dir / source.name)
            entry = load(request_key)
        return entry

    child._load = MethodType(cached, child)
    return child


def publish(parent, child):
    """Publish after all workers drain, with image/mask families before their entry files."""
    for source in sorted(child.cache_dir.iterdir(), key=lambda p: p.suffix == '.json'):
        if source.is_file() and source.name != 'calls-ledger.jsonl':
            target = parent.cache_dir / source.name
            temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.part')
            try:
                shutil.copy2(source, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
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
        self.completed = {}
        self.selected = None

    def submit(self, pool, client, prep, frozen, tex, cam, rendered, timeout, cover, head_found,
               body_cam, recipe, persistence_lock, user_note=""):
        from . import _view, imaging as I
        for index in range(I.HAIR_ATTEMPTS if self.hair else I.ATTEMPTS):
            child = isolated_client(client, self.key + f'/attempt-{index}', persistence_lock)
            if hasattr(child, 'network_attempts'):
                child._initial_network_attempts = child.network_attempts
            private = {'accepted': False, 'reason': 'incomplete', 'tries': 0, 'attempts': []}
            folder = self.folder / f'attempt-{index}'
            future = pool.submit(_view, child, prep, frozen, tex, cam, self.hair, folder, private,
                                 timeout, cover, face_exclusion=head_found, body_cam=body_cam,
                                 repaint=recipe, rendered=rendered, apply=lambda *args: 0,
                                 attempts=1, attempt_start=index, cancelled=self.cancelled, user_note=user_note)
            self.tasks.append((future, child, private, folder))

    def collect(self, index):
        future, _, private, folder = self.tasks[index]
        try:
            future.result()
        except CancelledError:
            private['reason'] = 'cancelled: lower attempt accepted'
        except BudgetStop as exc:
            private['reason'] = str(exc)
        self.completed[index] = private
        self.record['attempts'] = [trial for i in sorted(self.completed)
                                   for trial in self.completed[i]['attempts']]
        self.record['tries'] = len(self.record['attempts'])
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
                    self.folder.mkdir(parents=True, exist_ok=True)
                    for source in self.tasks[i][3].iterdir():
                        if source.is_file():
                            shutil.copy2(source, self.folder / source.name)
                    self.cancelled.set()
                    for task, *_ in self.tasks[i + 1:]:
                        task.cancel()
                break
        if self.selected is None and len(self.completed) == len(self.tasks):
            self.record['reason'] = self.completed[len(self.tasks) - 1]['reason']


def reproject(tex, work, records, body_cam):
    """Rebuild from the immutable base and saved accepted paint, in the declared pass order."""
    import numpy as np
    from . import imaging as I

    views = I.load_accepted_views(work, records, tex.height, body_cam=body_cam)
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
        records[view['key']]['texels'] = tex.apply(view['cam'], rgb, feather, view['depth'], view['valid'],
                                                 weight, I.HAIR_BONUS if view['hair'] else 1.0)


class Publisher:
    """Publish immutable GLBs before atomic progress pointers, retaining every progress snapshot."""
    def __init__(self, run, prep, tex, work, data, body_cam, timeout):
        self.run, self.prep, self.tex, self.work = run, prep, tex, work
        self.data, self.body_cam, self.timeout = data, body_cam, timeout
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
        if not set(accepted) > set(previous):
            return None
        reproject(self.tex, self.work, records, self.body_cam)
        n = len(self.progress['versions']) + 1
        texture = self.tex.save(self.work / f'texture-v{n}.png')
        name = f'03-refined-v{n}.glb'
        target = self.run / name
        temporary = target.with_name(target.stem + '.part.glb')
        try:
            blender('export', {'blend': str(self.prep / 'work.blend'), 'texture': str(texture),
                               'out': str(temporary)}, self.work / 'blender.log', timeout=self.timeout)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        version = {'n': n, 'glb': name, 'accepted': accepted,
                   'rejected': [key for key, r in records.items() if not r['accepted'] and r['reason'] != 'incomplete'],
                   'pending': [key for key, r in records.items() if not r['accepted'] and r['reason'] == 'incomplete'],
                   'time': datetime.now(timezone.utc).isoformat(),
                   'atlas_sha256': hashlib.sha256(texture.read_bytes()).hexdigest(),
                   'code_sha': (self.data.get('code') or {}).get('commit'), 'wave': wave}
        self.progress['versions'].append(version)
        self.write()
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
          user_note=""):
    from . import imaging as I
    from .backend import blender

    stopped, error = False, None
    persistence_lock = threading.RLock()
    data['waves'] = []
    all_passes = []
    for sequence in WAVES:
        for name, hair in sequence:
            key = ('hair-' if hair else '') + name
            record = {'accepted': False, 'reason': 'incomplete', 'tries': 0, 'attempts': []}
            data['views'][key] = record
            if not head_found and name.startswith('head'):
                record['reason'] = 'head_not_found'
    publisher = Publisher(work.parent.parent, prep, tex, work, data, cams['body+0'], timeout)

    def collect(job, index):
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

    try:
        # Every attempt can run, while the inherited client limiter still bounds HTTP concurrency/RPM.
        with ThreadPoolExecutor(max_workers=CONCURRENCY * I.ATTEMPTS + I.HAIR_ATTEMPTS) as pool:
            remaining = {}
            for wave_index, sequence in enumerate(WAVES, 1):
                jobs = []
                for name, hair in sequence:
                    key = ('hair-' if hair else '') + name
                    record = data['views'][key]
                    if record['reason'] == 'head_not_found':
                        continue
                    if stopped or error is not None:
                        record['reason'] = 'not attempted: API cap' if stopped else 'not attempted: model error'
                        continue
                    jobs.append(Pass(key, name, hair, record, work / key))
                if not jobs:
                    continue
                folder = work / f'wave-{wave_index}'
                data['waves'].append({'views': [j.key for j in jobs], 'frozen_atlas': os.path.relpath(current, work),
                                      'concurrency': CONCURRENCY, 'parallel_attempts': True})
                save(ok=False, status='running')
                blender('render', {'blend': str(prep / 'work.blend'), 'texture': str(current),
                                   'cams': [cams[j.name] for j in jobs], 'out': str(folder), 'pos': True},
                        folder / 'blender.log', timeout=timeout)
                for job in jobs:
                    job.submit(pool, client, prep, current, tex, cams[job.name], folder, timeout, cover,
                               head_found, cams['body+0'], recipe, persistence_lock, user_note=user_note)
                    all_passes.append(job)
                    for index, (future, *_) in enumerate(job.tasks):
                        remaining[future] = (job, index, wave_index)
                # Only first tries contribute at this barrier, even if retries already finished.
                if wave_index == 1:
                    firsts = {job.tasks[0][0] for job in jobs}
                    while firsts:
                        completed, _ = wait(firsts, return_when=FIRST_COMPLETED)
                        for future in sorted(completed, key=lambda f: all_passes.index(remaining[f][0])):
                            job, index, _ = remaining.pop(future)
                            collect(job, index)
                            firsts.remove(future)
                else:
                    firsts = {job.tasks[0][0] for job in jobs}
                    while firsts:
                        eligible = [f for f, (_, index, wave) in remaining.items() if wave < wave_index or index == 0]
                        completed, _ = wait(eligible, return_when=FIRST_COMPLETED)
                        for future in sorted(completed, key=lambda f: (all_passes.index(remaining[f][0]), remaining[f][1])):
                            job, index, wave = remaining.pop(future)
                            firsts.discard(future)
                            before = job.selected
                            collect(job, index)
                            if wave < wave_index and error is None and before is None and job.selected is not None:
                                publisher.publish(wave)
                if error is None:
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
                    collect(job, index)
                    if error is None and before is None and job.selected is not None:
                        publisher.publish(wave)
                save(ok=False, status='running')
    finally:
        for job in all_passes:
            for _, child, *_ in job.tasks:
                publish(client, child)
    save(ok=False, status='running')
    if error is not None:
        raise error
    publisher.finish()
    if publisher.progress['versions']:
        last = publisher.progress['versions'][-1]['n']
        shutil.copyfile(work / f'texture-v{last}.png', work / 'texture.png')
        current = work / 'texture.png'
    return current, stopped
