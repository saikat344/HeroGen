"""Frozen-atlas repaint waves with isolated requests and ordered texture projection."""
from concurrent.futures import ThreadPoolExecutor
import copy
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



def parallel_view(client, prep, current, tex, cam, hair, folder, record, timeout, cover, **kwargs):
    """Select the first passing index, regardless of completion order, and drain paid requests."""
    from . import _view, imaging as I

    count = I.HAIR_ATTEMPTS if hair else I.ATTEMPTS
    cancelled = threading.Event()
    futures, children = [], []
    persistence_lock = threading.RLock()
    error = None
    selected = None
    with ThreadPoolExecutor(max_workers=count) as pool:
        for index in range(count):
            child = isolated_client(client, f'attempt-{index}', persistence_lock)
            if hasattr(child, 'network_attempts'):
                child._initial_network_attempts = child.network_attempts
            children.append(child)
            private = {'accepted': False, 'reason': 'incomplete', 'tries': 0, 'attempts': []}
            projection = []
            future = pool.submit(_view, child, prep, current, tex, cam, hair, folder / f'attempt-{index}',
                                 private, timeout, cover, **{**kwargs, 'apply': lambda *a, p=projection: p.append(a) or 0},
                                 attempts=1, attempt_start=index, cancelled=cancelled)
            futures.append((future, private, projection, index))
        for future, private, projection, index in futures:
            try:
                future.result()
            except Exception as exc:
                if error is None:
                    error = exc
            if selected is None and private['accepted'] and error is None:
                selected = (private, projection, index)
                cancelled.set()
    for child in children:
        publish(client, child)
    record['attempts'] = [a for _, private, _, _ in futures for a in private['attempts']]
    record['tries'] = len(record['attempts'])
    if error is not None:
        raise error
    if selected is not None:
        private, projection, index = selected
        record.update({k: v for k, v in private.items() if k not in ('tries', 'attempts')})
        record['selected_attempt'] = index
        for source in (folder / f'attempt-{index}').iterdir():
            if source.is_file():
                shutil.copy2(source, folder / source.name)
        kwargs['apply'](*projection[0])
    else:
        record['reason'] = futures[-1][1]['reason']
    return current


def waves(client, prep, current, tex, cams, work, data, timeout, cover, head_found, recipe, save, state_lock):
    from . import _view
    from .backend import blender

    stopped = False
    persistence_lock = threading.RLock()
    data['waves'] = []
    for wave_index, sequence in enumerate(WAVES):
        jobs = []
        for name, hair in sequence:
            key = ('hair-' if hair else '') + name
            record = {'accepted': False, 'reason': 'incomplete', 'tries': 0, 'attempts': []}
            data['views'][key] = record
            if not head_found and name.startswith('head'):
                record['reason'] = 'head_not_found'
            elif stopped:
                record['reason'] = 'not attempted: API cap'
            else:
                jobs.append((key, name, hair, record))
        if not jobs:
            continue
        folder = work / f'wave-{wave_index + 1}'
        data['waves'].append({'views': [key for key, *_ in jobs], 'frozen_atlas': os.path.relpath(current, work),
                              'concurrency': CONCURRENCY})
        save(ok=False, status='running')
        # Blender remains serial. One process renders all the cameras from the same saved atlas.
        blender('render', {'blend': str(prep / 'work.blend'), 'texture': str(current),
                           'cams': [cams[name] for _, name, *_ in jobs], 'out': str(folder), 'pos': True},
                folder / 'blender.log', timeout=timeout)
        children = []
        futures = []
        error = None
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            for key, name, hair, record in jobs:
                child = isolated_client(client, key, persistence_lock)
                if hasattr(child, 'network_attempts'):
                    child._initial_network_attempts = child.network_attempts
                children.append(child)
                projection = []

                def defer(*args, target=projection):
                    target.append(args)
                    return 0

                # A worker owns its record until it is joined. Persist only completed records in save().
                private = copy.deepcopy(record)
                future = pool.submit(parallel_view, child, prep, current, tex, cams[name], hair, work / key,
                                     private, timeout, cover, face_exclusion=head_found, body_cam=cams['body+0'],
                                     repaint=recipe, rendered=folder, apply=defer)
                futures.append((future, key, record, private, projection))
            # Join in projection order regardless of API completion order. Every submitted worker drains,
            # including after a budget stop or model error; all its successful responses remain cached.
            for future, key, record, private, projection in futures:
                try:
                    future.result()
                except BudgetStop as exc:
                    private.update(accepted=False, reason=str(exc))
                    data['stop_reason'] = str(exc)
                    stopped = True
                except Exception as exc:
                    private.update(accepted=False, reason=type(exc).__name__)
                    if error is None:
                        error = exc
                with state_lock:
                    record.update(private)
                    if error is None and record['accepted']:
                        record['texels'] = tex.apply(*projection[0])
                        current = tex.save(work / key / 'texture.png')
        for child in children:
            publish(client, child)
        save(ok=False, status='running')
        if error is not None:
            raise error
        # The next wave can see this wave's paint. A cap stop admits no new work.
    return current, stopped
