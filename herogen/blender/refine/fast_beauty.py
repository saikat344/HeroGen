"""Beauty face ownership and closed default-head composition for opt-in repaint waves."""
import copy
import hashlib
import shutil
import threading

import numpy as np
from PIL import Image
from scipy import ndimage

from herogen.refine_budget import BudgetStop
from . import beauty, imaging as I
from .cameras import axes, project, px_per_unit
from .fast import Pass


def eye_quality(candidate, default, boxes):
    """Compare corresponding eye detail at a fixed scale, without another model call.

    Boxes come from the existing feature check's largest connected eye in each half.
    A small context margin includes eyelids. Both eyes must retain the ordinary paint's
    Laplacian variance and edge density; a high-contrast soft blob cannot win on contrast.
    """
    if candidate.size != default.size:
        raise ValueError('Eye comparison needs identical view dimensions')
    if len(boxes) != 2:
        return {'accepted': False, 'reason': 'two eye boxes unavailable', 'eyes': []}
    eyes = []
    for box in boxes:
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            raise ValueError('Eye comparison needs nonempty boxes')
        pad = round((x1 - x0) * .15)
        region = (max(0, x0 - pad), max(0, y0 - pad),
                  min(candidate.width, x1 + pad), min(candidate.height, y1 + pad))
        measures = []
        for image in (candidate, default):
            grey = np.asarray(image.crop(region).convert('L').resize(
                (64, 48), Image.Resampling.BILINEAR), dtype=np.float64) / 255
            gradient = np.hypot(ndimage.sobel(grey, axis=0) / 8, ndimage.sobel(grey, axis=1) / 8)
            measures.append({'laplacian_variance': float(ndimage.laplace(grey)[1:-1, 1:-1].var()),
                             'edge_density': float((gradient > .04).mean())})
        enhanced, ordinary = measures
        eyes.append({'box': list(region), 'beauty': enhanced, 'default': ordinary,
                     'ratios': {k: (enhanced[k] / ordinary[k] if ordinary[k] else None) for k in ordinary},
                     'eligible': all(ordinary[k] > 0 for k in ordinary),
                     'accepted': all(ordinary[k] > 0 and enhanced[k] >= ordinary[k] for k in ordinary)})
    accepted = all(eye['accepted'] for eye in eyes)
    reason = ('ordinary eye detail unavailable' if any(not eye['eligible'] for eye in eyes) else
              'accepted' if accepted else 'less eye detail than ordinary fast')
    return {'accepted': accepted, 'reason': reason,
            'eyes': eyes}


class FacePass(Pass):
    """The head's beauty and default slots are independent samples, never retries.

    A conditional third slot, launched only after both are rejected, is another default candidate.
    """

    def collect(self, index):
        future, _, private, folder = self.tasks[index]
        # The wave collector handles budget stops and propagates every other model error.
        try:
            future.result()
        except BudgetStop as exc:
            private['reason'] = str(exc)
        self.completed[index] = private
        for trial in private['attempts']:
            trial['face_candidate'] = 'beauty' if index == 0 else 'default'
        self.record['attempts'] = [a for i in sorted(self.completed) for a in self.completed[i]['attempts']]
        self.record['tries'] = len(self.record['attempts'])
        if index == 0:
            self.record.update({k: v for k, v in private.items() if k not in ('attempts', 'tries')})
            if private['accepted']:
                self.selected = 0
                self.record['selected_attempt'] = 0
                self.record['winning_slot'] = 1
                self.folder.mkdir(parents=True, exist_ok=True)
                for source in folder.iterdir():
                    if source.is_file():
                        shutil.copy2(source, self.folder / source.name)


class FaceControl:
    def __init__(self, folder, references, default_base=None, source_info=None, *, data, state_lock=None):
        self.folder, self.references, self.data = folder, references, data
        self.default_base, self.default_render = default_base, None
        self.source_info = source_info or {}
        self.state_lock = state_lock if state_lock is not None else threading.RLock()
        self.key = None
        self.fallback_texture = None
        self.hair_views = []
        self.default_head_weight = None
        self.default_work = None
        self.default_stopped = False
        # The default-head run's texture, records, passes and arguments, kept for conditional slots.
        self.default_tex = self.default_data = self.default_context = None
        self.default_passes = []
        self.extensions = {}
        self.root_client = self.persistence_lock = None

    def render_default(self, prep, cam, folder, timeout):
        """Render the same frozen geometry with default fast's original, ungraded atlas."""
        from .backend import blender

        self.default_render = folder.with_name(folder.name + '-default-face')
        blender('render', {'blend': str(prep / 'work.blend'), 'texture': str(self.default_base),
                           'cams': [cam], 'out': str(self.default_render), 'pos': True},
                self.default_render / 'blender.log', timeout=timeout)

    @staticmethod
    def pass_type(name):
        return FacePass if name == 'head+0' else Pass

    def options(self, job, index, cover):
        # The RF2R candidates use the original frozen base and artwork. Slot 2's sample index
        # matches default fast's attempt 0, and conditional slot 3's its attempt 1, for exact cache reuse.
        options = {'face_keep': True, 'save_face': True}
        if index in (1, 2):
            rendered = (self.default_work / ('wave-2' if job.hair else 'wave-1')
                        if self.default_work is not None else self.default_render)
            options = {**options, 'attempt_start': index - 1, 'rendered': rendered,
                       'art': (), 'art_masks': None}
            if index == 2:
                # The default-head run pays ordinary attempt 1 first; this slot replays its cache entry.
                options['before'] = lambda: self.extend_default(job.key)
            return cover, options
        angle = {'head+0': 0, 'body+0': 0, 'body+90': 90, 'body-90': -90,
                 'body+180': 180, 'upper+180': 180}[job.name]
        if beauty.selected_yaw(self.references, angle) is None:
            job.record['note'] = f'no face-free beauty reference at {angle:+g} yaw; standard recipe'
            return cover, options
        reference, record = beauty.reference(self.folder, self.references, angle, head=job.name.startswith('head'))
        if job.name == 'body+0':
            yaw = beauty.selected_yaw(self.references, angle)
            with Image.open(self.folder / self.references['views'][str(yaw)]['mask']) as mask:
                mask = mask.convert('L')
            return reference, {**options, 'recipe_name': 'RF18B', 'reference_record': record,
                               'surface_reference_mask': mask, 'complete_face': True}
        return reference, {**options, 'recipe_name': 'RF6B', 'reference_record': record,
                           'complete_face': job.name in ('head+0', 'body+0')}

    def choose(self, jobs, work):
        """Choose by declared view order after all frontal attempts have drained."""
        for key in ('head+0', 'body+0'):
            record = self.data['views'][key]
            if record['accepted'] and record.get('prompt') in ('RF6B', 'RF17B', 'RF18B'):
                selected = next(a for a in record['attempts'] if a.get('accepted'))
                features = selected.get('checks', {}).get('facial_features', {}).get('repaint', {})
                if features.get('both_eyes') and features.get('mouth'):
                    ordinary = self.default_work / key / 'refined.png' if self.default_work is not None else None
                    if ordinary is None or not ordinary.exists():
                        quality = {'accepted': False, 'reason': 'ordinary face paint unavailable', 'eyes': []}
                    else:
                        with Image.open(work / key / 'refined.png') as candidate, Image.open(ordinary) as default:
                            quality = eye_quality(candidate, default, features.get('eye_boxes', []))
                    selected.setdefault('checks', {})['eye_quality'] = quality
                    if not quality['accepted']:
                        continue
                    self.key = key
                    self.data['face_source'] = {'source': 'beauty', 'view': key,
                                                'attempt': record['selected_attempt']}
                    break
        head = next((job for job in jobs if job.name == 'head+0'), None)
        # The lowest accepted default slot (2, then conditional 3) is the fallback candidate.
        slot = None
        if head is not None:
            slot = next((i for i in range(1, len(head.tasks)) if head.tasks[i][2]['accepted']),
                        len(head.tasks) - 1)
        fallback = head.tasks[slot][2] if head is not None else None
        self.data['face_fallback'] = fallback or {'accepted': False, 'reason': 'head_not_found'}
        if self.key is None and fallback is not None and fallback['accepted']:
            # The default candidate remains an audit of the frontal sample. Composition
            # uses the complete ordinary-fast atlas, including the back of the head.
            self.key = 'head+0'
            self.data['beauty_head_candidate'] = dict(self.data['views'][self.key])
            target = work / self.key
            target.mkdir(parents=True, exist_ok=True)
            for source in head.tasks[slot][3].iterdir():
                if source.is_file():
                    shutil.copy2(source, target / source.name)
            with Image.open(target / 'face.png') as face:
                Image.new('L', face.size, 0).save(target / 'weight.png')
            self.data['views'][self.key] = {**fallback, 'head_only': True, 'selected_attempt': slot,
                                           'winning_slot': slot + 1,
                                           'attempts': head.record['attempts'], 'tries': head.record['tries']}
            self.data['face_source'] = {'source': 'default', 'view': self.key, 'attempt': slot - 1}
        if self.key is None:
            self.data['face_source'] = {'source': 'default-fast-fallback',
                                        'reason': 'no accepted frontal face repaint'}
            self.key = 'head+0' if head is not None else 'body+0'

    def default_fast(self, client, prep, tex, cams, work, timeout, cover, head_found, recipe, save, user_note):
        """Resolve the cover-only atlas from ordinary attempt zero per view and base residuals.

        Ordinary attempt 1 is paid later, per view, only by `extend_default` when the main
        beauty and default slots were both rejected.
        """
        from .fast import waves

        if self.default_work is not None:
            with self.state_lock:
                self.data['face_source'].update(view=self.key, atlas_sha256=hashlib.sha256(
                    (self.default_work / 'texture.png').read_bytes()).hexdigest(),
                    cap_reached=self.default_stopped)
            return self.default_stopped
        folder = work / 'default-fast' / 'blender-work' / 'texture-refine'
        folder.mkdir(parents=True, exist_ok=True)
        default = copy.copy(tex)
        with Image.open(self.default_base) as image:
            default.base = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
        rows, cols = np.divmod(default.idx, default.base.shape[1])
        default.base_cov = default.base[rows, cols].copy()
        default.reset()
        data = {'views': {}, 'code': self.data.get('code')}

        def checkpoint(**kwargs):
            # Outer attempts can checkpoint concurrently. Publish complete snapshots rather
            # than exposing the nested wave's dictionary while its pass list is initialized.
            with self.state_lock:
                self.data['default_fast_fallback'] = copy.deepcopy(data)
                return save(**kwargs)

        self.default_tex, self.default_data = default, data
        self.default_context = {'client': client, 'cams': cams}
        _, stopped = waves(client, prep, self.default_base, default, cams, folder, data,
                           timeout, cover, head_found, recipe, checkpoint, self.state_lock,
                           user_note=user_note, attempts=1, passes=self.default_passes)
        checkpoint(ok=False, status='running')
        # Match the exported default atlas, including its quantization and gutter extension.
        default.save(folder / 'texture.png')
        with Image.open(folder / 'texture.png') as image:
            self.fallback_texture = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
        self.data.setdefault('face_source', {})
        self.prepare_hair_views(client, folder, data['views'], tex.height)
        self.default_work, self.default_stopped = folder, stopped
        with self.state_lock:
            self.data.setdefault('face_source', {}).update(atlas_sha256=hashlib.sha256(
                (folder / 'texture.png').read_bytes()).hexdigest(), cap_reached=stopped)
        return stopped

    def extend_default(self, key):
        """Pay ordinary attempt 1 for one view through the default-head run's own Pass.

        It runs on the main conditional slot's worker before that slot's identical request,
        which then replays this cache entry. Both runs therefore pay one shared sample.
        """
        from .fast import publish, publish_files

        job = next(p for p in self.default_passes if p.key == key)
        with self.state_lock:
            record = self.default_data['views'][key]
            if record['accepted'] or record['tries'] != 1:
                raise ValueError(f'Default attempt 1 for {key} needs one rejected attempt 0')
        run, child, private, folder = job.prepare(1)
        try:
            run()
        finally:
            # Publish this family before the main slot's request reads the root cache.
            with self.persistence_lock:
                publish(self.default_context['client'], child)
                publish_files(self.root_client, child)
        with self.state_lock:
            self.extensions[key] = (private, folder)

    def fold_default(self, key):
        """Add a paid ordinary attempt 1 to the default-head run, as ordinary fast selects it.

        The default atlas is rebuilt from every accepted record, so it matches ordinary fast
        wherever this run holds the same accepted attempts.
        """
        from .fast import reproject

        with self.state_lock:
            if key not in self.extensions:
                return
            private, folder = self.extensions.pop(key)
            record = self.default_data['views'][key]
            record['attempts'] = record['attempts'] + private['attempts']
            record['tries'] = len(record['attempts'])
            if not private['accepted']:
                record['reason'] = private['reason']
                self.data['default_fast_fallback'] = copy.deepcopy(self.default_data)
                return
            record.update({k: v for k, v in private.items() if k not in ('attempts', 'tries')})
            record['selected_attempt'], record['winning_slot'] = 1, 2
            target = self.default_work / key
            target.mkdir(parents=True, exist_ok=True)
            for source in folder.iterdir():
                if source.is_file():
                    shutil.copy2(source, target / source.name)
            records = copy.deepcopy(self.default_data['views'])
        default = self.default_tex
        reproject(default, self.default_work, records, self.default_context['cams']['body+0'])
        default.save(self.default_work / 'texture.png')
        with Image.open(self.default_work / 'texture.png') as image:
            fallback = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
        self.prepare_hair_views(self.default_context['client'], self.default_work, records, default.height)
        self.fallback_texture = fallback
        with self.state_lock:
            self.data['default_fast_fallback'] = copy.deepcopy(self.default_data)
            self.data['face_source']['atlas_sha256'] = hashlib.sha256(
                (self.default_work / 'texture.png').read_bytes()).hexdigest()

    def prepare_hair_views(self, client, work, records, height):
        """Read the hair masks already required by accepted fast paints, with no API fallback."""
        replay = copy.copy(client)
        replay.mode = 'replay'
        self.hair_views = []
        for view in I.load_accepted_views(work, records, height):
            with Image.open(work / view['key'] / 'input.png') as image:
                mask = beauty.union(replay, image.convert('RGB'), 'hair')
            self.hair_views.append({**view, 'head_hair': mask})
        with self.state_lock:
            self.data['face_source']['cached_hair_views'] = [v['key'] for v in self.hair_views]

    def paint_default_head(self, tex, work):
        """Copy a closed head volume through every depth, then fade outside it at the collar.

        Setup's head segmentation can be a hair ring or contain missing face/scalp pixels.
        Close its small gaps and fill the cranial band through the face's chin. That
        volume has no visibility or normal test, so ears, scalp, rear head and UV island
        edges receive the same ungraded default atlas. Hanging hair below the chin uses
        the existing accepted paints' cached hair masks and mesh depths. With no
        accepted paint, a front mesh-depth band contains the setup silhouette.
        """
        from . import character as C
        from .cameras import camera

        boxes = self.source_info.get('boxes_px', {})
        box = boxes.get('head') or boxes.get('face')
        if box is None:
            with self.state_lock:
                self.data['face_source'].update(texels=0, head_region='not detected')
            return
        H, centre = self.source_info['height'], self.source_info['body_centre']
        probe = camera('probe0', centre, 0, H * C.BODY_FRAME, C.BODY_RES)
        w, h = C.BODY_RES
        left, top, right, bottom = box
        chin = boxes.get('face', box)[3]
        chin = max(top + 1, min(bottom, chin))
        width = max(1., .25 * (chin - top))
        mask_path = work / 'probe' / 'head.png'
        hair = np.zeros((h, w), bool)
        if mask_path.exists():
            with Image.open(mask_path) as image:
                if image.size != (w, h):
                    raise ValueError('Setup head mask dimensions differ from its probe')
                hair = np.asarray(image.convert('L')) > 127
            hair = ndimage.binary_closing(np.pad(hair, 3), iterations=3)[3:-3, 3:-3]
        # The mesh, rather than a skin segmentation, supplies the closed upper volume.
        # Pad its lateral bounds for ears and back-head silhouette differences.
        pad = .1 * (right - left)
        x0, x1 = max(0, int(np.floor(left - pad))), min(w, int(np.ceil(right + pad)))
        y0, y1 = max(0, int(np.floor(top - pad))), min(h, int(np.ceil(chin)))
        region = np.zeros((h, w), bool)
        region[y0:y1, x0:x1] = True
        # Semantic head masks can omit hats and helmets. Extend the closed volume to
        # its connected mesh silhouette above the chin; separate raised props stay out.
        probe_path = work / 'probe' / 'probe0.png'
        headgear_pixels = 0
        if probe_path.exists():
            with Image.open(probe_path) as image:
                if image.size != (w, h):
                    raise ValueError('Setup silhouette dimensions differ from its probe')
                silhouette = np.asarray(image.convert('RGBA').getchannel('A')) > 127
            silhouette[y1:] = False
            labels, count = ndimage.label(silhouette)
            if count:
                overlap = np.bincount(labels[region], minlength=count + 1)
                overlap[0] = 0
                if overlap.max():
                    connected = ndimage.binary_fill_holes(labels == int(overlap.argmax()))
                    headgear_pixels = int((connected & ~region).sum())
                    region |= connected
        if not region.any():
            raise ValueError('Detected head bounds are outside the setup probe')
        # Feather outward only: no beauty grade or feather correction enters the head.
        outside = ndimage.distance_transform_edt(~region)
        ramp = np.clip(1 - outside / width, 0, 1).astype(np.float32)
        blend = ramp * ramp * (3 - 2 * ramp)
        x, y, depth = project(probe, tex.P)
        alpha = I.sample(blend, x, y)
        # Exact ownership includes texels whose centres are in the closed region;
        # bilinear interpolation must not leak beauty at its silhouette boundary.
        xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        xc, yc = np.clip(xi, 0, w - 1), np.clip(yi, 0, h - 1)
        core = inside & region[yc, xc]
        alpha[core] = 1
        # Existing accepted paint already queried SAM hair in these exact renders.
        # Reuse those semantic masks and mesh depths, so jewellery inside a broad
        # setup head mask cannot become a patch of default clothing.
        if self.hair_views:
            for view in self.hair_views:
                hx, hy, hd = project(view['cam'], tex.P)
                hh, hw = view['head_hair'].shape
                hxi, hyi = np.floor(hx).astype(int), np.floor(hy).astype(int)
                hit = (hxi >= 0) & (hxi < hw) & (hyi >= 0) & (hyi < hh)
                hxi, hyi = np.clip(hxi, 0, hw - 1), np.clip(hyi, 0, hh - 1)
                front = ndimage.minimum_filter(np.where(view['valid'], view['depth'], np.inf),
                                               size=I.VIS_WINDOW)
                eps = 3 / px_per_unit(view['cam']) + .001 * tex.height
                hit &= (view['head_hair'][hyi, hxi] & view['valid'][hyi, hxi]
                        & (hd <= front[hyi, hxi] + eps))
                alpha[hit] = 1
        else:
            # Hanging hair can project over the torso. Its front surface owns a thin
            # mesh-depth band, not the unrelated body surface on the other side.
            hair[:y1] = False
            hair_core = inside & hair[yc, xc]
            if hair_core.any():
                front = np.full((h, w), np.inf)
                np.minimum.at(front, (yi[inside], xi[inside]), depth[inside])
                front = ndimage.minimum_filter(front, size=I.VIS_WINDOW)
                thickness = max(3 / px_per_unit(probe), .1 * (chin - top) / px_per_unit(probe))
                gap = np.maximum(0, depth - front[yc, xc])
                t = np.clip((2 * thickness - gap) / thickness, 0, 1)
                support = t * t * (3 - 2 * t)
                alpha[hair_core] = np.maximum(alpha[hair_core], support[hair_core])
        alpha[~inside] = 0
        core = alpha == 1
        rows, cols = np.divmod(tex.idx, tex.T.shape[1])
        selected = alpha > 0
        self.default_head_weight = np.zeros(tex.T.shape[:2], np.float32)
        self.default_head_weight[rows, cols] = alpha
        r, c = rows[selected], cols[selected]
        a = alpha[selected, None]
        tex.T[r, c] = tex.T[r, c] * (1 - a) + self.fallback_texture[r, c] * a
        with self.state_lock:
            self.data['face_source'].update(texels=int(core.sum()), head_region={
                'source': 'setup-head-mask-and-mesh', 'mask': str(mask_path.relative_to(work)),
                'mask_available': mask_path.exists(), 'chin_px': float(chin),
                'transition_px': float(width), 'transition_texels': int((selected & ~core).sum()),
                'grade_inside': False, 'all_depths': True,
                'headgear_silhouette': probe_path.exists(), 'headgear_pixels': headgear_pixels,
                'hanging_hair': 'cached-hair-masks' if self.hair_views else 'mesh-front-depth'})

    def prepare_region(self, client, work):
        """Include exposed scalp, ears and neck so a face cannot be pasted into graded skin.

        The existing front-probe head bounds supply the crop, with the same 25% neck context
        as beauty references. Skin is a paint mask, not an additional acceptance gate.
        """
        if self.data['face_source']['source'] != 'beauty':
            return
        folder = work / self.key
        with Image.open(folder / 'face.png') as image:
            face = np.asarray(image.convert('L')) > 127
        if not face.any():
            return
        image = Image.open(folder / 'input.png').convert('RGB')
        h, w = face.shape
        box = self.source_info.get('boxes_px', {}).get('head')
        cam = self.data['views'][self.key]['camera']
        if box is not None:
            H, c = self.source_info['height'], self.source_info['body_centre']
            left, top, right, bottom = box
            bottom += (bottom - top) * .25
            k = 1536 / (1.06 * H)
            points = np.array([[c[0] + (x - 512) / k, c[1], c[2] + (768 - y) / k]
                               for x, y in [(left, top), (right, bottom)]])
            x, y, _ = project(cam, points)
            roi = [int(np.floor(x.min())), int(np.floor(y.min())),
                   int(np.ceil(x.max())), int(np.ceil(y.max()))]
        elif cam['name'].startswith('head'):
            roi = [0, 0, w, h]
        else:
            y, x = np.nonzero(face)
            pad = max(x.max() - x.min() + 1, y.max() - y.min() + 1)
            roi = [int(x.min() - pad / 2), int(y.min() - pad),
                   int(x.max() + 1 + pad / 2), int(y.max() + 1 + pad / 4)]
        x0, y0, x1, y1 = roi = [max(0, roi[0]), max(0, roi[1]), min(w, roi[2]), min(h, roi[3])]
        if x1 <= x0 or y1 <= y0:
            raise ValueError('Frontal head skin crop is outside its render')
        skin = beauty.union(client, image.crop(roi), 'skin')
        region = face.copy()
        region[y0:y1, x0:x1] |= skin
        Image.fromarray(region.astype(np.uint8) * 255).save(folder / 'face-paint.png')
        with self.state_lock:
            self.data['face_source']['skin_region'] = {'crop': roi, 'skin_pixels': int(skin.sum()),
                                                       'mask': 'face-paint.png'}

    def paint(self, tex, work, records):
        """Preserve the chosen face interior and soften the surrounding head-skin boundary.

        A default source owns the complete head volume, including unpainted residuals.
        Beauty sources retain the accepted frontal skin composition.
        """
        if self.key is None:
            raise ValueError('Beauty face source must be resolved before publishing')
        if self.data['face_source']['source'] != 'beauty':
            if self.fallback_texture is None:
                raise ValueError('Default head atlas must be resolved before publishing')
            self.paint_default_head(tex, work)
            return
        view, = I.load_accepted_views(work, {self.key: records[self.key]}, tex.height)
        folder = work / self.key
        with Image.open(folder / 'face.png') as image:
            face = np.asarray(image.convert('L')) > 127
        if not face.any():
            # An accepted beauty helmet can have no exposed face.
            with self.state_lock:
                self.data['face_source'].update(texels=0, exposed=False)
            return
        region = face
        if (folder / 'face-paint.png').exists():
            with Image.open(folder / 'face-paint.png') as image:
                region = np.asarray(image.convert('L')) > 127
        x, y, d = project(view['cam'], tex.P)
        h, w = face.shape
        xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        xi, yi = np.clip(xi, 0, w - 1), np.clip(yi, 0, h - 1)
        front = ndimage.minimum_filter(np.where(view['valid'], view['depth'], np.inf), size=I.VIS_WINDOW)
        _, _, forward = axes(view['cam'])
        visible = (inside & region[yi, xi] & view['valid'][yi, xi]
                   & (d <= front[yi, xi] + 3 / px_per_unit(view['cam']) + .001 * tex.height)
                   & ((tex.N @ forward) < 0))
        indices = np.flatnonzero(visible)
        if not len(indices):
            raise ValueError('Accepted frontal repaint has no visible face texels')
        row, col = np.divmod(tex.idx[indices], tex.T.shape[1])
        colours = I.sample(view['refined'], x[indices], y[indices])
        width = max(1, round(12 * max(region.shape) / 1024))
        distance = ndimage.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
        t = np.clip((distance - .5) / width, 0, 1).astype(np.float32)
        blend = t * t * (3 - 2 * t)
        # Preserve the facial interior exactly; only the outer head-skin band is matched.
        core = ndimage.distance_transform_edt(np.pad(face, 1))[1:-1, 1:-1] > width
        blend[core] = 1
        current, valid = tex.render_view(view['cam'], view['depth'], view['valid'])
        boundary = region & valid & (distance <= width)
        if boundary.any():
            correction = I.masked_blur(current - view['refined'], boundary, width)
            adjusted = np.clip(view['refined'] + correction * (1 - blend[..., None]), 0, 1)
            colours = I.sample(adjusted, x[indices], y[indices])
        alpha = I.sample(blend, x[indices], y[indices])[:, None]
        tex.T[row, col] = tex.T[row, col] * (1 - alpha) + colours * alpha
        with self.state_lock:
            self.data['face_source']['texels'] = int(len(indices))
