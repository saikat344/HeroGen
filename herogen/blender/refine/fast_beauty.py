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


class FacePass(Pass):
    """The head's two slots are independent beauty and default samples, never retries."""

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
        # The RF2R candidate uses the original frozen base and artwork. Its
        # sample index matches default fast's first head attempt for exact cache reuse.
        options = {'face_keep': True, 'save_face': True}
        if job.name == 'head+0' and index == 1:
            return cover, {**options, 'recipe_name': 'RF2R', 'attempt_start': 0, 'rendered': self.default_render}
        angle = {'head+0': 0, 'body+0': 0, 'body+90': 90, 'body-90': -90,
                 'body+180': 180, 'upper+180': 180}[job.name]
        if beauty.selected_yaw(self.references, angle) is None:
            job.record['note'] = f'no face-free beauty reference at {angle:+g} yaw; standard recipe'
            return cover, options
        reference, record = beauty.reference(self.folder, self.references, angle, head=job.name.startswith('head'))
        return reference, {**options, 'recipe_name': 'RF6B', 'reference_record': record,
                           'complete_face': job.name in ('head+0', 'body+0')}

    def choose(self, jobs, work):
        """Choose by declared view order after all frontal attempts have drained."""
        for key in ('head+0', 'body+0'):
            record = self.data['views'][key]
            if record['accepted'] and record.get('prompt') == 'RF6B':
                selected = next(a for a in record['attempts'] if a.get('accepted'))
                features = selected.get('checks', {}).get('facial_features', {}).get('repaint', {})
                if features.get('both_eyes') and features.get('mouth'):
                    self.key = key
                    self.data['face_source'] = {'source': 'beauty', 'view': key,
                                                'attempt': record['selected_attempt']}
                    break
        head = next((job for job in jobs if job.name == 'head+0'), None)
        fallback = head.tasks[1][2] if head is not None else None
        self.data['face_fallback'] = fallback or {'accepted': False, 'reason': 'head_not_found'}
        if self.key is None and fallback is not None and fallback['accepted']:
            # The default candidate remains an audit of the frontal sample. Composition
            # uses the complete ordinary-fast atlas, including the back of the head.
            self.key = 'head+0'
            self.data['beauty_head_candidate'] = dict(self.data['views'][self.key])
            target = work / self.key
            target.mkdir(parents=True, exist_ok=True)
            for source in head.tasks[1][3].iterdir():
                if source.is_file():
                    shutil.copy2(source, target / source.name)
            with Image.open(target / 'face.png') as face:
                Image.new('L', face.size, 0).save(target / 'weight.png')
            self.data['views'][self.key] = {**fallback, 'head_only': True, 'selected_attempt': 1,
                                           'attempts': head.record['attempts'], 'tries': head.record['tries']}
            self.data['face_source'] = {'source': 'default', 'view': self.key, 'attempt': 0}
        if self.key is None:
            self.data['face_source'] = {'source': 'default-fast-fallback',
                                        'reason': 'no accepted frontal face repaint'}

    def default_fast(self, client, prep, tex, cams, work, timeout, cover, head_found, recipe, save, user_note):
        """Resolve the ordinary fast atlas, including accepted side/body paint and base residuals."""
        from .fast import waves

        if self.data['face_source']['source'] == 'beauty':
            return
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

        _, stopped = waves(client, prep, self.default_base, default, cams, folder, data,
                           timeout, cover, head_found, recipe, checkpoint, self.state_lock, user_note=user_note)
        checkpoint(ok=False, status='running')
        # Match the exported default atlas, including its quantization and gutter extension.
        default.save(folder / 'texture.png')
        with Image.open(folder / 'texture.png') as image:
            self.fallback_texture = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
        self.prepare_hair_views(client, folder, data['views'], tex.height)
        self.key = self.key or ('head+0' if head_found else 'body+0')
        with self.state_lock:
            self.data['face_source'].update(view=self.key, atlas_sha256=hashlib.sha256(
                (folder / 'texture.png').read_bytes()).hexdigest(), cap_reached=stopped)
        return stopped

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
                'grade_inside': False, 'all_depths': True, 'hanging_hair': 'cached-hair-masks' if self.hair_views else 'mesh-front-depth'})

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
