"""Public gallery exposes only separately published, integrity-checked artifacts."""
from pathlib import Path
import hashlib
import json
import sys
import unittest
from unittest import mock

sys.path[:0] = [str(Path(__file__).resolve().parent)]
from test_herogen_service_web import WebCase, Browser
from test_herogen_service_support import HAVE_WEB, png, web_test
from herogen.service.gallery import artifact, catalog, GalleryUnavailable
from herogen.service.fakes import box_glb
if HAVE_WEB:
    from herogen.service.web import create_app


class GalleryFixture:
    def publish_gallery(self):
        self.root = self.tmp / 'public-gallery'
        folder = self.root / 'hero'
        folder.mkdir(parents=True)
        assets = {}
        for kind, content, suffix in [('sketch', png(), '.png'), ('artwork', png(), '.png'), ('model', box_glb(), '.glb')]:
            name = f'hero/{kind}{suffix}'
            (self.root / name).write_bytes(content)
            assets[kind] = {'file': name, 'sha256': hashlib.sha256(content).hexdigest()}
        self.data = {'version': 1, 'items': [{'id': 'hero', 'title': 'Original hero', 'description': 'Saved result.',
                                            'code_commit': 'a' * 40, 'assets': assets}]}
        self.save()
        self.svc.settings = self.s.with_(gallery_dir=self.root)
        self.app = create_app(self.svc)

    def save(self):
        (self.root / 'catalog.json').write_text(json.dumps(self.data))


@web_test
class PublicGalleryTest(GalleryFixture, WebCase):
    def setUp(self):
        super().setUp()
        self.publish_gallery()

    def test_anonymous_inspection_does_not_grant_private_job_access(self):
        b = Browser(self.app)
        r = b.get('/api/gallery')
        self.assertEqual(r.status_code, 200)
        item = r.json()['items'][0]
        self.assertEqual(item['label'], 'Previously generated with HeroGen')
        self.assertNotIn(str(self.tmp), r.text)
        self.assertEqual(b.get('/api/jobs').status_code, 401)
        self.assertEqual(b.get('/api/workspace').status_code, 401)
        self.assertEqual(self.submitted, [])
        self.assertEqual(self.store.users_list(), [])
        self.assertIn('/assets/gallery-viewer.js', b.get('/gallery/hero').text)
        self.assertEqual(b.get('/api/gallery/unknown').status_code, 404)

    def test_models_stream_with_ranges_and_correct_type(self):
        b = Browser(self.app)
        r = b.get('/gallery-assets/hero/model')
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content, box_glb())
        self.assertEqual(r.headers['content-type'], 'model/gltf-binary')
        part = b.c.get('/gallery-assets/hero/model', headers={'Range': 'bytes=0-3'})
        self.assertEqual(part.status_code, 206)
        self.assertEqual(part.content, b'glTF')
        self.assertEqual(b.get('/gallery-assets/hero/stage-3').status_code, 404)

    def test_corrupt_and_linked_assets_are_not_published(self):
        (self.root / 'hero/model.glb').write_bytes(b'corrupt')
        self.assertEqual(Browser(self.app).get('/gallery-assets/hero/model').status_code, 404)
        target = self.root / 'hero/sketch.png'
        target.unlink()
        target.symlink_to(self.root / 'hero/artwork.png')
        with self.assertRaises(GalleryUnavailable):
            catalog(self.root)
        self.assertEqual(Browser(self.app).get('/api/gallery').status_code, 503)

    def test_manifest_cannot_escape_its_published_catalog(self):
        self.data['items'][0]['assets']['model']['file'] = '../private.glb'
        self.save()
        with self.assertRaises(GalleryUnavailable):
            artifact(self.root, 'hero', 'model')
        self.assertEqual(catalog(None), [])


@web_test
class GalleryThumbnailTest(GalleryFixture, WebCase):
    def setUp(self):
        super().setUp()
        self.publish_gallery()
        # Full-size published images: a wide sketch and a tall artwork with transparency.
        self.publish('sketch', png(1400, 900))
        self.publish('artwork', png(800, 1600, mode='RGBA'))
        self.cache = self.s.data_dir / 'gallery-thumbs' / 'hero'

    def publish(self, kind, content):
        (self.root / f'hero/{kind}.png').write_bytes(content)
        self.data['items'][0]['assets'][kind]['sha256'] = hashlib.sha256(content).hexdigest()
        self.save()

    def urls(self):
        return Browser(self.app).get('/api/gallery').json()['items'][0]['assets']

    def test_catalog_lists_versioned_thumbs_next_to_each_image(self):
        assets = self.urls()
        for kind in ('sketch', 'artwork'):
            sha = self.data['items'][0]['assets'][kind]['sha256']
            self.assertEqual(assets[kind]['url'], f'/gallery-assets/hero/{kind}')
            self.assertEqual(assets[kind]['thumb_url'], f'/gallery-assets/hero/{kind}-thumb?v={sha[:16]}')
        self.assertNotIn('thumb_url', assets['model'])
        self.assertEqual(Browser(self.app).get('/api/gallery/hero').json()['assets'], assets)

    def test_thumb_is_a_small_webp_made_once_then_cached(self):
        import io
        from PIL import Image
        from herogen.service import gallery
        b = Browser(self.app)
        for kind, size, mode in (('sketch', (512, 329), 'RGB'), ('artwork', (256, 512), 'RGBA')):
            url = self.urls()[kind]['thumb_url']
            with mock.patch.object(gallery, 'thumbnail', wraps=gallery.thumbnail) as made:
                r = b.get(url)
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.headers['content-type'], 'image/webp')
            self.assertEqual(r.headers['cache-control'], 'public, max-age=31536000, immutable')
            image = Image.open(io.BytesIO(r.content))
            self.assertEqual((image.format, image.size, image.mode), ('WEBP', size, mode))
            self.assertLess(len(r.content), (self.root / f'hero/{kind}.png').stat().st_size)
            self.assertEqual(made.call_count, 1)
            files = sorted(p.name for p in self.cache.iterdir())
            self.assertIn(f'{kind}-{url.split("v=")[1]}.webp', files)
            stamp = (self.cache / f'{kind}-{url.split("v=")[1]}.webp').stat().st_mtime_ns
            with mock.patch('PIL.Image.open', side_effect=AssertionError('made again')):
                again = b.get(url)
            self.assertEqual((again.status_code, again.content), (200, r.content))
            self.assertEqual((self.cache / f'{kind}-{url.split("v=")[1]}.webp').stat().st_mtime_ns, stamp)
        self.assertEqual(sorted(p.name for p in (self.root / 'hero').iterdir()),
                         ['artwork.png', 'model.glb', 'sketch.png'], 'nothing is written into the gallery')
        self.assertFalse(any(p.suffix == '.part' for p in self.cache.iterdir()))

    def test_read_only_gallery_still_gets_thumbs(self):
        import stat
        folder = self.root / 'hero'
        for path in (*folder.iterdir(), folder, self.root):
            path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        try:
            r = Browser(self.app).get(self.urls()['artwork']['thumb_url'])
            self.assertEqual((r.status_code, r.headers['content-type']), (200, 'image/webp'))
        finally:
            for path in (self.root, folder, *folder.iterdir()):
                path.chmod(path.stat().st_mode | stat.S_IWUSR)

    def test_republished_image_gets_a_new_thumb_and_the_old_one_goes(self):
        b = Browser(self.app)
        old = self.urls()['sketch']['thumb_url']
        first = b.get(old).content
        self.publish('sketch', png(900, 600, colour=(200, 0, 0)))
        new = self.urls()['sketch']['thumb_url']
        self.assertNotEqual(new, old)
        self.assertNotEqual(b.get(new).content, first)
        self.assertEqual([p.name for p in self.cache.glob('sketch-*.webp')], [f'sketch-{new.split("v=")[1]}.webp'])
        stale = b.get(old)
        self.assertEqual((stale.status_code, stale.headers['cache-control']), (200, 'public, max-age=3600'))

    def test_failed_write_leaves_nothing_to_serve(self):
        def broken(image, stream, **kw):
            stream.write(b'RIFF half')
            raise OSError('disk full')
        with mock.patch('PIL.Image.Image.save', broken):
            r = Browser(self.app).get(self.urls()['artwork']['thumb_url'])
        self.assertEqual(r.status_code, 404)
        self.assertEqual(list(self.cache.iterdir()), [])
        self.assertEqual(Browser(self.app).get(self.urls()['artwork']['thumb_url']).status_code, 200)

    def test_only_published_images_have_thumbs(self):
        b = Browser(self.app)
        for path in ('/gallery-assets/hero/model-thumb', '/gallery-assets/hero/stage-3-thumb',
                     '/gallery-assets/unknown/artwork-thumb', '/gallery-assets/hero/-thumb',
                     '/gallery-assets/hero/artwork-thumb-thumb'):
            self.assertEqual(b.get(path).status_code, 404, path)
        (self.root / 'hero/artwork.png').write_bytes(png(10, 10))
        self.assertEqual(b.get('/gallery-assets/hero/artwork-thumb').status_code, 404, 'integrity is checked')
