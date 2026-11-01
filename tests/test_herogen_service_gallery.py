"""Public gallery exposes only separately published, integrity-checked artifacts."""
from pathlib import Path
import hashlib
import json
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parent)]
from test_herogen_service_web import WebCase, Browser
from test_herogen_service_support import HAVE_WEB, png, web_test
from herogen.service.gallery import artifact, catalog, GalleryUnavailable
from herogen.service.fakes import box_glb
if HAVE_WEB:
    from herogen.service.web import create_app


@web_test
class PublicGalleryTest(WebCase):
    def setUp(self):
        super().setUp()
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
