"""Suggested sketches come only from a configured folder, are hash-checked, and cannot reach outside it."""
from pathlib import Path
import hashlib
import json
import sys
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parent)]
from test_herogen_service_web import WebCase, Browser
from test_herogen_service_support import HAVE_WEB, TempDirCase, png, web_test
from herogen.service import config as service_config
from herogen.service.sketch_ideas import SketchIdeasUnavailable, catalog, image
if HAVE_WEB:
    from herogen.service.web import create_app


def publish(root: Path, names=('iron-warden', 'space-cadet')) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    items = []
    for i, key in enumerate(names):
        content = png(300 + i, 500)
        (root / f'{key}.png').write_bytes(content)
        items.append({'id': key, 'title': key.replace('-', ' ').title(), 'file': f'{key}.png',
                      'sha256': hashlib.sha256(content).hexdigest(), 'job': f'job_{i}abc'})
    data = {'version': 1, 'items': items}
    save(root, data)
    return data


def save(root: Path, data: dict) -> None:
    (root / 'catalog.json').write_text(json.dumps(data))


class SketchIdeasLoaderTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / 'sketch-ideas'
        self.data = publish(self.root)

    def test_lists_public_fields_only(self):
        items = catalog(self.root)
        self.assertEqual([i['id'] for i in items], ['iron-warden', 'space-cadet'])
        sha = self.data['items'][0]['sha256']
        self.assertEqual(items[0], {'id': 'iron-warden', 'title': 'Iron Warden',
                                    'url': f'/sketch-ideas/iron-warden?v={sha[:16]}'})
        self.assertNotIn('job_0abc', json.dumps(items))
        path, got = image(self.root, 'iron-warden')
        self.assertEqual((path, got), (self.root / 'iron-warden.png', sha))

    def test_sha_mismatch_is_refused(self):
        (self.root / 'iron-warden.png').write_bytes(png(320, 500))
        with self.assertRaises(SketchIdeasUnavailable):
            image(self.root, 'iron-warden')
        image(self.root, 'space-cadet')   # the others are unaffected

    def test_not_a_png_is_refused_even_with_a_matching_hash(self):
        body = b'GIF89a not a png'
        (self.root / 'iron-warden.png').write_bytes(body)
        self.data['items'][0]['sha256'] = hashlib.sha256(body).hexdigest()
        save(self.root, self.data)
        with self.assertRaises(SketchIdeasUnavailable):
            image(self.root, 'iron-warden')

    def test_missing_folder_hides_the_feature(self):
        self.assertEqual(catalog(None), [])
        with self.assertRaises(SketchIdeasUnavailable):
            image(None, 'iron-warden')
        with self.assertRaises(SketchIdeasUnavailable):
            catalog(self.tmp / 'nowhere')

    def test_paths_cannot_escape_the_folder(self):
        secret = self.tmp / 'secret.png'
        secret.write_bytes(png())
        for name in ('../secret.png', '/etc/passwd', 'sub/iron-warden.png', 'iron-warden.PNG', '..png', 'x.glb'):
            self.data['items'][0]['file'] = name
            save(self.root, self.data)
            with self.assertRaises(SketchIdeasUnavailable, msg=name):
                catalog(self.root)
        # A link inside the folder that points outside it is refused too.
        (self.root / 'iron-warden.png').unlink()
        (self.root / 'iron-warden.png').symlink_to(secret)
        self.data['items'][0].update(file='iron-warden.png', sha256=hashlib.sha256(secret.read_bytes()).hexdigest())
        save(self.root, self.data)
        with self.assertRaises(SketchIdeasUnavailable):
            image(self.root, 'iron-warden')
        for key in ('../secret', 'Iron-Warden', '', 'a/b'):
            with self.assertRaises(SketchIdeasUnavailable, msg=key):
                image(self.root, key)

    def test_bad_catalogs_are_refused(self):
        for change in ({'version': 2}, {'items': {}}, {'items': [self.data['items'][0]] * 2},
                       {'items': [{**self.data['items'][0], 'sha256': 'abc'}]},
                       {'items': [{**self.data['items'][0], 'title': ''}]},
                       {'items': [{**self.data['items'][0], 'job': '../x'}]}):
            save(self.root, {**self.data, **change})
            with self.assertRaises(SketchIdeasUnavailable, msg=change):
                catalog(self.root)

    def test_setting_reads_a_folder(self):
        s = service_config.from_env({'HEROGEN_SERVICE_SKETCH_IDEAS': str(self.root)}, require_key=False)
        self.assertEqual(s.sketch_ideas_dir, self.root)
        self.assertIsNone(service_config.from_env({}, require_key=False).sketch_ideas_dir)


@web_test
class SketchIdeasRoutesTest(WebCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / 'sketch-ideas'
        self.data = publish(self.root)

    def use(self, root):
        self.svc.settings = self.s.with_(sketch_ideas_dir=root)
        self.app = create_app(self.svc)

    def test_unset_lists_nothing_and_serves_nothing(self):
        b = Browser(self.app)
        r = b.get('/api/sketch-ideas')
        self.assertEqual((r.status_code, r.json()), (200, {'items': []}))
        r = b.get('/sketch-ideas/iron-warden')
        self.assertEqual(r.status_code, 404)
        self.assertNotIn('text/html', r.headers['content-type'])
        self.assertIn('id="suggestSketch"', b.get('/').text)

    def test_lists_and_serves_verified_pngs_without_a_session(self):
        self.use(self.root)
        b = Browser(self.app)
        items = b.get('/api/sketch-ideas').json()['items']
        self.assertEqual(len(items), 2)
        self.assertNotIn(str(self.tmp), json.dumps(items))
        r = b.get(items[0]['url'])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers['content-type'], 'image/png')
        self.assertIn('immutable', r.headers['cache-control'])
        self.assertEqual(r.content, (self.root / 'iron-warden.png').read_bytes())
        self.assertEqual(b.get('/sketch-ideas/unknown').status_code, 404)
        self.assertEqual(self.submitted, [])
        self.assertEqual(self.store.users_list(), [])

    def test_mismatch_is_not_served_and_a_broken_catalog_is_503(self):
        self.use(self.root)
        (self.root / 'iron-warden.png').write_bytes(png(333, 500))
        b = Browser(self.app)
        self.assertEqual(b.get('/sketch-ideas/iron-warden').status_code, 404)
        self.assertEqual(b.get('/sketch-ideas/space-cadet').status_code, 200)
        (self.root / 'catalog.json').write_text('{')
        r = b.get('/api/sketch-ideas')
        self.assertEqual((r.status_code, r.json()['error']['code']), (503, 'sketch_ideas_unavailable'))

    def test_escaping_route_ids_are_refused(self):
        self.use(self.root)
        (self.tmp / 'secret.png').write_bytes(png())
        b = Browser(self.app)
        for path in ('/sketch-ideas/..%2Fsecret', '/sketch-ideas/catalog.json', '/sketch-ideas/iron-warden.png'):
            self.assertEqual(b.get(path).status_code, 404, path)


if __name__ == '__main__':
    unittest.main()
