"""Landing page presentation in a real browser, with no model or GPU calls."""
import time
import unittest
from tests.test_herogen_service_e2e import ServiceBrowserCase, CDP_PORT, HAVE_WEB, uvicorn


@unittest.skipUnless(HAVE_WEB and uvicorn is not None and CDP_PORT, 'set E2E_CDP_PORT')
class TestLandingPage(ServiceBrowserCase):
    service_settings = {'public_demo': True, 'code_commit': 'a' * 40}

    def ready(self):
        self.tab.goto(self.url + '/')
        self.tab.wait("document.querySelector('[data-theme-toggle]')?.dataset.preference && !document.querySelector('#entry').hidden")

    def choose(self, value):
        # The header toggle cycles dark -> light -> system; click it until it shows the wanted preference.
        for _ in range(3):
            if self.tab.eval("document.querySelector('[data-theme-toggle]').dataset.preference") == value:
                return
            self.tab.eval("document.querySelector('[data-theme-toggle]').click()")
        self.assertEqual(self.tab.eval("document.querySelector('[data-theme-toggle]').dataset.preference"), value)

    def test_first_visit_is_dark_even_when_the_os_is_light(self):
        t = self.tab
        t.call('Emulation.setEmulatedMedia', features=[{'name': 'prefers-color-scheme', 'value': 'light'}])
        self.ready()
        self.assertEqual(t.eval('document.documentElement.dataset.theme'), 'dark')
        self.assertEqual(t.eval("document.querySelector('[data-theme-toggle]').getAttribute('aria-label')"),
                         'Theme: Dark. Switch to light.')
        self.assertEqual(self.tab.errors, [])

    def test_theme_persists_and_system_choice_tracks_os(self):
        t = self.tab
        t.call('Emulation.setEmulatedMedia', features=[{'name': 'prefers-color-scheme', 'value': 'dark'}])
        self.ready()
        self.choose('system')
        self.assertEqual(t.eval('document.documentElement.dataset.theme'), 'dark')
        self.choose('light')
        self.assertEqual(t.eval('document.documentElement.dataset.theme'), 'light')
        t.goto(self.url + '/')
        t.wait("document.querySelector('[data-theme-toggle]')?.dataset.preference === 'light'")
        self.assertEqual(t.eval('document.documentElement.dataset.theme'), 'light')
        self.choose('dark')
        self.assertEqual(t.eval("getComputedStyle(document.body).backgroundColor"), 'rgb(11, 14, 18)')
        self.choose('system')
        t.call('Emulation.setEmulatedMedia', features=[{'name': 'prefers-color-scheme', 'value': 'light'}])
        t.wait("document.documentElement.dataset.theme === 'light'")
        self.assertEqual(self.store.users_list(), [])
        self.assertEqual(self.tab.errors, [])

    def test_theme_remains_usable_when_storage_is_blocked(self):
        t = self.tab
        t.call('Page.addScriptToEvaluateOnNewDocument', source="Object.defineProperty(window,'localStorage',{get(){throw new Error('storage blocked fixture')}})")
        self.ready()
        self.choose('dark')
        self.assertEqual(t.eval('document.documentElement.dataset.theme'), 'dark')
        self.choose('light')
        self.assertEqual(t.eval('document.documentElement.dataset.theme'), 'light')
        self.assertEqual(self.tab.errors, [])

    def test_mobile_orders_steps_before_sketchbook_without_overflow(self):
        t = self.tab
        t.call('Emulation.setDeviceMetricsOverride', width=390, height=844, deviceScaleFactor=1, mobile=True)
        self.ready()
        self.assertTrue(t.eval("document.querySelector('.sketchbook').getBoundingClientRect().top > document.querySelector('.journey').getBoundingClientRect().bottom"))
        self.assertTrue(t.eval('document.documentElement.scrollWidth <= innerWidth'))
        self.assertTrue(t.eval("document.querySelector('#landingCanvas').getBoundingClientRect().top > document.querySelector('#jobForm').getBoundingClientRect().bottom"))
        self.assertTrue(t.eval("document.querySelector('#headerSignIn').hidden"))
        self.assertEqual(self.tab.errors, [])

    def test_empty_gallery_has_no_fake_models_or_account_creation(self):
        self.ready()
        self.tab.wait("document.querySelector('#landingStatus').textContent.includes('No saved models')")
        self.assertEqual(self.tab.eval("document.querySelector('#modelPicker').children.length"), 0)
        self.assertEqual(self.store.users_list(), [])
        self.assertEqual(self.store.jobs_admin_list(), [])

    def test_shading_icons_render_when_studio_static_is_absent(self):
        from unittest.mock import patch

        with patch('herogen.service.labapi.studio_static', return_value=None):
            self.ready()
            self.tab.wait("Array.from(document.querySelectorAll('.shade-switch img')).every(i => i.complete && i.naturalWidth > 0)")
            imgs = self.tab.eval("[...document.querySelectorAll('.shade-switch img')].map(i => ({src: i.getAttribute('src'), complete: i.complete, naturalWidth: i.naturalWidth}))")
            self.assertEqual(len(imgs), 3)
            for img in imgs:
                self.assertTrue(img['src'].startswith('/assets/icons/hg-shade-'), img)
                self.assertGreater(img['naturalWidth'], 0, img)
            self.assertEqual(self.tab.errors, [])

@unittest.skipUnless(HAVE_WEB and uvicorn is not None and CDP_PORT, 'set E2E_CDP_PORT')
class TestLandingModels(ServiceBrowserCase):
    def setUp(self):
        import hashlib
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from herogen.service.fakes import box_glb
        from test_herogen_service_support import png
        self.gallery_tmp = TemporaryDirectory()
        root = Path(self.gallery_tmp.name)
        items = []
        for name in ['woodland-scout', 'second-model']:
            folder = root / name
            folder.mkdir()
            assets = {}
            for kind, data, suffix in [('model', box_glb(), 'glb'), ('artwork', png(), 'png'), ('sketch', png(), 'png')]:
                path = f'{name}/{kind}.{suffix}'
                (root / path).write_bytes(data)
                assets[kind] = {'file': path, 'sha256': hashlib.sha256(data).hexdigest()}
            items.append({'id': name, 'title': name, 'description': 'Synthetic fixture.', 'code_commit': 'a' * 40, 'assets': assets})
        (root / 'catalog.json').write_text(json.dumps({'version': 1, 'items': items}))
        self.service_settings = {'public_demo': True, 'code_commit': 'a' * 40, 'gallery_dir': root}
        super().setUp()
        self.addCleanup(self.gallery_tmp.cleanup)

    def test_real_mesh_picker_shading_and_gallery_theme_without_guest_access(self):
        t = self.tab
        t.goto(self.url + '/')
        t.wait("document.querySelector('#landingStatus').textContent === 'Drag to rotate · Scroll to zoom'")
        t.wait("document.querySelectorAll('.model-thumb img').length === 2")
        for mode in ['clay', 'wire', 'textured']:
            t.eval(f"document.querySelector('[data-shade={mode}]').click()")
            self.assertTrue(t.eval(f"document.querySelector('[data-shade={mode}]').getAttribute('aria-pressed') === 'true'"))
        t.eval("document.querySelectorAll('#modelPicker button')[1].click();document.querySelectorAll('#modelPicker button')[0].click()")
        t.wait("document.querySelector('#featuredTitle').textContent === 'woodland-scout' && document.querySelector('#landingStatus').textContent === 'Drag to rotate · Scroll to zoom'")
        self.assertEqual(self.store.users_list(), [])
        self.assertEqual(self.store.jobs_admin_list(), [])
        t.eval("window.herogenTheme.choose('dark')")
        t.goto(self.url + '/gallery/woodland-scout')
        t.wait("document.querySelector('#publicModelStatus').textContent.includes('Drag to orbit')")
        self.assertEqual(t.eval('document.documentElement.dataset.theme'), 'dark')
        self.assertEqual(self.tab.errors, [])

    def test_model_failure_is_retryable(self):
        t = self.tab
        t.call('Page.addScriptToEvaluateOnNewDocument', source="const originalFetch=window.fetch; window.fetch=(url,options)=>(url instanceof Request?url.url:String(url)).includes('/model')&&!window.allowModels?Promise.reject(new Error('offline fixture')):originalFetch(url,options);")
        t.goto(self.url + '/')
        t.wait("!document.querySelector('#retryLanding').hidden && document.querySelector('#landingStatus').textContent.includes('could not be loaded')")
        self.assertEqual(t.eval("document.querySelectorAll('.model-thumb img').length"), 0)
        t.eval("window.allowModels=true;document.querySelector('#retryLanding').click()")
        t.wait("document.querySelector('#landingStatus').textContent === 'Drag to rotate · Scroll to zoom' && document.querySelectorAll('.model-thumb img').length === 2 && document.querySelector('.thumbnail-render') === null")
        self.assertEqual(self.store.jobs_admin_list(), [])
        self.assertEqual(self.tab.errors, [])

    def test_failed_model_switch_clears_previous_mesh_and_recovers_on_retry(self):
        t = self.tab
        t.call('Emulation.setDeviceMetricsOverride', width=1440, height=960, deviceScaleFactor=1, mobile=False)
        t.goto(self.url + '/')
        t.wait("document.querySelector('#landingStatus').textContent === 'Drag to rotate · Scroll to zoom' && document.querySelectorAll('.model-thumb img').length === 2")
        box = t.eval("document.querySelector('#landingCanvas').getBoundingClientRect().toJSON()")
        clip = {'x': box['x'] + 150, 'y': box['y'] + 60, 'width': 260, 'height': 320, 'scale': 1}
        scout_shot = t.call('Page.captureScreenshot', format='png', clip=clip)['data']
        t.eval("""(() => {
            const orig = window.fetch;
            window.fetch = (u, o) => (u instanceof Request ? u.url : String(u)).includes('/second-model/model') && !window.allowSecondModel
                ? Promise.reject(new Error('offline fixture'))
                : orig(u, o);
        })()""")
        t.eval("document.querySelector('#modelPicker button[data-id=second-model]').click()")
        t.wait("!document.querySelector('#retryLanding').hidden && document.querySelector('#landingStatus').textContent.includes('could not be loaded')")
        after_shot = t.call('Page.captureScreenshot', format='png', clip=clip)['data']
        self.assertEqual(t.eval("document.querySelector('#featuredTitle').textContent"), 'second-model')
        self.assertNotEqual(after_shot, scout_shot)
        t.eval("window.allowSecondModel = true; document.querySelector('#retryLanding').click()")
        t.wait("document.querySelector('#landingStatus').textContent === 'Drag to rotate · Scroll to zoom'")
        self.assertEqual(t.errors, [])
