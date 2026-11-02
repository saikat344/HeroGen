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

    def test_mobile_layout_has_no_overflow_or_missing_art(self):
        t = self.tab
        for width in (360, 390):
            t.call('Emulation.setDeviceMetricsOverride', width=width, height=844, deviceScaleFactor=1, mobile=True)
            self.ready()
            self.assertTrue(t.eval('document.documentElement.scrollWidth <= innerWidth'), width)
        self.assertIsNone(t.eval("document.querySelector('.sketchbook, img[src*=\"/assets/art/\"]')"))
        self.assertEqual(t.eval("document.querySelectorAll('.journey > .step').length"), 4)
        self.assertTrue(t.eval("document.querySelector('#landingCanvas').getBoundingClientRect().top > document.querySelector('#jobForm').getBoundingClientRect().bottom"))
        self.assertTrue(t.eval("document.querySelector('#headerSignIn').hidden"))
        self.assertEqual(self.tab.errors, [])

    def test_empty_gallery_has_no_fake_models_or_account_creation(self):
        self.ready()
        self.tab.wait("document.querySelector('#landingStatus').textContent.includes('No saved models')")
        self.assertEqual(self.tab.eval("document.querySelector('#modelPicker').children.length"), 0)
        self.assertEqual(self.store.users_list(), [])
        self.assertEqual(self.store.jobs_admin_list(), [])

    def test_shading_icons_come_from_the_site_sprite_when_studio_static_is_absent(self):
        from unittest.mock import patch

        with patch('herogen.service.labapi.studio_static', return_value=None):
            self.ready()
            hrefs = self.tab.eval("[...document.querySelectorAll('[data-shade] use')].map(u => u.getAttribute('href'))")
            self.assertEqual(hrefs, [f'/assets/icons/sprite.svg#i-hg-shade-{m}' for m in ('textured', 'clay', 'wire')])
            self.assertEqual(self.tab.eval("fetch('/assets/icons/sprite.svg').then(r => r.status)"), 200)
            self.assertEqual(self.tab.errors, [])

    def test_segmented_control_drives_the_form_kind_and_drop_zone(self):
        t = self.tab
        self.ready()
        kind = "document.querySelector('input[name=kind]:checked').value"
        self.assertEqual(t.eval(kind), 'prompt')
        self.assertTrue(t.eval("document.querySelector('#dropZone').hidden"))
        t.eval("document.querySelector('[data-start=sketch]').click()")
        self.assertEqual(t.eval(kind), 'sketch')
        self.assertFalse(t.eval("document.querySelector('#dropZone').hidden"))
        self.assertTrue(t.eval("document.querySelector('#promptIn').hidden"))
        self.assertEqual(t.eval("document.activeElement.id"), 'browseUpload')
        self.assertIn('sketch', t.eval("document.querySelector('#dropTitle').textContent"))
        # Arrow keys move along the group like a radio group.
        t.eval("document.querySelector('[data-start=sketch]').focus(); document.querySelector('.composer__modes').dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true}))")
        self.assertEqual(t.eval("document.querySelector('[data-start=upload]').getAttribute('aria-pressed')"), 'true')
        self.assertIn('artwork', t.eval("document.querySelector('#dropTitle').textContent"))
        # Submitting with no image is an inline error on the composer, not a dialog or a request.
        t.eval("document.querySelector('#jobForm').requestSubmit()")
        self.assertEqual(t.eval("document.querySelector('#jobMsg').textContent"), 'Choose an image to upload.')
        self.assertEqual(t.eval("document.querySelector('#jobMsg').dataset.tone"), 'error')
        self.assertFalse(t.eval("document.querySelector('#guestDialog').open"))
        t.eval("document.querySelector('[data-idea=forest]').click()")
        self.assertEqual(t.eval(kind), 'prompt')
        self.assertEqual(t.eval("document.querySelector('[data-start=prompt]').getAttribute('aria-pressed')"), 'true')
        self.assertEqual(t.eval("document.querySelector('#jobMsg').textContent"), '')
        self.assertEqual(self.store.users_list(), [])
        self.assertEqual(self.tab.errors, [])

    def test_generation_closed_shows_the_server_sentence_without_opening_a_workspace(self):
        t = self.tab
        self.store.generation_update(actor='fixture', enabled=False)
        self.ready()
        sentence = t.eval("fetch('/api/session').then(r => r.json()).then(s => s.generation.sentence)")
        t.wait("!document.querySelector('#generationClosed').hidden")
        self.assertEqual(t.eval("document.querySelector('#generationSentence').textContent"), sentence)
        t.eval("document.querySelector('[data-idea=moon]').click(); document.querySelector('#jobForm').requestSubmit()")
        t.wait(f"document.querySelector('#jobMsg').textContent === {sentence!r}")
        self.assertIn('moonlit mage', t.eval("document.querySelector('#prompt').value"))
        self.assertEqual(self.store.users_list(), [])
        self.store.generation_update(actor='fixture', enabled=True)
        t.eval("document.querySelector('#checkGeneration').click()")
        t.wait("document.querySelector('#generationClosed').hidden")
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
        items, self.artwork_sha = [], {}
        for name in ['woodland-scout', 'second-model']:
            folder = root / name
            folder.mkdir()
            assets = {}
            for kind, data, suffix in [('model', box_glb(), 'glb'), ('artwork', png(), 'png'), ('sketch', png(), 'png')]:
                path = f'{name}/{kind}.{suffix}'
                (root / path).write_bytes(data)
                assets[kind] = {'file': path, 'sha256': hashlib.sha256(data).hexdigest()}
                if kind == 'artwork':
                    self.artwork_sha[name] = assets[kind]['sha256']
            items.append({'id': name, 'title': name, 'description': 'Synthetic fixture.', 'code_commit': 'a' * 40, 'assets': assets})
        (root / 'catalog.json').write_text(json.dumps({'version': 1, 'items': items}))
        self.service_settings = {'public_demo': True, 'code_commit': 'a' * 40, 'gallery_dir': root}
        super().setUp()
        self.addCleanup(self.gallery_tmp.cleanup)

    def test_real_mesh_picker_shading_and_gallery_theme_without_guest_access(self):
        t = self.tab
        t.goto(self.url + '/')
        t.wait("document.querySelector('#landingStatus').textContent === 'Drag to rotate · Scroll to zoom'")
        t.wait("document.querySelectorAll('.g-card__model[src]').length === 2")
        # Picker thumbnails are the real artwork (the small thumbnail when the catalog offers one), loaded lazily.
        thumbs = t.eval("[...document.querySelectorAll('#modelPicker img')].map(i => [i.getAttribute('src'), i.loading])")
        self.assertEqual(thumbs, [[f'/gallery-assets/{name}/artwork-thumb?v={self.artwork_sha[name][:16]}', 'lazy']
                                  for name in ['woodland-scout', 'second-model']])
        self.assertEqual(t.eval("document.querySelectorAll('.step__art img').length"), 4)
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
        self.assertEqual(t.eval("document.querySelectorAll('.g-card__model[src]').length"), 0)
        t.eval("window.allowModels=true;document.querySelector('#retryLanding').click()")
        t.wait("document.querySelector('#landingStatus').textContent === 'Drag to rotate · Scroll to zoom' && document.querySelectorAll('.g-card__model[src]').length === 2 && document.querySelector('.thumbnail-render') === null")
        self.assertEqual(self.store.jobs_admin_list(), [])
        self.assertEqual(self.tab.errors, [])

    def test_failed_model_switch_clears_previous_mesh_and_recovers_on_retry(self):
        t = self.tab
        t.call('Emulation.setDeviceMetricsOverride', width=1440, height=960, deviceScaleFactor=1, mobile=False)
        t.goto(self.url + '/')
        t.wait("document.querySelector('#landingStatus').textContent === 'Drag to rotate · Scroll to zoom' && document.querySelectorAll('.g-card__model[src]').length === 2")
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
