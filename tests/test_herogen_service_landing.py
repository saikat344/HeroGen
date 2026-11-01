"""Landing page presentation in a real browser, with no model or GPU calls."""
import time
import unittest
from tests.test_herogen_service_e2e import ServiceBrowserCase, CDP_PORT, HAVE_WEB, uvicorn


@unittest.skipUnless(HAVE_WEB and uvicorn is not None and CDP_PORT, 'set E2E_CDP_PORT')
class TestLandingPage(ServiceBrowserCase):
    service_settings = {'public_demo': True, 'code_commit': 'a' * 40}

    def ready(self):
        self.tab.goto(self.url + '/')
        self.tab.wait("document.querySelector('#themePreference') && !document.querySelector('#entry').hidden")

    def choose(self, value):
        self.tab.eval(f"document.querySelector('#themePreference').value='{value}'; document.querySelector('#themePreference').dispatchEvent(new Event('change'))")

    def test_theme_persists_and_system_choice_tracks_os(self):
        t = self.tab
        t.call('Emulation.setEmulatedMedia', features=[{'name': 'prefers-color-scheme', 'value': 'dark'}])
        self.ready()
        self.choose('system')
        self.assertEqual(t.eval('document.documentElement.dataset.theme'), 'dark')
        self.choose('light')
        self.assertEqual(t.eval('document.documentElement.dataset.theme'), 'light')
        t.goto(self.url + '/')
        t.wait("document.querySelector('#themePreference')?.value === 'light'")
        self.assertEqual(t.eval('document.documentElement.dataset.theme'), 'light')
        self.choose('dark')
        self.assertEqual(t.eval("getComputedStyle(document.body).backgroundColor"), 'rgb(21, 23, 25)')
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
        t.eval("document.querySelector('#themePreference').value='dark';document.querySelector('#themePreference').dispatchEvent(new Event('change'))")
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

    def test_header_navigation_recovers_across_signin_routes_and_library_errors(self):
        self.runner.gate.set()
        t = self.tab
        t.call('Emulation.setDeviceMetricsOverride', width=1440, height=960, deviceScaleFactor=1, mobile=False)
        t.call(
            'Page.addScriptToEvaluateOnNewDocument',
            source="window.__unhandled=[]; window.addEventListener('unhandledrejection', e => window.__unhandled.push(String(e.reason)));",
        )
        t.goto(self.url + '/signin')
        t.wait("/incomplete/.test(document.querySelector('#verifyWho').textContent)")
        t.eval('document.querySelector(\'nav a[href="/#gallery"]\').click()')
        # This click leaves /signin by a real navigation to /#gallery, so a poll can run in the
        # freshly committed document before index.html parsing reaches #verify, #landingStatus or
        # #modelPicker. The optional chaining treats not-yet-parsed as not-ready instead of throwing;
        # when the wait passes, every condition below still holds at the same instant.
        t.wait(
            "location.pathname === '/' && document.querySelector('#verify')?.hidden "
            "&& document.querySelector('#landingStatus')?.textContent === 'Drag to rotate · Scroll to zoom' "
            "&& document.querySelector('#modelPicker')?.children.length === 2"
        )
        t.eval("document.querySelector('[data-idea=forest]').click(); document.querySelector('#jobForm').requestSubmit()")
        t.wait("!document.querySelector('#detail').hidden")
        t.eval("""(() => {
            const orig = window.fetch;
            window.jobsStatus = 503;
            window.fetch = (u, o) => u === '/api/jobs' && (!o || !o.method || o.method === 'GET') && window.jobsStatus
                ? Promise.resolve(new Response(JSON.stringify({error:{code:'busy', message:'The service is busy; please try again in a moment.'}}), {status:window.jobsStatus}))
                : orig(u, o);
        })()""")
        t.eval("document.querySelector('#headerSignIn').click()")
        t.wait("!document.querySelector('#banner').hidden && /busy/i.test(document.querySelector('#banner').textContent)")
        self.assertEqual(t.eval('window.__unhandled'), [])
        self.assertEqual(t.errors, [])

        t.goto(self.url + '/admin/signin')
        t.wait("document.querySelector('#signinDialog').open")
        t.eval("document.querySelector('#closeSignin').click()")
        t.wait("!document.querySelector('#signinDialog').open")
        self.assertEqual(t.eval("document.querySelector('#headerSignIn').textContent"), 'Sign in')
        t.eval("document.querySelector('#headerSignIn').click()")
        t.wait("document.querySelector('#signinDialog').open")
        self.assertTrue(t.eval("document.querySelector('#heroLibrary').hidden"))

        t.goto(self.url + '/')
        t.wait("!document.querySelector('#headerSignIn').hidden && !document.querySelector('#headerSignIn').disabled")
        t.eval("""(() => {
            const orig = window.fetch;
            window.fetch = (u, o) => u === '/api/jobs' && (!o || !o.method || o.method === 'GET')
                ? Promise.resolve(new Response(JSON.stringify({error:{code:'unauthorized', message:'Please sign in.'}}), {status:401}))
                : orig(u, o);
            document.querySelector('#headerSignIn').click();
        })()""")
        t.wait("document.querySelector('#headerSignIn').hidden && /expired/i.test(document.querySelector('#jobMsg').textContent)")
        self.assertEqual(t.eval('window.__unhandled'), [])
        self.assertEqual(t.errors, [])


@unittest.skipUnless(HAVE_WEB and uvicorn is not None and CDP_PORT, 'set E2E_CDP_PORT')
class TestGuestAccessSummary(ServiceBrowserCase):
    """The guest access summary is a refusal notice: present only while generation is not accepting."""
    service_settings = {'public_demo': True, 'code_commit': 'a' * 40}

    def open_guest_workspace(self):
        t = self.tab
        t.goto(self.url + '/')
        t.wait("!document.querySelector('#headerSignIn').disabled")
        t.eval("document.querySelector('#headerSignIn').click()")
        t.wait("document.querySelector('#guestDialog').open")
        t.eval("document.querySelector('#guestGo').click()")
        t.wait("document.querySelector('#who').textContent === 'Private guest workspace'")

    def apply_refusal(self, *statements):
        """Return the synthetic policy to accepting, then apply this scenario's refusal rows."""
        db = self.store.db
        db.execute('UPDATE admission_policy SET enabled=1, code_commit=?, max_active=5', (self.s.code_commit,))
        db.execute('DELETE FROM budget_overhead')
        db.execute("DELETE FROM admissions WHERE user_id='fixture-refusal'")
        for sql, args in statements:
            db.execute(sql, args)

    def test_summary_appears_only_while_generation_is_refused(self):
        t = self.tab
        self.open_guest_workspace()
        self.assertTrue(t.eval("document.querySelector('#guestAccessSummary').hidden"))
        self.assertEqual(t.eval("document.querySelector('#quota').textContent"), '')
        policy = self.store.db.execute('SELECT limit_micro, reserve_micro FROM admission_policy WHERE id=1').fetchone()
        suffix = ' You can still explore the generated heroes.'
        refusals = [
            ('generation_paused', 'Live generation is paused right now.',
             [('UPDATE admission_policy SET enabled=0', ())]),
            ('cost_bound_unverified', 'Live generation is paused while its cost bound is verified.',
             [('UPDATE admission_policy SET code_commit=?', ('b' * 40,))]),
            ('generation_budget_exhausted', 'The live-generation budget is used up for now.',
             [('INSERT INTO budget_overhead(id, amount_micro, evidence, updated_at) VALUES (1, ?, ?, ?)',
               (policy['limit_micro'] - policy['reserve_micro'] + 1, 'synthetic browser fixture', time.time()))]),
            ('generation_capacity', 'Every live-generation slot is busy. Try again in a moment.',
             [('UPDATE admission_policy SET max_active=1', ()),
              ('INSERT INTO admissions(job_id, user_id, client_key, amount_micro, cost_signature, created_at) '
               "VALUES ('fixture-capacity', 'fixture-refusal', 'fixture', ?, 'fixture', ?)",
               (policy['reserve_micro'], time.time()))]),
        ]
        for reason, text, statements in refusals:
            with self.subTest(reason):
                self.apply_refusal(*statements)
                t.goto(self.url + '/')
                t.wait("document.querySelector('#who').textContent === 'Private guest workspace'")
                t.wait("!document.querySelector('#guestAccessSummary').hidden")
                self.assertEqual(t.eval("document.querySelector('#guestAccessText').textContent"), text + suffix)
        self.apply_refusal()
        t.eval("document.querySelector('#checkAvailability').click()")
        t.wait("document.querySelector('#guestAccessSummary').hidden")
        self.assertEqual(t.errors, [])
