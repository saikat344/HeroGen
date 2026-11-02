"""Real-browser version selection, refinement, comparison and refusal recovery with synthetic runs."""
import unittest
import base64
import io

from test_herogen_service_e2e import CDP_PORT, HAVE_WEB, ServiceBrowserCase, uvicorn
from test_herogen_service_variations import RefusedRunner


@unittest.skipUnless(HAVE_WEB and uvicorn is not None and CDP_PORT, 'set E2E_CDP_PORT and install requirements-dev.txt')
class TestVariationsBrowser(ServiceBrowserCase):
    def ready(self):
        self.runner.gate.set(); self.signin(); self.make_hero('Brass courier')
        self.tab.wait("document.querySelector('#state').textContent === 'succeeded' && !document.querySelector('#retryVersion').disabled", timeout=45)
        self.tab.wait("!document.querySelector('#orbitHint').hidden", timeout=45)
        return self.tab.eval("document.querySelector('#versionSelect').value")

    def test_tree_compare_and_mobile_list_select_the_version(self):
        root = self.ready(); t = self.tab
        self.screenshot('variations-result.png')
        t.eval("document.querySelector('#refineVersion').click()")
        self.assertTrue(t.eval("document.querySelector('#refineNoteWrap').hidden"))
        self.assertEqual(t.eval("document.querySelector('#submitRefine').textContent"), 'Retry texture')
        t.eval("document.querySelector('#refineStage').value='turnaround';document.querySelector('#refineStage').dispatchEvent(new Event('change'));document.querySelector('#refineNote').value='Weathered brass';document.querySelector('#refineNote').dispatchEvent(new Event('input'))")
        self.assertFalse(t.eval("document.querySelector('#refineNoteWrap').hidden"))
        self.assertEqual(t.eval("document.querySelector('#refineCount').textContent"),'15 / 300 characters')
        t.eval("document.querySelector('#refineForm').requestSubmit()")
        t.wait(f"document.querySelector('#versionSelect').value !== '{root}' && document.querySelector('#versionSelect').options.length === 2 && !document.querySelector('#retryVersion').disabled")
        child = t.eval("document.querySelector('#versionSelect').value")
        t.eval("document.querySelector('#branchesTab').click()")
        t.wait("document.querySelectorAll('#versionTree .vt-node:not([data-key^=\"step-\"])').length === 2 && !document.querySelector('#compareVersions').disabled")
        self.screenshot('variations-branches.png')
        self.assertEqual(t.eval(f"document.querySelector('#versionTree [data-key=\"{child}\"]').getAttribute('aria-level')"), '2')
        self.assertEqual(t.eval("document.querySelector('#versionTree [aria-current=\"true\"]').dataset.key"), child)
        t.eval(f"document.querySelector('#versionTree [data-key=\"{root}\"]').click()")
        t.wait(f"document.querySelector('#versionSelect').value === '{root}'")
        self.assertFalse(t.eval("document.querySelector('#branchesPanel').hidden"))
        self.assertTrue(t.eval(f"document.querySelector('#preview').getAttribute('aria-label').includes('body') && location.hash.includes('{root}')"))
        t.eval("document.querySelector('#compareVersions').click()")
        t.wait("document.querySelector('#compareDialog').open && document.querySelector('#compareLabelB').textContent.includes('$')")
        t.wait("document.querySelector('#compareStatus').textContent === 'Cameras linked'")
        self.assertIn('Weathered brass', t.eval("document.querySelector('#compareDialog').textContent"))
        boxes = t.eval("['A','B'].map(s=>{const b=document.querySelector('#compareCanvas'+s).getBoundingClientRect();return {x:b.x,y:b.y,width:b.width,height:b.height,scale:1};})")
        def frames():
            from PIL import Image
            return [Image.open(io.BytesIO(base64.b64decode(t.call('Page.captureScreenshot', format='png', clip=b)['data']))).convert('RGB') for b in boxes]
        before = frames()
        x, y = boxes[0]['x'] + boxes[0]['width'] / 2, boxes[0]['y'] + boxes[0]['height'] / 2
        t.call('Input.dispatchMouseEvent', type='mousePressed', x=x, y=y, button='left', buttons=1, clickCount=1)
        t.call('Input.dispatchMouseEvent', type='mouseMoved', x=x+65, y=y+25, button='left', buttons=1)
        t.call('Input.dispatchMouseEvent', type='mouseReleased', x=x+65, y=y+25, button='left', buttons=0)
        t.eval('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
        from PIL import ImageChops
        for initial, moved in zip(before, frames()):
            self.assertIsNotNone(ImageChops.difference(initial, moved).getbbox(), 'Both comparison cameras must orbit')
        self.screenshot('variations-compare.png')
        t.eval("document.querySelector('#closeCompare').click()")
        t.call('Emulation.setDeviceMetricsOverride', width=390, height=844, deviceScaleFactor=1, mobile=True)
        t.eval("document.querySelector('#branchesPanel').scrollIntoView({block:'center'})")
        self.assertEqual(t.eval("getComputedStyle(document.querySelector('#versionTree .vt-fork')).display"), 'block', 'branches stack on phones')
        self.assertTrue(t.eval("""(() => { const box = document.querySelector('#versionTree').getBoundingClientRect();
          return document.documentElement.scrollWidth <= innerWidth && [...document.querySelectorAll('#versionTree .vt-node')]
            .every(n => { const r = n.getBoundingClientRect(); return r.left >= box.left - 1 && r.right <= box.right + 1; }); })()"""))
        self.screenshot('variations-mobile-tree.png')
        t.eval(f"document.querySelector('#versionTree [data-key=\"{root}\"]').focus()")
        t.call('Input.dispatchKeyEvent', type='rawKeyDown', key='ArrowLeft', code='ArrowLeft', windowsVirtualKeyCode=37)
        self.assertEqual(t.eval("document.activeElement.dataset.key"), child, 'Left moves into the Refine branch')
        t.call('Input.dispatchKeyEvent', type='rawKeyDown', key='ArrowUp', code='ArrowUp', windowsVirtualKeyCode=38)
        self.assertEqual(t.eval("document.activeElement.dataset.key"), root)
        t.eval(f"document.querySelector('#versionSelect').value='{child}'; document.querySelector('#versionSelect').dispatchEvent(new Event('change'))")
        t.wait(f"document.querySelector('#versionSelect').value === '{child}' && document.querySelector('#state').textContent === 'succeeded'")
        self.assertEqual(t.errors, [])

    def test_retry_and_failure_try_again_never_overwrite_the_parent(self):
        root = self.ready(); t = self.tab
        t.eval("document.querySelector('#retryVersion').click()")
        t.wait(f"document.querySelector('#versionSelect').value !== '{root}' && !document.querySelector('#retryVersion').disabled")
        child = t.eval("document.querySelector('#versionSelect').value")
        self.assertNotEqual(root, child)
        self.assertEqual(self.store.job_get(root)['state'], 'succeeded')
        self.runner.fail['2'] = 'error'
        t.eval("document.querySelector('#refineVersion').click();document.querySelector('#refineStage').value='turnaround';document.querySelector('#refineStage').dispatchEvent(new Event('change'));document.querySelector('#refineNote').value='Blue coat';document.querySelector('#refineForm').requestSubmit()")
        t.wait("!document.querySelector('#variationFailure').hidden && document.querySelector('#state').textContent === 'failed'")
        self.assertIn('previous result is saved', t.eval("document.querySelector('#variationFailure').textContent"))
        failed = t.eval("document.querySelector('#versionSelect').value")
        self.runner.fail.clear()
        t.eval("document.querySelector('#retryFailedVersion').click()")
        t.wait(f"document.querySelector('#versionSelect').value !== '{failed}' && !document.querySelector('#retryVersion').disabled")
        again = t.eval("document.querySelector('#versionSelect').value")
        self.assertEqual((self.store.job_get(again)['note'], self.store.job_get(again)['parent_job']), ('Blue coat', failed))
        self.assertEqual(self.store.job_get(failed)['state'], 'failed')
        self.assertEqual(t.errors, [])

    def test_server_note_rejection_is_inline_and_does_not_create_a_version(self):
        self.ready(); t = self.tab
        t.eval("document.querySelector('#refineVersion').click();document.querySelector('#refineStage').value='turnaround';document.querySelector('#refineStage').dispatchEvent(new Event('change'));document.querySelector('#refineNote').value='123!';document.querySelector('#refineForm').requestSubmit()")
        t.wait("!document.querySelector('#refineError').hidden")
        self.assertTrue(t.eval("document.querySelector('#refineDialog').open"))
        self.assertIn('300', t.eval("document.querySelector('#refineError').textContent"))
        self.assertEqual(len(self.store.jobs_admin_list()), 1)

    def test_refused_input_retry_and_change_idea_keep_the_saved_input(self):
        self.runner.gate.set()
        self.runner.run = RefusedRunner().run
        self.signin(); self.make_hero('Full body please'); t = self.tab
        t.wait("!document.querySelector('#inputRefusal').hidden")
        self.assertIn('cut off', t.eval("document.querySelector('#inputReason').textContent"))
        self.assertFalse(t.eval("[...document.querySelectorAll('#downloads a')].some(a=>a.textContent==='Download GLB')"))
        self.screenshot('variations-refused-input.png')
        first = self.store.jobs_admin_list()[0]['id']
        t.eval("document.querySelector('#retryInput').click()")
        t.wait(f"location.hash !== '#job={first}' && !document.querySelector('#inputRefusal').hidden")
        self.assertEqual(len(self.store.jobs_admin_list()), 2)
        t.eval("document.querySelector('#changeIdea').click()")
        t.wait("!document.querySelector('#entry').hidden && document.querySelector('#uploadPreview').naturalWidth > 0")
        self.assertEqual(t.eval("document.querySelector('#name').value"), 'Full body please')
        self.assertEqual(t.errors, [])


@unittest.skipUnless(HAVE_WEB and uvicorn is not None and CDP_PORT, 'set E2E_CDP_PORT and install requirements-dev.txt')
class TestTextureNotesBrowser(ServiceBrowserCase):
    service_settings = {'texture_notes': True}

    def test_texture_note_is_offered_only_with_flag_and_reaches_lineage(self):
        self.runner.gate.set(); self.signin(); self.make_hero('Texture variation'); t = self.tab
        t.wait("document.querySelector('#state').textContent === 'succeeded' && !document.querySelector('#refineVersion').disabled")
        root = self.store.jobs_admin_list()[0]['id']
        t.eval("document.querySelector('#refineVersion').click()")
        self.assertFalse(t.eval("document.querySelector('#refineNoteWrap').hidden"))
        self.assertEqual(t.eval("document.querySelector('#refineStage').value"), 'texture')
        t.eval("document.querySelector('#refineNote').value='Weathered brass';document.querySelector('#refineForm').requestSubmit()")
        t.wait(f"document.querySelector('#versionSelect').value !== '{root}' && !document.querySelector('#refineVersion').disabled")
        child = self.store.job_get(t.eval("document.querySelector('#versionSelect').value"))
        self.assertEqual((child['from_stage'], child['note']), ('3f','Weathered brass'))
        t.eval("document.querySelector('#recipeTab').click()")
        self.assertIn('Weathered brass', t.eval("document.querySelector('#recipePanel').textContent"))
        self.assertIsNone(t.eval("document.querySelector('#recipePanel input, #recipePanel textarea, #recipePanel select')"))
        self.assertEqual(t.errors, [])


@unittest.skipUnless(HAVE_WEB and uvicorn is not None and CDP_PORT, 'set E2E_CDP_PORT and install requirements-dev.txt')
class TestWorkbenchPresentation(ServiceBrowserCase):
    service_settings = {'texture_notes': True}

    def assert_layout(self):
        t = self.tab
        self.assertTrue(t.eval("""(() => {
          const b=document.querySelector('.workbench-breadcrumb').getBoundingClientRect();
          const e=document.querySelector('.generation-heading .eyebrow').getBoundingClientRect();
          return b.bottom <= e.top && b.height > 0 && e.height > 0;
        })()"""), 'Breadcrumb and eyebrow must occupy separate lines')
        self.assertTrue(t.eval('document.documentElement.scrollWidth <= innerWidth'), 'Page must fit the viewport')
        self.assertEqual(t.eval("""[...document.querySelectorAll('#detail *')].filter(e => {
          const r=e.getBoundingClientRect(); if (!r.width || !r.height || getComputedStyle(e).position==='absolute') return false;
          for(let p=e.parentElement;p;p=p.parentElement) if (['auto','scroll','hidden'].includes(getComputedStyle(p).overflowX)) return false;
          return r.left < -1 || r.right > innerWidth+1;
        }).map(e=>e.id||e.className)"""), [])

    def capture_matrix(self, state, panel='live'):
        t = self.tab
        for width, height, size in [(1586,1000,'desktop'),(375,812,'375px')]:
            t.call('Emulation.setDeviceMetricsOverride',width=width,height=height,deviceScaleFactor=1,mobile=width==375)
            for theme in ['light','dark']:
                t.eval(f"window.herogenTheme.choose('{theme}')")
                t.eval(f"document.querySelector('#{panel}Tab').click();window.scrollTo(0,0)")
                t.eval('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                self.assert_layout()
                self.screenshot(f'workbench-{state}-{theme}-{size}.png')

    def test_running_finished_tree_and_compare_fit_both_themes_and_sizes(self):
        import threading
        stage_gate = threading.Event()
        original = self.runner.run
        def run(spec, ctx):
            if spec.key == '2':
                stage_gate.wait(45)
            return original(spec, ctx)
        self.runner.run = run
        self.runner.gate.set()
        try:
            self.signin(); self.make_hero('Clockwork gardener'); t = self.tab
            t.wait("document.querySelector('#stages .st-running')?.dataset.stage === '2'")
            t.wait("document.querySelector('#stageLog').textContent.includes('started')")
            self.assertFalse(t.eval("document.querySelector('#cancelJob').hidden"))
            self.assertFalse(t.eval("document.querySelector('.site-header').contains(document.querySelector('#developerMode'))"))
            self.assertEqual(t.eval("document.querySelector('#who').textContent"),'Private workspace')
            self.capture_matrix('running')
        finally:
            stage_gate.set()
        t.wait("document.querySelector('#state').textContent === 'succeeded' && !document.querySelector('#refineVersion').disabled",timeout=45)
        t.wait("!document.querySelector('#orbitHint').hidden",timeout=45)
        self.assertEqual(t.eval("document.querySelector('#generationTitle').textContent"),'Your hero. Your next idea.')
        self.assertTrue(t.eval("document.querySelector('#cancelJob').hidden"))
        self.assertEqual(t.eval("document.querySelector('#viewSteps').getAttribute('aria-expanded')"),'false')
        t.eval("document.querySelector('#viewSteps').click()")
        self.assertTrue(t.eval("[...document.querySelectorAll('#stages li')].every(e=>e.textContent.includes('$') && e.querySelector('a[target=\"_blank\"]'))"))
        t.eval("document.querySelector('#viewSteps').click()")
        self.capture_matrix('finished')
        t.eval("Object.defineProperty(navigator.clipboard,'writeText',{value:async text=>{window.copiedStageLog=text;},configurable:true});document.querySelector('#copyLog').click()")
        t.wait("document.querySelector('#logStatus').textContent === 'Log copied'")
        self.assertIn('Refined version 1 ready',t.eval('window.copiedStageLog'))
        self.assertIn('started',t.eval('window.copiedStageLog'))
        root = t.eval("document.querySelector('#versionSelect').value")
        t.eval("document.querySelector('#refineNote').value='Weathered brass';document.querySelector('#refineForm').requestSubmit()")
        t.wait("document.querySelector('#versionSelect').options.length === 2 && !document.querySelector('#refineVersion').disabled")
        t.wait("!document.querySelector('#compareVersions').disabled")
        self.assertEqual(t.eval("document.querySelectorAll('#versionTree [data-key^=\"step-\"]').length"),2)
        self.assertEqual(t.eval("document.querySelector('.vt-fork .vt-label').textContent"),'Refine')
        t.eval(f"document.querySelector('#versionSelect').value='{root}';document.querySelector('#versionSelect').dispatchEvent(new Event('change'))")
        t.wait(f"location.hash === '#job={root}' && !document.querySelector('#detail').hidden && document.querySelector('#versionSelect').value === '{root}' && !document.querySelector('#retryVersion').disabled")
        t.eval("document.querySelector('#retryVersion').click()")
        t.wait("document.querySelector('#versionSelect').options.length === 3 && !document.querySelector('#retryVersion').disabled")
        self.assertEqual(t.eval("document.querySelectorAll('.vt-fork > .vt-branches > .vt-side > .vt-item').length"),2)
        self.assertEqual(t.eval("[...document.querySelectorAll('.vt-fork .vt-label')].map(e=>e.textContent)"),['Refine','Retry'])
        self.capture_matrix('branches','branches')
        t.call('Emulation.setDeviceMetricsOverride',width=1586,height=1000,deviceScaleFactor=1,mobile=False)
        before = t.eval("document.querySelector('#versionTree').getBoundingClientRect().width")
        t.eval("document.querySelector('#expandTree').click()")
        self.assertTrue(t.eval("document.querySelector('#treeDialog').open && document.querySelector('#treeDialog').contains(document.querySelector('#versionTree'))"))
        self.assertGreater(t.eval("document.querySelector('#versionTree').getBoundingClientRect().width"),before)
        self.assertEqual(t.eval("document.querySelector('#expandTree').getAttribute('aria-expanded')"),'true')
        t.eval("document.querySelector('#closeTree').click()")
        t.wait("document.querySelector('#treeHome').contains(document.querySelector('#versionTree'))")
        for width, height, size in [(1586,1000,'desktop'),(375,812,'375px')]:
            t.call('Emulation.setDeviceMetricsOverride',width=width,height=height,deviceScaleFactor=1,mobile=width==375)
            for theme in ['light','dark']:
                t.eval(f"window.herogenTheme.choose('{theme}');document.querySelector('#compareVersions').click()")
                t.wait("document.querySelector('#compareStatus').textContent === 'Cameras linked' || !document.querySelector('#compareError').hidden")
                self.assertEqual(t.eval("document.querySelector('#compareStatus').textContent"),'Cameras linked', t.eval("document.querySelector('#compareError').textContent"))
                self.assertTrue(t.eval('document.documentElement.scrollWidth <= innerWidth'))
                self.assertTrue(t.eval("[...document.querySelectorAll('#compareDialog canvas')].every(e=>{const r=e.getBoundingClientRect();return r.left>=0 && r.right<=innerWidth})"))
                self.screenshot(f'workbench-compare-{theme}-{size}.png',viewport=True)
                t.eval("document.querySelector('#closeCompare').click()")
        self.runner.fail['1'] = 'error'
        t.eval(f"document.querySelector('#versionSelect').value='{root}';document.querySelector('#versionSelect').dispatchEvent(new Event('change'))")
        t.wait(f"location.hash === '#job={root}' && !document.querySelector('#detail').hidden && document.querySelector('#versionSelect').value === '{root}' && !document.querySelector('#retryVersion').disabled")
        t.eval("document.querySelector('#retryVersion').click()")
        t.wait("!document.querySelector('#variationFailure').hidden && document.querySelector('#state').textContent === 'failed'")
        self.capture_matrix('failed-variation','branches')
        self.assertIn('previous result is saved',t.eval("document.querySelector('#variationFailure').textContent"))
        self.assertEqual(self.store.job_get(root)['state'],'succeeded')
        t.eval(f"document.querySelector('#versionSelect').value='{root}';document.querySelector('#versionSelect').dispatchEvent(new Event('change'))")
        t.wait(f"location.hash === '#job={root}' && !document.querySelector('#detail').hidden && document.querySelector('#versionSelect').value === '{root}' && !document.querySelector('#retryVersion').disabled")
        t.eval("document.querySelector('#fullscreenPreview').click()")
        t.wait("document.fullscreenElement === document.querySelector('#viewer') || document.querySelector('#viewer').classList.contains('viewer--expanded')")
        self.assertEqual(t.eval("document.querySelector('#fullscreenPreview').getAttribute('aria-pressed')"), 'true')
        t.eval("document.querySelector('#fullscreenPreview').click()")
        t.wait("!document.fullscreenElement && !document.querySelector('#viewer').classList.contains('viewer--expanded')")
        t.eval("document.querySelector('#previewLighting').click()")
        self.assertEqual(t.eval("document.querySelector('#previewLighting').getAttribute('aria-pressed')"),'true')
        self.assertEqual(t.errors, [])
