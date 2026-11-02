"""Real-browser Build log: rows per model call that change in place, the event cursor, Follow live, reduced motion
and Copy log, fed by a test route that stands in for the recorded events of a run."""
import threading
import time
import unittest

from test_herogen_service_e2e import CDP_PORT, HAVE_WEB, ServiceBrowserCase, uvicorn


@unittest.skipUnless(HAVE_WEB and uvicorn is not None and CDP_PORT, 'set E2E_CDP_PORT and install requirements-dev.txt')
class TestBuildLogBrowser(ServiceBrowserCase):
    def feed_events(self):
        """Serve `self.feed` from /api/jobs/<id>/events with a numeric cursor; `self.finished` marks the end."""
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        self.feed, self.reads, self.finished = [], [], threading.Event()

        async def events(request):
            after = int(request.query_params.get('after') or 0)
            self.reads.append(request.url.query)
            page = self.feed[after:]
            return JSONResponse({'v': 1, 'events': page, 'cursor': str(after + len(page)), 'more': False,
                                 'done': self.finished.is_set() and after + len(page) == len(self.feed)},
                                headers={'Cache-Control': 'no-store'})
        app = self.server.config.app
        while not hasattr(app, 'routes'):
            app = app.app
        app.routes.insert(0, Route('/api/jobs/{job}/events', events))

    def emit(self, kind, step=None, **fields):
        event = {'v': 1, 't': int(time.time() * 1000), 'kind': kind, **({'step': step} if step else {}), **fields}
        self.feed.append(event)

    def test_rows_morph_in_place_follow_live_reduced_motion_and_copy(self):
        self.feed_events()
        self.signin()
        self.make_hero('Lantern keeper')
        t = self.tab
        self.assertTrue(self.runner.waiting.wait(30))
        self.emit('job_start', a=1, of=2)
        self.emit('step_start', '0', a=1, i=1, of=8)
        self.emit('local', '0', what='input_checked', s=0.2, m={'min_side_px': 400, 'contrast': 120})
        self.emit('step_end', '0', outcome='passed', s=0.2, usd=0)
        self.emit('step_start', '1', a=1, i=2, of=8)
        self.emit('call_start', '1', id='a1b2c3d4e5f6', who='muse-image', what='cover_art')
        t.wait("document.querySelector('#stageLog .bl-call.st-running')")
        row = "document.querySelector('#stageLog .bl-call')"
        self.assertIn('Painting the colour artwork from the sketch', t.eval(f"{row}.textContent"))
        self.assertIn('usually 13 s', t.eval(f"{row}.textContent"))
        self.assertEqual(t.eval(f"getComputedStyle({row}.querySelector('.bl-mark .icon')).animationName"), 'spin')
        self.assertIn('Step 2 of 8 · Artwork', t.eval("document.querySelector('#stageLog .bl-step.st-running').textContent"))
        t.eval(f"window.__row = {row}")
        self.emit('call_end', '1', id='a1b2c3d4e5f6', who='muse-image', what='cover_art', outcome='ok', s=17.3, usd=0.01)
        t.wait("window.__row.classList.contains('st-passed')", timeout=10)
        self.assertTrue(t.eval(f"window.__row === {row} && window.__row.isConnected"), 'the row changed in place')
        self.assertIn('17.3 s · $0.01', t.eval("window.__row.textContent"))
        self.assertTrue(any(q.startswith('after=') for q in self.reads), self.reads)
        self.assertEqual(self.reads[0], '')

        # Reduced motion: the running row shows a static "…" instead of a spinner.
        t.call('Emulation.setEmulatedMedia', features=[{'name': 'prefers-reduced-motion', 'value': 'reduce'}])
        self.emit('call_start', '1', id='0a1b2c3d4e5f', who='sam-3.1', what='cover_art', target='feet')
        t.wait("document.querySelector('#stageLog .bl-call.st-running')", timeout=10)
        running = "document.querySelector('#stageLog .bl-call.st-running')"
        self.assertEqual(t.eval(f"getComputedStyle({running}.querySelector('.bl-mark .icon')).display"), 'none')
        self.assertNotEqual(t.eval(f"getComputedStyle({running}.querySelector('.bl-dots')).display"), 'none')
        t.call('Emulation.setEmulatedMedia', features=[])

        # Follow live: scrolling up to read stops following; ticking it again returns to the newest row.
        for i in range(12):
            self.emit('local', '1', what='sheet_split', m={'figures': 4})
        t.wait("document.querySelectorAll('#stageLog .bl-row').length >= 14", timeout=20)
        t.wait("(l => l.scrollHeight > l.clientHeight && l.scrollTop + l.clientHeight >= l.scrollHeight - 2)(document.querySelector('#stageLog'))")
        t.eval("(l => { l.scrollTop = 0; l.dispatchEvent(new Event('scroll')); })(document.querySelector('#stageLog'))")
        self.assertFalse(t.eval("document.querySelector('#followLive').checked"))
        t.eval("document.querySelector('#followLive').click()")
        self.assertTrue(t.eval("(l => l.scrollTop + l.clientHeight >= l.scrollHeight - 2)(document.querySelector('#stageLog'))"))

        # The run ends: the cursor reads to the end once and the loop stops.
        self.emit('call_end', '1', id='0a1b2c3d4e5f', who='sam-3.1', what='cover_art', target='feet', outcome='ok', s=2.6, usd=0.0025)
        self.emit('step_end', '1', outcome='passed', s=21.0, usd=0.0125)
        self.runner.gate.set()
        t.wait("document.querySelector('#state').textContent === 'succeeded'", timeout=60)
        self.emit('job_end', state='succeeded', export='ready', s=42.0, usd=0.0125)
        self.finished.set()
        t.wait("document.querySelector('#stageLog').textContent.includes('Roblox files ready')", timeout=15)
        settled = len(self.reads)
        time.sleep(4)
        self.assertEqual(len(self.reads), settled, 'no reads after done')

        t.eval("Object.defineProperty(navigator.clipboard,'writeText',{value:async text=>{window.copied=text;},configurable:true});document.querySelector('#copyLog').click()")
        t.wait("document.querySelector('#logStatus').textContent === 'Log copied'")
        lines = t.eval('window.copied').split('\n')
        self.assertIn('▸ Step 2 of 8 · Artwork', '\n'.join(lines))
        self.assertTrue(any(l.endswith('… Muse Image: painting the colour artwork from the sketch (same pose and outline)…') for l in lines))
        self.assertTrue(any(l.endswith('✓ Muse Image: colour artwork ready · 17.3 s · $0.01 (usually 13 s)') for l in lines))
        self.assertTrue(all(l.startswith('+') for l in lines), lines)
        self.assertEqual(t.eval("document.querySelector('#stageLog').getAttribute('role')"), 'log')
        self.assertEqual(self.tab.errors, [])


if __name__ == '__main__':
    unittest.main()
