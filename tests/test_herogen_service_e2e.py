"""The service page in a real browser: headless Chrome driven over the DevTools protocol, against the web app with
stand-ins (the fake stage runner, in-process workers, no email) on 127.0.0.1. It signs in by link, makes heroes,
cancels one mid-stage and follows one to the end, and checks how the page polls.

Skipped unless E2E_CDP_PORT names the port of a running Chrome started with --remote-debugging-port, and the dev
dependencies (herogen/service/requirements-dev.txt, which adds uvicorn) are installed; CI has neither, so they
skip there. For example on macOS:

    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --remote-debugging-port=9447 \\
        --user-data-dir="$(mktemp -d)" --no-first-run about:blank &
    E2E_CDP_PORT=9447 python3 -m unittest tests.test_herogen_service_e2e -v

Each test opens its own tab and closes it afterwards; the browser itself is left running.
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import struct
import sys
import threading
import time
import unittest
import urllib.parse
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from test_herogen_service_support import HAVE_WEB, TempDirCase, png, settings

try:
    import uvicorn
except ImportError:
    uvicorn = None

CDP_PORT = os.environ.get("E2E_CDP_PORT", "")
POLL_S = 3.0   # app.js polls an open job every 3 s

if HAVE_WEB:
    from herogen.service.auth import Signer
    from herogen.service.fakes import FakeRunner, FakeSender
    from herogen.service.storage import Storage
    from herogen.service.store import Store
    from herogen.service.uploads import Limits, sanitize
    from herogen.service.web import Services, create_app
    from herogen.service.worker import ThreadBackend

    class GatedRunner(FakeRunner):
        """The fake stages, except that stage 1 waits until `gate` opens (so a job can be caught mid-stage)."""

        def __init__(self) -> None:
            super().__init__()
            self.gate = threading.Event()
            self.waiting = threading.Event()

        def run(self, spec, ctx):
            if spec.key == "1":
                self.waiting.set()
                self.gate.wait(60)
            return super().run(spec, ctx)


class WebSocket:
    """Just enough of RFC 6455 for the DevTools protocol on localhost: text frames, client-side masking."""

    def __init__(self, url: str) -> None:
        u = urllib.parse.urlparse(url)
        self.sock = socket.create_connection((u.hostname, u.port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\nUpgrade: websocket\r\n"
                           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
                          .encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("the browser closed the connection during the handshake")
            head += chunk
        head, self.buf = head.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise ConnectionError(head.decode(errors="replace"))

    def send(self, text: str) -> None:
        data, mask = text.encode(), os.urandom(4)
        n = len(data)
        if n < 126:
            head = struct.pack(">BB", 0x81, 0x80 | n)
        elif n < 65536:
            head = struct.pack(">BBH", 0x81, 0x80 | 126, n)
        else:
            head = struct.pack(">BBQ", 0x81, 0x80 | 127, n)
        self.sock.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _read(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("the browser closed the connection")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def recv(self) -> str:
        msg = b""
        while True:
            b0, b1 = self._read(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            if b1 & 0x80:
                self._read(4)   # a server never masks; skip a mask if one is sent anyway
            payload, op = self._read(n), b0 & 0x0F
            if op == 8:
                raise ConnectionError("the browser closed the connection")
            if op in (9, 10):
                continue
            msg += payload
            if b0 & 0x80:
                return msg.decode()

    def close(self) -> None:
        self.sock.close()


class Tab:
    """One browser tab over the DevTools protocol. Exceptions thrown in the page are kept in `errors`."""

    def __init__(self, port: str) -> None:
        self.base = f"http://127.0.0.1:{port}"
        req = urllib.request.Request(f"{self.base}/json/new?about:blank", method="PUT")
        with urllib.request.urlopen(req, timeout=10) as r:
            self.target = json.load(r)
        self.ws = WebSocket(self.target["webSocketDebuggerUrl"])
        self.n = 0
        self.errors: list[str] = []
        self.call("Runtime.enable")
        self.call("Page.enable")
        self.call("Network.clearBrowserCookies")   # every test server is 127.0.0.1, whatever its port

    def call(self, method: str, **params):
        self.n += 1
        mid = self.n
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == mid:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})
            if m.get("method") == "Runtime.exceptionThrown":
                d = m["params"]["exceptionDetails"]
                self.errors.append((d.get("exception") or {}).get("description") or d.get("text", ""))

    def eval(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise AssertionError(f"{expr!r} threw: {r['exceptionDetails']}")
        return r["result"].get("value")

    def wait(self, expr: str, timeout: float = 20.0):
        deadline = time.monotonic() + timeout
        while True:
            v = self.eval(expr)
            if v:
                return v
            if time.monotonic() > deadline:
                raise AssertionError(f"timed out waiting for {expr}")
            time.sleep(0.1)

    def goto(self, url: str) -> None:
        self.call("Page.navigate", url=url)
        path = urllib.parse.urlparse(url).path   # the page may drop the fragment (the sign-in link does)
        self.wait(f"location.pathname === {json.dumps(path)} && document.readyState === 'complete'")

    def close(self) -> None:
        self.ws.close()
        try:
            urllib.request.urlopen(f"{self.base}/json/close/{self.target['id']}", timeout=10).close()
        except OSError:
            pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(HAVE_WEB and uvicorn is not None and CDP_PORT,
                     "set E2E_CDP_PORT to a Chrome remote-debugging port (and install requirements-dev.txt)")
class ServiceBrowserCase(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        port = free_port()
        self.url = f"http://127.0.0.1:{port}"
        self.s = settings(self.tmp, base_url=self.url, allowed_hosts=("127.0.0.1",), **getattr(self, "service_settings", {}))
        self.store = Store(self.s.meta_path, max_running=self.s.max_running_jobs)
        self.store.generation_update(actor='fixture', enabled=True, limit_usd=1000, max_running=self.s.max_running_jobs)
        self.sender = FakeSender()
        self.runner = GatedRunner()
        storage = Storage(self.s.data_dir)
        self.backend = ThreadBackend(self.store, self.s, storage, self.runner)
        lim = Limits(self.s.upload_max_bytes, self.s.upload_max_pixels, self.s.upload_max_side, self.s.upload_min_side)
        svc = Services(settings=self.s, store=self.store, signer=Signer(self.s.signing_key), storage=storage,
                       sender=self.sender, sanitize=lambda b: sanitize(b, lim), submit=self.backend.submit)
        self.server = uvicorn.Server(uvicorn.Config(create_app(svc), host="127.0.0.1", port=port,
                                                    log_level="warning", proxy_headers=False))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.server.started:
            if time.monotonic() > deadline or not self.thread.is_alive():
                self.fail("the test server did not start")
            time.sleep(0.05)
        self.tab = Tab(CDP_PORT)

    def tearDown(self) -> None:
        self.tab.close()
        self.runner.gate.set()
        self.backend.wait(30)
        self.server.should_exit = True
        self.thread.join(15)
        self.store.close()
        super().tearDown()

    def signin(self, email: str = "alice@example.com") -> None:
        t = self.tab

        if email not in self.s.admin_emails:
            t.goto(self.url + '/')
            t.wait("!document.querySelector('#entry').hidden && !document.querySelector('#headerSignIn').disabled")
            t.eval("fetch('/api/session').then(r=>r.json()).then(s=>fetch('/api/demo/access',{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':s.csrf},body:'{}'})).then(r=>r.json())")
            t.goto(self.url + '/')
            t.wait("!document.querySelector('#app').hidden")
            return
        t.goto(self.url + "/admin/signin")
        t.wait("document.querySelector('#signinDialog').open")
        t.eval(f"document.querySelector('#email').value = {json.dumps(email)}; "
               "document.querySelector('#signinForm').requestSubmit()")
        t.wait("/link/i.test(document.querySelector('#signinMsg').textContent)")
        deadline = time.monotonic() + 10
        while not self.sender.sent and time.monotonic() < deadline:   # the email may go out after the answer
            time.sleep(0.05)
        token = re.search(r"/signin#t=([A-Za-z0-9_-]{43})", self.sender.sent[-1].text).group(1)
        t.goto(f"{self.url}/admin/signin#t={token}")
        t.wait("!document.querySelector('#verify').hidden && /HeroGen account|Sign in as/"
               ".test(document.querySelector('#verifyWho').textContent)")
        t.eval("document.querySelector('#verifyForm').requestSubmit()")
        t.wait(f"location.href === {json.dumps(self.url + '/admin#settings')} && document.readyState === 'complete' && "
               "!document.querySelector('#adminShell').hidden")

    def screenshot(self, name, *, viewport=False):
        folder = os.environ.get('HEROGEN_TEST_SHOTS')
        if folder:
            from pathlib import Path
            target = Path(folder)
            target.mkdir(parents=True, exist_ok=True)
            if viewport:
                shot = self.tab.call('Page.captureScreenshot', format='png', captureBeyondViewport=False)
            else:
                metrics = self.tab.call('Page.getLayoutMetrics')['cssContentSize']
                shot = self.tab.call('Page.captureScreenshot', format='png', captureBeyondViewport=True,
                                     clip={**metrics, 'scale':1})
            (target / name).write_bytes(base64.b64decode(shot['data']))

    def make_hero(self, name: str) -> None:
        """Fill the form with a sketch and a name and press Make (the page then opens the new job)."""
        data = base64.b64encode(png(400, 600)).decode()
        self.tab.eval(f"""(() => {{
            const bytes = Uint8Array.from(atob('{data}'), (c) => c.charCodeAt(0));
            const dt = new DataTransfer();
            dt.items.add(new File([bytes], 'sketch.png', {{ type: 'image/png' }}));
            document.querySelector('#sketch').files = dt.files;
            document.querySelector('#sketch').dispatchEvent(new Event('change'));
            document.querySelector('#name').value = {json.dumps(name)};
            document.querySelector('#jobForm').requestSubmit();
        }})()""")
        self.tab.wait(f"document.querySelector('#detailH').textContent === {json.dumps(name)}")

class TestServicePage(ServiceBrowserCase):

    def test_degraded_hero_shows_model_download_and_notes(self):
        self.runner.fail = {'3f': 'error', '4': 'reject'}
        self.runner.gate.set()
        self.signin()
        self.make_hero('Rough and ready')
        self.tab.wait("document.querySelector('#generationTitle').textContent === 'Meet your hero.'", timeout=60)
        self.tab.wait("!document.querySelector('#previewWrap').hidden && !!document.querySelector('#previewWrap canvas')", timeout=60)
        self.tab.wait("!document.querySelector('#orbitHint').hidden", timeout=60)
        self.tab.wait("/not available for this hero/.test(document.querySelector('#exportStatus').textContent)", timeout=60)
        self.assertTrue(self.tab.eval("[...document.querySelectorAll('#downloads a')].some(a => a.textContent === 'Download GLB')"))
        self.assertIn('unrefined texture', self.tab.eval("document.querySelector('#stageIssues').textContent"))
        self.assertTrue(self.tab.eval("document.querySelector('#previewWrap canvas').width > 0"))
        self.assertFalse(self.tab.eval("[...document.querySelectorAll('#downloads a')].some(a => /Roblox/.test(a.textContent))"))
        status = self.tab.eval("fetch(document.querySelector('#downloads a').href).then(r=>r.status)")
        self.assertEqual(status, 200)
        self.screenshot('degraded-workbench.png')
        self.assertEqual(self.tab.errors, [])

    def test_model_download_is_ready_while_exports_are_preparing(self):
        gate, waiting = threading.Event(), threading.Event()
        original = self.runner.run
        def delayed(spec, ctx):
            if spec.key == '4':
                waiting.set()
                if not gate.wait(30):
                    raise RuntimeError('test export gate timed out')
            return original(spec, ctx)
        self.runner.run = delayed
        self.runner.gate.set()
        try:
            self.signin()
            self.make_hero('Ready before exports')
            self.assertTrue(waiting.wait(10))
            self.tab.wait("document.querySelector('#generationTitle').textContent === 'Your hero. Your next idea.'", timeout=20)
            self.assertEqual(self.tab.eval("document.querySelector('#exportStatus').textContent"), 'Roblox files: being prepared')
            self.assertEqual(self.tab.eval("fetch(document.querySelector('#downloads a').href).then(r=>r.status)"), 200)
            self.assertTrue(self.tab.eval("!document.querySelector('#modelTab').hidden"))
        finally:
            gate.set()
        self.tab.wait("document.querySelector('#exportStatus').textContent === 'Roblox files: ready'", timeout=30)

    def test_progressive_shape_versions_interaction_camera_and_hidden_polling(self):
        shape, version_one, version_two = (threading.Event() for _ in range(3))
        publish, refine = self.runner.publish_version, self.runner._stage_3f
        waiting = threading.Event()
        def stage(ctx, run):
            (run / 'stage-3f-progress.json').write_text('{')
            waiting.set()
            if not shape.wait(60):
                raise RuntimeError('shape publication gate timed out')
            return refine(ctx, run)
        def version(run, versions, n, accepted):
            publish(run, versions, n, accepted)
            if not (version_one if n == 1 else version_two).wait(60):
                raise RuntimeError('version publication gate timed out')
        self.runner._stage_3f = stage
        self.runner.publish_version = version
        self.runner.gate.set()
        try:
            self.signin()
            t = self.tab
            t.eval("""(() => {
                window.modelReads=[];const original=window.fetch;
                window.fetch=(url, options)=>{const path=typeof url==='string'?url:url.url;
                    if(path?.includes('.glb')) window.modelReads.push(path);return original(url,options);};
            })()""")
            self.make_hero('A hero taking shape')
            self.assertTrue(waiting.wait(15))
            t.wait("!document.querySelector('#modelTab').hidden && document.querySelector('#artifactLabel').textContent === 'First 3D shape'")
            t.wait("!document.querySelector('#orbitHint').hidden", timeout=45)
            self.assertEqual(t.eval("document.querySelector('#modelNote').textContent"), 'The texture is still being refined.')
            self.assertTrue(t.eval("document.querySelector('#modelUpdate').hidden"))
            self.assertIn('elapsed', t.eval("document.querySelector('[data-stage=\"3f\"] .step-duration').textContent"))
            self.screenshot('progressive-first-shape.png')
            shape.set()
            t.wait("document.querySelector('#artifactLabel').textContent === 'Refined version 1' && !document.querySelector('#orbitHint').hidden")
            t.wait("window.modelReads.some(u=>u.includes('03-refined-v1.glb'))")
            self.screenshot('progressive-version-1.png')
            t.eval("document.querySelector('#preview').scrollIntoView({block:'center',behavior:'instant'})")
            box = t.eval("(() => {const b=document.querySelector('#preview').getBoundingClientRect();return {x:b.x+b.width/2,y:b.y+b.height/2};})()")
            self.assertTrue(t.eval(f"document.querySelector('#preview').contains(document.elementFromPoint({box['x']},{box['y']}))"))
            t.call('Input.dispatchMouseEvent', type='mousePressed', x=box['x'], y=box['y'], button='left', buttons=1, clickCount=1)
            t.call('Input.dispatchMouseEvent', type='mouseMoved', x=box['x'] + 90, y=box['y'] + 35, button='left', buttons=1)
            t.call('Input.dispatchMouseEvent', type='mouseReleased', x=box['x'] + 90, y=box['y'] + 35, button='left', buttons=0)
            version_one.set()
            t.wait("!document.querySelector('#modelUpdate').hidden")
            self.assertEqual(t.eval("document.querySelector('#artifactLabel').textContent"), 'Refined version 1')
            self.assertFalse(t.eval("window.modelReads.some(u=>u.includes('03-refined-v2.glb'))"))
            self.assertEqual(t.eval("document.querySelector('#modelUpdateNote').textContent"), '2 more views added')
            self.screenshot('progressive-update-ready.png')
            self.count_polls()
            # Simulate a hidden document and exercise the same visibility handler used by a background tab.
            t.eval("Object.defineProperty(document,'hidden',{configurable:true,value:true});document.dispatchEvent(new Event('visibilitychange'))")
            t.eval(f'new Promise(r=>setTimeout(r,{int((POLL_S + .5) * 1000)}))')
            self.assertEqual(t.eval('window.__polls'), 0)
            t.eval("delete document.hidden;document.dispatchEvent(new Event('visibilitychange'))")
            t.wait('window.__polls > 0')
            t.eval("document.querySelector('#orbitToggle').click()")  # the idle orbit would move the camera
            camera = t.eval("import('/assets/preview.js').then(p=>p.view())")
            t.eval("document.querySelector('#updateModel').click()")
            t.wait("document.querySelector('#artifactLabel').textContent === 'Refined version 2' && !document.querySelector('#orbitHint').hidden")
            t.wait("window.modelReads.some(u=>u.includes('03-refined-v2.glb'))")
            updated_camera = t.eval("import('/assets/preview.js').then(p=>p.view())")
            for key in ('position', 'target'):
                for before, after in zip(camera[key], updated_camera[key]):
                    self.assertAlmostEqual(before, after, delta=.001)  # OrbitControls may finish damping.
            self.assertEqual(camera['zoom'], updated_camera['zoom'])
            t.eval("import('/assets/analytics.js').then(a=>a.flush())")
            recorded = self.store._all("SELECT event,prop_val FROM analytics_events WHERE event IN ('model_first_shown','refined_version_shown','update_clicked')")
            self.assertCountEqual([(r['event'], r['prop_val']) for r in recorded],
                                  [('model_first_shown', ''), ('refined_version_shown', '1'),
                                   ('refined_version_shown', '2'), ('update_clicked', '2')])
            version_two.set()
            t.wait("document.querySelector('#state').textContent === 'succeeded'", timeout=30)
            self.assertEqual(t.eval("document.querySelector('#artifactLabel').textContent"), 'Refined version 2')
            self.assertTrue(t.eval("document.querySelector('#modelUpdate').hidden"))
            self.assertEqual(t.errors, [])
        finally:
            shape.set(); version_one.set(); version_two.set()

    def test_typical_stage_hints_appear_only_after_three_finished_runs(self):
        self.runner.gate.set()
        self.signin()
        t = self.tab
        for n in range(1, 4):
            self.make_hero(f'Measured hero {n}')
            t.wait("document.querySelector('#exportStatus').textContent === 'Roblox files: ready'", timeout=30)
            timing = t.eval("document.querySelector('[data-stage=\"1\"] .step-duration').textContent")
            self.assertNotIn('recorded', timing)
            self.assertIn('$0.01', timing)
            self.assertEqual('usually about' in timing, n >= 3)
            if n < 3:
                t.eval("document.querySelector('#backToIdeas').click()")

    def test_entry_rejects_excess_prompt_and_focuses_how_section(self):
        t = self.tab
        t.goto(self.url + '/')
        t.wait("!document.querySelector('#entry').hidden")
        t.eval("document.querySelector('#prompt').value = 'x'.repeat(301); document.querySelector('#prompt').dispatchEvent(new Event('input')); document.querySelector('#jobForm').requestSubmit()")
        t.wait("/within 300/.test(document.querySelector('#jobMsg').textContent)")
        self.assertFalse(t.eval("document.querySelector('#signinDialog').open"))
        self.assertEqual(t.eval("document.querySelector('#prompt').getAttribute('aria-invalid')"), 'true')
        t.eval("document.querySelector('[data-idea=moon]').click()")
        self.assertEqual(t.eval("document.querySelector('#prompt').getAttribute('aria-invalid')"), 'false')
        t.eval("document.querySelector('#howButton').click()")
        self.assertTrue(t.eval("document.activeElement.id === 'howHeading'"))
        self.assertFalse(t.eval("document.querySelector('#howDialog').open"))


    def test_generation_navigation_artifact_selection_and_developer_mode(self):
        self.signin()
        t = self.tab
        t.eval("""(() => {
            window.inspectionReads=0; const original=window.fetch;
            window.fetch=(url,options)=>{if(String(url).startsWith('/api/runs/')) window.inspectionReads++; return original(url,options);};
        })()""")
        self.make_hero('Workbench hero')
        self.assertTrue(t.eval("document.querySelector('#entry').hidden"))
        self.assertIn('job=', t.eval('location.hash'))
        t.wait("document.querySelector('#images button') !== null")
        t.eval("document.querySelector('#images button').click()")
        selected = t.eval("document.querySelector('#artboardImage').getAttribute('src')")
        self.assertEqual(t.eval('window.inspectionReads'), 0)
        t.eval("document.querySelector('#developerMode').click()")
        t.wait("!document.querySelector('#developerPanel').hidden && window.inspectionReads > 0")
        self.runner.gate.set()
        t.wait("document.querySelector('#state').textContent === 'succeeded'", timeout=60)
        self.assertEqual(t.eval("document.querySelector('#artboardImage').getAttribute('src')"), selected)
        self.assertTrue(t.eval("document.querySelector('#previewWrap').hidden"))
        self.assertEqual(t.eval("document.querySelector('#generationTitle').textContent"), 'Your hero. Your next idea.')
        t.eval("document.querySelector('#modelTab').click()")
        self.assertFalse(t.eval("document.querySelector('#previewWrap').hidden"))
        t.eval("document.querySelector('#artTab').click()")
        self.assertTrue(t.eval("document.querySelector('#previewWrap').hidden"))
        saved_hash = t.eval('location.hash')
        t.eval("document.querySelector('#backToIdeas').click()")
        t.wait("!document.querySelector('#entry').hidden && document.querySelector('#detail').hidden")
        self.assertTrue(t.eval("document.querySelector('#developerPanel').hidden"))
        t.goto(self.url + '/' + saved_hash)
        t.wait("!document.querySelector('#detail').hidden && document.querySelector('#entry').hidden")
        self.assertEqual(t.eval("document.querySelector('#detailH').textContent"), 'Workbench hero')
        t.eval("document.querySelector('#backToIdeas').click(); document.querySelector('#showLibrary').click()")
        t.eval(r"""(() => {
            const original=window.fetch;
            window.fetch=async (url, options)=>{
                const response=await original(url,options);
                if(/^\/api\/jobs\/[^/]+$/.test(String(url))) await new Promise(resolve=>window.releaseJob=resolve);
                return response;
            };
            document.querySelector('#jobs a').click();
        })()""")
        t.wait("typeof window.releaseJob === 'function'")
        t.eval("document.querySelector('#backToIdeas').click(); window.releaseJob()")
        t.eval("new Promise(resolve=>setTimeout(resolve,100))")
        self.assertFalse(t.eval("document.querySelector('#entry').hidden"))
        self.assertTrue(t.eval("document.querySelector('#detail').hidden"))

    def test_confirmed_creation_still_opens_when_library_refresh_fails(self):
        self.signin()
        t = self.tab
        t.eval("""(() => {
            const original=window.fetch; window.submittedJobs=0;
            window.fetch=async (url,options)=>{
                if(url === '/api/jobs' && options?.method === 'POST') {
                    window.submittedJobs++; return original(url,options);
                }
                if(url === '/api/jobs' && window.submittedJobs) return new Response(JSON.stringify({error:{message:'Library temporarily unavailable'}}),{status:503});
                return original(url,options);
            };
        })()""")
        self.make_hero('Saved despite refresh')
        t.wait("/Library temporarily unavailable/.test(document.querySelector('#banner').textContent)")
        self.assertTrue(t.eval("document.querySelector('#entry').hidden"))
        self.assertEqual(t.eval('window.submittedJobs'),1)
        self.assertIsNone(t.eval("sessionStorage.getItem('hg-pending-job')"))

    def test_failed_image_retries_and_cannot_overlay_the_model(self):
        from starlette.responses import Response
        from starlette.routing import Route
        available = threading.Event()
        reads = []
        async def image(request):
            reads.append(request.url.path)
            return Response(png(400, 600), media_type='image/png') if available.is_set() else Response(status_code=503)
        app = self.server.config.app
        while not hasattr(app, 'routes'):
            app = app.app
        app.routes.insert(0, Route('/ui-test-image', image))
        self.signin()
        t = self.tab
        t.eval(r"""(() => {
            window.imageVersion=0; const original=window.fetch;
            window.fetch=async (url,options)=>{
                const response=await original(url,options);
                if((url === '/api/jobs' && options?.method === 'POST') || /^\/api\/jobs\/[^/]+$/.test(String(url))) {
                    const job=await response.json();
                    job.images.sketch='/ui-test-image?v='+window.imageVersion;
                    return new Response(JSON.stringify(job),{status:response.status});
                }
                return response;
            };
        })()""")
        self.make_hero('Image recovery')
        t.wait("!document.querySelector('#imageError').hidden")
        available.set()
        t.eval("document.querySelector('#images button[data-image=sketch]').click()")
        t.wait("document.querySelector('#artboardImage').naturalWidth > 0 && document.querySelector('#imageError').hidden")
        self.assertGreaterEqual(len(reads),2)
        available.clear()
        t.eval('window.imageVersion++')
        t.wait("!document.querySelector('#imageError').hidden",timeout=15)
        self.runner.gate.set()
        t.wait("!document.querySelector('#modelTab').hidden",timeout=60)
        t.eval("document.querySelector('#modelTab').click(); document.querySelector('#artboardImage').dispatchEvent(new Event('error'))")
        self.assertTrue(t.eval("document.querySelector('#imageError').hidden"))
        self.assertIsNone(t.eval("document.querySelector('#artboardImage').getAttribute('src')"))


    def count_polls(self) -> None:
        """From now on, count the page's reads of one job (GET /api/jobs/<id>) in window.__polls."""
        self.tab.eval("""(() => {
            window.__polls = 0;
            const f = window.fetch;
            window.fetch = (u, o) => {
                if (/^\\/api\\/jobs\\/[^/]+$/.test(String(u)) && (!o || !o.method || o.method === 'GET')) window.__polls++;
                return f(u, o);
            };
        })()""")

    def stage_states(self) -> dict:
        return json.loads(self.tab.eval(
            "JSON.stringify(Object.fromEntries([...document.querySelectorAll('#stages li')].map((li) => "
            "[li.querySelector('b').textContent, li.className.replace('st-', '')])))"))

    def test_a_job_canceled_mid_stage_stops_polling(self):
        self.signin()
        self.count_polls()
        self.make_hero("Sir Cancel")
        self.assertTrue(self.runner.waiting.wait(20), "stage 1 never started")
        self.tab.wait("document.querySelector('#state').textContent === 'running' && "
                      "document.querySelector('.st-running') !== null")
        self.tab.wait("window.__polls >= 1", timeout=POLL_S * 3)   # it polls while the job runs
        self.tab.eval("document.querySelector('#cancelJob').click()")
        self.tab.wait("document.querySelector('#state').textContent === 'canceled'")
        time.sleep(POLL_S + 1.5)   # a poll already scheduled before the cancel runs once more, then stops
        settled = self.tab.eval("window.__polls")
        time.sleep(2 * POLL_S + 1)
        self.assertEqual(self.tab.eval("window.__polls"), settled, "the page kept polling a canceled job")
        self.assertEqual(self.stage_states()["Artwork"], "canceled")
        self.assertTrue(self.tab.eval("document.querySelector('#cancelJob').hidden"))
        self.assertEqual(self.tab.errors, [])

    def test_a_finished_job_stops_polling_once_its_quality_check_ends(self):
        self.runner.gate.set()
        self.signin()
        self.count_polls()
        self.make_hero("Sir Finish")
        self.tab.wait("document.querySelector('#state').textContent === 'succeeded'", timeout=60)
        self.tab.wait("!document.querySelector('.st-running')", timeout=60)
        states = self.stage_states()
        self.assertIsNone(self.tab.eval("document.querySelector('#qualityReview')"))
        self.assertTrue(self.tab.eval("[...document.querySelectorAll('#downloads a')].some((a) => "
                                      "a.textContent === 'Download GLB')"))
        time.sleep(POLL_S + 1.5)
        settled = self.tab.eval("window.__polls")
        time.sleep(2 * POLL_S + 1)
        self.assertEqual(self.tab.eval("window.__polls"), settled, "the page kept polling a finished job")
        self.assertEqual(self.tab.errors, [])


@unittest.skipUnless(HAVE_WEB and uvicorn is not None and CDP_PORT, "set E2E_CDP_PORT and install requirements-dev.txt")
class TestPublicLivePage(ServiceBrowserCase):
    service_settings = {'public_demo': True, 'code_commit': 'a'*40}

    def test_create_opens_guest_and_starts_without_an_account_or_code(self):
        t=self.tab; t.goto(self.url+'/')
        t.wait("!document.querySelector('#headerSignIn').disabled")
        t.eval("document.querySelector('[data-idea=forest]').click(); document.querySelector('#jobForm').requestSubmit(); document.querySelector('#jobForm').requestSubmit()")
        t.wait("!document.querySelector('#detail').hidden")
        self.assertEqual(len(self.store.users_list()),1)
        self.assertEqual(len(self.store.jobs_admin_list()),1)
        self.assertEqual(self.store.users_list()[0]['kind'],'guest')
        self.runner.gate.set()
        t.wait("document.querySelector('#state').textContent === 'succeeded'",timeout=60)
        t.goto(self.url+'/')
        t.wait("document.querySelector('#who').textContent === 'Private workspace'")
        self.assertTrue(t.eval("document.querySelector('#guestAccessSummary').hidden"))
        self.assertEqual(len(self.store.users_list()),1)
        t.eval("document.querySelector('[data-idea=moon]').click(); document.querySelector('#jobForm').requestSubmit()")
        t.wait("!document.querySelector('#detail').hidden")
        self.assertEqual(len(self.store.jobs_admin_list()),2)
        self.assertEqual(self.tab.errors,[])

    def test_queue_waiting_state_and_workers_start_in_order(self):
        self.store.generation_update(actor='fixture', max_running=1)
        self.signin()
        self.make_hero('First hero')
        self.assertTrue(self.runner.waiting.wait(10))
        self.tab.eval("document.querySelector('#backToIdeas').click()")
        self.make_hero('Queued hero')
        self.tab.wait("document.querySelector('#generationTitle').textContent === 'Waiting for a free worker'")
        self.assertEqual([j['state'] for j in self.store.jobs_admin_list()], ['queued','running'])
        self.assertIsNone(self.tab.eval("document.querySelector('#qualityReview')"))
        self.screenshot('queued-workbench.png')
        self.runner.gate.set()
        self.tab.wait("document.querySelector('#state').textContent === 'succeeded'", timeout=60)
        self.assertEqual([j['state'] for j in self.store.jobs_admin_list()], ['succeeded','succeeded'])
        self.assertEqual(self.tab.errors, [])

    def test_same_click_network_retry_uses_one_key_and_creates_one_hero(self):
        self.signin()
        self.tab.eval("""(() => {
          const original = window.fetch; window.createKeys = [];
          window.fetch = async (url, options) => {
            const response = await original(url, options);
            if (url === '/api/jobs' && options?.method === 'POST') {
              window.createKeys.push(options.headers['Idempotency-Key']);
              if (window.createKeys.length === 1) throw new TypeError('Synthetic lost response');
            }
            return response;
          };
        })()""")
        self.make_hero('One click')
        keys = self.tab.eval('window.createKeys')
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(len(self.store.jobs_admin_list()), 1)

    def test_public_gallery_fragment_scrolls_after_entry_is_loaded(self):
        t=self.tab; t.goto(self.url+'/#gallery')
        t.wait("!document.querySelector('#headerSignIn').disabled && window.scrollY > 0 && document.querySelector('#gallery').getBoundingClientRect().top < window.innerHeight")
        self.assertEqual(self.store.users_list(),[])

    def test_unfunded_public_generation_preserves_draft_with_clear_error(self):
        self.store.generation_update(actor='fixture', enabled=False)
        t=self.tab; t.goto(self.url+'/')
        t.wait("!document.querySelector('#headerSignIn').disabled")
        t.eval("document.querySelector('[data-idea=forest]').click(); document.querySelector('#jobForm').requestSubmit()")
        t.wait("/paused right now/.test(document.querySelector('#jobMsg').textContent)")
        self.assertEqual(len(self.store.jobs_admin_list()),0)
        self.assertIn('forest courier',t.eval("document.querySelector('#prompt').value"))


if __name__ == "__main__":
    unittest.main()
