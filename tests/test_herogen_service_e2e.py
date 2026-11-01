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
        self.store = Store(self.s.meta_path)
        if self.s.public_demo:
            self.store.budget_configure(100_000_000, 10_000_000, 1, time.time()+86400*14, self.s.code_commit,
                'Synthetic browser fixture; no paid work', self.s.cost_signature, provider_caps_verified=True)
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
        self.store.allow_add(email)
        t.goto(self.url + "/admin/signin")
        t.wait("document.querySelector('#signinDialog').open")
        t.eval(f"document.querySelector('#email').value = {json.dumps(email)}; "
               "document.querySelector('#signinForm').requestSubmit()")
        t.wait("/link/i.test(document.querySelector('#signinMsg').textContent)")
        deadline = time.monotonic() + 10
        while not self.sender.sent and time.monotonic() < deadline:   # the email may go out after the answer
            time.sleep(0.05)
        token = re.search(r"/signin#t=([A-Za-z0-9_-]{43})", self.sender.sent[-1].text).group(1)
        t.goto(f"{self.url}/signin#t={token}")
        t.wait("!document.querySelector('#verify').hidden && /HeroGen account|Sign in as/"
               ".test(document.querySelector('#verifyWho').textContent)")
        t.eval("document.querySelector('#adult').checked = true; document.querySelector('#accept').checked = true; "
               "document.querySelector('#verifyForm').requestSubmit()")
        t.wait(f"location.href === {json.dumps(self.url + '/')} && document.readyState === 'complete' && "
               "!document.querySelector('#app').hidden")

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
    def test_public_entry_preserves_image_draft_through_signin(self):
        t = self.tab
        t.goto(self.url + '/')
        t.wait("!document.querySelector('#entry').hidden")
        self.assertTrue(t.eval("document.querySelector('#app').hidden"))
        t.eval("document.querySelector('[data-idea=forest]').click()")
        self.assertIn('forest courier', t.eval("document.querySelector('#prompt').value"))
        data = base64.b64encode(png(400, 600)).decode()
        t.eval(f"""(() => {{
            const bytes = Uint8Array.from(atob('{data}'), c => c.charCodeAt(0));
            const dt = new DataTransfer();
            dt.items.add(new File([bytes], 'draft.png', {{type:'image/png'}}));
            document.querySelector('#sketch').files = dt.files;
            document.querySelector('#sketch').dispatchEvent(new Event('change'));
            document.querySelector('#name').value = 'Saved draft';
            document.querySelector('#jobForm').requestSubmit();
        }})()""")
        t.wait("document.querySelector('#judgeDialog').open")
        self.assertTrue(t.eval("/saved/.test(document.querySelector('#jobMsg').textContent)"))
        self.signin()
        t.wait("document.querySelector('#uploadName').textContent === 'draft.png'")
        self.assertEqual(t.eval("document.querySelector('#name').value"), 'Saved draft')
        self.assertTrue(t.eval("document.querySelector('#promptIn').hidden"))
        t.eval("document.querySelector('#jobForm').requestSubmit()")
        t.wait("document.querySelector('#detailH').textContent === 'Saved draft'")
        self.assertEqual(t.eval("document.querySelectorAll('#jobs li').length"), 1)
        self.assertTrue(t.eval("document.activeElement.id === 'detail'"))

    def test_entry_rejects_excess_prompt_before_auth_and_supports_keyboard_dialog(self):
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
        self.assertTrue(t.eval("document.querySelector('#howDialog').open"))
        t.eval("document.querySelector('#closeHow').click()")
        self.assertFalse(t.eval("document.querySelector('#howDialog').open"))

    def test_signout_clears_the_saved_draft(self):
        t = self.tab
        t.goto(self.url + '/')
        t.wait("!document.querySelector('#entry').hidden")
        t.eval("document.querySelector('[data-idea=forest]').click(); document.querySelector('#jobForm').requestSubmit()")
        t.wait("document.querySelector('#judgeDialog').open")
        self.signin()
        self.assertIn('forest courier', t.eval("document.querySelector('#prompt').value"))
        t.eval("document.querySelector('#signOut').click()")
        t.wait("!document.querySelector('#entry').hidden && !document.querySelector('#headerSignIn').disabled && document.querySelector('#signOut').hidden")
        self.assertEqual(t.eval("document.querySelector('#prompt').value"), '')
        t.goto(self.url + '/')
        t.wait("!document.querySelector('#entry').hidden")
        self.assertEqual(t.eval("document.querySelector('#prompt').value"), '')

    def test_uncertain_request_restores_current_input_and_retires_conflicted_key(self):
        t = self.tab
        t.goto(self.url + '/')
        t.wait("!document.querySelector('#entry').hidden")
        t.eval("document.querySelector('#prompt').value='Original idea'; document.querySelector('#jobForm').requestSubmit()")
        t.wait("document.querySelector('#judgeDialog').open")
        self.signin()
        t.eval("""(() => {
            const original=window.fetch;
            window.fetch=(url, options) => url === '/api/jobs' && options?.method === 'POST'
                ? Promise.resolve(new Response(JSON.stringify({error:{message:'Busy'}}), {status:503}))
                : original(url,options);
            document.querySelector('#prompt').value='Revised emerald knight';
            document.querySelector('#jobForm').requestSubmit();
        })()""")
        t.wait("/busy/.test(document.querySelector('#jobMsg').textContent)")
        t.goto(self.url + '/')
        t.wait("!document.querySelector('#entry').hidden")
        self.assertEqual(t.eval("document.querySelector('#prompt').value"), 'Revised emerald knight')
        self.assertTrue(t.eval("/not confirmed/.test(document.querySelector('#jobMsg').textContent)"))
        t.eval("""(() => {
            const original=window.fetch;
            window.fetch=(url, options) => url === '/api/jobs' && options?.method === 'POST'
                ? Promise.resolve(new Response(JSON.stringify({error:{message:'Conflict'}}), {status:409}))
                : original(url,options);
            document.querySelector('#jobForm').requestSubmit();
        })()""")
        t.wait("/Check Your heroes/.test(document.querySelector('#jobMsg').textContent)")
        t.goto(self.url + '/')
        t.wait("!document.querySelector('#entry').hidden")
        self.assertIsNone(t.eval("sessionStorage.getItem('hg-pending-job')"))
        self.assertEqual(t.eval("document.querySelector('#prompt').value"), 'Revised emerald knight')

    def test_signin_page_header_can_request_another_link(self):
        t = self.tab
        t.goto(self.url + '/signin')
        t.wait("/incomplete/.test(document.querySelector('#verifyWho').textContent)")
        t.eval("document.querySelector('#headerSignIn').click()")
        t.wait("document.querySelector('#signinDialog').open")
        t.eval("document.querySelector('#closeSignin').click(); document.querySelector('#howButton').click()")
        self.assertTrue(t.eval("document.querySelector('#howDialog').open"))

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
        self.assertEqual(t.eval("document.querySelector('#generationTitle').textContent"), 'Meet your hero.')
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
            document.querySelector('#jobs button').click();
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
        t.wait("/Your hero is saved/.test(document.querySelector('#banner').textContent)")
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

    def test_unavailable_quality_review_is_flagged_beside_successful_job(self):
        self.runner.fail['2h']='reject'
        self.runner.gate.set()
        self.signin()
        self.make_hero('Review warning')
        t = self.tab
        t.wait("document.querySelector('#state').textContent === 'succeeded'",timeout=60)
        t.wait("/Unavailable/.test(document.querySelector('#qualityStatus').textContent)",timeout=60)
        self.assertTrue(t.eval("document.querySelector('#qualityReview').classList.contains('review-attention')"))
        self.assertIn('attention',t.eval("document.querySelector('#generationSubtitle').textContent"))
        self.assertEqual(t.eval("document.querySelector('#generationTitle').textContent"),'Meet your hero.')

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
        self.assertIn("Passed", self.tab.eval("document.querySelector('#qualityStatus').textContent"))
        self.assertTrue(self.tab.eval("[...document.querySelectorAll('#downloads a')].some((a) => "
                                      "a.textContent === 'Download GLB')"))
        time.sleep(POLL_S + 1.5)
        settled = self.tab.eval("window.__polls")
        time.sleep(2 * POLL_S + 1)
        self.assertEqual(self.tab.eval("window.__polls"), settled, "the page kept polling a finished job")
        self.assertEqual(self.tab.errors, [])


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_WEB and uvicorn is not None and CDP_PORT, "set E2E_CDP_PORT and install requirements-dev.txt")
class TestPublicLivePage(ServiceBrowserCase):
    service_settings = {'public_demo': True, 'code_commit': 'a'*40}

    def test_no_account_live_entry_preserves_draft_and_allowance_on_reload(self):
        t = self.tab
        t.goto(self.url + '/')
        t.wait("document.querySelector('#headerSignIn').textContent === 'Try live demo'")
        t.eval("document.querySelector('[data-idea=forest]').click(); document.querySelector('#jobForm').requestSubmit()")
        t.wait("document.querySelector('#judgeDialog').open")
        self.assertFalse(t.eval("document.querySelector('#judgeCode').required"))
        self.assertTrue(t.eval("document.querySelector('#judgeCodeFields').hidden"))
        t.eval("document.querySelector('#judgeForm').requestSubmit()")
        t.wait("!document.querySelector('#judgeDialog').open && !document.querySelector('#app').hidden")
        self.assertIn('forest courier', t.eval("document.querySelector('#prompt').value"))
        self.assertEqual(len(self.store.judge_codes_list()), 1)
        t.eval("document.querySelector('#jobForm').requestSubmit()")
        t.wait("!document.querySelector('#detail').hidden")
        self.runner.gate.set()
        t.wait("document.querySelector('#state').textContent === 'succeeded'", timeout=60)
        t.goto(self.url + '/')
        t.wait("/0 generations left/.test(document.querySelector('#judgeAccessText').textContent)")
        self.assertEqual(len(self.store.judge_codes_list()), 1)
        t.eval("document.querySelector('[data-idea=moon]').click(); document.querySelector('#jobForm').requestSubmit()")
        t.wait("/used its generation allowance/.test(document.querySelector('#jobMsg').textContent)")
        self.assertEqual(len(self.store.jobs_admin_list()), 1)

    def test_unfunded_public_entry_has_clear_recovery(self):
        self.store.db.execute('UPDATE admission_policy SET enabled=0')
        t = self.tab; t.goto(self.url + '/')
        t.wait("!document.querySelector('#headerSignIn').disabled")
        t.eval("document.querySelector('#headerSignIn').click()")
        t.wait("document.querySelector('#judgeDialog').open")
        t.eval("document.querySelector('#judgeForm').requestSubmit()")
        t.wait("/unavailable/.test(document.querySelector('#judgeMsg').textContent)")
        self.assertTrue(t.eval("document.querySelector('#judgeDialog').open"))
        self.assertEqual(self.store.judge_codes_list(), [])
