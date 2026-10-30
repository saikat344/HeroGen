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


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(HAVE_WEB and uvicorn is not None and CDP_PORT,
                     "set E2E_CDP_PORT to a Chrome remote-debugging port (and install requirements-dev.txt)")
class TestServicePage(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        port = free_port()
        self.url = f"http://127.0.0.1:{port}"
        self.s = settings(self.tmp, base_url=self.url, allowed_hosts=("127.0.0.1",))
        self.store = Store(self.s.meta_path)
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
        t.goto(self.url + "/")
        t.wait("!document.querySelector('#signin').hidden")
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
            document.querySelector('#name').value = {json.dumps(name)};
            document.querySelector('#jobForm').requestSubmit();
        }})()""")
        self.tab.wait(f"document.querySelector('#detailH').textContent === {json.dumps(name)}")

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
        self.assertEqual(self.stage_states()["cover art"], "canceled")
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
        self.assertEqual(states["quality check (held-out views)"], "passed", states)
        self.assertTrue(self.tab.eval("[...document.querySelectorAll('#downloads a')].some((a) => "
                                      "a.textContent === 'Download GLB')"))
        time.sleep(POLL_S + 1.5)
        settled = self.tab.eval("window.__polls")
        time.sleep(2 * POLL_S + 1)
        self.assertEqual(self.tab.eval("window.__polls"), settled, "the page kept polling a finished job")
        self.assertEqual(self.tab.errors, [])


if __name__ == "__main__":
    unittest.main()
