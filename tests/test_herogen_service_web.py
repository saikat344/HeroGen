"""The HTTP layer end to end (Starlette's test client, stand-ins for email and workers): security headers, host
check, CSRF, admin sign-in by link, rate limits, sessions, uploads, idempotency, downloads, per-user
isolation, path safety, deletion, the Lab API and the legal pages."""

from __future__ import annotations

import io
import json
import os
import re
import sys
import zipfile
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from test_herogen_service_support import HAVE_WEB, Clock, TempDirCase, png, settings, web_test

if HAVE_WEB:
    from starlette.testclient import TestClient

    from herogen.service.auth import Signer
    from herogen.service.fakes import FakeRunner, FakeSender
    from herogen.service.storage import Storage
    from herogen.service.store import Store
    from herogen.service.uploads import Limits, sanitize
    from herogen.service.web import Services, create_app
    from herogen.service.worker import InPlaceWorkdir, make_context, run_group, run_job

ORIGIN = "http://testserver"


class Browser:
    """One browser: its own cookie jar, CSRF token and the service's own Origin on unsafe requests."""

    def __init__(self, app, base=ORIGIN) -> None:
        self.c = TestClient(app, base_url=base)
        self.base = base
        self.csrf = None

    def session(self) -> dict:
        r = self.c.get("/api/session")
        self.csrf = r.json()["csrf"]
        return r.json()

    def send(self, method, path, *, csrf=True, origin=True, headers=None, **kw):
        h = dict(headers or {})
        if origin:
            h.setdefault("Origin", self.base)
        if csrf and self.csrf:
            h.setdefault("X-CSRF-Token", self.csrf)
        return self.c.request(method, path, headers=h, **kw)

    def post(self, path, **kw):
        return self.send("POST", path, **kw)

    def get(self, path, **kw):
        return self.c.get(path, **kw)


@web_test
class WebCase(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        self.build(admin_emails=("alice@example.com",))

    def build(self, **kw) -> None:
        self.clock = Clock()
        self.s = settings(self.tmp, **kw)
        self.store = Store(self.s.meta_path, clock=self.clock, max_running=self.s.max_running_jobs)
        self.store.generation_update(actor="fixture", enabled=True, limit_usd=1000)
        self.sender = FakeSender()
        self.submitted: list[str] = []
        lim = Limits(self.s.upload_max_bytes, self.s.upload_max_pixels, self.s.upload_max_side, self.s.upload_min_side)
        self.svc = Services(settings=self.s, store=self.store, signer=Signer(self.s.signing_key),
                            storage=Storage(self.s.data_dir), sender=self.sender,
                            sanitize=lambda b: sanitize(b, lim),
                            submit=lambda j: self.submitted.append(j) or f"call-{len(self.submitted)}")
        self.app = create_app(self.svc)

    def tearDown(self) -> None:
        self.store.close()
        super().tearDown()

    def link_token(self) -> str:
        m = re.search(r"/signin#t=([A-Za-z0-9_-]{43})", self.sender.sent[-1].text)
        self.assertIsNotNone(m)
        return m.group(1)

    def signin(self, email="alice@example.com", browser=None) -> Browser:
        b = browser or Browser(self.app)
        from dataclasses import replace
        self.s = replace(self.s, admin_emails=tuple(set(self.s.admin_emails) | {email}))
        self.svc.settings = self.s
        if browser is not None:
            browser.c.app = create_app(self.svc)
        self.app = create_app(self.svc)
        b = browser or Browser(self.app)
        b.session()
        self.assertEqual(b.post("/api/auth/request", json={"email": email}).status_code, 200)
        tok = self.link_token()
        r = b.post("/api/auth/verify", json={"token": tok, "accept_terms": True, "adult": True})
        self.assertEqual(r.status_code, 200, r.text)
        b.csrf = r.json()["csrf"]
        return b

    def make_job(self, b: Browser, key="key-00000001", **form):
        data = {"kind": "sketch", "profile": "roblox-ingame", "name": "Sir Bob", **form}
        files = {"sketch": ("../../etc/passwd.png", png(400, 600), "image/png")} if data["kind"] == "sketch" else None
        return b.post("/api/jobs", data=data, files=files, headers={"Idempotency-Key": key})

    def finish(self, job_id: str, runner=None) -> str:
        runner = runner or FakeRunner()

        def execute(job, attempt, group, resume):
            return run_group(self.store, self.s, job, attempt, group, resume, runner,
                             InPlaceWorkdir(make_context(self.s, self.svc.storage, job)))

        return run_job(self.store, self.s, self.svc.storage, job_id, "test", execute)


class TestHeaders(WebCase):
    def test_security_headers_everywhere(self):
        b = Browser(self.app)
        for path, status in (("/", 200), ("/api/session", 200), ("/nope", 404), ("/assets/app.css", 200),
                             ("/api/jobs", 401), ("/viewer", 200), ("/terms", 404)):
            r = b.get(path)
            self.assertEqual(r.status_code, status, path)
            h = r.headers
            self.assertEqual(h["x-content-type-options"], "nosniff", path)
            self.assertEqual(h["referrer-policy"], "no-referrer", path)
            self.assertEqual(h["x-frame-options"], "DENY", path)
            self.assertEqual(h["cross-origin-opener-policy"], "same-origin", path)
            self.assertIn("frame-ancestors 'none'", h["content-security-policy"], path)
            self.assertIn("no-", h["cache-control"], path)
            self.assertNotIn("server", h, path)
            self.assertNotIn("strict-transport-security", h, "no HSTS over plain http")

    def test_page_csp(self):
        r = Browser(self.app).get("/")
        csp = r.headers["content-security-policy"]
        self.assertNotIn("unsafe-inline", csp)
        self.assertNotIn("unsafe-eval", csp)
        self.assertIn("https://cdn.jsdelivr.net/npm/three@0.160.0/", csp)
        self.assertNotIn("https://cdn.jsdelivr.net ", csp, "only the pinned path, not the whole CDN")
        self.assertRegex(csp, r"script-src 'self' \S+ 'sha256-[A-Za-z0-9+/=]+'")
        self.assertIn("object-src 'none'", csp)
        self.assertIn("base-uri 'none'", csp)
        self.assertNotIn("<script>", r.text, "no inline script besides the hashed import map")
        api = Browser(self.app).get("/api/session").headers["content-security-policy"]
        self.assertIn("default-src 'none'", api)
        self.assertIn("sandbox", api)

    def test_server_errors_carry_the_headers(self):
        b = self.signin()

        def boom(*a, **k):
            raise RuntimeError("boom")

        self.store.jobs_list = boom
        t = TestClient(self.app, base_url=ORIGIN, raise_server_exceptions=False)
        t.cookies = b.c.cookies
        r = t.get("/api/jobs")
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.json()["error"]["code"], "internal")
        self.assertNotIn("boom", r.text)
        for h in ("content-security-policy", "x-content-type-options", "x-frame-options", "referrer-policy",
                  "cache-control"):
            self.assertIn(h, r.headers, h)

    def test_host_must_be_allowed(self):
        r = Browser(self.app).get("/api/session", headers={"Host": "evil.example.com"})
        self.assertEqual(r.status_code, 400)

    def test_https_cookies_and_hsts(self):
        self.store.close()
        self.build(admin_emails=("alice@example.com",), base_url="https://testserver")
        b = Browser(self.app, base="https://testserver")
        r = b.get("/api/session")
        self.assertIn("max-age=63072000", r.headers["strict-transport-security"])
        cookie = r.headers["set-cookie"]
        self.assertTrue(cookie.startswith("__Host-hg_csrf="))
        for flag in ("HttpOnly", "Secure", "Path=/", "SameSite=lax"):
            self.assertIn(flag, cookie)
        self.assertNotIn("Domain", cookie)
        b.csrf = r.json()["csrf"]

        b.post("/api/auth/request", json={"email": "alice@example.com"})
        r = b.post("/api/auth/verify", json={"token": self.link_token(), "accept_terms": True, "adult": True})
        self.assertEqual(r.status_code, 200)
        cookies = r.headers.get_list("set-cookie")
        sess = next(c for c in cookies if c.startswith("__Host-hg_session="))
        for flag in ("HttpOnly", "Secure", "Path=/", "SameSite=lax"):
            self.assertIn(flag, sess)
        self.assertTrue(self.sender.sent[-1].text.count("https://testserver/admin/signin#t="), 1)

    def test_static_assets_are_confined(self):
        b = Browser(self.app)
        self.assertEqual(b.get("/assets/app.js").headers["content-type"], "text/javascript; charset=utf-8")
        for p in ("/assets/../web.py", "/assets/%2e%2e/web.py", "/assets/index.html", "/assets/.hidden",
                  "/assets/app.js/", "/assets/..%2fweb.py", "/static/%2e%2e/server.py", "/static/index.html",
                  "/static/..%2fserver.py", "/static/.hidden", "/static/app.js/"):
            self.assertEqual(b.get(p).status_code, 404, p)

    def test_raster_assets_have_image_mime_and_remain_confined(self):
        from unittest.mock import patch
        from herogen.service import web
        assets = self.tmp / 'assets'
        assets.mkdir()
        (assets / 'sample.png').write_bytes(png(400, 600))
        (assets / 'outside.png').symlink_to(self.tmp / 'not-public.png')
        (self.tmp / 'not-public.png').write_bytes(b'private')
        with patch.object(web, 'STATIC_DIR', assets):
            b = Browser(self.app)
            r = b.get('/assets/sample.png')
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers['content-type'], 'image/png')
            self.assertEqual(b.get('/assets/outside.png').status_code, 404)
            self.assertEqual(b.get('/assets/%2e%2e/not-public.png').status_code, 404)

    def test_the_viewer_is_the_studio_when_it_is_deployed(self):
        from herogen.service import labapi

        b = Browser(self.app)
        studio = labapi.studio_static()
        page = b.get("/viewer")
        if studio is None:
            self.assertEqual(b.get("/static/app.js").status_code, 404, "no Studio: nothing under /static/")
            self.assertIn("HeroGen viewer", page.text)
            return
        self.assertEqual(page.text, (studio / "index.html").read_text(), "the Studio page, unchanged")
        self.assertRegex(page.headers["content-security-policy"], r"script-src 'self' \S+ 'sha256-")
        r = b.get("/static/app.js")
        self.assertEqual((r.status_code, r.headers["content-type"]), (200, "text/javascript; charset=utf-8"))
        self.assertEqual(b.get("/static/fonts/fonts.css").status_code, 200)

    def test_landing_shading_icons_load_without_studio_static(self):
        from unittest.mock import patch

        with patch("herogen.service.labapi.studio_static", return_value=None):
            b = Browser(self.app)
            page = b.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertNotIn("/static/", page.text)
            hrefs = re.findall(r'<button[^>]+data-shade="[^"]+"[^>]*><svg[^>]*><use href="([^"]+)"', page.text)
            self.assertEqual(hrefs, [f"/assets/icons/sprite.svg#i-hg-shade-{m}" for m in ("textured", "clay", "wire")])
            r = b.get("/assets/icons/sprite.svg")
            self.assertEqual((r.status_code, r.headers["content-type"]), (200, "image/svg+xml"))
            for m in ("textured", "clay", "wire"):
                self.assertIn(f'id="i-hg-shade-{m}"', r.text)
            self.assertEqual(b.get("/static/icons/hg-shade-textured.svg").status_code, 404)


class TestCsrf(WebCase):
    def test_unsafe_requests_need_token_and_origin(self):
        b = Browser(self.app)
        b.session()
        body = {"email": "alice@example.com"}
        self.assertEqual(b.post("/api/auth/request", json=body, csrf=False).status_code, 403)
        self.assertEqual(b.post("/api/auth/request", json=body, headers={"X-CSRF-Token": "0" * 64}).status_code, 403)
        self.assertEqual(b.post("/api/auth/request", json=body, headers={"Origin": "https://evil.example.com"}
                                ).status_code, 403)
        self.assertEqual(b.post("/api/auth/request", json=body, headers={"Sec-Fetch-Site": "cross-site"}
                                ).status_code, 403)
        self.assertEqual(b.post("/api/auth/request", json=body, origin=False).status_code, 403, "no Origin at all")
        self.assertEqual(b.post("/api/auth/request", json=body, origin=False,
                                headers={"Sec-Fetch-Site": "same-origin"}).status_code, 200)
        self.assertEqual(b.post("/api/auth/request", json=body).status_code, 200)

    def test_token_is_bound_to_the_browser(self):
        a, b = Browser(self.app), Browser(self.app)
        a.session()
        b.session()
        r = b.post("/api/auth/request", json={"email": "x@example.com"}, headers={"X-CSRF-Token": a.csrf})
        self.assertEqual(r.status_code, 403)

    def test_every_unsafe_route_is_guarded(self):
        b = self.signin()
        job = self.make_job(b).json()
        for method, path in (("POST", "/api/auth/logout"), ("DELETE", "/api/account"), ("POST", "/api/jobs"),
                             ("DELETE", f"/api/jobs/{job['id']}"), ("POST", f"/api/jobs/{job['id']}/cancel"),
                             ("POST", "/api/auth/verify"), ("POST", "/api/auth/link"), ("PUT", "/api/jobs"),
                             ("POST", "/api/events"), ("POST", "/api/admin/generation")):
            r = b.send(method, path, csrf=False, json={})
            self.assertEqual(r.status_code, 403, (method, path))
        self.assertIsNotNone(self.store.job_get(job["id"], self.store.users_list()[0]["id"]))

    def test_signin_rotates_the_csrf_cookie(self):
        b = Browser(self.app)
        b.session()
        before = b.csrf
        self.signin(browser=b)
        self.assertNotEqual(b.csrf, before)
        r = b.post("/api/auth/logout", json={}, headers={"X-CSRF-Token": before})
        self.assertEqual(r.status_code, 403, "the pre-sign-in token is dead")


class TestSignin(WebCase):

    def test_expired_link(self):

        b = Browser(self.app)
        b.session()
        b.post("/api/auth/request", json={"email": "alice@example.com"})
        tok = self.link_token()
        self.clock.advance(self.s.link_ttl_s + 1)
        r = b.post("/api/auth/verify", json={"token": tok, "accept_terms": True, "adult": True})
        self.assertEqual(r.json()["error"]["code"], "invalid_link")
        self.assertEqual(b.post("/api/auth/link", json={"token": tok}).status_code, 400)


    def test_rate_limits(self):
        b = Browser(self.app)
        b.session()
        codes = [b.post("/api/auth/request", json={"email": "alice@example.com"}).status_code for _ in range(6)]
        self.assertEqual(codes, [200] * 5 + [429])
        r = b.post("/api/auth/request", json={"email": "alice@example.com"})
        self.assertGreater(int(r.headers["retry-after"]), 0)
        self.clock.advance(3601)
        self.assertEqual(b.post("/api/auth/request", json={"email": "alice@example.com"}).status_code, 200)

    def test_sessions_cannot_be_forged_and_end_at_logout(self):
        b = self.signin()
        self.assertEqual(b.get("/api/jobs").status_code, 200)
        evil = Browser(self.app)
        evil.c.cookies.set("hg_session", Signer.new_token())
        self.assertEqual(evil.get("/api/jobs").status_code, 401)
        evil.c.cookies.set("hg_session", "../../" + "a" * 40)
        self.assertEqual(evil.get("/api/jobs").status_code, 401)
        stolen = b.c.cookies.get("hg_session")
        self.assertEqual(b.post("/api/auth/logout", json={}).status_code, 200)
        replay = Browser(self.app)
        replay.c.cookies.set("hg_session", stolen)
        self.assertEqual(replay.get("/api/jobs").status_code, 401, "a logged-out session is revoked server-side")


class TestJobsApi(WebCase):
    def test_create_status_finish_download(self):
        b = self.signin()
        r = self.make_job(b)
        self.assertEqual(r.status_code, 201, r.text)
        job = r.json()
        self.assertEqual((job["state"], job["name"], job["kind"]), ("queued", "Sir Bob", "sketch"))
        self.assertEqual(self.submitted, [job["id"]], "dispatched at once")
        self.assertEqual([s["key"] for s in job["stages"]], ["0", "1", "2", "3", "3f", "4", "6", "5"])
        self.assertFalse(job["downloads"]["glb"]["available"])
        self.assertEqual(b.get(f"/api/jobs/{job['id']}/download/glb").status_code, 409)
        self.assertEqual(self.finish(job["id"]), "succeeded")
        view = b.get(f"/api/jobs/{job['id']}").json()
        self.assertEqual(view["state"], "succeeded")
        self.assertTrue(all(s["state"] == "passed" for s in view["stages"] if s["key"] != "2h"))
        self.assertIn("sketch", view["images"])
        self.assertTrue(view["images"]["sketch"].startswith(f"/runs/r1/{job['id']}/"))
        self.assertIn("Placeholder", view["roblox_readme"])
        self.assertEqual(view["viewer"], f"/viewer#run=r1/{job['id']}")
        r = b.get(view["downloads"]["glb"]["url"])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content[:4], b"glTF")
        self.assertEqual(r.headers["content-disposition"], 'attachment; filename="herogen-sir-bob-high.glb"')
        r = b.get(view["downloads"]["roblox"]["url"])
        self.assertEqual(r.headers["content-type"], "application/zip")
        names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
        self.assertIn("06-roblox/bind_meshes.luau", names)
        self.assertIn("04-body/high.glb", names)
        self.assertIn("stage-6.json", names)
        self.assertEqual(b.get(f"/api/jobs/{job['id']}/download/other").status_code, 404)
        self.assertEqual(b.get(f"/api/jobs/{job['id']}/download/glb?lod=../x").status_code, 404)
        listed = b.get("/api/jobs").json()["jobs"]
        self.assertEqual([j["id"] for j in listed], [job["id"]])

    def test_an_accepted_hero_warms_the_3d_models_and_a_page_load_does_not(self):
        warm = mock.Mock()
        self.svc.prewarm = warm
        b = self.signin()
        b.get("/")
        warm.assert_not_called()
        self.assertEqual(self.make_job(b).status_code, 201)
        self.assertEqual(warm.call_count, 1)
        self.assertEqual(self.make_job(b).status_code, 200, "the same Idempotency-Key returns the saved hero")
        self.assertEqual(warm.call_count, 1, "a replayed request starts no second warmup")
        warm.side_effect = RuntimeError("synthetic prewarm failure")
        with self.assertLogs("herogen.service", "ERROR"):
            self.assertEqual(self.make_job(b, key="key-00000002").status_code, 201, "warming never fails a hero")

    def test_prewarm_models_returns_before_the_remote_call_finishes(self):
        import threading
        from herogen.service.web import prewarm_models
        started, release = threading.Event(), threading.Event()
        def slow():
            started.set()
            release.wait(5)
        with mock.patch("herogen.blender.remote.prewarm", slow):
            prewarm_models()
            self.assertTrue(started.wait(5))
            release.set()

    def test_upload_is_cleaned_before_storage(self):
        b = self.signin()
        from PIL import PngImagePlugin

        info = PngImagePlugin.PngInfo()
        info.add_text("GPS", "51.5N 0.1W")
        files = {"sketch": ("x.png", png(400, 600, pnginfo=info), "image/png")}
        r = b.post("/api/jobs", data={"kind": "sketch"}, files=files, headers={"Idempotency-Key": "key-abcdefgh"})
        self.assertEqual(r.status_code, 201)
        uid = self.store.users_list()[0]["id"]
        saved = self.svc.storage.input_path(uid, r.json()["id"]).read_bytes()
        self.assertNotIn(b"51.5N", saved)
        self.assertNotIn(b"tEXt", saved)
        self.assertEqual(list((self.svc.storage.user_dir(uid) / "inputs").iterdir()),
                         [self.svc.storage.input_path(uid, r.json()["id"])], "the client's file name is never used")

    def test_bad_uploads(self):
        b = self.signin()
        h = {"Idempotency-Key": "key-00000009"}
        r = b.post("/api/jobs", data={"kind": "sketch"}, files={"sketch": ("a.png", b"<svg/>", "image/png")}, headers=h)
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (400, "upload_rejected"))
        self.assertIn("PNG, JPEG or WebP", r.json()["error"]["message"])
        r = b.post("/api/jobs", data={"kind": "sketch"}, files={"sketch": ("a.png", png(100, 100), "image/png")},
                   headers=h)
        self.assertIn("at least 256", r.json()["error"]["message"])
        big = png(400, 400) + b"\x00" * (self.s.upload_max_bytes + 300 * 1024)
        r = b.post("/api/jobs", data={"kind": "sketch"}, files={"sketch": ("a.png", big, "image/png")}, headers=h)
        self.assertEqual(r.status_code, 413)
        r = b.post("/api/jobs", data={"kind": "sketch"}, files={"other": ("a", b"a")}, headers=h)
        self.assertEqual(r.json()["error"]["code"], "no_file")
        r = b.post("/api/jobs", data={"kind": "sketch"}, headers=h)
        self.assertEqual(r.json()["error"]["code"], "multipart")
        r = b.post("/api/jobs", data={"kind": "video"}, headers=h, files={"x": ("a", b"a")})
        self.assertEqual(r.json()["error"]["code"], "bad_kind")
        r = b.post("/api/jobs", data={"kind": "prompt", "prompt": "a knight", "profile": "secret"}, headers=h,
                   files={"x": ("a", b"a")})
        self.assertEqual(r.json()["error"]["code"], "bad_profile")
        r = b.post("/api/jobs", json={"kind": "prompt"}, headers=h)
        self.assertEqual(r.status_code, 415)
        r = b.post("/api/jobs", data={"kind": "prompt", "prompt": "a knight"}, files={"x": ("a", b"a")})
        self.assertEqual(r.json()["error"]["code"], "idempotency_key")
        r = b.post("/api/jobs", data={"kind": "prompt", "prompt": "a knight"}, files={"x": ("a", b"a")},
                   headers={"Idempotency-Key": "../../etc"})
        self.assertEqual(r.json()["error"]["code"], "idempotency_key")
        self.assertEqual(self.store.jobs_list(self.store.users_list()[0]["id"]), [])
        uid = self.store.users_list()[0]["id"]
        self.assertFalse((self.svc.storage.user_dir(uid) / "inputs").exists() and
                         any((self.svc.storage.user_dir(uid) / "inputs").iterdir()), "nothing stored")


    def busy_store(self, land: bool):
        """job_create answers busy once: after saving the job (`land`) or without saving it."""
        from herogen.service.store import StoreError

        store, calls = self.store, {"n": 0}

        class Busy:
            def __getattr__(self, name):
                fn = getattr(store, name)
                if name != "job_create":
                    return fn

                def create(**kw):
                    calls["n"] += 1
                    if calls["n"] == 1:
                        if land:
                            fn(**kw)
                        raise StoreError("The service is busy; please try again.", "busy")
                    return fn(**kw)
                return create

        self.svc.store = Busy()

    def test_an_unconfirmed_job_that_was_saved_is_returned_not_charged_twice(self):
        b = self.signin()
        uid = self.store.users_list()[0]["id"]
        self.busy_store(land=True)
        r = self.make_job(b, key="key-busy-0001")
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (503, "busy"))
        self.assertEqual(r.headers["retry-after"], "5")
        self.assertIn("you get that hero, not a second one", r.json()["error"]["message"])
        saved = self.store.jobs_list(uid)
        self.assertEqual(len(saved), 1, "the job was saved")
        self.assertTrue(self.svc.storage.input_path(uid, saved[0]["id"]).is_file(), "and its sketch kept")
        r = self.make_job(b, key="key-busy-0001")              # the page sends the same form and key again
        self.assertEqual((r.status_code, r.json()["id"]), (200, saved[0]["id"]))

        self.assertEqual(len(list((self.svc.storage.user_dir(uid) / "inputs").iterdir())), 1)
        self.assertEqual(self.finish(saved[0]["id"]), "succeeded", "the saved job runs with its sketch")

    def test_an_unconfirmed_job_that_was_not_saved_starts_on_retry(self):
        b = self.signin()
        uid = self.store.users_list()[0]["id"]
        self.busy_store(land=False)
        self.assertEqual(self.make_job(b, key="key-busy-0002").status_code, 503)
        self.assertEqual(self.store.jobs_list(uid), [])
        r = self.make_job(b, key="key-busy-0002")
        self.assertEqual(r.status_code, 201)

        inputs = sorted((self.svc.storage.user_dir(uid) / "inputs").iterdir())
        self.assertEqual(len(inputs), 2, "the first attempt's sketch waits for retention")
        from herogen.service.worker import purge

        orphan = next(p for p in inputs if p.stem != r.json()["id"])
        os.utime(orphan, (1, 1))
        self.assertEqual(purge(self.store, self.s, self.svc.storage)["orphan_inputs"], 1)
        self.assertEqual([p.stem for p in (self.svc.storage.user_dir(uid) / "inputs").iterdir()], [r.json()["id"]])

    def test_busy_anywhere_is_a_503_with_retry_after(self):
        from herogen.service.store import StoreError

        b = self.signin()

        def busy(*a, **k):
            raise StoreError("The service is restarting; please try again.", "busy")

        self.store.jobs_list = busy
        r = b.get("/api/jobs")
        self.assertEqual((r.status_code, r.json()["error"]["code"], r.headers["retry-after"]), (503, "busy", "5"))
        self.assertIn("frame-ancestors 'none'", r.headers["content-security-policy"])


    def test_rejected_job_shows_its_reason(self):
        b = self.signin()
        job = self.make_job(b).json()
        self.finish(job["id"], FakeRunner(fail={"2": "reject"}))
        view = b.get(f"/api/jobs/{job['id']}").json()
        self.assertEqual(view["state"], "rejected")
        self.assertEqual((view["error"]["stage"], view["error"]["stage_label"]), ("2", "turnaround"))
        self.assertIn("did not agree", view["error"]["message"])
        self.assertFalse(view["downloads"]["roblox"]["available"])

    def test_cancel_and_delete(self):
        b = self.signin()
        job = self.make_job(b).json()
        r = b.post(f"/api/jobs/{job['id']}/cancel", json={})
        self.assertEqual(r.json()["state"], "canceled")
        job2 = self.make_job(b, key="key-00000002").json()
        self.finish(job2["id"])
        uid = self.store.users_list()[0]["id"]
        run = self.svc.storage.run_dir(uid, job2["id"])
        self.assertTrue(run.exists())
        self.assertEqual(b.send("DELETE", f"/api/jobs/{job2['id']}", json={}).status_code, 200)
        self.assertFalse(run.exists())
        self.assertFalse(self.svc.storage.input_path(uid, job2["id"]).exists())
        self.assertEqual(b.get(f"/api/jobs/{job2['id']}").status_code, 404)

    def test_a_job_canceled_mid_stage_shows_nothing_of_its_own_running(self):
        # The page polls while the job is active or a stage beside it (`side`) runs; a stage of the job's own that
        # was running at the cancel used to stay "running" for good.
        b = self.signin()
        job = self.make_job(b).json()
        c = self.store.job_claim(job["id"], "c", 60, 2)
        self.store.job_stage_set(job["id"], c["attempt"], "1", "running", lease_s=60)
        view = b.post(f"/api/jobs/{job['id']}/cancel", json={}).json()
        self.assertEqual(view["state"], "canceled")
        stages = {s["key"]: s for s in view["stages"]}
        self.assertEqual(stages["1"]["state"], "canceled")
        self.assertEqual([k for k in stages if k == "2h"], [])
        self.assertEqual([k for k, s in stages.items() if s["state"] == "running"], [])
        self.assertEqual(b.get(f"/api/jobs/{job['id']}").json()["stages"], view["stages"])


class TestIsolation(WebCase):
    def test_users_never_reach_each_others_jobs(self):
        alice = self.signin("alice@example.com")
        job = self.make_job(alice).json()
        self.finish(job["id"])
        bob = self.signin("bob@example.com")
        jid = job["id"]
        for path in (f"/api/jobs/{jid}", f"/api/jobs/{jid}/download/glb", f"/api/jobs/{jid}/download/roblox",
                     f"/api/runs/r1/{jid}", f"/runs/r1/{jid}/00-sketch.png", f"/runs/r1/{jid}/04-body/high.glb"):
            self.assertEqual(bob.get(path).status_code, 404, path)
            self.assertEqual(alice.get(path).status_code, 200, path)
        self.assertEqual(bob.post(f"/api/jobs/{jid}/cancel", json={}).status_code, 404)
        self.assertEqual(bob.send("DELETE", f"/api/jobs/{jid}", json={}).status_code, 404)
        self.assertEqual(bob.get("/api/jobs").json()["jobs"], [])
        self.assertEqual(bob.get("/api/workspace").json()["runs"], {})
        self.assertIn(f"r1/{jid}", alice.get("/api/workspace").json()["runs"])
        anon = Browser(self.app)
        self.assertEqual(anon.get(f"/runs/r1/{jid}/00-sketch.png").status_code, 401)
        self.assertEqual(anon.get("/api/workspace").status_code, 401)

    def test_run_file_paths_are_confined(self):
        alice = self.signin()
        job = self.make_job(alice).json()
        self.finish(job["id"])
        jid = job["id"]
        ok = alice.get(f"/runs/r1/{jid}/views/front.png")
        self.assertEqual((ok.status_code, ok.headers["content-type"]), (200, "image/png"))
        self.assertIn("sandbox", ok.headers["content-security-policy"])
        for path in (f"/runs/r1/{jid}/%2e%2e/meta.db", f"/runs/r1/{jid}/%2e%2e/%2e%2e/%2e%2e/meta.db",
                     f"/runs/r2/{jid}/00-sketch.png", "/runs/r1/j_notanid/00-sketch.png",
                     f"/runs/r1/{jid}/stage-0.json%00.png", f"/runs/r1/{jid}/.gitignore", f"/runs/r1/{jid}/",
                     f"/runs/r1/{jid}/results.html", f"/api/runs/r2/{jid}", "/api/runs/r1/..",
                     f"/runs/r1/{jid}/views%2f..%2f..%2fmeta.db"):
            self.assertIn(alice.get(path).status_code, (404, 400), path)

    def test_lab_api_shape(self):
        alice = self.signin()
        job = self.make_job(alice).json()
        self.finish(job["id"])
        ws = alice.get("/api/workspace").json()
        self.assertEqual(ws["roots"], [{"key": "r1", "label": "Your heroes", "exists": True}])
        d = alice.get(f"/api/runs/r1/{job['id']}").json()
        self.assertEqual(d["id"], f"r1/{job['id']}")
        self.assertNotIn(str(self.tmp), json.dumps(d), "no server paths")
        self.assertNotIn(str(self.tmp), json.dumps(ws))

    def test_named_prompt_keeps_its_workspace_title_and_downloads(self):
        alice = self.signin()
        result = alice.post("/api/jobs", data={"kind": "prompt", "prompt": "a knight in violet armour",
                                               "name": "Violet Knight"}, files={"x": ("a", b"a")},
                            headers={"Idempotency-Key": "key-prompt-name"})
        self.assertEqual(result.status_code, 201)
        job = result.json()
        self.assertEqual(self.finish(job["id"]), "succeeded")
        view = alice.get(f"/api/jobs/{job['id']}").json()
        self.assertEqual(view["name"], "Violet Knight")
        self.assertEqual(alice.get(view["images"]["sketch"]).status_code, 200)
        self.assertEqual(alice.get(f"/runs/r1/{job['id']}/p-full-body.png").status_code, 200)
        self.assertTrue(view["downloads"]["glb"]["available"])
        self.assertTrue(view["downloads"]["roblox"]["available"])
        workspace = alice.get("/api/workspace").json()
        self.assertEqual(workspace["characters"][0]["title"], "Violet Knight")


class TestErrorPages(WebCase):
    """HTML for people on page routes (404, 500, the legal pages); JSON, as before, for the API and for files."""

    def with_gallery(self) -> None:
        import hashlib

        root = self.tmp / "gallery"
        (root / "ranger").mkdir(parents=True)
        assets = {}
        for kind, data, suffix in (("model", b"glTF-stand-in", "glb"), ("artwork", png(), "png"), ("sketch", png(), "png")):
            (root / f"ranger/{kind}.{suffix}").write_bytes(data)
            assets[kind] = {"file": f"ranger/{kind}.{suffix}", "sha256": hashlib.sha256(data).hexdigest()}
        (root / "catalog.json").write_text(json.dumps({"version": 1, "items": [
            {"id": "ranger", "title": "Woodland ranger", "description": "Fixture.", "code_commit": "a" * 40,
             "assets": assets}]}))
        self.store.close()
        self.build(gallery_dir=root)

    def assert_page_csp(self, r, name) -> None:
        csp = r.headers["content-security-policy"]
        self.assertIn("default-src 'self'", csp, name)
        self.assertIn("script-src 'self';", csp, name)
        self.assertIn("style-src 'self';", csp, name)
        self.assertIn("font-src 'self'", csp, name)
        self.assertIn("frame-ancestors 'none'", csp, name)
        self.assertNotIn("unsafe-inline", csp, name)
        self.assertNotIn("unsafe-eval", csp, name)
        self.assertNotRegex(r.text, r"<style|\sstyle=|\son[a-z]+=", name)         # no inline styles or handlers
        self.assertEqual(re.findall(r"<script(?![^>]*\ssrc=)[^>]*>", r.text), [], name)  # no inline scripts
        for h in ("x-content-type-options", "x-frame-options", "referrer-policy", "cache-control"):
            self.assertIn(h, r.headers, name)

    def test_unknown_page_is_an_html_404(self):
        r = Browser(self.app).get("/nope")
        self.assertEqual(r.status_code, 404)
        self.assertTrue(r.headers["content-type"].startswith("text/html"))
        self.assertIn("This page wandered off.", r.text)
        self.assertIn('class="site-header"', r.text)
        self.assertIn('class="site-footer"', r.text)
        self.assertRegex(r.text, r'<a class="btn btn--primary btn--l" href="/">Create a hero')
        self.assertRegex(r.text, r'<a class="btn btn--secondary btn--l" href="/#gallery">Back to gallery</a>')
        self.assertLess(r.text.index("/assets/tokens.css"), r.text.index("/assets/app.css"))
        self.assertIn("/assets/styles/pages.css", r.text)
        self.assertNotIn("<img", r.text, "no gallery, no art")
        self.assert_page_csp(r, "/nope")
        for path in ("/nope/deeper", "/gallery", "/viewer/x", "/admin/nothing"):
            r = Browser(self.app).get(path)
            self.assertEqual((r.status_code, r.headers["content-type"].startswith("text/html")),
                             (404, True), path)

    def test_the_404_shows_one_lazy_gallery_hero(self):
        self.with_gallery()
        b = Browser(self.app)
        r = b.get("/nope")
        self.assertEqual(r.status_code, 404)
        imgs = re.findall(r"<img[^>]*>", r.text)
        self.assertEqual(len(imgs), 1)
        self.assertIn('src="/gallery-assets/ranger/artwork"', imgs[0])
        self.assertIn('loading="lazy"', imgs[0])
        self.assertRegex(imgs[0], r'width="\d+" height="\d+"')
        self.assertEqual(b.get("/gallery-assets/ranger/artwork").status_code, 200)
        self.assert_page_csp(r, "/nope with art")

    def test_a_broken_gallery_never_breaks_the_404(self):
        self.with_gallery()
        (self.s.gallery_dir / "catalog.json").write_text("{not json")
        r = Browser(self.app).get("/nope")
        self.assertEqual(r.status_code, 404)
        self.assertIn("This page wandered off.", r.text)
        self.assertNotIn("<img", r.text)

    def test_api_and_file_routes_keep_json_errors(self):
        b = Browser(self.app)
        for path in ("/api", "/api/nope", "/api/jobs/x/y/z", "/assets/nope.css", "/static/nope.js",
                     "/gallery-assets/nope/artwork"):
            r = b.get(path)
            self.assertEqual(r.status_code, 404, path)
            self.assertEqual(r.headers["content-type"], "application/json", path)
            self.assertEqual(r.json()["error"]["code"], "not_found", path)
        r = b.get("/runs/r1/j_x/a.png")  # signed out: refused, as JSON
        self.assertEqual((r.status_code, r.headers["content-type"]), (401, "application/json"))
        b.session()
        r = b.post("/api/nope", json={})
        self.assertEqual((r.status_code, r.headers["content-type"]), (404, "application/json"))
        self.assertEqual(b.get("/api/nope").headers["content-security-policy"],
                         "default-src 'none'; frame-ancestors 'none'; sandbox")
        r = b.post("/nope", json={}, csrf=False)  # not a page request: whatever it was before, never HTML
        self.assertFalse(r.headers["content-type"].startswith("text/html"))

    def test_a_server_error_on_a_page_route_is_an_html_500(self):
        t = TestClient(self.app, base_url=ORIGIN, raise_server_exceptions=False)
        with mock.patch("herogen.service.web.App.page", side_effect=RuntimeError("boom-secret")):
            self.store.close()
            self.build()
            t = TestClient(self.app, base_url=ORIGIN, raise_server_exceptions=False)
            r = t.get("/")
            api = t.get("/api/jobs")
        self.assertEqual(r.status_code, 500)
        self.assertTrue(r.headers["content-type"].startswith("text/html"))
        self.assertIn("Something went wrong on our side.", r.text)
        self.assertIn("Your heroes and progress are saved.", r.text)
        self.assertIn('class="site-header"', r.text)
        self.assertNotIn("boom-secret", r.text)
        self.assert_page_csp(r, "500")
        self.assertEqual(api.status_code, 401, "the API is untouched by the page patch")

    def test_a_server_error_on_an_api_route_stays_json(self):
        b = self.signin()

        def boom(*a, **k):
            raise RuntimeError("boom")

        self.store.jobs_list = boom
        t = TestClient(self.app, base_url=ORIGIN, raise_server_exceptions=False)
        t.cookies = b.c.cookies
        r = t.get("/api/jobs")
        self.assertEqual((r.status_code, r.headers["content-type"]), (500, "application/json"))
        self.assertEqual(r.json()["error"]["code"], "internal")

    def test_legal_pages_use_the_site_chrome_and_the_page_csp(self):
        legal = self.tmp / "legal"
        legal.mkdir()
        (legal / "terms.md").write_text(
            "HeroGen Terms of Service (beta)\n\nDRAFT. Version {{terms_version}}.\n\n1. Who we are\n\n"
            "We run it. <b>x</b>\nOn two lines.\n\n5.1 You keep your rights.\n\n6. Rules\n\nYou must not:\n\n"
            "- break it;\n  or wrap it;\n- abuse it.\n")
        (legal / "privacy.md").write_text("Privacy text.")
        self.store.close()
        self.build(legal_dir=legal)
        b = Browser(self.app)
        r = b.get("/terms")
        self.assertEqual(r.status_code, 200)
        self.assertIn('class="site-header"', r.text)
        self.assertIn('class="site-footer"', r.text)
        self.assertEqual(r.text.count("<h1>"), 1)
        self.assertIn("<h1>Terms of Service</h1>", r.text)
        self.assertNotIn("HeroGen Terms of Service (beta)", r.text, "the document's title line is the H1")
        self.assertIn('<p class="doc-meta">DRAFT. Version t1.</p>', r.text)
        self.assertIn("<h2>1. Who we are</h2>", r.text)
        self.assertIn("<p>We run it. &lt;b&gt;x&lt;/b&gt; On two lines.</p>", r.text)
        self.assertIn('<p class="clause"><span class="clause__n">5.1</span>You keep your rights.</p>', r.text)
        self.assertIn("<ul><li>break it; or wrap it;</li><li>abuse it.</li></ul>", r.text)
        self.assert_page_csp(r, "/terms")
        # The visitor's theme choice applies: theme.js runs before the styles paint, and the header toggle exists.
        self.assertIn('<script src="/assets/theme.js"></script>', r.text)
        self.assertLess(r.text.index("/assets/theme.js"), r.text.index("/assets/tokens.css"))
        self.assertIn('<script src="/assets/site-header.js" defer></script>', r.text)
        self.assertIn("data-theme-toggle", r.text)
        p = b.get("/privacy")
        self.assertEqual(p.status_code, 200)
        self.assertIn("<h1>Privacy Policy</h1>", p.text)
        self.assert_page_csp(p, "/privacy")
        (legal / "terms.md").unlink()
        gone = b.get("/terms")
        self.assertEqual(gone.status_code, 404)
        self.assertIn("has not been published yet", gone.text)
        self.assertIn('class="site-header"', gone.text)
        self.assert_page_csp(gone, "/terms unpublished")

    def test_page_stylesheet_ships_and_uses_tokens_only(self):
        r = Browser(self.app).get("/assets/styles/pages.css")
        self.assertEqual((r.status_code, r.headers["content-type"]), (200, "text/css; charset=utf-8"))
        self.assertEqual(re.findall(r"#[0-9a-fA-F]{3,8}\b", r.text), [])


class TestPages(WebCase):
    def test_description_count_is_wired_to_the_field(self):
        from html.parser import HTMLParser

        class Elements(HTMLParser):
            def __init__(self, html):
                super().__init__()
                self.tags = []
                self.feed(html)

            def handle_starttag(self, tag, attrs):
                self.tags.append((tag, dict(attrs)))

        b = Browser(self.app)
        tags = Elements(b.get('/').text).tags
        self.assertTrue(any(t == 'label' and a.get('for') == 'prompt' for t, a in tags))
        field = next(a for t, a in tags if t == 'textarea' and a.get('id') == 'prompt')
        count = next(a for t, a in tags if a.get('id') == 'promptCount')
        self.assertIn('promptCount', field['aria-describedby'].split())
        self.assertEqual(count['aria-live'], 'polite')
        self.assertEqual(count['aria-atomic'], 'true')
        script = b.get('/assets/app.js').text
        self.assertIn('S.session.limits.prompt_max_chars', script)
        self.assertIn("prompt.addEventListener('input', updatePromptCount)", script)

    def test_description_api_limit_follows_the_selected_version(self):
        from unittest import mock
        from herogen import prompts

        version = prompts.selected('prompt_sketch')
        limit = version.data['user_text']['max_chars'] - 17
        with mock.patch.dict(version.data['user_text'], max_chars=limit), \
                mock.patch.object(prompts, 'selected', return_value=version):
            b = self.signin()
            self.assertEqual(b.session()['limits']['prompt_max_chars'], limit)
            r = b.post('/api/jobs', data={'kind': 'prompt', 'prompt': 'x' * (limit + 1)},
                       files={'x': ('a', b'a')}, headers={'Idempotency-Key': 'key-prompt-limit'})
            self.assertEqual(r.status_code, 400)
            self.assertIn(f'the limit is {limit}', r.json()['error']['message'])
            self.assertEqual(self.submitted, [])

    def test_legal_pages(self):
        self.assertEqual(Browser(self.app).get("/privacy").status_code, 404)
        legal = self.tmp / "legal"
        legal.mkdir()
        (legal / "privacy.md").write_text("Files are kept {{retention_days}} days. <script>alert(1)</script>")
        (legal / "terms.md").write_text("Terms {{terms_version}}.")
        self.store.close()
        self.build(legal_dir=legal)
        r = Browser(self.app).get("/privacy")
        self.assertEqual(r.status_code, 200)
        self.assertIn("kept 30 days", r.text)
        self.assertNotIn("<script>", r.text)
        self.assertIn("&lt;script&gt;", r.text)
        self.assertIn("font-src 'self'", r.headers["content-security-policy"])
        self.assertLess(r.text.index("/assets/tokens.css"), r.text.index("/assets/app.css"))
        self.assertIn("Terms t1.", Browser(self.app).get("/terms").text)

    def test_pages_load_the_design_system_from_this_origin(self):
        from herogen.service import web

        b = Browser(self.app)
        pages = {"/": b.get("/").text, "/gallery/x": b.get("/gallery/x").text, "/admin": b.get("/admin").text,
                 "/nope": b.get("/nope").text, "viewer.html": (web.STATIC_DIR / "viewer.html").read_text()}
        fonts = ("/assets/fonts/Figtree-Variable.latin.woff2", "/assets/fonts/Fraunces-Variable.latin.woff2")
        for name, page in pages.items():
            self.assertLess(page.index('href="/assets/tokens.css"'), page.index('href="/assets/app.css"'), name)
            self.assertIn('<link rel="icon" href="/assets/icons/favicon.svg" type="image/svg+xml">', page, name)
            for font in fonts:
                self.assertIn(f'<link rel="preload" href="{font}" as="font" type="font/woff2" crossorigin>', page, name)
            self.assertNotIn("fonts.googleapis", page, name)
            self.assertNotIn("style=", page, name)          # the CSP allows no inline styles
        tokens = b.get("/assets/tokens.css").text
        for font in re.findall(r'url\("(fonts/[^"]+)"\)', tokens):
            r = b.get(f"/assets/{font}")
            self.assertEqual((r.status_code, r.headers["content-type"]), (200, "font/woff2"), font)
            self.assertEqual(r.content[:4], b"wOF2", font)
        sprite = b.get("/assets/icons/sprite.svg")
        self.assertEqual((sprite.status_code, sprite.headers["content-type"]), (200, "image/svg+xml"))
        symbols = set(re.findall(r'<symbol id="([^"]+)"', sprite.text))
        for name in ("/", "/gallery/x", "/nope"):
            used = set(re.findall(r'/assets/icons/sprite\.svg#([\w-]+)', pages[name]))
            self.assertTrue(used and used <= symbols, (name, used - symbols))
        self.assertEqual(b.get("/assets/icons/favicon.svg").headers["content-type"], "image/svg+xml")

    def test_only_tokens_css_writes_hex_colours_in_the_new_styles(self):
        from herogen.service import web

        css = (web.STATIC_DIR / "app.css").read_text()
        marker = "/* legacy rules */"
        self.assertEqual(css.count(marker), 1)
        self.assertEqual(re.findall(r"#[0-9a-fA-F]{3,8}\b", css[:css.index(marker)]), [])

    def test_workspace_stage_styles_are_served_and_use_only_tokens(self):
        from herogen.service import web

        b = Browser(self.app)
        self.assertIn('href="/assets/styles/workspace-stage.css"', b.get("/").text)
        r = b.get("/assets/styles/workspace-stage.css")
        self.assertEqual(r.status_code, 200)
        css = (web.STATIC_DIR / "styles" / "workspace-stage.css").read_text()
        self.assertEqual(re.findall(r"#[0-9a-fA-F]{3,8}\b", css), [])
        self.assertEqual(re.findall(r"\b\d+(?:\.\d+)?m?s\b", css), [])   # durations come from tokens

    def test_viewer_and_health(self):
        b = Browser(self.app)
        r = b.get("/viewer")
        self.assertIn("sha256-", r.headers["content-security-policy"])
        self.assertEqual(b.get("/healthz").json()["ok"], True)
        self.assertEqual(b.get("/api/nothing").status_code, 404)
