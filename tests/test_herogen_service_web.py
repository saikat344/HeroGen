"""The HTTP layer end to end (Starlette's test client, stand-ins for email and workers): security headers, host
check, CSRF, sign-in by link, invite-only, rate limits, sessions, uploads, quotas, idempotency, downloads, per-user
isolation, path safety, deletion, the Lab API and the legal pages."""

from __future__ import annotations

import io
import json
import os
import re
import sys
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from test_herogen_service_support import HAVE_WEB, Clock, TempDirCase, png, settings, web_test

if HAVE_WEB:
    from starlette.testclient import TestClient

    from herogen.service.auth import Signer, new_invite_code, normalize_invite
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
        self.build()

    def build(self, **kw) -> None:
        self.clock = Clock()
        self.s = settings(self.tmp, **kw)
        self.store = Store(self.s.meta_path, clock=self.clock)
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
        self.store.allow_add(email)
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
        self.build(base_url="https://testserver")
        b = Browser(self.app, base="https://testserver")
        r = b.get("/api/session")
        self.assertIn("max-age=63072000", r.headers["strict-transport-security"])
        cookie = r.headers["set-cookie"]
        self.assertTrue(cookie.startswith("__Host-hg_csrf="))
        for flag in ("HttpOnly", "Secure", "Path=/", "SameSite=lax"):
            self.assertIn(flag, cookie)
        self.assertNotIn("Domain", cookie)
        b.csrf = r.json()["csrf"]
        self.store.allow_add("alice@example.com")
        b.post("/api/auth/request", json={"email": "alice@example.com"})
        r = b.post("/api/auth/verify", json={"token": self.link_token(), "accept_terms": True, "adult": True})
        self.assertEqual(r.status_code, 200)
        cookies = r.headers.get_list("set-cookie")
        sess = next(c for c in cookies if c.startswith("__Host-hg_session="))
        for flag in ("HttpOnly", "Secure", "Path=/", "SameSite=lax"):
            self.assertIn(flag, sess)
        self.assertTrue(self.sender.sent[-1].text.count("https://testserver/signin#t="), 1)

    def test_static_assets_are_confined(self):
        b = Browser(self.app)
        self.assertEqual(b.get("/assets/app.js").headers["content-type"], "text/javascript; charset=utf-8")
        for p in ("/assets/../web.py", "/assets/%2e%2e/web.py", "/assets/index.html", "/assets/.hidden",
                  "/assets/app.js/", "/assets/..%2fweb.py", "/static/%2e%2e/server.py", "/static/index.html",
                  "/static/..%2fserver.py", "/static/.hidden", "/static/app.js/"):
            self.assertEqual(b.get(p).status_code, 404, p)

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
                             ("POST", "/api/auth/verify"), ("POST", "/api/auth/link"), ("PUT", "/api/jobs")):
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
    def test_full_flow_and_single_use(self):
        self.store.allow_add("alice@example.com")
        b = Browser(self.app)
        self.assertFalse(b.session()["signed_in"])
        r = b.post("/api/auth/request", json={"email": " Alice@Example.com "})
        self.assertEqual(r.status_code, 200)
        msg = self.sender.sent[-1]
        self.assertEqual(msg.to, "alice@example.com")
        self.assertIn(f"{ORIGIN}/signin#t=", msg.text, "the link is built from the configured origin")
        tok = self.link_token()
        info = b.post("/api/auth/link", json={"token": tok}).json()
        self.assertEqual(info["email"], "a***e@example.com")
        self.assertTrue(info["needs_terms"])
        r = b.post("/api/auth/verify", json={"token": tok})
        self.assertEqual(r.json()["error"]["code"], "needs_terms")
        r = b.post("/api/auth/verify", json={"token": tok, "accept_terms": True, "adult": True})
        self.assertEqual(r.status_code, 200)
        b.csrf = r.json()["csrf"]
        s = b.session()
        self.assertTrue(s["signed_in"])
        self.assertEqual(s["user"]["email"], "alice@example.com")
        other = Browser(self.app)
        other.session()
        r = other.post("/api/auth/verify", json={"token": tok, "accept_terms": True, "adult": True})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (400, "invalid_link"), "used once only")

    def test_expired_link(self):
        self.store.allow_add("alice@example.com")
        b = Browser(self.app)
        b.session()
        b.post("/api/auth/request", json={"email": "alice@example.com"})
        tok = self.link_token()
        self.clock.advance(self.s.link_ttl_s + 1)
        r = b.post("/api/auth/verify", json={"token": tok, "accept_terms": True, "adult": True})
        self.assertEqual(r.json()["error"]["code"], "invalid_link")
        self.assertEqual(b.post("/api/auth/link", json={"token": tok}).status_code, 400)

    def test_existing_unknown_and_allowlisted_addresses_answer_identically(self):
        self.signin("member@example.com")                  # an existing account
        self.store.allow_add("invited@example.com")         # allowlisted, no account yet
        code = new_invite_code()
        self.store.invite_create(self.svc.signer.hash("invite", normalize_invite(code)), 5, 3600,
                                 email="someone@example.com")
        b = Browser(self.app)
        b.session()
        for invite in (None, "", "NOPE-NOPE-NOPE-NOPE", "not a code", code):
            self.clock.advance(3601)  # a fresh per-address rate-limit window for each round
            answers = []
            for email in ("member@example.com", "stranger@example.com", "invited@example.com"):
                body = {"email": email} if invite is None else {"email": email, "invite_code": invite}
                r = b.post("/api/auth/request", json=body)
                answers.append((r.status_code, r.json()))
            self.assertEqual(len({json.dumps(a, sort_keys=True) for a in answers}), 1, (invite, answers))

    def test_signin_request_writes_nothing_before_responding(self):
        events: list[str] = []
        store = self.store

        class Recording:
            def __getattr__(self, name):
                fn = getattr(store, name)

                def call(*a, **k):
                    events.append(name)
                    return fn(*a, **k)
                return call

        inner = self.app

        async def app(scope, receive, send):
            async def mark(message):
                if message["type"] == "http.response.start" and scope["path"] == "/api/auth/request":
                    events.append("RESPONSE")
                await send(message)
            await inner(scope, receive, mark)

        self.svc.store = Recording()
        self.store.allow_add("alice@example.com")
        code = new_invite_code()
        self.store.invite_create(self.svc.signer.hash("invite", normalize_invite(code)), 5, 3600)
        b = Browser(app)
        b.session()
        for body in ({"email": "alice@example.com"}, {"email": "bob@example.com", "invite_code": code},
                     {"email": "stranger@example.com"}):
            events.clear()
            self.assertEqual(b.post("/api/auth/request", json=body).status_code, 200)
            before = events[:events.index("RESPONSE")]
            self.assertTrue(set(before) <= {"rate_hit", "invite_valid"}, (body, before))
            self.assertIn("signin_eligibility", events[events.index("RESPONSE"):])
        self.assertEqual([m.to for m in self.sender.sent], ["alice@example.com", "bob@example.com"])

    def test_uninvited_addresses_get_the_same_answer_and_no_email(self):
        b = Browser(self.app)
        b.session()
        r1 = b.post("/api/auth/request", json={"email": "stranger@example.com"})
        self.store.allow_add("alice@example.com")
        r2 = b.post("/api/auth/request", json={"email": "alice@example.com"})
        self.assertEqual(r1.status_code, r2.status_code)
        self.assertEqual(r1.json(), r2.json())
        self.assertEqual([m.to for m in self.sender.sent], ["alice@example.com"])

    def test_invite_codes(self):
        code = new_invite_code()
        self.store.invite_create(self.svc.signer.hash("invite", normalize_invite(code)), 1, 3600)
        b = Browser(self.app)
        b.session()
        r = b.post("/api/auth/request", json={"email": "bob@example.com", "invite_code": "NOPE-NOPE-NOPE-NOPE"})
        self.assertEqual(r.json()["error"]["code"], "invalid_invite")
        r = b.post("/api/auth/request", json={"email": "bob@example.com", "invite_code": "nonsense"})
        self.assertEqual(r.json()["error"]["code"], "invalid_invite")
        self.assertEqual(self.sender.sent, [])
        r = b.post("/api/auth/request", json={"email": "bob@example.com", "invite_code": code.lower()})
        self.assertEqual(r.status_code, 200)
        tok = self.link_token()
        self.assertTrue(b.post("/api/auth/link", json={"token": tok}).json()["new_account"])
        r = b.post("/api/auth/verify", json={"token": tok, "accept_terms": True, "adult": True})
        self.assertEqual(r.status_code, 200)
        c = Browser(self.app)
        c.session()
        r = c.post("/api/auth/request", json={"email": "cat@example.com", "invite_code": code})
        self.assertEqual(r.json()["error"]["code"], "invalid_invite", "the code's one use is spent")

    def test_bad_inputs(self):
        b = Browser(self.app)
        b.session()
        for body in ({"email": "not an email"}, {"email": ["a@example.com"]}, {}):
            self.assertEqual(b.post("/api/auth/request", json=body).status_code, 400)
        self.assertEqual(b.post("/api/auth/request", content=b"[1]", headers={"Content-Type": "application/json"}
                                ).status_code, 400)
        self.assertEqual(b.post("/api/auth/request", content=b"{", headers={"Content-Type": "application/json"}
                                ).status_code, 400)
        self.assertEqual(b.post("/api/auth/request", data={"email": "a@example.com"}).status_code, 415)
        big = json.dumps({"email": "a@example.com", "pad": "x" * 20000})
        self.assertEqual(b.post("/api/auth/request", content=big, headers={"Content-Type": "application/json"}
                                ).status_code, 413)
        for tok in ("short", "x" * 43 + "!", None, 5):
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

    def test_signin_closed_without_published_terms_in_production(self):
        self.store.close()
        self.build(mode="production", base_url="https://testserver", email_provider="resend", email_api_key="k" * 20,
                   email_from="HeroGen <signin@example.org>", legal_dir=self.tmp / "legal")
        b = Browser(self.app, base="https://testserver")
        b.session()
        self.store.allow_add("alice@example.com")
        b.post("/api/auth/request", json={"email": "alice@example.com"})
        r = b.post("/api/auth/verify", json={"token": self.link_token(), "accept_terms": True, "adult": True})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (503, "signups_closed"))


class TestJobsApi(WebCase):
    def test_create_status_finish_download(self):
        b = self.signin()
        r = self.make_job(b)
        self.assertEqual(r.status_code, 201, r.text)
        job = r.json()
        self.assertEqual((job["state"], job["name"], job["kind"]), ("queued", "Sir Bob", "sketch"))
        self.assertEqual(self.submitted, [job["id"]], "dispatched at once")
        self.assertEqual([s["key"] for s in job["stages"]], ["0", "1", "2", "3", "4", "6", "5", "2h"])
        self.assertFalse(job["downloads"]["glb"]["available"])
        self.assertEqual(b.get(f"/api/jobs/{job['id']}/download/glb").status_code, 409)
        self.assertEqual(self.finish(job["id"]), "succeeded")
        view = b.get(f"/api/jobs/{job['id']}").json()
        self.assertEqual(view["state"], "succeeded")
        self.assertTrue(all(s["state"] == "passed" for s in view["stages"] if s["key"] != "2h"))
        self.assertEqual(view["stages"][-1]["state"], "pending", "this test starts no quality check")
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

    def test_idempotency_and_quota(self):
        b = self.signin()
        first = self.make_job(b, key="key-00000001")
        again = self.make_job(b, key="key-00000001")
        self.assertEqual((first.status_code, again.status_code), (201, 200))
        self.assertEqual(first.json()["id"], again.json()["id"])
        uid = self.store.users_list()[0]["id"]
        self.assertEqual(len(list((self.svc.storage.user_dir(uid) / "inputs").iterdir())), 1, "no orphan upload")
        r = self.make_job(b, key="key-00000001", name="Another")
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (409, "conflict"))
        self.assertEqual(self.make_job(b, key="key-00000002").status_code, 201)
        r = b.post("/api/jobs", data={"kind": "prompt", "prompt": "a knight in green armour"},
                   files={"x": ("a", b"a")}, headers={"Idempotency-Key": "key-00000003"})
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.json()["prompt"], "a knight in green armour")
        r = self.make_job(b, key="key-00000004")
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (429, "quota"))
        self.assertEqual(len(self.submitted), 2, "the third waits: at most max_running_jobs (2) run at once")
        self.assertEqual(len(list((self.svc.storage.user_dir(uid) / "inputs").iterdir())), 2,
                         "a refused job leaves no upload")
        q = b.get("/api/quota").json()
        self.assertEqual((q["jobs_today"], q["jobs_per_day"]), (3, 3))

    def test_failing_after_paid_stages_cannot_run_past_the_quota(self):
        from herogen.service.runner import Outcome

        class FailAtMesh(FakeRunner):
            def run(self, spec, ctx):
                if spec.key == "3":
                    return Outcome("failed", 1.0, 0.0, "internal", "stage_error", False)
                return super().run(spec, ctx)

        b = self.signin()
        ran = 0
        for i in range(12):
            r = b.post("/api/jobs", data={"kind": "prompt", "prompt": f"a knight number {i}"},
                       files={"x": ("a", b"a")}, headers={"Idempotency-Key": f"loop-key-{i:04d}"})
            if r.status_code != 201:
                self.assertEqual((r.status_code, r.json()["error"]["code"]), (429, "quota"))
                break
            self.assertEqual(self.finish(r.json()["id"], FailAtMesh()), "failed")
            ran += 1
        self.assertEqual(ran, 3, "jobs_per_day is 3 and every one of them spent money")

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
        self.assertEqual(self.store.usage(uid, 3, 5)["jobs_total"], 1, "charged once")
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
        self.assertEqual(self.store.usage(uid, 3, 5)["jobs_total"], 1)
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

    def test_an_unavailable_quality_check_is_not_a_job_error(self):
        from herogen.service.snapshot import LocalVolume
        from herogen.service.worker import (
            HeldoutWorkdir,
            InPlaceWorkdir,
            make_context,
            run_group,
            run_heldout,
            run_job,
        )

        b = self.signin()
        job = self.make_job(b).json()
        runner = FakeRunner(fail={"2h": "error"})

        def heldout(j):
            ctx = make_context(self.s, self.svc.storage, j)
            run_heldout(self.store, self.s, j, runner, lambda claim: HeldoutWorkdir(
                ctx, self.tmp / "h" / j["id"], LocalVolume(), lambda: self.store.side_live(j["id"], "2h", claim)))

        def execute(j, attempt, group, resume):
            return run_group(self.store, self.s, j, attempt, group, resume, runner,
                             InPlaceWorkdir(make_context(self.s, self.svc.storage, j)))

        self.assertEqual(run_job(self.store, self.s, self.svc.storage, job["id"], "t", execute, heldout=heldout),
                         "succeeded")
        view = b.get(f"/api/jobs/{job['id']}").json()
        self.assertEqual((view["state"], view["error"]), ("succeeded", None))
        side = next(s for s in view["stages"] if s["key"] == "2h")
        self.assertEqual((side["label"], side["state"]), ("quality check (held-out views)", "unavailable"))
        self.assertIn("The hero is not affected.", side["message"])
        self.assertTrue(view["downloads"]["glb"]["available"])
        self.assertTrue(view["downloads"]["roblox"]["available"])

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

    def test_delete_account(self):
        b = self.signin()
        job = self.make_job(b).json()
        self.finish(job["id"])
        uid = self.store.users_list()[0]["id"]
        r = b.send("DELETE", "/api/account", json={"confirm": "yes"})
        self.assertEqual(r.json()["error"]["code"], "confirm")
        r = b.send("DELETE", "/api/account", json={"confirm": "delete my account"})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(self.svc.storage.user_dir(uid).exists())
        self.assertEqual(b.get("/api/jobs").status_code, 401)
        self.assertEqual(self.store.users_list(), [])


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


class TestPages(WebCase):
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
        self.assertIn("default-src 'none'", r.headers["content-security-policy"])
        self.assertIn("Terms t1.", Browser(self.app).get("/terms").text)

    def test_viewer_and_health(self):
        b = Browser(self.app)
        r = b.get("/viewer")
        self.assertIn("sha256-", r.headers["content-security-policy"])
        self.assertEqual(b.get("/healthz").json()["ok"], True)
        self.assertEqual(b.get("/api/nothing").status_code, 404)
