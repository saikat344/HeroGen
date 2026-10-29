"""The service's HTTP layer: a Starlette ASGI app (served by Modal's ASGI support, or uvicorn in dev mode).

Pages: `/` (sign in, new hero, your heroes), `/signin` (the page a sign-in link opens), `/viewer` (the Studio
viewer when the deployed code has it, else a minimal one), `/terms`, `/privacy`, `/healthz`.
API (JSON; every unsafe method needs the CSRF header): see docs/ARCHITECTURE.md, "Service", for the table.
Lab API for the viewer: `/api/workspace`, `/api/runs/r1/<job>`, `/runs/r1/<job>/<path>`, scoped to the user.

Every response carries the security headers in `SecurityHeaders`; pages get a CSP that allows the service's own
files, the inline import map by its hash and the one pinned three.js path, nothing else. Requests must name an
allowed host. Unsafe methods must come from the service's own origin (Origin or Sec-Fetch-Site) and carry
`X-CSRF-Token`, an HMAC of the browser's CSRF cookie. Sessions are an HttpOnly, Secure (on https), SameSite=Lax
cookie whose value the store never sees (it keeps an HMAC). Sign-in links carry their token in the URL fragment,
which browsers never send to a server or in a Referer, and are used by an explicit POST from the page, so a mail
scanner that fetches the link cannot use it up.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from herogen.service import bundle, labapi
from herogen.service.auth import TOKEN_RE, Signer, mask_email, normalize_email, normalize_invite
from herogen.service.config import Settings
from herogen.service.emailer import EmailError, EmailSender, signin_message
from herogen.service.runner import STAGE_LABELS, plan_for
from herogen.service.storage import SAFE_SEGMENT, Storage
from herogen.service.store import Conflict, NotFound, QuotaExceeded, RateLimiter, StoreError, new_id, valid_id
from herogen.service.uploads import UploadRejected, clean_name, clean_prompt
from herogen.service.worker import dispatch_queued, event

log = logging.getLogger("herogen.service")
STATIC_DIR = Path(__file__).resolve().parent / "static"
THREE_VERSION = "0.160.0"
CDN_PREFIX = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/"
ASSET_TYPES = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
               ".svg": "image/svg+xml", ".json": "application/json", ".txt": "text/plain; charset=utf-8",
               ".ttf": "font/ttf", ".woff2": "font/woff2"}
FILE_CSP = "default-src 'none'; frame-ancestors 'none'; sandbox"
JSON_LIMIT = 16 * 1024
IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
UNSAFE = ("POST", "PUT", "PATCH", "DELETE")
DOWNLOADS = ("glb", "roblox")
_IMPORTMAP = re.compile(r"<script type=\"importmap\">(.*?)</script>", re.DOTALL)


@dataclass
class Services:
    settings: Settings
    store: Any
    signer: Signer
    storage: Storage
    sender: EmailSender
    sanitize: Callable[[bytes], tuple[bytes, dict[str, Any]]]
    submit: Callable[[str], str]
    clock: Callable[[], float] = time.time
    local_rates: RateLimiter = field(default_factory=lambda: RateLimiter(time.time))


def page_csp(page_html: str) -> str:
    hashes = " ".join("'sha256-" + base64.b64encode(hashlib.sha256(m.encode()).digest()).decode() + "'"
                      for m in _IMPORTMAP.findall(page_html))
    return ("default-src 'self'; "
            f"script-src 'self' {CDN_PREFIX} {hashes}".rstrip() + "; "
            "style-src 'self'; img-src 'self' blob: data:; font-src 'self'; "
            f"connect-src 'self' blob: data: {CDN_PREFIX}; "
            "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


class SecurityHeaders:
    """Pure ASGI middleware: the same headers on every response (handlers may set their own CSP and cache
    policy), plus one structured access-log line without query strings, cookies or addresses."""

    def __init__(self, app, settings: Settings) -> None:
        self.app = app
        self.hsts = settings.secure_cookies

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        t0 = time.monotonic()
        status = {"code": 500}

        async def wrapped(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"server"]
                have = {k.lower() for k, _ in headers}
                add = {
                    b"x-content-type-options": b"nosniff",
                    b"referrer-policy": b"no-referrer",
                    b"x-frame-options": b"DENY",
                    b"cross-origin-opener-policy": b"same-origin",
                    b"cross-origin-resource-policy": b"same-origin",
                    b"permissions-policy": b"camera=(), microphone=(), geolocation=(), payment=(), usb=()",
                    b"content-security-policy": FILE_CSP.encode(),
                    b"cache-control": b"no-store",
                }
                if self.hsts:
                    add[b"strict-transport-security"] = b"max-age=63072000; includeSubDomains"
                headers += [(k, v) for k, v in add.items() if k not in have]
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, wrapped)
        finally:
            log.info(json.dumps({"event": "http", "method": scope.get("method"), "path": scope.get("path"),
                                 "status": status["code"], "ms": round((time.monotonic() - t0) * 1000, 1)}))


def error(status: int, code: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


BUSY_RETRY_S = 5


def busy_response(message: str = "The service is busy; please try again in a moment.") -> JSONResponse:
    return error(503, "busy", message, {"Retry-After": str(BUSY_RETRY_S)})


class App:
    def __init__(self, svc: Services) -> None:
        self.svc = svc
        s = svc.settings
        self.s = s
        secure = s.secure_cookies
        self.session_cookie = "__Host-hg_session" if secure else "hg_session"
        self.csrf_cookie = "__Host-hg_csrf" if secure else "hg_csrf"

    # ---- helpers ----------------------------------------------------------------------------------------

    async def call(self, fn, *args, **kw):
        return await run_in_threadpool(fn, *args, **kw)

    def client_ip(self, request: Request) -> str:
        h = self.s.trusted_ip_header
        if h:
            raw = request.headers.get(h, "")
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if parts:
                return parts[-1]
        return request.client.host if request.client else "unknown"

    async def limit(self, bucket: str, rule: tuple[int, int]) -> JSONResponse | None:
        ok, retry = await self.call(self.svc.store.rate_hit, bucket, rule[0], rule[1])
        if ok:
            return None
        return error(429, "rate_limited", "Too many requests. Please wait a little and try again.",
                     {"Retry-After": str(retry)})

    def set_cookie(self, resp: Response, name: str, value: str, max_age: int) -> None:
        resp.set_cookie(name, value, max_age=max_age, path="/", secure=self.s.secure_cookies, httponly=True,
                        samesite="lax")

    def clear_cookie(self, resp: Response, name: str) -> None:
        resp.delete_cookie(name, path="/", secure=self.s.secure_cookies, httponly=True, samesite="lax")

    async def user(self, request: Request) -> dict[str, Any] | None:
        raw = request.cookies.get(self.session_cookie)
        if not raw or not TOKEN_RE.match(raw):
            return None
        return await self.call(self.svc.store.session_get, self.svc.signer.hash("session", raw), self.s.session_idle_s)

    async def require_user(self, request: Request) -> dict[str, Any]:
        u = await self.user(request)
        if u is None:
            raise HTTPException(401, "sign in first")
        return u

    async def json_body(self, request: Request) -> dict[str, Any]:
        if not request.headers.get("content-type", "").split(";")[0].strip() == "application/json":
            raise HTTPException(415, "send JSON")
        try:
            data = json.loads(await request.body())
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(400, "the request body is not valid JSON") from None
        if not isinstance(data, dict):
            raise HTTPException(400, "the request body must be a JSON object")
        return data

    def csrf_for(self, request: Request) -> tuple[str, str | None]:
        """(CSRF token for this browser, a new CSRF cookie value to set, or None when it already has one)."""
        cookie = request.cookies.get(self.csrf_cookie)
        new = None
        if not cookie or not TOKEN_RE.match(cookie):
            cookie = new = self.svc.signer.new_token()
        return self.svc.signer.csrf_token(cookie), new

    # ---- pages ------------------------------------------------------------------------------------------

    def page(self, name: str) -> Response:
        body = (STATIC_DIR / name).read_text()
        return Response(body, media_type="text/html; charset=utf-8", headers={"Content-Security-Policy": page_csp(body)})

    async def index(self, request: Request) -> Response:
        resp = self.page("index.html")
        _, new = self.csrf_for(request)
        if new:
            self.set_cookie(resp, self.csrf_cookie, new, self.s.session_ttl_s)
        return resp

    async def viewer(self, request: Request) -> Response:
        studio = labapi.studio_static()
        if studio is not None:
            body = (studio / "index.html").read_text()
            return Response(body, media_type="text/html; charset=utf-8",
                            headers={"Content-Security-Policy": page_csp(body)})
        return self.page("viewer.html")

    def _asset(self, base: Path | None, name: str) -> Response:
        if base is None:
            return error(404, "not_found", "Not found.")
        parts = name.split("/")
        if not name or "\\" in name or "\x00" in name or any(not SAFE_SEGMENT.match(p) or ".." in p for p in parts):
            return error(404, "not_found", "Not found.")
        ctype = ASSET_TYPES.get(Path(parts[-1]).suffix.lower())
        try:
            target = base.resolve().joinpath(*parts)
            resolved = target.resolve(strict=True)
        except (OSError, RuntimeError):
            return error(404, "not_found", "Not found.")
        if ctype is None or resolved != target or not resolved.is_file():
            return error(404, "not_found", "Not found.")
        return Response(resolved.read_bytes(), media_type=ctype, headers={"Cache-Control": "no-cache"})

    async def assets(self, request: Request) -> Response:
        return self._asset(STATIC_DIR, request.path_params["path"])

    async def studio_assets(self, request: Request) -> Response:
        return self._asset(labapi.studio_static(), request.path_params["path"])

    async def legal(self, request: Request) -> Response:
        which = request.url.path.strip("/")
        title = {"terms": "Terms of Service", "privacy": "Privacy Policy"}[which]
        text = None
        if self.s.legal_dir is not None:
            p = self.s.legal_dir / f"{which}.md"
            if p.is_file():
                text = p.read_text(encoding="utf-8").replace("{{retention_days}}", str(self.s.retention_days))
                text = text.replace("{{terms_version}}", self.s.terms_version)
        if text is None:
            body = f"<p>The {title} has not been published yet.</p>"
            status = 404
        else:
            body = f"<pre class=\"legal\">{html.escape(text)}</pre>"
            status = 200
        page = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
                f"content=\"width=device-width, initial-scale=1\"><title>HeroGen {title}</title><link rel=\"stylesheet\" "
                f"href=\"/assets/app.css\"></head><body><main class=\"doc\"><p><a href=\"/\">HeroGen</a></p>"
                f"<h1>{title}</h1>{body}</main></body></html>")
        return Response(page, status_code=status, media_type="text/html; charset=utf-8",
                        headers={"Content-Security-Policy": "default-src 'none'; style-src 'self'; "
                                                            "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"})

    async def healthz(self, request: Request) -> Response:
        return JSONResponse({"ok": True, "commit": self.s.code_commit or None})

    # ---- session and sign-in ----------------------------------------------------------------------------

    def terms_ready(self) -> bool:
        if not self.s.terms_version:
            return False
        if self.s.mode == "dev":
            return True
        d = self.s.legal_dir
        return d is not None and (d / "terms.md").is_file() and (d / "privacy.md").is_file()

    async def session(self, request: Request) -> Response:
        u = await self.user(request)
        data: dict[str, Any] = {"signed_in": u is not None, "limits": self.s.public(), "terms_ready": self.terms_ready()}
        if u is not None:
            data["user"] = {"email": u["email"], "created_at": u["created_at"]}
            data["usage"] = await self.call(self.svc.store.usage, u["id"], self.s.jobs_per_day, self.s.jobs_total)
        data["csrf"], new = self.csrf_for(request)
        resp = JSONResponse(data)
        if new:
            self.set_cookie(resp, self.csrf_cookie, new, self.s.session_ttl_s)
        return resp

    async def auth_request(self, request: Request) -> Response:
        body = await self.json_body(request)
        ip_tag = self.svc.signer.tag(self.client_ip(request))
        if (r := await self.limit(f"auth-ip:{ip_tag}", self.s.rate_auth_ip)) is not None:
            return r
        email = normalize_email(body.get("email"))
        if email is None:
            return error(400, "bad_email", "Enter a valid email address.")
        if (r := await self.limit(f"auth-email:{self.svc.signer.tag(email)}", self.s.rate_auth_email)) is not None:
            return r
        raw_code = body.get("invite_code")
        invite_hash = None
        if raw_code not in (None, ""):
            code = normalize_invite(raw_code)
            if code is None:
                return error(400, "invalid_invite", "That invite code is not valid.")
            invite_hash = self.svc.signer.hash("invite", code)
            # Judged from the code alone (a read, the same for every address), so it says nothing about who has
            # an account.
            if not await self.call(self.svc.store.invite_valid, invite_hash):
                return error(400, "invalid_invite", "That invite code is not valid, has expired or has been used up.")
        # The same answer, after the same work, for every address. Whether it may sign in, the link (a durable store
        # write) and the email all happen after the response has gone.
        task = BackgroundTask(self._signin_link, email, invite_hash, ip_tag)
        return JSONResponse({"ok": True, "message": "If this address can sign in, a sign-in link is on its way. "
                                                    "It expires in "
                                                    f"{max(1, self.s.link_ttl_s // 60)} minutes."}, background=task)

    def _signin_link(self, email: str, invite_hash: str | None, ip_tag: str) -> None:
        """After the response: check the address, make the link and send it, when the address may sign in."""
        try:
            elig = self.svc.store.signin_eligibility(email, invite_hash)
            event("signin_requested", eligible=elig["eligible"], ip=ip_tag)
            if not elig["eligible"]:
                return
            token = self.svc.signer.new_token()
            self.svc.store.login_token_create(self.svc.signer.hash("login", token), email, self.s.link_ttl_s,
                                              elig["invite_id"], f"ip:{ip_tag}")
            link = f"{self.s.origin}/signin#t={token}"
            self.svc.sender.send(signin_message(email, link, max(1, self.s.link_ttl_s // 60)))
            event("signin_email_sent")
        except EmailError as e:
            log.error(json.dumps({"event": "signin_email_failed", "error": str(e)}))
        except Exception:
            log.exception("sign-in link failed")

    async def _token(self, request: Request, body: dict[str, Any]) -> str | JSONResponse:
        ip_tag = self.svc.signer.tag(self.client_ip(request))
        if (r := await self.limit(f"link-ip:{ip_tag}", self.s.rate_link_ip)) is not None:
            return r
        token = body.get("token")
        if not isinstance(token, str) or not TOKEN_RE.match(token):
            return error(400, "invalid_link", "This sign-in link is not valid.")
        return token

    async def auth_link(self, request: Request) -> Response:
        token = await self._token(request, await self.json_body(request))
        if isinstance(token, JSONResponse):
            return token
        info = await self.call(self.svc.store.login_token_info, self.svc.signer.hash("login", token),
                               self.s.terms_version)
        if info is None:
            return error(400, "invalid_link", "This sign-in link has expired or was already used. Ask for a new one.")
        return JSONResponse({"email": mask_email(info["email"]), "new_account": info["new_account"],
                             "needs_terms": info["needs_terms"], "terms_version": self.s.terms_version,
                             "terms_ready": self.terms_ready()})

    async def auth_verify(self, request: Request) -> Response:
        body = await self.json_body(request)
        token = await self._token(request, body)
        if isinstance(token, JSONResponse):
            return token
        if not self.terms_ready():
            return error(503, "signups_closed", "Sign-in is not open yet: the terms have not been published.")
        session = self.svc.signer.new_token()
        res = await self.call(self.svc.store.login_complete, self.svc.signer.hash("login", token),
                              self.svc.signer.hash("session", session), terms_version=self.s.terms_version,
                              accept_terms=body.get("accept_terms") is True, adult=body.get("adult") is True,
                              session_ttl_s=self.s.session_ttl_s)
        status = res["status"]
        if status == "needs_terms":
            return error(400, "needs_terms", "Please confirm that you are 18 or older and accept the terms.")
        if status != "ok":
            # One message for an unknown, used, expired, disabled or uninvited link.
            return error(400, "invalid_link", "This sign-in link has expired or was already used. Ask for a new one.")
        fresh = self.svc.signer.new_token()  # a new CSRF cookie at the privilege change
        resp = JSONResponse({"ok": True, "csrf": self.svc.signer.csrf_token(fresh)})
        self.set_cookie(resp, self.session_cookie, session, self.s.session_ttl_s)
        self.set_cookie(resp, self.csrf_cookie, fresh, self.s.session_ttl_s)
        event("signed_in", user=res["user"]["id"])
        return resp

    async def logout(self, request: Request) -> Response:
        raw = request.cookies.get(self.session_cookie)
        if raw and TOKEN_RE.match(raw):
            await self.call(self.svc.store.session_revoke, self.svc.signer.hash("session", raw))
        resp = JSONResponse({"ok": True})
        self.clear_cookie(resp, self.session_cookie)
        self.clear_cookie(resp, self.csrf_cookie)
        return resp

    async def delete_account(self, request: Request) -> Response:
        u = await self.require_user(request)
        body = await self.json_body(request)
        if body.get("confirm") != "delete my account":
            return error(400, "confirm", "Type the confirmation text to delete your account.")
        jobs = await self.call(self.svc.store.user_delete, u["id"])
        await self.call(self._delete_user_files, u["id"])
        event("account_deleted", user=u["id"], jobs=len(jobs))
        resp = JSONResponse({"ok": True})
        self.clear_cookie(resp, self.session_cookie)
        self.clear_cookie(resp, self.csrf_cookie)
        return resp

    def _delete_user_files(self, user_id: str) -> None:
        self.svc.storage.volume.refresh(0)
        self.svc.storage.delete_user(user_id)
        self.svc.storage.volume.commit()

    # ---- jobs -------------------------------------------------------------------------------------------

    async def quota(self, request: Request) -> Response:
        u = await self.require_user(request)
        return JSONResponse(await self.call(self.svc.store.usage, u["id"], self.s.jobs_per_day, self.s.jobs_total))

    async def jobs_list(self, request: Request) -> Response:
        u = await self.require_user(request)
        jobs = await self.call(self.svc.store.jobs_list, u["id"])
        return JSONResponse({"jobs": [self.job_summary(j) for j in jobs]})

    @staticmethod
    def job_summary(j: dict[str, Any]) -> dict[str, Any]:
        return {"id": j["id"], "kind": j["kind"], "name": j["name"], "profile": j["profile"], "state": j["state"],
                "created_at": j["created_at"], "finished_at": j["finished_at"], "stage": j["stage"],
                "stage_label": STAGE_LABELS.get(j["stage"] or "", None)}

    async def job_create(self, request: Request) -> Response:
        u = await self.require_user(request)
        key = request.headers.get("idempotency-key", "")
        if not IDEMPOTENCY_RE.match(key):
            return error(400, "idempotency_key", "Missing Idempotency-Key header (8 to 64 letters, digits, - or _).")
        if (r := await self.limit(f"jobs-user:{u['id']}", self.s.rate_jobs_user)) is not None:
            return r
        ctype = request.headers.get("content-type", "")
        if not ctype.startswith("multipart/form-data"):
            return error(415, "multipart", "Send the form as multipart/form-data.")
        try:
            form = await request.form(max_files=1, max_fields=6, max_part_size=self.s.upload_max_bytes + 1)
        except HTTPException:
            raise
        except Exception:  # noqa: BLE001  (a malformed multipart body)
            return error(400, "bad_form", "The form could not be read.")
        try:
            kind = form.get("kind")
            if kind not in ("sketch", "prompt"):
                return error(400, "bad_kind", "Choose a sketch upload or a description.")
            profile = form.get("profile") or self.s.default_profile
            if profile not in self.s.profiles:
                return error(400, "bad_profile", "Choose one of the offered profiles.")
            name = clean_name(form.get("name") if isinstance(form.get("name"), str) else None, self.s.name_max_chars)
            png, prompt, sha = None, "", ""
            if kind == "sketch":
                f = form.get("sketch")
                if not isinstance(f, UploadFile):
                    return error(400, "no_file", "Choose a sketch to upload.")
                data = await f.read(self.s.upload_max_bytes + 1)
                png, info = await self.call(self.svc.sanitize, data)
                sha = info["sha256"]
            else:
                prompt = clean_prompt(form.get("prompt"), self.s.prompt_max_chars)
                sha = hashlib.sha256(prompt.encode()).hexdigest()
        except UploadRejected as e:
            return error(400, "upload_rejected", str(e))
        finally:
            await form.close()
        job_id = new_id("j")
        if png is not None:
            await self.call(self._save_input, u["id"], job_id, png)
        try:
            job, created = await self.call(
                self.svc.store.job_create, job_id=job_id, user_id=u["id"], client_key=key, kind=kind, profile=profile,
                jobs_per_day=self.s.jobs_per_day, jobs_total=self.s.jobs_total,
                storage_max_bytes=self.s.user_storage_max_bytes, input_sha=sha, prompt=prompt, name=name,
                code_commit=self.s.code_commit)
        except (QuotaExceeded, Conflict, StoreError) as e:
            if e.code == "busy":
                # Not confirmed either way: the job may have been saved. Its sketch stays (so a saved job can run;
                # retention removes it if no job claims it), and sending the same form again with the same
                # Idempotency-Key returns that job instead of starting another.
                return busy_response("The service is busy and could not confirm your hero. Try again in a moment "
                                     "with the same Idempotency-Key: if it was saved, you get that hero, not a "
                                     "second one.")
            if png is not None:
                await self.call(self._drop_input, u["id"], job_id)
            status = {"quota": 429, "conflict": 409}.get(e.code, 400)
            return error(status, e.code, str(e))
        if not created and png is not None:
            await self.call(self._drop_input, u["id"], job_id)
        if created:
            event("job_created", job=job["id"], user=u["id"], kind=kind, profile=profile)
            try:
                await self.call(dispatch_queued, self.svc.store, self.s, self.svc.submit)
            except Exception:
                log.exception("dispatch after create failed")
        view = await self.job_view(job["id"], u["id"])
        return JSONResponse(view, status_code=201 if created else 200)

    def _save_input(self, user_id: str, job_id: str, png: bytes) -> None:
        self.svc.storage.save_input(user_id, job_id, png)
        self.svc.storage.volume.commit()

    def _drop_input(self, user_id: str, job_id: str) -> None:
        p = self.svc.storage.input_path(user_id, job_id)
        if p.exists():
            p.unlink()
            self.svc.storage.volume.commit()

    async def job_view(self, job_id: str, user_id: str) -> dict[str, Any] | None:
        job = await self.call(self.svc.store.job_get, job_id, user_id)
        if job is None:
            return None
        rows = {r["stage"]: r for r in await self.call(self.svc.store.job_stages, job_id)}
        files = await self.call(self._files, user_id, job)
        stages = []
        for spec in plan_for(job["kind"]):
            r = rows.get(spec.key)
            stages.append({"key": spec.key, "label": spec.label, "state": r["state"] if r else "pending",
                           "seconds": r["seconds"] if r else None, "cost_usd": r["cost_usd"] if r else None,
                           "message": r["message"] if r else ""})
        err = None
        if job["error_code"] and job["state"] in ("rejected", "failed", "queued"):
            err = {"stage": job["error_stage"], "stage_label": STAGE_LABELS.get(job["error_stage"] or ""),
                   "code": job["error_code"], "message": job["error_message"]}
        rid = f"{labapi.ROOT_KEY}/{job_id}"
        return {**self.job_summary(job), "prompt": job["prompt"], "started_at": job["started_at"],
                "attempts": job["attempts"], "stages": stages, "error": err, "seconds": job["seconds"],
                "cost_usd": job["cost_usd"], "cost_is_estimate": True,
                "cost_detail": json.loads(job["cost_detail"]) if job["cost_detail"] else None,
                "code_commit": job["code_commit"] or None, "images": files["images"],
                "downloads": files["downloads"], "roblox_readme": files["readme"],
                "viewer": f"/viewer#run={rid}"}

    def _files(self, user_id: str, job: dict[str, Any]) -> dict[str, Any]:
        st = self.svc.storage
        st.volume.refresh()
        run = st.run_dir(user_id, job["id"])
        rid = f"{labapi.ROOT_KEY}/{job['id']}"
        with st.volume.reading():
            images = {k: f"/runs/{rid}/{rel}" for k, rel in labapi.IMAGES.items()
                      if st.run_file(user_id, job["id"], rel) is not None}
            downloads: dict[str, Any] = {}
            done = job["state"] == "succeeded"
            glb = bundle.glb_path(run, "high") if done else None
            downloads["glb"] = ({"available": True, "url": f"/api/jobs/{job['id']}/download/glb",
                                 "bytes": glb.stat().st_size} if glb else
                                {"available": False, "reason": "The body is not finished yet." if not done
                                 else "This hero has no body file."})
            readme = None
            if done:
                try:
                    bundle.roblox_files(run)
                    downloads["roblox"] = {"available": True, "url": f"/api/jobs/{job['id']}/download/roblox"}
                    readme = bundle.roblox_readme(run)
                except bundle.BundleUnavailable as e:
                    rows = {r["stage"]: r for r in self.svc.store.job_stages(job["id"])}
                    six = rows.get("6")
                    reason = (six["message"] if six and six["state"] in ("skipped", "rejected", "failed")
                              and six["message"] else str(e))
                    downloads["roblox"] = {"available": False, "reason": reason}
            else:
                downloads["roblox"] = {"available": False, "reason": "The body is not finished yet."}
        return {"images": images, "downloads": downloads, "readme": readme}

    def _job_id(self, request: Request) -> str:
        jid = request.path_params["job"]
        if not valid_id(jid, "j"):
            raise HTTPException(404, "no such job")
        return jid

    async def job_get(self, request: Request) -> Response:
        u = await self.require_user(request)
        view = await self.job_view(self._job_id(request), u["id"])
        if view is None:
            return error(404, "not_found", "No such hero.")
        return JSONResponse(view)

    async def job_cancel(self, request: Request) -> Response:
        u = await self.require_user(request)
        try:
            await self.call(self.svc.store.job_cancel, self._job_id(request), u["id"])
        except NotFound:
            return error(404, "not_found", "No such hero.")
        return JSONResponse(await self.job_view(self._job_id(request), u["id"]))

    async def job_delete(self, request: Request) -> Response:
        u = await self.require_user(request)
        jid = self._job_id(request)
        try:
            await self.call(self.svc.store.job_delete, jid, u["id"])
        except NotFound:
            return error(404, "not_found", "No such hero.")
        await self.call(self._delete_files, u["id"], jid)
        return JSONResponse({"ok": True})

    def _delete_files(self, user_id: str, job_id: str) -> None:
        self.svc.storage.volume.refresh(0)
        self.svc.storage.delete_job(user_id, job_id)
        self.svc.storage.volume.commit()

    async def download(self, request: Request) -> Response:
        u = await self.require_user(request)
        jid = self._job_id(request)
        job = await self.call(self.svc.store.job_get, jid, u["id"])
        if job is None:
            return error(404, "not_found", "No such hero.")
        if job["state"] != "succeeded":
            return error(409, "not_ready", "This hero is not finished.")
        what = request.path_params["what"]
        if what not in DOWNLOADS:
            return error(404, "not_found", "Not found.")
        base = re.sub(r"[^a-z0-9-]+", "-", (job["name"] or "hero").lower()).strip("-")[:40] or "hero"
        st = self.svc.storage

        def read() -> tuple[bytes, str, str] | str:
            st.volume.refresh()
            run = st.run_dir(u["id"], jid)
            with st.volume.reading():
                if what == "glb":
                    lod = request.query_params.get("lod", "high")
                    p = bundle.glb_path(run, lod)
                    if p is None:
                        return "This hero has no body file at that level of detail."
                    return p.read_bytes(), "model/gltf-binary", f"herogen-{base}-{lod}.glb"
                try:
                    return bundle.roblox_bundle(run), "application/zip", f"herogen-{base}-roblox.zip"
                except bundle.BundleUnavailable as e:
                    return str(e)

        got = await self.call(read)
        if isinstance(got, str):
            return error(404, "not_available", got)
        data, ctype, filename = got
        event("download", job=jid, user=u["id"], what=what, bytes=len(data))
        return Response(data, media_type=ctype, headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    # ---- Lab API ----------------------------------------------------------------------------------------

    async def lab_workspace(self, request: Request) -> Response:
        u = await self.require_user(request)
        jobs = await self.call(self.svc.store.jobs_list, u["id"])

        def build() -> dict[str, Any]:
            self.svc.storage.volume.refresh()
            with self.svc.storage.volume.reading():
                return labapi.workspace(self.svc.storage.runs_root(u["id"]), [j["id"] for j in jobs])

        return JSONResponse(await self.call(build))

    async def _owned(self, request: Request) -> tuple[dict[str, Any], str] | Response:
        u = await self.require_user(request)
        if request.path_params["root"] != labapi.ROOT_KEY:
            return error(404, "not_found", "Not found.")
        jid = self._job_id(request)
        if await self.call(self.svc.store.job_get, jid, u["id"]) is None:
            return error(404, "not_found", "Not found.")
        return u, jid

    async def lab_run(self, request: Request) -> Response:
        got = await self._owned(request)
        if isinstance(got, Response):
            return got
        u, jid = got

        def build() -> dict[str, Any] | None:
            self.svc.storage.volume.refresh()
            with self.svc.storage.volume.reading():
                return labapi.run_detail(self.svc.storage.runs_root(u["id"]), jid)

        detail = await self.call(build)
        if detail is None:
            return error(404, "not_found", "Not found.")
        return JSONResponse(detail)

    async def run_file(self, request: Request) -> Response:
        got = await self._owned(request)
        if isinstance(got, Response):
            return got
        u, jid = got
        rel = request.path_params["path"]

        def read() -> tuple[bytes, str] | None:
            st = self.svc.storage
            st.volume.refresh()
            with st.volume.reading():
                f = st.run_file(u["id"], jid, rel)
                return (f[0].read_bytes(), f[1]) if f else None

        got2 = await self.call(read)
        if got2 is None:
            return error(404, "not_found", "Not found.")
        return Response(got2[0], media_type=got2[1])

    async def not_found(self, request: Request) -> Response:
        return error(404, "not_found", "Not found.")


def csrf_guard(app_state: App):
    """Pure ASGI middleware refusing unsafe methods that are cross-site or lack a valid CSRF token."""

    def wrap(app):
        async def guard(scope, receive, send):
            if scope["type"] == "http" and scope["method"] in UNSAFE:
                req = Request(scope)
                site = req.headers.get("sec-fetch-site")
                origin = req.headers.get("origin")
                why = None
                if site is not None and site != "same-origin":
                    why = "cross-site request"
                elif origin is not None and origin != app_state.s.origin:
                    why = "wrong origin"
                elif origin is None and site is None:
                    why = "no origin"
                elif not app_state.svc.signer.csrf_ok(req.cookies.get(app_state.csrf_cookie),
                                                      req.headers.get("x-csrf-token")):
                    why = "bad csrf token"
                if why:
                    event("csrf_refused", path=scope.get("path"), reason=why)
                    resp = error(403, "csrf", "This request was refused. Reload the page and try again.")
                    return await resp(scope, receive, send)
            return await app(scope, receive, send)

        return guard

    return wrap


def api_rate_guard(app_state: App):
    """Per-address request budget for the whole API, counted in this container's memory."""

    def wrap(app):
        async def guard(scope, receive, send):
            if scope["type"] == "http" and scope.get("path", "").startswith(("/api/", "/runs/")):
                req = Request(scope)
                rule = app_state.s.rate_api_ip
                ok, retry = app_state.svc.local_rates.hit(f"api:{app_state.client_ip(req)}", rule[0], rule[1])
                if not ok:
                    resp = error(429, "rate_limited", "Too many requests. Please wait a little and try again.",
                                 {"Retry-After": str(retry)})
                    return await resp(scope, receive, send)
            return await app(scope, receive, send)

        return guard

    return wrap


class _Wrap:
    def __init__(self, app, factory) -> None:
        self.inner = factory(app)

    async def __call__(self, scope, receive, send):
        return await self.inner(scope, receive, send)


def create_app(svc: Services) -> SecurityHeaders:
    a = App(svc)
    s = svc.settings
    upload_limit = s.upload_max_bytes + 256 * 1024

    async def http_error(request: Request, exc: HTTPException) -> Response:
        codes = {401: "signed_out", 404: "not_found", 405: "method", 413: "too_large", 415: "media_type"}
        msg = {401: "Sign in first.", 413: "The request is too large."}.get(exc.status_code, str(exc.detail))
        return error(exc.status_code, codes.get(exc.status_code, "bad_request"), msg)

    async def server_error(request: Request, exc: Exception) -> Response:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return error(500, "internal", "Something went wrong on our side. Please try again.")

    async def store_error(request: Request, exc: StoreError) -> Response:
        """A store refusal no handler caught: "busy" (the store is restarting or could not confirm a write) is a
        503 with Retry-After; the others keep their meaning."""
        if exc.code == "busy":
            return busy_response()
        status = {"quota": 429, "conflict": 409, "not_found": 404}.get(exc.code)
        if status is None:
            log.error("store refused %s %s: %s", request.method, request.url.path, exc.code)
            return error(500, "internal", "Something went wrong on our side. Please try again.")
        return error(status, exc.code, str(exc))

    routes = [
        Route("/", a.index), Route("/signin", a.index), Route("/viewer", a.viewer),
        Route("/terms", a.legal), Route("/privacy", a.legal), Route("/healthz", a.healthz),
        Route("/assets/{path:path}", a.assets), Route("/static/{path:path}", a.studio_assets),
        Route("/api/session", a.session),
        Route("/api/auth/request", a.auth_request, methods=["POST"], max_body_size=JSON_LIMIT),
        Route("/api/auth/link", a.auth_link, methods=["POST"], max_body_size=JSON_LIMIT),
        Route("/api/auth/verify", a.auth_verify, methods=["POST"], max_body_size=JSON_LIMIT),
        Route("/api/auth/logout", a.logout, methods=["POST"], max_body_size=JSON_LIMIT),
        Route("/api/account", a.delete_account, methods=["DELETE"], max_body_size=JSON_LIMIT),
        Route("/api/quota", a.quota),
        Route("/api/jobs", a.jobs_list, methods=["GET"]),
        Route("/api/jobs", a.job_create, methods=["POST"], max_body_size=upload_limit),
        Route("/api/jobs/{job}", a.job_get, methods=["GET"]),
        Route("/api/jobs/{job}", a.job_delete, methods=["DELETE"], max_body_size=JSON_LIMIT),
        Route("/api/jobs/{job}/cancel", a.job_cancel, methods=["POST"], max_body_size=JSON_LIMIT),
        Route("/api/jobs/{job}/download/{what:str}", a.download),
        Route("/api/workspace", a.lab_workspace),
        Route("/api/runs/{root}/{job}", a.lab_run),
        Route("/runs/{root}/{job}/{path:path}", a.run_file),
        Route("/api/{rest:path}", a.not_found, methods=["GET", "POST", "PUT", "PATCH", "DELETE"]),
    ]
    middleware = [
        Middleware(TrustedHostMiddleware, allowed_hosts=list(s.hosts()), www_redirect=False),
        Middleware(_Wrap, factory=api_rate_guard(a)),
        Middleware(_Wrap, factory=csrf_guard(a)),
    ]
    app = Starlette(routes=routes, middleware=middleware, max_body_size=upload_limit,
                    exception_handlers={HTTPException: http_error, StoreError: store_error, Exception: server_error})
    app.state.herogen = a
    # Outside Starlette's own error middleware, so its 500 responses carry the headers too.
    return SecurityHeaders(app, s)


