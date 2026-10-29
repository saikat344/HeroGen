"""Shared helpers for the service tests (no tests here): a settable clock, synthetic images, a test store and
services wired with stand-ins. Nothing here calls a model, Modal or an email provider."""

from __future__ import annotations

import io
import logging
import secrets
import tempfile
import unittest
from pathlib import Path

from herogen.service.config import Settings

try:
    import starlette  # noqa: F401
    try:
        import httpx2  # noqa: F401
    except ImportError:
        import httpx  # noqa: F401
    HAVE_WEB = True
except ImportError:
    HAVE_WEB = False

# Tests check outcomes, not log lines: keep the service's warnings off the test output.
logging.getLogger("herogen.service").addHandler(logging.NullHandler())
logging.getLogger("herogen.service").propagate = False

web_test = unittest.skipUnless(HAVE_WEB, "the web tests need herogen/service/requirements-dev.txt")


class Clock:
    def __init__(self, t: float = 1_800_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def png(w: int = 400, h: int = 600, colour=(30, 30, 30), mode: str = "RGB", **save) -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new(mode, (w, h), (255, 255, 255) if mode == "RGB" else None)
    if mode == "RGB":
        ImageDraw.Draw(img).rectangle([w // 3, h // 6, 2 * w // 3, 5 * h // 6], fill=colour)
    buf = io.BytesIO()
    img.save(buf, format=save.pop("format", "PNG"), **save)
    return buf.getvalue()


def settings(tmp: Path, **kw) -> Settings:
    base = dict(mode="dev", base_url="http://testserver", data_dir=tmp / "data", meta_path=tmp / "meta.db",
                outbox_path=tmp / "outbox.jsonl", signing_key=secrets.token_urlsafe(48).encode(),
                terms_version="t1", allowed_hosts=("testserver",), jobs_per_day=3, jobs_total=5,
                max_running_jobs=2, rate_api_ip=(10_000, 60))
    base.update(kw)
    return Settings(**base)


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self) -> None:
        self._tmp.cleanup()


def make_user(store, email: str = "alice@example.com", terms: str = "t1", session: str | None = None) -> dict:
    """An active account, created through the real sign-in path (allowlist, link, terms). Its session's hash is
    sha256("ses-<email>") unless `session` names another value to hash."""
    import hashlib

    store.allow_add(email)
    h = hashlib.sha256(secrets.token_bytes(16)).hexdigest()
    store.login_token_create(h, email, 600)
    ses = hashlib.sha256((session or f"ses-{email}").encode()).hexdigest()
    res = store.login_complete(h, ses, terms_version=terms, accept_terms=True, adult=True, session_ttl_s=3600)
    assert res["status"] == "ok", res
    return res["user"]
