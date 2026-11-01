"""Secrets handling for sign-in: link tokens, session ids, CSRF tokens.

Every secret the service hands out is 256 random bits and is stored only
as an HMAC-SHA256 under a key derived from `HEROGEN_SESSION_KEY` for its purpose. A copy of the database alone
cannot recreate a link or a session, and a token made for one purpose is never valid for another.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets

TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")
EMAIL_MAX = 254
# Deliberately strict: one @, a dotted domain, no spaces, quotes, angle brackets or control characters.
_EMAIL = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64}@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
                    r"[A-Za-z]{2,63}$")
PURPOSES = ("login", "session", "csrf", "tag")


class Signer:
    def __init__(self, key: bytes) -> None:
        if len(key) < 32:
            raise ValueError("the signing key must be at least 32 bytes")
        self._keys = {p: hmac.new(key, b"herogen-service/" + p.encode(), hashlib.sha256).digest() for p in PURPOSES}

    def hash(self, purpose: str, value: str) -> str:
        return hmac.new(self._keys[purpose], value.encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def new_token() -> str:
        return secrets.token_urlsafe(32)

    def csrf_token(self, csrf_cookie: str) -> str:
        return self.hash("csrf", csrf_cookie)

    def csrf_ok(self, csrf_cookie: str | None, presented: str | None) -> bool:
        if not csrf_cookie or not presented or not TOKEN_RE.match(csrf_cookie):
            return False
        return hmac.compare_digest(self.csrf_token(csrf_cookie), presented)

    def tag(self, value: str) -> str:
        """A short keyed digest for rate-limit buckets and audit entries (never the raw address or IP)."""
        return self.hash("tag", value)[:16]


def normalize_email(raw: object) -> str | None:
    """The address as stored (trimmed, lower-cased), or None when it is not an address we accept."""
    if not isinstance(raw, str):
        return None
    email = raw.strip().lower()
    if not email or len(email) > EMAIL_MAX or not _EMAIL.match(email) or ".." in email:
        return None
    return email


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    shown = local[0] + "***" + (local[-1] if len(local) > 2 else "")
    return f"{shown}@{domain}"
