"""Secrets handling for sign-in: link tokens, session ids, CSRF tokens and invite codes.

Every secret the service hands out is 256 random bits (invite codes: 80 bits, typed by people) and is stored only
as an HMAC-SHA256 under a key derived from `HEROGEN_SESSION_KEY` for its purpose. A copy of the database alone
cannot recreate a link, a session or an invite, and a token made for one purpose is never valid for another.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets

TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")
# Crockford base32 without I, L, O, U: unambiguous when read aloud or typed from a message.
INVITE_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
INVITE_GROUPS = 4
INVITE_GROUP_LEN = 4
INVITE_RE = re.compile(rf"^[{INVITE_ALPHABET}]{{{INVITE_GROUPS * INVITE_GROUP_LEN}}}$")
EMAIL_MAX = 254
# Deliberately strict: one @, a dotted domain, no spaces, quotes, angle brackets or control characters.
_EMAIL = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64}@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
                    r"[A-Za-z]{2,63}$")
PURPOSES = ("login", "session", "csrf", "invite", "tag")


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


def new_invite_code() -> str:
    """`XXXX-XXXX-XXXX-XXXX` in the invite alphabet (80 random bits)."""
    raw = "".join(secrets.choice(INVITE_ALPHABET) for _ in range(INVITE_GROUPS * INVITE_GROUP_LEN))
    return "-".join(raw[i:i + INVITE_GROUP_LEN] for i in range(0, len(raw), INVITE_GROUP_LEN))


def normalize_invite(raw: object) -> str | None:
    """An invite code as hashed: upper case, dashes and spaces removed, look-alike letters mapped (O to 0, I and L
    to 1). None when it cannot be a code."""
    if not isinstance(raw, str):
        return None
    code = re.sub(r"[\s-]", "", raw.strip().upper()).translate(str.maketrans("OIL", "011"))
    return code if INVITE_RE.match(code) else None
