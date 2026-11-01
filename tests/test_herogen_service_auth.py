"""Admin sign-in: keyed hashes, addresses, magic-link expiry and reuse, sessions and email."""

from __future__ import annotations

import io
import json
import os
import stat
import sys
import urllib.error

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from test_herogen_service_support import Clock, TempDirCase, make_user

from herogen.service.auth import Signer, mask_email, normalize_email
from herogen.service.emailer import EmailError, FileSender, ResendSender, signin_message
from herogen.service.store import Store

KEY = b"k" * 48


class TestSigner(TempDirCase):
    def test_hashes_are_keyed_and_purpose_bound(self):
        a, b = Signer(KEY), Signer(b"j" * 48)
        t = Signer.new_token()
        self.assertEqual(len(t), 43)
        self.assertEqual(a.hash("login", t), a.hash("login", t))
        self.assertNotEqual(a.hash("login", t), b.hash("login", t), "another key gives another hash")
        self.assertNotEqual(a.hash("login", t), a.hash("session", t), "a link is never a session")
        self.assertNotIn(t, a.hash("login", t))
        with self.assertRaises(ValueError):
            Signer(b"short")

    def test_csrf_tokens(self):
        s = Signer(KEY)
        cookie = s.new_token()
        tok = s.csrf_token(cookie)
        self.assertTrue(s.csrf_ok(cookie, tok))
        self.assertFalse(s.csrf_ok(cookie, tok[:-1] + ("0" if tok[-1] != "0" else "1")))
        self.assertFalse(s.csrf_ok(s.new_token(), tok), "bound to the browser's cookie")
        self.assertFalse(s.csrf_ok(None, tok))
        self.assertFalse(s.csrf_ok(cookie, None))
        self.assertFalse(s.csrf_ok("short", s.csrf_token("short")), "only real cookie values count")
        self.assertFalse(Signer(b"j" * 48).csrf_ok(cookie, tok), "made with another key")

    def test_email_normalisation(self):
        self.assertEqual(normalize_email("  Alice@Example.COM "), "alice@example.com")
        for bad in ("", "alice", "a@b", "a@@example.com", "a b@example.com", "<a@example.com>", "a@example..com",
                    "a@example.com\r\nBcc: x@example.com", "a" * 65 + "@example.com", None, 5, ["a@example.com"],
                    "a@-example.com", "a@example.c"):
            self.assertIsNone(normalize_email(bad), repr(bad))
        self.assertEqual(mask_email("alice@example.com"), "a***e@example.com")
        self.assertEqual(mask_email("al@example.com"), "a***@example.com")


class TestMagicLinks(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        self.clock = Clock()
        self.store = Store(self.tmp / "meta.db", clock=self.clock)
        self.s = Signer(KEY)

    def tearDown(self) -> None:
        self.store.close()
        super().tearDown()

    def link(self, email: str, ttl=900) -> str:
        t = self.s.new_token()
        self.store.login_token_create(self.s.hash("login", t), email, ttl)
        return t

    def use(self, t: str, accept=True, adult=True, terms="t1"):
        return self.store.login_complete(self.s.hash("login", t), self.s.hash("session", self.s.new_token()),
                                         admin_emails=("alice@example.com", "bob@example.com"), session_ttl_s=3600)

    def test_link_is_single_use(self):

        t = self.link("alice@example.com")
        self.assertEqual(self.use(t)["status"], "ok")
        self.assertEqual(self.use(t)["status"], "invalid", "a used link never works again")
        self.assertIsNone(self.store.login_token_info(self.s.hash("login", t)))

    def test_link_expires(self):

        t = self.link("alice@example.com", ttl=900)
        self.clock.advance(899)
        self.assertIsNotNone(self.store.login_token_info(self.s.hash("login", t)))
        self.clock.advance(2)
        self.assertIsNone(self.store.login_token_info(self.s.hash("login", t)))
        self.assertEqual(self.use(t)["status"], "invalid")

    def test_using_one_link_kills_the_others(self):

        t1, t2 = self.link("alice@example.com"), self.link("alice@example.com")
        self.assertEqual(self.use(t2)["status"], "ok")
        self.assertEqual(self.use(t1)["status"], "invalid")

    def test_unknown_or_forged_tokens(self):

        t = self.link("alice@example.com")
        self.assertEqual(self.use(self.s.new_token())["status"], "invalid")
        other = Signer(b"z" * 48)
        res = self.store.login_complete(other.hash("login", t), "x" * 64, admin_emails=("alice@example.com",), session_ttl_s=60)
        self.assertEqual(res["status"], "invalid", "a token hashed with another key does not match")


    def test_disabled_users_cannot_sign_in(self):
        make_user(self.store)
        self.store.user_set_status("alice@example.com", "disabled")

        t = self.link("alice@example.com")
        self.assertEqual(self.use(t)["status"], "invalid")

    def test_audit_holds_no_address_or_token(self):

        t = self.link("alice@example.com")
        self.use(t)
        rows = [r for r in self.store.audit_list(100) if r["action"] != "allow_add"]
        blob = json.dumps(rows)
        self.assertNotIn("alice@example.com", blob)
        self.assertNotIn(t, blob)


class TestEmail(TempDirCase):
    def test_file_sender_is_private(self):
        p = self.tmp / "out" / "outbox.jsonl"
        f = FileSender(p)
        f.send(signin_message("alice@example.com", "https://hero.example.com/signin#t=abc", 15))
        self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
        msg = f.messages()[0]
        self.assertEqual(msg["to"], "alice@example.com")
        self.assertIn("https://hero.example.com/signin#t=abc", msg["text"])
        self.assertIn("15 minutes", msg["text"])

    def test_resend_request_and_errors(self):
        seen = {}

        class Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def opener(req, timeout):
            seen["url"], seen["headers"], seen["body"] = req.full_url, dict(req.header_items()), req.data
            return Resp()

        key = "re_" + "q" * 30
        r = ResendSender(key, "HeroGen <signin@example.com>", opener=opener)
        r.send(signin_message("alice@example.com", "https://x.example.com/signin#t=abc", 15))
        self.assertEqual(seen["url"], "https://api.resend.com/emails")
        self.assertEqual(seen["headers"]["Authorization"], f"Bearer {key}")
        body = json.loads(seen["body"])
        self.assertEqual((body["to"], body["from"]), (["alice@example.com"], "HeroGen <signin@example.com>"))

        def failing(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(key.encode()))

        with self.assertRaises(EmailError) as cm:
            ResendSender(key, "HeroGen <signin@example.com>", opener=failing).send(
                signin_message("alice@example.com", "https://x.example.com/signin#t=abc", 15))
        self.assertIn("401", str(cm.exception))
        self.assertNotIn(key, str(cm.exception))

        def down(req, timeout):
            raise urllib.error.URLError("no route")

        with self.assertRaises(EmailError):
            ResendSender(key, "x <a@example.com>", opener=down).send(signin_message("a@example.com", "l", 1))
        with self.assertRaises(ValueError):
            ResendSender("", "x <a@example.com>")
