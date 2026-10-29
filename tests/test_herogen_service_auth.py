"""Sign-in: keyed hashes, addresses, invite codes, magic-link expiry and reuse, invite-only accounts, terms, email."""

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

from herogen.service.auth import Signer, mask_email, new_invite_code, normalize_email, normalize_invite
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

    def test_invite_codes(self):
        codes = {new_invite_code() for _ in range(200)}
        self.assertEqual(len(codes), 200)
        c = next(iter(codes))
        self.assertRegex(c, r"^[0-9A-HJKMNP-TV-Z]{4}(-[0-9A-HJKMNP-TV-Z]{4}){3}$")
        self.assertEqual(normalize_invite(c.lower().replace("-", " ")), c.replace("-", ""))
        self.assertEqual(normalize_invite("oooo-iiii-llll-0000"), "0000111111110000")
        for bad in ("", "abc", "XXXX-XXXX-XXXX-XXXU", None, 7, "X" * 17):
            self.assertIsNone(normalize_invite(bad), repr(bad))


class TestMagicLinks(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        self.clock = Clock()
        self.store = Store(self.tmp / "meta.db", clock=self.clock)
        self.s = Signer(KEY)

    def tearDown(self) -> None:
        self.store.close()
        super().tearDown()

    def link(self, email: str, invite_id=None, ttl=900) -> str:
        t = self.s.new_token()
        self.store.login_token_create(self.s.hash("login", t), email, ttl, invite_id)
        return t

    def use(self, t: str, accept=True, adult=True, terms="t1"):
        return self.store.login_complete(self.s.hash("login", t), self.s.hash("session", self.s.new_token()),
                                         terms_version=terms, accept_terms=accept, adult=adult, session_ttl_s=3600)

    def test_link_is_single_use(self):
        self.store.allow_add("alice@example.com")
        t = self.link("alice@example.com")
        self.assertEqual(self.use(t)["status"], "ok")
        self.assertEqual(self.use(t)["status"], "invalid", "a used link never works again")
        self.assertIsNone(self.store.login_token_info(self.s.hash("login", t), "t1"))

    def test_link_expires(self):
        self.store.allow_add("alice@example.com")
        t = self.link("alice@example.com", ttl=900)
        self.clock.advance(899)
        self.assertIsNotNone(self.store.login_token_info(self.s.hash("login", t), "t1"))
        self.clock.advance(2)
        self.assertIsNone(self.store.login_token_info(self.s.hash("login", t), "t1"))
        self.assertEqual(self.use(t)["status"], "invalid")

    def test_using_one_link_kills_the_others(self):
        self.store.allow_add("alice@example.com")
        t1, t2 = self.link("alice@example.com"), self.link("alice@example.com")
        self.assertEqual(self.use(t2)["status"], "ok")
        self.assertEqual(self.use(t1)["status"], "invalid")

    def test_unknown_or_forged_tokens(self):
        self.store.allow_add("alice@example.com")
        t = self.link("alice@example.com")
        self.assertEqual(self.use(self.s.new_token())["status"], "invalid")
        other = Signer(b"z" * 48)
        res = self.store.login_complete(other.hash("login", t), "x" * 64, terms_version="t1", accept_terms=True,
                                        adult=True, session_ttl_s=60)
        self.assertEqual(res["status"], "invalid", "a token hashed with another key does not match")

    def test_terms_and_age_are_required_for_a_new_account(self):
        self.store.allow_add("alice@example.com")
        t = self.link("alice@example.com")
        info = self.store.login_token_info(self.s.hash("login", t), "t1")
        self.assertEqual((info["new_account"], info["needs_terms"]), (True, True))
        self.assertEqual(self.use(t, accept=False)["status"], "needs_terms")
        self.assertEqual(self.use(t, adult=False)["status"], "needs_terms")
        self.assertEqual(self.use(t, terms="")["status"], "needs_terms", "no published terms, no accounts")
        self.assertEqual(self.use(t)["status"], "ok", "the link survives a needs_terms answer")
        user = self.store.users_list()[0]
        self.assertEqual(user["terms_version"], "t1")

    def test_new_terms_version_must_be_accepted_again(self):
        make_user(self.store, terms="t1")
        t = self.link("alice@example.com")
        self.assertTrue(self.store.login_token_info(self.s.hash("login", t), "t2")["needs_terms"])
        self.assertEqual(self.use(t, accept=False, terms="t2")["status"], "needs_terms")
        self.assertEqual(self.use(t, terms="t2")["status"], "ok")
        t = self.link("alice@example.com")
        self.assertFalse(self.store.login_token_info(self.s.hash("login", t), "t2")["needs_terms"])
        self.assertEqual(self.use(t, accept=False, terms="t2")["status"], "ok")

    def test_invite_only(self):
        elig = self.store.signin_eligibility("stranger@example.com", None)
        self.assertFalse(elig["eligible"])
        # A link made anyway (it never is for an ineligible address) still cannot create an account.
        t = self.link("stranger@example.com")
        self.assertEqual(self.use(t)["status"], "not_invited")
        self.assertEqual(self.store.users_list(), [])

    def test_invite_codes_are_counted_and_bound(self):
        code = normalize_invite(new_invite_code())
        inv = self.store.invite_create(self.s.hash("invite", code), max_uses=1, ttl_s=3600)
        e = self.store.signin_eligibility("bob@example.com", self.s.hash("invite", code))
        self.assertTrue(e["eligible"])
        self.assertEqual(e["invite_id"], inv["id"])
        self.assertTrue(self.store.signin_eligibility("bob@example.com", self.s.hash("invite", "0" * 16))["invalid_invite"])
        t_bob = self.link("bob@example.com", inv["id"])
        t_cat = self.link("cat@example.com", inv["id"])
        self.assertEqual(self.use(t_bob)["status"], "ok")
        self.assertEqual(self.use(t_cat)["status"], "not_invited", "the single use is spent")
        self.assertEqual(self.store.invite_list()[0]["uses"], 1)
        bound = self.store.invite_create(self.s.hash("invite", "1" * 16), 5, 3600, email="dan@example.com")
        self.assertFalse(self.store.signin_eligibility("eve@example.com", self.s.hash("invite", "1" * 16))["eligible"])
        self.assertTrue(self.store.signin_eligibility("dan@example.com", self.s.hash("invite", "1" * 16))["eligible"])
        self.store.invite_revoke(bound["id"])
        self.assertFalse(self.store.signin_eligibility("dan@example.com", self.s.hash("invite", "1" * 16))["eligible"])

    def test_expired_invite(self):
        self.store.invite_create(self.s.hash("invite", "2" * 16), 5, 60)
        self.clock.advance(61)
        self.assertFalse(self.store.signin_eligibility("bob@example.com", self.s.hash("invite", "2" * 16))["eligible"])

    def test_disabled_users_cannot_sign_in(self):
        make_user(self.store)
        self.store.user_set_status("alice@example.com", "disabled")
        self.assertFalse(self.store.signin_eligibility("alice@example.com", None)["eligible"])
        t = self.link("alice@example.com")
        self.assertEqual(self.use(t)["status"], "disabled")

    def test_audit_holds_no_address_or_token(self):
        self.store.allow_add("alice@example.com")
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
