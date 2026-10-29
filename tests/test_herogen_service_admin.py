"""Admin commands (invites, allowlist, users, quotas, audit) and the dev-mode wiring."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from test_herogen_service_support import TempDirCase

from herogen.service import admin
from herogen.service.auth import Signer, normalize_invite
from herogen.service.store import Store


class TestAdmin(TempDirCase):
    def test_invites_allowlist_users_quota(self):
        store = Store(self.tmp / "meta.db")
        signer = Signer(b"k" * 48)
        out = admin.run("invite --count 2 --uses 3 --days 7 --note friends", store, signer)
        codes = [line.split()[0] for line in out.splitlines()]
        self.assertEqual(len(codes), 2)
        e = store.signin_eligibility("bob@example.com", signer.hash("invite", normalize_invite(codes[0])))
        self.assertTrue(e["eligible"])
        self.assertNotIn(codes[0], json.dumps(store.invite_list()), "codes are stored only as hashes")
        self.assertIn("0/3 used", admin.run("invites", store, signer))
        self.assertIn("allowed carol@example.com", admin.run("allow Carol@Example.com", store, signer))
        self.assertIn("carol@example.com", admin.run("allowlist", store, signer))
        with self.assertRaises(SystemExit):
            admin.run("allow nonsense", store, signer)
        from test_herogen_service_support import make_user

        make_user(store, "carol@example.com")
        self.assertIn("carol@example.com", admin.run("users", store, signer))
        q = json.loads(admin.run("quota carol@example.com --daily 1 --total default", store, signer))
        self.assertEqual((q["quota_daily"], q["quota_total"]), (1, None))
        self.assertIn("disabled", admin.run("disable carol@example.com", store, signer))
        self.assertIn("active", admin.run("enable carol@example.com", store, signer))
        self.assertIn("allow_add", admin.run("audit --limit 50", store, signer))
        iid = store.invite_list()[0]["id"]
        self.assertEqual(admin.run(f"revoke-invite {iid}", store, signer), "revoked")
        store.close()


class TestSnapshotCommands(TempDirCase):
    def test_list_and_quarantine(self):
        from herogen.service.snapshot import LocalVolume, Snapshotter

        meta = self.tmp / "meta"
        snap = Snapshotter(meta, self.tmp / "local.db")
        snap.acquire()
        snap.restore()
        store = Store(snap.local, persist=snap.save)
        store.allow_add("bob@example.com")
        store.close()
        signer = Signer(b"k" * 48)
        where = (meta, LocalVolume())
        listing = admin.run("snapshots", None, signer, where).splitlines()
        names = [n for n in listing if n.startswith("meta-")]
        self.assertGreaterEqual(len(names), 2)
        self.assertTrue(any(n.startswith("lease-") for n in listing))
        with self.assertLogs("herogen.service", "WARNING"):
            out = admin.run(f"snapshot-quarantine {names[-1]}", None, signer, where)
        self.assertIn(f"quarantined {names[-1]}", out)
        self.assertIn(f"starts from {names[-2]}", out)
        for n in names[:-2]:
            admin.run(f"snapshot-quarantine {n}", None, signer, where)
        with self.assertRaises(SystemExit) as cm:
            admin.run(f"snapshot-quarantine {names[-2]}", None, signer, where)
        self.assertIn("--allow-empty", str(cm.exception))
        with self.assertLogs("herogen.service", "WARNING"):
            out = admin.run(f"snapshot-quarantine {names[-2]} --allow-empty", None, signer, where)
        self.assertIn("empty database", out)
        with self.assertRaises(SystemExit):
            admin.run("snapshot-quarantine meta-nope.db", None, signer, where)
        self.assertIn("dev mode", admin.run("snapshots", None, signer))


class TestDevMode(TempDirCase):
    def test_dev_services_are_local_and_private(self):
        from herogen.service.__main__ import _first_start, dev_services
        from herogen.service.fakes import FakeRunner

        try:
            svc, backend = dev_services(self.tmp / "dev", 8899, "fake", 0.0, echo=False)
        except ImportError:
            self.skipTest("dev mode needs the web dependencies")
        self.assertEqual(svc.settings.mode, "dev")
        self.assertEqual(svc.settings.base_url, "http://127.0.0.1:8899")
        self.assertIsInstance(backend.runner, FakeRunner)
        key = self.tmp / "dev" / "dev-session-key"
        self.assertEqual(oct(key.stat().st_mode & 0o777), "0o600")
        self.assertEqual((self.tmp / "dev" / ".gitignore").read_text(), "*\n")
        with contextlib.redirect_stderr(io.StringIO()):
            _first_start(svc)
        self.assertEqual([r["email"] for r in svc.store.allow_list()], ["dev@example.com"])
        self.assertEqual(len(svc.store.invite_list()), 1)
        svc.store.close()
