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
from herogen.service.auth import Signer
from herogen.service.store import Store


class TestSnapshotCommands(TempDirCase):
    def test_list_and_quarantine(self):
        from herogen.service.snapshot import LocalVolume, Snapshotter

        meta = self.tmp / "meta"
        snap = Snapshotter(meta, self.tmp / "local.db")
        snap.acquire()
        snap.restore()
        store = Store(snap.local, persist=snap.save)
        store.audit_add("fixture", "snapshot_test")
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
        from herogen.service.__main__ import dev_services
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
        self.assertTrue(svc.store.generation_get()["enabled"])
        svc.store.close()
