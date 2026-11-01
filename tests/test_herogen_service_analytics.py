"""First-party website analytics, schema v7 migration, retention, admin panel API/CLI and browser behaviour."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from test_herogen_service_support import Clock, TempDirCase, make_user, png, settings
from test_herogen_service_web import Browser, WebCase
from tests.test_herogen_service_e2e import CDP_PORT, HAVE_WEB, ServiceBrowserCase, uvicorn

from herogen.service import admin, config
from herogen.service.analytics import MAIN_STAGE_ORDER, SIDE_STAGE_ORDER
from herogen.service.auth import Signer
from herogen.service.fakes import box_glb
from herogen.service.storage import Storage
from herogen.service.store import SCHEMA, Store, new_id
from herogen.service.web import EVENTS_LIMIT
from herogen.service.worker import purge


VID_A = "a" * 32
VID_B = "b" * 32


class TestAnalyticsConfigMigrationAndRetention(TempDirCase):
    def test_settings_parsing_validation_and_admin_allowlist(self):
        base = {"HEROGEN_SESSION_KEY": "k" * 40}
        default_s = config.from_env(base)
        self.assertEqual(default_s.admin_emails, ())
        self.assertEqual(default_s.analytics_retention_days, 30)
        self.assertFalse(default_s.is_admin({"id": "u_1", "email": "ops@example.com", "kind": "account"}))

        configured = config.from_env({
            **base,
            "HEROGEN_SERVICE_ADMIN_EMAILS": " Ops@Example.com , second@example.org , ",
            "HEROGEN_SERVICE_ANALYTICS_RETENTION_DAYS": "14",
        })
        self.assertEqual(configured.admin_emails, ("ops@example.com", "second@example.org"))
        self.assertEqual(configured.analytics_retention_days, 14)
        self.assertNotIn("ops@example.com", repr(configured))
        self.assertTrue(configured.is_admin({"id": "u_1", "email": "OPS@example.com", "kind": "account"}))
        self.assertFalse(configured.is_admin({"id": "u_2", "email": "ops@example.com", "kind": "guest"}))
        self.assertFalse(configured.is_admin({"id": "u_3", "email": "other@example.com", "kind": "account"}))
        self.assertFalse(configured.is_admin(None))

        for bad_env, expected in (
            ({**base, "HEROGEN_SERVICE_ADMIN_EMAILS": "not-an-email"}, "HEROGEN_SERVICE_ADMIN_EMAILS"),
            ({**base, "HEROGEN_SERVICE_ANALYTICS_RETENTION_DAYS": "0"}, "HEROGEN_SERVICE_ANALYTICS_RETENTION_DAYS"),
            ({**base, "HEROGEN_SERVICE_ANALYTICS_RETENTION_DAYS": "-5"}, "HEROGEN_SERVICE_ANALYTICS_RETENTION_DAYS"),
        ):
            with self.assertRaises(config.ConfigError) as cm:
                config.from_env(bad_env)
            self.assertIn(expected, str(cm.exception))
            self.assertNotIn("not-an-email", str(cm.exception))

    def test_v6_database_migrates_cleanly_to_v7_and_purges_by_retention(self):
        db_path = self.tmp / "v6.db"
        con = sqlite3.connect(str(db_path))
        v6_schema = SCHEMA.split("CREATE TABLE IF NOT EXISTS analytics_events")[0]
        con.executescript(v6_schema)
        con.execute("PRAGMA user_version = 6")
        con.execute("INSERT INTO allowlist(email, added_at, note) VALUES ('ops@example.com', 1000.0, 'v6 row')")
        con.commit()
        con.close()

        clock = Clock()
        s = settings(self.tmp, meta_path=db_path, analytics_retention_days=10)
        store = Store(db_path, clock=clock)
        try:
            ver = store.db.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(ver, 7)
            self.assertEqual([r["email"] for r in store.allow_list()], ["ops@example.com"])

            store.analytics_record(VID_A, "desktop", [("page_view", "page", "landing")])
            clock.advance(6 * 86400)
            store.analytics_record(VID_B, "mobile", [("page_view", "page", "gallery_detail")])
            clock.advance(5 * 86400)  # VID_A is 11d old (>10d), VID_B is 5d old (<10d)

            out = purge(store, s, Storage(s.data_dir))
            self.assertEqual(out["analytics_events"], 1)
            rep = store.analytics_report(30)
            self.assertEqual(rep["daily"]["visitors"], 1)
            self.assertEqual(rep["daily"]["by_page"]["landing"], 0)
            self.assertEqual(rep["daily"]["by_page"]["gallery_detail"], 1)
        finally:
            store.close()


class TestEventsApiAndPrivacy(WebCase):
    def test_valid_events_allowlist_csrf_rate_limit_body_cap_gpc_and_privacy(self):
        self.store.close()
        self.build(rate_api_ip=(9, 60))
        b = Browser(self.app)
        b.session()

        # CSRF required
        self.assertEqual(
            b.post("/api/events", json={"visitor_id": VID_A, "device": "desktop", "events": [{"name": "create_clicked"}]},
                   csrf=False).status_code,
            403,
        )

        # Valid batch covering allow-listed event shapes
        valid_batch = {
            "visitor_id": VID_A,
            "device": "desktop",
            "events": [
                {"name": "page_view", "props": {"page": "landing"}},
                {"name": "suggestion_click", "props": {"id": "forest"}},
                {"name": "starter_select", "props": {"id": "trail-keeper"}},
                {"name": "upload_selected"},
                {"name": "idea_typed"},
                {"name": "create_clicked"},
                {"name": "create_result", "props": {"result": "accepted"}},
                {"name": "shade_toggle", "props": {"mode": "clay"}},
                {"name": "picker_select", "props": {"id": "woodland-scout"}},
                {"name": "inspect_saved_run", "props": {"id": "woodland-scout"}},
                {"name": "viewer_drag", "props": {"surface": "landing"}},
                {"name": "result_viewed"},
                {"name": "download_click", "props": {"format": "glb"}},
                {"name": "theme_change", "props": {"theme": "dark"}},
            ],
        }
        r = b.post("/api/events", json=valid_batch, headers={"User-Agent": "SensitiveBrowser/99.0"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"ok": True, "recorded": 14})
        self.assertEqual(self.store.users_list(), [], "analytics never creates guest or user rows")
        self.assertNotIn("hg_session", dict(b.c.cookies), "no new session or tracking cookie")

        # Sec-GPC: 1 stores nothing
        r_gpc = b.post(
            "/api/events",
            json={"visitor_id": VID_B, "device": "mobile", "events": [{"name": "page_view", "props": {"page": "landing"}}]},
            headers={"Sec-GPC": "1"},
        )
        self.assertEqual(r_gpc.status_code, 200)
        self.assertEqual(r_gpc.json(), {"ok": True, "recorded": 0})

        # Invalid payloads return 400 and increment the rejection counter without echoing user data
        bad_payloads = [
            {"visitor_id": "short", "device": "desktop", "events": [{"name": "create_clicked"}]},
            {"visitor_id": VID_A, "device": "tablet", "events": [{"name": "create_clicked"}]},
            {"visitor_id": VID_A, "device": "desktop", "events": [{"name": "unknown_event"}]},
            {"visitor_id": VID_A, "device": "desktop", "events": [{"name": "upload_selected", "props": {"file": "secret.png"}}]},
            {"visitor_id": VID_A, "device": "desktop", "events": [{"name": "idea_typed", "props": {"prompt": "my secret"}}]},
        ]
        with self.assertLogs("herogen.service", "INFO") as logs:
            for bad in bad_payloads:
                resp = b.post("/api/events", json=bad)
                self.assertEqual((resp.status_code, resp.json()["error"]["code"]), (400, "bad_events"))
        self.assertEqual(self.app.app.state.herogen.events_rejected, len(bad_payloads))
        joined_logs = "\n".join(logs.output)
        self.assertIn("analytics_rejected", joined_logs)
        self.assertNotIn("secret.png", joined_logs)
        self.assertNotIn("my secret", joined_logs)

        # Verify database columns and stored content hold zero PII / UA / IP
        cols = [row[1] for row in self.store.db.execute("PRAGMA table_info(analytics_events)").fetchall()]
        self.assertEqual(cols, ["id", "at", "day", "visitor_id", "device", "event", "prop_key", "prop_val"])
        stored_dump = json.dumps(self.store._all("SELECT * FROM analytics_events"))
        self.assertNotIn("SensitiveBrowser", stored_dump)
        self.assertNotIn("127.0.0.1", stored_dump)
        self.assertNotIn("testclient", stored_dump)
        self.assertNotIn(VID_B, stored_dump)

        # Rate limit kicks in after configured per-IP budget (1 session + 1 valid + 1 gpc + 5 bad = 8 hits used)
        r_limited = b.post("/api/events", json={"visitor_id": VID_A, "device": "desktop", "events": [{"name": "create_clicked"}]})
        self.assertEqual((r_limited.status_code, r_limited.json()["error"]["code"]), (429, "rate_limited"))

    def test_events_body_cap_enforced(self):
        b = Browser(self.app)
        b.session()
        oversized = json.dumps({"visitor_id": VID_A, "device": "desktop", "pad": "x" * (EVENTS_LIMIT + 200)})
        r = b.post("/api/events", content=oversized, headers={"Content-Type": "application/json"})
        self.assertEqual(r.status_code, 413)


class TestAdminAuthSettingsAndAnalyticsReport(WebCase):
    def test_default_empty_admin_emails_grants_nobody(self):
        b = self.signin("alice@example.com")
        self.assertFalse(b.session()["is_admin"])
        for path in ("/admin", "/api/admin/analytics", "/api/admin/generation-budget"):
            self.assertEqual(b.get(path).status_code, 403, path)
        self.assertEqual(b.post("/api/admin/generation-budget", json={}).status_code, 403)

    def test_admin_allowlist_budget_save_and_report_parity(self):
        self.store.close()
        commit = "c" * 40
        self.build(
            public_demo=True,
            code_commit=commit,
            admin_emails=("ops@example.com",),
            max_running_jobs=5,
        )

        # 1. Signed-out visitor -> 403
        anon = Browser(self.app)
        anon.session()
        self.assertEqual(anon.get("/admin").status_code, 403)
        self.assertEqual(anon.get("/api/admin/analytics").status_code, 403)
        self.assertEqual(anon.get("/api/admin/generation-budget").status_code, 403)

        # 2. Guest workspace -> never admin (403)
        guest = Browser(self.app)
        guest.session()
        rg = guest.post("/api/demo/access", json={})
        self.assertEqual(rg.status_code, 200)
        guest.csrf = rg.json()["csrf"]
        self.assertFalse(guest.session()["is_admin"])
        self.assertEqual(guest.get("/admin").status_code, 403)
        self.assertEqual(guest.get("/api/admin/analytics").status_code, 403)
        self.assertEqual(guest.post("/api/admin/generation-budget", json={}).status_code, 403)

        # 3. Non-admin signed-in email -> 403
        member = self.signin("member@example.com")
        self.assertFalse(member.session()["is_admin"])
        self.assertEqual(member.get("/admin").status_code, 403)
        self.assertEqual(member.get("/api/admin/analytics").status_code, 403)

        # 4. Allowlisted admin email -> 200
        ops = self.signin("ops@example.com")
        self.assertTrue(ops.session()["is_admin"])
        admin_page = ops.get("/admin")
        self.assertEqual(admin_page.status_code, 200)
        self.assertNotIn("unsafe-inline", admin_page.headers["content-security-policy"])
        self.assertIn("style-src 'self'", admin_page.headers["content-security-policy"])

        # Empty-state report works cleanly before any jobs/events/budget
        empty_rep = ops.get("/api/admin/analytics").json()
        self.assertEqual(empty_rep["daily"]["visitors"], 0)
        self.assertEqual(empty_rep["refusals"], [])
        self.assertEqual([s["stage"] for s in empty_rep["stages"]["main"]], list(MAIN_STAGE_ORDER))
        self.assertEqual([s["stage"] for s in empty_rep["stages"]["side"]], list(SIDE_STAGE_ORDER))
        self.assertFalse(empty_rep["budget"]["configured"])

        # Days validation (1..90)
        for bad_days in ("0", "91", "-3", "abc"):
            self.assertEqual(ops.get(f"/api/admin/analytics?days={bad_days}").status_code, 400)
        self.assertEqual(ops.get("/api/admin/analytics?days=90").status_code, 200)

        # Initial unconfigured budget view
        b0 = ops.get("/api/admin/generation-budget").json()
        self.assertFalse(b0["configured"])
        self.assertIn("Paused", b0["sentence"])

        # Inline validation errors on POST /api/admin/generation-budget
        for bad_body, needle in (
            ({"enabled": True, "limit_usd": "10", "reserve_usd": "3", "max_active": 1, "days": 7,
              "bound_note": "note", "confirm_provider_caps": False}, "provider spending caps"),
            ({"enabled": True, "limit_usd": "2", "reserve_usd": "3", "max_active": 1, "days": 7,
              "bound_note": "note", "confirm_provider_caps": True}, "at least"),
            ({"enabled": True, "limit_usd": "10", "reserve_usd": "3", "max_active": 33, "days": 7,
              "bound_note": "note", "confirm_provider_caps": True}, "1 to 32"),
            ({"enabled": True, "limit_usd": "10", "reserve_usd": "3", "max_active": 1, "days": 0,
              "bound_note": "note", "confirm_provider_caps": True}, "Duration"),
            ({"enabled": True, "limit_usd": "10", "reserve_usd": "3", "max_active": 1, "days": 7,
              "bound_note": "   ", "confirm_provider_caps": True}, "bound note"),
        ):
            r_bad = ops.post("/api/admin/generation-budget", json=bad_body)
            self.assertEqual((r_bad.status_code, r_bad.json()["error"]["code"]), (400, "invalid_budget"))
            self.assertIn(needle, r_bad.json()["error"]["message"])

        # Save valid budget; verify spoofed client commit/signature are ignored and actor is admin's user id
        r_save = ops.post("/api/admin/generation-budget", json={
            "enabled": True,
            "limit_usd": "10",
            "reserve_usd": "3",
            "max_active": 1,
            "days": 14,
            "bound_note": "Measured stage bound",
            "confirm_provider_caps": True,
            "code_commit": "f" * 40,
            "cost_signature": "0" * 64,
        })
        self.assertEqual(r_save.status_code, 200, r_save.text)
        b1 = r_save.json()
        self.assertEqual(b1["sentence"], "Accepting heroes now: one at a time, up to $3 each, $10 in total")
        self.assertEqual(b1["policy_commit"], commit)
        self.assertEqual(b1["policy_cost_signature"], self.s.cost_signature)
        self.assertTrue(b1["commit_matches"])
        self.assertTrue(b1["cost_signature_matches"])

        ops_user = next(u for u in self.store.users_list() if u["email"] == "ops@example.com")
        audit_entry = next(a for a in self.store.audit_list(20) if a["action"] == "generation_budget_configured")
        self.assertEqual(audit_entry["actor"], ops_user["id"])

        # Seed jobs, stages, open reservation and events to verify full analytics report & CLI parity
        j_done = self.make_job(member, key="key-member-01").json()
        self.assertEqual(self.finish(j_done["id"]), "succeeded")
        self.store.admission_side_done(j_done["id"], unavailable_message="synthetic heldout skip")

        j_open = self.make_job(guest, key="key-guest-01").json()
        self.clock.advance(45)

        b_open = ops.get("/api/admin/generation-budget").json()
        self.assertEqual(b_open["active_reservations"], 1)
        self.assertEqual(b_open["open_reservations"][0]["job_id"], j_open["id"])
        self.assertAlmostEqual(b_open["open_reservations"][0]["age_seconds"], 45.0, places=1)
        self.assertEqual(b_open["used_usd"], 3.0)
        self.assertEqual(b_open["reserved_usd"], 3.0)
        self.assertEqual(b_open["remaining_usd"], 4.0)
        self.assertIn("generation_capacity", b_open["sentence"])

        # Record client events across desktop & mobile
        self.store.analytics_record(VID_A, "desktop", [
            ("page_view", "page", "landing"),
            ("suggestion_click", "id", "forest"),
            ("create_clicked", "", ""),
            ("create_result", "result", "accepted"),
            ("page_view", "page", "workbench"),
            ("result_viewed", "", ""),
            ("download_click", "format", "glb"),
            ("picker_select", "id", "woodland-scout"),
            ("shade_toggle", "mode", "clay"),
            ("inspect_saved_run", "id", "woodland-scout"),
            ("viewer_drag", "surface", "landing"),
            ("theme_change", "theme", "dark"),
        ])
        self.store.analytics_record(VID_B, "mobile", [
            ("page_view", "page", "landing"),
            ("create_clicked", "", ""),
            ("create_result", "result", "generation_capacity"),
        ])

        http_report = ops.get("/api/admin/analytics?days=7").json()
        cli_report = json.loads(admin.run(["analytics-report", "--days", "7"], self.store, self.svc.signer))
        self.assertEqual(http_report, cli_report)

        self.assertEqual(http_report["daily"]["visitors"], 2)
        self.assertEqual(http_report["daily"]["page_views"], 3)
        self.assertEqual(
            [(s["key"], s["count"]) for s in http_report["funnel"]["steps"]],
            [
                ("landing_view", 2),
                ("idea_started", 1),
                ("create_clicked", 2),
                ("accepted", 2),
                ("main_completed", 1),
                ("result_viewed", 1),
                ("download", 1),
            ],
        )
        self.assertEqual(http_report["refusals"], [{"reason": "generation_capacity", "count": 1}])
        self.assertEqual(http_report["jobs"]["total"], 2)
        self.assertEqual(http_report["jobs"]["by_final_state"]["succeeded"], 1)
        self.assertEqual(http_report["jobs"]["active"]["queued"], 1)
        self.assertEqual(http_report["jobs"]["by_user_kind"], {"account": 1, "guest": 1})
        self.assertEqual([s["stage"] for s in http_report["stages"]["main"]], list(MAIN_STAGE_ORDER))
        self.assertEqual([s["stage"] for s in http_report["stages"]["side"]], list(SIDE_STAGE_ORDER))
        self.assertEqual(http_report["stages"]["side"][0]["unavailable"], 1)
        self.assertIsNotNone(http_report["stages"]["job_totals"]["median_seconds"])
        self.assertEqual(http_report["gallery"]["picker_selections"], [{"id": "woodland-scout", "count": 1}])
        self.assertEqual(http_report["gallery"]["shade_toggles"]["clay"], 1)
        self.assertEqual(http_report["gallery"]["inspect_clicks"], 1)
        self.assertEqual(http_report["devices"]["desktop"]["visitors"], 1)
        self.assertEqual(http_report["devices"]["mobile"]["visitors"], 1)

        # Pause without provider-caps checkbox
        r_pause = ops.post("/api/admin/generation-budget", json={
            "enabled": False,
            "limit_usd": "10",
            "reserve_usd": "3",
            "max_active": 1,
            "days": 14,
            "bound_note": "Paused for maintenance",
            "confirm_provider_caps": False,
        })
        self.assertEqual(r_pause.status_code, 200)
        self.assertEqual(r_pause.json()["sentence"], "Paused")

        with self.assertRaises(SystemExit):
            admin.run(["analytics-report", "--days", "0"], self.store, self.svc.signer)


@unittest.skipUnless(HAVE_WEB and uvicorn is not None and CDP_PORT, "set E2E_CDP_PORT")
class TestAnalyticsAndAdminBrowser(ServiceBrowserCase):
    def setUp(self):
        self.gallery_tmp = TemporaryDirectory()
        root = Path(self.gallery_tmp.name)
        items = []
        for slug, title in [("woodland-scout", "Woodland Scout"), ("clockwork-bot", "Clockwork Bot")]:
            folder = root / slug
            folder.mkdir()
            assets = {}
            for kind, data, suffix in [("model", box_glb(), "glb"), ("artwork", png(), "png"), ("sketch", png(), "png")]:
                rel = f"{slug}/{kind}.{suffix}"
                (root / rel).write_bytes(data)
                assets[kind] = {"file": rel, "sha256": hashlib.sha256(data).hexdigest()}
            items.append({"id": slug, "title": title, "description": "Synthetic fixture.", "code_commit": "a" * 40, "assets": assets})
        (root / "catalog.json").write_text(json.dumps({"version": 1, "items": items}))
        self.service_settings = {
            "public_demo": True,
            "code_commit": "a" * 40,
            "gallery_dir": root,
            "admin_emails": ("ops@example.com",),
        }
        super().setUp()
        self.addCleanup(self.gallery_tmp.cleanup)

    def test_landing_events_gpc_and_admin_tabs_keyboard_mobile(self):
        t = self.tab
        # 1. Public visitor generates first-party events with sessionStorage-only visitor id
        t.goto(self.url + "/")
        t.wait("document.querySelector('#landingStatus').textContent === 'Drag to rotate · Scroll to zoom'")
        t.eval("document.querySelector('[data-idea=forest]').click()")
        t.eval("document.querySelector('[data-shade=clay]').click()")
        t.eval("document.querySelectorAll('#modelPicker button')[1].click()")
        t.eval("document.querySelector('#themePreference').value='dark'; document.querySelector('#themePreference').dispatchEvent(new Event('change'))")
        t.wait("sessionStorage.getItem('hg-vid') !== null")
        self.assertIsNone(t.eval("localStorage.getItem('hg-vid')"))
        t.eval("import('/assets/analytics.js').then(m => m.flush())")
        deadline = __import__("time").time() + 5
        while self.store.analytics_report(7)["daily"]["page_views"] < 1 and __import__("time").time() < deadline:
            __import__("time").sleep(0.05)
        rep = self.store.analytics_report(7)
        self.assertEqual(rep["daily"]["visitors"], 1)
        self.assertEqual(rep["funnel"]["idea_sources"]["suggestion"], 1)
        self.assertEqual(rep["gallery"]["shade_toggles"]["clay"], 1)
        self.assertEqual(rep["gallery"]["picker_selections"], [{"id": "clockwork-bot", "count": 1}])
        self.assertEqual(self.tab.errors, [])

        # 2. Unsigned visitor hitting /admin sees forbidden card, not admin panels
        t.goto(self.url + "/admin")
        t.wait("!document.querySelector('#adminForbidden').hidden")
        self.assertTrue(t.eval("document.querySelector('#adminShell').hidden"))

        # 3. Sign in as allowlisted admin and exercise Settings + Analytics tabs, keyboard arrows & 375px layout
        self.signin("ops@example.com")
        t.goto(self.url + "/admin")
        t.wait("!document.querySelector('#adminShell').hidden")
        self.assertEqual(t.eval("location.hash"), "#settings")
        self.assertEqual(t.eval("document.querySelector('#tab-settings').getAttribute('aria-selected')"), "true")

        # Missing provider-caps confirmation shows inline error
        t.eval(
            "document.querySelector('#modeAccepting').checked = true;"
            "document.querySelector('#limitUsd').value = '10';"
            "document.querySelector('#reserveUsd').value = '3';"
            "document.querySelector('#maxActive').value = '1';"
            "document.querySelector('#durationDays').value = '14';"
            "document.querySelector('#boundNote').value = 'Verified synthetic bound';"
            "document.querySelector('#confirmProviderCaps').checked = false;"
            "document.querySelector('#budgetForm').requestSubmit();"
        )
        t.wait("document.querySelector('#budgetFormMsg').textContent.includes('provider spending caps')")

        # Confirming provider caps saves and updates plain-English sentence
        t.eval(
            "document.querySelector('#confirmProviderCaps').checked = true;"
            "document.querySelector('#budgetForm').requestSubmit();"
        )
        t.wait("document.querySelector('#budgetSentence').textContent.includes('Accepting heroes now: one at a time, up to $3 each, $10 in total')")

        # Keyboard ArrowRight moves to #analytics tab and updates URL hash
        t.eval("document.querySelector('#tab-settings').focus(); document.querySelector('#tab-settings').dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowRight', bubbles: true}))")
        t.wait("!document.querySelector('#panel-analytics').hidden && location.hash === '#analytics'")
        t.wait("document.querySelector('#mainStagesWrap table') !== null")
        self.assertEqual(t.eval("document.activeElement.id"), "tab-analytics")

        # 375px mobile viewport has zero horizontal overflow on both tabs
        t.call("Emulation.setDeviceMetricsOverride", width=375, height=812, deviceScaleFactor=2, mobile=True)
        self.assertTrue(t.eval("document.documentElement.scrollWidth <= innerWidth"))
        t.eval("document.querySelector('#tab-analytics').dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowLeft', bubbles: true}))")
        t.wait("!document.querySelector('#panel-settings').hidden && location.hash === '#settings'")
        self.assertTrue(t.eval("document.documentElement.scrollWidth <= innerWidth"))
        self.assertEqual(self.tab.errors, [])

    def test_global_privacy_control_sends_zero_client_events(self):
        t = self.tab
        t.call("Page.addScriptToEvaluateOnNewDocument",
               source="Object.defineProperty(Navigator.prototype, 'globalPrivacyControl', {get: () => true})")
        t.goto(self.url + "/")
        t.wait("document.querySelector('#landingStatus').textContent === 'Drag to rotate · Scroll to zoom'")
        t.eval("document.querySelector('[data-idea=moon]').click()")
        t.eval("document.querySelector('[data-shade=wire]').click()")
        t.eval("import('/assets/analytics.js').then(m => m.flush())")
        __import__("time").sleep(0.3)
        self.assertIsNone(t.eval("sessionStorage.getItem('hg-vid')"))
        self.assertEqual(self.store.analytics_report(7)["daily"]["visitors"], 0)
        self.assertEqual(self.tab.errors, [])


if __name__ == "__main__":
    unittest.main()
