"""Deployment: the Modal app's shape (with a stand-in `modal` module), settings and the remote-model bridge."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import sys
import types
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]

from test_herogen_service_support import TempDirCase

from herogen.service import config, modal_bridge


class _Chain:
    def __init__(self, log):
        self.log = log

    def __getattr__(self, name):
        def call(*a, **k):
            self.log.append((name, a, k))
            return self
        return call


def fake_modal():
    m = types.ModuleType("modal")
    m.functions = {}
    m.image_log = []

    class Named:
        def __init__(self, kind, name, **kw):
            self.kind, self.name, self.kw = kind, name, kw

    class App:
        def __init__(self, name):
            self.name = name

        def function(self, **kw):
            def deco(fn):
                m.functions[fn.__name__] = kw
                return fn
            return deco

        def cls(self, **kw):
            def deco(c):
                m.functions[c.__name__] = kw
                return c
            return deco

        def local_entrypoint(self, **kw):
            return lambda fn: fn

    def passthrough(*a, **k):
        return lambda fn: fn

    m.App = App
    m.Image = types.SimpleNamespace(debian_slim=lambda **k: _Chain(m.image_log))
    m.Volume = types.SimpleNamespace(from_name=lambda n, **k: Named("volume", n, **k))
    m.Secret = types.SimpleNamespace(from_name=lambda n, **k: Named("secret", n, **k))
    m.Period = lambda **k: ("period", k)
    m.Cron = lambda s: ("cron", s)
    m.enter = m.method = m.concurrent = m.asgi_app = passthrough
    m.is_local = lambda: True
    m.current_function_call_id = lambda: "fc-test"
    m.Error = type("Error", (Exception,), {})
    return m


PROD = {"HEROGEN_SERVICE_MODE": "production", "HEROGEN_SERVICE_BASE_URL": "https://hero.example.org",
        "HEROGEN_SERVICE_EMAIL_PROVIDER": "resend", "HEROGEN_SERVICE_EMAIL_FROM": "HeroGen <signin@hero.example.org>"}


class TestModalApp(TempDirCase):
    def load(self, env=None):
        fake = fake_modal()
        old = sys.modules.get("modal")
        sys.modules["modal"] = fake
        saved = dict(os.environ)
        os.environ.update(PROD if env is None else env)
        try:
            sys.modules.pop("herogen.service.modal_app", None)
            mod = importlib.import_module("herogen.service.modal_app")
        finally:
            os.environ.clear()
            os.environ.update(saved)
            if old is None:
                sys.modules.pop("modal", None)
            else:
                sys.modules["modal"] = old
            sys.modules.pop("herogen.service.modal_app", None)
        return mod, fake

    def test_function_configuration(self):
        mod, fake = self.load()
        f = fake.functions
        self.assertEqual(set(f), {"StoreService", "sanitize_image", "web", "run_job", "model_stages", "blender_stages",
                                  "export_stages", "heldout_stages", "sweep", "retention", "admin_command",
                                  "selfcheck_report"})
        for name, kw in f.items():
            self.assertNotIn("retries", kw, f"{name}: retries are the store's job, never Modal's")
        self.assertEqual((f["StoreService"]["max_containers"], f["StoreService"]["min_containers"]), (1, 1))
        san = f["sanitize_image"]
        self.assertTrue(san["block_network"])
        self.assertTrue(san["restrict_modal_access"])
        self.assertNotIn("secrets", san)
        self.assertNotIn("volumes", san)
        secrets = {n: [s.name for s in kw.get("secrets", [])] for n, kw in f.items()}
        self.assertEqual(secrets["web"], ["herogen-service-web"])
        self.assertEqual(secrets["model_stages"], ["herogen-service-model"])
        self.assertEqual(secrets["heldout_stages"], ["herogen-service-model"])
        self.assertEqual(f["heldout_stages"]["max_containers"], 2, "held-out views yield to the jobs' own work")
        self.assertEqual(secrets["admin_command"], ["herogen-service-web", "herogen-service-admin"])
        self.assertEqual(secrets["StoreService"], ["herogen-service-admin"])
        for n in ("blender_stages", "export_stages", "run_job", "sweep", "retention"):
            self.assertEqual(secrets[n], [], f"{n} holds no secret")
        self.assertNotIn("herogen-service-admin", secrets["web"])
        for n in ("run_job", "model_stages", "blender_stages", "export_stages", "heldout_stages"):
            self.assertTrue(f[n]["nonpreemptible"], n)
            self.assertTrue(f[n]["single_use_containers"], n)
        self.assertEqual(f["sweep"]["schedule"][0], "period")
        self.assertEqual(f["retention"]["schedule"][0], "cron")
        vols = {n: sorted(v.name for v in kw.get("volumes", {}).values()) for n, kw in f.items()}
        self.assertEqual(vols["StoreService"], ["herogen-service-meta"])
        self.assertEqual(vols["web"], ["herogen-service-data", "herogen-service-legal"])
        self.assertNotIn("herogen-service-meta", vols["web"], "the web never touches the metadata volume")
        self.assertEqual(vols["admin_command"], ["herogen-service-meta"], "for the snapshot commands")
        cmds = " ".join(str(a) for name, a, k in fake.image_log if name == "run_commands")
        self.assertIn(f"echo '{mod.BLENDER_SHA256}  /tmp/blender.tar.xz' | sha256sum -c -", cmds)
        self.assertIn("blender-4.5.", cmds)
        self.assertRegex(mod.IMAGE_ENV["HEROGEN_SERVICE_COMMIT"], r"^([0-9a-f]{40}(-dirty)?)?$")

    def test_staging_app_has_its_own_resources_and_no_secrets_in_image(self):
        mod, fake = self.load({**PROD, "HEROGEN_SERVICE_APP": "herogen-service-staging", "HEROGEN_SESSION_KEY": "s" * 40,
                               "HEROGEN_EMAIL_API_KEY": "e" * 40, "HEROGEN_SERVICE_JOBS_PER_DAY": "9"})
        self.assertEqual(mod.APP_NAME, "herogen-service-staging")
        f = fake.functions
        self.assertEqual([s.name for s in f["web"]["secrets"]], ["herogen-service-staging-web"])
        self.assertEqual(sorted(v.name for v in f["web"]["volumes"].values()),
                         ["herogen-service-staging-data", "herogen-service-staging-legal"])
        self.assertEqual(mod.IMAGE_ENV["HEROGEN_SERVICE_JOBS_PER_DAY"], "9")
        self.assertNotIn("HEROGEN_SESSION_KEY", mod.IMAGE_ENV)
        self.assertNotIn("HEROGEN_EMAIL_API_KEY", mod.IMAGE_ENV)
        envs = [k for name, a, k in fake.image_log if name == "env"] + [a[0] for name, a, k in fake.image_log
                                                                        if name == "env" and a]
        self.assertFalse(any("s" * 40 in json.dumps(e) or "e" * 40 in json.dumps(e) for e in envs))

    def test_deploy_refuses_dev_mode(self):
        for env in ({}, {**PROD, "HEROGEN_SERVICE_MODE": "dev"}):
            with self.assertRaises(RuntimeError) as cm:
                self.load(env)
            self.assertIn("dev mode", str(cm.exception))
        with self.assertRaises(config.ConfigError):
            self.load({**PROD, "HEROGEN_SERVICE_BASE_URL": "http://hero.example.org"})

    def test_timeouts_cover_the_stages(self):
        from herogen.service.runner import PLAN

        mod, fake = self.load()
        f = fake.functions
        for group, fn in (("model", "model_stages"), ("blender", "blender_stages"), ("export", "export_stages")):
            per = mod.DEPLOY.blender_timeout_s if group == "blender" else mod.DEPLOY.stage_timeout_s
            worst = sum(1 for s in PLAN if s.group == group) * per
            self.assertGreater(f[fn]["timeout"], worst, group)
        self.assertGreater(f["run_job"]["timeout"], sum(f[n]["timeout"] for n in ("model_stages", "blender_stages",
                                                                               "export_stages")))
        self.assertLessEqual(f["run_job"]["timeout"], 24 * 3600)
        self.assertGreater(f["heldout_stages"]["timeout"], mod.DEPLOY.stage_timeout_s)
        with self.assertRaises(RuntimeError):
            self.load({**PROD, "HEROGEN_SERVICE_BLENDER_TIMEOUT": str(12 * 3600)})

    def test_admin_methods_need_the_admin_key(self):
        from herogen.service.store import Store, StoreError

        mod, _ = self.load()
        svc = mod.StoreService()
        svc.store = Store(self.tmp / "m.db")
        self.assertEqual(svc.call("usage", ("u_x", 1, 1), {})["jobs_today"], 0)
        with self.assertRaises(StoreError) as cm:
            svc.call("allow_add", ("eve@example.com",), {})
        self.assertEqual(cm.exception.code, "bad_method")
        key = "k" * 48
        os.environ["HEROGEN_ADMIN_KEY"] = key
        try:
            for bad in ("", "x" * 48, None):
                with self.assertRaises(StoreError) as cm:
                    svc.admin_call(bad, "allow_add", ("eve@example.com",), {})
                self.assertEqual(cm.exception.code, "forbidden")
            with self.assertRaises(StoreError):
                svc.admin_call(key, "job_create", (), {})       # the admin path serves admin methods only
            svc.admin_call(key, "allow_add", ("eve@example.com",), {})
            self.assertEqual([r["email"] for r in svc.store.allow_list()], ["eve@example.com"])
            os.environ["HEROGEN_ADMIN_KEY"] = "short"
            with self.assertRaises(StoreError):
                svc.admin_call("short", "allow_list", (), {})  # a missing or weak key never opens the path
        finally:
            del os.environ["HEROGEN_ADMIN_KEY"]
            svc.store.close()

    def test_modal_failures_reach_the_web_as_busy(self):
        from herogen.service.store import StoreError

        mod, fake = self.load()

        class Method:
            def __init__(self, exc):
                self.exc = exc

            def remote(self, *a):
                raise self.exc

        rs = mod.RemoteStore()
        rs.handle = type("H", (), {"call": Method(fake.Error("container failed to start"))})()
        with self.assertLogs("herogen.service", "ERROR"), self.assertRaises(StoreError) as cm:
            rs.jobs_list("u_x")
        self.assertEqual(cm.exception.code, "busy")
        rs.handle = type("H", (), {"call": Method(StoreError("full", "quota"))})()
        with self.assertRaises(StoreError) as cm:
            rs.jobs_list("u_x")
        self.assertEqual(cm.exception.code, "quota", "the store's own refusals pass through")

    def test_compute_price(self):
        mod, _ = self.load()
        self.assertAlmostEqual(mod.price_per_s("blender"), (8 * 0.047 + 32 * 0.008) / 3600)


class TestSettings(TempDirCase):
    def test_every_setting_has_a_variable(self):
        from dataclasses import fields

        self.assertEqual({f.name for f in fields(config.Settings)}, set(config.ENV))
        self.assertEqual(len(set(config.ENV.values())), len(config.ENV))

    def test_parsing(self):
        s = config.from_env({"HEROGEN_SESSION_KEY": "k" * 40, "HEROGEN_SERVICE_JOBS_PER_DAY": "7",
                             "HEROGEN_SERVICE_RATE_AUTH_IP": "3/60", "HEROGEN_SERVICE_HOSTS": "a.example.com, b",
                             "HEROGEN_SERVICE_PROFILES": "roblox-avatar,best", "HEROGEN_SERVICE_PROFILE": "best",
                             "HEROGEN_SERVICE_DATA": "/data"})
        self.assertEqual((s.jobs_per_day, s.rate_auth_ip, s.allowed_hosts), (7, (3, 60), ("a.example.com", "b")))
        self.assertEqual((s.profiles, s.data_dir), (("roblox-avatar", "best"), Path("/data")))
        self.assertNotIn("k" * 40, repr(s), "secrets stay out of reprs and logs")

    def test_validation(self):
        good = {"HEROGEN_SESSION_KEY": "k" * 40}
        bad = [({}, "HEROGEN_SESSION_KEY"), ({"HEROGEN_SESSION_KEY": "short"}, "HEROGEN_SESSION_KEY"),
               ({**good, "HEROGEN_SERVICE_JOBS_PER_DAY": "x"}, "JOBS_PER_DAY"),
               ({**good, "HEROGEN_SERVICE_JOBS_PER_DAY": "0"}, "JOBS_PER_DAY"),
               ({**good, "HEROGEN_SERVICE_BASE_URL": "https://a.example.com/path"}, "BASE_URL"),
               ({**good, "HEROGEN_SERVICE_PROFILE": "secret"}, "PROFILE"),
               ({**good, "HEROGEN_SERVICE_MODE": "production"}, "https"),
               ({**good, "HEROGEN_SERVICE_MODE": "production", "HEROGEN_SERVICE_BASE_URL": "https://h.example.com"},
                "EMAIL_PROVIDER"),
               ({**good, "HEROGEN_SERVICE_MODE": "production", "HEROGEN_SERVICE_BASE_URL": "https://h.example.com",
                 "HEROGEN_SERVICE_EMAIL_PROVIDER": "resend"}, "HEROGEN_EMAIL_API_KEY"),
               ({**good, "HEROGEN_SERVICE_MODE": "production", "HEROGEN_SERVICE_BASE_URL": "https://h.example.com",
                 "HEROGEN_SERVICE_EMAIL_PROVIDER": "resend", "HEROGEN_EMAIL_API_KEY": "x" * 20}, "sending domain")]
        for env, needle in bad:
            with self.assertRaises(config.ConfigError) as cm:
                config.from_env(env)
            self.assertIn(needle, str(cm.exception), env)
            for v in env.values():
                if len(v) >= 20:
                    self.assertNotIn(v, str(cm.exception))
        config.from_env({}, require_key=False)
        prod = config.from_env({**good, "HEROGEN_SERVICE_MODE": "production",
                                "HEROGEN_SERVICE_BASE_URL": "https://h.example.org", "HEROGEN_EMAIL_API_KEY": "x" * 20,
                                "HEROGEN_SERVICE_EMAIL_PROVIDER": "resend",
                                "HEROGEN_SERVICE_EMAIL_FROM": "HeroGen <signin@h.example.org>"})
        self.assertTrue(prod.secure_cookies)
        self.assertEqual(prod.hosts(), ("h.example.org",))

    def test_deploy_env_drops_secrets(self):
        env = {"HEROGEN_SERVICE_JOBS_TOTAL": "9", "HEROGEN_SESSION_KEY": "s", "HEROGEN_EMAIL_API_KEY": "e",
               "MODEL_API_KEY": "m", "HOME": "/x"}
        self.assertEqual(config.env_for_deploy(env), {"HEROGEN_SERVICE_JOBS_TOTAL": "9"})


class TestBridge(TempDirCase):
    def runner(self, name: str, app: str) -> Path:
        p = self.tmp / f"{name}.py"
        p.write_text(f'import modal\nAPP = "{app}"\nGPU = __import__("os").environ.get("X", "A10G")\n')
        return p

    def test_objects_call_and_outputs(self):
        path = self.runner("objects", "herogen-sam3d-objects")
        (self.tmp / "img.png").write_bytes(b"IMG")
        (self.tmp / "mask.png").write_bytes(b"MASK")
        (self.tmp / "job.json").write_text(json.dumps({"jobs": [{"name": "job", "image": str(self.tmp / "img.png"),
                                                                 "mask": str(self.tmp / "mask.png")}], "seed": 7}))
        out = self.tmp / "out"
        out.mkdir()
        seen = {}

        class Fn:
            def remote(self, *a, **k):
                seen["a"], seen["k"] = a, k
                return {"runs": {"job": {"seconds": 1}}, "gpu": "x"}, {"job": {"mesh.glb": b"GLB"}}

        def lookup(app, fn):
            seen["app"], seen["fn"] = app, fn
            return Fn()

        os.environ["HEROGEN_OBJECTS_GPU"] = "A100-40GB"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = modal_bridge.main(["run", f"{path}::main", "--job", str(self.tmp / "job.json"), "--out",
                                        str(out)], lookup=lookup)
        finally:
            del os.environ["HEROGEN_OBJECTS_GPU"]
        self.assertEqual(rc, 0)
        self.assertEqual((seen["app"], seen["fn"]), ("herogen-sam3d-objects", "infer"))
        self.assertEqual(seen["a"][0], [{"name": "job", "image": b"IMG", "mask": b"MASK"}])
        self.assertEqual(seen["k"], {"seed": 7})
        self.assertEqual((out / "job" / "mesh.glb").read_bytes(), b"GLB")
        self.assertEqual(json.loads((out / "job" / "run.json").read_text()), {"seconds": 1})
        top = json.loads((out / "run.json").read_text())
        self.assertEqual(top["gpu_type"], "A100-40GB")
        self.assertIn("wall_seconds", top)

    def test_body_payload_and_refusals(self):
        (self.tmp / "img.png").write_bytes(b"IMG")
        args, kw = modal_bridge.payload("body", {"jobs": [{"name": "job", "image": str(self.tmp / "img.png"),
                                                           "bbox": [1, 2, 3, 4], "box_source": "mask"}],
                                                 "detector": None})
        self.assertEqual(args, [[{"name": "job", "png": b"IMG", "bbox": [1, 2, 3, 4], "box_source": "mask"}], None])
        self.assertEqual(kw, {})
        other = self.runner("other", "x")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(modal_bridge.main(["run", f"{other}::main", "--job", "j", "--out", "o"]), 2)
            self.assertEqual(modal_bridge.main(["deploy", "x"]), 2)
        with self.assertRaises(modal_bridge.BridgeError):
            modal_bridge.write_outputs(self.tmp, {"runs": {"job": {}}}, {"job": {"../evil": b"x"}})
