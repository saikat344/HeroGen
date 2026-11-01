"""Best-effort warm submissions use fake Modal handles, never deployed apps."""
import ast
import os
import sys
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from herogen.blender import remote
from tests.test_herogen_remote_lifecycle import REMOTE, runner


class PrewarmTest(unittest.TestCase):
    def setUp(self):
        remote._last_spawn.clear()
        self.models = {name: Mock() for name in ("objects", "body")}
        self.lookup = Mock(side_effect=lambda app, cls: Mock(return_value=self.models[app.rsplit("-", 1)[-1]]))
        self.env = patch.dict(os.environ, {"HEROGEN_PREWARM": "1", "HEROGEN_OBJECTS_APP": "test-objects",
                                         "HEROGEN_BODY_APP": "test-body"})
        self.env.start()
        self.modules = patch.dict(sys.modules, {"modal": types.SimpleNamespace(Cls=Mock(from_name=self.lookup))})
        self.modules.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.modules.stop)
        self.addCleanup(remote._last_spawn.clear)

    def test_spawns_both_without_waiting_for_results(self):
        self.assertIsNone(remote.prewarm())
        self.assertEqual(self.lookup.call_args_list, [unittest.mock.call("test-objects", "Model"),
                                                      unittest.mock.call("test-body", "Model")])
        for model in self.models.values():
            model.warm.spawn.assert_called_once_with()
            model.warm.remote.assert_not_called()
            model.warm.spawn.return_value.get.assert_not_called()

    def test_cooldown_is_per_app_and_expires_at_sixty_seconds(self):
        with patch.object(remote.time, "monotonic", side_effect=[0, 59.999, 59.999, 60]):
            remote.prewarm(("objects",))
            remote.prewarm(("objects", "body"))
            remote.prewarm(("objects",))
        self.assertEqual(self.models["objects"].warm.spawn.call_count, 2)
        self.models["body"].warm.spawn.assert_called_once_with()

    def test_concurrent_calls_share_cooldown(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: remote.prewarm(), range(20)))
        for model in self.models.values():
            model.warm.spawn.assert_called_once_with()

    def test_disabled_does_not_import_modal_or_consume_cooldown(self):
        with patch.dict(os.environ, {"HEROGEN_PREWARM": "0"}), patch.dict(sys.modules, {"modal": None}):
            self.assertIsNone(remote.prewarm())
        self.assertEqual(remote._last_spawn, {})
        self.lookup.assert_not_called()
        remote.prewarm()
        self.assertEqual(self.lookup.call_count, 2)

    def test_failures_warn_once_continue_other_app_and_are_cooled_down(self):
        self.models["objects"].warm.spawn.side_effect = RuntimeError("submission failed")
        with self.assertLogs(remote.__name__, level="WARNING") as logs:
            self.assertIsNone(remote.prewarm())
            remote.prewarm()
        self.assertEqual(len(logs.records), 1)
        self.models["objects"].warm.spawn.assert_called_once_with()
        self.models["body"].warm.spawn.assert_called_once_with()

    def test_lookup_or_import_failure_and_invalid_kinds_do_not_raise(self):
        self.lookup.side_effect = RuntimeError("lookup failed")
        with self.assertLogs(remote.__name__, level="WARNING") as logs:
            remote.prewarm()
        self.assertEqual(len(logs.records), 1)
        remote._last_spawn.clear()
        with patch.dict(sys.modules, {"modal": None}), self.assertLogs(remote.__name__, level="WARNING"):
            remote.prewarm()
        with self.assertLogs(remote.__name__, level="WARNING"):
            remote.prewarm(("invalid",))
        with self.assertLogs(remote.__name__, level="WARNING"):
            remote.prewarm(None)

    def test_default_app_names(self):
        with patch.dict(os.environ, clear=True):
            remote.prewarm()
        self.assertEqual([call.args[0] for call in self.lookup.call_args_list],
                         ["herogen-sam3d-objects", "herogen-sam3d-body"])


class WarmMethodTest(unittest.TestCase):
    def test_warm_returns_after_entry_hook_without_inference(self):
        for name, args in (("objects.py", ()), ("body.py", (None,))):
            with self.subTest(name=name):
                ns = runner(name, {})
                state = ns["_model_state"] = Mock()
                ns["infer"] = Mock()
                model = ns["Model"]()
                model.load()
                self.assertIsNone(model.warm())
                state.assert_called_once_with(*args)
                ns["infer"].assert_not_called()

    def test_capacity_defaults_and_environment_overrides(self):
        for kind in ("objects", "body"):
            tree = ast.parse((REMOTE / f"{kind}.py").read_text())
            if kind == "objects":
                cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Model")
                keywords = cls.decorator_list[0].keywords
            else:
                node = next(n for n in tree.body if isinstance(n, ast.Assign) and
                            any(isinstance(t, ast.Name) and t.id == "_fn_kwargs" for t in n.targets))
                keywords = node.value.keywords
            selected = {k.arg: k.value for k in keywords if k.arg in
                        ("min_containers", "max_containers", "scaledown_window")}
            def values():
                return {k: eval(compile(ast.Expression(v), "settings", "eval"), {"os": os})
                        for k, v in selected.items()}
            with patch.dict(os.environ, clear=True):
                self.assertEqual(values(), {"min_containers": 0, "max_containers": 1, "scaledown_window": 300})
                with patch.dict(os.environ, {f"HEROGEN_{kind.upper()}_MIN_CONTAINERS": "2"}):
                    self.assertEqual(values()["min_containers"], 2)


if __name__ == "__main__":
    unittest.main()
