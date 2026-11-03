import unittest
from unittest.mock import patch

from herogen.blender.remote import deploy


class DeployCapacityTest(unittest.TestCase):
    def test_objects_runs_five_containers_at_once(self):
        # Fixed by the project owner: up to five heroes build their 3D shape at the same time.
        self.assertEqual(deploy.CAPACITY["objects"], {"MAX_CONTAINERS": 5, "MIN_CONTAINERS": 0})

    def test_the_callers_capacity_and_gpu_are_ignored(self):
        base = {"PATH": "/bin", "HEROGEN_OBJECTS_MAX_CONTAINERS": "1", "HEROGEN_OBJECTS_MIN_CONTAINERS": "3",
                "HEROGEN_OBJECTS_GPU": "H100", "HEROGEN_OBJECTS_APP": "other", "HEROGEN_BODY_GPU": "L4"}
        env = deploy.environment("objects", base)
        self.assertEqual(env["HEROGEN_OBJECTS_MAX_CONTAINERS"], "5")
        self.assertEqual(env["HEROGEN_OBJECTS_MIN_CONTAINERS"], "0")
        self.assertNotIn("HEROGEN_OBJECTS_GPU", env)
        self.assertNotIn("HEROGEN_OBJECTS_APP", env)
        self.assertEqual(env["HEROGEN_BODY_GPU"], "L4")    # another runner's settings are left alone
        self.assertEqual(env["PATH"], "/bin")

    def test_deploys_each_runner_file_with_its_settings(self):
        calls = []

        def run(cmd, env):
            calls.append((cmd, env))
            return type("Done", (), {"returncode": 0})()

        with patch.dict("os.environ", {"HEROGEN_MODAL": "/x/modal", "HEROGEN_BODY_MAX_CONTAINERS": "1"}), \
                patch.object(deploy.subprocess, "run", side_effect=run), patch("builtins.print"):
            self.assertEqual(deploy.main([]), 0)
        self.assertEqual([c[0][:2] for c in calls], [["/x/modal", "deploy"]] * 2)
        self.assertEqual([c[0][2].rsplit("/", 1)[1] for c in calls], ["body.py", "objects.py"])
        self.assertEqual(calls[0][1]["HEROGEN_BODY_MAX_CONTAINERS"], "5")
        self.assertEqual(calls[1][1]["HEROGEN_OBJECTS_MAX_CONTAINERS"], "5")

    def test_a_failed_deploy_stops_and_reports(self):
        with patch.dict("os.environ", {"HEROGEN_MODAL": "/x/modal"}), \
                patch.object(deploy.subprocess, "run", return_value=type("Done", (), {"returncode": 2})()) as run, \
                patch("builtins.print"):
            self.assertEqual(deploy.main(["objects", "body"]), 2)
        self.assertEqual(run.call_count, 1)

    def test_dry_run_deploys_nothing(self):
        with patch.object(deploy.subprocess, "run") as run, patch("builtins.print") as out:
            self.assertEqual(deploy.main(["objects", "--dry-run"]), 0)
        run.assert_not_called()
        self.assertIn("HEROGEN_OBJECTS_MAX_CONTAINERS=5", out.call_args_list[0].args[0])


if __name__ == "__main__":
    unittest.main()
