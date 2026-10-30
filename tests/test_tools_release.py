"""Release rules (tools/release/release.py): tag format, rc and final tags, the rc check, deploy settings."""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(_HERE), "tools", "release")]

import release as R  # noqa: E402

COMMIT = "c0ffee" + "0" * 34
OTHER = "decade" + "0" * 34
# Stand-in values; every test that prints checks none of them appears.
SETTINGS = {
    "MODAL_TOKEN_ID": "test-token-id-value",
    "MODAL_TOKEN_SECRET": "test-token-secret-value",
    "HEROGEN_SERVICE_APP": "herogen-service-staging",
    "HEROGEN_SERVICE_BASE_URL": "https://hero.example.org",
    "HEROGEN_SERVICE_EMAIL_FROM": "HeroGen <signin@hero.example.org>",
    "HEROGEN_SERVICE_TERMS_VERSION": "2026-10-01",
}


class FakeGit:
    """Tags by commit; commit_of answers for every tag listed."""

    def __init__(self, tags: dict[str, list[str]]) -> None:
        self.tags = tags
        self.asked: list[str] = []

    def commit_of(self, tag: str) -> str:
        return next((c for c, ts in self.tags.items() if tag in ts), "")

    def tags_at(self, commit: str) -> list[str]:
        self.asked.append(commit)
        return list(self.tags.get(commit, []))


def run_main(argv, env=None, git=None, root=None, run=subprocess.run):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = R.main(argv, env={} if env is None else env, git=git, root=root, run=run)
    return code, out.getvalue(), err.getvalue()


class TagFormat(unittest.TestCase):
    def test_rc_and_final(self):
        rc = R.parse_tag("v1.4.0-rc.2")
        self.assertEqual((rc.name, rc.version, rc.rc, rc.is_rc), ("v1.4.0-rc.2", "1.4.0", 2, True))
        final = R.parse_tag("v10.0.31")
        self.assertEqual((final.version, final.rc, final.is_rc), ("10.0.31", None, False))
        self.assertEqual(R.parse_tag("v0.0.0-rc.0").rc, 0)

    def test_refuses_other_v_tags(self):
        for name in ("v1", "v1.2", "v1.2.3.4", "1.2.3", "v01.2.3", "v1.02.3", "v1.2.03", "v1.2.3-rc", "v1.2.3-rc.",
                     "v1.2.3-rc.01", "v1.2.3-RC.1", "v1.2.3-rc1", "v1.2.3-beta.1", "v1.2.3-rc.1-x", "v1.2.3+build",
                     "V1.2.3", "v1.2.3 ", " v1.2.3", "v1.2.3\n", "v", "vnext", ""):
            with self.subTest(name=name):
                with self.assertRaises(R.ReleaseError) as e:
                    R.parse_tag(name)
                self.assertIn("is not a release tag", str(e.exception))
                self.assertIn("vX.Y.Z-rc.N", str(e.exception))

    def test_environment_selection(self):
        self.assertEqual(R.parse_tag("v1.4.0-rc.1").environment, R.STAGING)
        self.assertEqual(R.parse_tag("v1.4.0").environment, R.PRODUCTION)
        self.assertEqual((R.STAGING, R.PRODUCTION), ("staging", "production"))


class Plan(unittest.TestCase):
    def test_rc_goes_to_staging_without_asking_for_other_tags(self):
        git = FakeGit({COMMIT: ["v1.4.0-rc.1"]})
        p = R.plan("v1.4.0-rc.1", git)
        self.assertEqual((p.environment, p.commit, p.rc_tags), ("staging", COMMIT, ()))
        self.assertEqual(git.asked, [])

    def test_final_with_rc_on_the_same_commit_goes_to_production(self):
        git = FakeGit({COMMIT: ["v1.4.0", "v1.4.0-rc.3", "v1.4.0-rc.1", "v1.3.0-rc.1", "latest"]})
        p = R.plan("v1.4.0", git)
        self.assertEqual((p.environment, p.commit), ("production", COMMIT))
        self.assertEqual(p.rc_tags, ("v1.4.0-rc.1", "v1.4.0-rc.3"))
        self.assertEqual(git.asked, [COMMIT])
        self.assertIn("v1.4.0-rc.1, v1.4.0-rc.3", p.describe())

    def test_final_refused_without_an_rc_on_its_commit(self):
        cases = {
            "no rc at all": {COMMIT: ["v1.4.0"]},
            "rc on another commit": {COMMIT: ["v1.4.0"], OTHER: ["v1.4.0-rc.1"]},
            "rc of another version": {COMMIT: ["v1.4.0", "v1.3.0-rc.1", "v1.4.1-rc.1", "v1.4.0-rc"]},
        }
        for case, tags in cases.items():
            with self.subTest(case), self.assertRaises(R.ReleaseError) as e:
                R.plan("v1.4.0", FakeGit(tags))
            msg = str(e.exception)
            self.assertIn("cannot deploy to production", msg)
            self.assertIn(COMMIT, msg)
            self.assertIn("v1.4.0-rc.N", msg)

    def test_unknown_tag_and_bad_format(self):
        with self.assertRaises(R.ReleaseError) as e:
            R.plan("v1.4.0-rc.1", FakeGit({}))
        self.assertIn("not in this checkout", str(e.exception))
        git = FakeGit({COMMIT: ["v1.4"]})
        with self.assertRaises(R.ReleaseError):
            R.plan("v1.4", git)
        self.assertEqual(git.asked, [])

    def test_cli_writes_the_github_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            out_file, summary = Path(d, "out"), Path(d, "summary")
            env = {"GITHUB_OUTPUT": str(out_file), "GITHUB_STEP_SUMMARY": str(summary)}
            code, out, _ = run_main(["plan", "v1.4.0"], env, FakeGit({COMMIT: ["v1.4.0", "v1.4.0-rc.2"]}))
            self.assertEqual(code, 0)
            self.assertEqual(out_file.read_text(encoding="utf-8").splitlines(),
                             ["environment=production", "version=1.4.0", f"commit={COMMIT}"])
            self.assertIn("deploys to production", out)
            self.assertIn("deploys to production", summary.read_text(encoding="utf-8"))

    def test_cli_refusal_exits_1_and_annotates_on_github(self):
        with tempfile.TemporaryDirectory() as d:
            out_file = Path(d, "out")
            env = {"GITHUB_OUTPUT": str(out_file), "GITHUB_ACTIONS": "true"}
            code, out, err = run_main(["plan", "v1.4.0"], env, FakeGit({COMMIT: ["v1.4.0"]}))
            self.assertEqual(code, 1)
            self.assertFalse(out_file.exists())
            self.assertIn("cannot deploy to production", err)
            self.assertTrue(out.startswith("::error title=Release refused::"))
            self.assertEqual(out.count("\n"), 1)


class RootCase(unittest.TestCase):
    """A temporary repository root, with or without the service's Modal app."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.calls = []

    def entry(self):
        (self.root / R.ENTRY_PATH).parent.mkdir(parents=True)
        (self.root / R.ENTRY_PATH).write_text("app = None\n", encoding="utf-8")

    def run_(self, code=0):
        def run(cmd, cwd=None, env=None):
            self.calls.append((cmd, cwd, env))
            return subprocess.CompletedProcess(cmd, code)
        return run


class Config(unittest.TestCase):
    def test_complete(self):
        self.assertEqual(R.missing_config(SETTINGS), [])

    def test_names_exactly_what_is_missing(self):
        env = dict(SETTINGS, MODAL_TOKEN_SECRET="", HEROGEN_SERVICE_TERMS_VERSION="  ")
        del env["HEROGEN_SERVICE_BASE_URL"]
        missing = R.missing_config(env)
        self.assertEqual(missing, [("MODAL_TOKEN_SECRET", "secret"), ("HEROGEN_SERVICE_BASE_URL", "variable"),
                                   ("HEROGEN_SERVICE_TERMS_VERSION", "variable")])
        msg = R.config_message(missing, "production")
        self.assertEqual(msg.splitlines()[1:], ["  secret MODAL_TOKEN_SECRET", "  variable HEROGEN_SERVICE_BASE_URL",
                                                "  variable HEROGEN_SERVICE_TERMS_VERSION"])
        self.assertIn("missing 3 setting(s)", msg)
        self.assertIn("Settings > Environments > production", msg)
        for name in ("MODAL_TOKEN_ID", "HEROGEN_SERVICE_APP", "HEROGEN_SERVICE_EMAIL_FROM"):
            self.assertNotIn(name, msg)

    def test_nothing_set(self):
        msg = R.config_message(R.missing_config({}), "staging")
        self.assertEqual(len(msg.splitlines()), 1 + len(R.REQUIRED))
        self.assertEqual(R.REQUIRED, (
            ("MODAL_TOKEN_ID", "secret"), ("MODAL_TOKEN_SECRET", "secret"), ("HEROGEN_SERVICE_APP", "variable"),
            ("HEROGEN_SERVICE_BASE_URL", "variable"), ("HEROGEN_SERVICE_EMAIL_FROM", "variable"),
            ("HEROGEN_SERVICE_TERMS_VERSION", "variable")))


class Check(RootCase):
    def test_ready(self):
        self.entry()
        self.assertEqual(R.check("v2.0.0-rc.4", SETTINGS, self.root).environment, "staging")

    def test_reports_every_problem_at_once(self):
        with self.assertRaises(R.ReleaseError) as e:
            R.check("v2.0.0", dict(SETTINGS, HEROGEN_SERVICE_APP=""), self.root)
        msg = str(e.exception)
        self.assertIn("herogen/service/modal_app.py is not in this commit", msg)
        self.assertIn("must be merged", msg)
        self.assertIn("The production deploy is missing 1 setting(s)", msg)
        self.assertIn("  variable HEROGEN_SERVICE_APP", msg)
        self.entry()
        with self.assertRaises(R.ReleaseError) as e:
            R.check("v2.0.0", dict(SETTINGS, HEROGEN_SERVICE_APP=""), self.root)
        self.assertNotIn("modal_app.py", str(e.exception))

    def test_cli_never_prints_a_value(self):
        self.entry()
        env = dict(SETTINGS, MODAL_TOKEN_ID="", GITHUB_ACTIONS="true")
        for argv in (["check", "v2.0.0"], ["check", "v2.0.0-rc.4"]):
            with self.subTest(argv=argv):
                code, out, err = run_main(argv, env, root=self.root)
                self.assertEqual(code, 1)
                self.assertIn("secret MODAL_TOKEN_ID", err)
                self.assertIn(R.parse_tag(argv[1]).environment, err)
                for value in SETTINGS.values():
                    if value:
                        self.assertNotIn(value, out + err)
        code, out, _ = run_main(["check", "v2.0.0"], SETTINGS, root=self.root)
        self.assertEqual(code, 0)
        self.assertIn("production deploy needs is set", out)
        for value in SETTINGS.values():
            self.assertNotIn(value, out)

    def test_cli_entry_point_missing(self):
        code, _, err = run_main(["check", "v2.0.0"], SETTINGS, root=self.root)
        self.assertEqual(code, 1)
        self.assertIn("modal_app.py is not in this commit", err)

    def test_cli_refuses_a_bad_tag(self):
        self.entry()
        code, _, err = run_main(["check", "v2"], SETTINGS, root=self.root)
        self.assertEqual(code, 1)
        self.assertIn("is not a release tag", err)


class Deploy(RootCase):
    def test_runs_modal_deploy_with_only_the_service_settings(self):
        self.entry()
        env = dict(SETTINGS, PATH="/usr/bin", HEROGEN_SERVICE_MODE="dev", HEROGEN_SERVICE_JOBS_TOTAL="9",
                   HEROGEN_SERVICE_EMAIL_PROVIDER="file", MODAL_ENVIRONMENT="main")
        R.deploy("v1.4.0-rc.1", env, self.root, self.run_())
        (cmd, cwd, denv), = self.calls
        self.assertEqual(cmd, [sys.executable, "-m", "modal", "deploy", "-m", "herogen.service.modal_app",
                               "--tag", "v1.4.0-rc.1"])
        self.assertEqual(cwd, self.root)
        service = {k: v for k, v in denv.items() if k.startswith("HEROGEN_SERVICE_")}
        self.assertEqual(service, {
            "HEROGEN_SERVICE_APP": "herogen-service-staging", "HEROGEN_SERVICE_BASE_URL": "https://hero.example.org",
            "HEROGEN_SERVICE_EMAIL_FROM": "HeroGen <signin@hero.example.org>",
            "HEROGEN_SERVICE_TERMS_VERSION": "2026-10-01", "HEROGEN_SERVICE_MODE": "production",
            "HEROGEN_SERVICE_EMAIL_PROVIDER": "resend"})
        for k in ("PATH", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_ENVIRONMENT"):
            self.assertEqual(denv[k], env[k])

    def test_entry_point_missing(self):
        with self.assertRaises(R.ReleaseError) as e:
            R.deploy("v1.4.0", SETTINGS, self.root, self.run_())
        self.assertIn("herogen/service/modal_app.py is not in this commit", str(e.exception))
        self.assertIn("must be merged", str(e.exception))
        self.assertEqual(self.calls, [])

    def test_missing_settings_stop_before_modal(self):
        self.entry()
        with self.assertRaises(R.ReleaseError) as e:
            R.deploy("v1.4.0", dict(SETTINGS, MODAL_TOKEN_SECRET=""), self.root, self.run_())
        self.assertIn("secret MODAL_TOKEN_SECRET", str(e.exception))
        self.assertIn("production", str(e.exception))
        self.assertEqual(self.calls, [])

    def test_bad_tag_and_failed_deploy(self):
        self.entry()
        with self.assertRaises(R.ReleaseError):
            R.deploy("v1.4", SETTINGS, self.root, self.run_())
        self.assertEqual(self.calls, [])
        with self.assertRaises(R.ReleaseError) as e:
            R.deploy("v1.4.0", SETTINGS, self.root, self.run_(code=3))
        self.assertIn("status 3", str(e.exception))

    def test_cli(self):
        self.entry()
        summary = self.root / "summary"
        env = dict(SETTINGS, GITHUB_STEP_SUMMARY=str(summary))
        code, out, _ = run_main(["deploy", "v1.4.0"], env, root=self.root, run=self.run_())
        self.assertEqual(code, 0)
        self.assertIn("Deployed v1.4.0 to production as Modal app herogen-service-staging", out)
        self.assertIn("Deployed v1.4.0", summary.read_text(encoding="utf-8"))
        code, _, err = run_main(["deploy", "v1.4.0"], env, root=self.root, run=self.run_(code=1))
        self.assertEqual(code, 1)
        self.assertIn("modal deploy exited", err)


@unittest.skipUnless(shutil.which("git"), "git is not on PATH")
class LocalGitTags(unittest.TestCase):
    """LocalGit on a real repository: lightweight and annotated tags, peeled to their commit."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
                        GIT_AUTHOR_NAME="Test", GIT_AUTHOR_EMAIL="test@example.com", GIT_COMMITTER_NAME="Test",
                        GIT_COMMITTER_EMAIL="test@example.com")
        self.git("init", "-q")

    def git(self, *args):
        return subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
                              cwd=self.root, env=self.env, check=True, capture_output=True, text=True).stdout.strip()

    def commit(self, name):
        (self.root / name).write_text(name, encoding="utf-8")
        self.git("add", name)
        self.git("commit", "-q", "-m", name)
        return self.git("rev-parse", "HEAD")

    def test_plan_on_real_tags(self):
        first = self.commit("a")
        self.git("tag", "v1.0.0-rc.1")
        self.git("tag", "-a", "-m", "release", "v1.0.0")
        second = self.commit("b")
        self.git("tag", "-a", "-m", "release", "v1.1.0")
        self.git("tag", "v1.1.0-rc.1", first)
        g = R.LocalGit(self.root)
        self.assertEqual(g.commit_of("v1.0.0"), first)
        self.assertEqual(g.commit_of("v1.1.0"), second)
        self.assertEqual(g.commit_of("v9.9.9"), "")
        self.assertEqual(sorted(g.tags_at(first)), ["v1.0.0", "v1.0.0-rc.1", "v1.1.0-rc.1"])
        self.assertEqual(R.plan("v1.0.0", g).rc_tags, ("v1.0.0-rc.1",))
        self.assertEqual(R.plan("v1.1.0-rc.1", g).environment, "staging")
        with self.assertRaises(R.ReleaseError):
            R.plan("v1.1.0", g)


if __name__ == "__main__":
    unittest.main()
