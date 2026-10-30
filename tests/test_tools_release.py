"""Release rules (tools/release/release.py): tag format, rc and final tags, the rc check, the staging deployment
check (with faked GitHub API answers, never the network), deploy settings."""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

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


class FakeDeployments:
    """Deployments as {id: (sha, [states newest first])}, all in staging; records what was asked."""

    def __init__(self, deployments: dict[int, tuple[str, list[str]]] | None = None) -> None:
        self.data = deployments or {}
        self.asked: list = []

    def deployments(self, environment, sha):
        self.asked.append(("deployments", environment, sha))
        return [{"id": i, "sha": s, "environment": environment} for i, (s, _) in self.data.items()]

    def states(self, deployment_id):
        self.asked.append(("states", deployment_id))
        return list(self.data[deployment_id][1])


def never():
    raise AssertionError("the deployments API was asked")


def staged():
    return FakeDeployments({7: (COMMIT, ["success", "in_progress"])})


TOKEN = "test-gh-token"
REPO = "owner/herogen"
API = "https://api.github.com"


def json_response(data, status=200, link=""):
    return R.Response(status, json.dumps(data).encode(), link)


class FakeFetch:
    """Answers by URL; records every request (URL and headers)."""

    def __init__(self, answers: dict[str, R.Response]) -> None:
        self.answers = answers
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, url, headers):
        self.requests.append((url, dict(headers)))
        if url not in self.answers:
            raise AssertionError(f"unexpected request {url}")
        return self.answers[url]


def deployments_url(sha=COMMIT, page=""):
    return f"{API}/repos/{REPO}/deployments?environment=staging&sha={sha}&per_page=100{page}"


def statuses_url(i, page=""):
    return f"{API}/repos/{REPO}/deployments/{i}/statuses?per_page=100{page}"


def run_main(argv, env=None, git=None, root=None, run=subprocess.run, fetch=None):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = R.main(argv, env={} if env is None else env, git=git, root=root, run=run,
                      fetch=fetch or FakeFetch({}))
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
    def test_rc_goes_to_staging_without_asking_for_other_tags_or_deployments(self):
        git = FakeGit({COMMIT: ["v1.4.0-rc.1"]})
        p = R.plan("v1.4.0-rc.1", git, never)
        self.assertEqual((p.environment, p.commit, p.rc_tags), ("staging", COMMIT, ()))
        self.assertEqual(git.asked, [])

    def test_final_with_rc_and_staging_success_goes_to_production(self):
        git = FakeGit({COMMIT: ["v1.4.0", "v1.4.0-rc.3", "v1.4.0-rc.1", "v1.3.0-rc.1", "latest"]})
        api = staged()
        p = R.plan("v1.4.0", git, lambda: api)
        self.assertEqual((p.environment, p.commit, p.staging_deployment), ("production", COMMIT, 7))
        self.assertEqual(p.rc_tags, ("v1.4.0-rc.1", "v1.4.0-rc.3"))
        self.assertEqual(git.asked, [COMMIT])
        self.assertEqual(api.asked[0], ("deployments", "staging", COMMIT))
        self.assertIn("v1.4.0-rc.1, v1.4.0-rc.3 and staging deployment 7 of it succeeded", p.describe())

    def test_final_refused_without_an_rc_on_its_commit(self):
        cases = {
            "no rc at all": {COMMIT: ["v1.4.0"]},
            "rc on another commit": {COMMIT: ["v1.4.0"], OTHER: ["v1.4.0-rc.1"]},
            "rc of another version": {COMMIT: ["v1.4.0", "v1.3.0-rc.1", "v1.4.1-rc.1", "v1.4.0-rc"]},
        }
        for case, tags in cases.items():
            with self.subTest(case):
                with self.assertRaises(R.ReleaseError) as e:
                    R.plan("v1.4.0", FakeGit(tags), never)
                msg = str(e.exception)
                self.assertIn("cannot deploy to production", msg)
                self.assertIn(COMMIT, msg)
                self.assertIn("v1.4.0-rc.N", msg)

    def test_rc_and_final_pushed_together_do_not_deploy_production(self):
        # The final's plan runs while the rc's staging deploy is still going.
        api = FakeDeployments({7: (COMMIT, ["in_progress", "queued"])})
        with self.assertRaises(R.ReleaseError) as e:
            R.plan("v1.4.0", FakeGit({COMMIT: ["v1.4.0", "v1.4.0-rc.1"]}), lambda: api)
        self.assertTrue(str(e.exception).startswith("v1.4.0 cannot deploy to production yet"))
        self.assertIn("Re-run this workflow once it has succeeded", str(e.exception))

    def test_unknown_tag_and_bad_format(self):
        with self.assertRaises(R.ReleaseError) as e:
            R.plan("v1.4.0-rc.1", FakeGit({}), never)
        self.assertIn("not in this checkout", str(e.exception))
        git = FakeGit({COMMIT: ["v1.4"]})
        with self.assertRaises(R.ReleaseError):
            R.plan("v1.4", git, never)
        self.assertEqual(git.asked, [])

    def test_cli_writes_the_github_outputs(self):
        fetch = FakeFetch({deployments_url(): json_response([{"id": 7, "sha": COMMIT}]),
                           statuses_url(7): json_response([{"state": "inactive"}, {"state": "success"}])})
        with tempfile.TemporaryDirectory() as d:
            out_file, summary = Path(d, "out"), Path(d, "summary")
            env = {"GITHUB_OUTPUT": str(out_file), "GITHUB_STEP_SUMMARY": str(summary), "GITHUB_REPOSITORY": REPO,
                   "GITHUB_TOKEN": TOKEN}
            code, out, err = run_main(["plan", "v1.4.0"], env, FakeGit({COMMIT: ["v1.4.0", "v1.4.0-rc.2"]}),
                                      fetch=fetch)
            self.assertEqual(code, 0, err)
            self.assertEqual(out_file.read_text(encoding="utf-8").splitlines(),
                             ["environment=production", "version=1.4.0", f"commit={COMMIT}"])
            self.assertIn("deploys to production", out)
            self.assertIn("deploys to production", summary.read_text(encoding="utf-8"))
            self.assertNotIn(TOKEN, out + err + summary.read_text(encoding="utf-8"))

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

    def test_cli_final_needs_the_api_settings(self):
        code, _, err = run_main(["plan", "v1.4.0"], {"GITHUB_REPOSITORY": REPO},
                                FakeGit({COMMIT: ["v1.4.0", "v1.4.0-rc.1"]}))
        self.assertEqual(code, 1)
        self.assertIn("needs GITHUB_TOKEN", err)
        code, out, _ = run_main(["plan", "v1.4.0-rc.1"], {}, FakeGit({COMMIT: ["v1.4.0-rc.1"]}))
        self.assertEqual(code, 0)
        self.assertIn("deploys to staging", out)


class StagingCheck(unittest.TestCase):
    """staging_succeeded on deployments with each kind of history."""

    def refused(self, data):
        with self.assertRaises(R.ReleaseError) as e:
            R.staging_succeeded(COMMIT, FakeDeployments(data))
        return str(e.exception)

    def test_success(self):
        self.assertEqual(R.staging_succeeded(COMMIT, staged()), 7)

    def test_success_later_made_inactive_still_counts(self):
        api = FakeDeployments({9: (COMMIT, ["failure"]), 7: (COMMIT, ["inactive", "success", "in_progress"])})
        self.assertEqual(R.staging_succeeded(COMMIT, api), 7)

    def test_failure_only(self):
        msg = self.refused({7: (COMMIT, ["failure", "in_progress"]), 8: (COMMIT, ["error"])})
        self.assertIn("no staging deployment of commit", msg)
        self.assertIn("succeeded (latest states 7: failure, 8: error)", msg)
        self.assertIn("then re-run this workflow", msg)

    def test_in_progress(self):
        for states in (["in_progress"], ["queued"], ["pending"], []):
            with self.subTest(states=states):
                msg = self.refused({7: (COMMIT, ["failure"]), 8: (COMMIT, states)})
                self.assertIn("still in progress (deployment 8)", msg)

    def test_none(self):
        msg = self.refused({})
        self.assertIn(f"the staging environment has no deployment of commit {COMMIT}", msg)
        self.assertIn("Let its rc tag deploy to staging, then re-run this workflow", msg)

    def test_a_different_sha_does_not_count(self):
        msg = self.refused({7: (OTHER, ["success"])})
        self.assertIn("has no deployment of commit", msg)


class GitHubApi(unittest.TestCase):
    """GitHubDeployments against faked API answers: requests, pages and errors."""

    def api(self, answers, **kw):
        self.fetch = FakeFetch(answers)
        return R.GitHubDeployments(REPO, TOKEN, API, self.fetch, **kw)

    def test_requests_and_token(self):
        api = self.api({deployments_url(): json_response([{"id": 7, "sha": COMMIT}]),
                        statuses_url(7): json_response([{"state": "success"}])})
        self.assertEqual(R.staging_succeeded(COMMIT, api), 7)
        for url, headers in self.fetch.requests:
            self.assertNotIn(TOKEN, url)
            self.assertEqual(headers["Authorization"], f"Bearer {TOKEN}")
            self.assertEqual(headers["Accept"], "application/vnd.github+json")
            self.assertEqual(headers["X-GitHub-Api-Version"], "2022-11-28")

    def test_pages(self):
        page2 = deployments_url(page="&page=2")
        spage2 = statuses_url(8, "&page=2")
        api = self.api({
            deployments_url(): json_response([{"id": 7, "sha": COMMIT}],
                                             link=f'<{page2}>; rel="next", <x>; rel="last"'),
            page2: json_response([{"id": 8, "sha": COMMIT}], link=f'<{deployments_url()}>; rel="first"'),
            statuses_url(7): json_response([{"state": "failure"}]),
            statuses_url(8): json_response([{"state": "inactive"}], link=f'<{spage2}>; rel="next"'),
            spage2: json_response([{"state": "success"}]),
        })
        self.assertEqual([d["id"] for d in api.deployments("staging", COMMIT)], [7, 8])
        self.assertEqual(api.states(8), ["inactive", "success"])
        self.assertEqual(R.staging_succeeded(COMMIT, api), 8)

    def test_next_page_elsewhere_is_refused(self):
        api = self.api({deployments_url(): json_response([], link='<https://evil.example.com/x>; rel="next"')})
        with self.assertRaises(R.ReleaseError) as e:
            api.deployments("staging", COMMIT)
        self.assertIn("next page is not on https://api.github.com", str(e.exception))
        self.assertEqual(len(self.fetch.requests), 1)

    def test_too_many_pages(self):
        api = self.api({deployments_url(): json_response([], link=f'<{deployments_url()}>; rel="next"')}, max_pages=3)
        with self.assertRaises(R.ReleaseError) as e:
            api.deployments("staging", COMMIT)
        self.assertIn("more than 3 pages", str(e.exception))

    def test_http_errors_are_refused_clearly(self):
        cases = {401: "token was refused (401)", 403: "`deployments: read` permission", 404: "(404)",
                 500: "failed (500); re-run this workflow later", 502: "failed (502)", 503: "failed (503)",
                 302: "answered 302"}
        for status, expect in cases.items():
            with self.subTest(status=status):
                api = self.api({deployments_url(): R.Response(status, b"")})
                with self.assertRaises(R.ReleaseError) as e:
                    R.plan("v1.4.0", FakeGit({COMMIT: ["v1.4.0", "v1.4.0-rc.1"]}), lambda: api)
                msg = str(e.exception)
                self.assertIn(expect, msg)
                self.assertIn("could not read the staging deployments", msg)
                self.assertIn("Production is not deployed until this check passes", msg)
                self.assertNotIn(TOKEN, msg)
        api = self.api({deployments_url(): json_response([{"id": 7, "sha": COMMIT}]),
                        statuses_url(7): R.Response(403, b"")})
        with self.assertRaises(R.ReleaseError) as e:
            R.staging_succeeded(COMMIT, api)
        self.assertIn("could not read the statuses of deployment 7", str(e.exception))

    def test_unreadable_answers(self):
        for body in (b"not json", b'{"message": "x"}'):
            with self.subTest(body=body):
                api = self.api({deployments_url(): R.Response(200, body)})
                with self.assertRaises(R.ReleaseError) as e:
                    api.deployments("staging", COMMIT)
                self.assertIn("is not a list", str(e.exception))

    def test_from_env(self):
        api = R.GitHubDeployments.from_env({"GITHUB_REPOSITORY": REPO, "GITHUB_TOKEN": TOKEN,
                                            "GITHUB_API_URL": "https://ghe.example.com/api/v3/"})
        self.assertEqual((api.repo, api.api_url), (REPO, "https://ghe.example.com/api/v3"))
        self.assertEqual(R.GitHubDeployments.from_env({"GITHUB_REPOSITORY": REPO, "GITHUB_TOKEN": TOKEN}).api_url,
                         API)
        with self.assertRaises(R.ReleaseError) as e:
            R.GitHubDeployments.from_env({"GITHUB_TOKEN": " "})
        self.assertIn("needs GITHUB_REPOSITORY and GITHUB_TOKEN", str(e.exception))

    def test_urllib_fetch_without_the_network(self):
        class Answer(io.BytesIO):
            status = 200
            headers = {"Link": '<n>; rel="next"'}

        with mock.patch.object(R.urllib.request, "urlopen", return_value=Answer(b"[]")) as urlopen:
            r = R.urllib_fetch(deployments_url(), {"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual((r.status, r.body, r.link), (200, b"[]", '<n>; rel="next"'))
        req = urlopen.call_args.args[0]
        self.assertEqual(req.full_url, deployments_url())
        self.assertEqual(req.get_header("Authorization"), f"Bearer {TOKEN}")
        http_error = urllib.error.HTTPError(deployments_url(), 403, "Forbidden", {}, io.BytesIO(b"{}"))
        with mock.patch.object(R.urllib.request, "urlopen", side_effect=http_error):
            self.assertEqual(R.urllib_fetch(deployments_url(), {}).status, 403)
        for error in (urllib.error.URLError("no route"), TimeoutError()):
            with self.subTest(error=error), mock.patch.object(R.urllib.request, "urlopen", side_effect=error):
                with self.assertRaises(R.ReleaseError) as e:
                    R.urllib_fetch(deployments_url(), {})
                self.assertIn("the GitHub API did not answer", str(e.exception))


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
        api = FakeDeployments({7: (first, ["success"])})
        self.assertEqual(R.plan("v1.0.0", g, lambda: api).rc_tags, ("v1.0.0-rc.1",))
        self.assertEqual(R.plan("v1.1.0-rc.1", g, never).environment, "staging")
        with self.assertRaises(R.ReleaseError):
            R.plan("v1.1.0", g, never)


if __name__ == "__main__":
    unittest.main()
