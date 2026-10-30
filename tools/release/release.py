#!/usr/bin/env python3
"""Release rules for the hosted service: which tag deploys where, and what a deploy needs.

Releases are trunk plus tags: every release tag must point at a commit on main (an ancestor of origin/main).
A tag `vX.Y.Z-rc.N` deploys the service to the `staging` GitHub environment. A tag `vX.Y.Z` deploys it to
`production`, and only when the same commit also carries a tag `vX.Y.Z-rc.N` (any N) and the `staging`
environment has a successful deployment of that commit. Any other tag is refused. X, Y, Z and N are whole numbers
without leading zeros. `.github/workflows/release.yml` runs these commands; the rules live here so they are tested
code.

Standard library only (the deploy itself runs the Modal CLI from tools/release/requirements.txt).

Usage, from the repository root:
  release.py plan <tag>            check the tag and that its commit is on origin/main (and, for production,
                                   its rc tag and its staging deployment); print the environment
  release.py check <tag>           check that the service's Modal app is in the commit, every deploy
                                   setting is set (names what is missing, never a secret's value) and the
                                   custom domain is the base URL's host
  release.py deploy <tag>          check again, then `python -m modal deploy -m herogen.service.modal_app`

`plan` for a production tag reads the repository's deployments from the GitHub API: it needs GITHUB_REPOSITORY
(`owner/name`) and GITHUB_TOKEN (a token that may read deployments); GITHUB_API_URL defaults to
https://api.github.com. On GitHub Actions, `plan` writes `environment`, `version` and `commit` to $GITHUB_OUTPUT;
`plan` and `deploy` add a line to the job summary, and a refusal goes to the summary and an error annotation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

STAGING = "staging"
PRODUCTION = "production"
_NUM = r"(?:0|[1-9][0-9]*)"
# Matched whole (fullmatch), so a trailing newline is refused too.
TAG = re.compile(rf"v(?P<version>{_NUM}\.{_NUM}\.{_NUM})(?:-rc\.(?P<rc>{_NUM}))?")
# The trunk every release tag's commit must be on, as the plan job fetches it.
MAIN_REF = "refs/remotes/origin/main"
TAG_FORMAT = "vX.Y.Z-rc.N (staging) or vX.Y.Z (production), where X, Y, Z and N are numbers without leading zeros"

# The service's Modal app (herogen/service/modal_app.py), deployed as a module from the repository root.
ENTRY_MODULE = "herogen.service.modal_app"
ENTRY_PATH = Path("herogen", "service", "modal_app.py")

# What the deploy reads from the GitHub environment: (name, kind). Secrets and variables are named, never shown.
REQUIRED = (
    ("MODAL_TOKEN_ID", "secret"),
    ("MODAL_TOKEN_SECRET", "secret"),
    ("HEROGEN_SERVICE_APP", "variable"),
    ("HEROGEN_SERVICE_BASE_URL", "variable"),
    ("HEROGEN_SERVICE_EMAIL_FROM", "variable"),
    ("HEROGEN_SERVICE_TERMS_VERSION", "variable"),
    ("HEROGEN_SERVICE_DOMAIN", "variable"),
)
# Settings every deploy runs with. The service refuses to deploy in any mode but `production` (staging included:
# the mode is "not dev", not the environment), and production mode needs a real email provider; `resend` is the
# only one the service has.
FIXED = {"HEROGEN_SERVICE_MODE": "production", "HEROGEN_SERVICE_EMAIL_PROVIDER": "resend"}
# The service bakes every HEROGEN_SERVICE_* variable of the deploying shell into its image, so the deploy gets
# only the ones above.
SERVICE_PREFIX = "HEROGEN_SERVICE_"

# GitHub deployment states (REST API, deployment statuses). A deployment with no status yet is pending.
SUCCESS = "success"
IN_PROGRESS = ("queued", "pending", "in_progress")
GITHUB_API = "https://api.github.com"
API_VERSION = "2022-11-28"
PER_PAGE = 100
MAX_PAGES = 20
_NEXT = re.compile(r'<([^<>]+)>\s*;\s*rel="next"')


class ReleaseError(Exception):
    """A release that must not go ahead. The message says why and what to do, and never holds a secret."""


@dataclass(frozen=True)
class Tag:
    name: str
    version: str
    rc: int | None

    @property
    def is_rc(self) -> bool:
        return self.rc is not None

    @property
    def environment(self) -> str:
        return STAGING if self.is_rc else PRODUCTION


def parse_tag(name: str) -> Tag:
    m = TAG.fullmatch(name)
    if not m:
        raise ReleaseError(f"{name!r} is not a release tag, so it deploys nothing. Release tags are {TAG_FORMAT}, "
                           "for example v1.4.0-rc.1 then v1.4.0.")
    rc = m.group("rc")
    return Tag(name, m.group("version"), None if rc is None else int(rc))


def try_parse_tag(name: str) -> Tag | None:
    try:
        return parse_tag(name)
    except ReleaseError:
        return None


class Git(Protocol):
    def commit_of(self, tag: str) -> str: ...

    def tags_at(self, commit: str) -> list[str]: ...

    def on_main(self, commit: str) -> bool:
        """Whether `commit` is an ancestor of (or is) the tip of MAIN_REF; a ReleaseError when MAIN_REF is
        unknown."""
        ...


class LocalGit:
    """The repository's own tags, through the git command (the checkout must hold every tag)."""

    def __init__(self, cwd: Path | None = None) -> None:
        self.cwd = cwd

    def _git(self, *args: str) -> str:
        r = subprocess.run(["git", *args], cwd=self.cwd, capture_output=True, text=True)
        if r.returncode != 0:
            raise ReleaseError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
        return r.stdout

    def commit_of(self, tag: str) -> str:
        """The commit tag `tag` points at (through an annotated tag), or "" when there is no such tag."""
        r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}"], cwd=self.cwd,
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""

    def tags_at(self, commit: str) -> list[str]:
        return [t for t in self._git("tag", "--points-at", commit).splitlines() if t.strip()]

    def on_main(self, commit: str) -> bool:
        r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{MAIN_REF}^{{commit}}"], cwd=self.cwd,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise ReleaseError(f"{MAIN_REF} is not in this checkout, so a release tag cannot be checked against main. "
                               "Fetch it first: git fetch --no-tags origin +refs/heads/main:refs/remotes/origin/main")
        r = subprocess.run(["git", "merge-base", "--is-ancestor", commit, MAIN_REF], cwd=self.cwd,
                           capture_output=True, text=True)
        if r.returncode in (0, 1):
            return r.returncode == 0
        raise ReleaseError(f"git merge-base --is-ancestor {commit} {MAIN_REF} failed: {r.stderr.strip()}")


def rc_tags_at(tag: Tag, tags: Sequence[str]) -> list[str]:
    """The rc tags of `tag`'s version among `tags`, lowest N first."""
    rcs = [t for t in (try_parse_tag(n) for n in tags) if t and t.is_rc and t.version == tag.version]
    return [t.name for t in sorted(rcs, key=lambda t: t.rc)]


# ---------------------------------------------------------------------------
# Deployments, from the GitHub REST API


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes
    link: str = ""   # the Link header, for the next page


Fetch = Callable[[str, Mapping[str, str]], Response]


def urllib_fetch(url: str, headers: Mapping[str, str]) -> Response:
    """GET `url`. An HTTP error comes back as its Response; no answer at all is a ReleaseError."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=dict(headers)), timeout=30) as r:
            return Response(r.status, r.read(), r.headers.get("Link", "") or "")
    except urllib.error.HTTPError as e:
        return Response(e.code, b"", (e.headers.get("Link", "") if e.headers else "") or "")
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", None) or type(e).__name__
        raise ReleaseError(f"the GitHub API did not answer ({reason}), so the {STAGING} deployment of this commit "
                           "cannot be checked; re-run this workflow.") from None


class Deployments(Protocol):
    def deployments(self, environment: str, sha: str) -> list[dict]: ...

    def states(self, deployment_id: int) -> list[str]: ...


class GitHubDeployments:
    """A repository's deployments and their statuses. The token goes in the Authorization header, only to
    `api_url`; a next-page link elsewhere is refused."""

    def __init__(self, repo: str, token: str, api_url: str = GITHUB_API, fetch: Fetch = urllib_fetch,
                 max_pages: int = MAX_PAGES) -> None:
        self.repo, self.api_url, self.fetch, self.max_pages = repo, api_url.rstrip("/"), fetch, max_pages
        self._headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                         "X-GitHub-Api-Version": API_VERSION, "User-Agent": "herogen-release"}

    @classmethod
    def from_env(cls, env: Mapping[str, str], fetch: Fetch = urllib_fetch) -> GitHubDeployments:
        missing = [n for n in ("GITHUB_REPOSITORY", "GITHUB_TOKEN") if not env.get(n, "").strip()]
        if missing:
            raise ReleaseError(f"checking the {STAGING} deployment needs {' and '.join(missing)} (the workflow sets "
                               "them; locally, set GITHUB_REPOSITORY to owner/name and GITHUB_TOKEN to a token that "
                               "may read deployments).")
        return cls(env["GITHUB_REPOSITORY"].strip(), env["GITHUB_TOKEN"].strip(),
                   env.get("GITHUB_API_URL", "").strip() or GITHUB_API, fetch)

    def _error(self, status: int, what: str) -> ReleaseError:
        if status == 401:
            why = "the GitHub token was refused (401)"
        elif status == 403:
            why = ("the GitHub token may not read deployments, or the API rate limit was reached (403); the job "
                   "needs the `deployments: read` permission")
        elif status == 404:
            why = f"GitHub found no repository {self.repo!r} or deployment there (404)"
        elif status >= 500:
            why = f"the GitHub API failed ({status}); re-run this workflow later"
        else:
            why = f"the GitHub API answered {status}"
        return ReleaseError(f"could not read {what}: {why}. Production is not deployed until this check passes.")

    def _get_all(self, path: str, query: Mapping[str, str], what: str) -> list:
        url = f"{self.api_url}{path}?{urllib.parse.urlencode({**query, 'per_page': PER_PAGE})}"
        items: list = []
        for _ in range(self.max_pages):
            r = self.fetch(url, self._headers)
            if r.status != 200:
                raise self._error(r.status, what)
            try:
                page = json.loads(r.body)
            except ValueError:
                page = None
            if not isinstance(page, list):
                raise ReleaseError(f"could not read {what}: the GitHub API's answer is not a list.")
            items += page
            m = _NEXT.search(r.link)
            if not m:
                return items
            url = m.group(1)
            if not url.startswith(self.api_url + "/"):
                raise ReleaseError(f"could not read {what}: the GitHub API's next page is not on {self.api_url}.")
        raise ReleaseError(f"could not read {what}: more than {self.max_pages} pages of {PER_PAGE}.")

    def deployments(self, environment: str, sha: str) -> list[dict]:
        found = self._get_all(f"/repos/{self.repo}/deployments", {"environment": environment, "sha": sha},
                              f"the {environment} deployments of {sha}")
        return [d for d in found if isinstance(d, dict)]

    def states(self, deployment_id: int) -> list[str]:
        """The deployment's status states, newest first, as the API lists them."""
        found = self._get_all(f"/repos/{self.repo}/deployments/{int(deployment_id)}/statuses", {},
                              f"the statuses of deployment {deployment_id}")
        return [str(s.get("state", "")) for s in found if isinstance(s, dict)]


def staging_succeeded(commit: str, api: Deployments) -> int:
    """The id of a successful `staging` deployment of `commit`, or a ReleaseError saying what was found instead.

    A deployment counts once any of its statuses is `success`: GitHub marks an earlier successful deployment
    `inactive` when a later one to the same environment succeeds, and it still counts."""
    found = [d for d in api.deployments(STAGING, commit) if d.get("sha") == commit and "id" in d]
    rerun = f"Let its rc tag deploy to {STAGING}, then re-run this workflow."
    if not found:
        raise ReleaseError(f"cannot deploy to {PRODUCTION}: the {STAGING} environment has no deployment of commit "
                           f"{commit}. {rerun}")
    latest = {}
    for d in found:
        states = api.states(d["id"])
        if SUCCESS in states:
            return int(d["id"])
        latest[d["id"]] = states[0] if states else "pending"
    running = sorted(i for i, s in latest.items() if s in IN_PROGRESS)
    if running:
        raise ReleaseError(f"cannot deploy to {PRODUCTION} yet: the {STAGING} deployment of commit {commit} is still "
                           f"in progress (deployment {', '.join(map(str, running))}). Re-run this workflow once it "
                           "has succeeded.")
    ended = ", ".join(f"{i}: {s}" for i, s in sorted(latest.items()))
    raise ReleaseError(f"cannot deploy to {PRODUCTION}: no {STAGING} deployment of commit {commit} succeeded "
                       f"(latest states {ended}). Fix the {STAGING} deploy (re-run it, or release a new rc), then "
                       "re-run this workflow.")


# ---------------------------------------------------------------------------
# The plan


@dataclass(frozen=True)
class Plan:
    tag: Tag
    commit: str
    rc_tags: tuple[str, ...]
    staging_deployment: int | None = None

    @property
    def environment(self) -> str:
        return self.tag.environment

    def describe(self) -> str:
        if self.tag.is_rc:
            return f"{self.tag.name} ({self.commit}) deploys to {STAGING}."
        return (f"{self.tag.name} ({self.commit}) deploys to {PRODUCTION}: the commit carries "
                f"{', '.join(self.rc_tags)} and {STAGING} deployment {self.staging_deployment} of it succeeded.")


def plan(name: str, git: Git, deployments: Callable[[], Deployments]) -> Plan:
    """Where tag `name` deploys. Any tag whose commit is not on main is refused; production is refused unless the
    tag's commit also carries an rc tag of its version and has a successful `staging` deployment. `deployments` is
    called only for a production tag."""
    tag = parse_tag(name)
    commit = git.commit_of(tag.name)
    if not commit:
        raise ReleaseError(f"tag {tag.name} is not in this checkout, so its commit is unknown.")
    if not git.on_main(commit):
        raise ReleaseError(f"tag {tag.name} points at commit {commit}, which is not on main ({MAIN_REF}). Release "
                           "tags must point at a commit on main: merge the change into main, then tag a commit on "
                           "main with a new tag.")
    if tag.is_rc:
        return Plan(tag, commit, ())
    rcs = rc_tags_at(tag, git.tags_at(commit))
    if not rcs:
        raise ReleaseError(
            f"{tag.name} cannot deploy to {PRODUCTION}: its commit {commit} carries no {tag.name}-rc.N tag. "
            f"A production tag goes on a commit that was released to {STAGING} first: tag this commit "
            f"{tag.name}-rc.N, let it deploy, then re-run this workflow (a re-run reads the tags again).")
    try:
        deployment = staging_succeeded(commit, deployments())
    except ReleaseError as e:
        raise ReleaseError(f"{tag.name} {e}") from None
    return Plan(tag, commit, tuple(rcs), deployment)


# ---------------------------------------------------------------------------
# The deploy


def missing_config(env: Mapping[str, str]) -> list[tuple[str, str]]:
    """The REQUIRED settings that are unset or blank in `env`, in REQUIRED's order. A domain list with no entry
    (only commas and spaces) is blank."""
    def blank(name: str) -> bool:
        value = env.get(name, "")
        return not (domains(value) if name == "HEROGEN_SERVICE_DOMAIN" else value.strip())

    return [(n, kind) for n, kind in REQUIRED if blank(n)]


def config_message(missing: Sequence[tuple[str, str]], environment: str) -> str:
    """Names what is missing and where to set it. Takes names only, so it cannot show a value."""
    lines = [f"The {environment} deploy is missing {len(missing)} setting(s). Set each in the GitHub environment "
             f"{environment!r} (Settings > Environments > {environment}):"]
    lines += [f"  {kind} {name}" for name, kind in missing]
    return "\n".join(lines)


# A bare host name, as Modal's custom domains take it: no scheme, port or path.
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
HOST = re.compile(rf"{_LABEL}(?:\.{_LABEL})+")
# The base URL on a custom domain: https, the default port, no path.
BASE_URL = re.compile(r"https://(?P<host>[^/:?#\s]+)/?")


def domains(raw: str) -> list[str]:
    """HEROGEN_SERVICE_DOMAIN as the service reads it (modal_app.py, CUSTOM_DOMAINS): a comma-separated list, each
    entry stripped, empty entries dropped."""
    return [d.strip() for d in raw.split(",") if d.strip()]


def domain_problems(env: Mapping[str, str]) -> list[str]:
    """What is wrong with HEROGEN_SERVICE_DOMAIN against HEROGEN_SERVICE_BASE_URL (only once both are set).

    The web function serves on every domain listed, but the service's Host check accepts only the base URL's host
    (HEROGEN_SERVICE_HOSTS is not passed to the deploy), so the base URL's host must be one of them."""
    raw_base, raw_domains = env.get("HEROGEN_SERVICE_BASE_URL", "").strip(), env.get("HEROGEN_SERVICE_DOMAIN", "")
    listed = domains(raw_domains)
    if not raw_base or not listed:
        return []
    problems = [f"HEROGEN_SERVICE_DOMAIN entry {d!r} is not a bare host name (no scheme, port or path), for "
                "example hero.example.org." for d in listed if not HOST.fullmatch(d)]
    m = BASE_URL.fullmatch(raw_base)
    if not m or not HOST.fullmatch(m.group("host")):
        problems.append(f"HEROGEN_SERVICE_BASE_URL {raw_base!r} must be https://<host> on the custom domain, with "
                        "no port or path.")
        return problems
    host = m.group("host").lower()
    if host not in [d.lower() for d in listed]:
        problems.append(f"HEROGEN_SERVICE_BASE_URL's host {host} is not in HEROGEN_SERVICE_DOMAIN "
                        f"({', '.join(listed)}): the service would answer only on a host its custom domains do not "
                        "attach. Set HEROGEN_SERVICE_DOMAIN to the base URL's host.")
    return problems


ENTRY_MISSING = (f"{ENTRY_PATH.as_posix()} is not in this commit, so there is no service to deploy. The service "
                 "code (herogen/service) must be merged before a release tag can deploy it.")


def check(name: str, env: Mapping[str, str], root: Path) -> Tag:
    """Whether tag `name` can deploy from `root`: the service's Modal app is there, every setting is set, and the
    custom domain matches the base URL. Every problem is reported at once."""
    tag = parse_tag(name)
    problems = [] if (root / ENTRY_PATH).is_file() else [ENTRY_MISSING]
    missing = missing_config(env)
    if missing:
        problems.append(config_message(missing, tag.environment))
    problems += domain_problems(env)
    if problems:
        raise ReleaseError("\n".join(problems))
    return tag


def deploy_env(env: Mapping[str, str]) -> dict[str, str]:
    """The environment `modal deploy` runs with: `env` without any HEROGEN_SERVICE_* variable but REQUIRED's,
    then FIXED."""
    out = {k: v for k, v in env.items() if not k.startswith(SERVICE_PREFIX)}
    out.update({n: env[n] for n, _ in REQUIRED})
    out.update(FIXED)
    return out


def deploy_command(tag: Tag) -> list[str]:
    return [sys.executable, "-m", "modal", "deploy", "-m", ENTRY_MODULE, "--tag", tag.name]


def deploy(name: str, env: Mapping[str, str], root: Path,
           run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> None:
    """Deploy the service's Modal app for tag `name` (its environment decides which settings it is given)."""
    tag = check(name, env, root)
    r = run(deploy_command(tag), cwd=root, env=deploy_env(env))
    if r.returncode != 0:
        raise ReleaseError(f"modal deploy exited with status {r.returncode}; its output is above.")


def _github_output(values: Mapping[str, str], env: Mapping[str, str]) -> None:
    path = env.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.writelines(f"{k}={v}\n" for k, v in values.items())


def _summary(line: str, env: Mapping[str, str]) -> None:
    path = env.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _escape(text: str) -> str:
    """A workflow command's message: GitHub reads %, CR and LF as escapes there."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None, git: Git | None = None,
         root: Path | None = None, run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
         fetch: Fetch = urllib_fetch) -> int:
    env = os.environ if env is None else env
    root = Path.cwd() if root is None else root
    p = argparse.ArgumentParser(prog="release.py", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for cmd, help_ in (("plan", "check the tag and print where it deploys"),
                       ("check", "check that the service is in the commit and the deploy settings are set"),
                       ("deploy", "deploy the service's Modal app")):
        sub.add_parser(cmd, help=help_).add_argument("tag")
    args = p.parse_args(argv)
    try:
        if args.command == "plan":
            pl = plan(args.tag, git or LocalGit(root), lambda: GitHubDeployments.from_env(env, fetch))
            print(pl.describe())
            _github_output({"environment": pl.environment, "version": pl.tag.version, "commit": pl.commit}, env)
            _summary(pl.describe(), env)
        elif args.command == "check":
            tag = check(args.tag, env, root)
            print(f"The service's Modal app is in this commit and every setting the {tag.environment} deploy "
                  "needs is set.")
        else:
            tag = parse_tag(args.tag)
            deploy(args.tag, env, root, run)
            line = f"Deployed {tag.name} to {tag.environment} as Modal app {env['HEROGEN_SERVICE_APP']}."
            print(line)
            _summary(line, env)
    except ReleaseError as e:
        print(f"release: {e}", file=sys.stderr)
        if env.get("GITHUB_ACTIONS") == "true":
            print("::error title=Release refused::" + _escape(str(e)))
        _summary(f"Release refused: {e}", env)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
