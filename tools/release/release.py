#!/usr/bin/env python3
"""Release rules for the hosted service: which tag deploys where, and what a deploy needs.

A tag `vX.Y.Z-rc.N` deploys the service to the `staging` GitHub environment. A tag `vX.Y.Z` deploys it to
`production`, and only when the same commit also carries a tag `vX.Y.Z-rc.N` (any N). Any other tag is refused.
X, Y, Z and N are whole numbers without leading zeros. `.github/workflows/release.yml` runs these commands; the
rules live here so they are tested code.

Standard library only (the deploy itself runs the Modal CLI from tools/release/requirements.txt).

Usage, from the repository root:
  release.py plan <tag>            check the tag (and, for production, its rc tag); print the environment
  release.py check <tag>           check that the service's Modal app is in the commit and every deploy
                                   setting is set (names what is missing, never a value)
  release.py deploy <tag>          check again, then `python -m modal deploy -m herogen.service.modal_app`

On GitHub Actions, `plan` writes `environment`, `version` and `commit` to $GITHUB_OUTPUT; `plan` and `deploy` add
a line to the job summary, and a refusal goes to the summary and an error annotation.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

STAGING = "staging"
PRODUCTION = "production"
_NUM = r"(?:0|[1-9][0-9]*)"
# Matched whole (fullmatch), so a trailing newline is refused too.
TAG = re.compile(rf"v(?P<version>{_NUM}\.{_NUM}\.{_NUM})(?:-rc\.(?P<rc>{_NUM}))?")
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
)
# Settings every deploy runs with. The service refuses to deploy in any mode but `production` (staging included:
# the mode is "not dev", not the environment), and production mode needs a real email provider; `resend` is the
# only one the service has.
FIXED = {"HEROGEN_SERVICE_MODE": "production", "HEROGEN_SERVICE_EMAIL_PROVIDER": "resend"}
# The service bakes every HEROGEN_SERVICE_* variable of the deploying shell into its image, so the deploy gets
# only the ones above.
SERVICE_PREFIX = "HEROGEN_SERVICE_"


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


def rc_tags_at(tag: Tag, tags: Sequence[str]) -> list[str]:
    """The rc tags of `tag`'s version among `tags`, lowest N first."""
    rcs = [t for t in (try_parse_tag(n) for n in tags) if t and t.is_rc and t.version == tag.version]
    return [t.name for t in sorted(rcs, key=lambda t: t.rc)]


@dataclass(frozen=True)
class Plan:
    tag: Tag
    commit: str
    rc_tags: tuple[str, ...]

    @property
    def environment(self) -> str:
        return self.tag.environment

    def describe(self) -> str:
        if self.tag.is_rc:
            return f"{self.tag.name} ({self.commit}) deploys to {STAGING}."
        return (f"{self.tag.name} ({self.commit}) deploys to {PRODUCTION}: the commit carries "
                f"{', '.join(self.rc_tags)}.")


def plan(name: str, git: Git) -> Plan:
    """Where tag `name` deploys. Production is refused unless the tag's commit carries an rc tag of its version."""
    tag = parse_tag(name)
    commit = git.commit_of(tag.name)
    if not commit:
        raise ReleaseError(f"tag {tag.name} is not in this checkout, so its commit is unknown.")
    if tag.is_rc:
        return Plan(tag, commit, ())
    rcs = rc_tags_at(tag, git.tags_at(commit))
    if not rcs:
        raise ReleaseError(
            f"{tag.name} cannot deploy to {PRODUCTION}: its commit {commit} carries no {tag.name}-rc.N tag. "
            f"A production tag goes on a commit that was released to {STAGING} first: tag this commit "
            f"{tag.name}-rc.N, let it deploy, then re-run this workflow (a re-run reads the tags again).")
    return Plan(tag, commit, tuple(rcs))


def missing_config(env: Mapping[str, str]) -> list[tuple[str, str]]:
    """The REQUIRED settings that are unset or blank in `env`, in REQUIRED's order."""
    return [(n, kind) for n, kind in REQUIRED if not env.get(n, "").strip()]


def config_message(missing: Sequence[tuple[str, str]], environment: str) -> str:
    """Names what is missing and where to set it. Takes names only, so it cannot show a value."""
    lines = [f"The {environment} deploy is missing {len(missing)} setting(s). Set each in the GitHub environment "
             f"{environment!r} (Settings > Environments > {environment}):"]
    lines += [f"  {kind} {name}" for name, kind in missing]
    return "\n".join(lines)


ENTRY_MISSING = (f"{ENTRY_PATH.as_posix()} is not in this commit, so there is no service to deploy. The service "
                 "code (herogen/service) must be merged before a release tag can deploy it.")


def check(name: str, env: Mapping[str, str], root: Path) -> Tag:
    """Whether tag `name` can deploy from `root`: the service's Modal app is there and every setting is set. Every
    problem is reported at once."""
    tag = parse_tag(name)
    problems = [] if (root / ENTRY_PATH).is_file() else [ENTRY_MISSING]
    missing = missing_config(env)
    if missing:
        problems.append(config_message(missing, tag.environment))
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
         root: Path | None = None, run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> int:
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
            pl = plan(args.tag, git or LocalGit(root))
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
