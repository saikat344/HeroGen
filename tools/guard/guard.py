#!/usr/bin/env python3
"""Pre-publication safeguards for the HeroGen public repository.

Everything committed here becomes public, including history. These checks run on
staged files (pre-commit), commit messages (commit-msg), every commit being pushed
(pre-push), and again in CI, because local hooks can be skipped with --no-verify.

Standard library only, so it runs unchanged on macOS, Linux and Windows.

Usage:
  guard.py staged                 check the index and the author/committer identity
  guard.py commit-msg <file>      check a commit message
  guard.py push                   pre-push hook (reads refs on stdin)
  guard.py tree                   check every tracked file in the working tree
  guard.py range <base> <head>    check every commit in base..head ("" base = all of head)
  guard.py strip <image>...       remove EXIF/XMP/IPTC/text metadata from JPEG/PNG in place

Exceptions live in .guard-allow, one per line: `<glob> <rule>  # reason` (reason required).
Secrets and author identity can never be allowlisted.

Publishing: after all checks pass, pre-push prints every commit and file it is about to make public
and asks the maintainer to type "publish" on the terminal. Scripts and CI have no terminal, so they
can't confirm.

Extra forbidden paths that should stay out of this file can be listed locally in
`<git common dir>/info/guard-forbid`, one regular expression per line: `<regex>  # reason`.
"""
from __future__ import annotations

import fnmatch
import os
import re
import struct
import subprocess
import sys
from dataclasses import dataclass

MAX_BYTES = 1_000_000
ZERO_SHA = "0" * 40
ALLOW_FILE = ".guard-allow"
PROTECTED_REF = "refs/heads/main"
PUBLISH_WORD = "publish"
PLANNING_THRESHOLD = 3
NEVER_ALLOWLISTED = {"secret", "identity"}

# GitHub noreply addresses only, so commits never publish a personal email.
ALLOWED_IDENTITY = re.compile(
    r"^(\d+\+)?[A-Za-z0-9-]+@users\.noreply\.github\.com$|^noreply@github\.com$"
)

FORBIDDEN_PATHS = [
    (r"(^|/)\.env(\.[^/]+)?$", "environment file (commit .env.example with placeholders instead)"),
    (r"\.(pem|key|p12|pfx|jks|keystore|ppk)$", "key or certificate file"),
    (r"(^|/)id_(rsa|dsa|ecdsa|ed25519)(\.pub)?$", "SSH key"),
    (r"(^|/)(auth|credentials?|secrets?|tokens?)\.(json|ya?ml|toml|ini|txt)$", "credentials file"),
    (r"(^|/)(private|raw)(/|$)", "private or raw material"),
    # The public repo holds engineering only: code, tests, final implementation and architecture docs.
    (r"(^|/)(plans?|planning|reviews?|notes|brainstorm(ing)?|discussions?|meetings?|retros?|drafts?|scratch|ideas?)(/|$)",
     "planning or discussion material (keep it in private/)"),
    (r"(^|/)([^/]*[-_.])?(plans?|planning|reviews?|brainstorm(ing)?|meetings?|retros?|notes|ideas?)([-_.][^/]*)?\.(md|txt|rst)$",
     "planning or discussion document (keep it in private/; allowlist only a final engineering doc, with a reason)"),
    (r"(^|/)(\.DS_Store|Thumbs\.db)$", "OS metadata file"),
    (r"\.(pdf|docx?|pptx?|xlsx?)$", "document file (third-party documents are not redistributable; write Markdown)"),
    (r"\.(mp4|mov|m4v|avi|mkv|webm|heic|heif|webp|dng|cr2|nef|wav|mp3|m4a|aac|flac)$",
     "media file (keep video and audio out of git; link the published demo instead)"),
    (r"\.(zip|tar|tgz|gz|bz2|xz|7z|rar)$", "archive"),
    (r"\.(sqlite3?|db)$", "database file"),
]
FORBIDDEN_PATHS = [(re.compile(p, re.IGNORECASE), why) for p, why in FORBIDDEN_PATHS]


def _local_forbidden() -> list[tuple[re.Pattern[str], str]]:
    """Extra forbidden-path patterns from `<git common dir>/info/guard-forbid` (local, never committed)."""
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, check=True).stdout.strip()
        lines = open(os.path.join(common, "info", "guard-forbid"), encoding="utf-8").read().splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []
    out = []
    for line in lines:
        pattern, _, why = line.partition("#")
        if pattern.strip():
            out.append((re.compile(pattern.strip(), re.IGNORECASE), why.strip() or "locally forbidden path"))
    return out


FORBIDDEN_PATHS += _local_forbidden()

SECRET_PATTERNS = [
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b")),
    ("Slack token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("sk- style API key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}\b")),
    ("bearer token", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9_\-.=]{20,})")),
    ("credentials in URL", re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:([^/\s@]{3,})@")),
    ("credential assignment", re.compile(
        r"(?i)\b(?:api[_-]?key|apikey|secret|token|password|passwd|access[_-]?key|client[_-]?secret)\b"
        r"[\"']?\s*[:=]\s*[\"']([^\"'\s]{12,})[\"']")),
]
PLACEHOLDER = re.compile(
    r"(?i)^(<[^>]*>|\$\{?[A-Z0-9_]+\}?|x{4,}|\*{4,}|\.{3,}|changeme|placeholder|redacted"
    r"|your[-_a-z]*|example[-_a-z0-9]*|dummy[-_a-z0-9]*|fake[-_a-z0-9]*|test[-_a-z0-9]*)$"
)

EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})")
ALLOWED_EMAIL_DOMAINS = ("users.noreply.github.com", "noreply.github.com", "example.com", "example.org", "example.net")
ALLOWED_EMAIL_LOCALS = ("noreply", "no-reply", "git")

HOME_PATH = re.compile(r"(?:/Users/|/home/)([A-Za-z0-9._-]+)/|\b[A-Za-z]:\\+Users\\+([A-Za-z0-9._ -]+)")
PLACEHOLDER_USERS = {"you", "user", "username", "runner", "me", "name", "shared"}

MARKERS = [
    ("commit-blocking marker", re.compile(r"(?i)\bDO[-_ ]?NOT[-_ ]?COMMIT\b|\bNOCOMMIT\b")),
    ("merge conflict marker", re.compile(r"^(?:<{7}|>{7})(?: |$)")),
]

# Planning language in a text doc. Several distinct signals together mean "this is a plan", whatever the file is called.
PLANNING_SIGNALS = [
    ("day-by-day schedule", re.compile(r"(?i)\bday\s*[0-9]+\s*[:(–-]")),
    ("stretch goals", re.compile(r"(?i)\bstretch\s+(goals?|window|items?|menu)\b")),
    ("prizes or judging", re.compile(r"(?i)\b(prizes?|judges?|judging)\b")),
    ("deadline", re.compile(r"(?i)\bdeadlines?\b")),
    ("kill criteria", re.compile(r"(?i)\bkill\s+(criteria|criterion|switch)\b")),
    ("feature freeze", re.compile(r"(?i)\bfeature\s+freeze\b")),
    ("brainstorm or retro", re.compile(r"(?i)\b(brainstorm(s|ing)?|retrospectives?|stand-?ups?|action\s+items?)\b")),
    ("fallback planning", re.compile(r"(?i)\b(if it fails|plan b|fallbacks?)\b")),
]
PLANNING_EXTENSIONS = (".md", ".markdown", ".txt", ".rst")

SUBJECT = re.compile(
    r"^(?:feat|fix|docs|test|refactor|perf|build|ci|chore|revert)(?:\([a-z0-9._/-]+\))?!?: \S.*$"
)


@dataclass
class Violation:
    path: str
    rule: str
    detail: str
    line: int = 0

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"  {where}  [{self.rule}] {self.detail}"


class GitError(RuntimeError):
    pass


def git(*args: str, stdin: bytes | None = None) -> bytes:
    result = subprocess.run(["git", *args], input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def redact(value: str) -> str:
    """Never print a secret in full: CI logs are public. ASCII only, for Windows consoles."""
    return f"{value[:4]}... ({len(value)} chars)"


# ---------------------------------------------------------------------------
# Allowlist


@dataclass
class AllowRule:
    glob: str
    rule: str


def parse_allowlist(text: str) -> tuple[list[AllowRule], list[Violation]]:
    rules: list[AllowRule] = []
    problems: list[Violation] = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        body, _, reason = line.partition("#")
        parts = body.split()
        if len(parts) != 2 or not reason.strip():
            problems.append(Violation(ALLOW_FILE, "allowlist", "expected `<glob> <rule>  # reason`", n))
            continue
        if parts[1] in NEVER_ALLOWLISTED:
            problems.append(Violation(ALLOW_FILE, "allowlist", f"rule '{parts[1]}' cannot be allowlisted", n))
            continue
        rules.append(AllowRule(parts[0], parts[1]))
    return rules, problems


def allowed(path: str, rule: str, rules: list[AllowRule]) -> bool:
    return any(r.rule == rule and fnmatch.fnmatchcase(path, r.glob) for r in rules)


# ---------------------------------------------------------------------------
# Image metadata (EXIF can carry GPS coordinates of the user's home)


def exif_orientation(tiff: bytes) -> int | None:
    try:
        endian = {b"II": "<", b"MM": ">"}[tiff[:2]]
        (ifd,) = struct.unpack(endian + "I", tiff[4:8])
        (count,) = struct.unpack(endian + "H", tiff[ifd:ifd + 2])
        for k in range(count):
            entry = tiff[ifd + 2 + 12 * k: ifd + 14 + 12 * k]
            (tag,) = struct.unpack(endian + "H", entry[:2])
            if tag == 0x0112:
                return struct.unpack(endian + "H", entry[8:10])[0]
    except (KeyError, struct.error):
        return None
    return None


def jpeg_segments(data: bytes):
    """Yield (marker, start, end, payload) for header segments up to the start of scan."""
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD9, 0xDA):
            return
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:
            i += 2
            continue
        (length,) = struct.unpack(">H", data[i + 2:i + 4])
        yield marker, i, i + 2 + length, data[i + 4:i + 2 + length]
        i += 2 + length


def jpeg_metadata_kind(marker: int, payload: bytes) -> str | None:
    if marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
        return "EXIF"
    if marker == 0xE1:
        return "XMP"
    if marker == 0xED:
        return "IPTC"
    if marker == 0xFE:
        return "comment"
    return None


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_METADATA_CHUNKS = {b"eXIf", b"tEXt", b"iTXt", b"zTXt", b"tIME"}


def png_chunks(data: bytes):
    pos = len(PNG_SIGNATURE)
    while pos + 8 <= len(data):
        length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
        yield ctype, pos, pos + 12 + length, data[pos + 8:pos + 8 + length]
        pos += 12 + length


def image_kind(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(PNG_SIGNATURE):
        return "png"
    if data[:4] == b"GIF8":
        return "gif"
    return None


def image_metadata(data: bytes) -> tuple[list[str], int | None]:
    """Return (metadata kinds found, EXIF orientation)."""
    kinds: list[str] = []
    orientation = None
    kind = image_kind(data)
    if kind == "jpeg":
        for marker, _, _, payload in jpeg_segments(data):
            found = jpeg_metadata_kind(marker, payload)
            if found:
                kinds.append(found)
                if found == "EXIF":
                    orientation = exif_orientation(payload[6:])
    elif kind == "png":
        for ctype, _, _, payload in png_chunks(data):
            if ctype in PNG_METADATA_CHUNKS:
                kinds.append(ctype.decode())
                if ctype == b"eXIf":
                    orientation = exif_orientation(payload)
    return kinds, orientation


def strip_image(data: bytes) -> bytes:
    kind = image_kind(data)
    if kind == "jpeg":
        out = bytearray(data[:2])
        end = 2
        for marker, start, stop, payload in jpeg_segments(data):
            if not jpeg_metadata_kind(marker, payload):
                out += data[start:stop]
            end = stop
        out += data[end:]
        return bytes(out)
    if kind == "png":
        out = bytearray(PNG_SIGNATURE)
        for ctype, start, stop, _ in png_chunks(data):
            if ctype not in PNG_METADATA_CHUNKS:
                out += data[start:stop]
        return bytes(out)
    raise ValueError("only JPEG and PNG can be stripped")


# ---------------------------------------------------------------------------
# Content checks


def check_text(path: str, text: str, rules: list[AllowRule]) -> list[Violation]:
    found: list[Violation] = []
    for n, line in enumerate(text.splitlines(), 1):
        for name, pattern in SECRET_PATTERNS:
            for m in pattern.finditer(line):
                value = m.group(m.lastindex or 0)
                if PLACEHOLDER.match(value):
                    continue
                found.append(Violation(path, "secret", f"{name}: {redact(value)}", n))
        if not allowed(path, "email", rules):
            for m in EMAIL.finditer(line):
                local, domain = m.group(1).lower(), m.group(2).lower()
                if domain.endswith(ALLOWED_EMAIL_DOMAINS) or local in ALLOWED_EMAIL_LOCALS:
                    continue
                found.append(Violation(path, "email", f"personal email address: {redact(m.group(0))}", n))
        if not allowed(path, "home-path", rules):
            for m in HOME_PATH.finditer(line):
                user = (m.group(1) or m.group(2) or "").strip()
                if user.lower() in PLACEHOLDER_USERS:
                    continue
                found.append(Violation(path, "home-path", "absolute home-directory path (use repo-relative paths)", n))
        if not allowed(path, "marker", rules):
            for name, pattern in MARKERS:
                if pattern.search(line):
                    found.append(Violation(path, "marker", name, n))
    return found


def check_blob(path: str, data: bytes, rules: list[AllowRule]) -> list[Violation]:
    found: list[Violation] = []
    name = path.rsplit("/", 1)[-1]
    if not allowed(path, "path", rules) and not name.endswith(".env.example"):
        for pattern, why in FORBIDDEN_PATHS:
            if pattern.search(path):
                found.append(Violation(path, "path", why))
                break
    if len(data) > MAX_BYTES and not allowed(path, "size", rules):
        found.append(Violation(path, "size", f"{len(data):,} bytes exceeds {MAX_BYTES:,}"))
    if image_kind(data):
        kinds, _ = image_metadata(data)
        if kinds:
            found.append(Violation(path, "image-metadata",
                                   f"contains {', '.join(sorted(set(kinds)))}; run `python tools/guard/guard.py strip {path}`"))
        return found
    if b"\x00" in data[:8192]:
        if not allowed(path, "binary", rules):
            found.append(Violation(path, "binary", "binary file (add to .guard-allow with a reason if it must be committed)"))
        return found
    text = data.decode("utf-8", errors="replace")
    found += check_text(path, text, rules)
    if path.lower().endswith(PLANNING_EXTENSIONS) and not allowed(path, "planning", rules):
        hits = [name for name, pattern in PLANNING_SIGNALS if pattern.search(text)]
        if len(hits) >= PLANNING_THRESHOLD:
            found.append(Violation(path, "planning",
                                   f"reads like planning material ({', '.join(hits)}); keep it in private/"))
    return found


def check_identity(email: str, role: str) -> list[Violation]:
    if ALLOWED_IDENTITY.match(email):
        return []
    return [Violation(f"<{role}>", "identity",
                      f"{redact(email)} is not a GitHub noreply address; "
                      "set `git config user.email <id>+<login>@users.noreply.github.com`")]


def check_message(text: str, rules: list[AllowRule], label: str = "<commit message>") -> list[Violation]:
    lines = []
    for line in text.splitlines():
        if line.startswith("# ------------------------ >8"):
            break
        if not line.startswith("#"):
            lines.append(line.rstrip())
    while lines and not lines[0]:
        lines.pop(0)
    found: list[Violation] = []
    if not lines:
        return [Violation(label, "message", "empty commit message")]
    subject = lines[0]
    if not (subject.startswith("Merge ") or subject.startswith('Revert "')):
        if not SUBJECT.match(subject):
            found.append(Violation(label, "message",
                                   "subject must be `type(scope): summary` (feat, fix, docs, test, refactor, perf, build, ci, chore, revert)", 1))
        if len(subject) > 72:
            found.append(Violation(label, "message", f"subject is {len(subject)} chars (max 72)", 1))
        if len(lines) > 1 and lines[1]:
            found.append(Violation(label, "message", "leave a blank line between the subject and the body", 2))
    return found + check_text(label, "\n".join(lines), rules)


# ---------------------------------------------------------------------------
# Modes


def git_email(var: str) -> str:
    ident = git("var", var).decode(errors="replace")
    m = re.search(r"<([^>]*)>", ident)
    return m.group(1) if m else ""


def load_allow_from(spec: str) -> tuple[list[AllowRule], list[Violation]]:
    try:
        text = git("show", spec).decode("utf-8", errors="replace")
    except GitError:
        return [], []
    return parse_allowlist(text)


def mode_staged() -> list[Violation]:
    rules, found = load_allow_from(f":{ALLOW_FILE}")
    found += check_identity(git_email("GIT_AUTHOR_IDENT"), "author")
    found += check_identity(git_email("GIT_COMMITTER_IDENT"), "committer")
    names = git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR").split(b"\x00")
    for raw in filter(None, names):
        path = raw.decode("utf-8", errors="replace")
        found += check_blob(path, git("show", f":{path}"), rules)
    return found


def mode_tree() -> list[Violation]:
    found: list[Violation] = []
    rules: list[AllowRule] = []
    if os.path.exists(ALLOW_FILE):
        with open(ALLOW_FILE, encoding="utf-8") as f:
            rules, found = parse_allowlist(f.read())
    for raw in filter(None, git("ls-files", "-z").split(b"\x00")):
        path = raw.decode("utf-8", errors="replace")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                found += check_blob(path, f.read(), rules)
    return found


def commit_files(sha: str) -> list[str]:
    parents = git("rev-list", "--parents", "-n", "1", sha).decode().split()[1:]
    base = [parents[0]] if parents else ["--root"]
    out = git("diff-tree", "-r", "-z", "--name-only", "--no-commit-id", "--diff-filter=ACMR", *base, sha)
    return [p.decode("utf-8", errors="replace") for p in out.split(b"\x00") if p]


def mode_range(base: str, head: str) -> list[Violation]:
    rules, found = load_allow_from(f"{head}:{ALLOW_FILE}")
    spec = [head] if not base else [f"{base}..{head}"]
    for sha in git("rev-list", "--reverse", *spec).decode().split():
        short = sha[:10]
        meta = git("show", "-s", "--format=%ae%x00%ce%x00%B", sha).decode("utf-8", errors="replace")
        author, committer, message = meta.split("\x00", 2)
        found += check_identity(author.strip(), f"author of {short}")
        found += check_identity(committer.strip(), f"committer of {short}")
        found += check_message(message, rules, f"<message of {short}>")
        for path in commit_files(sha):
            for v in check_blob(path, git("show", f"{sha}:{path}"), rules):
                v.path = f"{short}:{v.path}"
                found.append(v)
    return found


def push_base(local_sha: str, remote_sha: str) -> str | None:
    """The commit the remote already has (\"\" = publishing from the root), or None if nothing is new."""
    if remote_sha != ZERO_SHA:
        return remote_sha if git("rev-list", f"{remote_sha}..{local_sha}").strip() else None
    new = git("rev-list", local_sha, "--not", "--remotes").decode().split()
    if not new:
        return None
    parents = git("rev-list", "--parents", "-n", "1", new[-1]).decode().split()[1:]
    return parents[0] if parents else ""


def human_size(n: int) -> str:
    return f"{n} B" if n < 1024 else f"{n / 1024:.1f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:.1f} MB"


def publish_manifest(remote_ref: str, base: str, head: str) -> str:
    """Exactly what this push makes public: the commits, and every file added, changed or deleted."""
    spec = [head] if not base else [f"{base}..{head}"]
    commits = git("log", "--reverse", "--format=%h  %s", *spec).decode("utf-8", errors="replace").splitlines()
    if base:
        rows = [line.split("\t") for line in git("diff", "--name-status", "--no-renames", base, head)
                .decode("utf-8", errors="replace").splitlines()]
    else:
        rows = [["A", p] for p in git("ls-tree", "-r", "--name-only", head)
                .decode("utf-8", errors="replace").splitlines()]
    lines = [f"  -> {remote_ref}: {len(commits)} commit(s)"]
    lines += [f"       {c}" for c in commits]
    lines.append(f"     {len(rows)} file(s):")
    for status, path in ((r[0], r[-1]) for r in rows):
        if status.startswith("D"):
            lines.append(f"       D  {path}  (deleted, but still public in history)")
        else:
            size = int(git("cat-file", "-s", f"{head}:{path}").strip())
            lines.append(f"       {status[0]}  {path}  ({human_size(size)})")
    return "\n".join(lines)


def confirm_publish() -> bool:
    """Ask on the controlling terminal. Scripts and CI have none, so they can't confirm for the maintainer."""
    if os.environ.get("GUARD_PUBLISH_CONFIRM") == "1":
        return True
    if os.environ.get("GUARD_NO_TTY") == "1":
        return False
    try:
        if os.name == "nt":
            tin, tout = open("CONIN$"), open("CONOUT$", "w")
        else:
            tin, tout = open("/dev/tty"), open("/dev/tty", "w")
    except OSError:
        return False
    with tin, tout:
        tout.write(f"guard: type '{PUBLISH_WORD}' to make the above public: ")
        tout.flush()
        return tin.readline().strip() == PUBLISH_WORD


def mode_push(stdin_text: str) -> list[Violation]:
    found: list[Violation] = []
    manifests: list[str] = []
    for line in stdin_text.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        _, local_sha, remote_ref, remote_sha = parts
        if local_sha == ZERO_SHA:
            continue
        if remote_ref == PROTECTED_REF and os.environ.get("GUARD_ALLOW_MAIN") != "1":
            found.append(Violation(remote_ref, "push", "direct pushes to main are blocked; open a pull request"))
        base = push_base(local_sha, remote_sha)
        if base is None:
            continue
        found += mode_range(base, local_sha)
        manifests.append(publish_manifest(remote_ref, base, local_sha))
    if found or not manifests:
        return found
    print("guard: checks passed. This push makes the following PUBLIC:", file=sys.stderr)
    print("\n".join(manifests), file=sys.stderr)
    if not confirm_publish():
        found.append(Violation("<push>", "publish",
                               f"not confirmed. The maintainer pushes from a terminal and types '{PUBLISH_WORD}'"))
    return found


def staged_summary() -> str:
    rows = git("diff", "--cached", "--name-status", "--no-renames").decode("utf-8", errors="replace").splitlines()
    return "\n".join(f"  {r.split(chr(9))[0][0]}  {r.split(chr(9))[-1]}" for r in rows)


def mode_strip(paths: list[str]) -> int:
    status = 0
    for path in paths:
        with open(path, "rb") as f:
            data = f.read()
        if not image_kind(data) or image_kind(data) == "gif":
            print(f"guard: {path}: only JPEG and PNG are supported", file=sys.stderr)
            status = 1
            continue
        kinds, orientation = image_metadata(data)
        if orientation not in (None, 1):
            print(f"guard: {path}: EXIF orientation is {orientation}; rotate the pixels first "
                  "(stripping would display the photo sideways)", file=sys.stderr)
            status = 1
            continue
        if not kinds:
            print(f"guard: {path}: already clean")
            continue
        with open(path, "wb") as f:
            f.write(strip_image(data))
        print(f"guard: {path}: removed {', '.join(sorted(set(kinds)))}")
    return status


def report(found: list[Violation], context: str) -> int:
    if not found:
        return 0
    print(f"guard: BLOCKED ({context}) - {len(found)} problem(s):", file=sys.stderr)
    for v in found:
        print(v, file=sys.stderr)
    print("guard: fix these, never bypass with --no-verify. See CONTRIBUTING.md.", file=sys.stderr)
    return 1


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    mode, args = argv[0], argv[1:]
    try:
        if mode == "staged":
            status = report(mode_staged(), "pre-commit")
            if status == 0:
                print("guard: OK. Once pushed, these changes become public:\n" + staged_summary(), file=sys.stderr)
            return status
        if mode == "commit-msg" and len(args) == 1:
            with open(args[0], encoding="utf-8", errors="replace") as f:
                return report(check_message(f.read(), load_allow_from(f":{ALLOW_FILE}")[0]), "commit-msg")
        if mode == "push":
            return report(mode_push(sys.stdin.read()), "pre-push")
        if mode == "tree":
            return report(mode_tree(), "tree")
        if mode == "range" and len(args) == 2:
            return report(mode_range(args[0], args[1]), f"range {args[0][:10] or '(root)'}..{args[1][:10]}")
        if mode == "strip" and args:
            return mode_strip(args)
    except GitError as e:
        print(f"guard: {e}", file=sys.stderr)
        return 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
