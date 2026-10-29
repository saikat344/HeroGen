"""The lab's HTTP server: standard library only, bound to 127.0.0.1, read-only.

Routes:
  /                      the page (static/index.html)
  /static/<file>         the page's CSS and JS
  /api/runs              JSON list of runs
  /api/runs/<id>         JSON detail of one run (stage records, chain images, bodies, scores)
  /runs/<id>/<path>      a PNG, JPEG, GLB or JSON file inside that run folder

File requests are refused (404) for `..` or `.` segments, hidden names, absolute or backslash paths, any type
outside the list above, and anything that resolves outside the runs folder (symlinks included).
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from herogen.lab import runindex

STATIC_DIR = Path(__file__).resolve().parent / "static"

RUN_FILE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".glb": "model/gltf-binary",
    ".json": "application/json",
}
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}

# The only external origin the page may load from: three.js, pinned, through the import map.
THREE_VERSION = "0.160.0"
CDN_PREFIX = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/"
CDN_ORIGIN = "https://cdn.jsdelivr.net"

_IMPORTMAP = re.compile(r"<script type=\"importmap\">(.*?)</script>", re.S)


def content_security_policy(page_html: str) -> str:
    """CSP for the page: own origin, the pinned CDN for scripts, and the inline import map by its hash."""
    hashes = " ".join(
        "'sha256-" + base64.b64encode(hashlib.sha256(m.encode()).digest()).decode() + "'"
        for m in _IMPORTMAP.findall(page_html)
    )
    return (
        "default-src 'self'; "
        f"script-src 'self' {CDN_ORIGIN} {hashes}; "
        "style-src 'self'; "
        "img-src 'self' blob: data:; "
        f"connect-src 'self' blob: data: {CDN_ORIGIN}; "
        "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )


def safe_run_file(root: Path, url_path: str) -> tuple[Path, str] | None:
    """(file, content type) for `/runs/<id>/<rel>` (already stripped to `<id>/<rel>`), or None if refused."""
    rel = unquote(url_path)
    if not rel or "\x00" in rel or "\\" in rel or rel.startswith("/"):
        return None
    parts = rel.split("/")
    if len(parts) < 2 or any(p in ("", ".", "..") or p.startswith(".") for p in parts):
        return None
    if runindex.run_dir(root, parts[0]) is None:
        return None
    ctype = RUN_FILE_TYPES.get(Path(parts[-1]).suffix.lower())
    if ctype is None:
        return None
    candidate = root.joinpath(*parts)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    root_r = root.resolve()
    if not resolved.is_relative_to(root_r) or not resolved.is_file():
        return None
    if RUN_FILE_TYPES.get(resolved.suffix.lower()) != ctype:
        return None
    return resolved, ctype


def safe_static_file(name: str) -> tuple[Path, str] | None:
    name = unquote(name)
    if not runindex.SAFE_NAME.match(name):
        return None
    ctype = STATIC_TYPES.get(Path(name).suffix.lower())
    p = STATIC_DIR / name
    if ctype is None or not p.is_file() or p.resolve().parent != STATIC_DIR:
        return None
    return p, ctype


class LabHandler(BaseHTTPRequestHandler):
    server_version = "HeroGenLab"
    sys_version = ""

    def __init__(self, *args: Any, runs_root: Path, quiet: bool = False, **kwargs: Any) -> None:
        self.runs_root = runs_root
        self.quiet = quiet
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (stdlib signature)
        if not self.quiet:
            super().log_message(format, *args)

    def do_HEAD(self) -> None:  # noqa: N802 (stdlib naming)
        self._route(head=True)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        self._route(head=False)

    def _route(self, head: bool) -> None:
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            f = safe_static_file("index.html")
            if f is None:
                return self._error(HTTPStatus.NOT_FOUND)
            body = f[0].read_bytes()
            return self._send(body, f[1], head, extra={"Content-Security-Policy":
                                                         content_security_policy(body.decode())})
        if path.startswith("/static/"):
            f = safe_static_file(path[len("/static/"):])
            return self._send_file(f, head)
        if path == "/api/runs":
            return self._json({"runs": runindex.run_list(self.runs_root),
                               "runs_folder_exists": self.runs_root.is_dir()}, head)
        if path.startswith("/api/runs/"):
            run_id = unquote(path[len("/api/runs/"):])
            run = runindex.run_dir(self.runs_root, run_id)
            if run is None:
                return self._error(HTTPStatus.NOT_FOUND)
            return self._json(runindex.run_detail(run), head)
        if path.startswith("/runs/"):
            return self._send_file(safe_run_file(self.runs_root, path[len("/runs/"):]), head)
        return self._error(HTTPStatus.NOT_FOUND)

    def _json(self, data: Any, head: bool) -> None:
        self._send(json.dumps(data).encode(), "application/json", head)

    def _send_file(self, f: tuple[Path, str] | None, head: bool) -> None:
        if f is None:
            return self._error(HTTPStatus.NOT_FOUND)
        try:
            body = f[0].read_bytes()
        except OSError:
            return self._error(HTTPStatus.NOT_FOUND)
        self._send(body, f[1], head)

    def _send(self, body: bytes, ctype: str, head: bool, status: HTTPStatus = HTTPStatus.OK,
              extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _error(self, status: HTTPStatus) -> None:
        self._send(f"{status.value} {status.phrase}\n".encode(), "text/plain; charset=utf-8", head=False,
                   status=status)


def make_server(runs_root: Path, port: int = 8790, quiet: bool = False) -> ThreadingHTTPServer:
    handler = partial(LabHandler, runs_root=Path(runs_root), quiet=quiet)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server
