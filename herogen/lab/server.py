"""The lab's HTTP server: standard library only, bound to 127.0.0.1, read-only.

Routes:
  /                          the page (static/index.html), with its Content-Security-Policy; also at /index.html
  /static/<path>             the page's CSS, JS, SVG icons, fonts, JSON tokens and licence texts, subfolders included
                             (HTML is never served from here, so the page never comes without its CSP)
  /api/workspace             the runs roots, every run, the runs grouped into characters, and the versions
  /api/runs                  JSON list of every run in every runs root
  /api/runs/<root>/<run>     JSON detail of one run (records, images, bodies, scores, held-out views, quality)
  /api/versions/<version>    JSON detail of one frozen version (`<group>/<version>` or `<version>`)
  /runs/<root>/<run>/<path>  a PNG, JPEG, GLB or JSON file inside that run folder
  /versions/<version>/<file> a PNG, GLB or JSON file directly inside that version folder

`<root>` is a runs root's key: `r1`, `r2`, ... in the order the roots were given. File requests are refused (404)
for `..` or `.` segments, hidden names, absolute or backslash paths, any type outside the lists above, run folders
that are symlinks, and anything that resolves outside its own run or version folder (a link to a sibling run
included).

Every request must name the server as `127.0.0.1:<port>` or `localhost:<port>` in its Host header (and in an
absolute-form request line, if one is used), else it is refused with 421, so a page on another site cannot
reach the lab through DNS rebinding. Every error, including the ones the standard library raises itself for a
malformed request line (400, 414, 501, 505), is JSON (`{"error": ...}`) with the same security headers as any
other response; HEAD responses never carry a body.

Logging: the access log and error lines name the method and the path without its query string, and never a
header or the raw request line. An unexpected failure in a handler answers 500 and is logged with that
method and path plus the Python traceback.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
import traceback
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from herogen.lab import runindex, versions
from herogen.lab.workspace import Workspace

STATIC_DIR = Path(__file__).resolve().parent / "static"

RUN_FILE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".glb": "model/gltf-binary",
    ".json": "application/json",
}
# /static/ serves these types only. The page itself is served at / (with its CSP), never from /static/.
STATIC_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json",
    ".txt": "text/plain; charset=utf-8",
    ".ttf": "font/ttf",
    ".woff2": "font/woff2",
}
# Every response except the page itself: nothing in it may run or load anything, even if it is opened on its own
# (an SVG or a JSON file opened directly in a tab).
FILE_CSP = "default-src 'none'; sandbox"
PAGE = STATIC_DIR / "index.html"
PAGE_TYPE = "text/html; charset=utf-8"

# The only external source the page may load from: three.js, pinned, through the import map. The CSP allows
# this path prefix only, not the whole CDN.
THREE_VERSION = "0.160.0"
CDN_PREFIX = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/"

_IMPORTMAP = re.compile(r"<script type=\"importmap\">(.*?)</script>", re.S)


def content_security_policy(page_html: str) -> str:
    """CSP for the page: own origin, the pinned CDN for scripts, and the inline import map by its hash."""
    hashes = " ".join(
        "'sha256-" + base64.b64encode(hashlib.sha256(m.encode()).digest()).decode() + "'"
        for m in _IMPORTMAP.findall(page_html)
    )
    return (
        "default-src 'self'; "
        f"script-src 'self' {CDN_PREFIX} {hashes}; "
        "style-src 'self'; "
        "img-src 'self' blob: data:; "
        f"connect-src 'self' blob: data: {CDN_PREFIX}; "
        "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )


def safe_run_file(root: Path, url_path: str) -> tuple[Path, str] | None:
    """(file, content type) for `<run>/<rel>` in one runs folder (`url_path` still percent-encoded), or None if
    refused."""
    return run_file_in(root, unquote(url_path))


def run_file_in(root: Path, rel: str) -> tuple[Path, str] | None:
    """(file, content type) for the decoded path `<run>/<rel>` inside `root`, or None if refused."""
    if not rel or "\x00" in rel or "\\" in rel or rel.startswith("/"):
        return None
    parts = rel.split("/")
    if len(parts) < 2 or any(p in ("", ".", "..") or p.startswith(".") for p in parts):
        return None
    run = runindex.run_dir(root, parts[0])
    if run is None:
        return None
    ctype = RUN_FILE_TYPES.get(Path(parts[-1]).suffix.lower())
    if ctype is None:
        return None
    try:
        resolved = root.joinpath(*parts).resolve(strict=True)
        run_r = run.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    # Inside this run's own folder: a link to a sibling run or to any other folder of the root is refused.
    if not resolved.is_relative_to(run_r) or not resolved.is_file():
        return None
    if RUN_FILE_TYPES.get(resolved.suffix.lower()) != ctype:
        return None
    return resolved, ctype


def safe_static_file(name: str) -> tuple[Path, str] | None:
    """(file, content type) for `/static/<name>`, where `name` may name a file in a subfolder (`icons/box.svg`).

    Every segment must be a plain name (no `..`, `.`, hidden names, backslashes or NUL), the type must be in
    STATIC_TYPES after decoding and case-folding the extension (so `%69ndex.html` or `Index.HTML` is refused like
    `index.html`), and the file must resolve inside the static folder (symlinks included)."""
    name = unquote(name)
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        return None
    parts = name.split("/")
    if any(not runindex.SAFE_NAME.match(p) for p in parts):
        return None
    ctype = STATIC_TYPES.get(Path(parts[-1]).suffix.lower())
    if ctype is None:
        return None
    try:
        resolved = STATIC_DIR.joinpath(*parts).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_relative_to(STATIC_DIR) or not resolved.is_file():
        return None
    if STATIC_TYPES.get(resolved.suffix.lower()) != ctype:
        return None
    return resolved, ctype


class LabHandler(BaseHTTPRequestHandler):
    server_version = "HeroGenLab"
    sys_version = ""
    timeout = 30  # seconds a connection may sit idle or half-sent before it is closed

    def __init__(self, *args: Any, workspace: Workspace, quiet: bool = False, **kwargs: Any) -> None:
        self.workspace = workspace
        self.quiet = quiet
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (stdlib signature)
        if not self.quiet:
            super().log_message(format, *args)

    def _log_target(self) -> str:
        """The request's path without query or fragment, for logs; "-" when the request line never parsed."""
        raw = getattr(self, "path", None)
        if not raw:
            return "-"
        try:
            return urlsplit(raw).path or "-"
        except ValueError:
            return "-"

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        # The method and the path without its query string only. send_error can call this before the request
        # line parsed, when there is no command or path yet.
        if isinstance(code, HTTPStatus):
            code = code.value
        self.log_message('"%s %s" %s %s', self.command or "-", self._log_target(), str(code), str(size))

    def log_error(self, format: str, *args: Any) -> None:  # noqa: A002 (stdlib signature)
        # The stdlib passes request data (the raw request line, with its query) as arguments: log the fixed
        # text only.
        self.log_message("%s", format.split("%")[0].rstrip(" :(") or "error")

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """Errors the stdlib raises itself (malformed request line, URI too long, unsupported method) answered like
        ours: JSON with our headers, the Host checked when there are headers to check, and only the status logged.
        `message` and `explain` are dropped: they can quote the raw request line."""
        self.close_connection = True
        if self.request_version == "HTTP/0.9":
            self.request_version = "HTTP/1.0"  # the request line never parsed: still send a status line
        try:
            status = HTTPStatus(code)
        except ValueError:
            status = HTTPStatus.BAD_REQUEST
        if getattr(self, "headers", None) is not None and self.request_refusal() is not None:
            status = self.request_refusal()
        try:
            self._error(status, head=self.command == "HEAD")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_HEAD(self) -> None:  # noqa: N802 (stdlib naming)
        self._handle(head=True)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        self._handle(head=False)

    def _handle(self, head: bool) -> None:
        """Route the request; any unexpected exception becomes a 500 JSON error instead of a dropped connection."""
        self._responded = False
        try:
            self._route(head)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser went away mid-response
        except Exception:  # noqa: BLE001 (last resort: answer 500 rather than drop the connection)
            if not self.quiet:
                sys.stderr.write(f"lab: {self.command} {self._log_target()} failed\n{traceback.format_exc()}")
            if not self._responded:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, head)

    def _own_names(self) -> tuple[str, str]:
        port = self.server.server_address[1]
        return f"127.0.0.1:{port}", f"localhost:{port}"

    def host_allowed(self) -> bool:
        """True when the one Host header is this server's own loopback name and port."""
        hosts = self.headers.get_all("Host") or []
        return len(hosts) == 1 and hosts[0].strip().lower() in self._own_names()

    def request_refusal(self) -> HTTPStatus | None:
        """None for a request this server answers; else why not. An absolute-form request line
        (`GET http://name/path`) must name this server too, whatever the Host header says."""
        try:
            target = urlsplit(self.path)
        except ValueError:
            return HTTPStatus.BAD_REQUEST
        if not self.host_allowed():
            return HTTPStatus.MISDIRECTED_REQUEST
        if (target.scheme or target.netloc) and (target.scheme != "http"
                                                 or target.netloc.lower() not in self._own_names()):
            return HTTPStatus.MISDIRECTED_REQUEST
        return None

    def _route(self, head: bool) -> None:
        refusal = self.request_refusal()
        if refusal is not None:
            return self._error(refusal, head)
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            body = PAGE.read_bytes()
            return self._send(body, PAGE_TYPE, head, extra={"Content-Security-Policy":
                                                             content_security_policy(body.decode())})
        if path.startswith("/static/"):
            f = safe_static_file(path[len("/static/"):])
            return self._send_file(f, head)
        if path == "/api/workspace":
            return self._json(self.workspace.workspace(), head)
        if path == "/api/runs":
            return self._json({"runs": self.workspace.runs(),
                               "roots": [{"key": k, "exists": r.is_dir()} for k, r in self.workspace.roots]}, head)
        if path.startswith("/api/runs/"):
            detail = self.workspace.run_detail(unquote(path[len("/api/runs/"):]))
            if detail is None:
                return self._error(HTTPStatus.NOT_FOUND, head)
            return self._json(detail, head)
        if path.startswith("/api/versions/"):
            detail = self.workspace.version_detail(unquote(path[len("/api/versions/"):]))
            if detail is None:
                return self._error(HTTPStatus.NOT_FOUND, head)
            return self._json(detail, head)
        if path.startswith("/runs/"):
            return self._send_file(self.run_file(path[len("/runs/"):]), head)
        if path.startswith("/versions/"):
            found = versions.safe_version_file(self.workspace.versions_root, unquote(path[len("/versions/"):]))
            ctype = RUN_FILE_TYPES.get(found.suffix.lower()) if found else None
            return self._send_file((found, ctype) if found and ctype else None, head)
        return self._error(HTTPStatus.NOT_FOUND, head)

    def run_file(self, url_path: str) -> tuple[Path, str] | None:
        """`<root>/<run>/<rel>` -> (file, content type) inside that run, or None."""
        key, sep, rest = unquote(url_path).partition("/")
        root = self.workspace.root(key) if sep else None
        return run_file_in(root, rest) if root is not None else None

    def _json(self, data: Any, head: bool, status: HTTPStatus = HTTPStatus.OK) -> None:
        # allow_nan=False: NaN and Infinity are not JSON and would break the browser's parser.
        self._send(json.dumps(data, allow_nan=False).encode(), "application/json", head, status=status)

    def _send_file(self, f: tuple[Path, str] | None, head: bool) -> None:
        if f is None:
            return self._error(HTTPStatus.NOT_FOUND, head)
        try:
            body = f[0].read_bytes()
        except OSError:
            return self._error(HTTPStatus.NOT_FOUND, head)
        self._send(body, f[1], head)

    def _send(self, body: bytes, ctype: str, head: bool, status: HTTPStatus = HTTPStatus.OK,
              extra: dict[str, str] | None = None) -> None:
        self._responded = True
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        headers = {"Content-Security-Policy": FILE_CSP, **(extra or {})}
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _error(self, status: HTTPStatus, head: bool) -> None:
        self._json({"error": f"{status.value} {status.phrase}"}, head, status=status)


def make_server(workspace: Workspace | Path, port: int = 8790, quiet: bool = False) -> ThreadingHTTPServer:
    """A server for `workspace` (or for one runs folder) on 127.0.0.1:`port`; port 0 picks a free one."""
    ws = workspace if isinstance(workspace, Workspace) else Workspace([Path(workspace)])
    handler = partial(LabHandler, workspace=ws, quiet=quiet)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server
