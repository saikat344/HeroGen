"""A local HTTP server that stands in for Meta's Model API, so tests drive the real client end to end (its
HTTP calls, status handling and retries) and can inject faults: HTTP errors, empty SAM results.

Not a test module itself (no `test` prefix); the fault and retry tests import it.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterator
from unittest import mock

import numpy as np
from PIL import Image

from herogen.pipeline import client as client_mod

# responder(endpoint, body) -> (status, extra headers, JSON payload)
Responder = Callable[[str, dict], tuple[int, dict[str, str], Any]]


def pixels_key(img: Image.Image) -> str:
    """A hash of an image's pixels (not its PNG bytes, which differ between encoders)."""
    a = np.asarray(img.convert("RGB"))
    return hashlib.sha256(repr(a.shape).encode() + a.tobytes()).hexdigest()[:16]


def data_url_image(url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


def sam_response(text: str) -> dict:
    """A SAM 3.1 answer from the responses endpoint, which always says its status."""
    return {"status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


def error_body(status: int) -> dict:
    return {"error": {"code": status, "message": f"fake error {status}"}}


class FakeModelServer:
    """Serves POST /v1/<endpoint> from ``responder``. ``requests`` lists (endpoint, headers) of every request,
    with the Authorization header replaced by whether it carried the expected key."""

    def __init__(self, responder: Responder, key: str) -> None:
        self.responder = responder
        self.key = key
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                endpoint = self.path.split("/v1/", 1)[-1]
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                headers = {k: v for k, v in self.headers.items() if k.lower() != "authorization"}
                headers["auth_ok"] = str(self.headers.get("Authorization") == f"Bearer {server.key}")
                with server.lock:
                    server.requests.append((endpoint, headers))
                status, extra, payload = server.responder(endpoint, body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in extra.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_a: Any) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def count(self, endpoint: str) -> int:
        return sum(1 for e, _ in self.requests if e == endpoint)

    @contextmanager
    def serving(self) -> Iterator["FakeModelServer"]:
        """Run the server and point the client at it (no proxy, the real `_post`)."""
        self.thread.start()
        try:
            with mock.patch.object(client_mod, "API_BASE", self.base), \
                    mock.patch.dict("os.environ", {"no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1"}):
                yield self
        finally:
            self.httpd.shutdown()
            self.httpd.server_close()
