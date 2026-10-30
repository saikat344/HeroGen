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
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterator
from unittest import mock

import numpy as np
from PIL import Image

from herogen.pipeline import __main__ as main_mod
from herogen.pipeline import client as client_mod
from herogen.pipeline import retry

# responder(endpoint, body) -> (status, extra headers, JSON payload[, delay in seconds before answering[, seconds
# between pieces[, framing]]]). Status 0 closes the connection without an answer. With a piece time, the answer
# is written by hand in 4-byte pieces: ``framing`` "length" (the default; the body trickles), "headers" (the
# status line and headers trickle), "chunked" (chunked transfer coding; every chunk and the trailers trickle),
# "trailers" (chunked; only the trailers trickle) or "short" (Content-Length claims more than is sent, then the
# connection closes; nothing trickles).
FRAMINGS = ("length", "headers", "chunked", "trailers", "short")
Responder = Callable[[str, dict], tuple]


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


def is_question(body: dict) -> bool:
    """A Muse Spark question (a responses call with a JSON schema), not a SAM 3.1 call."""
    return "text" in body and body.get("model") == client_mod.TEXT_MODEL


def question_answer(answer: dict) -> dict:
    """A Muse Spark answer from the responses endpoint, with its token usage."""
    return {"status": "completed", "usage": {"input_tokens": 1200, "output_tokens": 150},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(answer)}]}]}


NOT_A_PHOTO = {"real_person_photo_probability": 0.0, "image_kind": "drawing_or_artwork", "reason": "a drawing"}


def error_body(status: int) -> dict:
    return {"error": {"code": status, "message": f"fake error {status}"}}


class FakeModelServer:
    """Serves POST /v1/<endpoint> from ``responder``. ``requests`` lists (endpoint, headers) of every request,
    with the Authorization header replaced by whether it carried the expected key; ``max_in_flight`` is the
    most requests it was answering at once."""

    def __init__(self, responder: Responder, key: str) -> None:
        self.responder = responder
        self.key = key
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                endpoint = self.path.split("/v1/", 1)[-1]
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                headers = {k: v for k, v in self.headers.items() if k.lower() != "authorization"}
                headers["auth_ok"] = str(self.headers.get("Authorization") == f"Bearer {server.key}")
                with server.lock:
                    server.requests.append((endpoint, headers))
                    server.in_flight += 1
                    server.max_in_flight = max(server.max_in_flight, server.in_flight)
                try:
                    self._answer(endpoint, body)
                finally:
                    with server.lock:
                        server.in_flight -= 1

            def _answer(self, endpoint: str, body: dict) -> None:
                status, extra, payload, *rest = server.responder(endpoint, body)
                if rest and rest[0]:
                    time.sleep(rest[0])
                if status == 0:
                    self.close_connection = True
                    self.connection.close()
                    return
                data = json.dumps(payload).encode()
                if len(rest) > 1 and (rest[1] or len(rest) > 2):
                    self._write_by_hand(status, extra, data, rest[1], rest[2] if len(rest) > 2 else "length")
                    return
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in extra.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def _write_by_hand(self, status: int, extra: dict, data: bytes, piece_s: float, framing: str) -> None:
                assert framing in FRAMINGS, framing
                self.close_connection = True
                chunked = framing in ("chunked", "trailers")
                head = [f"HTTP/1.1 {status} OK", "Content-Type: application/json", "Connection: close"]
                head += ["Transfer-Encoding: chunked" if chunked else
                         f"Content-Length: {len(data) + (50 if framing == 'short' else 0)}"]
                head += [f"{k}: {v}" for k, v in extra.items()]
                parts: list[tuple[bytes, bool]] = [("\r\n".join(head).encode() + b"\r\n\r\n", framing == "headers")]
                if chunked:
                    for i in range(0, len(data), 4):
                        piece = data[i:i + 4]
                        parts.append((b"%x\r\n" % len(piece) + piece + b"\r\n", framing == "chunked"))
                    parts.append((b"0\r\n", False))
                    parts.append((b"X-Trailer: " + b"t" * 200 + b"\r\n\r\n", True))
                else:
                    parts.append((data, framing == "length"))
                for part, slow in parts:
                    pieces = [part[i:i + 4] for i in range(0, len(part), 4)] if slow else [part]
                    for piece in pieces:
                        self.wfile.write(piece)
                        self.wfile.flush()
                        if slow:
                            time.sleep(piece_s)

            def log_message(self, *_a: Any) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.handle_error = lambda *_a: None  # a client that gave up (a timeout test) is not an error
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def count(self, endpoint: str) -> int:
        return sum(1 for e, _ in self.requests if e == endpoint)

    @contextmanager
    def serving(self, limiter_dir: str) -> Iterator["FakeModelServer"]:
        """Run the server and point the client at it (no proxy, the real `_post`), with the shared limiter's
        state under ``limiter_dir``."""
        self.thread.start()
        try:
            with mock.patch.object(client_mod, "API_BASE", self.base), \
                    mock.patch.dict("os.environ", {"no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1",
                                                   retry.LIMITER_DIR_ENV: limiter_dir}):
                yield self
        finally:
            self.httpd.shutdown()
            self.httpd.server_close()


def no_wait_policy() -> retry.RetryPolicy:
    """The retry policy with every backoff wait zero and no sleeping (no Retry-After is sent unless a test
    sends one)."""
    return retry.RetryPolicy(base_s=0.0, sleep=lambda s: None)


@contextmanager
def pipeline_without_waits() -> Iterator[None]:
    """`run_pipeline` builds its client with `no_wait_policy`."""
    real = main_mod.ModelClient
    with mock.patch.object(main_mod, "ModelClient", lambda *a, **k: real(*a, policy=no_wait_policy(), **k)):
        yield
