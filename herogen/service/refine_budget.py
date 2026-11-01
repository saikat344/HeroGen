"""Private subprocess pipes to the service's existing durable metadata store.

The bridge owns no budget state. A reply is sent only after the store call returns, which on Modal means
its metadata Volume snapshot was committed. Pipe descriptors are inherited only by the stage subprocess.
"""
from __future__ import annotations

import json
import os
import select
import threading
from contextlib import contextmanager

REQUEST_FD = "HEROGEN_REFINE_BUDGET_REQUEST_FD"
RESPONSE_FD = "HEROGEN_REFINE_BUDGET_RESPONSE_FD"
_CLIENT_LOCK = threading.Lock()


def remote_budget(action, *, prefix="HEROGEN_REFINE_BUDGET", **kwargs):
    # Parallel clients in one stage must not consume one another's acknowledgements.
    with _CLIENT_LOCK:
        return _remote_budget(action, prefix=prefix, **kwargs)


def _remote_budget(action, *, prefix, **kwargs):
    request_fd, response_fd = int(os.environ[prefix + "_REQUEST_FD"]), int(os.environ[prefix + "_RESPONSE_FD"])
    os.write(request_fd, json.dumps({"action": action, **kwargs}).encode() + b"\n")
    if not select.select([response_fd], [], [], 120)[0]:
        raise RuntimeError("Durable refinement budget acknowledgement timed out")
    with os.fdopen(os.dup(response_fd), "rb") as stream:
        response = json.loads(stream.readline(1024 * 1024))
    if "error" in response:
        raise RuntimeError("Durable refinement budget could not be acknowledged")
    return response["state"]


@contextmanager
def budget_bridge(callback, prefix="HEROGEN_REFINE_BUDGET"):
    """Expose only this job attempt's budget operations to its stage subprocess."""
    request_read, request_write = os.pipe()
    response_read, response_write = os.pipe()

    def serve():
        with os.fdopen(request_read, "rb") as requests, os.fdopen(response_write, "wb") as responses:
            for line in requests:
                try:
                    result = {"state": callback(**json.loads(line))}
                except Exception:
                    # A failed durable write must never authorize an HTTP send.
                    result = {"error": "budget unavailable"}
                try:
                    responses.write(json.dumps(result).encode() + b"\n")
                    responses.flush()
                except BrokenPipeError:
                    return

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield {prefix + "_REQUEST_FD": str(request_write), prefix + "_RESPONSE_FD": str(response_read)}
    finally:
        os.close(request_write)
        os.close(response_read)
        thread.join(timeout=5)
