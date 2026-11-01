"""Private subprocess IPC to the service's existing durable metadata store.

The bridge owns no budget state. A reply is sent only after the store call returns, which on Modal means
its metadata Volume snapshot was committed. Pipe descriptors are inherited only by the stage subprocess.
"""
from __future__ import annotations

import json
import os
import select
import threading
import secrets
import sys
import tempfile
from multiprocessing.connection import Client, Listener
from contextlib import contextmanager

REQUEST_FD = "HEROGEN_REFINE_BUDGET_REQUEST_FD"
RESPONSE_FD = "HEROGEN_REFINE_BUDGET_RESPONSE_FD"
_CLIENT_LOCK = threading.Lock()


def bridge_enabled(prefix="HEROGEN_REFINE_BUDGET"):
    return prefix + "_REQUEST_FD" in os.environ or prefix + "_ADDRESS" in os.environ


def inherited_fds(env):
    fds = tuple(int(value) for key, value in env.items() if key.endswith('_FD'))
    return {'pass_fds': fds} if fds else {}


def remote_budget(action, *, prefix="HEROGEN_REFINE_BUDGET", **kwargs):
    # Parallel clients in one stage must not consume one another's acknowledgements.
    with _CLIENT_LOCK:
        return _remote_budget(action, prefix=prefix, **kwargs)


def _remote_budget(action, *, prefix, **kwargs):
    if prefix + "_ADDRESS" in os.environ:
        with Client(json.loads(os.environ[prefix + "_ADDRESS"]),
                    authkey=bytes.fromhex(os.environ[prefix + "_AUTHKEY"])) as conn:
            conn.send_bytes(json.dumps({"action": action, **kwargs}).encode())
            if not conn.poll(120):
                raise RuntimeError("Durable refinement budget acknowledgement timed out")
            response = json.loads(conn.recv_bytes(1024 * 1024))
        if "error" in response:
            raise RuntimeError("Durable refinement budget could not be acknowledged")
        return response["state"]
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
    if os.name == 'nt':
        with _connection_bridge(callback, prefix) as env:
            yield env
        return
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


@contextmanager
def _connection_bridge(callback, prefix):
    """Authenticated Windows named pipe; JSON only, with no inherited descriptor assumptions."""
    from multiprocessing import AuthenticationError
    key = secrets.token_bytes(32)
    done = threading.Event()
    # AF_UNIX lets the same transport and failure paths be exercised on POSIX without a network listener.
    with tempfile.TemporaryDirectory(prefix='herogen-ipc-') as folder:
        address = (r'\\.\pipe\herogen-' + secrets.token_hex(16) if sys.platform == 'win32'
                   else os.path.join(folder, 'bridge'))
        family = 'AF_PIPE' if sys.platform == 'win32' else 'AF_UNIX'
        with Listener(address, family=family, authkey=key) as listener:
            def serve():
                while True:
                    try:
                        conn = listener.accept()
                    except (AuthenticationError, OSError):
                        continue
                    with conn:
                        if done.is_set():
                            return
                        try:
                            if not conn.poll(120):
                                continue
                            result = {'state': callback(**json.loads(conn.recv_bytes(1024 * 1024)))}
                        except Exception:
                            result = {'error': 'budget unavailable'}
                        try:
                            conn.send_bytes(json.dumps(result).encode())
                        except (BrokenPipeError, OSError):
                            pass
            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            try:
                yield {prefix + '_ADDRESS': json.dumps(address), prefix + '_AUTHKEY': key.hex()}
            finally:
                done.set()
                with Client(address, family=family, authkey=key):
                    pass
                thread.join(timeout=5)
