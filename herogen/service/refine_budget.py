"""Private subprocess IPC to the service's existing durable metadata store.

The bridge owns no budget state. A reply is sent only after the store call returns, which on Modal means
its metadata Volume snapshot was committed. Pipe descriptors are inherited only by the stage subprocess.
"""
from __future__ import annotations

import json
import logging
import os
import select
import socket
import struct
import time
import threading
import secrets
import sys
import tempfile
from multiprocessing.connection import Client, Listener, answer_challenge, deliver_challenge
from contextlib import contextmanager

REQUEST_FD = "HEROGEN_REFINE_BUDGET_REQUEST_FD"
RESPONSE_FD = "HEROGEN_REFINE_BUDGET_RESPONSE_FD"
log = logging.getLogger("herogen.service")
_CLIENT_LOCK = threading.Lock()
_AUTH_TIMEOUT = 5


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
                    # A failed durable write must never authorize an HTTP send. The stage only learns that the
                    # budget is unavailable, so the service log keeps the store's actual error.
                    log.exception("texture-refine budget operation failed")
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


class _BoundedConnection:
    """Cancellable framed I/O for authentication as well as JSON requests and replies."""

    def __init__(self, conn, done):
        self.conn, self.done = conn, done
        self.deadline = time.monotonic() + _AUTH_TIMEOUT
        # AF_PIPE messages are atomic. AF_UNIX is a byte stream: poll() alone cannot bound a
        # recv_bytes() after only part of a frame arrives, so read its framing with socket timeouts.
        self.socket = None if sys.platform == 'win32' else socket.socket(fileno=conn.fileno())
        if self.socket is not None:
            self.socket.settimeout(.05)

    def close(self):
        if self.socket is not None:
            self.socket.detach()  # the Connection retains ownership of the descriptor

    def check(self):
        if self.done.is_set() or time.monotonic() >= self.deadline:
            raise TimeoutError('Budget bridge I/O stopped or timed out')

    def _read(self, size):
        data = bytearray()
        while len(data) < size:
            self.check()
            try:
                part = self.socket.recv(size - len(data))
            except TimeoutError:
                continue
            if not part:
                raise EOFError
            data.extend(part)
        return bytes(data)

    def recv_bytes(self, maxlength):
        if self.socket is not None:
            size, = struct.unpack('!i', self._read(4))
            if size == -1:
                size, = struct.unpack('!Q', self._read(8))
            if not 0 <= size <= maxlength:
                raise OSError('Budget bridge message too large')
            return self._read(size)
        while True:
            self.check()
            if self.conn.poll(.05):
                return self.conn.recv_bytes(maxlength)

    def send_bytes(self, data):
        if self.socket is not None:
            pending = memoryview(struct.pack('!i', len(data)) + data)
            while pending:
                self.check()
                try:
                    sent = self.socket.send(pending)
                except TimeoutError:
                    continue
                if not sent:
                    raise EOFError
                pending = pending[sent:]
            return
        # Connection.send_bytes() waits forever on an overlapped pipe write. Keep ownership
        # of the operation so shutdown can cancel it even if an authenticated peer stops reading.
        import _winapi
        self.check()
        operation, error = _winapi.WriteFile(self.conn.fileno(), data, overlapped=True)
        try:
            if error == _winapi.ERROR_IO_PENDING:
                while _winapi.WaitForMultipleObjects([operation.event], False, 50) == _winapi.WAIT_TIMEOUT:
                    self.check()
            written, error = operation.GetOverlappedResult(True)
            if error or written != len(data):
                raise OSError('Budget bridge pipe write failed')
        finally:
            operation.cancel()
            operation.GetOverlappedResult(True)


@contextmanager
def _connection_bridge(callback, prefix):
    """Authenticated Windows named pipe; JSON only, with bounded, cancellable peer I/O."""
    from multiprocessing import AuthenticationError
    key = secrets.token_bytes(32)
    done = threading.Event()
    # AF_UNIX lets the same transport and failure paths be exercised on POSIX without a network listener.
    with tempfile.TemporaryDirectory(prefix='herogen-ipc-') as folder:
        address = (r'\\.\pipe\herogen-' + secrets.token_hex(16) if sys.platform == 'win32'
                   else os.path.join(folder, 'bridge'))
        family = 'AF_PIPE' if sys.platform == 'win32' else 'AF_UNIX'
        # Listener's built-in authentication blocks inside accept(), before we own the connection.
        # Accept raw handles, then authenticate with deadlines before exposing any callback.
        with Listener(address, family=family) as listener:
            def serve():
                while True:
                    try:
                        conn = listener.accept()
                    except OSError:
                        if done.is_set():
                            return
                        continue
                    with conn:
                        if done.is_set():
                            return
                        bounded = _BoundedConnection(conn, done)
                        try:
                            try:
                                deliver_challenge(bounded, key)
                                answer_challenge(bounded, key)
                            except (AuthenticationError, EOFError, OSError, ValueError, AssertionError):
                                continue
                            bounded.deadline = time.monotonic() + 120
                            try:
                                request = json.loads(bounded.recv_bytes(1024 * 1024))
                                bounded.check()
                                result = {'state': callback(**request)}
                            except Exception:
                                log.exception("refinement budget operation failed")
                                result = {'error': 'budget unavailable'}
                            bounded.deadline = time.monotonic() + 120
                            try:
                                bounded.send_bytes(json.dumps(result).encode())
                            except (EOFError, OSError):
                                pass
                        finally:
                            bounded.close()
                    if done.is_set():
                        return
            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            try:
                yield {prefix + '_ADDRESS': json.dumps(address), prefix + '_AUTHKEY': key.hex()}
            finally:
                done.set()
                # Wake a raw accept; never wait for authentication from a stopped or failed peer.
                with Client(address, family=family):
                    pass
                thread.join()
