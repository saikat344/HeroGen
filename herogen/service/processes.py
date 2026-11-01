"""Local worker and subprocess-group exit evidence, conservative across hosts and PID reuse."""
import os
import socket
import time


def identity() -> dict:
    return {'host': socket.gethostname(), 'pid': os.getpid()}


def stopped(process: dict, *, group: bool = False) -> bool:
    if process.get('host') != socket.gethostname() or type(process.get('pid')) is not int or process['pid'] <= 0:
        return False
    try:
        if group:
            os.killpg(process['pid'], 0)
        else:
            os.kill(process['pid'], 0)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    return False


def wait_stopped(process: dict, *, group: bool = False, timeout: float = 10) -> bool:
    """Keep the owning host available while killed descendants exit and are reaped."""
    deadline = time.monotonic() + timeout
    while not stopped(process, group=group):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(.02, remaining))
    return True
