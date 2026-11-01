"""Local worker and subprocess-group exit evidence, conservative across hosts and PID reuse."""
import os
import socket


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
