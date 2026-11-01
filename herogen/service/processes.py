"""Local worker and subprocess-group exit evidence, conservative across hosts and PID reuse."""
import os
import socket
import time
import sys
import subprocess
from contextlib import contextmanager

if sys.platform != 'win32':
    from os import killpg
    from signal import SIGKILL


def identity() -> dict:
    return {'host': socket.gethostname(), 'pid': os.getpid()}


def stopped(process: dict, *, group: bool = False) -> bool:
    if process.get('host') != socket.gethostname() or type(process.get('pid')) is not int or process['pid'] <= 0:
        return False
    if not group:
        # The limiter already uses OpenProcess/WaitForSingleObject on Windows, never a console signal.
        from herogen.pipeline.retry import _alive
        try:
            return not _alive(process['pid'])
        except (OSError, OverflowError):
            return False
    if os.name == 'nt':
        # A PID alone cannot prove an entire Windows job object empty. Its owner records that proof.
        return False
    try:
        killpg(process['pid'], 0)
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


@contextmanager
def stage_process(argv, **kwargs):
    """Contain a stage's descendants before allowing its first durable acknowledgement."""
    job = None
    if sys.platform == 'win32':
        from herogen.service.windows_job import WindowsJob
        job = WindowsJob()
        # The wrapper cannot spawn the real stage until registration and job assignment both succeed.
        argv = [sys.executable, '-m', 'herogen.service.stage_process', *argv]
    else:
        kwargs['start_new_session'] = True
    try:
        with subprocess.Popen(argv, **kwargs) as proc:
            try:
                if job is not None:
                    job.assign(proc.pid)
                def stop():
                    if job is not None:
                        job.terminate()
                        proc.wait()
                        return job.wait_empty()
                    try:
                        killpg(proc.pid, SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                    return wait_stopped({**identity(), 'pid': proc.pid}, group=True)
                yield proc, stop
            finally:
                if proc.poll() is None:
                    if job is not None:
                        job.terminate()
                    proc.kill()
                    proc.wait()
    finally:
        if job is not None:
            job.close()
