"""An exclusive lock on an open file, shared by every process on the machine.

``lock(fd)`` waits until no other open file (in this process or any other) holds the lock on the same file,
then takes it; ``unlock(fd)`` releases it, and ``with exclusive(fd):`` holds it for a block. A process that
dies holding it loses it.

- POSIX: ``fcntl.flock`` on the whole file.
- Windows: ``LockFileEx`` on one byte at offset 2**63 - 2**32, far past the end of any real file. Windows
  byte-range locks are mandatory, so locking the file's contents would make reads and writes of them fail;
  a byte past the end guards the same file without touching them.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class _Overlapped(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p),
                    ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.DWORD, ctypes.POINTER(_Overlapped)]
    _kernel32.LockFileEx.restype = wintypes.BOOL
    _kernel32.UnlockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                       ctypes.POINTER(_Overlapped)]
    _kernel32.UnlockFileEx.restype = wintypes.BOOL
    _LOCKFILE_EXCLUSIVE_LOCK = 0x2
    _OFFSET_HIGH = 0x7FFFFFFF  # with a low part of 0: byte 2**63 - 2**32

    def _region() -> _Overlapped:
        return _Overlapped(Offset=0, OffsetHigh=_OFFSET_HIGH)

    def lock(fd: int) -> None:
        # The handle is synchronous, so LockFileEx waits until the byte is free.
        if not _kernel32.LockFileEx(msvcrt.get_osfhandle(fd), _LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0,
                                    ctypes.byref(_region())):
            raise ctypes.WinError(ctypes.get_last_error())

    def unlock(fd: int) -> None:
        if not _kernel32.UnlockFileEx(msvcrt.get_osfhandle(fd), 0, 1, 0, ctypes.byref(_region())):
            raise ctypes.WinError(ctypes.get_last_error())
else:
    import fcntl

    def lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def exclusive(fd: int) -> Iterator[None]:
    """Hold the exclusive lock on the open file ``fd`` for the block (see the module docstring)."""
    lock(fd)
    try:
        yield
    finally:
        unlock(fd)
