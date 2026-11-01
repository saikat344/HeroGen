"""Windows job-object ownership: no breakaway, kill on close, and explicit empty-tree evidence."""
import ctypes
import time
from ctypes import c_uint32, c_int64, c_size_t, c_void_p, Structure


class _Limits(Structure):
    _fields_ = [('process_time', c_int64), ('job_time', c_int64), ('flags', c_uint32),
                ('min_working_set', c_size_t), ('max_working_set', c_size_t), ('active_limit', c_uint32),
                ('affinity', c_size_t), ('priority', c_uint32), ('scheduling', c_uint32)]


class _IO(Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ('reads','writes','other','read_bytes','write_bytes','other_bytes')]


class _ExtendedLimits(Structure):
    _fields_ = [('basic', _Limits), ('io', _IO), ('process_memory', c_size_t), ('job_memory', c_size_t),
                ('peak_process_memory', c_size_t), ('peak_job_memory', c_size_t)]


class _Accounting(Structure):
    _fields_ = [(name, c_int64) for name in ('user','kernel','period_user','period_kernel')] + [
        (name, c_uint32) for name in ('faults','total','active','terminated')]


def _kernel():
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    for name, args, result in (
        ('CreateJobObjectW', [c_void_p, ctypes.c_wchar_p], c_void_p),
        ('SetInformationJobObject', [c_void_p, c_uint32, c_void_p, c_uint32], ctypes.c_int),
        ('OpenProcess', [c_uint32, ctypes.c_int, c_uint32], c_void_p),
        ('AssignProcessToJobObject', [c_void_p, c_void_p], ctypes.c_int),
        ('TerminateJobObject', [c_void_p, c_uint32], ctypes.c_int),
        ('QueryInformationJobObject', [c_void_p, c_uint32, c_void_p, c_uint32, c_void_p], ctypes.c_int),
        ('CloseHandle', [c_void_p], ctypes.c_int),
    ):
        fn = getattr(k, name)
        fn.argtypes, fn.restype = args, result
    return k


class WindowsJob:
    def __init__(self):
        self.kernel = _kernel()
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; neither breakaway flag is set.
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def assign(self, pid):
        handle = self.kernel.OpenProcess(0x0101, False, pid)  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.kernel.CloseHandle(handle)

    def terminate(self):
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def wait_empty(self, timeout=10):
        deadline = time.monotonic() + timeout
        while True:
            state = _Accounting()
            if not self.kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(state), ctypes.sizeof(state), None):
                return False
            if state.active == 0:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(.02, remaining))

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
