"""The physical footprint of a process, the memory measure of the memory checks.

macOS counts a process's memory by its physical footprint (proc_pid_rusage, RUSAGE_INFO_V0). ps rss
also counts the freed pages that libmalloc has marked reusable and the kernel has not taken back,
which can grow by tens of MB while the process holds no more memory: on a hosted macOS runner,
test_memory_flat_long_run measured RSS growth of +1.41 MB and then +30.86 MB over the same windows in
which the footprint changed by +0.00 MB and then -2.12 MB.
"""
from __future__ import annotations

import ctypes
import os
import sys


class _RusageInfoV0(ctypes.Structure):
    """rusage_info_v0 of <sys/resource.h>, which proc_pid_rusage fills for RUSAGE_INFO_V0."""
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
        "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime")]


_LIBSYSTEM = ctypes.CDLL("/usr/lib/libSystem.B.dylib") if sys.platform == "darwin" else None


def footprint_mb(pid: int | None = None) -> float | None:
    """The physical footprint of ``pid`` (this process by default) in MB, or None off macOS."""
    if _LIBSYSTEM is None:
        return None
    info = _RusageInfoV0()
    if _LIBSYSTEM.proc_pid_rusage(os.getpid() if pid is None else pid, 0, ctypes.byref(info)) != 0:
        return None
    return info.phys_footprint / 2**20
