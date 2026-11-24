"""Process and MPS limits with clean-stop requests."""
from __future__ import annotations

import json
from contextlib import contextmanager
from functools import lru_cache
import os
import re
import signal
import subprocess
import sys
import time

import torch

try:
    import resource
except ImportError:
    resource = None


class CleanStop(RuntimeError):
    """A resource limit or signal requests a checkpoint and exit."""


@lru_cache(maxsize=1)
def _autorelease_functions():
    import ctypes
    library = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    push, pop = library.objc_autoreleasePoolPush, library.objc_autoreleasePoolPop
    push.argtypes, push.restype = [], ctypes.c_void_p
    pop.argtypes, pop.restype = [ctypes.c_void_p], None
    return push, pop


@contextmanager
def autorelease_scope(device):
    """Release temporary Objective-C objects after one MPS work unit."""
    if torch.device(device).type != "mps":
        yield
        return
    if sys.platform != "darwin":
        raise ValueError("MPS autorelease scopes require macOS")
    push, pop = _autorelease_functions()
    pool = push()
    try:
        yield
    finally:
        pop(pool)


def free_percent():
    if sys.platform == "darwin":
        result = subprocess.check_output(["memory_pressure"], text=True)
        match = re.search(r"free percentage: (\d+)%", result)
        if match is None:
            raise RuntimeError("memory_pressure did not report a free percentage")
        return float(match[1])
    import psutil
    return 100.0 * psutil.virtual_memory().available / psutil.virtual_memory().total


def resident_bytes():
    import psutil
    return psutil.Process().memory_info().rss


def physical_footprint():
    """Read the macOS physical footprint, which includes unified device memory."""
    if sys.platform == "darwin":
        import ctypes
        # Field order follows rusage_info_v0 in the macOS SDK sys/resource.h.
        class Usage(ctypes.Structure):
            _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
                (name, ctypes.c_uint64) for name in (
                    "user", "system", "idle", "interrupt", "pageins", "wired",
                    "resident", "physical", "start", "exit",
                )]
        usage = Usage()
        try:
            library = ctypes.CDLL("/usr/lib/libproc.dylib")
            library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            library.proc_pid_rusage.restype = ctypes.c_int
            if library.proc_pid_rusage(os.getpid(), 0, ctypes.byref(usage)) == 0:
                return usage.physical
        except (OSError, AttributeError):
            pass
    return None


def peak_resident_bytes():
    if resource:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if sys.platform == "darwin" else value * 1024
    import psutil
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", info.rss)


class MemoryGuard:
    """Check total process memory and device allocation at bounded intervals."""

    def __init__(self, limit_gb, device, *, interval=15, pressure=free_percent,
                 resident=resident_bytes, footprint=physical_footprint, peak=peak_resident_bytes,
                 driver=None, clock=time.monotonic):
        if not 0 < limit_gb <= 24:
            raise ValueError("memory limit must be positive and at most 24 GB")
        self.limit = int(limit_gb * 1_000_000_000)
        self.device = torch.device(device)
        self.interval, self.pressure, self.resident, self.clock = interval, pressure, resident, clock
        self.footprint, self.peak_reader, self.driver_reader = footprint, peak, driver
        self.last = -float("inf")
        self._pressure_time = -float("inf")
        self._pressure_value = None
        self.low = 0
        self.peak = 0
        if self.device.type == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError("MPS is unavailable")
            recommended = torch.mps.recommended_max_memory()
            # Leave room for CPU allocations inside the total process limit.
            torch.mps.set_per_process_memory_fraction(min(1.0, 0.8 * self.limit / recommended))

    def check(self, stage, *, force=True):
        now = self.clock()
        if not force and now - self.last < self.interval:
            return
        self.last = now
        driver = (self.driver_reader() if self.driver_reader else
                  torch.mps.driver_allocated_memory() if self.device.type == "mps" else 0)
        rss = self.resident()
        footprint = self.footprint()
        # A missing physical-footprint API uses a conservative sum, not a maximum.
        process = footprint if footprint is not None else rss + driver
        self.peak = max(self.peak, process, rss, driver, self.peak_reader())
        # Forced stage checks refresh allocations without a pressure subprocess.
        # A low sample bypasses the cache to confirm pressure or recovery.
        if (self._pressure_value is None or self._pressure_value < 20
                or now - self._pressure_time >= self.interval):
            self._pressure_value = self.pressure()
            self._pressure_time = now
            self.low = self.low + 1 if self._pressure_value < 20 else 0
        free = self._pressure_value
        print(json.dumps({"stage": stage, "process_bytes": process, "rss_bytes": rss,
                          "physical_footprint_bytes": footprint, "peak_memory_bytes": self.peak,
                          "driver_bytes": driver, "free_percent": free,
                          "measurement": "physical_footprint" if footprint is not None else "rss_plus_driver_fallback"}),
              file=sys.stderr, flush=True)
        if self.peak > self.limit:
            raise CleanStop("process or MPS allocation exceeds the memory limit")
        if free < 15 or self.low >= 2:
            raise CleanStop("memory pressure requires a clean stop")


class StopControl:
    """Signals set a flag. They cannot interrupt a partial optimizer update."""

    def __init__(self):
        self.reason = None
        self.previous = {}

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self._signal)
        return self

    def _signal(self, number, frame):
        self.reason = f"signal {number}"

    def check(self):
        if self.reason:
            raise CleanStop(self.reason)

    def __exit__(self, *_):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)
