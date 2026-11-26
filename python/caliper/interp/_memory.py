"""Process and MPS limits with clean-stop requests (40 GB process cap, 8 GB host reserve)."""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import json
import math
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

MAX_PROCESS_GB = 40
HOST_RESERVE_BYTES = 8_000_000_000


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


@contextmanager
def mps_batch_pool_scope(device):
    """Drain per-batch Objective-C Metal allocations during multi-batch attribution."""
    if torch.device(device).type != "mps" or sys.platform != "darwin" or sys.getprofile() is not None:
        yield
        return
    push, pop = _autorelease_functions()
    pools: dict[int, object] = {}
    tracked = {
        "compute_batch",
        "compute_partial_influences",
        "cache_activations",
        "calculate_delta_hook",
        "_get_decoder_vectors",
    }

    def _profiler(frame, event, _arg):
        if event not in ("call", "return"):
            return
        fn = frame.f_code.co_name
        if fn not in tracked or "circuit_tracer/" not in frame.f_code.co_filename:
            return
        key = id(frame)
        if event == "call":
            pools[key] = push()
        elif key in pools:
            pop(pools.pop(key))
            if fn == "compute_batch" and torch.backends.mps.is_available():
                torch.mps.synchronize()

    sys.setprofile(_profiler)
    try:
        yield
    finally:
        sys.setprofile(None)
        for pool in reversed(list(pools.values())):
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


def host_available_bytes():
    import psutil
    return int(psutil.virtual_memory().available)


def resident_bytes():
    import psutil
    return int(psutil.Process().memory_info().rss)


def physical_footprint():
    """Read the macOS physical footprint, which includes unified device memory."""
    if sys.platform == "darwin":
        import ctypes

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
                return int(usage.physical)
        except (OSError, AttributeError):
            pass
    return None


def peak_resident_bytes():
    if resource:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    import psutil
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


class MemoryGuard:
    """Check total process memory (<= 40 GB) and host reserve (>= 8 GB) at bounded intervals."""

    def __init__(self, limit_gb=MAX_PROCESS_GB, device="cpu", *, interval=15, pressure=free_percent,
                 available=None, resident=resident_bytes, footprint=physical_footprint,
                 peak=peak_resident_bytes, driver=None, clock=time.monotonic):
        if isinstance(limit_gb, bool) or not isinstance(limit_gb, (int, float)) or not math.isfinite(limit_gb) or not 0 < limit_gb <= MAX_PROCESS_GB:
            raise ValueError(f"memory limit must be positive and at most {MAX_PROCESS_GB} GB")
        self.limit = int(limit_gb * 1_000_000_000)
        self.device = torch.device(device)
        self.interval, self.pressure, self.resident, self.clock = interval, pressure, resident, clock
        # Default to checking the 8 GB host reserve whenever using real system sampling.
        self.available_reader = available if available is not None else (
            host_available_bytes if pressure is free_percent else None
        )
        self.footprint, self.peak_reader, self.driver_reader = footprint, peak, driver
        self.last = -float("inf")
        self._pressure_time = -float("inf")
        self._pressure_value = None
        self.low = 0
        self.peak = 0
        self.current = 0
        if self.device.type == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError("MPS is unavailable")
            recommended = torch.mps.recommended_max_memory()
            torch.mps.set_per_process_memory_fraction(min(1.0, 0.85 * self.limit / recommended))

    def check(self, stage, *, force=True):
        now = self.clock()
        if not force and now - self.last < self.interval:
            return self.current
        self.last = now
        driver = (self.driver_reader() if self.driver_reader else
                  torch.mps.driver_allocated_memory() if self.device.type == "mps" else 0)
        rss = self.resident()
        footprint = self.footprint()
        process = footprint if footprint is not None else rss + driver
        self.current = max(process, rss, driver)
        self.peak = max(self.peak, self.current, self.peak_reader())
        avail = self.available_reader() if self.available_reader is not None else None
        if (self._pressure_value is None or self._pressure_value < 20
                or now - self._pressure_time >= self.interval):
            self._pressure_value = self.pressure()
            self._pressure_time = now
            self.low = self.low + 1 if self._pressure_value < 20 else 0
        free = self._pressure_value
        record = {
            "stage": stage,
            "process_bytes": process,
            "rss_bytes": rss,
            "physical_footprint_bytes": footprint,
            "peak_memory_bytes": self.peak,
            "driver_bytes": driver,
            "free_percent": free,
            "measurement": "physical_footprint" if footprint is not None else "rss_plus_driver_fallback",
        }
        if avail is not None:
            record["host_available_bytes"] = avail
        print(json.dumps(record), file=sys.stderr, flush=True)
        if self.peak > self.limit:
            raise CleanStop("process or MPS allocation exceeds the memory limit")
        if avail is not None and avail < HOST_RESERVE_BYTES:
            raise CleanStop("host available memory fell below the 8 GB reserve")
        if free < 15 or self.low >= 2:
            raise CleanStop("memory pressure requires a clean stop")
        return self.current

    def reserve(self, stage, size):
        self.check(stage)
        if self.current + int(size) > self.limit:
            raise CleanStop(f"{stage} exceeds the process memory limit")


class BudgetMemory:
    """Narrower budget wrapper around an existing ``MemoryGuard``."""

    def __init__(self, guard: MemoryGuard, limit_gb: float, *, extra: MemoryGuard | None = None):
        if not 0 < limit_gb <= MAX_PROCESS_GB:
            raise ValueError(f"budget limit must be positive and at most {MAX_PROCESS_GB} GB")
        self._guard = guard
        self.extra = extra
        self.limit = min(int(limit_gb * 1_000_000_000), guard.limit)
        self.device = guard.device

    @property
    def peak(self) -> int:
        return max(self._guard.peak, getattr(self.extra, "peak", 0))

    @property
    def current(self) -> int:
        return max(self._guard.current, getattr(self.extra, "current", 0))

    def check(self, stage, *, force=True):
        val = self._guard.check(stage, force=force)
        if self.extra is not None:
            self.extra.check(stage, force=force)
        if self.peak > self.limit:
            raise CleanStop("allocation exceeds the scoped memory budget")
        return val

    def reserve(self, stage, size):
        self.check(stage)
        if self.current + int(size) > self.limit:
            raise CleanStop(f"{stage} exceeds the scoped memory budget")


class StopControl:
    """Signals set a flag for clean boundary termination."""

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
