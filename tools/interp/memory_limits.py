"""Process and MPS limits with clean-stop requests."""
from __future__ import annotations

import json
import os
import re
import resource
import signal
import subprocess
import sys
import time

import torch


class CleanStop(RuntimeError):
    """A resource limit or signal requests a checkpoint and exit."""


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


class MemoryGuard:
    """Check total process memory and device allocation at bounded intervals."""

    def __init__(self, limit_gb, device, *, interval=15, pressure=free_percent,
                 resident=resident_bytes, clock=time.monotonic):
        if not 0 < limit_gb <= 24:
            raise ValueError("memory limit must be positive and at most 24 GB")
        self.limit = int(limit_gb * 1_000_000_000)
        self.device = torch.device(device)
        self.interval, self.pressure, self.resident, self.clock = interval, pressure, resident, clock
        self.last = -float("inf")
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
        driver = torch.mps.driver_allocated_memory() if self.device.type == "mps" else 0
        rss = self.resident()
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            peak_rss *= 1024
        # Unified allocations can also appear in RSS. A sum would count them twice.
        self.peak = max(self.peak, rss, driver, peak_rss)
        free = self.pressure()
        self.low = self.low + 1 if free < 20 else 0
        print(json.dumps({"stage": stage, "rss_bytes": rss, "peak_memory_bytes": self.peak,
                          "driver_bytes": driver, "free_percent": free}), flush=True)
        if max(rss, driver) > self.limit:
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
