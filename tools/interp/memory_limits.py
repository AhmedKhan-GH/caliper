"""Compatibility imports for the shared Caliper process memory guard."""
from caliper.interp._memory import (
    CleanStop, MemoryGuard, StopControl, autorelease_scope, free_percent, peak_resident_bytes,
    physical_footprint, resident_bytes,
)

__all__ = ["CleanStop", "MemoryGuard", "StopControl", "autorelease_scope", "free_percent",
           "peak_resident_bytes", "physical_footprint", "resident_bytes"]
