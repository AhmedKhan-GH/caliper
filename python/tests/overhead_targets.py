"""The overhead targets of the acceptance tests and of the timing tests that measure this machine.

An overhead share and a throughput are targets, not gates. A test prints each measured value with its
target in one line, such as ``overhead share 0.0187 (target 0.02): tower_arcs``, and a line that starts
with ``warning:`` when the value misses its target. It fails only on a clear regression, an overhead of
more than twice the target: a share above twice its target, or a throughput below 90 % for the target of
95 %. Each function returns the text of that failure, or None.

The acceptance drivers under tests/ load this file by its path; the tests of this folder import it.
"""
from __future__ import annotations

OVERHEAD_SHARE_TARGET = 0.02     # the share of the step time while views are attached
THROUGHPUT_TARGET = 0.95         # the speed of a watched run against the same run without Caliper


def overhead_share(name: str, share, target: float = OVERHEAD_SHARE_TARGET, out=print):
    """Prints the overhead share of ``name`` with its target; the failure text when the share is
    missing or above twice the target, else None."""
    if not isinstance(share, (int, float)) or share < 0:
        return f"{name} has no measured overhead share: {share!r}"
    out(f"overhead share {share:.4f} (target {target:g}): {name}")
    if share > target:
        out(f"warning: the overhead share {share:.4f} of {name} misses its target {target:g}")
    if share > 2 * target:
        return f"the overhead share {share:.4f} of {name} is above {2 * target:g}, twice its target"
    return None


def throughput(name: str, speed: float, target: float = THROUGHPUT_TARGET, out=print):
    """Prints the throughput of ``name``, the speed with Caliper over the speed without it, with its
    target; the failure text when the overhead is more than twice the target's, else None."""
    floor = 1 - 2 * (1 - target)
    out(f"throughput {100 * speed:.1f} % (target {100 * target:g} %): {name}")
    if speed < target:
        out(f"warning: the throughput {100 * speed:.1f} % of {name} misses its target {100 * target:g} %")
    if speed < floor:
        return f"the throughput {100 * speed:.1f} % of {name} is below {100 * floor:g} %, twice the target's overhead"
    return None
