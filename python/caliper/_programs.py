"""The facade of GPU programs in the script's process (``design/gpu-programs.md``).

A program is WGSL compute that reads tensors of the script in place and writes one output. The
wire path of Caliper LIB calls :func:`install`, :func:`remove` and :func:`status` when a PROGRAM
message arrives, and sends the status items back in PROGRAM_STATE. ``caliper.step()`` calls
:func:`at_step` after the framework's work of the step is complete. The facade loads the native
module ``caliper._producer`` on the first :func:`install` only, so that a script that only
streams tensors loads nothing new.

The facade keeps the cadence (``design/gpu-programs.md``, Cadence): it measures the step time
``T`` and the added time ``C`` of a pass step, and runs the programs every
``N = max(ceil(C / (0.04 * T)), every, 1)`` steps. It gives each geometry output to the publish
path of ``caliper.scene`` and each pixels output to ``caliper.show`` as the live view
``program.<id>``. A refused or stopped program is an error of its panel, never of the script:
no function here raises for it.

The functions raise ``NotImplementedError`` until the producer runtime is built.
"""
from __future__ import annotations

__all__ = ["install", "remove", "status", "at_step", "loaded"]


def install(program_id: str, declaration: dict, wgsl: str, *, scene: str | None = None,
            sha256: str = "") -> dict:
    """Install a program, or replace the program with the same id.

    Args:
        program_id: The program's id.
        declaration: The program declaration of the PROGRAM message, without ``wgsl``
            (``inputs``, ``output``, ``limits``, ``entry``, ``workgroups``).
        wgsl: The WGSL source.
        scene: The scene name of a geometry output's panel, or ``None`` for a pixels output.
        sha256: The SHA-256 of ``wgsl`` that the app sent, in hex.

    Returns:
        dict: The program's status item (the keys of ``program_status`` without ``host``), with
        the state ``compiled`` or ``refused`` and, for a refusal, the ``error``.
    """
    raise NotImplementedError("the producer runtime is not built yet (design/gpu-programs.md)")


def remove(program_id: str) -> None:
    """Stop and remove a program, and clear its output; an unknown id does nothing."""
    raise NotImplementedError("the producer runtime is not built yet (design/gpu-programs.md)")


def status() -> list:
    """Return one status item for each installed program, in the order of installation."""
    raise NotImplementedError("the producer runtime is not built yet (design/gpu-programs.md)")


def at_step(step: int) -> None:
    """Run the programs that are due at ``step`` and publish their outputs.

    ``caliper.step()`` calls it on the training thread after the framework's work of the step is
    complete. Without an installed program it returns at once and touches no device.
    """
    raise NotImplementedError("the producer runtime is not built yet (design/gpu-programs.md)")


def loaded() -> bool:
    """Whether the native module ``caliper._producer`` is loaded in this process."""
    return False
