# D7. Dynamic CRT on Windows, ImGui allocators handed over

| Date | Status |
|---|---|
| 2026-06-10 | Accepted |

## Decision

On Windows the host and applets link the C runtime dynamically (`/MD`), and `ui.v1` hands the host's ImGui allocators to each applet.

## Reason

A static runtime gives the host and each applet DLL separate heaps, so memory allocated in one and freed in the other crashes. Sharing one runtime and one ImGui allocator removes that class of bug.

## Notes

The code implements this decision: the dynamic runtime is the default in `CMakeLists.txt`, with the option `CALIPER_STATIC_CRT` as an escape hatch, and the allocator handoff is in `ui_v1.h`.
