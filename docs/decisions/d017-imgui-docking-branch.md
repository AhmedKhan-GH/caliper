# D17. ImGui docking branch, epoch stays 2

| Date | Status |
|---|---|
| 2026-07-02 | Accepted |

## Decision

The UI-stack pin switches to ImGui's docking branch, on the same 1.92 line. The epoch stays 2 because no SDK release or outside consumer existed yet. The host enables docking and a viewport dockspace with a default tiled layout, and applet windows dock without code changes. Layouts persist in `imgui.ini`.

## Reason

Floating windows look like a demo, and docking is the ImGui-native way to get a desktop layout. Changing the pin before the first release is free; afterwards it costs an epoch.

## Notes

This amends the pin in [D4](d004-ui-stack-pin-defines-the-epoch.md). From the first release on, any pin change is a formal epoch bump. The code enables `ImGuiConfigFlags_DockingEnable` in the host's core lifecycle.
