# caliper.ui.v1_1

Service id `caliper.ui.v1_1` is an additive revision of [`caliper.ui.v1`](service-ui-v1.md). Its table opens with a byte-identical copy of `CaliperUiV1` and appends one entry, `dock_hint`, through which an applet tells the host where each of its windows belongs in the docked layout. This page embeds the header verbatim; the docs build fails if the file moves.

```c
--8<-- "sdk/include/caliper/services/ui_v1_1.h"
```

## Semantics

- **The v1 entries are unchanged.** The context getters and `imgui_allocators` behave exactly as in [`caliper.ui.v1`](service-ui-v1.md), and the host keeps serving `caliper.ui.v1` unchanged beside this id.
- **Roles, not positions.** `dock_hint(window_title, role)` declares the role of the window whose `Begin()` title is `window_title`. The host lays out roles and never names a window. `CALIPER_DOCK_CENTRAL` docks the window in the central area, `CALIPER_DOCK_SIDE_TOP` in the upper part of a side column and `CALIPER_DOCK_SIDE_BOTTOM` in its lower part. Windows with the same role share a node as tabs. A role value the host does not know is treated as `CALIPER_DOCK_CENTRAL`.
- **Beside `Begin()`, every frame if that is simplest.** Call it on the UI thread next to the window's `Begin()`. A hint given before the window's first `Begin()` docks the window in the frame it first appears; a hint given after it takes effect on the next frame. The first hint for a title counts, and every later call costs one hash lookup and changes nothing.
- **The user's layout wins.** A hint places only a window that has no saved layout, that is, no entry in `imgui.ini`. The host places each window once per process, the first time it appears, so a window the user moves or undocks afterwards stays where the user put it.
- **No hint means the centre.** A window that the applet begins without a hint, and that is not a child window, a popup, a tooltip, or a window with `ImGuiWindowFlags_NoDocking` or `ImGuiWindowFlags_NoSavedSettings`, docks in the centre the first time it appears.
- **Optional for the applet.** An applet that uses hints still lists only `caliper.ui.v1` in `[services].required`. The sugar `caliper::ui::dock_hint(host, title, role)` probes `caliper.ui.v1_1` and does nothing on a host that lacks it, where the windows keep the placement ImGui gives them.
- **Where hints apply.** The `caliper` executable applies hints on its applet page. An embedding host that has no docked layout, such as the embed core, accepts the calls and ignores them.

## Example

```cpp
caliper::ui::dock_hint(host, "Viewer: Scene", CALIPER_DOCK_CENTRAL);
ImGui::Begin("Viewer: Scene");
// ...
ImGui::End();

caliper::ui::dock_hint(host, "Viewer: Controls", CALIPER_DOCK_SIDE_TOP);
ImGui::Begin("Viewer: Controls");
// ...
ImGui::End();
```

The testing fixture (`caliper/fixture_host.h`) serves this id and records every hint, so an applet test can check the role of each window with `FixtureHost::dock_role(title)`.
