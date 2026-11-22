# caliper.ui.v1_1

Service id `caliper.ui.v1_1` is an additive revision of [`caliper.ui.v1`](caliper-ui-v1.md). Its table opens with a byte-identical copy of `CaliperUiV1` and adds one entry, `dock_hint`. With `dock_hint`, an applet tells the host where each of its windows belongs in the docked layout.

## Calls

<!-- c-api calls sdk/include/caliper/services/ui_v1_1.h -->

## Types

<!-- c-api types sdk/include/caliper/services/ui_v1_1.h level=3 -->

## Ids

<!-- c-api ids sdk/include/caliper/services/ui_v1_1.h -->

## Semantics

- **The v1 entries are unchanged.** The context getters and `imgui_allocators` behave as in [`caliper.ui.v1`](caliper-ui-v1.md). The host also keeps serving `caliper.ui.v1` beside this id.
- **Roles, not positions.** The host lays out roles and never names a window.
- **Shared nodes.** Windows with the same role share a node as tabs, also after a restart. A new side window joins the node where a window of its role is docked already, such as one restored from `imgui.ini`. The host splits a side column off the central area only when no such node exists. The host treats a role value it does not know as `CALIPER_DOCK_CENTRAL`.
- **Call timing.** Call `dock_hint` on the UI thread next to the window's `Begin()`. It is safe to call it every frame. A hint given before the first `Begin()` docks the window in the frame where it first appears. A hint given after it takes effect on the next frame. The first hint for a title counts, and every later call changes nothing.
- **The user's layout wins.** A hint places only a window that has no saved layout, that is, no entry in `imgui.ini`. The host places each window once per process, the first time it appears. Thus a window that the user moves or undocks stays where the user put it. A hinted window with `ImGuiWindowFlags_NoSavedSettings` is docked too, but the host writes nothing about it to `imgui.ini`.
- **Windows without a hint.** The host docks an ordinary window in the center the first time it appears. An ordinary window is not a child window, a popup, a tooltip or a modal. It also has none of `ImGuiWindowFlags_NoDocking`, `NoSavedSettings`, `NoMove`, `NoResize`, `NoTitleBar` and `AlwaysAutoResize`. Any other window, such as a tool window that sizes itself, keeps the placement ImGui gives it. To keep an ordinary window floating, such as a non-modal file dialog, give it `ImGuiWindowFlags_NoDocking`.
- **Optional for the applet.** An applet that uses hints lists only `caliper.ui.v1` in `[services].required`. The sugar `caliper::ui::dock_hint(host, title, role)` probes `caliper.ui.v1_1`. On a host that lacks it, the call does nothing and the windows keep the placement ImGui gives them.
- **Where hints apply.** The `caliper` executable applies hints on its applet page. An embedding host that has no docked layout, such as Caliper Embed, accepts the calls and ignores them.

## Worked Example

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

The testing fixture (`caliper/fixture_host.h`) serves this id and records every hint. In an applet test, `FixtureHost::dock_role(title)` returns the first role hinted for a title, as the host keeps it.
