# caliper.ui.v1

Service id `caliper.ui.v1` — the ImGui/ImPlot/ImPlot3D contexts and host allocator handoff (PLATFORM.md §6d); fuller semantics arrive at Task 13. This page embeds the header verbatim; the docs build fails if the file moves.

```c
--8<-- "sdk/include/caliper/services/ui_v1.h"
```

## Semantics

- **Context lifetime.** The three context pointers (`imgui_context`,
  `implot_context`, `implot3d_context`) are owned by the host and stay valid for
  the whole lifetime of the applet — from `initialize()` through `cleanup()`.
  They never change under a running applet, so an applet may cache them.
- **Allocator handoff is mandatory.** An applet links its *own* copy of Dear
  ImGui, which has its own allocator globals. Before the applet makes **any**
  ImGui/ImPlot/ImPlot3D allocation it MUST call `imgui_allocators(...)` and
  install the returned `(alloc, free, user_data)` triple into its copy
  (`ImGui::SetAllocatorFunctions`) *and* set the current context to the host's.
  Skipping this means the applet allocates on its own heap while the host frees
  on another — the classic cross-DLL heap mismatch. The SDK's `ui::connect`
  (caliper.hpp sugar) performs the whole handoff for you during `initialize()`;
  hand-written C applets must do it themselves.
- **UI thread only.** Every function in this table, and every ImGui call made
  through the shared context, must run on the host's UI thread — the same thread
  that calls the applet's `frame()`. There is no locking; touching the shared
  context from a worker thread is undefined behavior. Long or blocking work
  belongs in background jobs, not on this thread.
