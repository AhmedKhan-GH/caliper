# caliper.ui.v1

Service id `caliper.ui.v1` gives an applet the ImGui, ImPlot and ImPlot3D contexts and the host allocator handoff.

## Calls

<!-- c-api calls sdk/include/caliper/services/ui_v1.h -->

## Types

<!-- c-api types sdk/include/caliper/services/ui_v1.h level=3 -->

## Ids

<!-- c-api ids sdk/include/caliper/services/ui_v1.h -->

## Semantics

- **Context lifetime.** The host owns the three context pointers (`imgui_context`, `implot_context` and `implot3d_context`). They stay valid from `initialize()` through `cleanup()`. They never change while the applet runs, so an applet may cache them.
- **Allocator handoff is mandatory.** An applet links its own copy of Dear ImGui, and that copy has its own allocator globals. Before the applet makes any ImGui, ImPlot or ImPlot3D allocation, it MUST call `imgui_allocators(...)`. It then installs the returned `(alloc, free, user_data)` triple into its copy with `ImGui::SetAllocatorFunctions` and sets the current context to the host context. If the applet skips this, it allocates on its own heap while the host frees on another heap. This is the cross-DLL heap mismatch. The SDK function `ui::connect` does the whole handoff during `initialize()`. A hand-written C applet must do the handoff itself.
- **UI thread only.** Every function in this header and every ImGui call on the shared context must run on the host UI thread. This is the same thread that calls the `frame()` function of the applet. The context has no locking, and a call from a worker thread is undefined behavior. Put long or blocking work in background jobs.
