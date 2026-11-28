#pragma once
/* caliper.ui.v1 - the ImGui, ImPlot, and ImPlot3D contexts and the host allocators. The allocator handoff makes
 * context sharing across the DLL boundary sound, as the DLL guidance of Dear ImGui requires. The service is
 * frozen once published. The function-pointer types match ImGuiMemAllocFunc and ImGuiMemFreeFunc exactly,
 * without pulling imgui.h into the C ABI. */
#include <stdint.h>
#include <stddef.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_UI_V1 "caliper.ui.v1"

#ifdef __cplusplus
extern "C" {
#endif

struct ImGuiContext;
struct ImPlotContext;
struct ImPlot3DContext;

/** The type of an allocation function. It takes a size in bytes and the `user_data` pointer and returns the new memory. */
typedef void* (*CaliperImGuiAllocFn)(size_t sz, void* user_data);
/** The type of a free function. It takes a pointer from the matching allocation function and the `user_data` pointer. */
typedef void  (*CaliperImGuiFreeFn)(void* ptr, void* user_data);

/** The function table of `caliper.ui.v1`. Call every entry from the UI thread. */
typedef struct CaliperUiV1 {
    /** `sizeof(CaliperUiV1)` as the host built it. */
    uint32_t struct_size;
    /** Return the shared `ImGuiContext` of the host. The pointer stays valid from `initialize()` through `cleanup()`. */
    struct ImGuiContext*    (*imgui_context)(void);
    /** Return the shared `ImPlotContext` of the host. The pointer stays valid from `initialize()` through `cleanup()`. */
    struct ImPlotContext*   (*implot_context)(void);
    /** Return the shared `ImPlot3DContext` of the host. The pointer stays valid from `initialize()` through `cleanup()`. */
    struct ImPlot3DContext* (*implot3d_context)(void);
    /**
     * Write the allocator pair of the host to `out_alloc`, `out_free`, and `out_user_data`. The instrument must install
     * the pair into its copy of the ImGui globals before it makes any allocation, so that every allocation lands on
     * the host heap.
     */
    void (*imgui_allocators)(CaliperImGuiAllocFn* out_alloc,
                             CaliperImGuiFreeFn*  out_free,
                             void** out_user_data);
} CaliperUiV1;

#ifdef __cplusplus
}
#endif
