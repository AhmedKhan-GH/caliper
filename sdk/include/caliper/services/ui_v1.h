#pragma once
/* caliper.ui.v1 — ImGui/ImPlot/ImPlot3D contexts + allocators (§6d).
 * The allocator handoff is what makes context-sharing across the DLL
 * boundary sound (Dear ImGui's own DLL guidance). IMMUTABLE once published.
 * Function-pointer typedefs mirror ImGuiMemAllocFunc/ImGuiMemFreeFunc
 * layout-exactly, without pulling imgui.h into the C ABI (§6c). */
#include <stdint.h>
#include <stddef.h>

#define CALIPER_UI_V1 "caliper.ui.v1"

#ifdef __cplusplus
extern "C" {
#endif

struct ImGuiContext;
struct ImPlotContext;
struct ImPlot3DContext;

typedef void* (*CaliperImGuiAllocFn)(size_t sz, void* user_data);
typedef void  (*CaliperImGuiFreeFn)(void* ptr, void* user_data);

typedef struct CaliperUiV1 {
    uint32_t struct_size;
    struct ImGuiContext*    (*imgui_context)(void);
    struct ImPlotContext*   (*implot_context)(void);
    struct ImPlot3DContext* (*implot3d_context)(void);
    /* Host's allocator pair — the applet side MUST install these into its
     * copy of ImGui's globals so every allocation lands on the host heap. */
    void (*imgui_allocators)(CaliperImGuiAllocFn* out_alloc,
                             CaliperImGuiFreeFn*  out_free,
                             void** out_user_data);
} CaliperUiV1;

#ifdef __cplusplus
}
#endif
