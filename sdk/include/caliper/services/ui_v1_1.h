#pragma once
/* caliper.ui.v1_1 - additive revision of caliper.ui.v1 (docs/applets/services/caliper-ui-v1-1.md). The table
 * opens with a byte-identical copy of CaliperUiV1 and appends one entry, dock_hint(). The host keeps serving
 * caliper.ui.v1 unchanged. An applet that wants hints probes this id at run time and goes on without them on a
 * host that lacks it. The service is frozen once published. */
#include <stdint.h>
#include <stddef.h>
#include <caliper/services/ui_v1.h>

/** The id of this service, which `get_service` takes. */
#define CALIPER_UI_V1_1 "caliper.ui.v1_1"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The place where a window docks the first time it appears. The values are fixed integers, and the host treats a
 * value it does not know as `CALIPER_DOCK_CENTRAL`.
 */
typedef enum CaliperDockRole {
    /** The central area, for the main view of the applet. */
    CALIPER_DOCK_CENTRAL     = 0,
    /** The upper part of the side column, for controls. */
    CALIPER_DOCK_SIDE_TOP    = 1,
    /** The lower part of the side column, for details. */
    CALIPER_DOCK_SIDE_BOTTOM = 2
} CaliperDockRole;

/** The function table of `caliper.ui.v1_1`. Call every entry from the UI thread. */
typedef struct CaliperUiV1_1 {
    /** `sizeof(CaliperUiV1_1)` as the host built it. */
    uint32_t struct_size;
    /* --- v1-identical prefix (byte-for-byte CaliperUiV1). --- */
    /** Return the shared `ImGuiContext` of the host. The pointer stays valid from `initialize()` through `cleanup()`. */
    struct ImGuiContext*    (*imgui_context)(void);
    /** Return the shared `ImPlotContext` of the host. The pointer stays valid from `initialize()` through `cleanup()`. */
    struct ImPlotContext*   (*implot_context)(void);
    /** Return the shared `ImPlot3DContext` of the host. The pointer stays valid from `initialize()` through `cleanup()`. */
    struct ImPlot3DContext* (*implot3d_context)(void);
    /**
     * Write the allocator pair of the host to `out_alloc`, `out_free`, and `out_user_data`. The applet must install
     * the pair into its copy of the ImGui globals before it makes any allocation, so that every allocation lands on
     * the host heap.
     */
    void (*imgui_allocators)(CaliperImGuiAllocFn* out_alloc,
                             CaliperImGuiFreeFn*  out_free,
                             void** out_user_data);

    /* --- v1_1 addition. --- */
    /**
     * Declare the dock role of the window whose `Begin()` title is `window_title`. The argument `role` is a
     * `CaliperDockRole` value. Call it on the UI thread next to that `Begin()`, before or after it, and a call in
     * every frame is cheap and changes nothing after the first call for a title. The hint places only a window that
     * has no saved layout, so a layout that the user arranged always wins. A window without a hint docks in the
     * center.
     */
    void (*dock_hint)(const char* window_title, uint32_t role);
} CaliperUiV1_1;

#ifdef __cplusplus
}
static_assert(sizeof(CaliperUiV1_1) == sizeof(CaliperUiV1) + sizeof(void*),
              "CaliperUiV1_1 appends exactly one entry to the v1 table");
static_assert(offsetof(CaliperUiV1_1, imgui_allocators) ==
              offsetof(CaliperUiV1, imgui_allocators),
              "v1_1 prefix must match v1 slot-for-slot");
static_assert(offsetof(CaliperUiV1_1, dock_hint) == sizeof(CaliperUiV1),
              "the v1_1 entry must follow the frozen v1 table");
#endif
