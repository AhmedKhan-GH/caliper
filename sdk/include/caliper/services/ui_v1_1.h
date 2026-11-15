#pragma once
/* caliper.ui.v1_1 — additive revision of caliper.ui.v1 (docs/references/
 * service-ui-v1-1.md). The table opens with a byte-identical copy of
 * CaliperUiV1 and appends one entry, dock_hint(), which tells the host where a
 * window belongs in its docked layout. The host keeps serving caliper.ui.v1
 * unchanged; an applet that wants hints probes this id at run time and goes on
 * without them on a host that lacks it (caliper::ui::dock_hint in the sugar).
 * IMMUTABLE once published. */
#include <stdint.h>
#include <stddef.h>
#include <caliper/services/ui_v1.h>

#define CALIPER_UI_V1_1 "caliper.ui.v1_1"

#ifdef __cplusplus
extern "C" {
#endif

/* Where a window docks the first time it appears. Fixed integers: a value the
 * host does not know is treated as CALIPER_DOCK_CENTRAL. */
typedef enum CaliperDockRole {
    CALIPER_DOCK_CENTRAL     = 0,  /* the central area: the applet's main view */
    CALIPER_DOCK_SIDE_TOP    = 1,  /* the side column, upper part: controls */
    CALIPER_DOCK_SIDE_BOTTOM = 2   /* the side column, lower part: details */
} CaliperDockRole;

typedef struct CaliperUiV1_1 {
    uint32_t struct_size;
    /* --- v1-identical prefix (byte-for-byte CaliperUiV1). --- */
    struct ImGuiContext*    (*imgui_context)(void);
    struct ImPlotContext*   (*implot_context)(void);
    struct ImPlot3DContext* (*implot3d_context)(void);
    void (*imgui_allocators)(CaliperImGuiAllocFn* out_alloc,
                             CaliperImGuiFreeFn*  out_free,
                             void** out_user_data);

    /* --- v1_1 addition. --- */
    /* Declares the dock role (a CaliperDockRole value) of the window whose
     * Begin() title is `window_title`. Call it on the UI thread beside that
     * Begin(), before it or after it; calling it every frame is cheap and
     * changes nothing after the first call for a title. The hint only places a
     * window that has no saved layout, so a layout the user arranged always
     * wins. A window with no hint docks in the centre. */
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
