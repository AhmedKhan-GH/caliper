#pragma once

#define CALIPER_APPLET_ABI 1

struct ImGuiContext;
struct ImPlotContext;
struct ImPlot3DContext;

struct CaliperHostContext {
    ImGuiContext*    imgui;
    ImPlotContext*   implot;
    ImPlot3DContext* implot3d;
    const char*      data_dir;
};

struct CaliperAppletInfo {
    const char* name;
    const char* version;
    const char* description;
    const char* tag;
    int         abi;
};

#ifdef __cplusplus
extern "C" {
#endif

typedef CaliperAppletInfo (*PFN_applet_info)(void);
typedef void* (*PFN_applet_create)(void);
typedef void  (*PFN_applet_destroy)(void* ctx);
typedef bool  (*PFN_applet_initialize)(void* ctx, const CaliperHostContext* host);
typedef void  (*PFN_applet_draw_ui)(void* ctx, int w, int h);
typedef void  (*PFN_applet_cleanup)(void* ctx);

#ifdef CALIPER_APPLET_EXPORT
  #ifdef _WIN32
    #define APPLET_API __declspec(dllexport)
  #else
    #define APPLET_API __attribute__((visibility("default")))
  #endif
#else
  #define APPLET_API
#endif

#ifdef __cplusplus
}
#endif
