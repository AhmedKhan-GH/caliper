#pragma once

#include "applet_api.h"
#include <string>
#include <vector>

struct LoadedApplet {
    std::string path;
    void*       handle = nullptr;

    PFN_applet_info       fn_info       = nullptr;
    PFN_applet_create     fn_create     = nullptr;
    PFN_applet_destroy    fn_destroy    = nullptr;
    PFN_applet_initialize fn_initialize = nullptr;
    PFN_applet_draw_ui    fn_draw_ui    = nullptr;
    PFN_applet_cleanup    fn_cleanup    = nullptr;

    CaliperAppletInfo info{};
    void*             instance = nullptr;
    bool              initialized = false;
};

class AppletHost {
public:
    void scan(const std::string& dir);
    void close_all();

    int count() const { return (int)applets_.size(); }
    const LoadedApplet& operator[](int i) const { return applets_[i]; }
    LoadedApplet&       operator[](int i)       { return applets_[i]; }

    bool launch(int idx, const CaliperHostContext& host);
    void draw(int idx, int w, int h);
    void teardown(int idx);

private:
    std::vector<LoadedApplet> applets_;
};
