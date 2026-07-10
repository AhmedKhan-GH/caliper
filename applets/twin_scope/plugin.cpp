#include <caliper/caliper.hpp>
#include "twin_scope.h"

class TwinScopePlugin final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override { return applet_.initialize(host); }
    void on_frame(const caliper::Frame&) override { applet_.draw_ui(); }
    void on_cleanup() override { applet_.cleanup(); }

private:
    twinscope::TwinScopeApplet applet_;
};

CALIPER_APPLET(TwinScopePlugin,
    .id       = "dev.caliper.twin-scope",
    .version  = "0.2.0",
    .name     = "TwinScope",
    .summary  = "A surface-aware synthetic heat twin on a UV-mapped finned "
                "housing: a batched cotangent-Laplacian sim flows heat over the "
                "3-D surface while a small net learns it live. The hero splits "
                "physics|belief side by side, drawn zero-copy from imported "
                "device tensors at texture resolution; a live textured (v1.2) vs "
                "per-vertex (v1.1) toggle exposes the R2 resolution gap. Falls "
                "back honestly to per-vertex color or a CPU heatmap wherever the "
                "zero-copy geometry path is absent.",
    .tag      = "ML",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_JOBS_V1,
                 CALIPER_DEVICE_V1, CALIPER_TENSOR_BRIDGE_V1,
                 CALIPER_TENSOR_BRIDGE_V1_2, CALIPER_GEOMETRY_V1_1,
                 CALIPER_GEOMETRY_V1_2})
