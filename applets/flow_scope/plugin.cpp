#include <caliper/caliper.hpp>
#include "flow_scope.h"

// Epoch-2 entry (PLATFORM.md §6a) — the established adapter shape: all logic
// lives in FlowScopeApplet, this file is only the ABI bridge. id/version are
// byte-identical to flow_scope.caliper.toml (the loader rejects disagreement).
class FlowScopePlugin final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override { return impl_.initialize(host); }
    void on_frame(const caliper::Frame&) override { impl_.draw_ui(); }
    void on_cleanup() override { impl_.cleanup(); }

private:
    flowscope::FlowScopeApplet impl_;
};

CALIPER_APPLET(FlowScopePlugin,
    .id       = "dev.caliper.flow-scope",
    .version  = "0.1.0",
    .name     = "FlowScope",
    .summary  = "A million-particle flow field, drawn with zero copies: the "
                "torch CUDA simulation writes exportable-pool tensors and the "
                "renderer's instanced point pass (caliper.geometry.v1) reads "
                "the SAME memory in place, per frame. Left-drag pushes the "
                "field with a cursor-ray impulse; right-drag orbits. Falls "
                "back to a subsampled CPU scatter wherever the zero-copy path "
                "is absent.",
    .tag      = "ML",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_JOBS_V1,
                 CALIPER_DEVICE_V1})
