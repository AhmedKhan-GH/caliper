#include <caliper/caliper.hpp>
#include "sculpt_scope.h"

// Epoch-2 entry (PLATFORM.md §6a): all logic lives in SculptScopeApplet; this
// file is only the ABI bridge. id/version are byte-identical to the manifest
// (sculpt_scope.caliper.toml) — the loader rejects disagreement.
class SculptScopePlugin final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override { return impl_.initialize(host); }
    void on_frame(const caliper::Frame&) override { impl_.draw_ui(); }
    void on_cleanup() override { impl_.cleanup(); }

private:
    sculptscope::SculptScopeApplet impl_;
};

CALIPER_APPLET(SculptScopePlugin,
    .id       = "dev.caliper.sculpt-scope",
    .version  = "0.1.0",
    .name     = "SculptScope",
    .summary  = "A small libtorch MLP maps N latent codes to N 3-D points and "
                "trains live (Adam, real backprop) to match a target shape. Its "
                "forward output is written straight into the pool-born tensor "
                "the renderer's instanced point pass (caliper.geometry.v1) reads "
                "in place — the buffer the final Linear layer writes IS the "
                "buffer the GPU draws, with zero copies between the ML and the "
                "picture. Watch a formless blob flow into a crisp shape as the "
                "network learns; color tracks per-point motion so a converged "
                "cloud dims. Right-drag orbits, wheel zooms. Falls back to a "
                "subsampled CPU scatter wherever the zero-copy path is absent.",
    .tag      = "ML",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_JOBS_V1,
                 CALIPER_DEVICE_V1})
