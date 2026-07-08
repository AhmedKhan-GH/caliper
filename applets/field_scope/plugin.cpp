#include <caliper/caliper.hpp>
#include "field_scope.h"

// Epoch-2 entry (PLATFORM.md §6a) — the established adapter shape: all logic
// lives in FieldScopeApplet, this file is only the ABI bridge. id/version are
// byte-identical to field_scope.caliper.toml (the loader rejects disagreement).
class FieldScopePlugin final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override { return impl_.initialize(host); }
    void on_frame(const caliper::Frame&) override { impl_.draw_ui(); }
    void on_cleanup() override { impl_.cleanup(); }

private:
    fieldscope::FieldScopeApplet impl_;
};

CALIPER_APPLET(FieldScopePlugin,
    .id       = "dev.caliper.field-scope",
    .version  = "0.1.0",
    .name     = "FieldScope",
    .summary  = "A self-consistent electrostatic Particle-In-Cell plasma, drawn "
                "with zero copies: the torch simulation writes exportable-pool "
                "tensors and the renderer's instanced point pass "
                "(caliper.geometry.v1) reads the SAME memory in place, per "
                "frame. Each step the particles deposit charge onto a grid, an "
                "FFT Poisson solve produces their own electric field, and a "
                "Boris pusher advances them under that self-field plus an "
                "optional background B — so real collective behaviour (the "
                "two-stream instability, plasma oscillations) emerges rather "
                "than being scripted. Choose the initial condition; left-drag "
                "perturbs the plasma, right-drag orbits, wheel zooms. Falls back "
                "to a subsampled CPU scatter wherever zero-copy is absent.",
    .tag      = "ML",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_JOBS_V1,
                 CALIPER_DEVICE_V1})
