#include <caliper/caliper.hpp>
#include "gpt_model.h"

// Epoch-2 entry (PLATFORM.md §6a). All applet logic lives in GPTScopeApplet;
// this file is only the bridge — the established adapter shape. ui::connect is
// handled by the macro. The id/version below are byte-identical to
// gpt_scope.caliper.toml (the loader rejects any disagreement).
class GPTScopePlugin final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override { return impl_.initialize(host); }
    void on_frame(const caliper::Frame&) override { impl_.draw_ui(); }
    void on_cleanup() override { impl_.cleanup(); }

private:
    gptscope::GPTScopeApplet impl_;
};

CALIPER_APPLET(GPTScopePlugin,
    .id       = "dev.caliper.gpt-scope",
    .version  = "0.1.0",
    .name     = "GPTScope",
    .summary  = "Flagship: a nanoGPT-style char-level transformer trained on "
                "TinyShakespeare off the frame thread via caliper.jobs.v1, "
                "device-negotiated, with live train/val loss and a live text "
                "sample. metrics.v1 streams the run; tensor_bridge.v1 is probed "
                "for the E2 attention panel.",
    .tag      = "LLM",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_JOBS_V1,
                 CALIPER_DEVICE_V1})
