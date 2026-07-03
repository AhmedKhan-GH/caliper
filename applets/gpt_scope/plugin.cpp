#include <caliper/caliper.hpp>
#include "gpt_model.h"

// Epoch-2 entry (PLATFORM.md §6a). All applet logic lives in GPTScopeApplet;
// this file is only the ABI bridge — the established adapter shape. ui::connect
// is handled by the macro. The id/version below are byte-identical to
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
    .version  = "0.2.0",
    .name     = "GPTScope",
    .summary  = "Mechanistic interpretability for a live-training char GPT "
                "(4L/4H/128d, TinyShakespeare): the logit lens across depth, "
                "head roles as a distance/entropy scatter with on-demand "
                "attention drill-down, W_E embedding geometry as a glyph cloud, "
                "attn-vs-MLP residual write accounting with per-group gradient "
                "norms, and confidence-colored live sampling. Trains off the "
                "frame thread via caliper.jobs.v1, device-negotiated; metrics.v1 "
                "streams the run; tensor_bridge.v1 draws the head heatmap; "
                "artifacts.v1 saves/loads the model.",
    .tag      = "ML",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_JOBS_V1,
                 CALIPER_DEVICE_V1})
