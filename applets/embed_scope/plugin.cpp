#include <caliper/caliper.hpp>
#include "embed_model.h"

// Epoch-2 entry (PLATFORM.md §6a). All applet logic lives in EmbedScopeApplet;
// this file is only the ABI bridge — the established adapter shape. ui::connect
// is handled by the macro. The id/version below are byte-identical to
// embed_scope.caliper.toml (the loader rejects any disagreement).
class EmbedScopePlugin final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override { return impl_.initialize(host); }
    void on_frame(const caliper::Frame&) override { impl_.draw_ui(); }
    void on_cleanup() override { impl_.cleanup(); }

private:
    embedscope::EmbedScopeApplet impl_;
};

CALIPER_APPLET(EmbedScopePlugin,
    .id       = "dev.caliper.embed-scope",
    .version  = "0.1.0",
    .name     = "EmbedScope",
    .summary  = "3D embedding projector: a small MNIST net with a learned 3-D "
                "bottleneck whose test-set embeddings are drawn as a live "
                "ImPlot3D scatter (10 classes) that splits blob->lobes while "
                "training runs off the frame thread via caliper.jobs.v1. Hover a "
                "point for its digit via tensor_bridge.v1; Save/Load the model "
                "via artifacts.v1 (Load skips training); class centroids and "
                "misclassified counts via SQL over caliper.data.v1.",
    .tag      = "ML",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_JOBS_V1,
                 CALIPER_DEVICE_V1})
