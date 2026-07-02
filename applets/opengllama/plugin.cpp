#include <caliper/caliper.hpp>
#include "opengllama.h"

// Bridge-native since Phase 2D: heatmaps upload through caliper.tensor_bridge.v1
// (required), so this applet issues no raw GL — the §6c grandfather has expired.
class OpenGllamaPlugin final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override {
        return impl_.initialize(caliper::Bridge(host));
    }
    void on_frame(const caliper::Frame& f) override {
        impl_.draw_ui(f.fb_width, f.fb_height);
    }
    void on_cleanup() override { impl_.cleanup(); }

private:
    OpenGllamaApplet impl_;
};

CALIPER_APPLET(OpenGllamaPlugin,
    .id       = "dev.ahmed.opengllama",
    .version  = "0.1.0",
    .name     = "OpenGllama",
    .summary  = "Load GGUF models via llama.cpp and visualize layer activations "
                "with OpenGL-rendered heatmaps on Metal/CUDA backends.",
    .tag      = "LLM",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_TENSOR_BRIDGE_V1})
