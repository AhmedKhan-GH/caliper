// MyScope — the "smallest complete applet" from the Development Basics page.
// docs/wiki/tutorials/development-basics.md embeds this file VERBATIM
// (pymdownx.snippets), so the tutorial and this source cannot drift.
#include <caliper/caliper.hpp>
#include <cmath>

class MyScope final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override {
        host_ = &host;
        host.log_info("my_scope: on_init");
        return true;                       // false aborts loading
    }
    void on_frame(const caliper::Frame& f) override {
        ImGui::Begin("MyScope");           // host dockspace places it
        ImGui::Text("%d x %d px, t = %.1fs",
                    f.fb_width, f.fb_height, f.time_sec);
        if (ImPlot::BeginPlot("wave", {-1, 220})) {
            static float xs[256], ys[256];
            for (int i = 0; i < 256; i++) {
                xs[i] = i / 255.0f * 6.28318f;
                ys[i] = std::sin(xs[i] + (float)f.time_sec);
            }
            ImPlot::PlotLine("sin", xs, ys, 256);
            ImPlot::EndPlot();
        }
        ImGui::End();
    }
    void on_cleanup() override { host_->log_info("my_scope: bye"); }
private:
    caliper::Host* host_ = nullptr;
};

CALIPER_APPLET(MyScope,                    // generates the entire C ABI glue
    .id       = "dev.example.my-scope",    // byte-identical to the manifest
    .version  = "0.1.0",
    .name     = "MyScope",
    .summary  = "Smallest complete applet.",
    .tag      = "Demo",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1})
