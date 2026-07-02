// Epoch-2 fixture applet (PLATFORM.md §13.1): loader-test substrate and the
// "hello world" of the sugar layer. Kept deliberately tiny.
#include <caliper/caliper.hpp>
#include <cmath>
#include <cstdlib>
#include <vector>

class HelloApplet final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override {
        host_ = &host;
        crash_on_frame_ = std::getenv("CALIPER_HELLO_CRASH") != nullptr;
        host.log_info("hello.on_init");
        return true;
    }

    void on_frame(const caliper::Frame& f) override {
        if (crash_on_frame_) {           // test hook: fault before any ImGui call
            volatile int* p = nullptr;
            *p = 1;
        }
        ImGui::SetNextWindowPos({40, 60}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({520, 360}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Hello, Caliper");
        ImGui::Text("ABI epoch %d applet via CALIPER_APPLET macro", CALIPER_ABI_EPOCH);
        ImGui::Text("framebuffer: %d x %d px   dpi_scale: %.1f",
                    f.fb_width, f.fb_height, f.dpi_scale);
        if (ImPlot::BeginPlot("sine", {-1, 220})) {
            static std::vector<float> xs(256), ys(256);
            for (int i = 0; i < 256; i++) {
                xs[i] = i / 255.0f * 6.28318f;
                ys[i] = std::sin(xs[i] + (float)f.time_sec);
            }
            ImPlot::PlotLine("sin", xs.data(), ys.data(), 256);
            ImPlot::EndPlot();
        }
        ImGui::End();
    }

    void on_cleanup() override {
        if (host_) host_->log_info("hello.on_cleanup");
    }

private:
    caliper::Host* host_ = nullptr;
    bool crash_on_frame_ = false;
};

CALIPER_APPLET(HelloApplet,
    .id       = "dev.caliper.hello",
    .version  = "0.1.0",
    .name     = "Hello",
    .summary  = "Epoch-2 fixture applet: sugar demo + loader-test substrate.",
    .tag      = "Demo",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1})
