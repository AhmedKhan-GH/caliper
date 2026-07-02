#include <caliper/caliper.hpp>
#include "repnet_demo.h"

class RepNetDemoPlugin final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override {
        (void)host;
        return impl_.initialize();
    }
    void on_frame(const caliper::Frame& f) override {
        impl_.draw_ui(f.fb_width, f.fb_height);
    }
    void on_cleanup() override { impl_.cleanup(); }

private:
    RepNetDemoApplet impl_;
};

CALIPER_APPLET(RepNetDemoPlugin,
    .id       = "dev.ahmed.repnet-demo",
    .version  = "1.0.0",
    .name     = "RepNet Demo",
    .summary  = "Load and visualize the UCDH Senior Design dataset, run signal "
                "preprocessing, inspect raw data via DuckDB, and run model "
                "inference on ECG recordings.",
    .tag      = "ECG",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1})
