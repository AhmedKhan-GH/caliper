#include <caliper/caliper.hpp>
#include "circuitnet.h"

// Epoch-2 entry (PLATFORM.md §6a). All applet logic stays in CircuitNetApplet;
// this file is only the bridge. ui::connect is handled by the macro.
class CircuitNetPlugin final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override {
        (void)host;
        return impl_.initialize();
    }
    void on_frame(const caliper::Frame& f) override {
        impl_.draw_ui(f.fb_width, f.fb_height);   // physical px, same as v1
    }
    void on_cleanup() override { impl_.cleanup(); }

private:
    CircuitNetApplet impl_;
};

CALIPER_APPLET(CircuitNetPlugin,
    .id       = "dev.ahmed.circuitnet",
    .version  = "1.0.0",
    .name     = "CircuitNet 3.0",
    .summary  = "Gate-level circuit architecture explorer with DuckDB-powered "
                "querying, Verilog netlist parsing, and interactive graph "
                "visualization.",
    .tag      = "EDA",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1})
