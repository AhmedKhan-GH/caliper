#include <caliper/caliper.hpp>
#include "instance_scope.h"

class InstanceScopePlugin final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override { return impl_.initialize(host); }
    void on_frame(const caliper::Frame&) override { impl_.draw_ui(); }
    void on_cleanup() override { impl_.cleanup(); }

private:
    instancescope::InstanceScopeApplet impl_;
};

CALIPER_APPLET(InstanceScopePlugin,
    .id       = "dev.caliper.instance-scope",
    .version  = "0.1.0",
    .name     = "InstanceScope",
    .summary  = "Instancing, made obvious: a field of N procedural gems bobs and "
                "spins in a traveling wave, drawn with ONE instanced draw call "
                "and ZERO copies of the mesh. Per-frame rigid (N,16) poses and a "
                "(N,) sin-phase tint are recomputed on device and imported "
                "zero-copy (caliper.geometry.v1_3); the tint rides a fixed "
                "[-1,1] MAGMA window that can never saturate. Falls back to a "
                "single non-instanced gem where the instanced path is absent.",
    .tag      = "GFX",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_JOBS_V1,
                 CALIPER_DEVICE_V1, CALIPER_TENSOR_BRIDGE_V1,
                 CALIPER_TENSOR_BRIDGE_V1_2, CALIPER_GEOMETRY_V1_1,
                 CALIPER_GEOMETRY_V1_2, CALIPER_GEOMETRY_V1_3})
