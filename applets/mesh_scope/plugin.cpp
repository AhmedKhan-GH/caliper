#include <caliper/caliper.hpp>
#include "mesh_scope.h"

class MeshScopePlugin final : public caliper::Applet {
public:
    bool on_init(caliper::Host& host) override { return impl_.initialize(host); }
    void on_frame(const caliper::Frame&) override { impl_.draw_ui(); }
    void on_cleanup() override { impl_.cleanup(); }

private:
    meshscope::MeshScopeApplet impl_;
};

CALIPER_APPLET(MeshScopePlugin,
    .id       = "dev.caliper.mesh-scope",
    .version  = "0.1.0",
    .name     = "MeshScope",
    .summary  = "Arbitrary imported geometry demo for caliper.geometry.v1_1: "
                "a live deforming height-field mesh rendered as indexed "
                "triangles with depth, Lambert normals, a colormapped "
                "per-vertex attribute, and a wireframe line overlay. The "
                "vertex/index/normal/attribute tensors are imported through "
                "tensor_bridge.v1_2 and read directly by the renderer.",
    .tag      = "ML",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_DEVICE_V1,
                 CALIPER_TENSOR_BRIDGE_V1, CALIPER_TENSOR_BRIDGE_V1_2,
                 CALIPER_GEOMETRY_V1_1})
