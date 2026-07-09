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
    .summary  = "A small MLP learns a fixed 2-D target surface live: every "
                "optimizer step its 72x72-grid prediction is written into "
                "imported device tensors and drawn the same frame as Lambert-lit "
                "indexed triangles colored by per-vertex squared error through "
                "the MAGMA LUT, with a wireframe overlay and the training "
                "minibatch as additive points — the caliper.geometry.v1_1 "
                "exemplar. Falls back to an input-locked CPU error heatmap.",
    .tag      = "ML",
    .services = {CALIPER_UI_V1, CALIPER_LOG_V1, CALIPER_DEVICE_V1,
                 CALIPER_TENSOR_BRIDGE_V1, CALIPER_TENSOR_BRIDGE_V1_2,
                 CALIPER_GEOMETRY_V1_1})
