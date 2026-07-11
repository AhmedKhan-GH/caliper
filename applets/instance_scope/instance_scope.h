#pragma once
// InstanceScope — the caliper.geometry.v1_3 instancing exemplar: a field of N
// procedural gems, ONE instanced draw call, ZERO copies of the mesh. Poses and
// tint are recomputed each frame ON DEVICE and imported zero-copy; the HUD leads
// with the layman's pitch "N objects — 1 draw call — 0 copies of the mesh".
#include <memory>

namespace caliper { class Host; }

namespace instancescope {

struct InstanceScopeState;

class InstanceScopeApplet {
public:
    InstanceScopeApplet();
    ~InstanceScopeApplet();
    bool initialize(caliper::Host& host);
    void draw_ui();
    void cleanup();

private:
    std::unique_ptr<InstanceScopeState> s_;
};

}  // namespace instancescope
