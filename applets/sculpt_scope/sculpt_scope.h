#pragma once
// SculptScope — a generator MLP g_θ: R^k -> R^3 trained live to match a target
// shape, drawn with zero copies (id dev.caliper.sculpt-scope). Built on the
// field_scope/flow_scope zero-copy spine: the (N,3) tensor the net's final
// layer writes (via addmm_out, into a pool-born slot) IS the buffer the
// renderer's instanced point pass reads in place, per frame. Right-drag orbits,
// wheel zooms. Falls back to a subsampled CPU scatter where zero-copy is absent.
#include <memory>

namespace caliper { class Host; }

namespace sculptscope {

struct SculptScopeState;

class SculptScopeApplet {
public:
    SculptScopeApplet();
    ~SculptScopeApplet();
    bool initialize(caliper::Host& host);
    void draw_ui();
    void cleanup();
private:
    std::unique_ptr<SculptScopeState> s_;
};

}  // namespace sculptscope
