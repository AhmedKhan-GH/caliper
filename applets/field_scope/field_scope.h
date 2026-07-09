#pragma once
// FieldScope — a self-consistent electrostatic Particle-In-Cell (PIC) plasma,
// drawn with zero copies (id dev.caliper.field-scope). The particles generate
// their own field: each step deposits charge to a grid, an FFT Poisson solve
// gives the self-field, it is gathered back and a Boris pusher advances them —
// so real collective behaviour (two-stream instability, plasma oscillations)
// emerges. Built on the flow_scope backbone: the SAME pool tensors the torch
// sim writes are the buffers the renderer's point pass reads, per frame.
// Right-drag orbits, wheel zooms, left-drag perturbs the plasma. See em_pic.h.
#include <memory>

namespace caliper { class Host; }

namespace fieldscope {

struct FieldScopeState;

class FieldScopeApplet {
public:
    FieldScopeApplet();
    ~FieldScopeApplet();
    bool initialize(caliper::Host& host);
    void draw_ui();
    void cleanup();
private:
    std::unique_ptr<FieldScopeState> s_;
};

}  // namespace fieldscope
