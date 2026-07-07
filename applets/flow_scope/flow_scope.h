#pragma once
// FlowScope — a million-particle flow-field simulation drawn with zero copies
// (id dev.caliper.flow-scope). The digital-twin exemplar for
// caliper.geometry.v1: the SAME pool tensors the torch sim writes are the
// buffers the renderer's point pass reads, per frame, at byte offsets.
#include <memory>

namespace caliper { class Host; }

namespace flowscope {

struct FlowScopeState;

class FlowScopeApplet {
public:
    FlowScopeApplet();
    ~FlowScopeApplet();
    bool initialize(caliper::Host& host);
    void draw_ui();
    void cleanup();
private:
    std::unique_ptr<FlowScopeState> s_;
};

}  // namespace flowscope
