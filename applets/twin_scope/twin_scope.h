#pragma once

#include <memory>

namespace caliper { class Host; }

namespace twinscope {

struct TwinScopeState;

// The TwinScope v2 applet: a surface-aware synthetic heat twin on a UV-mapped
// finned housing. A batched cotangent-Laplacian heat sim runs on a subdivided
// sim mesh; a small MLP chases the field. The hero splits sim|net side by side
// (two textured draws, ±x offsets); a live textured↔per-vertex toggle exposes
// the R2 resolution gap. Everything torch happens on the worker/init job — the
// frame thread only snapshots and draws (design §6-§9).
class TwinScopeApplet {
public:
    TwinScopeApplet();
    ~TwinScopeApplet();
    bool initialize(caliper::Host& host);
    void draw_ui();
    void cleanup();

private:
    std::unique_ptr<TwinScopeState> state_;
};

}  // namespace twinscope
