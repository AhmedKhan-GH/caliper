#pragma once
// PulseScope — the caliper.feed.v1 dashboard exemplar (feed spec §5). pImpl over
// a torch-free state struct; the ABI-facing shell is plugin.cpp.
#include <memory>

namespace caliper { class Host; }

namespace pulsescope {

struct PulseState;

class PulseScopeApplet {
public:
    PulseScopeApplet();
    ~PulseScopeApplet();

    bool initialize(caliper::Host& host);   // on_init: enumerate channels, spawn poller
    void draw_ui();                         // on_frame: snapshot under mutex, then draw
    void cleanup();                         // on_cleanup: stop + join the poller

private:
    std::unique_ptr<PulseState> s_;
};

}  // namespace pulsescope
