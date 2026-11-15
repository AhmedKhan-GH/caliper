#pragma once
// ============================================================================
// window_captures.h: Caliper API's GET /v0/capture.png in the app's window
// (main.cpp). The API hands the frame thread a Capture call at the start of a
// frame; here it waits for the renderer's read-back of a presented frame
// (HostRenderer::window_capture_request / window_capture_poll):
//
//   before_frame()  before new_frame(): the calls that wait arm a read-back of
//                   the frame about to be drawn, unless one is in flight (they
//                   then wait for the next one);
//   after_frame()   after render(): a finished read-back answers every call it
//                   serves, with the frame's pixels at the framebuffer's size.
//
// The frame loop never waits. The copy completes on the GPU and a later frame
// picks it up; the API's own deadline (504 after frame_timeout_ms) bounds how
// long a call can wait, for example while the window draws no frames. A
// renderer that cannot read its window back answers 501. Header-only.
// ============================================================================
#include "renderer/host_renderer.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace caliper_host {

class WindowCaptures {
public:
    using Done = std::function<void(uint64_t ticket, uint32_t w, uint32_t h, std::vector<uint8_t> rgba)>;
    using Fail = std::function<void(uint64_t ticket, int status, const std::string& message)>;

    // The answer to a capture when the renderer cannot read its window back.
    static std::string unsupported(const char* renderer_name) {
        return std::string("the ") + renderer_name +
               " renderer cannot read the window's frame back; start the app windowless with "
               "caliper shot --api to capture its frame";
    }

    // A Capture call taken this frame.
    void add(uint64_t ticket) { waiting_.push_back(ticket); }
    // Calls not answered yet.
    size_t pending() const { return waiting_.size() + serving_.size(); }

    void before_frame(HostRenderer& r, const Fail& fail) {
        if (waiting_.empty() || !serving_.empty()) return;
        if (!r.window_capture_request()) {
            for (uint64_t t : waiting_) fail(t, 501, unsupported(r.name()));
            waiting_.clear();
            return;
        }
        serving_.swap(waiting_);
    }

    void after_frame(HostRenderer& r, const Done& done, const Fail& fail) {
        if (serving_.empty()) return;
        std::vector<uint8_t> px;
        int w = 0, h = 0;
        const HostRenderer::WindowCapture got = r.window_capture_poll(&px, &w, &h);
        if (got == HostRenderer::WindowCapture::Pending) return;
        std::vector<uint64_t> served;
        served.swap(serving_);
        if (got != HostRenderer::WindowCapture::Ready || w <= 0 || h <= 0 ||
            px.size() != (size_t)w * (size_t)h * 4) {
            for (uint64_t t : served)
                fail(t, 500, std::string("could not read the window's frame back from the ") + r.name() +
                                 " renderer");
            return;
        }
        for (size_t i = 0; i < served.size(); ++i)
            done(served[i], (uint32_t)w, (uint32_t)h, i + 1 == served.size() ? std::move(px) : px);
    }

private:
    std::vector<uint64_t> waiting_;   // taken, waiting for a read-back to be armed
    std::vector<uint64_t> serving_;   // served by the read-back in flight
};

}  // namespace caliper_host
