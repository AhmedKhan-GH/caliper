// gfx_selftest.h: a self-test of the graphics layer's WebGPU backend in the page, for the browser
// tests (web/test/wasm.spec.ts). It draws through the layer only, reads every target back with the
// layer's asynchronous readback and compares each texel exactly with what the CPU expects.
#pragma once

#include "gfx/gfx.h"

#include <cstdint>
#include <string>
#include <vector>

namespace caliper_web {

struct SelftestImage {
    std::string name;
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> rgba;   // tightly packed, top row first
};

// Starts the self-test on the backend: one frame with two dispatches, a buffer-to-texture copy
// and one draw list of an indexed quad, an instanced draw and an alpha-blended instanced draw
// with a scissor, then the readbacks of every target. False, with why, when it could not start.
bool gfx_selftest_start(caliper_gfx::Backend& b, std::string* why);

// Called every frame while it runs. Returns true once every readback is in, with the report:
// "ok" or "fail", then key=value facts separated by spaces (see gfx_selftest.cpp).
bool gfx_selftest_poll(caliper_gfx::Backend& b, std::string* report);

bool gfx_selftest_running();
const std::vector<SelftestImage>& gfx_selftest_images();

}  // namespace caliper_web
