// draw_parity.h: the browser's side of the pixel comparison between the app and the browser
// module (design/programmable-graphics.md, Pixel Comparison), for web/test/draw_parity.spec.ts.
//
// The page gives the fixture that the app's side wrote (tests/draw_parity/parity_draw.h): its
// frames and its plan. The module decodes each frame with the shared codec, decodes each PROGRAM
// with the checks of the hash (program_frames.h), runs the browser checks of each draw half
// (web_programs.h, ModuleCheck), and draws each program that passes with the draw runtime on the
// module's WebGPU backend into a host target of the plan's size, as the app's side draws it on
// Metal. Then it reads the color and ID targets back with the layer's asynchronous readback.
#pragma once

#include "gfx/gfx.h"

#include <webgpu/webgpu.h>

#include <cstdint>
#include <string>
#include <vector>

namespace caliper_web {

// One program's result: its verdict, and its targets when it drew.
struct ParityResult {
    std::string id;
    std::string code, message, where;      // the refusal or the stop; code "" when it drew
    std::uint32_t line = 0, column = 0;
    std::vector<std::uint8_t> color;       // RGBA8, top row first
    std::vector<std::uint8_t> ids;         // RG32Uint, top row first
};

// Starts the comparison's draw. False, with why, when it could not start.
bool draw_parity_start(caliper_gfx::Backend& b, WGPUDevice device, const std::uint8_t* frames, std::size_t n,
                       const std::string& plan_json, std::string* why);
// Called every frame while it runs. True once every readback is in, with the report: "ok" or
// "fail", then one entry for each program, "<id>=drawn" or "<id>=<code>".
bool draw_parity_poll(caliper_gfx::Backend& b, const std::vector<std::string>& device_errors, std::string* report);
bool draw_parity_running();
const std::vector<ParityResult>& draw_parity_results();
// Releases the runtimes and the targets; before the backend goes.
void draw_parity_stop();

}  // namespace caliper_web
