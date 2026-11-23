// web_programs.h: the draw halves of pieces in the browser module
// (design/programmable-graphics.md, Resource Message and Browser Checks).
//
// A viewer of a run gets each draw half from the script as the instance form
// of PROGRAM set, and each version of a published resource as RESOURCE. The
// module has no WGSL compiler. Before a program draws, it runs the checks of
// pieces with the same code as the app (check 1, name and files; check 2, the
// manifest; check 4, the files; check 5, the imports and the hash of the pin,
// in DrawRuntime::decode), composes each pass with the app's composer, and
// gives each module to WebGPU. A compilation error refuses the program with
// program.browser_refused, with its line mapped to the piece's file. A
// pipeline that WebGPU refuses later, which reaches the backend's error list,
// refuses the program with the same code. WebGPU's robust buffer access keeps
// each draw inside its bindings.
//
// WebPrograms keeps the programs and the newest version of each resource (the
// module keeps no history: Decisions, No history in the browser) and is the
// DrawFeed of each scene view whose name is a program's panel. The scene view
// owns the runtimes (src/host/live/live_scene.h). Everything runs on the
// page's main thread.
#pragma once

#include "draw/draw_host.h"   // DrawFeed and PanelProgram
#include "draw/draw_runtime.h"
#include "net/json.h"
#include "net/wire.h"

#include <webgpu/webgpu.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace caliper_web {

// One draw half given to WebGPU, pass by pass: the checks of pieces first,
// then each composed module, whose compilation messages come back later.
class ModuleCheck {
public:
    ModuleCheck() = default;
    ~ModuleCheck();
    ModuleCheck(const ModuleCheck&) = delete;
    ModuleCheck& operator=(const ModuleCheck&) = delete;

    // Runs checks 1, 2 and 4 and the resolution of imports, composes each
    // pass and starts its compilation on `device`. Not ok: a check or the
    // composer refused the piece, and nothing went to WebGPU.
    caliper_draw::Verdict start(WGPUDevice device, const caliper_builder::spec::Piece& piece,
                                const caliper_draw::DrawChecks& checks = caliper_draw::DrawChecks::pieces());
    // True once every pass's compilation answered.
    bool done() const;
    // Once done: ok, or program.browser_refused naming the first pass with an
    // error, its file, line and column, and WebGPU's message.
    caliper_draw::Verdict verdict() const;

private:
    struct State;
    std::shared_ptr<State> state_;
};

class WebPrograms {
public:
    explicit WebPrograms(WGPUDevice device);
    ~WebPrograms();

    // PROGRAM from the script: set (the instance form; a program of the
    // one-pass form is not a draw half and is ignored), remove, and params.
    // A new program past CALIPER_NET_MAX_PROGRAMS is ignored.
    void on_program(const caliper_net::wire::Program& m);
    // RESOURCE: the newest version of a resource of a program, copied, when
    // the program's piece publishes that resource and the version holds at
    // most its declared count and bytes (program_frames.h, keeps_version).
    // Thus the page keeps at most the published bytes of 16 pieces. The parts of
    // a version in parts are put together first (caliper_net::wire::SplitResources),
    // which holds at most two incomplete versions.
    void on_resource(const caliper_net::wire::Resource& r);
    // Once a frame: takes the browser checks that finished, and refuses a
    // program whose module or pipeline WebGPU refused since the last call
    // (`device_errors` is the backend's whole list).
    void pump(const std::vector<std::string>& device_errors);
    // The feed of the scene view `panel`, made at its first use.
    caliper_draw::DrawFeed* feed(const std::string& panel);
    // The panels that hold a program.
    std::vector<std::string> panels() const;
    // The programs for the page: [{id, panel, state, code, message, where,
    // line, column, step, frames}] with the state of the browser check or of
    // the scene view's runtime, and the frames that it drew.
    std::string status_json() const;
    // Changes whenever a program is added, replaced, refused or removed.
    std::uint64_t revision() const { return revision_; }

    struct Program;
    struct Version {
        std::shared_ptr<const std::vector<std::uint8_t>> bytes;   // never null once received
        std::int64_t step = -1;
        std::uint32_t count = 0;
    };

private:
    class PanelFeed;
    WGPUDevice device_ = nullptr;
    std::map<std::string, std::unique_ptr<Program>> programs_;
    std::map<std::string, std::unique_ptr<PanelFeed>> feeds_;
    caliper_net::wire::SplitResources parts_;   // versions in parts being put together
    std::uint64_t revision_ = 1;
    std::uint32_t next_pick_ = 1u << 24;   // above the entities' ids, as the app's draw host gives them
    std::size_t errors_seen_ = 0;
};

}  // namespace caliper_web
