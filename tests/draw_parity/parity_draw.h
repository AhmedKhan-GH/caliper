// parity_draw.h: the draw of the pixel comparison between the app and the
// browser module (design/programmable-graphics.md, Pixel Comparison), shared by
// both sides so that they draw the same frames the same way.
//
// The fixture is a file of wire frames, PROGRAM sets of the instances and
// RESOURCE versions of their published resources, each frame preceded by its
// length as a little-endian u32, and a plan in JSON that names each program to
// draw with its pick id and the frame's values of its fixed camera:
//
//   {"size": 512, "frames": 3, "step": 100,
//    "programs": [{"id": "tower_arcs", "pick_id": 16777216, "expect": "",
//                  "inputs": {"clip_from_world": [16 numbers], "world_from_clip": [...],
//                             "view_from_world": [...], "eye": [4], "viewport": [4],
//                             "pixel_ratio": 2}}]}
//
// A program whose "expect" names a code is one that the viewer must refuse
// with it. Each side decodes the frames with the shared codec
// (src/host/net/wire.h), decodes each PROGRAM with program_frames.h, installs
// the draw runtime and draws `frames` frames into a host target of
// size x size: color RGBA8Unorm cleared to kBackground, ID RG32Uint and depth
// Depth32Float, with the host's colormaps.
#pragma once

#include "draw/draw_runtime.h"
#include "gfx/colormap_tables.h"
#include "gfx/gfx.h"
#include "net/json.h"

#include "../../web/wasm/src/program_frames.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace draw_parity {

inline constexpr std::uint32_t kBackground = 0xff201814u;   // r 0x14, g 0x18, b 0x20, as the graphics suite's rows

// One program of the plan.
struct ProgramPlan {
    std::string id;
    std::uint32_t pick_id = 0;
    std::string expect;               // a code that the viewer must refuse the program with, or ""
    std::uint32_t expect_line = 0;    // the line of the piece's file that the refusal names, or 0
    caliper_draw::FrameInputs inputs; // the camera's values; params come from the PROGRAM
};

struct Plan {
    std::uint32_t size = 512;
    int frames = 3;
    std::int64_t step = 100;
    std::vector<ProgramPlan> programs;
};

// The frames of a fixture file: (bytes, length) of each, inside `data`.
inline bool split_frames(const std::uint8_t* data, std::size_t n,
                         std::vector<std::pair<const std::uint8_t*, std::size_t>>* out, std::string* why) {
    out->clear();
    std::size_t at = 0;
    while (at < n) {
        if (n - at < 4) {
            *why = "the fixture ends inside a frame's length";
            return false;
        }
        std::uint32_t len = 0;
        for (int k = 0; k < 4; ++k) len |= std::uint32_t(data[at + std::size_t(k)]) << (8 * k);
        at += 4;
        if (len > n - at) {
            *why = "a frame of the fixture is longer than the file";
            return false;
        }
        out->push_back({data + at, len});
        at += len;
    }
    return true;
}

// Appends one frame with its length.
inline void append_frame(std::vector<std::uint8_t>* file, const std::vector<std::uint8_t>& frame) {
    const std::uint32_t len = std::uint32_t(frame.size());
    for (int k = 0; k < 4; ++k) file->push_back(std::uint8_t(len >> (8 * k)));
    file->insert(file->end(), frame.begin(), frame.end());
}

template <std::size_t N>
caliper_net::Json numbers(const std::array<float, N>& v) {
    caliper_net::Json a = caliper_net::Json::array();
    for (float f : v) a.push(double(f));
    return a;
}

template <std::size_t N>
bool read_numbers(const caliper_net::Json& o, const char* key, std::array<float, N>* out) {
    const caliper_net::Json* a = o.find(key);
    if (!a || !a->is_array() || a->size() != N) return false;
    for (std::size_t i = 0; i < N; ++i) {
        if (!a->at(i).is_number()) return false;
        (*out)[i] = float(a->at(i).as_double());
    }
    return true;
}

inline std::string plan_json(const Plan& plan) {
    using caliper_net::Json;
    Json root = Json::object();
    root.set("size", std::uint64_t(plan.size));
    root.set("frames", std::int64_t(plan.frames));
    root.set("step", plan.step);
    Json list = Json::array();
    for (const ProgramPlan& p : plan.programs) {
        Json j = Json::object();
        j.set("id", p.id);
        j.set("pick_id", std::uint64_t(p.pick_id));
        j.set("expect", p.expect);
        if (p.expect_line) j.set("line", std::uint64_t(p.expect_line));
        Json in = Json::object();
        in.set("clip_from_world", numbers(p.inputs.clip_from_world));
        in.set("world_from_clip", numbers(p.inputs.world_from_clip));
        in.set("view_from_world", numbers(p.inputs.view_from_world));
        in.set("eye", numbers(p.inputs.eye));
        in.set("viewport", numbers(p.inputs.viewport));
        in.set("pixel_ratio", double(p.inputs.pixel_ratio));
        j.set("inputs", std::move(in));
        list.push(std::move(j));
    }
    root.set("programs", std::move(list));
    return root.dump();
}

inline bool read_plan(const std::string& text, Plan* plan, std::string* why) {
    using caliper_net::Json;
    Json root;
    if (!Json::parse(text.data(), text.size(), &root, why)) return false;
    std::uint64_t u = 0;
    std::int64_t i = 0;
    const Json* v = root.find("size");
    if (!v || !v->get_u64(&u) || u == 0 || u > 4096) return (*why = "the plan's size is not 1 to 4096", false);
    plan->size = std::uint32_t(u);
    v = root.find("frames");
    if (!v || !v->get_i64(&i) || i < 1 || i > 16) return (*why = "the plan's frames are not 1 to 16", false);
    plan->frames = int(i);
    v = root.find("step");
    if (!v || !v->get_i64(&i)) return (*why = "the plan has no step", false);
    plan->step = i;
    const Json* list = root.find("programs");
    if (!list || !list->is_array()) return (*why = "the plan has no programs", false);
    plan->programs.clear();
    for (const Json& j : list->items()) {
        ProgramPlan p;
        const Json* id = j.find("id");
        const Json* pick = j.find("pick_id");
        const Json* in = j.find("inputs");
        if (!id || !id->is_string() || !pick || !pick->get_u64(&u) || u > 0xffffffffu || !in || !in->is_object())
            return (*why = "a program of the plan lacks its id, pick_id or inputs", false);
        p.id = id->as_string();
        p.pick_id = std::uint32_t(u);
        if (const Json* e = j.find("expect"); e && e->is_string()) p.expect = e->as_string();
        if (const Json* l = j.find("line"); l && l->get_u64(&u) && u <= 0xffffffffu) p.expect_line = std::uint32_t(u);
        const Json* ratio = in->find("pixel_ratio");
        if (!read_numbers(*in, "clip_from_world", &p.inputs.clip_from_world) ||
            !read_numbers(*in, "world_from_clip", &p.inputs.world_from_clip) ||
            !read_numbers(*in, "view_from_world", &p.inputs.view_from_world) ||
            !read_numbers(*in, "eye", &p.inputs.eye) || !read_numbers(*in, "viewport", &p.inputs.viewport) || !ratio ||
            !ratio->is_number())
            return (*why = "the inputs of " + p.id + " are incomplete", false);
        p.inputs.pixel_ratio = float(ratio->as_double());
        plan->programs.push_back(std::move(p));
    }
    return true;
}

// The values of frame `index` (from 0) of a program: its camera, a fixed time, the step of the
// resources as the cursor, and the packed params.
inline caliper_draw::FrameInputs frame_inputs(const ProgramPlan& p, const Plan& plan, int index,
                                              std::vector<std::uint8_t> params) {
    caliper_draw::FrameInputs in = p.inputs;
    in.time_s = double(index) / 60.0;
    in.dt_s = index == 0 ? 0.0 : 1.0 / 60.0;
    in.cursor_step = plan.step;
    in.latest_step = plan.step;
    in.live = true;
    in.params = std::move(params);
    return in;
}

// The host target of a scene panel and the host's colormaps on one backend.
struct Target {
    caliper_gfx::Backend& g;
    caliper_draw::HostTarget host;
    bool ok = false;

    Target(caliper_gfx::Backend& backend, std::uint32_t size) : g(backend) {
        caliper_gfx::TextureDesc d;
        d.width = size;
        d.height = size;
        d.format = caliper_gfx::Format::RGBA8Unorm;
        d.usage = caliper_gfx::kRenderTarget | caliper_gfx::kSampled | caliper_gfx::kCopySrcTex;
        d.label = "parity color";
        host.color = g.create_texture(d);
        d.format = caliper_gfx::Format::RG32Uint;
        d.usage = caliper_gfx::kRenderTarget | caliper_gfx::kCopySrcTex;
        d.label = "parity id";
        host.id = g.create_texture(d);
        d.format = caliper_gfx::Format::Depth32Float;
        d.usage = caliper_gfx::kRenderTarget;
        d.label = "parity depth";
        host.depth = g.create_texture(d);
        host.width = size;
        host.height = size;
        caliper_gfx::TextureDesc c;
        c.width = 256;
        c.height = caliper_gfx::cmap::kBuiltinColormaps;
        c.format = caliper_gfx::Format::RGBA8Unorm;
        c.usage = caliper_gfx::kSampled | caliper_gfx::kCopyDstTex;
        c.label = "parity colormaps";
        host.colormaps = g.create_texture(c);
        std::vector<std::uint32_t> rows(256u * caliper_gfx::cmap::kBuiltinColormaps);
        for (int id = 0; id < caliper_gfx::cmap::kBuiltinColormaps; ++id)
            std::memcpy(rows.data() + 256u * std::uint32_t(id), caliper_gfx::cmap::builtin_lut(id), 256 * 4);
        ok = host.color && host.id && host.depth && host.colormaps && g.write_texture(host.colormaps, rows.data(), 256 * 4);
    }
    ~Target() {
        for (caliper_gfx::Texture t : {host.color, host.id, host.depth, host.colormaps})
            if (t) g.release(t);
    }
    Target(const Target&) = delete;
    Target& operator=(const Target&) = delete;

    // One frame: the scene's clear of the host target in a submit, then the program's frames, each
    // in a submit of its own, as the scene view submits them. The serials go to *serials.
    bool frame(caliper_draw::DrawRuntime& rt, const caliper_draw::FrameInputs& in,
               std::vector<caliper_gfx::Serial>* serials, std::string* why) {
        caliper_gfx::RenderTarget clear;
        clear.color = host.color;
        clear.id = host.id;
        clear.depth = host.depth;
        clear.clear_rgba = kBackground;
        caliper_gfx::Frame f;
        f.render(caliper_gfx::DrawList(clear));
        const caliper_gfx::Serial s = g.submit(std::move(f));
        if (s == 0) return (*why = "the clear was refused: " + g.last_error(), false);
        serials->push_back(s);
        std::vector<caliper_gfx::Frame> frames;
        if (!rt.encode(&frames, in, host)) {
            const caliper_draw::DrawStatus st = rt.status();
            *why = "the program drew nothing: " + st.error.code + " " + st.error.message;
            return false;
        }
        for (caliper_gfx::Frame& pf : frames) {
            const caliper_gfx::Serial ps = g.submit(std::move(pf));
            if (ps == 0) return (*why = "a frame of the program was refused: " + g.last_error(), false);
            serials->push_back(ps);
        }
        return true;
    }
};

}  // namespace draw_parity
