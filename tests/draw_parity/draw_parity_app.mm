// draw_parity_app.mm: the app's side of the pixel comparison between the app and the browser
// module (design/programmable-graphics.md, Pixel Comparison).
//
//   caliper_draw_parity_app <out>
//
// writes the fixture (parity_fixture.h) as <out>/frames.bin and <out>/plan.json, then decodes
// those frames again exactly as the browser module does (wire codec, program_frames.h), and draws
// each program of the plan that a viewer must draw with the app's draw runtime on its Metal device
// in the offscreen mode of the graphics layer: no window. For each it writes into <out>/app/:
//
//   <id>.color   the color target, RGBA8, top row first
//   <id>.id      the ID target, RG32Uint (pick id, item), top row first
//   <id>.bloom   one byte for each pixel: 1 where the bloom of a glow target lights it, which the
//                comparison holds to 3 levels in place of 2 (the frame again with the param
//                glow at 0 tells those pixels; 0 everywhere for a piece without glow)
//   <id>.png     the color target, for the captures
//
// and <out>/app/report.txt with one line for each program, or <out>/skipped.txt with the reason
// when it drew nothing. The browser test
// (web/test/draw_parity.spec.ts) draws the same frames in headless Chromium and compares.
// Without a Metal device, or with one that MPS does not support (the GPU programs' tests skip
// there too), it writes the fixture only, prints "skipped: no Metal device ..." and exits 77, which
// ctest reports as skipped.
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "parity_fixture.h"

#include "gfx/metal_backend.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-function"
#include <stb_image_write.h>
#pragma clang diagnostic pop

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <thread>

namespace {

namespace cd = caliper_draw;
namespace cs = caliper_builder::spec;
namespace dp = draw_parity;
namespace fs = std::filesystem;
namespace gfx = caliper_gfx;
namespace wire = caliper_net::wire;

bool write_file(const fs::path& p, const void* data, std::size_t n) {
    std::ofstream f(p, std::ios::binary);
    f.write(static_cast<const char*>(data), std::streamsize(n));
    return bool(f);
}

struct Decoded {
    std::map<std::string, wire::Program> programs;
    std::map<std::string, std::vector<wire::Resource>> resources;   // the sections point into the file's bytes
};

bool decode_all(const std::vector<std::uint8_t>& file, Decoded* out, std::string* why) {
    std::vector<std::pair<const std::uint8_t*, std::size_t>> frames;
    if (!dp::split_frames(file.data(), file.size(), &frames, why)) return false;
    for (const auto& [data, n] : frames) {
        wire::Message m;
        const wire::DecodeResult r = wire::decode(data, n, &m);
        if (!r.ok()) return (*why = "a frame does not decode: " + r.reason + r.error_code + " " + r.message, false);
        if (auto* p = std::get_if<wire::Program>(&m)) out->programs[p->id] = *p;
        else if (auto* res = std::get_if<wire::Resource>(&m)) out->resources[res->program].push_back(*res);
        else return (*why = "the fixture holds a frame that is neither PROGRAM nor RESOURCE", false);
    }
    return true;
}

// Waits for one submit and gives the runtime its GPU time when the layer measured it.
bool wait(gfx::Backend& g, gfx::Serial s) {
    if (!g.wait(s)) return false;
    return true;
}

std::vector<std::uint8_t> read(gfx::Backend& g, gfx::Texture t, std::uint32_t size, std::string* why) {
    std::vector<std::uint8_t> out;
    const gfx::Readback rb = g.read_texture(t, 0, 0, size, size);
    if (!rb || !g.wait(g.last_use(t)) || !g.take_readback(rb, &out)) *why = "a readback failed: " + g.last_error();
    return out;
}

// Draws one program `frames` times with `values` as its params and reads both targets back.
bool draw(gfx::Backend& g, const dp::Plan& plan, const dp::ProgramPlan& pp, const cs::Piece& piece,
          const cd::Instance& instance, const std::vector<wire::Resource>& resources, const caliper_net::Json& values,
          std::vector<std::uint8_t>* color, std::vector<std::uint8_t>* ids, std::string* why) {
    dp::Target target(g, plan.size);
    if (!target.ok) return (*why = "the host target was refused: " + g.last_error(), false);
    cd::DrawRuntime rt(g, false);
    const cd::Verdict v = rt.install(piece, instance, pp.pick_id);
    if (!v.ok()) return (*why = "install: " + v.code + " " + v.message, false);
    for (const wire::Resource& r : resources)
        if (const cd::Verdict s = rt.set_resource(caliper_web::resource_version(r)); !s.ok())
            return (*why = "resource " + r.resource + ": " + s.code + " " + s.message, false);
    for (int i = 0; i < plan.frames; ++i) {
        @autoreleasepool {
            std::vector<gfx::Serial> serials;
            const cd::FrameInputs in = dp::frame_inputs(pp, plan, i, caliper_web::pack_params(piece, values));
            if (!target.frame(rt, in, &serials, why)) return false;
            for (std::size_t k = 0; k < serials.size(); ++k) {
                if (!wait(g, serials[k])) return (*why = "a submit did not complete: " + g.last_error(), false);
                if (k == 0) continue;   // the clear is the scene's, not the program's
                double ms = -1;
                const auto start = std::chrono::steady_clock::now();
                while (!g.submit_gpu_ms(serials[k], &ms) &&
                       std::chrono::steady_clock::now() - start < std::chrono::milliseconds(200))
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (ms >= 0) rt.frame_done(ms);
            }
        }
    }
    @autoreleasepool {
        *color = read(g, target.host.color, plan.size, why);
        *ids = read(g, target.host.id, plan.size, why);
    }
    const std::size_t px = std::size_t(plan.size) * plan.size;
    if (color->size() != px * 4 || ids->size() != px * 8) {
        if (why->empty()) *why = "a readback has the wrong size";
        return false;
    }
    if (rt.status().state != cd::DrawState::running)
        return (*why = "the program is not running: " + rt.status().error.code + " " + rt.status().error.message, false);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: caliper_draw_parity_app <out>\n");
        return 2;
    }
    const fs::path out = argv[1];
    std::error_code ec;
    fs::create_directories(out / "app", ec);
    for (const auto& e : fs::directory_iterator(out / "app", ec)) fs::remove(e.path(), ec);   // no stale picture
    fs::remove(out / "skipped.txt", ec);

    const dp::Fixture fx = dp::make_fixture();
    if (!fx.error.empty()) {
        std::fprintf(stderr, "fixture: %s\n", fx.error.c_str());
        return 1;
    }
    const std::string plan_text = dp::plan_json(fx.plan);
    if (!write_file(out / "frames.bin", fx.frames.data(), fx.frames.size()) ||
        !write_file(out / "plan.json", plan_text.data(), plan_text.size())) {
        std::fprintf(stderr, "cannot write the fixture into %s\n", out.string().c_str());
        return 1;
    }
    std::printf("fixture: %zu bytes of frames, %zu programs\n", fx.frames.size(), fx.plan.programs.size());

    // The app's side decodes the frames it wrote, as the browser does.
    Decoded d;
    std::string why;
    if (!decode_all(fx.frames, &d, &why)) {
        std::fprintf(stderr, "decode: %s\n", why.c_str());
        return 1;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        const std::string why = "no Metal device";
        write_file(out / "skipped.txt", why.data(), why.size());
        std::printf("skipped: %s\n", why.c_str());
        return 77;
    }
    // A device that MPS does not support, as the paravirtual device of a hosted runner, is no usable
    // device for programs: the GPU programs' tests skip there by the same test, through torch.
    if (!MPSSupportsMTLDevice(device)) {
        const std::string why = "no Metal device that MPS supports (" + std::string(device.name.UTF8String) + ")";
        write_file(out / "skipped.txt", why.data(), why.size());
        std::printf("skipped: %s\n", why.c_str());
        return 77;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    std::unique_ptr<gfx::Backend> g = gfx::make_metal_backend((__bridge void*)device, (__bridge void*)queue);
    if (!g) {
        const std::string why = "no Metal device (the graphics layer made no Metal backend)";
        write_file(out / "skipped.txt", why.data(), why.size());
        std::printf("skipped: %s\n", why.c_str());
        return 77;
    }

    std::string report;
    int failed = 0;
    for (const dp::ProgramPlan& pp : fx.plan.programs) {
        const auto pm = d.programs.find(pp.id);
        if (pm == d.programs.end()) {
            std::fprintf(stderr, "%s: no PROGRAM in the fixture\n", pp.id.c_str());
            ++failed;
            continue;
        }
        cs::Piece piece;
        cd::Instance instance;
        const cd::Verdict v = caliper_web::decode_program(pm->second, &piece, &instance);
        if (!pp.expect.empty()) {
            // A program that the browser must refuse; the app checks only what its decode refuses.
            const std::string line = pp.id + " expect " + pp.expect + " decode " + (v.ok() ? "ok" : v.code);
            std::printf("%s\n", line.c_str());
            report += line + "\n";
            continue;
        }
        if (!v.ok()) {
            std::fprintf(stderr, "%s: decode refused it: %s %s\n", pp.id.c_str(), v.code.c_str(), v.message.c_str());
            ++failed;
            continue;
        }
        std::vector<std::uint8_t> color, ids, plain, plain_ids;
        const std::vector<wire::Resource>& res = d.resources[pp.id];
        if (!draw(*g, fx.plan, pp, piece, instance, res, pm->second.params, &color, &ids, &why)) {
            std::fprintf(stderr, "%s: %s\n", pp.id.c_str(), why.c_str());
            ++failed;
            continue;
        }
        // The pixels that the bloom lights: those that change when the glow goes to 0.
        std::vector<std::uint8_t> bloom(std::size_t(fx.plan.size) * fx.plan.size, 0);
        bool has_glow = false;
        for (const cs::PieceParam& p : piece.params) has_glow = has_glow || p.name == "glow";
        std::size_t glow_px = 0;
        if (has_glow) {
            caliper_net::Json dark = pm->second.params;
            dark.set("glow", 0.0);
            if (!draw(*g, fx.plan, pp, piece, instance, res, dark, &plain, &plain_ids, &why)) {
                std::fprintf(stderr, "%s without glow: %s\n", pp.id.c_str(), why.c_str());
                ++failed;
                continue;
            }
            for (std::size_t p = 0; p < bloom.size(); ++p)
                if (std::memcmp(color.data() + 4 * p, plain.data() + 4 * p, 4) != 0) {
                    bloom[p] = 1;
                    ++glow_px;
                }
        }
        std::size_t lit = 0, picked = 0;
        for (std::size_t p = 0; p < bloom.size(); ++p) {
            std::uint32_t rgba = 0, id0 = 0;
            std::memcpy(&rgba, color.data() + 4 * p, 4);
            std::memcpy(&id0, ids.data() + 8 * p, 4);
            lit += rgba != dp::kBackground;
            picked += id0 == pp.pick_id;
        }
        const fs::path base = out / "app" / pp.id;
        const int n = int(fx.plan.size);
        if (!write_file(base.string() + ".color", color.data(), color.size()) ||
            !write_file(base.string() + ".id", ids.data(), ids.size()) ||
            !write_file(base.string() + ".bloom", bloom.data(), bloom.size()) ||
            stbi_write_png((base.string() + ".png").c_str(), n, n, 4, color.data(), n * 4) == 0) {
            std::fprintf(stderr, "%s: cannot write its pictures\n", pp.id.c_str());
            ++failed;
            continue;
        }
        char line[256];
        std::snprintf(line, sizeof line, "%s drawn %ux%u lit %zu picked %zu bloom %zu", pp.id.c_str(), fx.plan.size,
                      fx.plan.size, lit, picked, glow_px);
        std::printf("%s\n", line);
        report += std::string(line) + "\n";
    }
    write_file(out / "app" / "report.txt", report.data(), report.size());
    std::printf("%s\n", failed ? "failed" : "ok");
    return failed ? 1 : 0;
}
