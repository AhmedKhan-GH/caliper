// uniform_check.h: one fixed scene compiled by the scene core, whose uniform bytes must be the
// same on every backend of the graphics layer.
//
// The scene core composes a view's camera in double and rounds each draw's model-view-projection
// to f32 once (compile.cpp); it writes the frame block, every draw's parameter block and the
// colormap blocks into the draw list's arena, and the backends upload that arena as it is: the
// Metal backend copies it into its arena buffer (metal_backend.mm, on_submit), the WebGPU backend
// writes it with one wgpuQueueWriteBuffer (webgpu_backend.cpp, on_submit), and the recording
// backend keeps the frame (recording_backend.cpp). So the bytes a draw reads do not depend on the
// backend, and a browser draws a scene from the same f32 matrices as Metal. uniform_check()
// compiles two views of a points batch, a 3-D one under an orbit and a 2-D one under a pan and
// zoom, against a backend's resources and hashes their arenas; the tests (the graphics suite's
// points view rows natively, the browser module's self-test in a page) compare the hash with
// kUniformCheckFnv1a, which the step that wrote this pinned from a native run.
#pragma once

#include "gfx/colormap_keys.h"
#include "gfx/gfx.h"
#include "scene/backend_resources.h"
#include "scene/camera.h"
#include "scene/compile.h"
#include "scene/scene.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

namespace caliper_scene {

inline constexpr uint64_t kUniformCheckFnv1a = 0xda7424328d45d9eaull;

// The FNV-1a hash of the arenas of the fixture's two compiled views on backend `b`. detail, when
// given, receives each view's blocks: the hash of the frame block, of the draw's parameter block and
// of the colormap block, and the parameter block's first 48 words in hex (the f32
// model-view-projection, then the normal matrix columns).
inline bool uniform_check(caliper_gfx::Backend& b, const caliper_gfx::ProgramDesc* programs, uint64_t* hash,
                          std::string* why, std::string* detail = nullptr) {
    namespace gfx = caliper_gfx;
    BackendResources res(b, programs, gfx::cmap::colormap_source(std::make_shared<gfx::cmap::KeyBlockCache>()));
    if (!res.init(why)) return false;
    gfx::BufferDesc bd;
    bd.size = 8 * 12;
    bd.usage = gfx::kVertex | gfx::kCopyDst;
    const gfx::Buffer pos = b.create_buffer(bd);
    bd.size = 8 * 4;
    const gfx::Buffer val = b.create_buffer(bd);
    gfx::TextureDesc td;
    td.width = 64;
    td.height = 48;
    td.format = gfx::Format::RGBA8Unorm;
    td.usage = gfx::kRenderTarget | gfx::kSampled;
    const gfx::Texture color = b.create_texture(td);
    td.format = gfx::Format::Depth32Float;
    td.usage = gfx::kRenderTarget;
    const gfx::Texture depth = b.create_texture(td);
    uint64_t h = 0xcbf29ce484222325ull;
    bool ok = pos && val && color && depth;
    if (!ok && why) *why = "the fixture's resources: " + b.last_error();
    for (int view = 0; ok && view < 2; ++view) {
        const bool three = view == 0;
        SceneTree tree(three ? Space::k3D : Space::k2D);
        Batch batch;
        BatchArray p;
        p.role = Role::Positions;
        p.buffer = pos;
        p.count = 8;
        p.stride = 12;
        p.format = three ? gfx::VertexFormat::F32x3 : gfx::VertexFormat::F32x2;
        batch.arrays.push_back(p);
        BatchArray v;
        v.role = Role::Values;
        v.buffer = val;
        v.count = 8;
        v.stride = 4;
        v.format = gfx::VertexFormat::F32;
        batch.arrays.push_back(v);
        batch.colormap = 1;
        batch.vmin = -1.5f;
        batch.vmax = 2.25f;
        batch.point_shape = PointShape::Square;
        batch.bounds.add({-1.25, -0.75, three ? -0.5 : 0.0});
        batch.bounds.add({1.5, 2.0, three ? 0.875 : 0.0});
        std::string err;
        if (!tree.set_batch(tree.root(), batch, false, &err)) {
            if (why) *why = err;
            ok = false;
            break;
        }
        tree.update();
        ViewCamera cam;
        if (three) {
            Orbit o;
            o.target[0] = 0.3;
            o.target[1] = -0.2;
            o.target[2] = 0.1;
            o.distance = 4.7;
            o.azimuth = 0.81;
            o.elevation = 0.43;
            cam = o;
        } else {
            PanZoom z;
            z.center[0] = 0.25;
            z.center[1] = -0.5;
            z.half_height = 1.7;
            z.y_down = false;
            cam = z;
        }
        gfx::RenderTarget target;
        target.color = color;
        target.depth = three ? depth : gfx::Texture{};
        target.clear_rgba = 0xff262320u;
        gfx::DrawList list(target);
        CompileOptions opt;
        opt.width = 64;
        opt.height = 48;
        opt.pixel_scale = 2.0f;
        opt.clear_rgba = target.clear_rgba;
        SceneCompiler compiler;
        const CompileResult r = compiler.compile(tree, cam, opt, res, list, nullptr);
        if (!r.ok || r.draws != 1) {
            if (why) *why = r.notes.empty() ? std::string("the fixture did not compile") : r.notes[0];
            ok = false;
            break;
        }
        for (size_t i = 0; i < list.arena().size; ++i) {
            h ^= list.arena().data[i];
            h *= 0x100000001b3ull;
        }
        if (detail && list.records().size == 1) {
            const caliper_gfx::DrawRecord& rec = list.records()[0];
            const uint32_t sizes[3] = {uint32_t(sizeof(caliper_gfx::shaders::Frame)),
                                       uint32_t(sizeof(caliper_gfx::shaders::PrimParams)),
                                       uint32_t(sizeof(caliper_gfx::shaders::Cmap))};
            char t[64];
            std::snprintf(t, sizeof t, "%sview%d", detail->empty() ? "" : " ", view);
            *detail += t;
            for (int k = 0; k < 3; ++k) {
                uint64_t bh = 0xcbf29ce484222325ull;
                for (uint32_t i = 0; i < sizes[k]; ++i) {
                    bh ^= list.arena().data[rec.dynamic_offsets[k] + i];
                    bh *= 0x100000001b3ull;
                }
                std::snprintf(t, sizeof t, " %s=%016llx", k == 0 ? "frame" : k == 1 ? "draw" : "cmap", (unsigned long long)bh);
                *detail += t;
            }
            *detail += " block=";
            for (int i = 0; i < 48; ++i) {
                uint32_t bits;
                std::memcpy(&bits, list.arena().data + rec.dynamic_offsets[1] + 4 * i, 4);
                std::snprintf(t, sizeof t, "%s%08x", i ? "," : "", bits);
                *detail += t;
            }
        }
    }
    if (pos) b.release(pos);
    if (val) b.release(val);
    if (color) b.release(color);
    if (depth) b.release(depth);
    *hash = h;
    return ok;
}

}  // namespace caliper_scene
