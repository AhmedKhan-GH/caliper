// Shared fixtures for library piece tests. CPU scenes supply camera and label data.
// Saved images and picks come from the original renderer. Piece output keeps the
// same color tolerances, with exact interiors and a one-level allowance at edges.
#pragma once

#include <doctest/doctest.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "../draw_parity/parity_draw.h"
#include "draw/draw_runtime.h"
#include "gfx/colormap_keys.h"
#include "gfx/metal_backend.h"
#include "producer/caliper_producer.h"
#include "producer/caliper_producer_derive.h"
#include "scene/camera.h"

#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/spec/spec.h>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-function"
#include <stb_image_write.h>
#include "../match_reference.h"
#pragma clang diagnostic pop

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace match_rig {

namespace cd = caliper_draw;
namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;
namespace dp = draw_parity;
namespace gfx = caliper_gfx;
namespace sc = caliper_scene;

inline constexpr std::uint32_t kPiecePick = 1u << 24;
// The divisor of a capacity case's count, from CALIPER_CAPACITY_FRACTION (1 for the full count), for
// a run at a fraction of the count that measures the memory first.
inline int capacity_fraction() {
    const char* e = std::getenv("CALIPER_CAPACITY_FRACTION");
    const int f = e ? std::atoi(e) : 1;
    return f < 1 ? 1 : f;
}

inline constexpr std::uint32_t kClear = dp::kBackground;

// The device and the backend of the graphics layer on it, or nothing without a usable device.
struct Metal {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    std::unique_ptr<gfx::Backend> backend;
    std::string why;

    Metal() {
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            why = "no Metal device";
            return;
        }
        if (!MPSSupportsMTLDevice(device)) {
            why = "no Metal device that MPS supports";
            return;
        }
        queue = [device newCommandQueue];
        backend = gfx::make_metal_backend((__bridge void*)device, (__bridge void*)queue);
        if (!backend) why = "the graphics layer made no Metal backend";
    }
};

// Reads one texture back as bytes, top row first.
inline std::vector<std::uint8_t> read_back(gfx::Backend& g, gfx::Texture t, std::uint32_t w, std::uint32_t h) {
    const gfx::Readback rb = g.read_texture(t, 0, 0, w, h);
    REQUIRE_MESSAGE(bool(rb), g.last_error());
    std::vector<std::uint8_t> out;
    REQUIRE(g.wait(g.last_use(t)));
    REQUIRE(g.take_readback(rb, &out));
    return out;
}

inline std::uint32_t pixel_of(const std::vector<std::uint8_t>& img, std::uint32_t size, int x, int y) {
    std::uint32_t v = 0;
    std::memcpy(&v, img.data() + (std::size_t(y) * size + std::size_t(x)) * 4, 4);
    return v;
}

inline int channel_gap(std::uint32_t a, std::uint32_t b) {
    int d = 0;
    for (int k = 0; k < 4; ++k) d = std::max(d, std::abs(int((a >> (8 * k)) & 0xffu) - int((b >> (8 * k)) & 0xffu)));
    return d;
}

// The identity of pixel (x, y) in an RG32Uint ID image: 0 for the background, else item + 1.
inline std::uint64_t identity_of(const std::vector<std::uint8_t>& ids, std::uint32_t size, int x, int y) {
    std::uint32_t pick = 0, item = 0;
    const std::uint8_t* p = ids.data() + (std::size_t(y) * size + std::size_t(x)) * 8;
    std::memcpy(&pick, p, 4);
    std::memcpy(&item, p + 4, 4);
    return pick == 0 ? 0 : std::uint64_t(item) + 1;
}

inline bool is_interior(const std::vector<std::uint8_t>& ids, std::uint32_t size, int x, int y) {
    const std::uint64_t own = identity_of(ids, size, x, y);
    const int s = int(size);
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            if (identity_of(ids, size, std::clamp(x + dx, 0, s - 1), std::clamp(y + dy, 0, s - 1)) != own) return false;
    return true;
}

// Writes an RGBA8 image as a PNG into the folder that the environment variable names, if any.
inline void capture(const char* env, const std::string& name, std::uint32_t size, const std::vector<std::uint8_t>& rgba) {
    const char* dir = std::getenv(env);
    if (dir == nullptr || *dir == 0) return;
    const std::string path = std::string(dir) + "/" + name;
    CHECK(stbi_write_png(path.c_str(), int(size), int(size), 4, rgba.data(), int(size) * 4) != 0);
    MESSAGE("capture written: ", path);
}

// The difference of two images scaled up 32 times, opaque, for a capture.
inline std::vector<std::uint8_t> difference(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
    std::vector<std::uint8_t> d(a.size());
    for (std::size_t i = 0; i < d.size(); ++i)
        d[i] = (i % 4 == 3) ? 255 : std::uint8_t(std::min(255, std::abs(int(a[i]) - int(b[i])) * 32));
    return d;
}

// A piece of the built-in library, version 1, with its files and its hash.
inline bool load_piece(const std::string& name, cs::Piece* piece, std::string* why) {
    std::string_view manifest;
    std::map<std::string, std::string> files;
    const std::string prefix = "pieces/" + name + "/1/";
    for (const cp::LibraryFile& f : cp::library_files()) {
        const std::string_view path = f.path;
        if (path.substr(0, prefix.size()) != prefix) continue;
        const std::string file(path.substr(prefix.size()));
        if (file == "piece.toml") manifest = f.text;
        else files[file] = std::string(f.text);
    }
    if (manifest.empty()) return (*why = "the library has no " + prefix + "piece.toml", false);
    cs::PieceParse parse = cs::piece_from_toml(manifest);
    if (!parse.ok()) return (*why = prefix + "piece.toml does not parse", false);
    *piece = std::move(parse.piece);
    for (const char* file : {"common.wgsl", "derive.wgsl", "draw.wgsl"})
        if (files.count(file)) piece->files[file] = files[file];
    piece->hash = cp::piece_hash(*piece);
    return true;
}

// The checks that piece.check runs before a compiler: the manifest, each file, the imports, and the
// composition of the derive half and of each pass of the draw half.
inline void check_piece(const cs::Piece& piece) {
    const cp::Verdict m = cp::check_piece_manifest(piece);
    CHECK_MESSAGE(m.ok(), m.code, " ", m.where, " ", m.message);
    for (const auto& file : piece.files) {
        const cp::Verdict v = cp::check_piece_source(file.second, 64 * 1024);
        CHECK_MESSAGE(v.ok(), file.first, ": ", v.code, " ", v.message);
    }
    const cp::Resolved r = cp::resolve_imports(piece);
    REQUIRE_MESSAGE(r.verdict.ok(), r.verdict.code, " ", r.verdict.message);
    const cp::Composed d = cp::compose_derive(piece, r);
    CHECK_MESSAGE(d.verdict.ok(), d.verdict.code, " ", d.verdict.message);
    for (std::size_t k = 0; k < piece.draw->passes.size(); ++k) {
        const cp::Composed c = cp::compose_draw(piece, r, k);
        CHECK_MESSAGE(c.verdict.ok(), "pass ", k, ": ", c.verdict.code, " ", c.verdict.message);
    }
}

// A tensor of the script that a derive half reads: its input name, dtype, shape and bytes.
struct HostTensor {
    std::string var;
    std::string dtype = "f32";
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> bytes;

    template <class T>
    static HostTensor of(const std::string& var, const char* dtype, std::vector<std::int64_t> shape, const std::vector<T>& v) {
        HostTensor t;
        t.var = var;
        t.dtype = dtype;
        t.shape = std::move(shape);
        t.bytes.resize(v.size() * sizeof(T));
        if (!v.empty()) std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
        return t;
    }
};

// One published resource after a run of the derive half: its valid elements and their bytes.
struct Published {
    std::string id;
    std::vector<std::uint8_t> bytes;
    std::uint32_t count = 0;
    std::uint32_t width = 0, height = 0;    // a texture's version
    std::string format;                     // a texture's format, empty for rgba8unorm
    std::optional<std::uint32_t> elements;  // a buffer of narrow elements
};

// Runs the derive half of `piece` once, as the script's producer runtime runs it, on the tensors
// with the params (the manifest's defaults where `params` has no value), and gives each published
// resource. The resolved files go as the app sends them in PROGRAM.
inline std::vector<Published> run_derive(id<MTLDevice> device, const cs::Piece& piece,
                                         const std::vector<HostTensor>& tensors,
                                         const std::map<std::string, double>& params) {
    std::vector<Published> out;
    caliper_producer* rt = nullptr;
    caliper_producer_error e{};
    REQUIRE(caliper_producer_create((__bridge void*)device, &rt, &e) == CALIPER_PRODUCER_OK);
    struct Guard {
        caliper_producer* rt;
        caliper_producer_error* e;
        ~Guard() {
            caliper_producer_error_free(e);
            caliper_producer_destroy(rt);
        }
    } guard{rt, &e};
    std::vector<id<MTLBuffer>> keep;
    std::vector<caliper_producer_input> inputs;
    std::vector<const char*> input_types;
    std::vector<std::uint8_t> unbound;
    for (const cs::PieceInput& in : piece.inputs) {
        const HostTensor* t = nullptr;
        for (const HostTensor& h : tensors)
            if (h.var == in.name) t = &h;
        if (t == nullptr && in.optional) {   // an optional input that the case leaves unbound
            inputs.push_back({in.name.c_str(), 0});
            input_types.push_back(in.dtype.c_str());
            unbound.push_back(1);
            continue;
        }
        REQUIRE_MESSAGE(t != nullptr, "no tensor for the input ", in.name);
        unbound.push_back(0);
        id<MTLBuffer> b = [device newBufferWithLength:std::max<std::size_t>((t->bytes.size() + 15) / 16 * 16, 16)
                                              options:MTLResourceStorageModeShared];
        if (!t->bytes.empty()) std::memcpy([b contents], t->bytes.data(), t->bytes.size());
        keep.push_back(b);
        caliper_producer_tensor ct{};
        ct.name = t->var.c_str();
        ct.buffer = (__bridge void*)b;
        ct.byte_offset = 0;
        ct.byte_size = t->bytes.size();
        ct.dtype = t->dtype == "i32"   ? CALIPER_PRODUCER_I32
                   : t->dtype == "u32" ? CALIPER_PRODUCER_U32
                   : t->dtype == "u8"  ? CALIPER_PRODUCER_U8
                   : t->dtype == "u16" ? CALIPER_PRODUCER_U16
                                       : CALIPER_PRODUCER_F32;
        ct.ndim = std::uint32_t(t->shape.size());
        for (std::size_t k = 0; k < t->shape.size(); ++k) ct.shape[k] = t->shape[k];
        std::uint32_t handle = 0;
        REQUIRE_MESSAGE(caliper_producer_import(rt, &ct, &handle, &e) == CALIPER_PRODUCER_OK,
                        in.name, ": ", std::string(e.message ? e.message : ""));
        inputs.push_back({in.name.c_str(), handle});
        input_types.push_back(in.dtype.c_str());
    }
    std::vector<caliper_producer_resource_desc> resources;
    std::vector<std::vector<std::string>> field_text(piece.resources.size());
    std::vector<std::vector<const char*>> fields(piece.resources.size());
    for (std::size_t i = 0; i < piece.resources.size(); ++i) {
        const cs::Resource& r = piece.resources[i];
        caliper_producer_resource_desc d{};
        d.id = r.id.c_str();
        d.kind = r.kind == cs::ResourceKind::texture ? CALIPER_PRODUCER_RESOURCE_TEXTURE : CALIPER_PRODUCER_RESOURCE_BUFFER;
        d.element = r.layout.element ? r.layout.element->c_str() : nullptr;
        d.struct_name = r.layout.type_name ? r.layout.type_name->c_str() : nullptr;
        for (const cs::Field& f : r.layout.fields) field_text[i].push_back(f.name + ": " + f.type);
        for (const std::string& f : field_text[i]) fields[i].push_back(f.c_str());
        d.fields = fields[i].empty() ? nullptr : fields[i].data();
        d.field_count = fields[i].size();
        d.count = r.count;
        d.width = r.width;
        d.height = r.height;
        d.format = r.format ? r.format->c_str() : nullptr;
        d.texels = r.texels.value_or(0);
        d.counted = r.counted;
        d.publish = r.publish;
        d.clear = r.clear;
        resources.push_back(d);
    }
    std::vector<caliper_producer_derive_pass> passes;
    std::vector<std::vector<const char*>> writes(piece.derive->passes.size());
    for (std::size_t i = 0; i < piece.derive->passes.size(); ++i) {
        const cs::DerivePass& dp_ = piece.derive->passes[i];
        for (const std::string& w : dp_.writes) writes[i].push_back(w.c_str());
        caliper_producer_derive_pass p{};
        p.entry = dp_.entry.c_str();
        for (int k = 0; k < 3; ++k) p.workgroups[k] = dp_.workgroups[std::size_t(k)];
        p.writes = writes[i].empty() ? nullptr : writes[i].data();
        p.write_count = writes[i].size();
        passes.push_back(p);
    }
    std::vector<caliper_producer_param> values;
    std::vector<const char*> param_types;
    for (const cs::PieceParam& p : piece.params) {
        const auto it = params.find(p.name);
        values.push_back({p.name.c_str(), it == params.end() ? p.default_value : it->second});
        param_types.push_back(cs::param_type_name(p.type));
    }
    const cp::Resolved resolved = cp::resolve_imports(piece);
    REQUIRE_MESSAGE(resolved.verdict.ok(), resolved.verdict.message);
    auto file = [&](const char* name) {
        const auto it = resolved.files.find(name);
        return it == resolved.files.end() ? std::string() : it->second;
    };
    const std::string wgsl = file("derive.wgsl"), common = resolved.lib_text + file("common.wgsl");
    caliper_producer_derive_desc d{};
    d.id = piece.name.c_str();
    d.input_unbound = unbound.data();
    d.piece_hash = piece.hash.c_str();
    d.wgsl = wgsl.c_str();
    d.wgsl_size = wgsl.size();
    d.common = common.empty() ? nullptr : common.c_str();
    d.common_size = common.size();
    d.inputs = inputs.data();
    d.input_count = inputs.size();
    d.input_types = input_types.data();
    d.resources = resources.data();
    d.resource_count = resources.size();
    d.passes = passes.data();
    d.pass_count = passes.size();
    d.params = values.data();
    d.param_types = param_types.data();
    d.param_count = values.size();
    d.time_limit_ms = std::min(50.0, piece.derive->time_ms.value_or(4.0) * 10.0);   // a test machine under load, within the producer's 50 ms
    d.memory_limit_bytes = std::uint64_t(piece.derive->memory_mb.value_or(64)) << 20;
    std::uint32_t prog = 0;
    REQUIRE_MESSAGE(caliper_producer_compile_derive(rt, &d, &prog, &e) == CALIPER_PRODUCER_OK,
                    std::string(e.code ? e.code : ""), ": ", std::string(e.message ? e.message : ""));
    caliper_producer_frame frame{1, 0.0, 1};
    caliper_producer_pass pass{};
    REQUIRE_MESSAGE(caliper_producer_run(rt, &prog, 1, &frame, 0, &pass, &e) == CALIPER_PRODUCER_OK,
                    std::string(e.code ? e.code : ""), ": ", std::string(e.message ? e.message : ""));
    REQUIRE(pass.ran == 1);
    for (const cs::Resource& r : piece.resources) {
        if (!r.publish) continue;
        caliper_producer_output o{};
        REQUIRE_MESSAGE(caliper_producer_read_resource(rt, prog, r.id.c_str(), &o, &e) == CALIPER_PRODUCER_OK,
                        r.id, ": ", std::string(e.message ? e.message : ""));
        Published p;
        p.id = r.id;
        p.count = o.count;
        p.width = o.width;
        p.height = o.height;
        if (r.kind == cs::ResourceKind::texture && r.format && *r.format != "rgba8unorm") p.format = *r.format;
        if (o.narrow) p.elements = o.elements;
        p.bytes.resize(o.byte_size);
        if (o.byte_size) {
            const auto* base = static_cast<const std::uint8_t*>([(__bridge id<MTLBuffer>)o.buffer contents]);
            std::memcpy(p.bytes.data(), base + o.byte_offset, o.byte_size);
        }
        out.push_back(std::move(p));
    }
    return out;
}

// Host arrays of one batch in one buffer, each at a multiple of 256 bytes.
struct Upload {
    struct Part {
        sc::Role role;
        std::uint64_t offset;
        std::uint32_t count, stride;
        gfx::VertexFormat format;
    };
    std::vector<std::uint8_t> bytes;
    std::vector<Part> parts;

    template <class T>
    void add(sc::Role role, const std::vector<T>& v, std::uint32_t count, std::uint32_t stride, gfx::VertexFormat format) {
        const std::uint64_t off = (bytes.size() + 255) / 256 * 256;
        bytes.resize(off + std::max<std::size_t>(v.size() * sizeof(T), 4), 0);
        if (!v.empty()) std::memcpy(bytes.data() + off, v.data(), v.size() * sizeof(T));
        parts.push_back({role, off, count, stride, format});
    }
};

// A CPU scene fixes the camera and label anchors for a saved reference image.
struct ReferenceScene {
    std::uint32_t size;
    std::string adapter;
    sc::SceneTree tree;
    sc::ViewCamera camera;
    std::vector<std::vector<std::uint8_t>> cpu_copies;
    std::vector<std::uint8_t> image;
    std::string reference;

    ReferenceScene(gfx::Backend& backend, std::uint32_t side, sc::Space space)
        : size(side), adapter(backend.caps().adapter ? backend.caps().adapter : ""), tree(space) {
        if (space == sc::Space::k2D) camera = sc::PanZoom{};
        else camera = sc::Orbit{};
    }

    void set(const std::string& path, sc::Batch batch, const Upload& up) {
        cpu_copies.push_back(up.bytes);
        const auto* cpu = cpu_copies.back().data();
        for (const Upload::Part& p : up.parts) {
            sc::BatchArray a;
            a.role = p.role;
            a.cpu = cpu + p.offset;
            a.count = p.count;
            a.stride = p.stride;
            a.format = p.format;
            batch.arrays.push_back(a);
        }
        batch.seq = 1;
        REQUIRE(tree.set_batch(tree.ensure(path), batch));
    }

    void load() {
        tree.update();
        sc::fit(camera, tree.scene_bounds(), 1.0);
        image = match_reference::picture("rig", size, size, nullptr, adapter.c_str());
        reference = match_reference::last_name("rig");
    }

    cd::FrameInputs frame_inputs() const {
        const sc::CameraMatrices cm = sc::camera_matrices(camera, tree, size, size);
        REQUIRE(cm.ok);
        const sc::Mat4 clip = cm.proj * cm.view;
        cd::FrameInputs in;
        for (int i = 0; i < 16; ++i) {
            in.clip_from_world[std::size_t(i)] = float(clip.m[i]);
            in.view_from_world[std::size_t(i)] = float(cm.view.m[i]);
        }
        in.eye = {float(cm.eye.x), float(cm.eye.y), float(cm.eye.z), cm.perspective ? 1.0f : 0.0f};
        in.viewport = {0, 0, float(size), float(size)};
        in.pixel_ratio = 1.0f;
        in.cursor_step = 1;
        in.latest_step = 1;
        in.live = true;
        return in;
    }
};

// The piece's draw: one frame of the program in a DrawRuntime into a host target of size x size.
struct PieceDraw {
    gfx::Backend& g;
    cd::DrawRuntime rt;
    dp::Target target;
    std::vector<std::uint8_t> color, ids;

    PieceDraw(gfx::Backend& backend, std::uint32_t size) : g(backend), rt(backend, false), target(backend, size) {
        REQUIRE_MESSAGE(target.ok, g.last_error());
    }

    void install(const cs::Piece& piece, const std::string& program) {
        cd::Instance instance;
        instance.program = program;
        instance.panel = program;
        const cd::Verdict v = rt.install(piece, instance, kPiecePick);
        REQUIRE_MESSAGE(v.ok(), v.code << " " << v.message);
    }

    void set(const std::string& resource, const std::vector<std::uint8_t>& bytes, std::uint32_t count,
             const Published* from = nullptr) {
        cd::ResourceVersion version;
        version.resource = resource;
        version.step = 1;
        version.count = count;
        if (from) {
            version.width = from->width;
            version.height = from->height;
            version.format = from->format;
            version.elements = from->elements;
        }
        version.bytes = bytes.data();
        version.size = bytes.size();
        const cd::Verdict v = rt.set_resource(version);
        REQUIRE_MESSAGE(v.ok(), resource << ": " << v.code << " " << v.message);
    }

    // Draws one frame; false when the program drew nothing (no instance, as for an empty input).
    bool draw(const cd::FrameInputs& in) {
        std::vector<gfx::Serial> serials;
        std::string why;
        const bool drawn = target.frame(rt, in, &serials, &why);
        for (gfx::Serial s : serials) REQUIRE_MESSAGE(g.wait(s), g.last_error());
        color = read_back(g, target.host.color, target.host.width, target.host.height);
        ids = read_back(g, target.host.id, target.host.width, target.host.height);
        if (drawn)
            CHECK_MESSAGE(rt.status().state == cd::DrawState::running,
                          rt.status().error.code << " " << rt.status().error.message);
        return drawn;
    }
};

// The comparison of the piece's image with the built-in's under the tolerances, with every
// difference measured. With `color_edges`, a pixel whose 3 x 3 neighborhood in the built-in's
// image holds more than one color is an edge pixel too: a mesh with flat faces lights each face
// from the derivatives of its position, which a GPU takes over a quad of 2 x 2 pixels, thus a pixel
// at the edge of two faces of one item may take the light of either face.
struct Comparison {
    std::size_t interior = 0, lit_interior = 0, edge = 0, failed = 0, differ = 0;
    std::size_t lit = 0;                  // pixels that either image draws
    std::size_t interior_failed = 0;      // interior pixels that differ
    std::size_t edge_far = 0;             // edge pixels over 1 level from every built-in neighbor
    int max_interior = 0, max_edge = 0;   // the largest channel gap inside, and at an edge to the nearest neighbor
    std::string first;
};

inline bool uniform_color(const std::vector<std::uint8_t>& img, std::uint32_t size, int x, int y) {
    const std::uint32_t own = pixel_of(img, size, x, y);
    const int s = int(size);
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            if (pixel_of(img, size, std::clamp(x + dx, 0, s - 1), std::clamp(y + dy, 0, s - 1)) != own) return false;
    return true;
}

inline Comparison compare(const std::vector<std::uint8_t>& piece, const std::vector<std::uint8_t>& ids,
                          const std::vector<std::uint8_t>& builtin, std::uint32_t size, bool color_edges = false) {
    REQUIRE(piece.size() == builtin.size());
    REQUIRE(ids.size() == std::size_t(size) * size * 8);
    Comparison c;
    const int S = int(size);
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x) {
            const std::uint32_t pc = pixel_of(piece, size, x, y), bc = pixel_of(builtin, size, x, y);
            c.differ += pc != bc;
            c.lit += pc != kClear || bc != kClear;
            bool ok;
            const bool inside = is_interior(ids, size, x, y) && (!color_edges || uniform_color(builtin, size, x, y));
            if (inside) {
                ++c.interior;
                c.lit_interior += pc != kClear;
                c.max_interior = std::max(c.max_interior, channel_gap(pc, bc));
                ok = pc == bc;
                c.interior_failed += !ok;
            } else {
                ++c.edge;
                int best = 255;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                        best = std::min(best, channel_gap(pc, pixel_of(builtin, size, std::clamp(x + dx, 0, S - 1),
                                                                       std::clamp(y + dy, 0, S - 1))));
                c.max_edge = std::max(c.max_edge, best);
                c.edge_far += best > 1;
                ok = best <= 1;
            }
            if (!ok) {
                ++c.failed;
                if (c.first.empty()) {
                    char buf[160];
                    std::snprintf(buf, sizeof buf, "first at (%d, %d): piece %08x, built-in %08x, %s", x, y, pc, bc,
                                  inside ? "interior" : "edge");
                    c.first = buf;
                }
            }
        }
    return c;
}

// The first pixel, in row order, that is interior in the piece's ID image and shows `item`.
inline std::optional<std::array<int, 2>> pixel_of_item(const std::vector<std::uint8_t>& ids, std::uint32_t size,
                                                       std::uint64_t item) {
    for (int y = 1; y + 1 < int(size); ++y)
        for (int x = 1; x + 1 < int(size); ++x)
            if (identity_of(ids, size, x, y) == item + 1 && is_interior(ids, size, x, y)) return std::array<int, 2>{x, y};
    return std::nullopt;
}

// The picks of `items`: at an interior pixel of each in the piece's ID image, the built-in's pick
// names the same item, and the piece's item map gives it as the field `index`. Returns the picks
// that agree; an item that shows no interior pixel is reported and not counted.
inline int check_picks(ReferenceScene& b, const PieceDraw& p, const std::vector<std::uint64_t>& items,
                       sc::Primitive primitive) {
    const auto& probes = match_reference::metadata(b.reference).at("picks");
    int agree = 0;
    for (std::uint64_t item : items) {
        CAPTURE(item);
        const auto probe = std::find_if(probes.begin(), probes.end(), [&](const auto& value) {
            return value.at("item").template get<std::uint64_t>() == item;
        });
        if (probe == probes.end()) {
            MESSAGE("item ", item, " has no saved interior pick");
            continue;
        }
        const int x = probe->at("x").get<int>(), y = probe->at("y").get<int>();
        CHECK(probe->at("primitive").get<int>() == int(primitive));
        const std::uint64_t id = identity_of(p.ids, b.size, x, y);
        CHECK_MESSAGE(id == item + 1, "pixel (", x, ", ", y, "): saved item ", item, ", piece identity ", id);
        const std::optional<cd::PickedItem> picked = p.rt.pick(std::uint32_t(item));
        REQUIRE(picked.has_value());
        REQUIRE(picked->fields.size() == 1);
        CHECK(picked->fields[0].first == "index");
        CHECK(picked->fields[0].second == double(item));
        agree += id == item + 1 && picked->fields[0].second == double(item);
    }
    return agree;
}

}  // namespace match_rig
