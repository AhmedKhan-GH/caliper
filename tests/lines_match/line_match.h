// line_match.h: the shared parts of the matches of the library pieces lines@1, arrows@1 and
// path@1 with the built-in lines and arrows (design/programmable-graphics.md, Library Pieces).
//
// Each case draws one scene twice on Metal, offscreen and without a window. The scene core draws
// the built-in batches (SceneCompiler over BackendResources), and the pieces draw the same data
// end to end: the producer runtime runs each piece's derive half over the same arrays as tensors,
// and the draw runtime draws the published resources with the camera matrix of the built-in draw.
// The comparison holds the tolerances of the points match: an interior pixel, whose 3 x 3
// neighborhood shows one item in the piece's ID image, has the same color in both images, and an
// edge pixel has a piece color within 1 level of a built-in color in its 3 x 3 neighborhood. A pick
// of the built-in draw at a pixel that shows an element of the piece must name that element, and
// the piece's item map must give its fields.
//
// With CALIPER_LINE_MATCH_CAPTURE_DIR set, a case that names a capture writes <name>-builtin.png,
// <name>-piece.png and <name>-diff.png (the difference scaled up 32 times) there.
#pragma once

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "../gpu_skip.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "../draw_parity/line_pieces.h"
#include "../draw_parity/parity_draw.h"
#include "draw/draw_runtime.h"
#include "gfx/colormap_keys.h"
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "producer/caliper_producer.h"
#include "producer/caliper_producer_derive.h"
#include "scene/backend_resources.h"
#include "scene/camera.h"
#include "scene/compile.h"
#include "scene/pick.h"

#include <caliper_builder/programs/pieces.h>
#include <caliper_builder/session/piece_commands.h>
#include <caliper_builder/spec/spec.h>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-function"
#include <stb_image_write.h>
#pragma clang diagnostic pop

#include <algorithm>
#include <cctype>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace line_match {

namespace cd = caliper_draw;
namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;
namespace dp = draw_parity;
namespace gfx = caliper_gfx;
namespace sc = caliper_scene;
namespace sh = caliper_gfx::shaders;
using Json = caliper_net::Json;

constexpr std::uint32_t kSize = 256;              // the target, square
constexpr std::uint32_t kPick = 1u << 24;         // the pick id of the first program
constexpr std::uint32_t kClear = dp::kBackground;
constexpr std::uint32_t kNone = 0xffffffffu;

// ---- the device -----------------------------------------------------------------------------

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
    std::vector<std::uint8_t> out;
    @autoreleasepool {
        const gfx::Readback rb = g.read_texture(t, 0, 0, w, h);
        REQUIRE_MESSAGE(bool(rb), g.last_error());
        REQUIRE(g.wait(g.last_use(t)));
        REQUIRE(g.take_readback(rb, &out));
    }
    return out;
}

// ---- pixels ---------------------------------------------------------------------------------

inline std::uint32_t pixel_of(const std::vector<std::uint8_t>& img, int x, int y) {
    std::uint32_t v = 0;
    std::memcpy(&v, img.data() + (std::size_t(y) * kSize + std::size_t(x)) * 4, 4);
    return v;
}

inline int channel_gap(std::uint32_t a, std::uint32_t b) {
    int d = 0;
    for (int k = 0; k < 4; ++k) d = std::max(d, std::abs(int((a >> (8 * k)) & 0xffu) - int((b >> (8 * k)) & 0xffu)));
    return d;
}

// The identity of pixel (x, y) in an RG32Uint ID image: 0 for the background, else a number for
// the program and the item.
inline std::uint64_t identity_of(const std::vector<std::uint8_t>& ids, int x, int y) {
    std::uint32_t pick = 0, item = 0;
    const std::uint8_t* p = ids.data() + (std::size_t(y) * kSize + std::size_t(x)) * 8;
    std::memcpy(&pick, p, 4);
    std::memcpy(&item, p + 4, 4);
    return pick == 0 ? 0 : (std::uint64_t(pick) << 32 | item) + 1;
}

inline bool is_interior(const std::vector<std::uint8_t>& ids, int x, int y) {
    const std::uint64_t own = identity_of(ids, x, y);
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            if (identity_of(ids, std::clamp(x + dx, 0, int(kSize) - 1), std::clamp(y + dy, 0, int(kSize) - 1)) != own)
                return false;
    return true;
}

inline void capture(const std::string& name, const std::vector<std::uint8_t>& rgba) {
    const char* dir = std::getenv("CALIPER_LINE_MATCH_CAPTURE_DIR");
    if (dir == nullptr || *dir == 0) return;
    const std::string path = std::string(dir) + "/" + name;
    CHECK(stbi_write_png(path.c_str(), int(kSize), int(kSize), 4, rgba.data(), int(kSize) * 4) != 0);
    MESSAGE("capture written: ", path);
}

// The comparison of the plan's tolerances.
struct Comparison {
    std::size_t interior = 0, lit_interior = 0, edge = 0, failed = 0, differ = 0;
    int max_interior_gap = 0, max_edge_gap = 0;   // the largest gap of an interior pixel, of an edge pixel to its best neighbor
    std::string first;
};

inline Comparison compare(const std::vector<std::uint8_t>& builtin, const std::vector<std::uint8_t>& piece,
                          const std::vector<std::uint8_t>& ids) {
    Comparison c;
    const int S = int(kSize);
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x) {
            const std::uint32_t pc = pixel_of(piece, x, y), bc = pixel_of(builtin, x, y);
            c.differ += pc != bc;
            bool ok;
            const bool inside = is_interior(ids, x, y);
            if (inside) {
                ++c.interior;
                c.lit_interior += pc != kClear;
                c.max_interior_gap = std::max(c.max_interior_gap, channel_gap(pc, bc));
                ok = pc == bc;
            } else {
                ++c.edge;
                int best = 256;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                        best = std::min(best, channel_gap(pc, pixel_of(builtin, std::clamp(x + dx, 0, S - 1),
                                                                         std::clamp(y + dy, 0, S - 1))));
                c.max_edge_gap = std::max(c.max_edge_gap, best);
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

inline void capture_pair(const std::string& name, const std::vector<std::uint8_t>& builtin,
                         const std::vector<std::uint8_t>& piece) {
    capture(name + "-builtin.png", builtin);
    capture(name + "-piece.png", piece);
    std::vector<std::uint8_t> diff(piece.size());
    for (std::size_t i = 0; i < diff.size(); ++i)
        diff[i] = (i % 4 == 3) ? 255 : std::uint8_t(std::min(255, std::abs(int(piece[i]) - int(builtin[i])) * 32));
    capture(name + "-diff.png", diff);
}

// ---- the library piece and piece.check -------------------------------------------------------

// The piece <name>@1 of the built-in library, with its files and its hash.
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
    if (!parse.ok()) {
        *why = prefix + "piece.toml does not parse";
        for (const cs::Diagnostic& d : parse.diagnostics) *why += "; " + d.code + " " + d.message;
        return false;
    }
    *piece = std::move(parse.piece);
    for (const char* f : {"common.wgsl", "derive.wgsl", "draw.wgsl"})
        if (files.count(f)) piece->files[f] = files[f];
    piece->hash = cp::piece_hash(*piece);
    return true;
}

// piece.check on the piece: the manifest, the files, the hash, and each pass of both halves
// composed and compiled. Every pass must compile; a build without the composers or the compiler
// fails here, because the match needs both.
inline void check_piece(const cs::Piece& piece) {
    const nlohmann::json r = caliper_builder::check_piece_json(cs::to_json(piece), false, nullptr);
    std::string messages;
    for (const nlohmann::json& m : r["messages"]) messages += m.dump() + "\n";
    CHECK_MESSAGE(r["ok"] == true, messages);
    CHECK_MESSAGE(r["messages"].empty(), messages);
    std::size_t derive = 0, draw = 0;
    for (const nlohmann::json& p : r["passes"]) {
        CHECK_MESSAGE(p["ok"] == true, p.dump());
        (p["half"] == "derive" ? derive : draw) += 1;
    }
    CHECK(derive == (piece.derive ? piece.derive->passes.size() : 0));
    CHECK(draw == (piece.draw ? piece.draw->passes.size() : 0));
    CHECK(r["hash"] == piece.hash);
    MESSAGE("piece.check ", piece.name, "@", piece.version, ": ", derive, " derive and ", draw,
            " draw passes compiled, hash ", piece.hash);
}

// The canonical manifest and the hash that the pixel comparison of the app and the browser module
// carries for the piece (tests/draw_parity/line_pieces.h) are those of the library's piece, and
// they decode there as a viewer decodes PROGRAM to the same piece.
inline void check_pinned(const cs::Piece& piece, std::string_view manifest, std::string_view hash) {
    const std::string canonical = cp::canonical_manifest(piece);
    CHECK_MESSAGE(canonical == manifest, "tests/draw_parity/line_pieces.h holds another manifest of ", piece.name,
                  "@1; the library's is\n", canonical);
    CHECK_MESSAGE(piece.hash == hash, "tests/draw_parity/line_pieces.h holds another hash of ", piece.name,
                  "@1; the library's is ", piece.hash);
    cs::Piece decoded;
    std::string why;
    REQUIRE_MESSAGE(dp::line_piece(piece.name, manifest, hash, &decoded, &why), why);
    CHECK(cp::piece_hash(decoded) == piece.hash);
    CHECK(cp::canonical_manifest(decoded) == canonical);
}

// ---- the derive half on the producer runtime ------------------------------------------------

// A tensor of the script: its input name, bytes, dtype and shape.
struct Tensor {
    std::string var;
    std::vector<std::uint8_t> bytes;
    std::uint32_t dtype = CALIPER_PRODUCER_F32;
    std::vector<std::int64_t> shape;
};

template <typename T>
Tensor tensor(const std::string& var, const std::vector<T>& v, std::vector<std::int64_t> shape) {
    static_assert(sizeof(T) == 4, "a tensor of 4-byte elements");
    Tensor t;
    t.var = var;
    t.bytes.resize(v.size() * 4);
    if (!v.empty()) std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
    t.dtype = std::is_same<T, float>::value ? CALIPER_PRODUCER_F32
              : std::is_same<T, std::int32_t>::value ? CALIPER_PRODUCER_I32
                                                     : CALIPER_PRODUCER_U32;
    t.shape = std::move(shape);
    return t;
}

// One published resource after a run: its valid elements and their bytes.
struct Published {
    std::string id;
    std::uint32_t count = 0;
    std::vector<std::uint8_t> bytes;
};

// The value of each param of the piece: the given value, else its default.
inline std::vector<std::pair<std::string, double>> param_values(const cs::Piece& piece, const Json& values) {
    std::vector<std::pair<std::string, double>> out;
    for (const cs::PieceParam& p : piece.params) {
        double v = p.default_value;
        if (const Json* j = values.find(p.name); j && j->is_number()) v = j->as_double();
        out.emplace_back(p.name, v);
    }
    return out;
}

// Runs the derive half of `piece` once over `inputs`, with the params `values`, in the producer
// runtime on the default Metal device, and gives each published resource.
inline bool run_derive(id<MTLDevice> device, const cs::Piece& piece, const std::vector<Tensor>& inputs,
                       const Json& values, std::vector<Published>* out, std::string* why) {
    out->clear();
    caliper_producer* rt = nullptr;
    caliper_producer_error e{};
    struct Guard {
        caliper_producer*& rt;
        caliper_producer_error& e;
        ~Guard() {
            caliper_producer_error_free(&e);
            if (rt) caliper_producer_destroy(rt);
        }
    } guard{rt, e};
    auto fail = [&](const std::string& what) {
        *why = what + ": " + (e.code ? e.code : "") + " " + (e.message ? e.message : "");
        return false;
    };
    if (caliper_producer_create((__bridge void*)device, &rt, &e) != CALIPER_PRODUCER_OK) return fail("create");

    // The tensors, each in a shared buffer of its own (at least 4 bytes, as a device allocation is).
    std::vector<id<MTLBuffer>> buffers;
    std::vector<caliper_producer_input> ins;
    std::vector<const char*> types;
    std::vector<std::string> names;
    names.reserve(piece.inputs.size());
    for (const cs::PieceInput& pin : piece.inputs) {
        const auto t = std::find_if(inputs.begin(), inputs.end(), [&](const Tensor& x) { return x.var == pin.name; });
        if (t == inputs.end()) return (*why = "no tensor for the input " + pin.name, false);
        id<MTLBuffer> b = [device newBufferWithLength:std::max<std::size_t>(t->bytes.size(), 4)
                                              options:MTLResourceStorageModeShared];
        if (!t->bytes.empty()) std::memcpy([b contents], t->bytes.data(), t->bytes.size());
        buffers.push_back(b);
        caliper_producer_tensor ct{};
        names.push_back("live." + pin.name);
        ct.name = names.back().c_str();
        ct.buffer = (__bridge void*)b;
        ct.byte_offset = 0;
        ct.byte_size = t->bytes.size();
        ct.dtype = t->dtype;
        ct.ndim = std::uint32_t(t->shape.size());
        for (std::size_t k = 0; k < t->shape.size(); ++k) ct.shape[k] = t->shape[k];
        std::uint32_t handle = 0;
        if (caliper_producer_import(rt, &ct, &handle, &e) != CALIPER_PRODUCER_OK) return fail("import " + pin.name);
        ins.push_back({pin.name.c_str(), handle});
        types.push_back(pin.dtype.c_str());
    }

    // The declaration of the derive half, as the script's facade makes it from PROGRAM.
    const cp::Resolved resolved = cp::resolve_imports(piece);
    if (!resolved.verdict.ok()) return (*why = "imports: " + resolved.verdict.message, false);
    auto file = [&](const char* f) {
        auto it = resolved.files.find(f);
        return it == resolved.files.end() ? std::string() : it->second;
    };
    const std::string wgsl = file("derive.wgsl");
    const std::string common = resolved.lib_text + file("common.wgsl");
    std::vector<std::vector<std::string>> field_text(piece.resources.size());
    std::vector<std::vector<const char*>> fields(piece.resources.size());
    std::vector<caliper_producer_resource_desc> res;
    for (std::size_t i = 0; i < piece.resources.size(); ++i) {
        const cs::Resource& r = piece.resources[i];
        for (const cs::Field& f : r.layout.fields) field_text[i].push_back(f.name + ": " + f.type);
        for (const std::string& f : field_text[i]) fields[i].push_back(f.c_str());
        caliper_producer_resource_desc d{};
        d.id = r.id.c_str();
        d.kind = r.kind == cs::ResourceKind::texture ? CALIPER_PRODUCER_RESOURCE_TEXTURE : CALIPER_PRODUCER_RESOURCE_BUFFER;
        d.element = r.layout.element ? r.layout.element->c_str() : nullptr;
        d.struct_name = r.layout.type_name ? r.layout.type_name->c_str() : nullptr;
        d.fields = fields[i].empty() ? nullptr : fields[i].data();
        d.field_count = fields[i].size();
        d.count = r.count;
        d.width = r.width;
        d.height = r.height;
        d.counted = r.counted;
        d.publish = r.publish;
        d.clear = r.clear;
        res.push_back(d);
    }
    std::vector<std::vector<const char*>> writes(piece.derive->passes.size());
    std::vector<caliper_producer_derive_pass> passes;
    for (std::size_t i = 0; i < piece.derive->passes.size(); ++i) {
        const cs::DerivePass& p = piece.derive->passes[i];
        for (const std::string& w : p.writes) writes[i].push_back(w.c_str());
        caliper_producer_derive_pass dp{};
        dp.entry = p.entry.c_str();
        for (int k = 0; k < 3; ++k) dp.workgroups[k] = p.workgroups[std::size_t(k)];
        dp.writes = writes[i].empty() ? nullptr : writes[i].data();
        dp.write_count = writes[i].size();
        passes.push_back(dp);
    }
    const std::vector<std::pair<std::string, double>> pv = param_values(piece, values);
    std::vector<caliper_producer_param> params;
    std::vector<const char*> param_types;
    for (std::size_t i = 0; i < pv.size(); ++i) {
        params.push_back({pv[i].first.c_str(), pv[i].second});
        const cs::ParamType t = piece.params[i].type;
        param_types.push_back(t == cs::ParamType::u32 ? "u32" : t == cs::ParamType::i32 ? "i32" : "f32");
    }
    const std::string program_id = "program." + piece.name;
    caliper_producer_derive_desc d{};
    d.id = program_id.c_str();
    d.piece_hash = piece.hash.c_str();
    d.wgsl = wgsl.data();
    d.wgsl_size = wgsl.size();
    d.common = common.empty() ? nullptr : common.data();
    d.common_size = common.size();
    d.inputs = ins.data();
    d.input_count = ins.size();
    d.input_types = types.data();
    d.resources = res.data();
    d.resource_count = res.size();
    d.passes = passes.data();
    d.pass_count = passes.size();
    d.params = params.data();
    d.param_types = param_types.data();
    d.param_count = params.size();
    d.time_limit_ms = piece.derive->time_ms.value_or(4.0);
    d.memory_limit_bytes = std::uint64_t(piece.derive->memory_mb.value_or(64)) << 20;
    std::uint32_t program = 0;
    if (caliper_producer_compile_derive(rt, &d, &program, &e) != CALIPER_PRODUCER_OK) return fail("compile_derive");
    caliper_producer_frame frame{1, 0.0, 1};
    caliper_producer_pass pass{};
    // The time limit of a run is the declared one; the run itself waits without a timeout of its own.
    if (caliper_producer_run(rt, &program, 1, &frame, 0, &pass, &e) != CALIPER_PRODUCER_OK) return fail("run");
    if (pass.ran != 1) return (*why = "the derive half did not run", false);
    for (const cs::Resource& r : piece.resources) {
        if (!r.publish) continue;
        caliper_producer_output o{};
        if (caliper_producer_read_resource(rt, program, r.id.c_str(), &o, &e) != CALIPER_PRODUCER_OK)
            return fail("read " + r.id);
        Published p;
        p.id = r.id;
        p.count = o.count;
        if (o.byte_size) {
            const auto* base = static_cast<const std::uint8_t*>([(__bridge id<MTLBuffer>)o.buffer contents]);
            p.bytes.assign(base + o.byte_offset, base + o.byte_offset + o.byte_size);
        }
        out->push_back(std::move(p));
    }
    return true;
}

inline const Published* find_published(const std::vector<Published>& v, const std::string& id) {
    for (const Published& p : v)
        if (p.id == id) return &p;
    return nullptr;
}

// ---- the built-in draw ----------------------------------------------------------------------

// A buffer of the built-in batch: its bytes and the array's layout.
struct Array {
    sc::Role role;
    std::vector<std::uint8_t> bytes;
    std::uint64_t count = 0;
    std::uint32_t stride = 0;
    gfx::VertexFormat format = gfx::VertexFormat::F32;
};

template <typename T>
Array array(sc::Role role, const std::vector<T>& v, std::uint64_t count, std::uint32_t stride, gfx::VertexFormat f) {
    Array a;
    a.role = role;
    a.bytes.resize(v.size() * sizeof(T));
    if (!v.empty()) std::memcpy(a.bytes.data(), v.data(), a.bytes.size());
    a.count = count;
    a.stride = stride;
    a.format = f;
    return a;
}

// The built-in scene of one case: its batches, its fitted orbit camera and its draw.
struct Builtin {
    gfx::Backend& g;
    std::unique_ptr<sc::BackendResources> res;
    sc::SceneTree tree{sc::Space::k3D};
    sc::ViewCamera camera = sc::Orbit{};
    sc::CompileOptions options;
    sc::SceneCompiler compiler;
    gfx::Texture color, depth;
    std::vector<gfx::Buffer> buffers;
    std::uint64_t seq = 0;

    explicit Builtin(gfx::Backend& backend, float pixel_scale = 1.0f) : g(backend) {
        res = std::make_unique<sc::BackendResources>(
            g, sh::kMetalPrograms, gfx::cmap::colormap_source(std::make_shared<gfx::cmap::KeyBlockCache>()));
        std::string err;
        REQUIRE_MESSAGE(res->init(&err), err);
        color = g.create_texture({kSize, kSize, gfx::Format::RGBA8Unorm, gfx::kRenderTarget | gfx::kCopySrcTex,
                                  "line match color"});
        depth = g.create_texture({kSize, kSize, gfx::Format::Depth32Float, gfx::kRenderTarget, "line match depth"});
        REQUIRE_MESSAGE((color && depth), g.last_error());
        options.width = options.height = kSize;
        options.pixel_scale = pixel_scale;
        options.clear_rgba = kClear;
    }
    ~Builtin() {
        res.reset();
        for (gfx::Buffer b : buffers) g.release(b);
        g.release(color);
        g.release(depth);
    }
    Builtin(const Builtin&) = delete;
    Builtin& operator=(const Builtin&) = delete;

    // Adds the batch of an entity with its arrays, each in a buffer of its own at offset 0.
    void add(const std::string& path, sc::Batch b, const std::vector<Array>& arrays) {
        for (const Array& a : arrays) {
            const std::size_t size = std::max<std::size_t>(a.bytes.size(), 4);
            gfx::Buffer buf = g.create_buffer({size, gfx::kVertex | gfx::kIndex | gfx::kStorage | gfx::kCopyDst, false,
                                               "line match array"});
            REQUIRE_MESSAGE(bool(buf), g.last_error());
            if (!a.bytes.empty()) REQUIRE(g.write_buffer(buf, 0, a.bytes.data(), a.bytes.size()));
            buffers.push_back(buf);
            sc::BatchArray ba;
            ba.role = a.role;
            ba.buffer = buf;
            ba.offset = 0;
            ba.count = a.count;
            ba.stride = a.stride;
            ba.format = a.format;
            b.arrays.push_back(ba);
        }
        b.seq = ++seq;
        std::string why;
        REQUIRE_MESSAGE(tree.set_batch(tree.ensure(path), b, false, &why), path, ": ", why);
    }

    // Fits the camera to the scene and gives the camera matrix that a program's frame gets,
    // proj * view, rounded to f32 once as the scene core rounds it.
    std::array<float, 16> fit() {
        tree.update();
        res->update_class_table(tree);
        sc::fit(camera, tree.scene_bounds(), 1.0);
        const sc::CameraMatrices cm = sc::camera_matrices(camera, tree, kSize, kSize);
        std::array<float, 16> m{};
        sc::to_f32(cm.proj * cm.view, m.data());
        return m;
    }

    std::vector<std::uint8_t> draw() {
        gfx::Frame frame;
        compiler.prepare(tree, *res, frame);
        gfx::RenderTarget target;
        target.color = color;
        target.depth = depth;
        target.clear_rgba = kClear;
        gfx::DrawList list(target);
        sc::DrawMap map;
        const sc::CompileResult r = compiler.compile(tree, camera, options, *res, list, &map);
        REQUIRE_MESSAGE(r.ok, (r.notes.empty() ? std::string() : r.notes[0]));
        CHECK_MESSAGE(r.skipped == 0, (r.notes.empty() ? std::string() : r.notes[0]));
        CHECK_MESSAGE(res->error().empty(), res->error());
        frame.render(std::move(list));
        @autoreleasepool {
            const gfx::Serial s = g.submit(std::move(frame));
            REQUIRE_MESSAGE(s != 0, g.last_error());
            REQUIRE(g.wait(s));
        }
        return read_back(g, color, kSize, kSize);
    }

    // The built-in pick at pixel (x, y), through the scene's pick pass and the picker.
    std::optional<sc::Hit> pick(int x, int y) {
        gfx::Texture ids = g.create_texture({16, 16, gfx::Format::RG32Uint, gfx::kRenderTarget | gfx::kCopySrcTex, "pick ids"});
        gfx::Texture dep = g.create_texture({16, 16, gfx::Format::Depth32Float, gfx::kRenderTarget | gfx::kCopySrcTex,
                                             "pick depth"});
        REQUIRE_MESSAGE((ids && dep), g.last_error());
        sc::Picker picker;
        const sc::PickWindow w = picker.request(float(x) + 0.5f, float(y) + 0.5f, 8, kSize, kSize);
        const std::int32_t window[4] = {w.x0, w.y0, w.width, w.height};
        gfx::RenderTarget pt;
        pt.color = ids;
        pt.depth = dep;
        gfx::DrawList list(pt);
        sc::DrawMap map;
        const sc::CompileResult r = compiler.compile_pick(tree, camera, options, window, *res, list, &map);
        REQUIRE_MESSAGE(r.ok, (r.notes.empty() ? std::string() : r.notes[0]));
        gfx::Frame f;
        f.render(std::move(list));
        @autoreleasepool {
            REQUIRE_MESSAGE(g.submit(std::move(f)) != 0, g.last_error());
        }
        const std::vector<std::uint8_t> id_bytes = read_back(g, ids, 16, 16);
        const std::vector<std::uint8_t> depth_bytes = read_back(g, dep, 16, 16);
        g.release(ids);
        g.release(dep);
        return picker.deliver(picker.current(), map.revision, id_bytes, &depth_bytes, map);
    }
};

// ---- the pieces' draw -----------------------------------------------------------------------

// One program: a library piece installed in a draw runtime with the published resources of its
// derive half and its params.
struct Program {
    cs::Piece piece;
    std::unique_ptr<cd::DrawRuntime> rt;
    Json values = Json::object();
    std::vector<Published> published;
    std::uint32_t pick_id = kPick;
};

// Runs the derive half of the library piece `name` over `inputs` and installs its draw half.
inline void make_program(Metal& m, const std::string& name, const std::vector<Tensor>& inputs, const Json& values,
                         std::uint32_t pick_id, Program* p) {
    std::string why;
    REQUIRE_MESSAGE(load_piece(name, &p->piece, &why), why);
    p->values = values;
    p->pick_id = pick_id;
    REQUIRE_MESSAGE(run_derive(m.device, p->piece, inputs, values, &p->published, &why), why);
    p->rt = std::make_unique<cd::DrawRuntime>(*m.backend, false);
    cd::Instance instance;
    instance.program = name;
    instance.panel = "scene";
    const cd::Verdict installed = p->rt->install(p->piece, instance, pick_id);
    REQUIRE_MESSAGE(installed.ok(), installed.code << " " << installed.message);
    for (const Published& r : p->published) {
        cd::ResourceVersion v;
        v.resource = r.id;
        v.step = 1;
        v.count = r.count;
        v.bytes = r.bytes.data();
        v.size = r.bytes.size();
        const cd::Verdict s = p->rt->set_resource(v);
        REQUIRE_MESSAGE(s.ok(), r.id << ": " << s.code << " " << s.message);
    }
}

// The pictures of programs drawn in order into one host target after one clear.
struct Drawn {
    std::vector<std::uint8_t> color, ids;
};

inline Drawn draw_programs(gfx::Backend& g, const std::vector<Program*>& programs, const std::array<float, 16>& mvp,
                           float pixel_ratio = 1.0f) {
    dp::Target target(g, kSize);
    REQUIRE_MESSAGE(target.ok, g.last_error());
    std::vector<gfx::Serial> serials;
    std::vector<bool> encoded(programs.size(), false);
    @autoreleasepool {
        gfx::RenderTarget clear;
        clear.color = target.host.color;
        clear.id = target.host.id;
        clear.depth = target.host.depth;
        clear.clear_rgba = kClear;
        gfx::Frame f;
        f.render(gfx::DrawList(clear));
        const gfx::Serial s = g.submit(std::move(f));
        REQUIRE_MESSAGE(s != 0, g.last_error());
        serials.push_back(s);
        for (std::size_t k = 0; k < programs.size(); ++k) {
            Program* p = programs[k];
            cd::FrameInputs in;
            in.clip_from_world = mvp;
            in.viewport = {0, 0, float(kSize), float(kSize)};
            in.pixel_ratio = pixel_ratio;
            in.cursor_step = 1;
            in.latest_step = 1;
            in.live = true;
            in.params = caliper_web::pack_params(p->piece, p->values);
            std::vector<gfx::Frame> frames;
            // A program whose passes all have no valid element encodes no frame, without an error.
            encoded[k] = p->rt->encode(&frames, in, target.host);
            REQUIRE_MESSAGE((encoded[k] || p->rt->status().error.code.empty()),
                            p->rt->status().error.code << " " << p->rt->status().error.message);
            for (gfx::Frame& pf : frames) {
                const gfx::Serial ps = g.submit(std::move(pf));
                REQUIRE_MESSAGE(ps != 0, g.last_error());
                serials.push_back(ps);
            }
        }
    }
    for (gfx::Serial s : serials) REQUIRE_MESSAGE(g.wait(s), g.last_error());
    for (std::size_t k = 0; k < programs.size(); ++k) {
        const cd::DrawStatus st = programs[k]->rt->status();
        CHECK_MESSAGE((st.state == (encoded[k] ? cd::DrawState::running : cd::DrawState::compiled)),
                      st.error.code << " " << st.error.message);
    }
    Drawn d;
    d.color = read_back(g, target.host.color, kSize, kSize);
    d.ids = read_back(g, target.host.id, kSize, kSize);
    return d;
}

// ---- one case -------------------------------------------------------------------------------

// The expected fields of the item map for element `item`, or nothing for an element that must not
// be picked.
using Fields = std::vector<std::pair<std::string, double>>;

// Compares the two pictures, prints the measured differences and checks the picks of `items`
// (each must show an interior pixel unless it is listed in `hidden`).
inline Comparison check_case(const std::string& name, Builtin& b, const std::vector<std::uint8_t>& builtin,
                             const Drawn& piece, Program& p, const std::vector<std::uint32_t>& items,
                             const std::function<Fields(std::uint32_t)>& fields, std::size_t min_lit = 1,
                             const std::string& capture_name = "") {
    REQUIRE(builtin.size() == piece.color.size());
    const Comparison c = compare(builtin, piece.color, piece.ids);
    MESSAGE(name, ": ", c.interior, " interior pixels (", c.lit_interior, " lit), ", c.edge, " edge pixels, ", c.differ,
            " differ, largest interior gap ", c.max_interior_gap, ", largest edge gap to a neighbor ", c.max_edge_gap,
            ", ", c.failed, " failed");
    CHECK_MESSAGE(c.failed == 0, name, ": ", c.failed, " pixels outside the tolerances; ", c.first);
    CHECK_MESSAGE(c.lit_interior >= min_lit, name, ": ", c.lit_interior, " lit interior pixels");
    if (!capture_name.empty()) {
        capture_pair(capture_name, builtin, piece.color);
    } else if (c.failed > 0) {   // a failed case leaves its pictures for a look
        std::string file = "failed-";
        for (char ch : name) file += std::isalnum(static_cast<unsigned char>(ch)) ? ch : '-';
        capture_pair(file, builtin, piece.color);
    }

    // The picks: a pixel inside each element, the built-in pick there, and the piece's item map.
    const int S = int(kSize);
    int agree = 0;
    for (std::uint32_t item : items) {
        CAPTURE(item);
        const std::uint64_t want = (std::uint64_t(p.pick_id) << 32 | item) + 1;
        int px = -1, py = -1;
        for (int y = 1; y < S - 1 && px < 0; ++y)
            for (int x = 1; x < S - 1 && px < 0; ++x)
                if (identity_of(piece.ids, x, y) == want && is_interior(piece.ids, x, y)) px = x, py = y;
        CHECK_MESSAGE(px >= 0, name, ": element ", item, " shows no interior pixel");
        if (px < 0) continue;
        const std::optional<sc::Hit> hit = b.pick(px, py);
        REQUIRE_MESSAGE(hit.has_value(), name, ": no built-in hit at (", px, ", ", py, ")");
        CHECK_MESSAGE(hit->item == item, name, ": pixel (", px, ", ", py, "): built-in item ", hit->item,
                      ", piece item ", item);
        const std::optional<cd::PickedItem> picked = p.rt->pick(item);
        REQUIRE_MESSAGE(picked.has_value(), name, ": the piece gives no fields for element ", item);
        CHECK(picked->item == item);
        const Fields want_fields = fields(item);
        CHECK(picked->fields.size() == want_fields.size());
        for (std::size_t k = 0; k < std::min(picked->fields.size(), want_fields.size()); ++k) {
            CHECK(picked->fields[k].first == want_fields[k].first);
            CHECK_MESSAGE(picked->fields[k].second == want_fields[k].second, name, ": field ", want_fields[k].first,
                          " of element ", item, " is ", picked->fields[k].second, ", not ", want_fields[k].second);
        }
        agree += hit->item == item;
    }
    MESSAGE(name, ": ", agree, " of ", items.size(), " picks name the piece's element");
    return c;
}

}  // namespace line_match
