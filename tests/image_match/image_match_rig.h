// image_match_rig.h: what the match tests of the library pieces image@1, depth@1 and
// segmentation@1 share (tests/image_match, tests/depth_match, tests/segmentation_match).
//
// A case draws one image twice on Metal, offscreen and without a window. The built-in draw is the
// scene view of the app (live/live_scene.cpp) after the protocol 3.2 messages of the case, which
// convert the pixels and draw them as the app does. The piece's draw runs the derive half of the
// piece in the producer runtime over the case's tensors, as a training script runs it, and gives
// the published versions to a draw runtime, with the camera matrices that the scene view gives its
// programs. The comparison then holds the piece's picture to the tolerances of the plan: a pixel
// whose 3 x 3 neighborhood shows one pixel of the image in the piece's ID image differs from the
// built-in by at most the case's interior tolerance in each channel, and any other pixel lies
// within 1 level of a built-in color in its 3 x 3 neighborhood. Pixels that the built-in draws
// without the image (a camera's frustum, the box that gives a 3-D scene its bounds) stay out. The
// picks of the built-in at pixels of the image name the pixel that the piece's ID shows there,
// and the item map of the piece gives that pixel's value.
//
// With CALIPER_IMAGE_MATCH_CAPTURE_DIR set, a case writes <name>-builtin.png, <name>-piece.png and
// <name>-diff.png (the difference scaled up 32 times) there.
#pragma once

#include "../gpu_skip.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "../draw_parity/parity_draw.h"
#include "../scene/primitive_fixtures.h"
#include "draw/draw_runtime.h"
#include "gfx/metal_backend.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "live/live_scene.h"
#include "net/wire.h"
#include "producer/caliper_producer.h"
#include "producer/caliper_producer_derive.h"
#include "scene/camera.h"
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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace image_match {

namespace cd = caliper_draw;
namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;
namespace dp = draw_parity;
namespace gfx = caliper_gfx;
namespace pf = primitive_fixtures;
namespace sc = caliper_scene;
namespace sh = caliper_gfx::shaders;
namespace wire = caliper_net::wire;

constexpr std::uint32_t kSize = 256;   // the target, square
constexpr std::uint32_t kPiecePick = 1u << 24;
constexpr std::uint32_t kClear = dp::kBackground;

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

// ---- Pieces ---------------------------------------------------------------------------------------

// A piece of the built-in library, with its files and its hash.
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

// The checks of piece.check on a piece: ok, and the passes that it composed and compiled.
struct Checked {
    bool ok = false;
    std::size_t passes = 0;
    std::string messages;
};
inline Checked check_piece(const cs::Piece& piece) {
    Checked out;
    nlohmann::json j = nlohmann::json::parse(cp::canonical_manifest(piece), nullptr, false);
    if (!j.is_object()) return (out.messages = "no canonical manifest", out);
    nlohmann::json files = nlohmann::json::object();
    for (const auto& [name, text] : piece.files) files[name.substr(0, name.find('.'))] = text;
    j["files"] = files;
    cs::Piece decoded;
    const nlohmann::json r = caliper_builder::check_piece_json(j, false, &decoded);
    out.ok = r.value("ok", false);
    if (r.contains("passes") && r["passes"].is_array()) out.passes = r["passes"].size();
    if (r.contains("messages")) out.messages = r["messages"].dump();
    return out;
}

// The params of a piece in the manifest's order, packed as the draw runtime reads them: the
// values given by name, the defaults for the others.
inline std::vector<std::uint8_t> pack_params(const cs::Piece& piece, const std::map<std::string, double>& values) {
    std::vector<std::uint8_t> out(piece.params.size() * 4);
    for (std::size_t i = 0; i < piece.params.size(); ++i) {
        const cs::PieceParam& p = piece.params[i];
        const auto it = values.find(p.name);
        const double v = it == values.end() ? p.default_value : it->second;
        if (p.type == cs::ParamType::f32) {
            const float f = float(v);
            std::memcpy(out.data() + 4 * i, &f, 4);
        } else if (p.type == cs::ParamType::i32) {
            const std::int32_t n = std::int32_t(v);
            std::memcpy(out.data() + 4 * i, &n, 4);
        } else {
            const std::uint32_t n = std::uint32_t(v);
            std::memcpy(out.data() + 4 * i, &n, 4);
        }
    }
    return out;
}

// ---- The derive half ------------------------------------------------------------------------------

// A tensor of the training script that an input of the piece binds.
struct Tensor {
    std::string input;                  // the piece's input
    std::uint32_t dtype = CALIPER_PRODUCER_F32;
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> bytes;
};

template <class T>
Tensor tensor(const std::string& input, std::vector<std::int64_t> shape, const std::vector<T>& v,
              std::uint32_t dtype = CALIPER_PRODUCER_F32) {
    Tensor t;
    t.input = input;
    t.dtype = dtype;
    t.shape = std::move(shape);
    t.bytes.resize(v.size() * sizeof(T));
    if (!v.empty()) std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
    return t;
}

// A pose as the tensor (4, 4) row by row, from a column-major matrix of 16 floats.
inline Tensor pose_tensor(const std::vector<float>& column_major) {
    std::vector<float> rows(16);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) rows[std::size_t(r * 4 + c)] = column_major[std::size_t(c * 4 + r)];
    return tensor("pose", {4, 4}, rows);
}

// One published version after a run of the derive half.
struct Version {
    std::string resource;
    std::uint32_t count = 0;
    std::vector<std::uint8_t> bytes;
    std::uint32_t width = 0, height = 0;    // a texture's version
    std::string format;                     // a texture's format, empty for rgba8unorm
    std::optional<std::uint32_t> elements;  // a buffer of narrow elements
};

// Runs the derive half of the piece once in the producer runtime over the tensors, with the
// manifest's limits, and gives each published resource.
inline bool run_derive(id<MTLDevice> device, const cs::Piece& piece, const std::vector<Tensor>& tensors,
                       const std::map<std::string, double>& values, std::vector<Version>* out, std::string* why) {
    out->clear();
    if (!piece.derive) return (*why = piece.name + " has no derive half", false);
    caliper_producer* rt = nullptr;
    caliper_producer_error e{};
    auto failed = [&](const std::string& what) {
        *why = what + ": " + (e.code ? e.code : "?") + " " + (e.message ? e.message : "");
        caliper_producer_error_free(&e);
        if (rt) caliper_producer_destroy(rt);
        return false;
    };
    if (caliper_producer_create((__bridge void*)device, &rt, &e) != CALIPER_PRODUCER_OK) return failed("create");

    std::vector<id<MTLBuffer>> buffers;
    std::vector<std::string> live;
    live.reserve(piece.inputs.size());
    std::vector<caliper_producer_input> inputs;
    std::vector<const char*> input_types;
    std::vector<std::uint8_t> unbound;
    for (const cs::PieceInput& in : piece.inputs) {
        const Tensor* t = nullptr;
        for (const Tensor& x : tensors)
            if (x.input == in.name) t = &x;
        if (t == nullptr && in.optional) {   // an optional input that the case leaves unbound
            inputs.push_back({in.name.c_str(), 0});
            input_types.push_back(in.dtype.c_str());
            unbound.push_back(1);
            continue;
        }
        if (t == nullptr) return failed("no tensor for the input " + in.name);
        unbound.push_back(0);
        id<MTLBuffer> b = [device newBufferWithLength:std::max<std::size_t>(16, t->bytes.size())
                                              options:MTLResourceStorageModeShared];
        if (!t->bytes.empty()) std::memcpy([b contents], t->bytes.data(), t->bytes.size());
        buffers.push_back(b);
        live.push_back("tensors." + in.name);
        caliper_producer_tensor ct{};
        ct.name = live.back().c_str();
        ct.buffer = (__bridge void*)b;
        ct.byte_offset = 0;
        ct.byte_size = t->bytes.size();
        ct.dtype = t->dtype;
        ct.ndim = std::uint32_t(t->shape.size());
        for (std::size_t k = 0; k < t->shape.size(); ++k) ct.shape[k] = t->shape[k];
        std::uint32_t handle = 0;
        if (caliper_producer_import(rt, &ct, &handle, &e) != CALIPER_PRODUCER_OK) return failed("import " + in.name);
        inputs.push_back({in.name.c_str(), handle});
        input_types.push_back(in.dtype.c_str());
    }

    std::vector<std::vector<std::string>> field_text(piece.resources.size());
    std::vector<std::vector<const char*>> fields(piece.resources.size());
    std::vector<caliper_producer_resource_desc> resources;
    for (std::size_t i = 0; i < piece.resources.size(); ++i) {
        const cs::Resource& r = piece.resources[i];
        caliper_producer_resource_desc d{};
        d.id = r.id.c_str();
        d.kind = r.kind == cs::ResourceKind::texture ? CALIPER_PRODUCER_RESOURCE_TEXTURE : CALIPER_PRODUCER_RESOURCE_BUFFER;
        d.element = r.layout.element ? r.layout.element->c_str() : nullptr;
        d.struct_name = r.layout.type_name ? r.layout.type_name->c_str() : nullptr;
        for (const cs::Field& f : r.layout.fields) field_text[i].push_back(f.name + ": " + f.type);
        for (const std::string& s : field_text[i]) fields[i].push_back(s.c_str());
        d.fields = fields[i].empty() ? nullptr : fields[i].data();
        d.field_count = fields[i].size();
        d.count = r.count;
        d.width = r.width;
        d.height = r.height;
        d.format = r.format ? r.format->c_str() : nullptr;
        d.texels = r.texels.value_or(0);
        d.counted = r.counted ? 1 : 0;
        d.publish = r.publish ? 1 : 0;
        d.clear = r.clear ? 1 : 0;
        resources.push_back(d);
    }
    std::vector<std::vector<const char*>> writes(piece.derive->passes.size());
    std::vector<caliper_producer_derive_pass> passes;
    for (std::size_t i = 0; i < piece.derive->passes.size(); ++i) {
        const cs::DerivePass& p = piece.derive->passes[i];
        for (const std::string& w : p.writes) writes[i].push_back(w.c_str());
        caliper_producer_derive_pass d{};
        d.entry = p.entry.c_str();
        for (int k = 0; k < 3; ++k) d.workgroups[k] = p.workgroups[std::size_t(k)];
        d.writes = writes[i].empty() ? nullptr : writes[i].data();
        d.write_count = writes[i].size();
        passes.push_back(d);
    }
    std::vector<caliper_producer_param> params;
    std::vector<const char*> param_types;
    for (const cs::PieceParam& p : piece.params) {
        const auto it = values.find(p.name);
        params.push_back({p.name.c_str(), it == values.end() ? p.default_value : it->second});
        param_types.push_back(p.type == cs::ParamType::f32 ? "f32" : p.type == cs::ParamType::i32 ? "i32" : "u32");
    }
    const cp::Resolved resolved = cp::resolve_imports(piece);
    if (!resolved.verdict.ok()) return failed("resolve: " + resolved.verdict.message);
    auto file = [&](const char* name) {
        const auto it = resolved.files.find(name);
        return it == resolved.files.end() ? std::string() : it->second;
    };
    const std::string derive = file("derive.wgsl");
    const std::string common = resolved.lib_text + file("common.wgsl");

    caliper_producer_derive_desc d{};
    d.id = "match";
    d.input_unbound = unbound.data();
    d.piece_hash = piece.hash.c_str();
    d.wgsl = derive.c_str();
    d.wgsl_size = derive.size();
    d.common = common.empty() ? nullptr : common.c_str();
    d.common_size = common.size();
    d.inputs = inputs.data();
    d.input_count = inputs.size();
    d.input_types = input_types.data();
    d.resources = resources.data();
    d.resource_count = resources.size();
    d.passes = passes.data();
    d.pass_count = passes.size();
    d.params = params.data();
    d.param_types = param_types.data();
    d.param_count = params.size();
    d.time_limit_ms = piece.derive->time_ms.value_or(4.0);
    d.memory_limit_bytes = std::uint64_t(piece.derive->memory_mb.value_or(64)) << 20;
    std::uint32_t program = 0;
    if (caliper_producer_compile_derive(rt, &d, &program, &e) != CALIPER_PRODUCER_OK) return failed("compile_derive");
    caliper_producer_frame frame{1, 0.0, 1};
    caliper_producer_pass pass{};
    if (caliper_producer_run(rt, &program, 1, &frame, 0, &pass, &e) != CALIPER_PRODUCER_OK) return failed("run");
    for (const cs::Resource& r : piece.resources) {
        if (!r.publish) continue;
        caliper_producer_output o{};
        if (caliper_producer_read_resource(rt, program, r.id.c_str(), &o, &e) != CALIPER_PRODUCER_OK)
            return failed("read " + r.id);
        Version v;
        v.resource = r.id;
        v.count = o.count;
        v.width = o.width;
        v.height = o.height;
        if (r.kind == cs::ResourceKind::texture && r.format && *r.format != "rgba8unorm") v.format = *r.format;
        if (o.narrow) v.elements = o.elements;
        if (o.byte_size) {
            const auto* base = static_cast<const std::uint8_t*>([(__bridge id<MTLBuffer>)o.buffer contents]);
            v.bytes.assign(base + o.byte_offset, base + o.byte_offset + o.byte_size);
        }
        out->push_back(std::move(v));
    }
    caliper_producer_error_free(&e);
    caliper_producer_destroy(rt);
    return true;
}

inline const Version* find_version(const std::vector<Version>& vs, const std::string& id) {
    for (const Version& v : vs)
        if (v.resource == id) return &v;
    return nullptr;
}

// ---- The built-in draw ----------------------------------------------------------------------------

// The scene view of the app on the backend, fed with the messages of a fixture through the codec.
struct Builtin {
    gfx::Backend& g;
    std::unique_ptr<caliper_live_views::SceneView> view;
    std::vector<std::shared_ptr<std::vector<std::uint8_t>>> frames;
    std::deque<wire::Message> decoded;
    std::string refusal;   // the why of the first message that the view refused

    explicit Builtin(gfx::Backend& backend) : g(backend) {}
    ~Builtin() { view.reset(); }

    bool draw(const pf::Fixture& f, std::string* why) {
        view = std::make_unique<caliper_live_views::SceneView>(g, sh::kMetalPrograms);
        if (!view->init(why)) return false;
        view->describe(f.info);
        for (const wire::Message& m : f.messages) {
            auto bytes = std::make_shared<std::vector<std::uint8_t>>();
            std::string err;
            if (!wire::encode(m, bytes.get(), &err, CALIPER_NET_MINOR_3_2)) return (*why = "encode: " + err, false);
            wire::Message out;
            const wire::DecodeResult r = wire::decode(bytes->data(), bytes->size(), &out);
            if (!r.ok()) return (*why = "decode: " + r.message, false);
            frames.push_back(bytes);
            decoded.push_back(std::move(out));
            const wire::Message& d = decoded.back();
            bool ok = true;
            std::string w;
            if (const auto* u = std::get_if<wire::SceneUpdate>(&d)) ok = view->apply(*u, &w);
            else if (const auto* t = std::get_if<wire::SceneTransforms>(&d)) ok = view->apply(*t, &w);
            else if (const auto* c = std::get_if<wire::SceneClasses>(&d)) ok = view->apply(*c, &w);
            if (!ok && refusal.empty()) refusal = w.empty() ? "refused" : w;
        }
        return view->render(kSize, kSize, 1.0f, kClear, why);
    }

    std::vector<std::uint8_t> pixels() {
        std::vector<std::uint8_t> px;
        std::string why;
        REQUIRE_MESSAGE(view->read_pixels(&px, &why), why);
        REQUIRE(px.size() == std::size_t(kSize) * kSize * 4);
        return px;
    }

    // The frame inputs that the scene view gives its programs (SceneView::draw_programs).
    cd::FrameInputs inputs() const {
        const sc::ViewCamera cam = view->effective_camera();
        sc::CameraMatrices cm;
        const double aspect = double(view->width()) / double(std::max<std::uint32_t>(1, view->height()));
        const sc::Aabb bounds = view->tree().scene_bounds();
        if (const sc::Orbit* o = std::get_if<sc::Orbit>(&cam)) {
            cm.view = sc::orbit_view(*o);
            cm.proj = sc::orbit_projection(*o, bounds, aspect);
            cm.eye = sc::orbit_eye(*o);
            cm.perspective = true;
        } else if (const sc::PanZoom* pz = std::get_if<sc::PanZoom>(&cam)) {
            cm.proj = sc::panzoom_projection(*pz, aspect);
        } else {
            cm = sc::camera_matrices(cam, view->tree(), view->width(), view->height());
        }
        const sc::Mat4 clip = cm.proj * cm.view;
        sc::Mat4 inv;
        if (!sc::inverse(clip, &inv)) inv = sc::Mat4{};
        cd::FrameInputs in;
        for (int i = 0; i < 16; ++i) {
            in.clip_from_world[std::size_t(i)] = float(clip.m[i]);
            in.world_from_clip[std::size_t(i)] = float(inv.m[i]);
            in.view_from_world[std::size_t(i)] = float(cm.view.m[i]);
        }
        in.eye = {float(cm.eye.x), float(cm.eye.y), float(cm.eye.z), cm.perspective ? 1.0f : 0.0f};
        in.viewport = {0, 0, float(kSize), float(kSize)};
        in.pixel_ratio = 1.0f;
        in.cursor_step = 1;
        in.latest_step = 1;
        in.live = true;
        return in;
    }

    // The built-in pick at the center of pixel (x, y), through the scene view's pick pass.
    std::optional<sc::Hit> pick(int x, int y) {
        sc::Hit hit;
        std::string why;
        REQUIRE_MESSAGE(view->request_pick(float(x) + 0.5f, float(y) + 0.5f, &why), why);
        sc::PickPass::Result r = sc::PickPass::Result::Pending;
        for (int tries = 0; tries < 100 && r == sc::PickPass::Result::Pending; ++tries) {
            g.wait(view->pick_pass()->last_use());
            g.poll();
            r = view->poll_pick(&hit);
        }
        if (r != sc::PickPass::Result::Hit) return std::nullopt;
        return hit;
    }
};

// A 3-D room: one box around the origin, which gives the scene its bounds.
inline pf::Fixture room(const std::string& name) {
    pf::Fixture f;
    f.name = name;
    f.info = {CALIPER_NET_SPACE_3D, "+z", "m"};
    std::vector<float> cen = {0.0f, 0.0f, 0.0f}, half = {1.2f, 1.2f, 0.6f};
    wire::SceneUpdate box = f.update("room/box", CALIPER_NET_PART_BOXES, true);
    box.params.set("color", pf::rgba(pf::RGBA(240, 180, 60)));
    box.arrays.push_back(f.section("centers", CALIPER_NET_DTYPE_F32, {1, 3}, cen));
    box.arrays.push_back(f.section("half_sizes", CALIPER_NET_DTYPE_F32, {1, 3}, half));
    f.messages.push_back(box);
    return f;
}

// The pose of a camera between the room and the eye of the view of the room, turned a little,
// so that its image plane faces the view at a slant and its frustum's apex lies beside the plane.
inline std::vector<float> facing_pose(Metal& m) {
    Builtin b(*m.backend);
    std::string why;
    REQUIRE_MESSAGE(b.draw(room("probe"), &why), why);
    const sc::ViewCamera cam = b.view->effective_camera();
    const sc::Orbit* o = std::get_if<sc::Orbit>(&cam);
    REQUIRE(o != nullptr);
    const sc::Vec3 e = sc::orbit_direction(*o);
    // p: a unit vector across the view, perpendicular to e and to the up axis z.
    double px = e.y, py = -e.x;
    const double n = std::sqrt(px * px + py * py);
    px /= n;
    py /= n;
    const float eye[3] = {float(2.0 * e.x - 0.9 * px), float(2.0 * e.y - 0.9 * py), float(2.0 * e.z)};
    const float fwd[3] = {float(-e.x + 0.45 * px), float(-e.y + 0.45 * py), float(-e.z)};
    return pf::look_from(eye, fwd);
}

// A camera "rig/cam" at a pose in a room, with the pinhole of the image cases: 16 x 12 pixels,
// fx = fy = 8, the principal point at the center and the image plane at 0.9.
inline pf::Fixture camera_room(const std::vector<float>& pose, const char* convention = CALIPER_NET_CONVENTION_RDF) {
    pf::Fixture f = room("cam");
    f.transform("rig", pf::translate(0, 0, 0), true);
    f.transform("rig/cam", pose, true);
    wire::SceneUpdate cam = f.update("rig/cam", CALIPER_NET_PART_CAMERA, true);
    cam.params.set("fx", caliper_net::Json(8.0));
    cam.params.set("fy", caliper_net::Json(8.0));
    cam.params.set("cx", caliper_net::Json(8.0));
    cam.params.set("cy", caliper_net::Json(6.0));
    cam.params.set("width", caliper_net::Json(16));
    cam.params.set("height", caliper_net::Json(12));
    cam.params.set("convention", caliper_net::Json(convention));
    cam.params.set("plane_distance", caliper_net::Json(0.9));
    f.messages.push_back(cam);
    return f;
}
inline const std::map<std::string, double> kCameraParams = {{"placement", 1}, {"fx", 8}, {"fy", 8},
                                                            {"cx", 8},        {"cy", 6}, {"plane", 0.9}};

// ---- The piece's draw -----------------------------------------------------------------------------

inline std::vector<std::uint8_t> read_back(gfx::Backend& g, gfx::Texture t, std::uint32_t w, std::uint32_t h) {
    const gfx::Readback rb = g.read_texture(t, 0, 0, w, h);
    REQUIRE_MESSAGE(bool(rb), g.last_error());
    std::vector<std::uint8_t> out;
    REQUIRE(g.wait(g.last_use(t)));
    REQUIRE(g.take_readback(rb, &out));
    return out;
}

// The piece installed in a draw runtime with the versions of one run, drawn in one frame.
struct PieceDraw {
    std::unique_ptr<cd::DrawRuntime> rt;
    std::vector<std::uint8_t> color, ids;
    std::string error;
};
inline bool draw_piece(gfx::Backend& g, const cs::Piece& piece, const std::vector<Version>& versions,
                       cd::FrameInputs in, const std::map<std::string, double>& values, PieceDraw* out) {
    out->rt = std::make_unique<cd::DrawRuntime>(g, false);
    cd::Instance instance;
    instance.program = piece.name;
    instance.panel = "view";
    const cd::Verdict installed = out->rt->install(piece, instance, kPiecePick);
    if (!installed.ok()) return (out->error = "install: " + installed.code + " " + installed.message, false);
    for (const Version& v : versions) {
        cd::ResourceVersion rv;
        rv.resource = v.resource;
        rv.step = 1;
        rv.count = v.count;
        rv.width = v.width;
        rv.height = v.height;
        rv.format = v.format;
        rv.elements = v.elements;
        rv.bytes = v.bytes.empty() ? nullptr : v.bytes.data();
        rv.size = v.bytes.size();
        const cd::Verdict s = out->rt->set_resource(rv);
        if (!s.ok()) return (out->error = "set_resource " + v.resource + ": " + s.code + " " + s.message, false);
    }
    dp::Target target(g, kSize);
    if (!target.ok) return (out->error = "target: " + g.last_error(), false);
    in.params = pack_params(piece, values);
    std::vector<gfx::Serial> serials;
    std::string why;
    if (!target.frame(*out->rt, in, &serials, &why)) return (out->error = why, false);
    for (gfx::Serial s : serials)
        if (!g.wait(s)) return (out->error = "wait: " + g.last_error(), false);
    const cd::DrawStatus st = out->rt->status();
    if (st.state != cd::DrawState::running)
        return (out->error = "state " + std::to_string(int(st.state)) + ": " + st.error.code + " " + st.error.message, false);
    out->color = read_back(g, target.host.color, kSize, kSize);
    out->ids = read_back(g, target.host.id, kSize, kSize);
    return true;
}

// ---- The comparison -------------------------------------------------------------------------------

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

// The identity of pixel (x, y) in the piece's ID image: 0 for none, else item + 1.
inline std::uint64_t identity_of(const std::vector<std::uint8_t>& ids, int x, int y) {
    std::uint32_t pick = 0, item = 0;
    const std::uint8_t* p = ids.data() + (std::size_t(y) * kSize + std::size_t(x)) * 8;
    std::memcpy(&pick, p, 4);
    std::memcpy(&item, p + 4, 4);
    return pick == 0 ? 0 : std::uint64_t(item) + 1;
}

inline bool is_interior(const std::vector<std::uint8_t>& ids, int x, int y) {
    const std::uint64_t own = identity_of(ids, x, y);
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            if (identity_of(ids, std::clamp(x + dx, 0, int(kSize) - 1), std::clamp(y + dy, 0, int(kSize) - 1)) != own)
                return false;
    return true;
}

// Pixels that the built-in draws without the image: 1 where the plain frame is not the clear color.
inline std::vector<std::uint8_t> mask_of(const std::vector<std::uint8_t>& plain) {
    std::vector<std::uint8_t> m(std::size_t(kSize) * kSize, 0);
    for (int y = 0; y < int(kSize); ++y)
        for (int x = 0; x < int(kSize); ++x) m[std::size_t(y) * kSize + std::size_t(x)] = pixel_of(plain, x, y) != kClear;
    return m;
}

struct Comparison {
    std::size_t interior = 0, lit = 0, edge = 0, masked = 0, failed = 0;
    int interior_gap = 0;   // the largest difference of a channel at an interior pixel
    std::size_t interior_differ = 0;
    std::string first;
};
inline Comparison compare(const std::vector<std::uint8_t>& piece_color, const std::vector<std::uint8_t>& piece_ids,
                          const std::vector<std::uint8_t>& builtin_color, const std::vector<std::uint8_t>* mask,
                          int interior_tolerance) {
    Comparison c;
    const int S = int(kSize);
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x) {
            if (mask && (*mask)[std::size_t(y) * kSize + std::size_t(x)]) {
                ++c.masked;
                continue;
            }
            const std::uint32_t pc = pixel_of(piece_color, x, y), bc = pixel_of(builtin_color, x, y);
            bool ok;
            const bool interior = is_interior(piece_ids, x, y);
            if (interior) {
                ++c.interior;
                c.lit += identity_of(piece_ids, x, y) != 0;
                const int gap = channel_gap(pc, bc);
                c.interior_gap = std::max(c.interior_gap, gap);
                c.interior_differ += gap != 0;
                ok = gap <= interior_tolerance;
            } else {
                ++c.edge;
                ok = false;
                for (int dy = -1; dy <= 1 && !ok; ++dy)
                    for (int dx = -1; dx <= 1 && !ok; ++dx)
                        ok = channel_gap(pc, pixel_of(builtin_color, std::clamp(x + dx, 0, S - 1),
                                                      std::clamp(y + dy, 0, S - 1))) <= 1;
            }
            if (!ok) {
                ++c.failed;
                if (c.first.empty()) {
                    char buf[160];
                    std::snprintf(buf, sizeof buf, "first at (%d, %d): piece %08x, built-in %08x, %s", x, y, pc, bc,
                                  interior ? "interior" : "edge");
                    c.first = buf;
                }
            }
        }
    return c;
}

inline void capture(const std::string& name, const std::vector<std::uint8_t>& rgba) {
    const char* dir = std::getenv("CALIPER_IMAGE_MATCH_CAPTURE_DIR");
    if (dir == nullptr || *dir == 0) return;
    const std::string path = std::string(dir) + "/" + name;
    CHECK(stbi_write_png(path.c_str(), int(kSize), int(kSize), 4, rgba.data(), int(kSize) * 4) != 0);
    MESSAGE("capture written: ", path);
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

// ---- One case -------------------------------------------------------------------------------------

struct Case {
    std::string name;
    pf::Fixture scene;                     // the built-in's messages
    std::optional<pf::Fixture> plain;      // the same scene without the image, for the mask
    std::vector<Tensor> tensors;
    std::map<std::string, double> params;
    std::uint32_t width = 0, height = 0;   // of the image
    int interior_tolerance = 0;
    std::size_t min_lit = 2000;            // interior pixels of the image that the piece must draw
    std::size_t picks = 16;                // pixels to pick, the first and the last pixel among them
    // The value of the item map's first field for a pixel, from the case's data.
    std::function<double(std::uint32_t)> field;
};

struct Outcome {
    Comparison c;
    std::size_t probes = 0, agree = 0, fields = 0;
    bool first = false, last = false;
    std::vector<std::uint8_t> builtin, piece;
};

// Draws the case both ways, compares the pictures and the picks, and writes the captures.
inline Outcome run_case(Metal& m, const cs::Piece& piece, const Case& k) {
    Outcome o;
    gfx::Backend& g = *m.backend;
    std::string why;
    std::vector<Version> versions;
    REQUIRE_MESSAGE(run_derive(m.device, piece, k.tensors, k.params, &versions, &why), why);

    Builtin b(g);
    REQUIRE_MESSAGE(b.draw(k.scene, &why), why);
    CHECK_MESSAGE(b.refusal.empty(), b.refusal);
    o.builtin = b.pixels();
    std::vector<std::uint8_t> mask;
    if (k.plain) {
        Builtin p(g);
        REQUIRE_MESSAGE(p.draw(*k.plain, &why), why);
        mask = mask_of(p.pixels());
    }
    PieceDraw d;
    REQUIRE_MESSAGE(draw_piece(g, piece, versions, b.inputs(), k.params, &d), d.error);
    o.piece = d.color;
    o.c = compare(d.color, d.ids, o.builtin, k.plain ? &mask : nullptr, k.interior_tolerance);
    MESSAGE(k.name, ": ", o.c.interior, " interior pixels (", o.c.lit, " lit, ", o.c.interior_differ,
            " differ, largest gap ", o.c.interior_gap, "), ", o.c.edge, " edge pixels, ", o.c.masked, " masked, ",
            o.c.failed, " failed");
    CHECK_MESSAGE(o.c.failed == 0, k.name, ": ", o.c.failed, " pixels outside the tolerances; ", o.c.first);
    CHECK_MESSAGE(o.c.lit >= k.min_lit, k.name, ": ", o.c.lit, " lit interior pixels");
    capture_pair(k.name, o.builtin, o.piece);

    // The picks: interior pixels of the piece's ID image, away from the mask, spread over the items,
    // with the first and the last pixel of the image when the picture shows them.
    const std::uint64_t n = std::uint64_t(k.width) * k.height;
    std::map<std::uint64_t, std::pair<int, int>> where;   // item to a pixel that shows it
    for (int y = 1; y < int(kSize) - 1; ++y)
        for (int x = 1; x < int(kSize) - 1; ++x) {
            const std::uint64_t id = identity_of(d.ids, x, y);
            if (id == 0 || where.count(id - 1) || !is_interior(d.ids, x, y)) continue;
            bool clear = true;
            for (int dy = -2; dy <= 2 && clear && !mask.empty(); ++dy)
                for (int dx = -2; dx <= 2 && clear; ++dx)
                    clear = !mask[std::size_t(std::clamp(y + dy, 0, int(kSize) - 1)) * kSize +
                                  std::size_t(std::clamp(x + dx, 0, int(kSize) - 1))];
            if (clear) where[id - 1] = {x, y};
        }
    std::vector<std::uint64_t> items;
    if (where.count(0)) items.push_back(0);
    if (n && where.count(n - 1)) items.push_back(n - 1);
    o.first = where.count(0) != 0;
    o.last = n && where.count(n - 1) != 0;
    std::vector<std::uint64_t> all;
    for (const auto& [item, at] : where) all.push_back(item);
    for (std::size_t i = 0; i < k.picks && !all.empty() && items.size() < k.picks; ++i) {
        const std::uint64_t item = all[i * all.size() / k.picks];
        if (std::find(items.begin(), items.end(), item) == items.end()) items.push_back(item);
    }
    for (const std::uint64_t item : items) {
        const int x = where[item].first, y = where[item].second;
        CAPTURE(x);
        CAPTURE(y);
        ++o.probes;
        const std::optional<sc::Hit> hit = b.pick(x, y);
        REQUIRE_MESSAGE(hit.has_value(), k.name, ": the built-in picks nothing at (", x, ", ", y, ")");
        std::uint64_t builtin_item = hit->item;
        if (hit->primitive == sc::Primitive::Image && hit->part == sc::DrawPart::Batch) {
            CHECK(hit->has_pixel);
            builtin_item = std::uint64_t(hit->pixel[1]) * k.width + hit->pixel[0];
        }
        CHECK_MESSAGE(builtin_item == item, k.name, " at (", x, ", ", y, "): built-in pixel ", builtin_item,
                      ", piece pixel ", item);
        o.agree += builtin_item == item;
        const std::optional<cd::PickedItem> picked = d.rt->pick(std::uint32_t(item));
        REQUIRE(picked.has_value());
        if (k.field) {
            // The item map leaves out a field that is not a finite number, as a pick event does.
            const double want = k.field(std::uint32_t(item));
            const bool finite = std::isfinite(want);
            REQUIRE(picked->fields.size() == (finite ? 1u : 0u));
            const bool same = !finite || picked->fields[0].second == want;
            CHECK_MESSAGE(same, k.name, ": the field of pixel ", item, " is not ", want);
            o.fields += same;
        }
    }
    MESSAGE(k.name, ": ", o.agree, " of ", o.probes, " picks name the piece's pixel (first ", o.first, ", last ",
            o.last, "), ", o.fields, " fields agree");
    return o;
}

// The case's name in the messages and the captures.
inline std::string case_name(const std::string& piece, const std::string& what) { return piece + "-" + what; }

// The pixels, color or ID, that the piece draws from these versions in an empty 2-D view.
inline std::size_t drawn_pixels(Metal& m, const cs::Piece& piece, const std::vector<Version>& versions,
                                const std::map<std::string, double>& values) {
    Builtin b(*m.backend);
    pf::Fixture none;
    none.name = "none";
    none.info = {CALIPER_NET_SPACE_2D, "-y", "px"};
    std::string why;
    REQUIRE_MESSAGE(b.draw(none, &why), why);
    PieceDraw d;
    if (!draw_piece(*m.backend, piece, versions, b.inputs(), values, &d)) {
        // A runtime without a version of a resource that a pass draws skips the pass: no frame.
        const cd::DrawStatus st = d.rt->status();
        CHECK_MESSAGE(st.error.code.empty(), d.error);
        CHECK(st.state != cd::DrawState::refused);
        CHECK(st.state != cd::DrawState::stopped);
        return 0;
    }
    std::size_t drawn = 0;
    for (int y = 0; y < int(kSize); ++y)
        for (int x = 0; x < int(kSize); ++x) drawn += pixel_of(d.color, x, y) != kClear || identity_of(d.ids, x, y);
    return drawn;
}

// The checks of an image that the piece cannot hold and of an empty one. The derive half of an
// image above the limit publishes no pixels and a size of 0, and the piece then draws nothing. The
// producer refuses to import an empty tensor, as the built-in refuses an empty image, and a piece
// without a version, or with a size that its pixels do not fill, draws nothing.
inline void check_refusals(Metal& m, const cs::Piece& piece, const std::vector<Tensor>& too_large,
                           const std::vector<Tensor>& empty, std::size_t info_bytes,
                           const std::map<std::string, double>& values) {
    std::string why;
    std::vector<Version> versions;
    REQUIRE_MESSAGE(run_derive(m.device, piece, too_large, values, &versions, &why), why);
    const Version* pixels = find_version(versions, "pixels");
    const Version* info = find_version(versions, "info");
    REQUIRE(pixels);
    REQUIRE(info);
    CHECK(pixels->count == 0);
    REQUIRE(info->bytes.size() == info_bytes);
    std::uint32_t size[2] = {1, 1};
    std::memcpy(size, info->bytes.data(), 8);
    CHECK(size[0] == 0);
    CHECK(size[1] == 0);
    CHECK(drawn_pixels(m, piece, versions, values) == 0);

    std::vector<Version> none;
    CHECK_FALSE(run_derive(m.device, piece, empty, values, &none, &why));
    MESSAGE(piece.name, "@1 on an empty tensor: ", why);
    CHECK(why.find("program.import_refused") != std::string::npos);
    CHECK(drawn_pixels(m, piece, {}, values) == 0);
    // An info of 4 x 4 pixels beside a version of no pixels: the draw reads no pixel past the count.
    std::vector<Version> stale = versions;
    const std::uint32_t four[2] = {4, 4};
    for (Version& v : stale)
        if (v.resource == "info") std::memcpy(v.bytes.data(), four, 8);
    CHECK(drawn_pixels(m, piece, stale, values) == 0);
}

}  // namespace image_match
