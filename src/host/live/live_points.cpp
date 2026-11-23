// live_points.cpp: the points live view as a scene of the scene core (live_points.h).
#include "live/live_points.h"

#include "gfx/colormap_keys.h"
#include "scene/scene_math.h"

#if defined(CALIPER_HAVE_DRAW) && defined(CALIPER_HAVE_LIBRARY_PIECES)
#define CALIPER_POINTS_PIECES 1
#include "draw/draw_runtime.h"
#include "draw/library_pieces.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace caliper_live_views {

using namespace caliper_scene;
namespace gfx = caliper_gfx;

// The colormap source: the keys of a range from the shared cache,
// with the colours of a built-in colormap, or of the palette at every level, dimmed toward the
// ground for the rest of a selection.
struct PointsScene::Tables {
    std::shared_ptr<gfx::cmap::KeyBlockCache> cache = std::make_shared<gfx::cmap::KeyBlockCache>();
    uint32_t palette[256] = {};
    uint32_t ground = 0xff000000u;
};

uint32_t dimmed_rgba(uint32_t c, uint32_t ground) {
    uint32_t out = 0xff000000u;
    for (int k = 0; k < 3; ++k) {
        const float a = (float)((c >> (8 * k)) & 0xff), b = (float)((ground >> (8 * k)) & 0xff);
        out |= (uint32_t)std::lround(a * 0.28f + b * 0.72f) << (8 * k);
    }
    return out;
}

namespace {

constexpr double kOrbitRate = 0.01;        // radians a logical pixel
constexpr double kElevationBound = 1.55;   // the old view's bound, inside the scene core's 89 degrees

std::string chunk_path(const char* base, uint64_t k) {
    if (k == 0) return base;
    char t[32];
    std::snprintf(t, sizeof t, "%s%sc%05llu", base, *base ? "/" : "", (unsigned long long)k);
    return t;
}

}  // namespace


// ---- points@1 -------------------------------------------------------------------------------------

#if defined(CALIPER_POINTS_PIECES)

namespace {

namespace cd = caliper_draw;
namespace cs = caliper_builder::spec;

// points@1's element Point: position vec4<f32> (x, y, z, radius), bits u32, index u32, two pad words.
constexpr uint64_t kPointBytes = 32;
// Errors from the points runtime.
constexpr const char* kNoPiece = "no library piece";
constexpr const char* kStopped = "piece stopped by its runtime";

// The bytes of a buffer window that the CPU cannot read (an MTLBuffer in private storage): copied
// into RGBA8 textures of 4,096 texels a row, at most 4,096 rows each, and read back, all without a
// wait. The layer keeps the buffer alive until the copy's submit completes.
struct Fetch {
    struct Part {
        gfx::Texture tex;
        gfx::Readback rb;
        uint64_t at = 0, bytes = 0;
        uint32_t w = 0, h = 0;
        bool done = false;
    };
    std::vector<Part> parts;
    std::vector<uint8_t> bytes;
    bool failed = false;

    bool start(gfx::Backend& g, gfx::Buffer src, uint64_t offset, uint64_t size, gfx::Frame* frame) {
        // A copy's row pitch is a multiple of 256 bytes and a copy reads no byte past the window: rows of
        // 16 KiB, then rows of 256 bytes, then one row of what remains (the arrays hold 4-byte elements).
        constexpr uint32_t kRow = 4096, kRows = 4096;
        if (size % 4 != 0 || offset % 4 != 0) return false;
        bytes.assign(size, 0);
        for (uint64_t at = 0; at < size;) {
            const uint64_t left = size - at;
            uint32_t w = 0, h = 1, pitch = 256;
            if (left >= uint64_t(kRow) * 4) {
                w = kRow;
                h = uint32_t(std::min<uint64_t>(left / (uint64_t(kRow) * 4), kRows));
                pitch = kRow * 4;
            } else if (left >= 256) {
                w = 64;
                h = uint32_t(left / 256);
            } else {
                w = uint32_t(left / 4);
            }
            Part p;
            p.at = at;
            p.bytes = uint64_t(w) * h * 4;
            p.w = w;
            p.h = h;
            gfx::TextureDesc d;
            d.width = w;
            d.height = h;
            d.format = gfx::Format::RGBA8Unorm;
            d.usage = gfx::kCopyDstTex | gfx::kCopySrcTex;
            d.label = "points readback";
            p.tex = g.create_texture(d);
            if (!p.tex) return false;
            frame->copy_buffer_to_texture(src, offset + at, pitch, p.tex);
            at += p.bytes;
            parts.push_back(p);
        }
        return true;
    }
    // After the copies' submit: the readbacks, ordered after it.
    void read(gfx::Backend& g) {
        for (Part& p : parts) {
            p.rb = g.read_texture(p.tex, 0, 0, p.w, p.h);
            if (!p.rb) failed = true;
        }
    }
    // Takes what arrived, without a wait; true once every part is here (or failed).
    bool poll(gfx::Backend& g) {
        bool all = true;
        for (Part& p : parts) {
            if (p.done) continue;
            std::vector<uint8_t> got;
            if (p.rb && !g.take_readback(p.rb, &got)) {
                all = false;
                continue;
            }
            if (!p.rb || got.size() < p.bytes) failed = true;
            else std::memcpy(bytes.data() + p.at, got.data(), p.bytes);
            p.rb = {};   // take_readback freed it
            if (p.tex) g.release(p.tex);
            p.tex = {};
            p.done = true;
        }
        return all;
    }
    void release(gfx::Backend& g) {
        for (Part& p : parts) {
            if (p.rb) g.release(p.rb);
            if (p.tex) g.release(p.tex);
        }
        parts.clear();
    }
};

}  // namespace

struct PointsScene::Pieces {
    const cs::Piece* piece = nullptr;
    struct Program {
        std::string id;
        std::unique_ptr<cd::DrawRuntime> runtime;
        std::vector<uint8_t> rows;   // the next version, given at the next frame
        uint32_t count = 0;
        bool selection = false;
        bool given = true;
    };
    std::vector<Program> programs;   // a pool; the first `active` draw, in order
    size_t active = 0;
    uint64_t generation = 0;
    // A payload whose bytes come back from the GPU first: what set() got, and the readbacks.
    struct Wait {
        PointsArrays all;
        PointsLook look;
        Fetch pos, aux;
    };
    std::unique_ptr<Wait> wait;
    float size = 0.0f;     // logical pixels of a point's side
    bool failed = false;   // the last frame could not draw
};

#else
struct PointsScene::Pieces {};
#endif  // CALIPER_POINTS_PIECES

PointsScene::PointsScene(gfx::Backend& backend, const gfx::ProgramDesc* programs)
    : gfx_(backend), tables_(std::make_shared<Tables>()) {
    (void)programs;
    for (int i = 0; i < 256; ++i) tables_->palette[i] = kPalette[i % 16];
}

PointsScene::~PointsScene() {
#if defined(CALIPER_POINTS_PIECES)
    if (pieces_) {
        if (pieces_->wait) {
            pieces_->wait->pos.release(gfx_);
            pieces_->wait->aux.release(gfx_);
        }
        pieces_.reset();
    }
#endif
    if (color_) gfx_.release(color_);
    if (depth_) gfx_.release(depth_);
}

bool PointsScene::init(std::string* why) {
#if defined(CALIPER_POINTS_PIECES)
    (void)why;
    return true;
#else
    if (why) *why = "scene.pieces_unavailable: this build has no library pieces";
    return false;
#endif
}

// ---- content ---------------------------------------------------------------------------------------

bool PointsScene::set(const PointsArrays& all, const PointsArrays* selected, const PointsLook& look, std::string* why) {
    if (look.dims != dims_ || tree_.entity_count() == 0) {
        dims_ = look.dims == 2 ? 2 : 3;
        tree_ = SceneTree(dims_ == 2 ? Space::k2D : Space::k3D);
        if (dims_ == 2) {
            PanZoom p;
            p.y_down = false;
            camera_ = p;
        } else {
            camera_ = Orbit{};
        }
    }
    tree_.reset();
    tables_->ground = look.ground;
    const bool dim = selected != nullptr;
    auto add = [&](const PointsArrays& a, const char* base, bool dimmed) {
        for (uint64_t k = 0, first = 0; first < a.count; ++k, first += kMaxItems) {
            const uint64_t m = std::min<uint64_t>(a.count - first, kMaxItems);
            Batch b;
            b.primitive = Primitive::Points;
            BatchArray p;
            p.role = Role::Positions;
            p.buffer = a.pos;
            p.offset = a.pos_offset + first * 12;
            p.count = m;
            p.stride = 12;
            p.format = dims_ == 2 ? gfx::VertexFormat::F32x2 : gfx::VertexFormat::F32x3;
            p.cpu = a.pos_cpu ? static_cast<const uint8_t*>(a.pos_cpu) + first * 12 : nullptr;
            b.arrays.push_back(p);
            if (look.colour != PointsColour::Flat) {
                BatchArray v;
                v.role = Role::Values;
                v.buffer = a.aux;
                v.offset = a.aux_offset + first * 4;
                v.count = m;
                v.stride = 4;
                v.format = gfx::VertexFormat::F32;
                b.arrays.push_back(v);
                b.colormap = (look.colour == PointsColour::Classes ? kPaletteTable : uint32_t(look.colormap)) |
                             (dimmed ? kDimmed : 0u);
                b.vmin = look.colour == PointsColour::Classes ? 0.0f : look.vmin;
                b.vmax = look.colour == PointsColour::Classes ? 255.0f : look.vmax;
            }
            b.flat_rgba = dimmed ? dimmed_rgba(kDefaultRgba, tables_->ground) : kDefaultRgba;
            b.point_shape = PointShape::Square;
            b.size_px = look.size_px;
            b.full_strength = std::string_view(base) == "selection";   // the selected points: no fade
            if (look.has_bbox) {
                b.bounds.add({look.bbox[0], look.bbox[2], dims_ == 3 ? look.bbox[4] : 0.0});
                b.bounds.add({look.bbox[1], look.bbox[3], dims_ == 3 ? look.bbox[5] : 0.0});
            }
            const std::string path = chunk_path(base, k);
            std::string err;
            const EntityId e = tree_.ensure(path, &err);
            if (!e || !tree_.set_batch(e, b, false, &err)) {
                if (why) *why = err;
                return false;
            }
        }
        return true;
    };
    if (!add(all, "", dim)) return false;
    if (selected && selected->count && !add(*selected, "selection", false)) return false;
    tree_.update();
#if defined(CALIPER_POINTS_PIECES)
    set_pieces(all, selected, look);
#endif
    return true;
}

// ---- points@1 -----------------------------------------------------------------------------------------


bool PointsScene::drew_pieces() const {
#if defined(CALIPER_POINTS_PIECES)
    return pieces_ && !pieces_->failed;
#else
    return false;
#endif
}

bool PointsScene::pending() const {
#if defined(CALIPER_POINTS_PIECES)
    return pieces_ && pieces_->wait != nullptr;
#else
    return false;
#endif
}

#if defined(CALIPER_POINTS_PIECES)

namespace {

std::vector<uint8_t> pack_params(const cs::Piece& piece, std::initializer_list<std::pair<const char*, double>> values) {
    std::vector<uint8_t> out(piece.params.size() * 4, 0);
    for (size_t i = 0; i < piece.params.size(); ++i) {
        const cs::PieceParam& p = piece.params[i];
        double v = p.default_value;
        for (const auto& [name, given] : values)
            if (p.name == name) v = given;
        uint32_t bits = 0;
        if (p.type == cs::ParamType::f32) {
            const float f = float(v);
            std::memcpy(&bits, &f, 4);
        } else if (p.type == cs::ParamType::i32) {
            bits = uint32_t(int32_t(std::isfinite(v) ? std::clamp(v, -2147483648.0, 2147483647.0) : 0.0));
        } else {
            bits = uint32_t(std::isfinite(v) ? std::clamp(v, 0.0, 4294967295.0) : 0.0);
        }
        std::memcpy(out.data() + 4 * i, &bits, 4);
    }
    return out;
}

}  // namespace

void PointsScene::set_pieces(const PointsArrays& all, const PointsArrays* selected, const PointsLook& look) {
    if (!pieces_) {
        pieces_ = std::make_unique<Pieces>();
        pieces_->piece = caliper_draw::library_piece("points");
    }
    Pieces& P = *pieces_;
    if (P.wait) {
        P.wait->pos.release(gfx_);
        P.wait->aux.release(gfx_);
        P.wait.reset();
    }
    P.failed = false;
    ++P.generation;
    if (!P.piece) return;
    const bool need_aux = look.colour != PointsColour::Flat;
    if (all.count && (!all.pos_cpu || (need_aux && !all.aux_cpu))) {
        // The CPU cannot read the payload: its bytes come back through textures first, and until they
        // are in, the frames show the points that the programs hold. A selection needs CPU views, from
        // which the view stages it, thus none comes with such a payload.
        auto w = std::make_unique<Pieces::Wait>();
        w->all = all;
        w->look = look;
        gfx::Frame f;
        bool ok = true;
        if (!all.pos_cpu) ok = w->pos.start(gfx_, all.pos, all.pos_offset, all.count * 12, &f);
        if (ok && need_aux && !all.aux_cpu) ok = w->aux.start(gfx_, all.aux, all.aux_offset, all.count * 4, &f);
        if (ok && gfx_.submit(std::move(f)) != 0) {
            w->pos.read(gfx_);
            w->aux.read(gfx_);
            P.wait = std::move(w);
        } else {
            error_detail_ = "the copy of the payload for its readback was refused: " + gfx_.last_error();
            w->pos.release(gfx_);
            w->aux.release(gfx_);
            P.failed = true;
            error_ = "batch without arrays to convert";
        }
        return;
    }
    P.active = 0;
    add_rows(all, false, selected != nullptr, look);
    if (selected && selected->count) add_rows(*selected, true, false, look);
}

// The rows of one set of points for points@1's resource, one program for each part of at most
// kMaxItems points: the position (x, y, z, 0), with z 0 in 2-D, the colour packed as RGBA as the
// shared tables give it (the flat grey, the colormap's level by the colour keys, or the palette
// at every level, dimmed toward the ground for the rest of a selection), and the index in the set.
void PointsScene::add_rows(const PointsArrays& a, bool selection, bool dimmed, const PointsLook& look) {
    Pieces& P = *pieces_;
    const float* pos = static_cast<const float*>(a.pos_cpu);
    const float* aux = static_cast<const float*>(a.aux_cpu);
    uint32_t lut[256];
    gfx::shaders::Cmap cm{};
    const bool mapped = look.colour != PointsColour::Flat && aux;
    if (mapped) {
        const bool classes = look.colour == PointsColour::Classes;
        tables_->cache->fill(classes ? 0 : look.colormap, classes ? 0.0f : look.vmin, classes ? 255.0f : look.vmax, &cm);
        for (uint32_t l = 0; l < 256; ++l) {
            const uint32_t c = classes ? tables_->palette[l] : gfx::cmap::level_rgba(cm, l);
            lut[l] = dimmed ? dimmed_rgba(c, tables_->ground) : c;
        }
    }
    const uint32_t flat = dimmed ? dimmed_rgba(kDefaultRgba, tables_->ground) : kDefaultRgba;
    for (uint64_t first = 0; first < a.count; first += kMaxItems) {
        const uint64_t m = std::min<uint64_t>(a.count - first, kMaxItems);
        if (P.active == P.programs.size()) {
            P.programs.emplace_back();
            P.programs.back().id = P.active == 0 ? std::string("points") : "points/" + std::to_string(P.active);
        }
        Pieces::Program& pr = P.programs[P.active++];
        pr.rows.assign(size_t(m * kPointBytes), 0);
        uint8_t* row = pr.rows.data();
        for (uint64_t i = 0; i < m; ++i, row += kPointBytes) {
            const uint64_t k = first + i;
            const float xyzw[4] = {pos[3 * k], pos[3 * k + 1], dims_ == 3 ? pos[3 * k + 2] : 0.0f, 0.0f};
            uint32_t bits = flat;
            if (mapped) {
                uint32_t vb;
                std::memcpy(&vb, aux + k, 4);
                bits = lut[gfx::cmap::colormap_index(vb, cm)];
            }
            const uint32_t index = uint32_t(k);
            std::memcpy(row, xyzw, 16);
            std::memcpy(row + 16, &bits, 4);
            std::memcpy(row + 20, &index, 4);
        }
        pr.count = uint32_t(m);
        pr.selection = selection;
        pr.given = false;
    }
    P.size = look.size_px > 0 ? look.size_px : default_point_size(a.count);
}

bool PointsScene::render_pieces(float pixel_scale, uint32_t clear_rgba, gfx::Serial* serial, std::string* why) {
    if (!pieces_) set_pieces(PointsArrays{}, nullptr, PointsLook{});
    Pieces& P = *pieces_;
    if (!P.piece) {
        error_ = kNoPiece;
        P.failed = true;
        return false;
    }
    if (P.wait && P.wait->pos.poll(gfx_) && P.wait->aux.poll(gfx_)) {
        std::unique_ptr<Pieces::Wait> w = std::move(P.wait);
        if (w->pos.failed || w->aux.failed) {
            P.failed = true;
            error_ = "batch without arrays to convert";
            error_detail_ = "the readback of the payload failed: " + gfx_.last_error();
        } else {
            PointsArrays all = w->all;
            if (!w->pos.bytes.empty()) all.pos_cpu = w->pos.bytes.data();
            if (!w->aux.bytes.empty()) all.aux_cpu = w->aux.bytes.data();
            P.active = 0;
            ++P.generation;
            add_rows(all, false, false, w->look);
        }
    }
    if (P.failed) return false;
    // A frame requires every piece runtime to be ready.
    const bool webgpu = std::strcmp(gfx_.caps().backend, "webgpu") == 0 || std::strcmp(gfx_.caps().backend, "dawn") == 0;
    for (size_t i = 0; i < P.active; ++i) {
        Pieces::Program& pr = P.programs[i];
        if (!pr.runtime) {
            pr.runtime = std::make_unique<cd::DrawRuntime>(gfx_, webgpu);
            cd::Instance inst;
            inst.program = pr.id;
            inst.panel = "points";
            const cd::Verdict v = pr.runtime->install(*P.piece, inst, (1u << 24) + uint32_t(i));
            if (!v.ok() && why) *why = "points@1: " + v.code + " " + v.message;
        }
        if (!pr.given) {
            cd::ResourceVersion version;
            version.resource = "points";
            version.step = int64_t(P.generation);
            version.serial = P.generation;
            version.count = pr.count;
            version.bytes = pr.rows.data();
            version.size = pr.rows.size();
            const cd::Verdict v = pr.runtime->set_resource(version);
            if (!v.ok() && why) *why = "points@1: " + v.code + " " + v.message;
            std::vector<uint8_t>().swap(pr.rows);
            pr.given = true;
        }
        const cd::DrawState st = pr.runtime->status().state;
        if (st == cd::DrawState::refused || st == cd::DrawState::stopped) {
            const cd::DrawStatus status = pr.runtime->status();
            if (why && why->empty()) *why = "points@1: " + status.error.code + " " + status.error.message;
            error_ = kStopped;
            error_detail_ = status.error.code + " " + status.error.message;
            P.failed = true;
            return false;
        }
    }
    tree_.update();
    // The frame: the clear, then each program's passes in their own submits.
    gfx::RenderTarget target;
    target.color = color_;
    target.depth = depth_;
    target.clear_rgba = clear_rgba;
    gfx::Frame clear;
    clear.render(gfx::DrawList(target));
    gfx::Serial last = gfx_.submit(std::move(clear));
    if (last == 0) {
        if (why) *why = "the layer refused the frame: " + gfx_.last_error();
        return true;
    }
    const CameraMatrices cm = camera_matrices(camera_, tree_, width_, height_);
    if (cm.ok && P.active) {
        cd::FrameInputs base;
        const Mat4 clip = cm.proj * cm.view;
        Mat4 inv;
        if (!inverse(clip, &inv)) inv = Mat4{};
        float c[16], w[16], v[16];
        to_f32(clip, c);
        to_f32(inv, w);
        to_f32(cm.view, v);
        for (int i = 0; i < 16; ++i) {
            base.clip_from_world[size_t(i)] = c[i];
            base.world_from_clip[size_t(i)] = w[i];
            base.view_from_world[size_t(i)] = v[i];
        }
        base.eye = {float(cm.eye.x), float(cm.eye.y), float(cm.eye.z), cm.perspective ? 1.0f : 0.0f};
        base.viewport = {0.0f, 0.0f, float(width_), float(height_)};
        base.pixel_ratio = pixel_scale;
        base.cursor_step = base.latest_step = int64_t(P.generation);
        base.live = true;
        float att_ref = 0.0f, fade_w0 = 0.0f, fade_w1 = 0.0f;
        cue_values(&att_ref, &fade_w0, &fade_w1);
        cd::HostTarget host;
        host.color = color_;
        host.depth = depth_;
        host.width = width_;
        host.height = height_;
        for (size_t i = 0; i < P.active; ++i) {
            Pieces::Program& pr = P.programs[i];
            cd::FrameInputs in = base;
            // The selected points keep their colour at every distance, as the full strength.
            in.params = pack_params(*P.piece, {{"size", P.size},
                                               {"shape", 0},
                                               {"color_mode", 2},
                                               {"att_ref", att_ref},
                                               {"fade_w0", pr.selection ? 0.0f : fade_w0},
                                               {"fade_w1", pr.selection ? 0.0f : fade_w1}});
            std::vector<gfx::Frame> frames;
            if (!pr.runtime->encode(&frames, in, host)) continue;
            for (gfx::Frame& f : frames) {
                const gfx::Serial s = gfx_.submit(std::move(f));
                if (s == 0) {
                    pr.runtime->frame_refused(gfx_.last_error());
                    break;
                }
                last = s;
                // The view's own points carry no draw budget of an agent's program: each frame is done
                // without a judgement of its time.
                pr.runtime->frame_done(gfx::kGpuMsForgotten);
            }
        }
    }
    if (serial) *serial = last;
    return true;
}

#endif  // CALIPER_POINTS_PIECES

// ---- the camera ------------------------------------------------------------------------------------

void PointsScene::fit(double aspect) {
    tree_.update();
    const Aabb b = tree_.scene_bounds();
    caliper_scene::fit(camera_, b, aspect);
    if (Orbit* o = std::get_if<Orbit>(&camera_)) {   // the default three-quarter view, as every renderer
        o->azimuth = kDefaultAzimuth;
        o->elevation = kDefaultElevation;
    }
    Aabb u = b;
    if (!u.valid) {
        u.add({-1, -1, -1});
        u.add({1, 1, 1});
    }
    const Vec3 e = u.half_extent();
    if (dims_ == 3) {
        const double r = length(e);
        radius_ = (r > 1e-12 && std::isfinite(r)) ? r : 1.0;
    } else {
        radius_ = std::max(std::max(std::fabs(e.x), std::fabs(e.y)), 1e-6);
    }
}

void PointsScene::orbit(float dx, float dy) {
    if (Orbit* o = std::get_if<Orbit>(&camera_)) {
        o->azimuth += double(dx) * kOrbitRate;
        o->elevation = std::clamp(o->elevation + double(dy) * kOrbitRate, -kElevationBound, kElevationBound);
    }
}

void PointsScene::turn(double radians) {
    if (Orbit* o = std::get_if<Orbit>(&camera_)) o->azimuth += radians;
}

void PointsScene::set_angles(double azimuth, double elevation) {
    if (Orbit* o = std::get_if<Orbit>(&camera_)) {
        o->azimuth = azimuth;
        o->elevation = elevation;
    }
}

void PointsScene::cue_values(float* att_ref, float* fade_w0, float* fade_w1) const {
    *att_ref = *fade_w0 = *fade_w1 = 0.0f;
    const Orbit* o = std::get_if<Orbit>(&camera_);
    if (!cues_ || dims_ != 3 || !o) return;
    *att_ref = float(o->distance);
    *fade_w0 = float(o->distance);
    *fade_w1 = float(o->distance + radius_);
}

void PointsScene::pan(float dx, float dy, float view_height) {
    if (!(view_height > 0)) return;
    if (Orbit* o = std::get_if<Orbit>(&camera_)) {
        const Mat4 v = orbit_view(*o);   // rows 0 and 1 of the view's rotation: right and up
        const Vec3 s{v.at(0, 0), v.at(0, 1), v.at(0, 2)}, up{v.at(1, 0), v.at(1, 1), v.at(1, 2)};
        const double k = 2.0 * o->distance * std::tan(kOrbitFovY * 0.5) / view_height;
        const Vec3 d = (s * -double(dx) + up * double(dy)) * k;
        o->target[0] += d.x;
        o->target[1] += d.y;
        o->target[2] += d.z;
    } else if (PanZoom* p = std::get_if<PanZoom>(&camera_)) {
        const double k = 2.0 * p->half_height / view_height;
        p->center[0] -= double(dx) * k;
        p->center[1] += double(dy) * k;
    }
}

void PointsScene::zoom(float wheel, float mx, float my, double aspect) {
    if (wheel == 0.0f) return;
    const double f = std::pow(0.9, double(wheel));
    if (Orbit* o = std::get_if<Orbit>(&camera_)) {
        o->distance = std::clamp(o->distance * f, radius_ * 1e-3, radius_ * 1e3);
    } else if (PanZoom* p = std::get_if<PanZoom>(&camera_)) {
        const double wx = p->center[0] + mx * p->half_height * aspect, wy = p->center[1] + my * p->half_height;
        p->half_height = std::clamp(p->half_height * f, radius_ * 1e-5, radius_ * 1e5);
        p->center[0] = wx - mx * p->half_height * aspect;
        p->center[1] = wy - my * p->half_height;
    }
}

bool PointsScene::mvp(uint32_t width, uint32_t height, float out[16]) const {
    const CameraMatrices cm = camera_matrices(camera_, tree_, width, height);
    if (!cm.ok) return false;
    to_f32(cm.proj * cm.view, out);
    return true;
}

// ---- drawing -----------------------------------------------------------------------------------------

bool PointsScene::render(uint32_t width, uint32_t height, float pixel_scale, uint32_t clear_rgba, gfx::Serial* serial,
                         std::string* why) {
    if (serial) *serial = 0;
    if (width == 0 || height == 0) return true;
    const bool depth = dims_ == 3;
    if (!color_ || width != width_ || height != height_ || bool(depth_) != depth) {
        if (color_) gfx_.release(color_);
        if (depth_) gfx_.release(depth_);
        depth_ = {};
        gfx::TextureDesc d;
        d.width = width;
        d.height = height;
        d.format = gfx::Format::RGBA8Unorm;
        d.usage = gfx::kRenderTarget | gfx::kSampled | gfx::kCopySrcTex;
        d.label = "points view colour";
        color_ = gfx_.create_texture(d);
        if (depth) {
            d.format = gfx::Format::Depth32Float;
            d.usage = gfx::kRenderTarget;
            d.label = "points view depth";
            depth_ = gfx_.create_texture(d);
        }
        if (!color_ || (depth && !depth_)) {
            if (why) *why = "no render target of " + std::to_string(width) + " x " + std::to_string(height) + ": " +
                            gfx_.last_error();
            width_ = height_ = 0;
            return false;
        }
        width_ = width;
        height_ = height;
    }
#if defined(CALIPER_POINTS_PIECES)
    std::string err;
    const bool drawn = render_pieces(pixel_scale, clear_rgba, serial, &err);
    if (!drawn && err.empty()) err = "scene.piece_input: " + error_ + " " + error_detail_;
    if (!err.empty() && why) *why = err;
    return drawn && err.empty();
#else
    if (why) *why = "scene.pieces_unavailable: this build has no library pieces";
    return false;
#endif
}

uint64_t PointsScene::imgui_texture() const { return color_ ? gfx_.imgui_texture_id(color_) : 0; }

bool PointsScene::read_pixels(std::vector<uint8_t>* rgba, std::string* why) {
    if (!color_) {
        if (why) *why = "no render target yet";
        return false;
    }
    const gfx::Readback rb = gfx_.read_texture(color_, 0, 0, width_, height_);
    if (!rb || !gfx_.wait(gfx_.last_use(color_)) || !gfx_.take_readback(rb, rgba)) {
        if (why) *why = "the readback failed: " + gfx_.last_error();
        return false;
    }
    return true;
}

gfx::Stats PointsScene::poll() {
    gfx_.poll();
    return gfx_.take_stats();
}

}  // namespace caliper_live_views
