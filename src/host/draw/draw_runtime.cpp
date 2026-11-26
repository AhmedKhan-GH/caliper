// The draw half of a program in a viewer (draw_runtime.h,
// design/programmable-graphics.md, Render Graph, Draw State and Safety of
// Draws): the decoder of a PROGRAM's draw half, the checks before anything is
// made, the programs and pipelines of each pass, the resources, the draw
// buffers and the offscreen targets, the passes of one frame, the reset rules,
// the frame-budget warning and the removal of a hung program.
#include "draw/draw_runtime.h"
#include "draw/draw_runtime_impl.h"

#include <caliper_builder/programs/sandbox.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>

#if defined(CALIPER_HAS_WGSL)
#include <caliper/wgsl.h>
#endif

namespace caliper_draw {

namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;
namespace pl = caliper_builder::spec::piece_limits;
namespace gfx = caliper_gfx;

namespace {

constexpr std::uint32_t kWindowFrames = 30;      // frames in a row over draw.frame_ms before a warning
// A frame not done after kTimeoutMs by the wall clock is late. A training run on the same GPU can
// delay a correct frame that long, thus the wall clock alone stops a program only at kHangMs, and a
// late frame waits for its GPU time. A frame over kHeavyMs of GPU time, more than 15 times the
// largest draw.frame_ms, is heavy: heavy frames in a row stop the program once their GPU time passes
// kTimeoutMs together, thus a single frame over kTimeoutMs stops it at once, late or not. Where the
// backend gives no GPU time, kLateFrames late frames in a row stop it. A frame not done after kHangMs
// stops it at once: its pass may never end.
constexpr double kTimeoutMs = 1000.0;
constexpr double kHeavyMs = 250.0;
constexpr std::uint32_t kLateFrames = 3;
constexpr double kHangMs = 5000.0;
constexpr std::uint64_t kMiB = 1ull << 20;
constexpr std::uint32_t kFallbackColormaps = 4;   // the rows of colormap@1: viridis, magma, rdbu, greys

Verdict refusal(const char* code, std::string message, std::string where = {}) {
    Verdict v;
    v.code = code;
    v.message = std::move(message);
    v.where = std::move(where);
    return v;
}

std::string pass_where(std::size_t k) { return "draw.passes[" + std::to_string(k) + "]"; }

std::string fmt_ms(double ms) {
    char b[32];
    std::snprintf(b, sizeof b, "%.1f", ms);
    return b;
}

std::string fmt_limit(double ms) {
    char b[32];
    std::snprintf(b, sizeof b, "%g", ms);
    return b;
}

std::uint32_t pack_rgba(const std::array<float, 4>& c) {
    std::uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        const float v = std::isfinite(c[i]) ? std::min(1.0f, std::max(0.0f, c[i])) : 0.0f;
        r |= std::uint32_t(std::lround(v * 255.0f)) << (8 * i);
    }
    return r;
}

// ---- the manifest's JSON form (the keys of piece.toml) -------------------------------------

using json = nlohmann::json;

struct Reader {
    std::string error;
    bool fail(std::string what) {
        if (error.empty()) error = std::move(what);
        return false;
    }
    bool str(const json& o, const char* key, std::string* out, bool required = false) {
        const auto it = o.find(key);
        if (it == o.end()) return required ? fail(std::string("the key ") + key + " is missing") : true;
        if (!it->is_string()) return fail(std::string("the key ") + key + " is not a string");
        *out = it->get<std::string>();
        return true;
    }
    bool opt_str(const json& o, const char* key, std::optional<std::string>* out) {
        const auto it = o.find(key);
        if (it == o.end()) return true;
        if (!it->is_string()) return fail(std::string("the key ") + key + " is not a string");
        *out = it->get<std::string>();
        return true;
    }
    bool u32(const json& o, const char* key, std::uint32_t* out) {
        const auto it = o.find(key);
        if (it == o.end()) return true;
        if (!it->is_number_integer() && !it->is_number_unsigned())
            return fail(std::string("the key ") + key + " is not an integer");
        const auto v = it->get<std::int64_t>();
        if (v < 0 || v > 0xffffffffll) return fail(std::string("the key ") + key + " is out of range");
        *out = std::uint32_t(v);
        return true;
    }
    bool opt_u32(const json& o, const char* key, std::optional<std::uint32_t>* out) {
        if (o.find(key) == o.end()) return true;
        std::uint32_t v = 0;
        if (!u32(o, key, &v)) return false;
        *out = v;
        return true;
    }
    bool num(const json& o, const char* key, double* out) {
        const auto it = o.find(key);
        if (it == o.end()) return true;
        if (!it->is_number()) return fail(std::string("the key ") + key + " is not a number");
        *out = it->get<double>();
        return true;
    }
    bool opt_num(const json& o, const char* key, std::optional<double>* out) {
        if (o.find(key) == o.end()) return true;
        double v = 0;
        if (!num(o, key, &v)) return false;
        *out = v;
        return true;
    }
    bool flag(const json& o, const char* key, bool* out) {
        const auto it = o.find(key);
        if (it == o.end()) return true;
        if (!it->is_boolean()) return fail(std::string("the key ") + key + " is not a boolean");
        *out = it->get<bool>();
        return true;
    }
    bool strings(const json& o, const char* key, std::vector<std::string>* out) {
        const auto it = o.find(key);
        if (it == o.end()) return true;
        if (!it->is_array()) return fail(std::string("the key ") + key + " is not an array");
        for (const json& s : *it) {
            if (!s.is_string()) return fail(std::string("an element of ") + key + " is not a string");
            out->push_back(s.get<std::string>());
        }
        return true;
    }
    template <typename E>
    bool enum_of(const json& o, const char* key, std::optional<E> (*from)(std::string_view), E* out) {
        std::string s;
        if (o.find(key) == o.end()) return true;
        if (!str(o, key, &s)) return false;
        const auto e = from(s);
        if (!e) return fail(std::string("the key ") + key + " has the unknown value " + s);
        *out = *e;
        return true;
    }
    // `element`, or `struct` with `fields` as "name: type" strings or {name, type} tables.
    bool layout(const json& o, cs::Layout* out) {
        if (!opt_str(o, "element", &out->element) || !opt_str(o, "struct", &out->type_name)) return false;
        const auto it = o.find("fields");
        if (it == o.end()) return true;
        if (!it->is_array()) return fail("the key fields is not an array");
        for (const json& f : *it) {
            cs::Field field;
            if (f.is_string()) {
                const std::string s = f.get<std::string>();
                const auto colon = s.find(':');
                if (colon == std::string::npos) return fail("the field " + s + " is not written name: type");
                auto trim = [](std::string t) {
                    const auto a = t.find_first_not_of(" \t");
                    const auto b = t.find_last_not_of(" \t");
                    return a == std::string::npos ? std::string() : t.substr(a, b - a + 1);
                };
                field.name = trim(s.substr(0, colon));
                field.type = trim(s.substr(colon + 1));
            } else if (f.is_object()) {
                if (!str(f, "name", &field.name, true) || !str(f, "type", &field.type, true)) return false;
            } else {
                return fail("a field is neither a string nor a table");
            }
            out->fields.push_back(std::move(field));
        }
        return true;
    }
    bool workgroups(const json& o, std::array<std::uint32_t, 3>* out) {
        const auto it = o.find("workgroups");
        if (it == o.end()) return true;
        if (!it->is_array() || it->size() < 1 || it->size() > 3) return fail("workgroups is not 1 to 3 integers");
        for (std::size_t i = 0; i < it->size(); ++i) {
            const json& v = (*it)[i];
            if (!v.is_number_integer() || v.get<std::int64_t>() < 0 || v.get<std::int64_t>() > 0xffffffffll)
                return fail("an element of workgroups is not a count");
            (*out)[i] = std::uint32_t(v.get<std::int64_t>());
        }
        return true;
    }
    template <typename T, typename F>
    bool tables(const json& o, const char* key, std::vector<T>* out, F each) {
        const auto it = o.find(key);
        if (it == o.end()) return true;
        if (!it->is_array()) return fail(std::string("the key ") + key + " is not an array of tables");
        for (const json& t : *it) {
            if (!t.is_object()) return fail(std::string("an element of ") + key + " is not a table");
            T v;
            if (!each(t, &v)) return false;
            out->push_back(std::move(v));
        }
        return true;
    }
};

bool read_manifest(const json& m, cs::Piece* p, std::string* why) {
    Reader r;
    if (!m.is_object()) {
        *why = "the manifest is not a table";
        return false;
    }
    bool ok = r.str(m, "name", &p->name, true) && r.u32(m, "version", &p->version) &&
              r.str(m, "description", &p->description) && r.strings(m, "imports", &p->imports) &&
              r.str(m, "hash", &p->hash);
    ok = ok && r.tables(m, "inputs", &p->inputs, [&](const json& t, cs::PieceInput* in) {
        return r.str(t, "name", &in->name, true) && r.enum_of(t, "kind", cs::input_kind_from_name, &in->kind) &&
               r.str(t, "dtype", &in->dtype) && r.u32(t, "rank_min", &in->rank_min) &&
               r.u32(t, "rank_max", &in->rank_max) && r.layout(t, &in->layout) && r.opt_str(t, "title", &in->title) &&
               r.flag(t, "optional", &in->optional);
    });
    ok = ok && r.tables(m, "params", &p->params, [&](const json& t, cs::PieceParam* pa) {
        return r.str(t, "name", &pa->name, true) && r.enum_of(t, "type", cs::param_type_from_name, &pa->type) &&
               r.num(t, "default", &pa->default_value) && r.opt_num(t, "min", &pa->min) &&
               r.opt_num(t, "max", &pa->max) && r.opt_str(t, "title", &pa->title);
    });
    ok = ok && r.tables(m, "resources", &p->resources, [&](const json& t, cs::Resource* res) {
        return r.str(t, "id", &res->id, true) && r.enum_of(t, "kind", cs::resource_kind_from_name, &res->kind) &&
               r.layout(t, &res->layout) && r.u32(t, "count", &res->count) && r.flag(t, "counted", &res->counted) &&
               r.u32(t, "width", &res->width) && r.u32(t, "height", &res->height) &&
               r.opt_str(t, "format", &res->format) && r.opt_u32(t, "texels", &res->texels) &&
               r.flag(t, "publish", &res->publish) && r.flag(t, "clear", &res->clear) &&
               r.opt_u32(t, "keep", &res->keep);
    });
    if (ok) {
        const auto it = m.find("depends");
        if (it != m.end()) {
            if (!it->is_array()) ok = r.fail("the key depends is not an array");
            for (std::size_t i = 0; ok && i < it->size(); ++i) {
                const json& d = (*it)[i];
                cs::PieceDependency dep;
                if (d.is_string()) {
                    const std::string s = d.get<std::string>();
                    const auto at = s.find('@');
                    dep.name = s.substr(0, at);
                    if (at != std::string::npos) dep.version = std::uint32_t(std::strtoul(s.c_str() + at + 1, nullptr, 10));
                } else if (d.is_object()) {
                    ok = r.str(d, "name", &dep.name, true) && r.u32(d, "version", &dep.version);
                } else {
                    ok = r.fail("an element of depends is neither a string nor a table");
                }
                if (ok) p->depends.push_back(dep);
            }
        }
    }
    if (ok) {
        // The canonical form (programs/pieces.cpp, canonical_manifest): an array of {input, shape}, whose
        // file is sample/<input>.bin, as the JSON form of a piece reads it.
        const auto it = m.find("sample");
        if (it != m.end()) {
            if (!it->is_array()) ok = r.fail("the key sample is not an array");
            for (std::size_t k = 0; ok && k < it->size(); ++k) {
                const json& s = (*it)[k];
                cs::SampleInput in;
                std::string input;
                if (!s.is_object()) {
                    ok = r.fail("a sample is not a table");
                    break;
                }
                ok = r.str(s, "input", &input, true);
                const auto shape = s.find("shape");
                if (ok && shape == s.end()) ok = r.fail("a sample has no shape");
                if (ok && !shape->is_array()) ok = r.fail("a sample's shape is not an array");
                for (std::size_t i = 0; ok && i < shape->size(); ++i) {
                    const json& v = (*shape)[i];
                    if (!v.is_number_integer() || v.get<std::int64_t>() < 0 || v.get<std::int64_t>() > 0xFFFFFFFFll)
                        ok = r.fail("a sample's shape is not axis lengths");
                    else
                        in.shape.push_back(std::uint32_t(v.get<std::int64_t>()));
                }
                if (ok && p->sample.count(input)) ok = r.fail("two samples name the input " + input);
                in.file = "sample/" + input + ".bin";
                if (ok) p->sample[input] = std::move(in);
            }
        }
    }
    if (ok && m.contains("derive")) {
        const json& d = m["derive"];
        if (!d.is_object()) {
            ok = r.fail("the key derive is not a table");
        } else {
            cs::Derive derive;
            ok = r.tables(d, "passes", &derive.passes, [&](const json& t, cs::DerivePass* dp) {
                     return r.str(t, "entry", &dp->entry, true) && r.workgroups(t, &dp->workgroups) &&
                            r.strings(t, "writes", &dp->writes);
                 }) &&
                 r.opt_num(d, "time_ms", &derive.time_ms) && r.opt_u32(d, "memory_mb", &derive.memory_mb) &&
                 r.opt_u32(d, "every", &derive.every);
            if (ok) p->derive = std::move(derive);
        }
    }
    if (ok && m.contains("draw")) {
        const json& d = m["draw"];
        if (!d.is_object()) {
            ok = r.fail("the key draw is not a table");
        } else {
            cs::Draw draw;
            ok = r.enum_of(d, "camera", cs::camera_mode_from_name, &draw.camera);
            ok = ok && r.tables(d, "buffers", &draw.buffers, [&](const json& t, cs::DrawBuffer* b) {
                return r.str(t, "id", &b->id, true) && r.layout(t, &b->layout) && r.u32(t, "count", &b->count) &&
                       r.enum_of(t, "reset", cs::buffer_reset_from_name, &b->reset);
            });
            ok = ok && r.tables(d, "targets", &draw.targets, [&](const json& t, cs::DrawTarget* tg) {
                double scale = 1.0;
                if (!r.str(t, "id", &tg->id, true) || !r.num(t, "scale", &scale)) return false;
                tg->scale = float(scale);
                const auto c = t.find("clear");
                if (c != t.end()) {
                    if (!c->is_array() || c->size() != 4) return r.fail("a target's clear is not four numbers");
                    for (int i = 0; i < 4; ++i) {
                        if (!(*c)[i].is_number()) return r.fail("a target's clear is not four numbers");
                        tg->clear[i] = (*c)[i].get<float>();
                    }
                }
                return true;
            });
            ok = ok && r.tables(d, "passes", &draw.passes, [&](const json& t, cs::DrawPass* dp) {
                if (!r.enum_of(t, "kind", cs::pass_kind_from_name, &dp->kind) || !r.str(t, "entry", &dp->entry) ||
                    !r.workgroups(t, &dp->workgroups) || !r.strings(t, "writes", &dp->writes) ||
                    !r.opt_str(t, "vertex", &dp->vertex) || !r.str(t, "fragment", &dp->fragment) ||
                    !r.str(t, "target", &dp->target) || !r.strings(t, "reads", &dp->reads) ||
                    !r.opt_u32(t, "vertices", &dp->vertices) ||
                    !r.enum_of(t, "topology", cs::draw_topology_from_name, &dp->topology) ||
                    !r.flag(t, "pick", &dp->pick) || !r.opt_str(t, "indices", &dp->indices))
                    return false;
                const auto inst = t.find("instances");
                if (inst != t.end()) {
                    if (inst->is_string()) dp->instances_of = inst->get<std::string>();
                    else if (!r.u32(t, "instances", &dp->instances)) return false;
                }
                if (t.contains("blend")) {
                    cs::DrawBlend b = cs::DrawBlend::opaque;
                    if (!r.enum_of(t, "blend", cs::draw_blend_from_name, &b)) return false;
                    dp->blend = b;
                }
                if (t.contains("depth")) {
                    cs::DrawDepth dd = cs::DrawDepth::none;
                    if (!r.enum_of(t, "depth", cs::draw_depth_from_name, &dd)) return false;
                    dp->depth = dd;
                }
                return true;
            });
            if (ok && d.contains("pick")) {
                cs::PickMap pm;
                const json& t = d["pick"];
                ok = t.is_object() ? (r.str(t, "resource", &pm.resource, true) && r.strings(t, "fields", &pm.fields))
                                   : r.fail("the key pick is not a table");
                if (ok) draw.pick = pm;
            }
            if (ok && d.contains("text")) {
                cs::TextTable tt;
                const json& t = d["text"];
                ok = t.is_object() ? (r.str(t, "anchors", &tt.anchors, true) && r.str(t, "position", &tt.position) &&
                                      r.str(t, "string", &tt.string) && r.opt_str(t, "priority", &tt.priority) &&
                                      r.strings(t, "strings", &tt.strings) &&
                                      r.opt_str(t, "strings_from", &tt.strings_from) &&
                                      r.enum_of(t, "align", cs::text_align_from_name, &tt.align))
                                   : r.fail("the key text is not a table");
                if (ok && t.contains("hover")) {
                    const json& h = t["hover"];
                    cs::TextHover hover;
                    ok = h.is_object() && h.contains("anchor") &&
                         r.u32(h, "anchor", &hover.anchor) && r.u32(h, "string_offset", &hover.string_offset);
                    if (!ok) r.fail("the hover label needs an anchor and an optional string_offset");
                    else tt.hover = hover;
                }
                if (ok && t.contains("selected")) {
                    const json& s = t["selected"];
                    cs::TextSelected selected;
                    ok = s.is_object() && s.contains("anchor") &&
                         r.u32(s, "anchor", &selected.anchor) && r.str(s, "param", &selected.param, true) &&
                         r.u32(s, "string_offset", &selected.string_offset);
                    if (!ok) r.fail("the selected label needs an anchor, param, and optional string_offset");
                    else tt.selected = selected;
                }
                if (ok) draw.text = tt;
            }
            ok = ok && r.opt_num(d, "frame_ms", &draw.frame_ms) && r.opt_u32(d, "memory_mb", &draw.memory_mb);
            if (ok) p->draw = std::move(draw);
        }
    }
    if (!ok) *why = r.error;
    return ok;
}

// The author's text of a section that may hold the imports resolved: without the lib modules'
// text in front, with "<module>__" written back as "<module>::".
std::string strip_libs(std::string_view section, const std::vector<std::string>& imports) {
    std::string libs;
    for (const std::string& m : imports) {
        std::string t(cp::lib_module(m));
        if (!t.empty() && t.back() != '\n') t += '\n';
        libs += t;
    }
    if (!libs.empty() && section.substr(0, libs.size()) == libs) return std::string(section.substr(libs.size()));
    return std::string(section);
}

std::string unrewrite(std::string text, const std::vector<std::string>& imports) {
    for (const std::string& m : imports) {
        const std::string name = m.substr(0, m.find('@'));
        const std::string from = name + "__", to = name + "::";
        for (std::size_t at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size()))
            text.replace(at, from.size(), to);
    }
    return text;
}

// ---- the prelude's values --------------------------------------------------------------

constexpr std::array<float, 16> kIdentity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

void put_f32(std::uint8_t* at, float v) { std::memcpy(at, &v, 4); }
void put_u32(std::uint8_t* at, std::uint32_t v) { std::memcpy(at, &v, 4); }

std::uint32_t step_u32(std::int64_t s) { return s < 0 ? 0u : s > 0xffffffffll ? 0xffffffffu : std::uint32_t(s); }

// One binding of a pass: what the prelude, the header or the piece names at (group, binding).
enum class Object { view, selection, linear, nearest, colormaps, params, counts, slot, buffer, target, none };

}  // namespace

// ---- checks ------------------------------------------------------------------------------

DrawChecks DrawChecks::pieces() {
    DrawChecks c;
    c.manifest = cp::check_piece_manifest;
    c.source = cp::check_piece_source;
    c.resolve = cp::resolve_imports;
    c.layout = cp::element_layout;
    c.hash = cp::piece_hash;
    return c;
}

// ---- construction ------------------------------------------------------------------------

DrawRuntime::DrawRuntime(caliper_gfx::Backend& backend, bool webgpu, DrawChecks checks)
    : impl_(std::make_unique<Impl>()) {
    impl_->gfx = &backend;
    impl_->webgpu = webgpu;
    impl_->checks = checks;
}

DrawRuntime::~DrawRuntime() { remove(); }

namespace {

void release_objects(DrawRuntime::Impl& m) {
    gfx::Backend& g = *m.gfx;
    for (auto& p : m.passes) {
        for (auto& bg : p.groups)
            if (bg) g.release(bg);
        for (auto& pipe : p.pipelines)
            if (pipe) g.release(pipe);
        if (p.program) g.release(p.program);
    }
    m.passes.clear();
    for (auto& s : m.slots) {
        if (s.buffer) g.release(s.buffer);
        if (s.tex) g.release(s.tex);
    }
    m.slots.clear();
    for (auto& b : m.buffers)
        if (b.buffer) g.release(b.buffer);
    m.buffers.clear();
    for (auto& t : m.targets)
        if (t.tex) g.release(t.tex);
    m.targets.clear();
    for (gfx::Buffer* b : {&m.view, &m.selection, &m.params, &m.counts})
        if (*b) {
            g.release(*b);
            *b = {};
        }
    if (m.linear) g.release(m.linear);
    if (m.nearest) g.release(m.nearest);
    if (m.colormaps) g.release(m.colormaps);
    m.linear = {};
    m.nearest = {};
    m.colormaps = {};
    m.bound_colormaps = {};
    m.memory_bytes = 0;
    m.expected.clear();
}

// Stops the program: its state, its error naming the pass, and its objects released. The GPU
// finishes what it runs; the layer destroys each object after its last use completed.
void stop(DrawRuntime::Impl& m, const char* code, const std::string& message, int pass, std::string where = {}) {
    m.status.state = DrawState::stopped;
    m.status.warning = Verdict{};
    m.panel_warned = false;
    m.status.error = refusal(code, message, where.empty() ? (pass >= 0 ? pass_where(std::size_t(pass)) : "draw")
                                                          : where);
    m.status.pass = pass;
    m.status.entry.clear();
    if (pass >= 0 && m.piece.draw && std::size_t(pass) < m.piece.draw->passes.size()) {
        const cs::DrawPass& p = m.piece.draw->passes[std::size_t(pass)];
        m.status.entry = p.kind == cs::PassKind::compute ? p.entry : p.fragment;
    }
    m.status.memory_bytes = 0;
    release_objects(m);
}

// Records a frame-budget warning on a running program without releasing its GPU objects.
void warn(DrawRuntime::Impl& m, const char* code, const std::string& message, int pass, std::string where = {}) {
    m.status.warning = refusal(code, message, where.empty() ? (pass >= 0 ? pass_where(std::size_t(pass)) : "draw")
                                                            : where);
    m.status.pass = pass;
    m.status.entry.clear();
    if (pass >= 0 && m.piece.draw && std::size_t(pass) < m.piece.draw->passes.size()) {
        const cs::DrawPass& p = m.piece.draw->passes[std::size_t(pass)];
        m.status.entry = p.kind == cs::PassKind::compute ? p.entry : p.fragment;
    }
}

void clear_warning(DrawRuntime::Impl& m) {
    if (m.status.warning.ok()) return;
    m.status.warning = Verdict{};
    m.panel_warned = false;
    if (m.status.error.ok()) {
        m.status.pass = -1;
        m.status.entry.clear();
    }
}

std::string pass_name(const DrawRuntime::Impl& m, int pass) {
    const cs::DrawPass& p = m.piece.draw->passes[std::size_t(pass)];
    return "draw pass " + std::to_string(pass) + " (" + (p.kind == cs::PassKind::compute ? p.entry : p.fragment) + ")";
}

// A sampled texture of a texture resource's format and the given size, which each version fills
// with write_texture.
gfx::Texture make_texture(gfx::Backend& g, const DrawRuntime::Impl::Slot& s, std::uint32_t w, std::uint32_t h) {
    gfx::TextureDesc d;
    d.width = w;
    d.height = h;
    d.format = s.format;
    d.usage = gfx::kSampled | gfx::kCopyDstTex;
    d.label = s.id.c_str();
    return g.create_texture(d);
}

// Accounts `add` more bytes against draw.memory_mb in place of `drop`; false (and the program
// stopped with program.draw_over_memory, naming `what`) when the total would pass it.
bool account(DrawRuntime::Impl& m, std::uint64_t drop, std::uint64_t add, const std::string& what,
             const std::string& where) {
    const std::uint64_t base = m.memory_bytes >= drop ? m.memory_bytes - drop : 0;
    const std::uint64_t limit = m.status.memory_limit_bytes;
    if (add > limit || base > limit - add) {
        stop(m, cp::kProgramDrawOverMemory,
             "Stopped " + m.instance.program + ": " + what + " would bring its draw memory to " +
                 std::to_string((base + std::min<std::uint64_t>(add, limit + 1)) / kMiB) + " MiB, above its " +
                 std::to_string(limit / kMiB) + " MiB",
             -1, where);
        return false;
    }
    m.memory_bytes = base + add;
    m.status.memory_bytes = m.memory_bytes;
    return true;
}

// Zeroes a buffer through writes of one shared block of zeros.
bool zero_buffer(gfx::Backend& g, gfx::Buffer b, std::uint64_t bytes) {
    static const std::vector<std::uint8_t> zeros(std::size_t(1) << 20, 0);
    for (std::uint64_t at = 0; at < bytes;) {
        const std::uint64_t n = std::min<std::uint64_t>(zeros.size(), bytes - at);
        if (!g.write_buffer(b, at, zeros.data(), n)) return false;
        at += n;
    }
    return true;
}

gfx::Topology topology_of(cs::DrawTopology t) {
    switch (t) {
        case cs::DrawTopology::triangle_list: return gfx::Topology::TriangleList;
        case cs::DrawTopology::triangle_strip: return gfx::Topology::TriangleStrip;
        case cs::DrawTopology::line_list: return gfx::Topology::LineList;
        case cs::DrawTopology::line_strip: return gfx::Topology::LineStrip;
        case cs::DrawTopology::point_list: return gfx::Topology::PointList;
    }
    return gfx::Topology::TriangleList;
}

bool offscreen(const cs::DrawPass& p) { return p.kind == cs::PassKind::render && p.target != "host"; }

cs::DrawBlend blend_of(const cs::DrawPass& p) {
    return p.blend ? *p.blend : offscreen(p) ? cs::DrawBlend::additive : cs::DrawBlend::opaque;
}

cs::DrawDepth depth_of(const cs::DrawPass& p) {
    if (offscreen(p)) return cs::DrawDepth::none;
    return p.depth ? *p.depth : p.vertex ? cs::DrawDepth::write : cs::DrawDepth::none;
}

// What a pass binds at (group, binding), and its binding type as the header declares it.
Object object_at(const DrawRuntime::Impl& m, const cs::DrawPass& p, std::uint32_t group, std::uint32_t binding,
                 gfx::BindingType* type, std::size_t* index) {
    *index = 0;
    if (group == cp::kGroupPrelude) {
        switch (binding) {
            case cp::kBindingView: *type = gfx::BindingType::Uniform; return Object::view;
            case cp::kBindingSelection: *type = gfx::BindingType::Storage; return Object::selection;
            case cp::kBindingLinear: *type = gfx::BindingType::Sampler; return Object::linear;
            case cp::kBindingNearest: *type = gfx::BindingType::Sampler; return Object::nearest;
            case cp::kBindingColormaps: *type = gfx::BindingType::Texture; return Object::colormaps;
            case cp::kBindingParams:
                if (m.piece.params.empty()) return Object::none;
                *type = gfx::BindingType::Uniform;
                return Object::params;
            default: return Object::none;
        }
    }
    if (group == cp::kGroupInputs) {
        if (binding == 0) {
            if (m.slots.empty()) return Object::none;
            *type = gfx::BindingType::Uniform;
            return Object::counts;
        }
        if (binding - 1 >= m.slots.size()) return Object::none;
        *index = binding - 1;
        const DrawRuntime::Impl::Slot& s = m.slots[*index];
        *type = !s.texture                              ? gfx::BindingType::Storage
                : s.format == gfx::Format::R32Uint ? gfx::BindingType::UintTexture
                                                        : gfx::BindingType::Texture;
        return Object::slot;
    }
    if (group == cp::kGroupResources) {
        if (binding < 4) {
            if (binding >= m.buffers.size()) return Object::none;
            *index = binding;
            const bool rw = p.kind == cs::PassKind::compute &&
                            std::find(p.writes.begin(), p.writes.end(), m.buffers[binding].id) != p.writes.end();
            *type = rw ? gfx::BindingType::StorageReadWrite : gfx::BindingType::Storage;
            return Object::buffer;
        }
        if (binding - 4 >= m.targets.size()) return Object::none;
        *index = binding - 4;
        if (std::find(p.reads.begin(), p.reads.end(), m.targets[*index].id) == p.reads.end()) return Object::none;
        *type = gfx::BindingType::Texture;
        return Object::target;
    }
    return Object::none;
}

// The smallest binding of a storage object: one element (WebGPU's minimum binding size of a
// runtime-sized array), 0 for the other kinds.
std::uint32_t storage_size(const DrawRuntime::Impl& m, Object o, std::size_t index) {
    switch (o) {
        case Object::selection: return cp::kSelectedBytes;
        case Object::slot: return m.slots[index].texture ? 0u : m.slots[index].layout.stride;
        case Object::buffer: return m.buffers[index].stride;
        default: return 0;
    }
}

// The place of a line of a composed module in the author's file.
std::string place(const cp::Composed& c, std::uint32_t line, std::uint32_t column, std::uint32_t* file_line) {
    for (const cp::Segment& s : c.segments)
        if (line >= s.first_line && line < s.first_line + s.lines) {
            *file_line = line - s.first_line + 1;
            return s.part + " line " + std::to_string(*file_line) + ", column " + std::to_string(column);
        }
    *file_line = line;
    return "line " + std::to_string(line);
}

#if defined(CALIPER_HAS_WGSL)
// Joins the MSL of each entry point of one module: their common leading lines once, then what
// follows in each (caliper-wgsl translates one entry point at a time).
std::string merge_msl(const std::vector<std::string>& texts) {
    if (texts.empty()) return {};
    std::string out = texts[0];
    for (std::size_t i = 1; i < texts.size(); ++i) {
        const std::string& t = texts[i];
        std::size_t common = 0, at = 0;
        while (at < t.size() && at < texts[0].size()) {
            const std::size_t e0 = texts[0].find('\n', at), e1 = t.find('\n', at);
            if (e0 == std::string::npos || e1 == std::string::npos || e0 != e1 ||
                texts[0].compare(at, e0 - at, t, at, e1 - at) != 0)
                break;
            at = e0 + 1;
            common = at;
        }
        out += t.substr(common);
    }
    return out;
}
#endif


// Compiles one composed pass with caliper-wgsl and its bounds checks, checks its reflection
// (Check Order 6 and 7), and fills the pass's MSL, entries and bindings with the stages and
// the Metal indices that the compiler gave. Each reflected binding must be one that the
// header declares, with the header's type.
Verdict compile_pass(DrawRuntime::Impl& m, std::size_t k, const cp::Composed& c, DrawRuntime::Impl::PassSlot* out) {
    const std::string where = pass_where(k);
    const cs::DrawPass& p = m.piece.draw->passes[k];
#if defined(CALIPER_HAS_WGSL)
    if (caliper_wgsl_abi_version() != CALIPER_WGSL_ABI_VERSION)
        return refusal(cp::kProgramCompilerUnavailable, "The WGSL compiler's interface differs from the app's.", where);
    caliper_wgsl_options options;
    caliper_wgsl_options_init(&options);
    options.bounds_checks = CALIPER_WGSL_BOUNDS_CHECKED;   // every access of a piece stays checked
    caliper_wgsl_module* module = nullptr;
    caliper_wgsl_error error{};
    if (caliper_wgsl_module_create(c.text.data(), c.text.size(), &options, &module, &error) != CALIPER_WGSL_OK) {
        std::uint32_t line = 0;
        const std::string at = place(c, error.line, error.column, &line);
        Verdict v = refusal(cp::kProgramCompileFailed,
                            "Draw pass " + std::to_string(k) + " does not compile, at " + at + ": " +
                                (error.message ? error.message : "the compiler refused it"),
                            where);
        v.line = line;
        v.column = error.column;
        caliper_wgsl_error_free(&error);
        return v;
    }
    struct Guard {
        caliper_wgsl_module* m;
        ~Guard() { caliper_wgsl_module_destroy(m); }
    } guard{module};

    cp::Reflection reflection;
    for (std::size_t i = 0, n = caliper_wgsl_module_entry_point_count(module); i < n; ++i) {
        caliper_wgsl_entry_point e{};
        if (caliper_wgsl_module_entry_point(module, i, &e) != CALIPER_WGSL_OK) continue;
        cp::ReflectedEntry r;
        r.name = e.name ? e.name : "";
        r.stage = cp::EntryStage(e.stage);
        for (int a = 0; a < 3; ++a) r.workgroup_size[a] = e.workgroup_size[a];
        r.workgroup_memory = e.workgroup_memory;
        // The inputs and outputs, for the check of the stage interfaces (Check Order 7).
        for (std::size_t k = 0, nio = caliper_wgsl_module_entry_io_count(module, i); k < nio; ++k) {
            caliper_wgsl_io io{};
            if (caliper_wgsl_module_entry_io(module, i, k, &io) != CALIPER_WGSL_OK) continue;
            cp::ReflectedIo v;
            v.location = io.location;
            v.scalar = cp::IoScalar(io.scalar);
            v.components = io.components;
            v.interpolation = cp::IoInterpolation(io.interpolation);
            v.sampling = cp::IoSampling(io.sampling);
            (io.direction == CALIPER_WGSL_IO_INPUT ? r.inputs : r.outputs).push_back(v);
        }
        reflection.entries.push_back(std::move(r));
    }
    std::vector<caliper_wgsl_binding> raw;
    for (std::size_t i = 0, n = caliper_wgsl_module_binding_count(module); i < n; ++i) {
        caliper_wgsl_binding b{};
        if (caliper_wgsl_module_binding(module, i, &b) != CALIPER_WGSL_OK) continue;
        cp::ReflectedBinding r;
        r.group = b.group;
        r.binding = b.binding;
        r.kind = cp::BindingKind(b.kind);
        r.read = (b.access & CALIPER_WGSL_ACCESS_READ) != 0;
        r.write = (b.access & CALIPER_WGSL_ACCESS_WRITE) != 0;
        r.size = b.size;
        r.name = b.name ? b.name : "";
        reflection.bindings.push_back(r);
        raw.push_back(b);
    }
    if (Verdict v = cp::check_piece_reflection(m.piece, cp::Half::draw, k, reflection); !v.ok()) return v;

    for (std::size_t i = 0; i < raw.size(); ++i) {
        const caliper_wgsl_binding& b = raw[i];
        gfx::BindingType type{};
        std::size_t index = 0;
        const Object o = object_at(m, p, b.group, b.binding, &type, &index);
        if (o == Object::none)
            return refusal(b.group == cp::kGroupResources && b.binding >= 4 ? cp::kProgramTargetCycle
                                                                            : cp::kProgramBindingNotAllowed,
                           "Draw pass " + std::to_string(k) + " uses " + reflection.bindings[i].name +
                               (b.group == cp::kGroupResources && b.binding >= 4
                                    ? ", a target that the pass does not name in reads."
                                    : ", which its header does not declare."),
                           where);
        gfx::BindingDesc d;
        d.group = std::uint8_t(b.group);
        d.binding = std::uint8_t(b.binding);
        d.type = type;
        d.stages = std::uint8_t(b.stages & 7u);
        d.size = type == gfx::BindingType::Uniform
                     ? std::uint32_t((b.size + 15) / 16 * 16)
                     : std::max<std::uint32_t>(std::uint32_t(std::min<std::uint64_t>(b.size, 0xffffffffu)),
                                               storage_size(m, o, index));
        d.msl_index = std::uint8_t(b.metal_index);
        out->bindings.push_back(d);
    }

    std::vector<std::pair<std::string, caliper_wgsl_stage>> entries;
    if (p.kind == cs::PassKind::compute) {
        entries.emplace_back(p.entry, CALIPER_WGSL_STAGE_COMPUTE);
    } else {
        entries.emplace_back(p.vertex ? *p.vertex : "caliper_fullscreen", CALIPER_WGSL_STAGE_VERTEX);
        entries.emplace_back(p.fragment, CALIPER_WGSL_STAGE_FRAGMENT);
    }
    std::vector<std::string> texts;
    for (const auto& [name, stage] : entries) {
        caliper_wgsl_buffer text{};
        if (caliper_wgsl_translate_msl(module, name.c_str(), stage, &text, &error) != CALIPER_WGSL_OK) {
            Verdict v = refusal(cp::kProgramCompileFailed,
                                "Draw pass " + std::to_string(k) + " does not translate to MSL at " + name + ": " +
                                    (error.message ? error.message : "the translation failed"),
                                where);
            caliper_wgsl_error_free(&error);
            return v;
        }
        texts.emplace_back(reinterpret_cast<const char*>(text.data), text.size);
        caliper_wgsl_buffer_free(&text);
        out->names.push_back(name);
        gfx::EntryDesc e;
        e.stage = stage == CALIPER_WGSL_STAGE_VERTEX ? gfx::kStageVertex
                  : stage == CALIPER_WGSL_STAGE_FRAGMENT ? gfx::kStageFragment
                                                         : gfx::kStageCompute;
        for (const cp::ReflectedEntry& r : reflection.entries)
            if (r.name == name)
                for (int a = 0; a < 3; ++a) e.workgroup_size[a] = std::uint16_t(std::max<std::uint32_t>(1, r.workgroup_size[a]));
        out->entries.push_back(e);
    }
    out->msl = merge_msl(texts);
    return {};
#else
    (void)m;
    (void)c;
    (void)out;
    (void)p;
    return refusal(cp::kProgramCompilerUnavailable, "This build has no WGSL compiler for the draw half.", where);
#endif
}

// The bindings of a pass without a compiler (WebGPU takes the WGSL as it is): every binding of
// the prelude and the header, in every stage of the pass.
void plan_bindings(DrawRuntime::Impl& m, std::size_t k, DrawRuntime::Impl::PassSlot* out) {
    const cs::DrawPass& p = m.piece.draw->passes[k];
    const std::uint8_t stages =
        p.kind == cs::PassKind::compute ? gfx::kStageCompute : std::uint8_t(gfx::kStageVertex | gfx::kStageFragment);
    auto add = [&](std::uint32_t group, std::uint32_t binding, std::uint32_t size) {
        gfx::BindingType type{};
        std::size_t index = 0;
        const Object o = object_at(m, p, group, binding, &type, &index);
        if (o == Object::none) return;
        gfx::BindingDesc d;
        d.group = std::uint8_t(group);
        d.binding = std::uint8_t(binding);
        d.type = type;
        d.stages = stages;
        d.size = std::max(size, storage_size(m, o, index));
        out->bindings.push_back(d);
    };
    for (std::uint32_t b = 0; b <= cp::kBindingParams; ++b)
        add(cp::kGroupPrelude, b, b == cp::kBindingView ? cp::kViewBytes : b == cp::kBindingParams ? m.params_bytes : 0);
    add(cp::kGroupInputs, 0, m.counts_bytes);
    for (std::uint32_t b = 1; b <= m.slots.size(); ++b) add(cp::kGroupInputs, b, 0);
    for (std::uint32_t b = 0; b < 4 + m.targets.size(); ++b) add(cp::kGroupResources, b, 0);
    if (p.kind == cs::PassKind::compute) {
        out->names.push_back(p.entry);
        gfx::EntryDesc e;
        e.stage = gfx::kStageCompute;
        out->entries.push_back(e);
    } else {
        out->names.push_back(p.vertex ? *p.vertex : "caliper_fullscreen");
        out->names.push_back(p.fragment);
        gfx::EntryDesc v, f;
        v.stage = gfx::kStageVertex;
        f.stage = gfx::kStageFragment;
        out->entries.push_back(v);
        out->entries.push_back(f);
    }
}

// The render pipeline of a pass for one variant of the host target (bit 0 depth, bit 1 ID).
gfx::Pipeline make_pipeline(DrawRuntime::Impl& m, std::size_t k, unsigned variant) {
    const cs::DrawPass& p = m.piece.draw->passes[k];
    DrawRuntime::Impl::PassSlot& s = m.passes[k];
    if (p.kind == cs::PassKind::compute) {
        gfx::ComputePipelineDesc d;
        d.program = s.program;
        d.entry = s.names[0].c_str();
        d.label = s.label.c_str();
        return m.gfx->create_compute_pipeline(d);
    }
    gfx::RenderPipelineDesc d;
    d.program = s.program;
    d.vertex_entry = s.names[0].c_str();
    d.fragment_entry = s.names[1].c_str();
    d.topology = topology_of(p.topology);
    d.color_format = gfx::Format::RGBA8Unorm;
    const cs::DrawBlend b = blend_of(p);
    d.blend = b == cs::DrawBlend::opaque ? gfx::Blend::Opaque : b == cs::DrawBlend::alpha ? gfx::Blend::Alpha
                                                                                          : gfx::Blend::Additive;
    if (!offscreen(p)) {
        d.id_target = p.pick && (variant & 2u) != 0;
        d.depth = (variant & 1u) != 0;
        const cs::DrawDepth depth = depth_of(p);
        d.depth_test = d.depth && depth != cs::DrawDepth::none;
        d.depth_write = d.depth && depth == cs::DrawDepth::write;
    }
    d.label = s.label.c_str();
    return m.gfx->create_render_pipeline(d);
}

// The pipelines of every pass for a host variant; false with the program stopped when the
// layer refuses one (an error of this program only).
bool ensure_pipelines(DrawRuntime::Impl& m, unsigned variant) {
    for (std::size_t k = 0; k < m.passes.size(); ++k) {
        const cs::DrawPass& p = m.piece.draw->passes[k];
        const unsigned v = p.kind == cs::PassKind::compute || offscreen(p) ? 0u : variant;
        if (m.passes[k].pipelines[v]) continue;
        m.passes[k].pipelines[v] = make_pipeline(m, k, v);
        if (!m.passes[k].pipelines[v]) {
            stop(m, cp::kProgramLayerRefused,
                 "Stopped " + m.instance.program + ": the graphics layer refused the pipeline of " + pass_name(m, int(k)) +
                     ": " + m.gfx->last_error(),
                 int(k));
            return false;
        }
    }
    return true;
}

// The bind groups of every pass, again after a buffer or a target changed.
bool rebuild_groups(DrawRuntime::Impl& m) {
    for (std::size_t k = 0; k < m.passes.size(); ++k) {
        DrawRuntime::Impl::PassSlot& s = m.passes[k];
        const cs::DrawPass& p = m.piece.draw->passes[k];
        for (std::uint32_t g = 0; g < 3; ++g) {
            if (s.groups[g]) {
                m.gfx->release(s.groups[g]);
                s.groups[g] = {};
            }
            gfx::BindGroupDesc d;
            d.program = s.program;
            d.group = g;
            d.label = s.label.c_str();
            for (const gfx::BindingDesc& b : s.bindings) {
                if (b.group != g) continue;
                gfx::BindingType type{};
                std::size_t i = 0;
                gfx::BindEntry& e = d.entries[d.entry_count++];
                e.binding = b.binding;
                switch (object_at(m, p, g, b.binding, &type, &i)) {
                    case Object::view: e.buffer = m.view; break;
                    case Object::selection: e.buffer = m.selection; break;
                    case Object::linear: e.sampler = m.linear; break;
                    case Object::nearest: e.sampler = m.nearest; break;
                    case Object::colormaps: e.texture = m.bound_colormaps; break;
                    case Object::params: e.buffer = m.params; break;
                    case Object::counts: e.buffer = m.counts; break;
                    case Object::slot:
                        if (m.slots[i].texture) e.texture = m.slots[i].tex;
                        else e.buffer = m.slots[i].buffer;
                        break;
                    case Object::buffer: e.buffer = m.buffers[i].buffer; break;
                    case Object::target: e.texture = m.targets[i].tex; break;
                    case Object::none: break;
                }
            }
            if (d.entry_count == 0) continue;
            s.groups[g] = m.gfx->create_bind_group(d);
            if (!s.groups[g]) {
                stop(m, cp::kProgramLayerRefused,
                     "Stopped " + m.instance.program + ": the graphics layer refused the bindings of " +
                         pass_name(m, int(k)) + ": " + m.gfx->last_error(),
                     int(k));
                return false;
            }
        }
    }
    m.groups_dirty = false;
    return true;
}

// The offscreen targets at the host target's size; false with the program stopped when they
// would pass draw.memory_mb or the layer refuses one.
bool ensure_targets(DrawRuntime::Impl& m, std::uint32_t width, std::uint32_t height) {
    for (std::size_t i = 0; i < m.targets.size(); ++i) {
        DrawRuntime::Impl::TargetSlot& t = m.targets[i];
        const std::uint32_t w = std::max<std::uint32_t>(1, std::uint32_t(std::floor(double(width) * t.scale)));
        const std::uint32_t h = std::max<std::uint32_t>(1, std::uint32_t(std::floor(double(height) * t.scale)));
        if (t.tex && t.width == w && t.height == h) continue;
        const std::uint64_t old = t.tex ? std::uint64_t(t.width) * t.height * 4 : 0;
        if (!account(m, old, std::uint64_t(w) * h * 4, "the target " + t.id + " of " + std::to_string(w) + " by " +
                                                           std::to_string(h) + " pixels",
                     "draw.targets[" + std::to_string(i) + "]"))
            return false;
        if (t.tex) m.gfx->release(t.tex);
        gfx::TextureDesc d;
        d.width = w;
        d.height = h;
        d.format = gfx::Format::RGBA8Unorm;
        d.usage = gfx::kRenderTarget | gfx::kSampled;
        d.label = t.id.c_str();
        t.tex = m.gfx->create_texture(d);
        t.width = w;
        t.height = h;
        if (!t.tex) {
            stop(m, cp::kProgramLayerRefused,
                 "Stopped " + m.instance.program + ": the graphics layer refused the target " + t.id + ": " +
                     m.gfx->last_error(),
                 -1, "draw.targets[" + std::to_string(i) + "]");
            return false;
        }
        m.groups_dirty = true;
    }
    return true;
}

gfx::Buffer make_buffer(gfx::Backend& g, std::uint64_t size, std::uint32_t usage, const char* label) {
    gfx::BufferDesc d;
    d.size = std::max<std::uint64_t>(16, (size + 15) / 16 * 16);
    d.usage = usage;
    d.label = label;
    return g.create_buffer(d);
}

}  // namespace

// ---- decode ------------------------------------------------------------------------------

Verdict DrawRuntime::decode(std::string_view declaration_json, std::string_view common_wgsl, std::string_view draw_wgsl,
                            caliper_builder::spec::Piece* piece, Instance* instance, const DrawChecks& checks,
                            std::string_view derive_wgsl) {
    cs::Piece p;
    Instance in;
    if (piece) *piece = cs::Piece{};
    if (instance) *instance = Instance{};
    const json d = json::parse(declaration_json.begin(), declaration_json.end(), nullptr, false);
    if (d.is_discarded() || !d.is_object())
        return refusal(cp::kPieceBadManifest, "The declaration of the program is no JSON object.");
    Reader r;
    if (!r.str(d, "id", &in.program) || (in.program.empty() && !r.str(d, "program", &in.program)) ||
        !r.str(d, "panel", &in.panel))
        return refusal(cp::kPieceBadManifest, "The declaration of the program: " + r.error + ".");
    if (d.contains("bind")) {
        if (!d["bind"].is_object()) return refusal(cp::kPieceBadBinding, "The key bind is not a table.");
        for (auto it = d["bind"].begin(); it != d["bind"].end(); ++it) {
            if (!it->is_string()) return refusal(cp::kPieceBadBinding, "The binding of " + it.key() + " is not a string.");
            in.bind[it.key()] = it->get<std::string>();
        }
    }
    std::string why;
    if (!d.contains("manifest") || !read_manifest(d["manifest"], &p, &why))
        return refusal(cp::kPieceBadManifest,
                       "The manifest of the piece: " + (why.empty() ? std::string("it is missing") : why) + ".");
    std::string pin_hash = p.hash;
    if (d.contains("piece") && d["piece"].is_object()) {
        const json& pin = d["piece"];
        std::string name;
        std::uint32_t version = p.version;
        if (!r.str(pin, "name", &name) || !r.u32(pin, "version", &version) || !r.str(pin, "hash", &pin_hash))
            return refusal(cp::kPieceBadManifest, "The pin of the piece: " + r.error + ".");
        if ((!name.empty() && name != p.name) || version != p.version)
            return refusal(cp::kPieceHashMismatch, "The pin names " + name + "@" + std::to_string(version) +
                                                       ", and the manifest " + p.name + "@" +
                                                       std::to_string(p.version) + ".");
    }

    // The author's files, from sections that may hold the imports resolved.
    auto files_of = [&](bool resolved) {
        std::map<std::string, std::string> f;
        std::string common = resolved ? strip_libs(common_wgsl, p.imports) : std::string(common_wgsl);
        std::string draw(draw_wgsl), derive(derive_wgsl);
        if (resolved) {
            common = unrewrite(common, p.imports);
            draw = unrewrite(draw, p.imports);
            derive = unrewrite(derive, p.imports);
        }
        if (!common.empty()) f["common.wgsl"] = common;
        if (!derive.empty()) f["derive.wgsl"] = derive;
        if (!draw.empty()) f["draw.wgsl"] = draw;
        return f;
    };
    p.hash.clear();
    p.files = files_of(false);
    if (!checks.hash) return refusal(cp::kProgramNotBuilt, "The piece hash is not available.");
    std::string computed = checks.hash(p);
    if (computed != pin_hash) {
        p.files = files_of(true);
        computed = checks.hash(p);
    }
    if (pin_hash.empty() || computed != pin_hash)
        return refusal(cp::kPieceHashMismatch,
                       "The hash of the piece " + p.name + "@" + std::to_string(p.version) + " is " +
                           (computed.empty() ? std::string("unknown") : computed) + ", and the pin says " +
                           (pin_hash.empty() ? std::string("none") : pin_hash) + ".");
    p.hash = pin_hash;
    if (piece) *piece = std::move(p);
    if (instance) *instance = std::move(in);
    return {};
}

// ---- install and remove ------------------------------------------------------------------

Verdict DrawRuntime::install(const caliper_builder::spec::Piece& piece, const Instance& instance,
                             std::uint32_t pick_id) {
    remove();
    Impl& m = *impl_;
    m.piece = piece;
    m.instance = instance;
    m.pick_id = pick_id;
    m.status = DrawStatus{};
    auto refuse = [&](Verdict v) {
        m.status.state = DrawState::refused;
        m.status.error = v;
        m.status.pass = -1;
        if (v.where.rfind("draw.passes[", 0) == 0) m.status.pass = std::atoi(v.where.c_str() + 12);
        release_objects(m);
        m.installed = false;
        return v;
    };
    const DrawChecks& c = m.checks;
    if (!c.manifest || !c.source || !c.resolve || !c.layout || !c.hash)
        return refuse(refusal(cp::kProgramNotBuilt, "The checks of pieces are not available."));

    // Check Order 2, 4 and 5: everything is checked before a program or a pipeline is made.
    if (Verdict v = c.manifest(piece); !v.ok()) return refuse(v);
    if (!piece.draw) return refuse(refusal(cp::kProgramBadPass, "The piece " + piece.name + " has no draw half.", "draw"));
    for (const auto& [name, text] : piece.files)
        if (Verdict v = c.source(text, pl::kFileBytes); !v.ok()) {
            if (v.where.empty()) v.where = name;
            return refuse(v);
        }
    if (!piece.hash.empty() && c.hash(piece) != piece.hash)
        return refuse(refusal(cp::kPieceHashMismatch, "The files of " + piece.name + " do not have the pinned hash."));
    const cp::Resolved resolved = c.resolve(piece);
    if (!resolved.verdict.ok()) return refuse(resolved.verdict);
    const cs::Draw& draw = *piece.draw;
    for (std::size_t k = 0; k < draw.passes.size(); ++k)
        if (draw.passes[k].pick && offscreen(draw.passes[k]))
            return refuse(refusal(cp::kProgramBadPass, "A pass with pick writes the host target only.",
                                  pass_where(k) + ".pick"));

    m.status.frame_limit_ms = std::min(pl::kFrameMsMax, draw.frame_ms.value_or(pl::kFrameMsDefault));
    m.status.memory_limit_bytes =
        std::uint64_t(std::min(pl::kDrawMemoryMbMax, draw.memory_mb.value_or(pl::kDrawMemoryMbDefault))) * kMiB;

    // The slots of group 1: the published resources, then the buffer inputs.
    std::set<std::string> cpu_ids;
    if (draw.pick) cpu_ids.insert(draw.pick->resource);
    if (draw.text) cpu_ids.insert(draw.text->anchors);
    if (draw.text && draw.text->strings_from) cpu_ids.insert(*draw.text->strings_from);
    for (const cs::Resource& res : piece.resources) {
        if (!res.publish) continue;
        Impl::Slot s;
        s.id = res.id;
        s.keep_cpu = cpu_ids.count(res.id) != 0;
        if (res.kind == cs::ResourceKind::texture) {
            // The manifest check bounds the sides by 8,192 and the texels by width * height, thus
            // declared_count is below 2^27 and declared_bytes at most 128 MiB.
            s.texture = true;
            s.width = res.width;
            s.height = res.height;
            s.format_name = cs::texture_format(res);
            s.format = s.format_name == "r8unorm"   ? gfx::Format::R8Unorm
                       : s.format_name == "r32uint" ? gfx::Format::R32Uint
                                                    : gfx::Format::RGBA8Unorm;
            s.declared_count = std::uint32_t(cs::texture_texels(res));
            s.declared_bytes = cs::texture_bytes(res);
            // The item map reads one scalar field per texel in row order.
            if (s.format != gfx::Format::R8Unorm) {
                s.layout.stride = 4;
                s.layout.align = 4;
                const std::string name = res.layout.fields.empty()
                                             ? (s.format == gfx::Format::R32Uint ? "value" : "rgba")
                                             : res.layout.fields[0].name;
                const std::string type = res.layout.fields.empty() ? "u32" : res.layout.fields[0].type;
                s.layout.fields.push_back(cp::FieldLayout{name, type, 0, 4});
            }
        } else {
            if (Verdict v = c.layout(res.layout, &s.layout); !v.ok()) return refuse(v);
            s.declared_count = res.count;
            s.declared_bytes = cp::layout_bytes(s.layout, res.count);
        }
        m.slots.push_back(std::move(s));
    }
    for (const cs::PieceInput& in : piece.inputs) {
        if (in.kind != cs::InputKind::buffer) continue;
        Impl::Slot s;
        s.id = in.name;
        s.keep_cpu = cpu_ids.count(in.name) != 0;
        s.unbound = in.optional && instance.bind.find(in.name) == instance.bind.end();
        if (Verdict v = c.layout(in.layout, &s.layout); !v.ok()) return refuse(v);
        // Another program's published buffer: at most kPublishedResourceBytesMax, whatever its stride.
        s.declared_bytes = std::min<std::uint64_t>(cp::layout_bytes(s.layout, pl::kCountMax),
                                                   pl::kPublishedResourceBytesMax);
        s.declared_count = s.layout.stride
                               ? std::uint32_t(std::min<std::uint64_t>(
                                     s.declared_bytes / s.layout.stride * s.layout.per_word, pl::kCountMax))
                               : 0;
        m.slots.push_back(std::move(s));
    }
    for (Impl::Slot& s : m.slots)
        for (const cs::DrawPass& p : draw.passes)
            if (p.indices && *p.indices == s.id) s.index = true;
    for (Impl::Slot& s : m.slots)
        if (!s.texture && (s.layout.stride == 0 || s.layout.stride % 4 != 0 || s.layout.stride > pl::kStrideMax))
            return refuse(refusal(cp::kProgramBadLayout, "The layout of " + s.id + " has no valid stride.", "resources"));
    for (const cs::DrawBuffer& b : draw.buffers) {
        Impl::DrawBufferSlot s;
        s.id = b.id;
        s.reset = b.reset;
        cp::ElementLayout layout;
        if (Verdict v = c.layout(b.layout, &layout); !v.ok()) return refuse(v);
        if (layout.stride == 0 || layout.stride % 4 != 0 || layout.stride > pl::kStrideMax)
            return refuse(refusal(cp::kProgramBadLayout, "The layout of " + b.id + " has no valid stride.", "draw.buffers"));
        s.stride = layout.stride;
        s.bytes = std::max<std::uint64_t>(layout.stride, std::uint64_t(b.count) * layout.stride);
        m.buffers.push_back(std::move(s));
    }
    for (const cs::DrawTarget& t : draw.targets) {
        Impl::TargetSlot s;
        s.id = t.id;
        s.scale = t.scale;
        s.clear_rgba = pack_rgba(t.clear);
        m.targets.push_back(std::move(s));
    }
    m.params_bytes = std::uint32_t(piece.params.size() * 4);
    m.counts_bytes = std::uint32_t(m.slots.size() * 4);
    m.params_default.assign(m.params_bytes, 0);
    for (std::size_t i = 0; i < piece.params.size(); ++i) {
        const cs::PieceParam& pa = piece.params[i];
        std::uint8_t* at = m.params_default.data() + 4 * i;
        if (pa.type == cs::ParamType::f32) put_f32(at, float(pa.default_value));
        else if (pa.type == cs::ParamType::i32) put_u32(at, std::uint32_t(std::int32_t(pa.default_value)));
        else put_u32(at, std::uint32_t(std::max(0.0, pa.default_value)));
    }

    // Check Order 6 and 7: compose and compile each pass, then make its program.
    m.passes.resize(draw.passes.size());
    for (std::size_t k = 0; k < draw.passes.size(); ++k) {
        Impl::PassSlot& s = m.passes[k];
        s.label = instance.program + " " + pass_where(k);
        const cp::Composed composed = cp::compose_draw(piece, resolved, k);
        if (!composed.verdict.ok()) return refuse(composed.verdict);
        s.wgsl = composed.text;
        if (m.webgpu) {
            plan_bindings(m, k, &s);
        } else if (Verdict v = compile_pass(m, k, composed, &s); !v.ok()) {
            return refuse(v);
        }
        for (std::size_t e = 0; e < s.entries.size(); ++e) s.entries[e].name = s.names[e].c_str();
        gfx::ProgramDesc d;
        d.label = s.label.c_str();
        d.entries = s.entries.data();
        d.entry_count = std::uint8_t(s.entries.size());
        d.bindings = s.bindings.data();
        d.binding_count = std::uint8_t(s.bindings.size());
        d.wgsl = s.wgsl.c_str();
        d.msl = s.msl.empty() ? nullptr : s.msl.c_str();
        s.program = m.gfx->create_program(d);
        if (!s.program)
            return refuse(refusal(m.webgpu ? cp::kProgramBrowserRefused : cp::kProgramLayerRefused,
                                  "The graphics layer refused the program of draw pass " + std::to_string(k) + ": " +
                                      m.gfx->last_error(),
                                  pass_where(k)));
    }

    // The objects of the prelude and the uniforms.
    gfx::Backend& g = *m.gfx;
    m.view = make_buffer(g, cp::kViewBytes, gfx::kUniform | gfx::kCopyDst, "caliper_view");
    m.selection = make_buffer(g, std::uint64_t(cp::kMaxSelected) * cp::kSelectedBytes, gfx::kStorage | gfx::kCopyDst,
                              "caliper_selection");
    if (m.params_bytes) m.params = make_buffer(g, m.params_bytes, gfx::kUniform | gfx::kCopyDst, "params");
    if (m.counts_bytes) m.counts = make_buffer(g, m.counts_bytes, gfx::kUniform | gfx::kCopyDst, "counts");
    gfx::SamplerDesc lin, near;
    lin.linear = true;
    near.linear = false;
    m.linear = g.create_sampler(lin);
    m.nearest = g.create_sampler(near);
    gfx::TextureDesc cm;
    cm.width = 256;
    cm.height = kFallbackColormaps;
    cm.format = gfx::Format::RGBA8Unorm;
    cm.usage = gfx::kSampled | gfx::kCopyDstTex;
    cm.label = "caliper_colormaps";
    m.colormaps = g.create_texture(cm);
    if (!m.view || !m.selection || (m.params_bytes && !m.params) || (m.counts_bytes && !m.counts) || !m.linear ||
        !m.nearest || !m.colormaps)
        return refuse(refusal(cp::kProgramLayerRefused, "The graphics layer refused the prelude's objects: " +
                                                            g.last_error()));
    // Without the host's colormaps (HostTarget::colormaps) each row is a grey ramp.
    std::vector<std::uint32_t> rows(std::size_t(256) * kFallbackColormaps);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const std::uint32_t v = std::uint32_t(i % 256);
        rows[i] = 0xff000000u | (v << 16) | (v << 8) | v;
    }
    g.write_texture(m.colormaps, rows.data(), 256 * 4);
    m.bound_colormaps = m.colormaps;
    if (!zero_buffer(g, m.selection, std::uint64_t(cp::kMaxSelected) * cp::kSelectedBytes))
        return refuse(refusal(cp::kProgramLayerRefused, "The graphics layer refused a write: " + g.last_error()));

    // The resources' first buffers, the draw buffers zeroed, within draw.memory_mb.
    for (std::size_t i = 0; i < m.slots.size(); ++i) {
        Impl::Slot& s = m.slots[i];
        const std::string where = "resources[" + s.id + "]";
        if (s.texture) {
            // The budget holds the largest version; the texture starts at 1 by 1 and takes the size of
            // each version within it.
            if (!account(m, 0, s.declared_bytes, "the texture " + s.id, where)) return m.status.error;
            s.tex = make_texture(g, s, 1, 1);
            if (!s.tex)
                return refuse(refusal(cp::kProgramLayerRefused, "The graphics layer refused the texture " + s.id + ": " +
                                                                    g.last_error(), where));
            s.tex_width = s.tex_height = 1;
            continue;
        }
        s.capacity = std::max<std::uint64_t>(16, (s.layout.stride + 15) / 16 * 16);
        if (!account(m, 0, s.capacity, "the resource " + s.id, where)) return m.status.error;
        s.buffer = make_buffer(g, s.capacity, gfx::kStorage | gfx::kCopyDst | (s.index ? gfx::kIndex : 0u), s.id.c_str());
        if (!s.buffer || !zero_buffer(g, s.buffer, s.capacity))
            return refuse(refusal(cp::kProgramLayerRefused, "The graphics layer refused the buffer of " + s.id + ": " +
                                                                g.last_error(), where));
        // An unbound optional input is an empty version for good: its count is 0 in `counts`.
        if (s.unbound) {
            s.has_version = true;
            s.count = 0;
            s.serial = ++m.versions;
        }
    }
    for (std::size_t i = 0; i < m.buffers.size(); ++i) {
        Impl::DrawBufferSlot& b = m.buffers[i];
        const std::string where = "draw.buffers[" + std::to_string(i) + "]";
        if (!account(m, 0, b.bytes, "the draw buffer " + b.id, where)) return m.status.error;
        b.buffer = make_buffer(g, b.bytes, gfx::kStorage | gfx::kCopyDst, b.id.c_str());
        if (!b.buffer || !zero_buffer(g, b.buffer, std::max<std::uint64_t>(16, (b.bytes + 15) / 16 * 16)))
            return refuse(refusal(cp::kProgramLayerRefused, "The graphics layer refused the draw buffer " + b.id + ": " +
                                                                g.last_error(), where));
    }

    // The pipelines for a host target with depth and ID; another variant is made when a host
    // target first needs it. A pipeline that the layer refuses refuses the program.
    m.installed = true;
    m.status.state = DrawState::compiled;
    if (!ensure_pipelines(m, 3u)) {
        Verdict v = m.status.error;
        return refuse(v);
    }
    m.groups_dirty = true;
    m.first_after_reset = true;
    m.frames_since_reset = 0;
    return {};
}

void DrawRuntime::remove() {
    Impl& m = *impl_;
    if (m.gfx) release_objects(m);
    m.installed = false;
    m.piece = cs::Piece{};
    m.instance = Instance{};
    m.status = DrawStatus{};
    m.window.clear();
    m.over = 0;
    m.measure = false;
    m.panel_warned = false;
    m.measured.clear();
    m.epoch_known = false;
    m.late = false;
    m.late_frames = 0;
    m.heavy_frames = 0;
    m.heavy_ms = 0;
}

// ---- resources ---------------------------------------------------------------------------

Verdict DrawRuntime::set_resource(const ResourceVersion& version) {
    Impl& m = *impl_;
    if (!m.installed || m.status.state == DrawState::stopped || m.status.state == DrawState::refused) return {};
    Impl::Slot* s = nullptr;
    std::size_t index = 0;
    for (std::size_t i = 0; i < m.slots.size(); ++i)
        if (m.slots[i].id == version.resource) {
            s = &m.slots[i];
            index = i;
        }
    if (s == nullptr)
        return refusal(cp::kProgramUnknownResource,
                       "The draw half of " + m.instance.program + " reads no resource " + version.resource + ".");
    if (s->unbound)
        return refusal(cp::kProgramUnknownResource,
                       "The optional input " + s->id + " of " + m.instance.program + " is unbound and takes no version.");
    const std::string where = "resources[" + s->id + "]";
    if (version.size > 0 && version.bytes == nullptr)
        return refusal(cp::kProgramBadResource, "A version of " + s->id + " has a size but no bytes.", where);
    std::uint32_t valid = version.count;   // the count that the frame's `counts` gives the draw half
    if (s->texture) {
        // The version's format and size against the manifest, each before any product: the sides are at
        // most 8,192, thus the texels and the bytes fit 64 bits with room.
        // A version without texels carries no size and thus no format, as RESOURCE gives it.
        const bool empty = version.width == 0 && version.height == 0;
        const std::string format = version.format.empty() ? (empty ? s->format_name : std::string("rgba8unorm"))
                                                          : version.format;
        if (format != s->format_name)
            return refusal(cp::kProgramBadResource,
                           "A version of the texture " + s->id + " has the format " + format + ", and the texture " +
                               s->format_name + ".",
                           where);
        if (version.elements)
            return refusal(cp::kProgramBadResource, "A version of the texture " + s->id + " has narrow elements.",
                           where);
        if (!empty && (version.width < 1 || version.width > s->width || version.height < 1 ||
                       version.height > s->height))
            return refusal(cp::kProgramResourceTooLarge,
                           "A version of the texture " + s->id + " is " + std::to_string(version.width) + " by " +
                               std::to_string(version.height) + " texels, outside its largest " +
                               std::to_string(s->width) + " by " + std::to_string(s->height) + ".",
                           where);
        const std::uint64_t texels = std::uint64_t(version.width) * version.height;
        const std::uint64_t words = cs::texture_words(s->format_name, texels);
        if (texels > s->declared_count || version.count != words || version.size != words * 4)
            return refusal(texels > s->declared_count ? cp::kProgramResourceTooLarge : cp::kProgramBadResource,
                           "A version of the texture " + s->id + " of " + std::to_string(version.width) + " by " +
                               std::to_string(version.height) + " texels has " + std::to_string(version.size) +
                               " bytes; it holds at most " + std::to_string(s->declared_count) + " texels in " +
                               std::to_string(words * 4) + " bytes.",
                           where);
        if (!empty) {
            if (version.width != s->tex_width || version.height != s->tex_height) {
                gfx::Texture t = make_texture(*m.gfx, *s, version.width, version.height);
                if (!t)
                    return refusal(cp::kProgramLayerRefused, "The graphics layer refused the texture " + s->id +
                                                                 " of a version: " + m.gfx->last_error(), where);
                m.gfx->release(s->tex);
                s->tex = t;
                s->tex_width = version.width;
                s->tex_height = version.height;
                m.groups_dirty = true;
            }
            const std::uint32_t row = version.width * (s->format == gfx::Format::R8Unorm ? 1u : 4u);
            if (!m.gfx->write_texture(s->tex, version.bytes, row))
                return refusal(cp::kProgramLayerRefused, "The graphics layer refused a version of " + s->id + ": " +
                                                             m.gfx->last_error(), where);
        }
        valid = std::uint32_t(texels);
    } else {
        if (version.width || version.height || !version.format.empty())
            return refusal(cp::kProgramBadResource, "A version of the buffer " + s->id + " has a texture's size.",
                           where);
        if ((s->layout.per_word > 1) != version.elements.has_value())
            return refusal(cp::kProgramBadResource,
                           "A version of " + s->id + (s->layout.per_word > 1 ? " has no" : " has a") +
                               " count of narrow elements, and its layout " +
                               (s->layout.per_word > 1 ? "packs them." : "has none."),
                           where);
        if (version.elements) {
            // The words of the elements, in 64 bits: at most 2^32 - 1 elements.
            const std::uint64_t words = (std::uint64_t(*version.elements) + s->layout.per_word - 1) / s->layout.per_word;
            if (*version.elements > s->declared_count || words != version.count)
                return refusal(*version.elements > s->declared_count ? cp::kProgramResourceTooLarge
                                                                     : cp::kProgramBadResource,
                               "A version of " + s->id + " has " + std::to_string(*version.elements) +
                                   " narrow elements in " + std::to_string(version.count) + " words, for at most " +
                                   std::to_string(s->declared_count) + " elements.",
                               where);
            valid = *version.elements;
        }
        const std::uint64_t expect = std::uint64_t(version.count) * s->layout.stride;   // < 2^32 * 256, no wrap
        if ((!version.elements && version.count > s->declared_count) || version.size > s->declared_bytes)
            return refusal(cp::kProgramResourceTooLarge,
                           "A version of " + s->id + " has " + std::to_string(version.count) + " elements and " +
                               std::to_string(version.size) + " bytes, above its " +
                               std::to_string(s->declared_count) + " elements.",
                           where);
        if (version.size != expect)
            return refusal(cp::kProgramBadResource,
                           "A version of " + s->id + " has " + std::to_string(version.size) + " bytes for " +
                               std::to_string(version.count) + " elements of " + std::to_string(s->layout.stride) +
                               " bytes.",
                           where);
        if (version.size > s->capacity) {
            std::uint64_t cap = s->capacity;
            while (cap < version.size) cap *= 2;
            cap = std::min(cap, std::max<std::uint64_t>(version.size, (s->declared_bytes + 15) / 16 * 16));
            if (!account(m, s->capacity, cap, "a version of " + s->id + " of " + std::to_string(version.size) + " bytes",
                         where))
                return m.status.error;
            gfx::Buffer b =
                make_buffer(*m.gfx, cap, gfx::kStorage | gfx::kCopyDst | (s->index ? gfx::kIndex : 0u), s->id.c_str());
            if (!b) {
                stop(m, cp::kProgramLayerRefused,
                     "Stopped " + m.instance.program + ": the graphics layer refused a buffer of " + s->id + ": " +
                         m.gfx->last_error(),
                     -1, where);
                return m.status.error;
            }
            m.gfx->release(s->buffer);
            s->buffer = b;
            s->capacity = cap;
            m.groups_dirty = true;
        }
        if (version.size && !m.gfx->write_buffer(s->buffer, 0, version.bytes, version.size))
            return refusal(cp::kProgramLayerRefused, "The graphics layer refused a version of " + s->id + ": " +
                                                         m.gfx->last_error(), where);
    }
    (void)index;
    if (s->keep_cpu) s->cpu.assign(version.bytes, version.bytes + version.size);
    s->has_version = true;
    s->step = version.step;
    s->count = valid;
    s->serial = ++m.versions;   // a version at the same step is another version
    return {};
}

// ---- one frame ---------------------------------------------------------------------------

bool DrawRuntime::encode(std::vector<caliper_gfx::Frame>* frames, const FrameInputs& in, const HostTarget& target) {
    Impl& m = *impl_;
    if (!m.installed || frames == nullptr) return false;
    if (m.status.state != DrawState::compiled && m.status.state != DrawState::running) return false;
    if (m.late) return false;   // nothing piles up behind a frame that is not done
    if (!target.color || target.width == 0 || target.height == 0) return false;
    for (const Impl::Slot& s : m.slots)
        if (!s.has_version) return false;   // waiting for the run's resources
    gfx::Backend& g = *m.gfx;
    const cs::Draw& draw = *m.piece.draw;

    if (!ensure_targets(m, target.width, target.height)) return false;
    const unsigned variant = (target.depth ? 1u : 0u) | (target.id ? 2u : 0u);
    if (!ensure_pipelines(m, variant)) return false;

    // The reset rules (Draw State).
    bool all = false;
    if (m.epoch_known && in.epoch != m.epoch) all = true;
    m.epoch = in.epoch;
    m.epoch_known = true;
    bool changed = false;
    for (const Impl::Slot& s : m.slots)
        if (s.drawn_serial != 0 && s.serial != s.drawn_serial) changed = true;
    bool reset = false;
    for (Impl::DrawBufferSlot& b : m.buffers) {
        const bool zero = all || (b.reset == cs::BufferReset::cursor && in.cursor_moved) ||
                          (b.reset == cs::BufferReset::resource && changed);
        if (!zero) continue;
        zero_buffer(g, b.buffer, std::max<std::uint64_t>(16, (b.bytes + 15) / 16 * 16));
        reset = true;
    }
    if (reset) {
        m.frames_since_reset = 0;
        m.first_after_reset = true;
    }
    const gfx::Texture colormaps = target.colormaps ? target.colormaps : m.colormaps;
    if (colormaps != m.bound_colormaps) {
        m.bound_colormaps = colormaps;
        m.groups_dirty = true;
    }
    if (m.groups_dirty && !rebuild_groups(m)) return false;

    // CaliperView, the selection, params and the counts.
    std::array<std::uint8_t, cp::kViewBytes> view{};
    const bool camera = draw.camera != cs::CameraMode::none;
    std::memcpy(view.data() + 0, (camera ? in.clip_from_world : kIdentity).data(), 64);
    std::memcpy(view.data() + 64, (camera ? in.world_from_clip : kIdentity).data(), 64);
    std::memcpy(view.data() + 128, (camera ? in.view_from_world : kIdentity).data(), 64);
    std::memcpy(view.data() + 192, in.eye.data(), 16);
    std::array<float, 4> viewport = in.viewport;
    if (viewport[2] <= 0 || viewport[3] <= 0) viewport = {0, 0, float(target.width), float(target.height)};
    std::memcpy(view.data() + 208, viewport.data(), 16);
    std::memcpy(view.data() + 224, in.pointer.data(), 16);
    put_f32(view.data() + 240, float(in.time_s));
    put_f32(view.data() + 244, float(in.dt_s));
    put_f32(view.data() + 248, in.pixel_ratio > 0 ? in.pixel_ratio : 1.0f);
    std::int64_t resource_step = -1;
    for (const Impl::Slot& s : m.slots) resource_step = std::max(resource_step, s.step);
    put_u32(view.data() + 256, step_u32(in.cursor_step));
    put_u32(view.data() + 260, step_u32(in.latest_step));
    put_u32(view.data() + 264, step_u32(resource_step));
    put_u32(view.data() + 268, in.live ? 1u : 0u);
    const std::size_t selected = std::min<std::size_t>(in.selection.size(), cp::kMaxSelected);
    put_u32(view.data() + 272, std::uint32_t(std::min<std::uint64_t>(m.frames_since_reset, 0xffffffffull)));
    put_u32(view.data() + 276, m.first_after_reset ? 1u : 0u);
    put_u32(view.data() + 280, m.pick_id);
    put_u32(view.data() + 284, std::uint32_t(selected));
    g.write_buffer(m.view, 0, view.data(), view.size());
    if (selected) {
        std::vector<std::uint8_t> sel(selected * cp::kSelectedBytes, 0);
        for (std::size_t i = 0; i < selected; ++i) {
            std::uint8_t* at = sel.data() + i * cp::kSelectedBytes;
            put_u32(at, in.selection[i].pick);
            put_u32(at + 4, in.selection[i].item);
            put_u32(at + 8, in.selection[i].flags);
            for (int j = 0; j < 4; ++j) put_u32(at + 16 + 4 * j, std::uint32_t(in.selection[i].index[std::size_t(j)]));
        }
        g.write_buffer(m.selection, 0, sel.data(), sel.size());
    }
    std::vector<std::uint8_t> params = m.params_default;
    if (m.params_bytes) {
        if (!in.params.empty())
            std::memcpy(params.data(), in.params.data(), std::min<std::size_t>(params.size(), in.params.size()));
        if (!g.write_buffer(m.params, 0, params.data(), params.size())) {
            stop(m, cp::kProgramLayerRefused,
                 "The graphics layer refused the draw parameters: " + g.last_error(), -1, "params");
            return false;
        }
    }
    if (m.counts_bytes) {
        std::vector<std::uint8_t> counts(m.counts_bytes, 0);
        for (std::size_t i = 0; i < m.slots.size(); ++i) put_u32(counts.data() + 4 * i, m.slots[i].count);
        g.write_buffer(m.counts, 0, counts.data(), counts.size());
    }

    // The passes in their order: one frame, or one frame for each pass while measuring.
    const bool measuring = m.measure;
    const std::size_t first_frame = frames->size();
    std::vector<bool> cleared(m.targets.size(), false);
    std::vector<int> frame_pass;   // the pass of each appended frame while measuring
    auto frame_for = [&](int pass) -> gfx::Frame& {
        if (!measuring) {
            if (frames->size() == first_frame) frames->emplace_back();
            return frames->back();
        }
        frames->emplace_back();
        frame_pass.push_back(pass);
        return frames->back();
    };
    auto target_index = [&](const std::string& id) {
        for (std::size_t i = 0; i < m.targets.size(); ++i)
            if (m.targets[i].id == id) return i;
        return m.targets.size();
    };
    for (std::size_t k = 0; k < draw.passes.size(); ++k) {
        const cs::DrawPass& p = draw.passes[k];
        Impl::PassSlot& s = m.passes[k];
        if (p.kind == cs::PassKind::compute) {
            gfx::Frame& f = frame_for(int(k));
            f.dispatch(s.pipelines[0], gfx::View<gfx::BindGroup>(s.groups.data(), s.groups.size()), {},
                       p.workgroups[0], p.workgroups[1], p.workgroups[2]);
            continue;
        }
        std::uint32_t instances = p.instances;
        if (p.instances_of) {
            instances = 0;
            for (const Impl::Slot& sl : m.slots)
                if (sl.id == *p.instances_of) instances = sl.has_version ? sl.count : 0;
        }
        // An indexed pass draws the valid indices of its version: the buffer holds at least count
        // elements of 4 bytes (set_resource), and each index is only the vertex entry's
        // vertex_index, whose every read of a buffer is bounds-checked; no index is read on the CPU.
        const Impl::Slot* index_slot = nullptr;
        if (p.indices) {
            for (const Impl::Slot& sl : m.slots)
                if (sl.id == *p.indices) index_slot = &sl;
            if (!index_slot || !index_slot->has_version || index_slot->count == 0 || !index_slot->buffer) instances = 0;
        }
        const bool host = !offscreen(p);
        const std::size_t ti = host ? m.targets.size() : target_index(p.target);
        if (!host && ti == m.targets.size()) continue;   // refused by the manifest check
        gfx::RenderTarget rt;
        if (host) {
            rt.color = target.color;
            rt.id = p.pick ? target.id : gfx::Texture{};
            rt.depth = target.depth;
            rt.clear = false;
        } else {
            rt.color = m.targets[ti].tex;
            rt.clear = !cleared[ti];
            rt.clear_rgba = m.targets[ti].clear_rgba;
            cleared[ti] = true;
        }
        gfx::DrawList list(rt);
        if (instances > 0) {
            gfx::DrawRecord r;
            r.pipeline = s.pipelines[host ? variant : 0u];
            for (std::size_t gi = 0; gi < 3; ++gi) r.groups[gi] = s.groups[gi];
            r.count = p.vertex ? p.vertices.value_or(3) : 3;
            if (index_slot) {
                r.index_buffer = index_slot->buffer;
                r.count = index_slot->count;
            }
            r.instance_count = instances;
            r.pick_id = p.pick ? m.pick_id : 0;
            if (host)
                for (int i = 0; i < 4; ++i) r.scissor[i] = in.scissor[std::size_t(i)];
            list.draw(r);
        } else if (host || !rt.clear) {
            continue;   // a pass whose instances have no version or no valid element is skipped
        }
        frame_for(int(k)).render(std::move(list));
    }
    if (frames->size() == first_frame) return false;
    m.params_shown = std::move(params);

    // What the frame shows, for the item map, the text table and the reset rules: a new copy for
    // each new version, also one at the same step, and the copy a pick of an earlier frame holds
    // stays as it was.
    for (Impl::Slot& s : m.slots) {
        if (s.drawn_serial == s.serial && s.shown) continue;
        s.drawn_serial = s.serial;
        auto shown = std::make_shared<Impl::Shown>();
        shown->step = s.step;
        shown->count = s.count;
        if (s.keep_cpu) shown->bytes = std::move(s.cpu);
        s.cpu.clear();
        s.shown = std::move(shown);
    }
    m.status.resource_step = resource_step;
    ++m.frame_no;
    if (measuring) {
        m.measure = false;
        m.measured.assign(draw.passes.size(), -1.0);
        for (std::size_t i = 0; i < frame_pass.size(); ++i)
            m.expected.push_back({m.frame_no, frame_pass[i], i + 1 == frame_pass.size()});
    } else {
        m.expected.push_back({m.frame_no, -1, true});
    }
    ++m.frames_since_reset;
    m.first_after_reset = false;
    ++m.status.frames;
    m.status.state = DrawState::running;
    return true;
}

// ---- timing and removal ------------------------------------------------------------------

void DrawRuntime::frame_done(double gpu_ms, double wall_ms) {
    Impl& m = *impl_;
    if (m.expected.empty()) return;
    const Impl::Expected e = m.expected.front();
    m.expected.pop_front();
    const bool late = m.late;   // the oldest frame, which this is, was late
    m.late = false;
    if (m.status.state != DrawState::running) return;
    // A time that the backend no longer keeps (the panel was not drawn while other views of the
    // backend completed many submits): the frame is done, its time is unknown, and it is not judged.
    if (gpu_ms == gfx::kGpuMsForgotten) return;
    const std::string what = e.pass >= 0 ? pass_name(m, e.pass) : std::string("a frame of its draw passes");
    // Heavy frames in a row: their GPU time together over 1 s stops the program, late or not.
    if (gpu_ms >= 0) {
        if (gpu_ms > kHeavyMs) {
            ++m.heavy_frames;
            m.heavy_ms += gpu_ms;
        } else {
            m.heavy_frames = 0;
            m.heavy_ms = 0;
        }
        if (m.heavy_ms > kTimeoutMs) {
            stop(m, cp::kProgramDrawTimeout,
                 "Stopped " + m.instance.program + ": " +
                     (m.heavy_frames == 1 ? what + " took " + fmt_ms(gpu_ms) + " ms of GPU time, above 1 s"
                                          : std::to_string(m.heavy_frames) + " frames of its draw passes in a row took " +
                                                fmt_ms(m.heavy_ms) + " ms of GPU time together, above 1 s"),
                 e.pass);
            return;
        }
    }
    // A late frame on a backend with timestamps that gives no time for it: the driver ended it.
    // Without timestamps, late frames in a row stop it, and a frame done in time ends the row.
    const bool timed = m.gfx->caps().has(gfx::kCapTimestamps);
    if (late && gpu_ms < 0 && timed) {
        stop(m, cp::kProgramDrawTimeout,
             "Stopped " + m.instance.program + ": " + what + " did not finish, and the driver ended it", e.pass);
        return;
    }
    if (late && gpu_ms < 0) {
        if (++m.late_frames >= kLateFrames) {
            stop(m, cp::kProgramDrawTimeout,
                 "Stopped " + m.instance.program + ": " + std::to_string(kLateFrames) +
                     " frames of its draw passes in a row did not finish in 1 s",
                 e.pass);
            return;
        }
    } else if (late || (gpu_ms >= 0 ? gpu_ms : wall_ms) <= kTimeoutMs) {
        m.late_frames = 0;
    }
    // Without timestamps the time from the submit until the host saw the work done also holds the
    // host's polling, once a frame in a browser (about 16.7 ms at 60 Hz), thus it cannot judge a
    // draw.frame_ms of a few milliseconds: every light frame would count as over. The frame time
    // stays unknown, so the panel's sum skips it too, and only frames over kHeavyMs of that time,
    // kWindowFrames in a row, record a warning on the program while it keeps drawing.
    const std::string prefix = m.instance.program.empty() ? std::string() : m.instance.program + ": ";
    if (gpu_ms < 0 && !timed) {
        if (e.pass >= 0 || wall_ms < 0) return;
        if (wall_ms <= kHeavyMs) {
            m.over = 0;
            if (!m.panel_warned) clear_warning(m);
            return;
        }
        ++m.over;
        if (m.over >= kWindowFrames && m.status.warning.ok())
            warn(m, cp::kProgramDrawOverTime,
                 prefix + std::to_string(kWindowFrames) + " frames of its draw passes in a row took over " +
                     fmt_ms(kHeavyMs) + " ms from the submit until done, and this backend gives no GPU time",
                 -1);
        return;
    }
    if (gpu_ms < 0) gpu_ms = wall_ms;   // the upper bound, where the backend gives no GPU time
    const double limit = m.status.frame_limit_ms;
    auto record_window = [&](double ms) {
        m.window.push_back(ms);
        while (m.window.size() > kWindowFrames) m.window.pop_front();
        std::vector<double> sorted(m.window.begin(), m.window.end());
        std::sort(sorted.begin(), sorted.end());
        m.status.frame_ms = sorted[sorted.size() / 2];
    };
    if (e.pass < 0) {
        if (gpu_ms < 0) return;
        record_window(gpu_ms);
        if (gpu_ms <= limit) {
            m.over = 0;
            if (!m.panel_warned) clear_warning(m);
            return;
        }
        ++m.over;
        if (m.over >= kWindowFrames && !m.measure && m.status.warning.ok()) {
            m.measure = true;
            m.measure_code = cp::kProgramDrawOverTime;
            m.measure_message.clear();
        }
        return;
    }
    if (std::size_t(e.pass) < m.measured.size()) m.measured[std::size_t(e.pass)] = gpu_ms;
    if (!e.last) return;
    int worst = -1;
    double total = 0;
    bool any_measured = false;
    for (std::size_t k = 0; k < m.measured.size(); ++k) {
        if (m.measured[k] < 0) continue;
        any_measured = true;
        total += m.measured[k];
        if (worst < 0 || m.measured[k] > m.measured[std::size_t(worst)]) worst = int(k);
    }
    if (any_measured) record_window(total);
    const std::string code = m.measure_code;
    if (code == cp::kProgramDrawOverTime && m.measure_message.empty() && total <= limit) {
        m.over = 0;   // the time came back under the limit while it measured
        if (!m.panel_warned) clear_warning(m);
        return;
    }
    if (worst < 0) worst = 0;
    const double pass_ms = m.measured[std::size_t(worst)] < 0 ? 0 : m.measured[std::size_t(worst)];
    std::string message = prefix + pass_name(m, worst) + " took " + fmt_ms(pass_ms) + " ms a frame";
    if (m.measure_message.empty()) message += ", above its " + fmt_limit(limit) + " ms";
    else message += "; " + m.measure_message;
    m.panel_warned = !m.measure_message.empty();
    warn(m, code.c_str(), message, worst);
}

void DrawRuntime::frame_pending(double waited_ms) {
    Impl& m = *impl_;
    if (m.status.state != DrawState::running || waited_ms <= kTimeoutMs) return;
    m.late = true;
    if (waited_ms <= kHangMs) return;
    const int pass = m.expected.empty() ? -1 : m.expected.front().pass;
    stop(m, cp::kProgramDrawTimeout,
         "Stopped " + m.instance.program + ": " +
             (pass >= 0 ? pass_name(m, pass) : std::string("a frame of its draw passes")) +
             " did not finish in 5 s; the GPU runs it until the driver ends it",
         pass);
}

void DrawRuntime::frame_refused(const std::string& why) {
    Impl& m = *impl_;
    if (m.status.state != DrawState::running && m.status.state != DrawState::compiled) return;
    stop(m, cp::kProgramLayerRefused, "Stopped " + m.instance.program + ": the graphics layer refused its frame: " + why,
         -1);
}

void DrawRuntime::request_warning(const char* code, const std::string& message) {
    Impl& m = *impl_;
    if (m.status.state != DrawState::running && m.status.state != DrawState::compiled) return;
    m.measure = true;
    m.measure_code = code;
    m.measure_message = message.empty() ? std::string("the programs of its panel exceeded 16.7 ms") : message;
}

void DrawRuntime::clear_panel_warning() {
    Impl& m = *impl_;
    if (!m.panel_warned) return;
    if (m.over == 0) clear_warning(m);
    else m.panel_warned = false;
}

std::int64_t DrawRuntime::resource_step(const std::string& resource) const {
    for (const Impl::Slot& s : impl_->slots)
        if (s.id == resource) return s.has_version ? s.step : -1;
    return -1;
}

caliper_builder::spec::CameraMode DrawRuntime::camera() const {
    return impl_->piece.draw ? impl_->piece.draw->camera : cs::CameraMode::none;
}

const Instance& DrawRuntime::instance() const { return impl_->instance; }

std::uint32_t DrawRuntime::pick_id() const { return impl_->pick_id; }

DrawStatus DrawRuntime::status() const { return impl_->status; }

}  // namespace caliper_draw
