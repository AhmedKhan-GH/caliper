// live_scene.cpp: the scene view of the shared view layer (live_scene.h).
#include "live/live_scene.h"
#include "live/scene_piece_form.h"

#include "gfx/colormap_keys.h"
#include "gfx/colormap_tables.h"
#if defined(CALIPER_HAVE_DRAW)
#include "draw/draw_host.h"
#include "draw/draw_runtime.h"
#endif
// The shapes drawn by library pieces need the draw layer and the library's manifests, which the app
// with the builder and the browser module have (scene_pieces.h).
#if defined(CALIPER_HAVE_DRAW) && defined(CALIPER_HAVE_SCENE_PIECES)
#define CALIPER_SCENE_PIECES 1
#include "draw/scene_pieces.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace caliper_live_views {

using namespace caliper_scene;
using caliper_gfx::Buffer;
using caliper_gfx::VertexFormat;
namespace wire = caliper_net::wire;

namespace {

bool class_color(const caliper_net::Json& params, uint32_t* out) {
    const caliper_net::Json* c = params.find("color");
    if (!c || !c->is_array() || c->size() != 4) return false;
    uint32_t rgba = 0;
    for (size_t k = 0; k < 4; ++k) {
        const double v = c->at(k).as_double();
        rgba |= uint32_t(std::clamp(v, 0.0, 255.0)) << (8 * k);
    }
    *out = rgba;
    return true;
}

// A VIEWS entry's view_camera, as wire.cpp normalised it: null for the fit, a camera entity's
// path, or an orbit object with any of target, distance, azimuth and elevation (degrees).
bool starting_from_wire(const caliper_net::Json& v, StartingCamera* out) {
    StartingCamera s;
    if (v.is_string()) {
        s.form = StartingCamera::Form::Entity;
        s.entity = v.as_string();
    } else if (v.is_object() && v.size() > 0) {
        s.form = StartingCamera::Form::Orbit;
        if (const caliper_net::Json* t = v.find("target"); t && t->is_array() && t->size() == 3)
            s.target = Vec3{t->at(0).as_double(), t->at(1).as_double(), t->at(2).as_double()};
        if (const caliper_net::Json* d = v.find("distance"); d && d->is_number()) s.distance = d->as_double();
        if (const caliper_net::Json* a = v.find("azimuth"); a && a->is_number()) s.azimuth_deg = a->as_double();
        if (const caliper_net::Json* e = v.find("elevation"); e && e->is_number()) s.elevation_deg = e->as_double();
    } else if (!v.is_null() && !v.is_object()) {
        return false;
    }
    *out = s;
    return true;
}

template <class T>
bool same_opt(const std::optional<T>& a, const std::optional<T>& b) {
    return a.has_value() == b.has_value() && (!a || *a == *b);
}

bool same_start(const StartingCamera& a, const StartingCamera& b) {
    const bool targets = a.target.has_value() == b.target.has_value() &&
                         (!a.target || (a.target->x == b.target->x && a.target->y == b.target->y &&
                                        a.target->z == b.target->z));
    return a.form == b.form && targets && same_opt(a.distance, b.distance) && same_opt(a.azimuth_deg, b.azimuth_deg) &&
           same_opt(a.elevation_deg, b.elevation_deg) && a.entity == b.entity;
}

constexpr size_t kMaxNotes = 8;

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}


std::string bytes_text(uint64_t b) {
    char t[32];
    if (b < 1024) std::snprintf(t, sizeof t, "%llu B", (unsigned long long)b);
    else if (b < 1024ull * 1024) std::snprintf(t, sizeof t, "%.1f KB", double(b) / 1024.0);
    else std::snprintf(t, sizeof t, "%.1f MB", double(b) / (1024.0 * 1024.0));
    return t;
}

std::string count_text(uint64_t n) {
    std::string s = std::to_string(n);
    for (int i = int(s.size()) - 3; i > 0; i -= 3) s.insert(size_t(i), ",");
    return s;
}

}  // namespace

namespace {
std::atomic<uint64_t> g_view_serial{0};
}  // namespace


void SceneView::ShapesDeleter::operator()(caliper_draw::ScenePieces* p) const {
#if defined(CALIPER_SCENE_PIECES)
    delete p;
#else
    (void)p;   // never made without the pieces
#endif
}

SceneView::SceneView(caliper_gfx::Backend& backend, const caliper_gfx::ProgramDesc* programs)
    : gfx_(backend), serial_(++g_view_serial) {
    (void)programs;
#if defined(CALIPER_SCENE_PIECES)
    shapes_.reset(new caliper_draw::ScenePieces());
#endif
}

// The programs' pick of the view: the readback of its ID target on its backend, the ticket of the
// request in flight, and the version of the item map that each program showed in the frame that
// the readback reads, by pick id.
struct SceneView::ProgramPick {
    std::unique_ptr<caliper_scene::FramePick> frame;
    uint64_t ticket = 0, last_ticket = 0;
#if defined(CALIPER_HAVE_DRAW)
    std::vector<std::pair<uint32_t, caliper_draw::ShownItems>> shown;
#endif
};

// One program of pieces that the view draws: its runtime, the submits of its frames that the
// GPU has not completed with their submit time, the state last reported to the feed, and its install.
struct SceneView::ProgramSlot {
#if defined(CALIPER_HAVE_DRAW)
    std::unique_ptr<caliper_draw::DrawRuntime> runtime;
#endif
    std::string program;
    uint64_t revision = 0;
    bool shape = false;   // a shape's piece (shapes_), else a program of the feed
    bool marks = false;   // a mark of the highlight (shapes_->marks()), drawn last into the colour alone
    std::deque<std::pair<caliper_gfx::Serial, std::chrono::steady_clock::time_point>> pending;
    int reported = -1;
    int64_t reported_step = -2;   // the resource step of the last report
    double reported_ms = -1;     // the completed draw timing of the last report
    std::string reported_warning; // the warning of the last report
    std::chrono::steady_clock::time_point installed;
};

SceneView::~SceneView() {
    program_pick_.reset();   // its readbacks go back to the layer with the view
    shape_pick_.reset();
    draws_.clear();
    shapes_.reset();
    if (id_) gfx_.release(id_);
    if (program_colormaps_) gfx_.release(program_colormaps_);
    for (auto& kv : slots_) release(kv.second);
    slots_.clear();
    if (color_) gfx_.release(color_);
    if (depth_) gfx_.release(depth_);
}

bool SceneView::init(std::string* why) {
#if defined(CALIPER_SCENE_PIECES)
    (void)why;
    return true;
#else
    if (why) *why = "scene.pieces_unavailable: this build has no library pieces";
    return false;
#endif
}

void SceneView::describe(const wire::SceneView& info) {
    const Space space = info.space == CALIPER_NET_SPACE_2D ? Space::k2D : Space::k3D;
    if (space != tree_.space()) {
        reset_scene();
        tree_ = SceneTree(space);
        camera_ = space == Space::k2D ? ViewCamera(PanZoom{}) : ViewCamera(Orbit{});
        fitted_ = moved_ = false;
    }
    Axis up = tree_.up();
    if (parse_axis(info.up, &up) && up != tree_.up()) {
        std::string err;
        if (!tree_.set_up(up, &err)) note("up '" + info.up + "': " + err);
    }
    // The scene's up is where a view starts: an orbit the person has not moved takes it, unless the
    // person chose a mode, which wins until a reset.
    if (Orbit* o = std::get_if<Orbit>(&camera_); o && !moved_ && !up_chosen_) o->up = tree_.up();
    if (PanZoom* p = std::get_if<PanZoom>(&camera_)) p->y_down = tree_.up() == Axis::NegY;
    if (!up_chosen_) up_mode_ = std::holds_alternative<Orbit>(camera_) ? std::get<Orbit>(camera_).up : tree_.up();
    tree_.set_units(info.units);
    // The starting camera from the scene's VIEWS entry; null is the fit.
    StartingCamera start;
    if (starting_from_wire(info.view_camera, &start) && !same_start(start, start_)) set_starting(start);
    changed();
}

void SceneView::set_starting(const StartingCamera& s) {
    start_ = s;
    // A view whose camera the person has not moved takes the new starting camera at its next frame;
    // a moved one keeps its camera until its next reset, because a frame frames the view again only
    // while it is not moved. Thus a starting camera that arrives during a click, whose wobble moves
    // the camera until the release puts it back (live/camera_drag.h), applies after the release.
    fitted_ = false;
    changed();
}

void SceneView::set_box_fit(bool on) {
    if (box_fit_ == on) return;
    box_fit_ = on;
    fitted_ = false;   // as set_starting: a frame frames again only a view that is not moved
    changed();
}

void SceneView::set_style(const std::string& path, const BatchStyle& style) {
    BatchStyle s = style;
    s.depth_offset = std::clamp(s.depth_offset, -8, 8);
    auto it = styles_.find(path);
    const bool plain = s.depth_offset == 0 && !(s.size_px > 0) && !s.point_shape;
    if (it != styles_.end() && it->second.depth_offset == s.depth_offset && it->second.size_px == s.size_px &&
        it->second.point_shape == s.point_shape)
        return;
    if (plain && it == styles_.end()) return;
    if (plain)
        styles_.erase(it);
    else
        styles_[path] = s;
    rebuild(path);
}

bool SceneView::set_orbit(const Orbit& o) {
    if (tree_.space() != Space::k3D) return false;
    look_through("", nullptr);
    Orbit next = o;
    next.elevation = std::clamp(next.elevation, -kMaxElevation, kMaxElevation);
    if (!(next.distance > 0) || !std::isfinite(next.distance)) return false;
    camera_ = next;
    moved_ = true;
    changed();
    return true;
}

bool SceneView::start_without_bounds() const {
    return start_.form == StartingCamera::Form::Orbit && start_.target && start_.distance;
}

void SceneView::apply_start(double aspect) {
    if (tree_.space() != Space::k3D) {
        caliper_scene::fit(camera_, tree_.scene_bounds(), aspect);
        return;
    }
    const StartResult r = caliper_scene::starting_camera(start_, tree_, aspect, box_fit_);
    if (const Orbit* o = std::get_if<Orbit>(&r.camera)) {
        Orbit next = *o;
        // A mode the person chose stays: the starting eye and target hold, and the picture rolls.
        if (up_chosen_ && next.up != up_mode_) caliper_scene::switch_up(next, up_mode_);
        camera_ = next;
    } else if (std::holds_alternative<ThroughPinhole>(r.camera)) {
        // The entity form: the view looks through that camera, over the orbit of the fit, which a
        // fit or the camera's absence returns to.
        Orbit o;
        o.up = up_chosen_ ? up_mode_ : tree_.up();
        camera_ = o;
        if (box_fit_)
            caliper_scene::fit_box(camera_, tree_.scene_bounds(), aspect);
        else
            caliper_scene::fit(camera_, tree_.scene_bounds(), aspect);
        through_ = start_.entity;
    }
    if (r.fell_back) note("view_camera names " + start_.entity + ", which holds no camera; the view opens on the fit");
}

void SceneView::note(const std::string& text) {
    if (std::find(notes_.begin(), notes_.end(), text) != notes_.end()) return;
    if (notes_.size() >= kMaxNotes) notes_.erase(notes_.begin());
    notes_.push_back(text);
}

void SceneView::begin_round() { counts_.bytes_last = 0; }

void SceneView::release(Slot& s) { s = Slot{}; }

void SceneView::reset_scene() {
    for (auto& kv : slots_) release(kv.second);
    slots_.clear();
#if defined(CALIPER_SCENE_PIECES)
    if (shapes_) shapes_->clear();
#endif
    tree_.reset();
    changed();
}

// The entity's batches from its slots: both layers of the entity's batches are cleared in the
// store, then the static layer of each part is set again, then the stamped one, which wins where
// both exist. Its transform and camera live in the store's own layers and stay.
void SceneView::rebuild(const std::string& path) {
    const EntityId e = tree_.find(path);
    if (!e) return;
    tree_.clear(e, false, true, kClearBatches);
    for (const bool fixed : {true, false})
        for (int p = 1; p <= kPrimitiveCount; ++p) {
            const auto it = slots_.find(Key{path, p, fixed});
            if (it == slots_.end() || !it->second.present) continue;
            std::string err;
            caliper_scene::Batch b = it->second.batch;
            if (auto st = styles_.find(path); st != styles_.end()) {
                b.depth_offset = int8_t(st->second.depth_offset);
                if (st->second.size_px > 0) b.size_px = st->second.size_px;
                if (st->second.point_shape) b.point_shape = *st->second.point_shape;
            }
            if (!tree_.set_batch(e, b, fixed, &err)) note((path.empty() ? "the root" : path) + ": " + err);
        }
    changed();
}

// ---- messages ------------------------------------------------------------------------------------

bool SceneView::apply(const wire::SceneUpdate& u, std::string* why) {
    if (!u.program) return decode_scene_update(u, nullptr, why);
    // The piece form: the arrays come back from the paired resources in the rows' dimension of the
    // entity, and the scene view's pieces draw the received bytes.
    const std::string where = (u.path.empty() ? std::string("the root") : u.path) + " " + u.part;
    std::string err;
    const EntityId e = tree_.ensure(u.path, &err);
    std::vector<wire::Section> arrays;
    std::vector<std::vector<uint8_t>> owned;
    const uint32_t dims = (tree_.space() == Space::k2D || (e && in_pixel_space(e))) ? 2u : 3u;
    if (!e || !caliper_live::scene_piece_arrays(u, dims, &arrays, &owned, &err)) {
        ++counts_.messages;
        const std::string t = where + ": " + err;
        note(t);
        if (why) *why = t;
        return false;
    }
    wire::SceneUpdate cu = u;
    cu.arrays = std::move(arrays);
    if (u.piece.find("name") && u.piece.find("name")->as_string() == "depth") cu.params.set("scale", 1.0);
    cu.program.reset();
    return decode_scene_update(cu, &u.paired, why);
}


bool SceneView::apply(const wire::SceneTransforms& t, std::string* why) {
    ++counts_.messages;
    const uint64_t k = t.paths.size();
    if (t.matrices.size < k * 64) {
        note("entity transforms: the matrices are short");
        if (why) *why = "entity transforms: the matrices are short";
        return false;
    }
    bool ok = true;
    for (uint64_t i = 0; i < k; ++i) {
        float m[16];
        std::memcpy(m, t.matrices.data + i * 64, 64);
        std::string err;
        const EntityId e = tree_.ensure(t.paths[i], &err);
        if (!e || !tree_.set_transform(e, m, t.is_static, &err)) {
            const std::string text = t.paths[i] + " transform: " + err;
            note(text);
            if (why && ok) *why = text;
            ok = false;
        }
    }
    changed();
    return ok;
}

bool SceneView::apply(const wire::SceneClasses& c, std::string* why) {
    ++counts_.messages;
    std::vector<ClassEntry> table;
    for (const caliper_net::Json& x : c.classes.items()) {
        ClassEntry ce;
        const caliper_net::Json* id = x.find("id");
        int64_t v = 0;
        if (!id || !id->get_i64(&v) || v < 0 || v > int64_t(kMaxClassId)) continue;
        ce.id = uint32_t(v);
        if (const caliper_net::Json* l = x.find("label"); l && l->is_string()) ce.label = l->as_string();
        if (!class_color(x, &ce.rgba)) ce.rgba = kPalette[ce.id % 16];
        table.push_back(std::move(ce));
    }
    std::string err;
    if (!tree_.set_classes(table, c.is_static, &err)) {
        note("class table: " + err);
        if (why) *why = "class table: " + err;
        return false;
    }
    changed();
    return true;
}

bool SceneView::apply(const wire::SceneClear& c, std::string* why) {
    ++counts_.messages;
    if (c.reset) {   // a new state: everything goes, static items included
        reset_scene();
        return true;
    }
    if (!tree_.find(c.path) && c.path != "") return true;   // nothing there
    std::vector<int> parts;
    uint32_t pose = 0;   // the transform and camera bits, which the store's layers hold
    if (c.parts) {
        for (const std::string& p : *c.parts) {
            if (p == CALIPER_NET_PART_TRANSFORM) pose |= kClearTransform;
            else if (p == CALIPER_NET_PART_CAMERA) pose |= kClearCamera;
            else if (const int k = wire::scene_part_index(p); k >= 0 && k < kPrimitiveCount) parts.push_back(k + 1);
        }
    } else {
        for (int p = 1; p <= kPrimitiveCount; ++p) parts.push_back(p);
        pose = kClearTransform | kClearCamera;
    }
    std::vector<std::string> touched;
    const std::string prefix = c.path.empty() ? std::string() : c.path + "/";
    for (auto& kv : slots_) {
        const Key& k = kv.first;
        const bool inside = k.path == c.path || (c.recursive && (c.path.empty() || k.path.compare(0, prefix.size(), prefix) == 0));
        if (!inside || !kv.second.present) continue;
        if (std::find(parts.begin(), parts.end(), k.primitive) == parts.end()) continue;
        if (k.fixed && !c.is_static) continue;   // a clear keeps static items unless it is static itself
        release(kv.second);
        if (std::find(touched.begin(), touched.end(), k.path) == touched.end()) touched.push_back(k.path);
    }
    for (const std::string& p : touched) rebuild(p);
    if (pose) {
        tree_.clear(c.path.empty() ? tree_.root() : tree_.find(c.path), c.recursive, c.is_static, pose);
        changed();
    }
    (void)why;
    return true;
}

bool SceneView::look_through(const std::string& path, std::string* why) {
    if (path.empty()) {
        if (!through_.empty()) changed();
        through_.clear();
        return true;
    }
    const EntityId e = tree_.find(path);
    if (!e || !tree_.camera(e)) {
        if (why) *why = path + " holds no camera";
        return false;
    }
    through_ = path;
    changed();
    return true;
}

ViewCamera SceneView::effective_camera() const {
    if (!through_.empty()) {
        const EntityId c = tree_.find(through_);
        if (c && tree_.camera(c)) {
            ThroughPinhole t;
            t.camera = c;
            return t;
        }
    }
    return camera_;
}

std::vector<std::string> SceneView::cameras() const {
    std::vector<std::string> out;
    for (uint32_t i : tree_.path_order()) {
        const EntityId e = tree_.entity_at(i);
        if (tree_.camera(e)) out.push_back(tree_.path(e));
    }
    return out;
}

std::vector<LabelAnchor> SceneView::labels() const {
    if (!width_ || !height_) return {};
    return project_labels(tree_, effective_camera(), width_, height_);
}

// ---- picking ---------------------------------------------------------------------------------------

bool SceneView::request_pick(float x, float y, std::string* why) {
    if (!width_ || !height_ || !id_) {
        if (why) *why = "the scene view has not drawn yet";
        return false;
    }
    if (!shape_pick_) shape_pick_ = std::make_unique<caliper_scene::FramePick>(gfx_);
    shape_asked_ = shape_pick_->request(id_, depth_, width_, height_, x, y, why);
    shape_generation_ = generation_;
    return shape_asked_;
}


SceneShapeCounts SceneView::shape_counts() const {
    SceneShapeCounts out;
#if defined(CALIPER_SCENE_PIECES)
    if (shapes_) {
        const caliper_draw::ShapeCounts c = shapes_->counts();
        out.pieces = c.pieces;
        out.marks = c.marks;
        out.received = c.received;
        return out;
    }
#endif
    return out;
}

void SceneView::set_piece_batch(const std::string& path, Primitive p, const caliper_draw::PieceBatch* batch) {
#if defined(CALIPER_SCENE_PIECES)
    if (!shapes_) return;
    if (batch) shapes_->set_override(path, p, *batch);
    else shapes_->clear_override(path, p);
    changed();
#else
    (void)path;
    (void)p;
    (void)batch;
#endif
}

bool SceneView::external_draws() const {
    for (const auto& d : draws_)
        if (!d->shape && !d->marks) return true;
    return false;
}

bool SceneView::active_external_draws() const {
#if defined(CALIPER_HAVE_DRAW)
    for (const auto& d : draws_) {
        if (d->shape || d->marks || !d->runtime) continue;
        const caliper_draw::DrawState st = d->runtime->status().state;
        if ((st == caliper_draw::DrawState::compiled || st == caliper_draw::DrawState::running) && !d->runtime->late())
            return true;
    }
#endif
    return false;
}

bool SceneView::program_pan_zoom() const {
#if defined(CALIPER_HAVE_DRAW)
    for (const auto& d : draws_)
        if (!d->shape && !d->marks && d->runtime)
            return d->runtime->camera() == caliper_builder::spec::CameraMode::pan_zoom;
#endif
    return false;
}

bool SceneView::normalized_program_camera() const {
    return program_pan_zoom() && !tree_.scene_bounds().valid;
}

void SceneView::sync_shapes() {
    piece_highlight_ = false;
#if defined(CALIPER_SCENE_PIECES)
    if (!shapes_) return;
    caliper_draw::ScenePieces::SyncState st;
    const ViewCamera cam = effective_camera();
    if (const ThroughPinhole* t = std::get_if<ThroughPinhole>(&cam)) st.through = t->camera;
    st.three_d = tree_.space() == Space::k3D;
    shapes_->sync(tree_, st);
    refresh_piece_notes();
    // The highlight of a selection with the pieces: the selected rows of each points and boxes batch.
    std::vector<caliper_draw::HighlightRows> rows;
    if (highlight_wanted())
        for (uint32_t idx : tree_.path_order()) {
            const EntityId e = tree_.entity_at(idx);
            if (!tree_.drawable(e)) continue;
            for (const Primitive p : {Primitive::Points, Primitive::Boxes}) {
                std::vector<int64_t> pick = highlight_rows(e, p);
                if (!pick.empty()) rows.push_back({tree_.path(e), p, std::move(pick)});
            }
        }
    uint64_t lit = 0;
    piece_highlight_ = !rows.empty() && shapes_->set_highlight(rows, clear_, width_, height_, &lit);
    if (!piece_highlight_) shapes_->set_highlight({}, clear_, width_, height_, nullptr);
    highlighted_ = piece_highlight_ ? lit : 0;
#endif
}

void SceneView::refresh_piece_notes() {
#if defined(CALIPER_SCENE_PIECES)
    if (!shapes_) return;
    for (const auto& message : piece_notes_)
        notes_.erase(std::remove(notes_.begin(), notes_.end(), message), notes_.end());
    piece_notes_.clear();
    for (const auto& error : shapes_->errors()) {
        piece_notes_.push_back(error.path + ": " + error.code + ": " + error.message);
        note(piece_notes_.back());
    }
#endif
}

void SceneView::set_draw_feed(caliper_draw::DrawFeed* feed) {
    if (feed == draw_feed_) return;
    draw_feed_ = feed;
    draw_revision_ = ~0ull;
    changed();
}

void SceneView::set_pointer(float x, float y, uint32_t buttons, bool over) {
    pointer_[0] = x;
    pointer_[1] = y;
    pointer_[2] = float(buttons);
    pointer_[3] = over ? 1.0f : 0.0f;
}

std::vector<const caliper_draw::DrawRuntime*> SceneView::draw_runtimes() const {
    std::vector<const caliper_draw::DrawRuntime*> out;
#if defined(CALIPER_HAVE_DRAW)
    for (const auto& d : draws_)
        if (d->runtime && !d->shape && !d->marks) out.push_back(d->runtime.get());
#endif
    return out;
}

std::vector<const caliper_draw::DrawRuntime*> SceneView::shape_runtimes() const {
    std::vector<const caliper_draw::DrawRuntime*> out;
#if defined(CALIPER_HAVE_DRAW)
    for (const auto& d : draws_)
        if (d->runtime && d->shape) out.push_back(d->runtime.get());
#endif
    return out;
}

uint64_t SceneView::program_frames() const {
    uint64_t n = 0;
#if defined(CALIPER_HAVE_DRAW)
    for (const caliper_draw::DrawRuntime* rt : draw_runtimes()) n += rt->status().frames;
#endif
    return n;
}

int64_t SceneView::program_step() const {
    int64_t step = -1;
#if defined(CALIPER_HAVE_DRAW)
    for (const caliper_draw::DrawRuntime* rt : draw_runtimes()) step = std::max<int64_t>(step, rt->status().resource_step);
#endif
    return step;
}

#if defined(CALIPER_HAVE_DRAW)
namespace {

// The box that a panel of programs alone frames: -1 to 1 on each axis, as clip space spans.
Aabb program_box() {
    Aabb b;
    for (int i = 0; i < 3; ++i) {
        b.lo[i] = -1.0;
        b.hi[i] = 1.0;
    }
    b.valid = true;
    return b;
}

// The feed receives changes of state, resource step, warning, and completed draw timing.
// A timing can arrive after the report of its resource step. The host emits
// an event only when the state or warning changes.
void report_state(caliper_draw::DrawFeed* feed, int* reported, int64_t* reported_step, double* reported_ms,
                  std::string* reported_warning, const caliper_draw::DrawRuntime& rt) {
    const caliper_draw::DrawStatus st = rt.status();
    const std::string warn =
        st.warning.ok() ? std::string() : st.warning.code + ":" + st.warning.where + ":" + st.warning.message;
    if (int(st.state) == *reported && st.resource_step == *reported_step && st.frame_ms == *reported_ms &&
        warn == *reported_warning)
        return;
    *reported = int(st.state);
    *reported_step = st.resource_step;
    *reported_ms = st.frame_ms;
    *reported_warning = warn;
    if (feed) feed->status(rt);
}

}  // namespace
#endif

void SceneView::reconcile_programs() {
#if defined(CALIPER_HAVE_DRAW)
    // The shapes' pieces first, in the tree's order, then the feed's programs.
    caliper_draw::DrawFeed* shapes = nullptr;
    caliper_draw::DrawFeed* marks = nullptr;
#if defined(CALIPER_SCENE_PIECES)
    if (shapes_) {
        shapes = shapes_.get();
        marks = &shapes_->marks();
    }
#endif
    const uint64_t srev = shapes ? shapes->revision() : ~1ull, frev = draw_feed_ ? draw_feed_->revision() : ~1ull,
                   mrev = marks ? marks->revision() : ~1ull;
    if (srev == shape_revision_ && frev == draw_revision_ && mrev == marks_revision_) return;
    shape_revision_ = srev;
    draw_revision_ = frev;
    marks_revision_ = mrev;
    const bool webgpu = std::strcmp(gfx_.caps().backend, "webgpu") == 0 || std::strcmp(gfx_.caps().backend, "dawn") == 0;
    std::vector<std::unique_ptr<ProgramSlot>> next;
    for (caliper_draw::DrawFeed* feed : {shapes, draw_feed_, marks}) {
        if (!feed) continue;
        const bool shape = feed == shapes, mark = feed == marks;
        for (const caliper_draw::PanelProgram& p : feed->programs()) {
            if (!p.refused.ok()) continue;   // the feed reports a refusal before any runtime
            std::unique_ptr<ProgramSlot> slot;
            for (auto& d : draws_)
                if (d && d->shape == shape && d->marks == mark && d->program == p.instance.program &&
                    d->revision == p.revision)
                    slot = std::move(d);
            if (!slot) {
                slot = std::make_unique<ProgramSlot>();
                slot->program = p.instance.program;
                slot->revision = p.revision;
                slot->shape = shape;
                slot->marks = mark;
                slot->runtime = std::make_unique<caliper_draw::DrawRuntime>(gfx_, webgpu, feed->checks());
                slot->runtime->install(p.piece, p.instance, p.pick_id);
                slot->installed = std::chrono::steady_clock::now();
#if defined(CALIPER_SCENE_PIECES)
                if (shape) shapes_->installed(p.instance.program);
#endif
                report_state(feed, &slot->reported, &slot->reported_step, &slot->reported_ms,
                             &slot->reported_warning, *slot->runtime);
            }
            next.push_back(std::move(slot));
        }
    }
    draws_ = std::move(next);   // a program no longer in the feed releases its objects
    refresh_piece_notes();
    panel_over_ = 0;
    changed();
#endif
}

bool SceneView::prepare_program_targets(std::string* why) {
#if defined(CALIPER_HAVE_DRAW)
    if (!id_ || id_w_ != width_ || id_h_ != height_) {
        if (id_) gfx_.release(id_);
        caliper_gfx::TextureDesc d;
        d.width = width_;
        d.height = height_;
        d.format = caliper_gfx::Format::RG32Uint;
        d.usage = caliper_gfx::kRenderTarget | caliper_gfx::kCopySrcTex;
        d.label = "scene view program ids";
        id_ = gfx_.create_texture(d);
        id_w_ = width_;
        id_h_ = height_;
        if (!id_) {
            if (why) *why = "no ID target for the programs: " + gfx_.last_error();
            return false;
        }
    }
    if (!program_colormaps_) {
        // The host's colormaps, one row for each colormap id of the layer (caliper_colormaps).
        caliper_gfx::TextureDesc c;
        c.width = 256;
        c.height = uint32_t(caliper_gfx::cmap::kBuiltinColormaps);
        c.format = caliper_gfx::Format::RGBA8Unorm;
        c.usage = caliper_gfx::kSampled | caliper_gfx::kCopyDstTex;
        c.label = "program colormaps";
        program_colormaps_ = gfx_.create_texture(c);
        std::vector<uint32_t> rows(256u * uint32_t(caliper_gfx::cmap::kBuiltinColormaps));
        for (int id = 0; id < caliper_gfx::cmap::kBuiltinColormaps; ++id)
            std::memcpy(rows.data() + 256u * uint32_t(id), caliper_gfx::cmap::builtin_lut(id), 256 * 4);
        if (program_colormaps_) gfx_.write_texture(program_colormaps_, rows.data(), 256 * 4);
    }
    return true;
#else
    (void)why;
    return false;
#endif
}

void SceneView::draw_programs() {
#if defined(CALIPER_HAVE_DRAW)
    const auto now = std::chrono::steady_clock::now();
    const double dt = last_program_frame_.time_since_epoch().count() == 0
                          ? 0.0
                          : std::chrono::duration<double>(now - last_program_frame_).count();
    last_program_frame_ = now;
    // The camera that the view draws with; a panel of programs alone has the box of program_box().
    const ViewCamera cam = effective_camera();
    caliper_scene::CameraMatrices cm;
    const double aspect = double(width_) / double(std::max<uint32_t>(1, height_));
    const Aabb bounds = tree_.scene_bounds().valid ? tree_.scene_bounds() : program_box();
    if (const Orbit* o = std::get_if<Orbit>(&cam)) {
        cm.view = caliper_scene::orbit_view(*o);
        cm.proj = caliper_scene::orbit_projection(*o, bounds, aspect);
        cm.eye = caliper_scene::orbit_eye(*o);
        cm.perspective = true;
    } else if (const PanZoom* pz = std::get_if<PanZoom>(&cam)) {
        // Standalone pieces use the panel's normalized square on both axes.
        // Scene geometry retains its world-space aspect ratio.
        cm.proj = caliper_scene::panzoom_projection(*pz, normalized_program_camera() ? 1.0 : aspect);
    } else {
        cm = caliper_scene::camera_matrices(cam, tree_, width_, height_);
    }
    const caliper_scene::Mat4 clip = cm.proj * cm.view;
    program_clip_ = clip;
    caliper_scene::Mat4 inv;
    if (!caliper_scene::inverse(clip, &inv)) inv = caliper_scene::Mat4{};
    caliper_draw::FrameInputs base;
    for (int i = 0; i < 16; ++i) {
        base.clip_from_world[size_t(i)] = float(clip.m[i]);
        base.world_from_clip[size_t(i)] = float(inv.m[i]);
        base.view_from_world[size_t(i)] = float(cm.view.m[i]);
    }
    base.eye = {float(cm.eye.x), float(cm.eye.y), float(cm.eye.z), cm.perspective ? 1.0f : 0.0f};
    base.viewport = {0, 0, float(width_), float(height_)};
    base.pointer = {pointer_[0], pointer_[1], pointer_[2], pointer_[3]};
    base.dt_s = dt;
    base.pixel_ratio = scale_;
    caliper_draw::HostTarget target;
    target.color = color_;
    target.id = id_;
    target.depth = depth_;
    target.width = width_;
    target.height = height_;
    target.colormaps = program_colormaps_;
    const caliper_scene::CameraMatrices shape_cm = caliper_scene::camera_matrices(cam, tree_, width_, height_);
#if defined(CALIPER_SCENE_PIECES)
    // The shapes' matrices are the scene core's, of the camera it drew with.
    if (shapes_) shapes_->set_camera(shape_cm);
#endif
    for (auto& d : draws_) {
        if (!d->runtime) continue;
        caliper_draw::DrawFeed* feed = slot_feed(*d);
        caliper_draw::FrameInputs in = base;
        in.time_s = std::chrono::duration<double>(now - d->installed).count();
        if (feed) feed->frame(*d->runtime, &in);
        std::vector<caliper_gfx::Frame> frames;
        // A mark of the highlight draws into the colour alone: no depth hides it, and the picks
        // read the shapes under it.
        caliper_draw::HostTarget bare = target;
        bare.id = {};
        bare.depth = {};
        if (d->runtime->encode(&frames, in, d->marks ? bare : target)) {
            for (caliper_gfx::Frame& f : frames) {
                const caliper_gfx::Serial s = gfx_.submit(std::move(f));
                if (s == 0) {
                    d->runtime->frame_refused(gfx_.last_error());
                    break;
                }
                d->pending.emplace_back(s, std::chrono::steady_clock::now());
            }
        }
        report_state(feed, &d->reported, &d->reported_step, &d->reported_ms, &d->reported_warning, *d->runtime);
    }
    refresh_piece_notes();
#endif
}

void SceneView::poll_programs() {
#if defined(CALIPER_HAVE_DRAW)
    bool completed = false;
    for (auto& d : draws_) {
        if (!d->runtime) continue;
        // Never a wait: the completed submits give their GPU time, or < 0 without timestamps, for
        // work that the driver ended, or for a submit whose time the backend no longer keeps (the
        // panel was not drawn for a while), and the time from the submit until the work was seen
        // done, an upper bound. A frame not done after 1 s is late, and the runtime judges it by
        // its GPU time once it is done.
        bool slot_completed = false;
        while (!d->pending.empty()) {
            double ms = -1;
            if (!gfx_.submit_gpu_ms(d->pending.front().first, &ms)) break;
            d->runtime->frame_done(ms, ms_since(d->pending.front().second));
            d->pending.pop_front();
            slot_completed = true;
            completed = true;
        }
        if (slot_completed && !d->pending.empty())
            d->pending.front().second = std::max(d->pending.front().second, std::chrono::steady_clock::now());
        if (!d->pending.empty()) d->runtime->frame_pending(ms_since(d->pending.front().second));
        const int prev_state = d->reported;
        report_state(slot_feed(*d), &d->reported, &d->reported_step, &d->reported_ms, &d->reported_warning,
                     *d->runtime);
        if (d->reported != prev_state && d->runtime->status().state == caliper_draw::DrawState::stopped) changed();
    }
    refresh_piece_notes();
    // The programs of the panel together: over 16.7 ms a frame on 30 polls in a row warns on the
    // costliest one, which names its costliest pass after one measured frame.
    if (!completed) return;
    double total = 0, worst_ms = -1;
    caliper_draw::DrawRuntime* worst = nullptr;
    for (auto& d : draws_) {
        if (!d->runtime) continue;
        const caliper_draw::DrawStatus st = d->runtime->status();
        if (st.state != caliper_draw::DrawState::running || st.frame_ms < 0) continue;
        total += st.frame_ms;
        if (st.frame_ms > worst_ms) {
            worst_ms = st.frame_ms;
            worst = d->runtime.get();
        }
    }
    if (total <= 16.7) {
        panel_over_ = 0;
        for (auto& d : draws_) {
            if (!d->runtime) continue;
            d->runtime->clear_panel_warning();
            report_state(slot_feed(*d), &d->reported, &d->reported_step, &d->reported_ms, &d->reported_warning,
                         *d->runtime);
        }
        return;
    }
    ++panel_over_;
    if (panel_over_ >= 30 && worst != nullptr) {
        if (worst->status().warning.ok()) {
            char b[96];
            std::snprintf(b, sizeof b, "the programs of its panel took %.1f ms a frame together, above 16.7 ms", total);
            worst->request_warning(caliper_builder::programs::kProgramDrawOverTime, b);
        }
        panel_over_ = 0;
    }
#endif
}

caliper_draw::DrawFeed* SceneView::slot_feed(const ProgramSlot& slot) const {
#if defined(CALIPER_SCENE_PIECES)
    if (slot.shape) return shapes_.get();
    if (slot.marks) return shapes_ ? &shapes_->marks() : nullptr;
#endif
    return slot.shape || slot.marks ? nullptr : draw_feed_;
}


PickResult SceneView::poll_pick(Hit* out) {
    if (!shape_asked_ || !shape_pick_) return PickResult::None;
    FrameHit frame;
    const auto result = shape_pick_->poll([this](uint32_t id) {
#if defined(CALIPER_SCENE_PIECES)
        return shapes_ && shapes_->owns(id);
#else
        (void)id;
        return false;
#endif
    }, &frame);
    if (result == PickResult::Pending) return result;
    shape_asked_ = false;
    if (shape_generation_ != generation_) return PickResult::Dropped;
#if defined(CALIPER_SCENE_PIECES)
    if (result == PickResult::Hit && shapes_ && shapes_->hit(frame, shape_pick_->window(), out)) return PickResult::Hit;
#endif
    return PickResult::Miss;
}

uint64_t SceneView::request_program_pick(float x, float y, std::string* why) {
#if defined(CALIPER_HAVE_DRAW)
    if (!external_draws() || !id_) {
        if (why) *why = "the view draws no program";
        return 0;
    }
    if (!program_pick_) program_pick_ = std::make_unique<ProgramPick>();
    ProgramPick& p = *program_pick_;
    if (!p.frame) p.frame = std::make_unique<caliper_scene::FramePick>(gfx_);
    p.ticket = 0;
    p.shown.clear();
    if (!p.frame->request(id_, depth_, width_, height_, x, y, why)) return 0;
    // The readback reads the frames submitted so far, whose last one showed these versions.
    for (const auto& d : draws_)
        if (d->runtime && !d->shape) p.shown.emplace_back(d->runtime->pick_id(), d->runtime->shown_items());
    p.ticket = ++p.last_ticket;
    return p.ticket;
#else
    (void)x;
    (void)y;
    if (why) *why = "the build draws no program";
    return 0;
#endif
}

PickResult SceneView::poll_program_pick(uint64_t ticket, caliper_draw::PickedItem* out, float* depth) {
#if defined(CALIPER_HAVE_DRAW)
    if (!program_pick_ || ticket == 0) return PickResult::None;
    ProgramPick& p = *program_pick_;
    if (ticket != p.ticket) return ticket < p.last_ticket ? PickResult::Dropped : PickResult::None;
    const auto is_program = [&p](uint32_t id) {
        for (const auto& s : p.shown)
            if (s.first == id) return true;
        return false;
    };
    caliper_scene::FrameHit fh;
    const PickResult r = p.frame->poll(is_program, &fh);
    if (r == PickResult::Pending || r == PickResult::None) return r;
    p.ticket = 0;
    PickResult answer = PickResult::Miss;
    if (r == PickResult::Hit)
        for (const auto& s : p.shown)
            if (s.first == fh.pick_id)
                if (std::optional<caliper_draw::PickedItem> picked = caliper_draw::DrawRuntime::pick_shown(fh.item, s.second)) {
                    if (out) *out = std::move(*picked);
                    if (depth) *depth = fh.depth;
                    answer = PickResult::Hit;
                    break;
                }
    p.shown.clear();
    return answer;
#else
    (void)ticket;
    (void)out;
    (void)depth;
    return PickResult::None;
#endif
}

int64_t SceneView::row_id(const std::string& path, Primitive p, int64_t index) const {
    const EntityId e = tree_.find(path);
    const bool fixed = e && tree_.batch_static(e, p);
    const auto it = slots_.find(Key{path, int(p), fixed});
    if (it == slots_.end() || it->second.ax_size <= 0) return index;
    const Slot& s = it->second;
    // start >= 0, step >= 1, count >= 0 and size >= 1 (see apply): every product is checked, and an
    // index whose source row int64_t cannot hold keeps its batch index.
    if (index < 0 || (s.ax_count && index > INT64_MAX / s.ax_count)) return index;
    const int64_t q = index * s.ax_count / s.ax_size;
    if (q && s.ax_step > (INT64_MAX - s.ax_start) / q) return index;
    return s.ax_start + q * s.ax_step;
}

std::string SceneView::pick_detail(const Hit& h) const {
    if (h.part == DrawPart::Frustum || h.part == DrawPart::Axes) return {};
    const std::string path = tree_.path(h.entity);
    // A depth cloud's point is a pixel of its depth image; an image's item is its pixel too.
    int64_t index = int64_t(h.item);
    if (h.primitive == Primitive::Image && h.part == DrawPart::Batch) {
        const Batch* b = tree_.batch(h.entity, Primitive::Image);
        index = h.has_pixel && b ? int64_t(h.pixel[1]) * b->image_width + h.pixel[0] : 0;
    }
    caliper_net::Json d = caliper_net::Json::object();
    d.set("entity", caliper_net::Json(path));
    d.set("primitive", caliper_net::Json(primitive_name(h.primitive)));
    d.set("index", caliper_net::Json(index));
    d.set("row_id", caliper_net::Json(row_id(path, h.primitive, index)));
    if (h.primitive == Primitive::Image && h.has_pixel) {
        caliper_net::Json px = caliper_net::Json::array();
        px.push(caliper_net::Json(int64_t(h.pixel[0])));
        px.push(caliper_net::Json(int64_t(h.pixel[1])));
        d.set("pixel", px);
    }
    if (h.has_position) {
        caliper_net::Json pos = caliper_net::Json::array();
        for (double c : {h.position.x, h.position.y, h.position.z}) pos.push(caliper_net::Json(c));
        d.set("position", pos);
    }
    return d.dump();
}

// ---- selections --------------------------------------------------------------------------------------

std::string SceneView::select_polygon(const std::vector<float>& poly) {
    selected_.clear();
    const size_t m = poly.size() / 2;
    if (m >= 3 && width_ && height_) {
        float xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY;
        for (size_t k = 0; k < m; ++k) {
            xmin = std::min(xmin, poly[2 * k]);
            xmax = std::max(xmax, poly[2 * k]);
            ymin = std::min(ymin, poly[2 * k + 1]);
            ymax = std::max(ymax, poly[2 * k + 1]);
        }
        auto inside = [&](double px, double py) {
            if (!(px >= xmin && px <= xmax && py >= ymin && py <= ymax)) return false;
            bool in = false;
            for (size_t a = 0, b = m - 1; a < m; b = a++) {
                const double ax = poly[2 * a], ay = poly[2 * a + 1], bx = poly[2 * b], by = poly[2 * b + 1];
                if ((ay > py) != (by > py) && px < (bx - ax) * (py - ay) / (by - ay) + ax) in = !in;
            }
            return in;
        };
        tree_.update();
        const CameraMatrices cm = camera_matrices(effective_camera(), tree_, width_, height_);
        if (cm.ok)
            for (uint32_t idx : tree_.path_order()) {
                const EntityId e = tree_.entity_at(idx);
                if (!tree_.drawable(e)) continue;
                const EntityId pcam = tree_.pixel_space_of(e);
                const bool overlay = pcam && pcam == cm.through;
                const Mat4 mvp = overlay ? cm.pixel_to_clip * tree_.pixel_local(e) : cm.proj * cm.view * tree_.world(e);
                for (const Primitive p : {Primitive::Points, Primitive::Boxes}) {
                    const Batch* b = tree_.batch(e, p);
                    if (!b) continue;
                    const auto it = slots_.find(Key{tree_.path(e), int(p), tree_.batch_static(e, p)});
                    if (it == slots_.end()) continue;
                    const auto rows = it->second.cpu.find(int(p == Primitive::Points ? Role::Positions : Role::Centers));
                    if (rows == it->second.cpu.end()) continue;
                    const uint32_t comps = b->dims();
                    const uint64_t n = b->count();
                    const float* f = reinterpret_cast<const float*>(rows->second.data());
                    std::vector<int64_t> idxs;
                    for (uint64_t i = 0; i < n; ++i) {
                        const Vec3 q{f[i * comps], f[i * comps + 1], comps == 3 ? f[i * comps + 2] : 0.0};
                        double h[4];
                        transform_h(mvp, q, h);
                        if (!(h[3] > 0)) continue;
                        const double z = h[2] / h[3];
                        if (!(z >= 0 && z <= 1)) continue;
                        if (inside((h[0] / h[3] + 1) * 0.5 * width_, (1 - h[1] / h[3]) * 0.5 * height_)) idxs.push_back(int64_t(i));
                    }
                    if (!idxs.empty()) selected_[{tree_.path(e), int(p)}] = std::move(idxs);
                }
            }
    }
    changed();
    if (selected_.empty()) return "{}";
    // The detail: at most 32,768 indices in all, in path order.
    caliper_net::Json ents = caliper_net::Json::array();
    size_t total = 0;
    bool truncated = false;
    std::string space;
    std::vector<int64_t> items;
    for (const auto& kv : selected_) {
        const Primitive p = Primitive(kv.first.second);
        caliper_net::Json ent = caliper_net::Json::object();
        caliper_net::Json idx = caliper_net::Json::array(), rows = caliper_net::Json::array();
        const auto it = slots_.find(Key{kv.first.first, kv.first.second, tree_.batch_static(tree_.find(kv.first.first), p)});
        const std::string items_space = it != slots_.end() ? it->second.items : std::string();
        for (const int64_t i : kv.second) {
            if (total >= CALIPER_NET_MAX_EVENT_INDICES) {
                truncated = true;
                break;
            }
            const int64_t r = row_id(kv.first.first, p, i);
            idx.push(caliper_net::Json(i));
            rows.push(caliper_net::Json(r));
            ++total;
            if (!items_space.empty() && (space.empty() || space == items_space)) {
                space = items_space;
                items.push_back(r);
            }
        }
        ent.set("entity", caliper_net::Json(kv.first.first));
        ent.set("primitive", caliper_net::Json(primitive_name(p)));
        ent.set("indices", idx);
        ent.set("row_ids", rows);
        ents.push(ent);
        if (truncated) break;
    }
    caliper_net::Json d = caliper_net::Json::object();
    d.set("entities", ents);
    if (truncated) d.set("truncated", caliper_net::Json(true));
    if (!space.empty()) {
        std::sort(items.begin(), items.end());
        items.erase(std::unique(items.begin(), items.end()), items.end());
        caliper_net::Json it = caliper_net::Json::array();
        for (const int64_t x : items) it.push(caliper_net::Json(x));
        d.set("space", caliper_net::Json(space));
        d.set("items", it);
    }
    return d.dump();
}

bool SceneView::clear_selection() {
    const bool had = !selected_.empty();
    selected_.clear();
    if (had) changed();
    return had;
}

void SceneView::set_linked(const std::string& space, std::vector<int64_t> items) {
    std::sort(items.begin(), items.end());
    items.erase(std::unique(items.begin(), items.end()), items.end());
    const auto it = linked_.find(space);
    if (items.empty()) {
        if (it == linked_.end()) return;
        linked_.erase(it);
    } else {
        if (it != linked_.end() && it->second == items) return;
        linked_[space] = std::move(items);
    }
    changed();
}

void SceneView::set_all_linked(const std::map<std::string, std::vector<int64_t>>& by_space) {
    std::vector<std::string> gone;
    for (const auto& kv : linked_)
        if (!by_space.count(kv.first)) gone.push_back(kv.first);
    for (const std::string& sp : gone) set_linked(sp, {});
    for (const auto& kv : by_space) set_linked(kv.first, kv.second);
}

uint64_t SceneView::selected_count() const {
    uint64_t n = 0;
    for (const auto& kv : selected_) n += kv.second.size();
    return n;
}

// The highlight: a scene of the selected points and boxes alone, each entity with the poses and
// cameras of its path in the view's scene, so that the view's camera draws them where they are. Their
// arrays are the selected rows of the CPU copies, uploaded into buffers the view owns (copies, counted).


std::vector<int64_t> SceneView::highlight_rows(EntityId e, Primitive p) const {
    std::vector<int64_t> pick;
    const Batch* b = tree_.batch(e, p);
    if (!b) return pick;
    const std::string& path = tree_.path(e);
    const auto it = slots_.find(Key{path, int(p), tree_.batch_static(e, p)});
    if (it == slots_.end()) return pick;
    const Slot& slot = it->second;
    const uint64_t n = b->count();
    if (const auto sel = selected_.find({path, int(p)}); sel != selected_.end()) pick = sel->second;
    if (const auto lk = linked_.find(slot.items); !slot.items.empty() && lk != linked_.end())
        for (uint64_t i = 0; i < n; ++i)
            if (std::binary_search(lk->second.begin(), lk->second.end(), row_id(path, p, int64_t(i))))
                pick.push_back(int64_t(i));
    std::sort(pick.begin(), pick.end());
    pick.erase(std::unique(pick.begin(), pick.end()), pick.end());
    while (!pick.empty() && pick.back() >= int64_t(n)) pick.pop_back();
    return pick;
}

bool SceneView::in_pixel_space(EntityId e) const {
    for (EntityId p = tree_.parent(e); p; p = tree_.parent(p))
        if (tree_.camera(p)) return true;
    return false;
}

// ---- camera --------------------------------------------------------------------------------------

void SceneView::orbit(float dx, float dy) {
    if (Orbit* o = std::get_if<Orbit>(&camera_)) {
        o->azimuth += double(dx) * 0.01;
        o->elevation = std::clamp(o->elevation + double(dy) * 0.01, -kMaxElevation, kMaxElevation);
        moved_ = true;
        changed();
    } else {
        pan(dx, dy, float(std::max<uint32_t>(height_, 1)) / scale_);
    }
}

void SceneView::pan(float dx, float dy, float view_height) {
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
        const double sx = normalized_program_camera() ? double(height_) / std::max<uint32_t>(width_, 1) : 1.0;
        p->center[0] -= double(dx) * k * sx;
        p->center[1] += (p->y_down ? -1.0 : 1.0) * double(dy) * k;
    }
    moved_ = true;
    changed();
}

void SceneView::zoom(float wheel, float mx, float my, float view_w, float view_h) {
    if (wheel == 0.0f) return;
    const double f = std::pow(0.9, double(wheel));
    const Aabb b = tree_.scene_bounds();
    const double r = b.valid ? std::max(b.radius(), 1e-6) : 1.0;
    if (Orbit* o = std::get_if<Orbit>(&camera_)) {
        o->distance = std::clamp(o->distance * f, r * 1e-3, r * 1e3);
    } else if (PanZoom* p = std::get_if<PanZoom>(&camera_)) {
        const double aspect = !normalized_program_camera() && view_h > 0 ? double(view_w) / double(view_h) : 1.0;
        const double sy = p->y_down ? -1.0 : 1.0;
        const double wx = p->center[0] + mx * p->half_height * aspect, wy = p->center[1] + sy * my * p->half_height;
        p->half_height = std::clamp(p->half_height * f, r * 1e-5, r * 1e5);
        p->center[0] = wx - mx * p->half_height * aspect;
        p->center[1] = wy - sy * my * p->half_height;
    }
    moved_ = true;
    changed();
}

void SceneView::fit() {
    look_through("", nullptr);   // a fit leaves a pinhole for the orbit of the fit
    tree_.update();
    const double aspect = (width_ && height_) ? double(width_) / double(height_) : 1.0;
    if (normalized_program_camera()) {
        PanZoom p;
        p.y_down = false;
        camera_ = p;
    } else if (box_fit_)
        caliper_scene::fit_box(camera_, tree_.scene_bounds(), aspect);
    else
        caliper_scene::fit(camera_, tree_.scene_bounds(), aspect);
    fitted_ = tree_.scene_bounds().valid;
    fitted_bounds_ = tree_.scene_bounds();
    fitted_aspect_ = aspect;
    changed();
}

bool SceneView::switch_up(Axis up) {
    if (tree_.space() != Space::k3D) return false;
    up_mode_ = up;
    up_chosen_ = true;
    if (Orbit* o = std::get_if<Orbit>(&camera_); o && o->up != up) {
        caliper_scene::switch_up(*o, up);
        changed();
    }
    return true;
}

std::vector<Axis> SceneView::up_choices() const {
    if (tree_.space() != Space::k3D) return {};
    std::vector<Axis> c = {Axis::PosY, Axis::PosZ};
    for (Axis a : {tree_.up(), up_mode_})
        if (std::find(c.begin(), c.end(), a) == c.end()) c.push_back(a);
    return c;
}

void SceneView::reset() {
    if (tree_.space() == Space::k3D && !program_pan_zoom()) {
        Orbit o;
        o.up = tree_.up();
        camera_ = o;
    } else {
        PanZoom p;
        p.y_down = !program_pan_zoom() && tree_.up() == Axis::NegY;
        camera_ = p;
    }
    through_.clear();
    up_mode_ = tree_.up();
    up_chosen_ = moved_ = fitted_ = was_through_ = false;
    changed();
}

void SceneView::restore(const Settings& s) {
    scale_bar_on_ = s.scale_bar;
    if (!s.up_chosen || tree_.space() != Space::k3D) return;
    up_mode_ = s.up;
    up_chosen_ = true;
    if (Orbit* o = std::get_if<Orbit>(&camera_)) {
        if (!moved_) o->up = s.up;   // a view that opens starts about its remembered axis
        else if (o->up != s.up) caliper_scene::switch_up(*o, s.up);
    }
    changed();
}

ScaleBar SceneView::scale_bar(double w, double h) const {
    if (!scale_bar_on_) return {};
    return caliper_scene::scale_bar(effective_camera(), tree_.scene_bounds().valid, tree_.units(), w, h);
}

// ---- drawing ---------------------------------------------------------------------------------------

void SceneView::set_entity_filter(const std::string& key, std::function<bool(const std::string&)> keep) {
    if (key == filter_key_ && bool(keep) == bool(filter_)) return;
    filter_key_ = key;
    filter_ = std::move(keep);
    filtered_rev_ = ~0ull;
}

void SceneView::apply_entity_filter() {
    if (tree_.revision() == filtered_rev_) return;
    const uint32_t n = uint32_t(tree_.entity_count());
    std::vector<uint8_t> keep(n, 1);
    if (filter_) {
        for (uint32_t i = 1; i < n; ++i) {
            const EntityId e = tree_.entity_at(i);
            keep[i] = tree_.valid(e) && filter_(tree_.path(e)) ? 1 : 0;
        }
        for (uint32_t i = 1; i < n; ++i) {   // the ancestors of a kept entity stay, else it would hide
            const EntityId e = tree_.entity_at(i);
            if (!keep[i] || !tree_.valid(e)) continue;
            for (EntityId p = tree_.parent(e); tree_.valid(p) && p.index != 0 && p.index < n; p = tree_.parent(p))
                keep[p.index] = 1;
        }
    }
    for (uint32_t i = 1; i < n; ++i) {
        const EntityId e = tree_.entity_at(i);
        if (tree_.valid(e)) tree_.set_visible(e, keep[i] != 0);
    }
    filtered_rev_ = tree_.revision();
}

bool SceneView::render(uint32_t width, uint32_t height, float pixel_scale, uint32_t clear_rgba, std::string* why) {
    if (width == 0 || height == 0) return true;
    apply_entity_filter();
    const bool depth = tree_.space() == Space::k3D;
    if (!color_ || width != width_ || height != height_ || bool(depth_) != depth) {
        if (color_) gfx_.release(color_);
        if (depth_) gfx_.release(depth_);
        depth_ = {};
        caliper_gfx::TextureDesc d;
        d.width = width;
        d.height = height;
        d.format = caliper_gfx::Format::RGBA8Unorm;
        d.usage = caliper_gfx::kRenderTarget | caliper_gfx::kSampled | caliper_gfx::kCopySrcTex;
        d.label = "scene view colour";
        color_ = gfx_.create_texture(d);
        if (depth) {
            d.format = caliper_gfx::Format::Depth32Float;
            d.usage = caliper_gfx::kRenderTarget | caliper_gfx::kCopySrcTex;   // a program's pick reads it
            d.label = "scene view depth";
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
        ++target_creates_;
        changed();
    }
    if (pixel_scale != scale_ || clear_rgba != clear_) {
        scale_ = pixel_scale;
        clear_ = clear_rgba;
        changed();
    }
    if (tree_.revision() != drawn_revision_ || tree_.pose_revision() != drawn_pose_) changed();
    // The feed's programs draw in every frame: their time, their draw state and their resources
    // move. Scene pieces draw when the scene changes.
    reconcile_programs();
    if (active_external_draws()) changed();
    // The piece selects the host camera even when the empty scene defaults to 3-D.
    const bool pan_zoom = program_pan_zoom();
    if (pan_zoom != program_camera_active_ || (pan_zoom && !std::holds_alternative<PanZoom>(camera_))) {
        program_camera_active_ = pan_zoom;
        reset();
    }
    // Back on the orbit after looking through a pinhole (the view's own camera, or a camera entity
    // looked through): the mode chosen meanwhile applies.
    const bool through_now = std::holds_alternative<ThroughPinhole>(effective_camera());
    if (Orbit* o = std::get_if<Orbit>(&camera_); o && !through_now) {
        if (was_through_ && o->up != up_mode_) {
            caliper_scene::switch_up(*o, up_mode_);
            changed();
        }
        was_through_ = false;
    } else {
        was_through_ = through_now;
    }
    if (!dirty_) return true;

    const auto t0 = std::chrono::steady_clock::now();
    tree_.update();
    // The view follows the scene until the person moves its camera: it opens on its starting
    // camera, filled from the fit, once the scene has bounds, and frames them again when they
    // grow past what it framed (a late attach, parts that arrive one by one) or when the picture
    // changes shape. Once the person orbits, pans or zooms, only a fit or a reset moves it.
    {
        Aabb bounds = tree_.scene_bounds();
#if defined(CALIPER_HAVE_DRAW)
        const bool box = !bounds.valid && !draws_.empty();   // a panel of programs alone
        if (box) bounds = program_box();
#else
        const bool box = false;
#endif
        const double aspect = double(width_) / double(height_);
        const bool can = bounds.valid || start_without_bounds();
        const bool stale = !fitted_ || !caliper_scene::aabb_contains(fitted_bounds_, bounds, 0.01) ||
                           std::fabs(aspect - fitted_aspect_) > 0.01 * fitted_aspect_;
        if (!moved_ && can && stale) {
            if (box && pan_zoom) {
                // The piece already reserves its margins inside [-1, 1].
                PanZoom p;
                p.y_down = false;
                camera_ = p;
            } else if (box) {
                // The starting camera over the box of the programs, as apply_start() gives it over the
                // scene's bounds: an orbit's angles, the fit at them, then its target and distance.
                const bool orbit_form = start_.form == StartingCamera::Form::Orbit;
                if (std::holds_alternative<Orbit>(camera_)) {
                    Orbit next;
                    next.up = tree_.up();
                    if (orbit_form && start_.azimuth_deg) next.azimuth = *start_.azimuth_deg * kPi / 180.0;
                    if (orbit_form && start_.elevation_deg) next.elevation = *start_.elevation_deg * kPi / 180.0;
                    ViewCamera v = next;
                    caliper_scene::fit(v, bounds, aspect);
                    Orbit& f = std::get<Orbit>(v);
                    if (orbit_form && start_.target) {
                        f.target[0] = start_.target->x;
                        f.target[1] = start_.target->y;
                        f.target[2] = start_.target->z;
                    }
                    if (orbit_form && start_.distance) f.distance = *start_.distance;
                    // A mode the person chose stays, as in apply_start().
                    if (up_chosen_ && f.up != up_mode_) caliper_scene::switch_up(f, up_mode_);
                    camera_ = v;
                } else {
                    caliper_scene::fit(camera_, bounds, aspect);
                }
            } else if (pan_zoom) {
                caliper_scene::fit(camera_, bounds, aspect);
            } else {
                apply_start(aspect);
            }
            fitted_ = true;
            fitted_bounds_ = bounds;
            fitted_aspect_ = aspect;
        }
    }
    // Reconcile library pieces with the updated scene tree.
    sync_shapes();
    reconcile_programs();
    cost_.compile_ms = ms_since(t0);
    cost_.draws = uint32_t(draws_.size());
    caliper_gfx::Frame frame;
    caliper_gfx::RenderTarget target;
    target.color = color_;
    target.depth = depth_;
    target.clear_rgba = clear_;
    frame.render(caliper_gfx::DrawList(target));
    if (!prepare_program_targets(why)) return false;
    caliper_gfx::RenderTarget ids;
    ids.color = id_;
    ids.clear = true;
    ids.clear_rgba = 0;
    frame.render(caliper_gfx::DrawList(ids));
    const auto t1 = std::chrono::steady_clock::now();
    const caliper_gfx::Serial s = gfx_.submit(std::move(frame));
    cost_.submit_ms = ms_since(t1);
    dirty_ = false;
    drawn_revision_ = tree_.revision();
    drawn_pose_ = tree_.pose_revision();
    if (s == 0) {
        if (why) *why = "the layer refused the frame: " + gfx_.last_error();
        return false;
    }
    ++cost_.frames;
    ++generation_;
    drawn_camera_ = effective_camera();
    draw_programs();
    return true;
}

void SceneView::poll() {
    gfx_.poll();
    poll_programs();
    const caliper_gfx::Stats st = gfx_.take_stats();
    if (st.gpu_ms >= 0) cost_.gpu_ms = st.gpu_ms;

}

uint64_t SceneView::imgui_texture() const { return color_ ? gfx_.imgui_texture_id(color_) : 0; }

bool SceneView::read_pixels(std::vector<uint8_t>* rgba, std::string* why) {
    if (!color_) {
        if (why) *why = "no render target yet";
        return false;
    }
    const caliper_gfx::Readback rb = gfx_.read_texture(color_, 0, 0, width_, height_);
    if (!rb || !gfx_.wait(gfx_.last_use(color_)) || !gfx_.take_readback(rb, rgba)) {
        if (why) *why = "the readback failed: " + gfx_.last_error();
        return false;
    }
    return true;
}

// ---- reading -------------------------------------------------------------------------------------------

SceneCounts SceneView::counts() const {
    SceneCounts c = counts_;
    std::vector<std::string> seen;
    c.batches = 0;
    c.items = 0;
    c.arrays_live = 0;
    for (const auto& kv : slots_) {
        c.arrays_live += kv.second.cpu.size();

    }
    for (size_t i = 0; i < tree_.entity_count(); ++i) {
        const EntityId e = tree_.entity_at(uint32_t(i));
        bool any = false;
        for (int p = 1; p <= kPrimitiveCount; ++p) {
            const Batch* b = tree_.batch(e, Primitive(p));
            if (!b) continue;
            any = true;
            ++c.batches;
            const BatchArray* idx = b->array(Role::Indices);
            switch (b->primitive) {
                case Primitive::Lines: {
                    const uint64_t v = b->count();
                    c.items += b->line_mode == LineMode::Edges ? (idx ? idx->count : 0)
                               : b->line_mode == LineMode::Strips ? (v ? v - 1 : 0)
                                                                  : v / 2;
                    break;
                }
                case Primitive::Mesh: c.items += idx ? idx->count : b->count() / 3; break;
                case Primitive::Image: c.items += 1; break;
                default: c.items += b->count(); break;
            }
        }
        if (any) ++c.entities;
    }
    return c;
}

std::string SceneView::info() const {
    const SceneCounts c = counts();
    std::string s = count_text(c.entities) + (c.entities == 1 ? " entity  " : " entities  ") + count_text(c.items) +
                    " items";
    if (c.copies_total) s += "  copied " + bytes_text(c.bytes_last);
    if (cost_.gpu_ms >= 0) {
        char t[32];
        std::snprintf(t, sizeof t, "  gpu %.2f ms", cost_.gpu_ms);
        s += t;
    }
    if (!notes_.empty()) s += "  " + std::to_string(notes_.size()) + (notes_.size() == 1 ? " note" : " notes");
    return s;
}

namespace {

std::string camera_words(const ViewCamera& camera) {
    char t[256];
    if (const Orbit* o = std::get_if<Orbit>(&camera)) {
        std::snprintf(t, sizeof t, "orbit target %.9g %.9g %.9g distance %.9g azimuth %.9g elevation %.9g up %s fovy %.9g",
                      o->target[0], o->target[1], o->target[2], o->distance, o->azimuth, o->elevation, axis_name(o->up),
                      kOrbitFovY);
    } else if (const PanZoom* p = std::get_if<PanZoom>(&camera)) {
        std::snprintf(t, sizeof t, "panzoom center %.9g %.9g half_height %.9g y_down %d", p->center[0], p->center[1],
                      p->half_height, p->y_down ? 1 : 0);
    } else {
        std::snprintf(t, sizeof t, "pinhole");
    }
    return t;
}

}  // namespace

std::string SceneView::drawn_camera_text() const { return camera_words(drawn_camera_); }

std::string SceneView::camera_text() const {
    std::string out = camera_words(camera_) + " target_px " + std::to_string(width_) + " " + std::to_string(height_);
    if (!through_.empty() && std::holds_alternative<ThroughPinhole>(effective_camera())) out += " through " + through_;
    return out;
}

}  // namespace caliper_live_views
