// The entities of a scene panel (spec_entities.h, spec_prepare.h): their arrays checked against
// their slots, mapped into the panel's data box, built into the parts of the scene view (points,
// lines, and a surface as a mesh of its grid), and the pick fields of a hit on them
// (design/instrument-interface.md, Scene Entities, Array Resolution, Pick Fields and Data Box).
//
// Decision: the surface draw kind builds its triangles and normals on the CPU in the app and
// draws through the mesh part, because the mesh program already draws colormapped, lit and
// picked triangles; a vertex stage that reads the heights of the grid directly is later work.
#include "spec_runtime/spec_entities.h"
#include "spec_runtime/spec_prepare.h"
#include "spec_runtime/spec_runtime.h"   // house_spec_look

#include "live/live_scene.h"
#include "net/wire.h"
#include "scene/camera.h"
#include "scene/scene.h"

#include <caliper_builder/spec/colormaps.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <unordered_map>

namespace caliper_host {

namespace spec = caliper_builder::spec;
using caliper_builder::EntityData;
using caliper_builder::Error;
using caliper_builder::PanelData;
using caliper_builder::PickFields;
using caliper_builder::SlotData;

// ---- The data box ---------------------------------------------------------------------------------

double EntityBox::to_box(int a, double v) const {
    const double span = hi[a] - lo[a];
    if (!(span > 0)) return 0;
    return ((v - lo[a]) / span * 2.0 - 1.0) * half[a];
}

double EntityBox::to_data(int a, double b) const {
    if (!(half[a] > 0)) return lo[a];
    return lo[a] + (b / half[a] + 1.0) * 0.5 * (hi[a] - lo[a]);
}

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// The numbers of one slot, in row-major order, and the shape they came in.
struct Numbers {
    std::vector<double> v;
    std::vector<int64_t> shape;
    bool present = false;
};

Numbers numbers_of(const SlotData& s) {
    Numbers n;
    n.present = true;
    if (s.kind == SlotData::Kind::tensor) {
        const uint64_t count = s.tensor.numel();
        n.v.resize(count);
        for (uint64_t i = 0; i < count; ++i) n.v[i] = s.tensor.element(i);
        n.shape = s.tensor.shape();
    } else {
        for (const auto& cell : s.values) n.v.push_back(cell.is_number() ? cell.get<double>() : kNaN);
        n.shape = {int64_t(n.v.size())};
    }
    return n;
}

const SlotData* find_slot(const EntityData& d, const std::string& name) {
    for (const SlotData& s : d.slots)
        if (s.slot == name) return &s;
    return nullptr;
}

// The finite range of some numbers, or false.
bool range_of(const std::vector<double>& v, double* lo, double* hi) {
    bool any = false;
    for (double x : v)
        if (std::isfinite(x)) {
            if (!any) *lo = *hi = x;
            *lo = std::min(*lo, x);
            *hi = std::max(*hi, x);
            any = true;
        }
    return any;
}

// The literal of an option that may be a reference: the literal, or the reference's default.
std::optional<double> literal_number(const std::optional<spec::ValueOr<double>>& v) {
    if (!v) return std::nullopt;
    if (const double* d = std::get_if<double>(&*v)) return *d;
    const spec::Reference& r = std::get<spec::Reference>(*v);
    if (r.fallback.is_number()) return r.fallback.get<double>();
    return std::nullopt;
}

std::optional<std::string> literal_text(const std::optional<spec::ValueOr<std::string>>& v) {
    if (!v) return std::nullopt;
    if (const std::string* s = std::get_if<std::string>(&*v)) return *s;
    const spec::Reference& r = std::get<spec::Reference>(*v);
    if (r.fallback.is_string()) return r.fallback.get<std::string>();
    return std::nullopt;
}

bool literal_bool(const std::optional<spec::ValueOr<bool>>& v, bool fallback) {
    if (!v) return fallback;
    if (const bool* b = std::get_if<bool>(&*v)) return *b;
    const spec::Reference& r = std::get<spec::Reference>(*v);
    return r.fallback.is_boolean() ? r.fallback.get<bool>() : fallback;
}

int axis_index(const std::string& a) { return a == "x" ? 0 : a == "y" ? 1 : 2; }

// One entity's arrays in data units, before the box maps them.
struct Staged {
    EntityDraw out;
    spec::EntityStyle style;
    std::vector<double> axis[3];       // each item's data value on each axis; empty: from a place
    std::optional<spec::AxisPlace> place[3];
    bool own[3] = {false, false, false};
    std::vector<double> height;        // surface (R * C)
};

Error shape_error(const std::string& path, const std::string& what) {
    return Error("entity.shape", "The entity " + path + " has the wrong shape: " + what + ".");
}

// Reads the slots of a points or lines entity. False with the entity's error.
bool stage_items(Staged& st, const EntityData& d) {
    EntityDraw& o = st.out;
    Numbers pos, ax[3], vals, cols, radii;
    if (const SlotData* s = find_slot(d, "positions")) pos = numbers_of(*s);
    static const char* const names[3] = {"x", "y", "z"};
    for (int a = 0; a < 3; ++a)
        if (const SlotData* s = find_slot(d, names[a])) ax[a] = numbers_of(*s);
    if (const SlotData* s = find_slot(d, "values")) vals = numbers_of(*s);
    if (const SlotData* s = find_slot(d, "colors")) cols = numbers_of(*s);
    if (const SlotData* s = find_slot(d, "radii")) radii = numbers_of(*s);
    int64_t n = -1;
    auto length = [&](const Numbers& x, int64_t per, const char* name) {
        if (!x.present) return true;
        const int64_t items = per > 0 ? int64_t(x.v.size()) / per : int64_t(x.v.size());
        if (n < 0) n = items;
        if (items != n) {
            o.error = Error("entity.length", "The slots of the entity " + o.entity + " have different lengths: " +
                                                 name + " has " + std::to_string(items) + " items, others " +
                                                 std::to_string(n) + ".");
            return false;
        }
        return true;
    };
    if (pos.present) {
        if (pos.shape.size() != 2 || (pos.shape[1] != 3 && pos.shape[1] != 2)) {
            o.error = shape_error(o.entity, "positions is (N, 3)");
            return false;
        }
        const int64_t dims = pos.shape[1];
        if (!length(pos, dims, "positions")) return false;
        for (int a = 0; a < 3; ++a) st.axis[a].resize(size_t(n), 0.0);
        for (int64_t i = 0; i < n; ++i)
            for (int a = 0; a < dims; ++a) st.axis[a][size_t(i)] = pos.v[size_t(i * dims + a)];
        if (dims == 2) st.axis[2].clear();
    }
    for (int a = 0; a < 3; ++a) {
        if (!ax[a].present) continue;
        if (!length(ax[a], 0, names[a])) return false;
        st.axis[a] = ax[a].v;
    }
    if (!length(vals, 0, "values") || !length(cols, 4, "colors") || !length(radii, 0, "radii")) return false;
    if (n < 0) {
        o.error = shape_error(o.entity, "it has no positions");
        return false;
    }
    if (n > spec::limits::kEntityItems) {
        o.error = Error("entity.limit", "The entity " + o.entity + " has " + std::to_string(n) +
                                            " items, over the limit of " + std::to_string(spec::limits::kEntityItems) +
                                            ".");
        return false;
    }
    for (int a = 0; a < 3; ++a)
        if (st.axis[a].empty() && !st.place[a]) st.place[a] = spec::AxisPlace{spec::AxisPlace::Kind::min, 0};
    if (vals.present) o.values.assign(vals.v.begin(), vals.v.end());
    if (cols.present) {
        o.colors.resize(size_t(n) * 4);
        for (size_t i = 0; i < o.colors.size(); ++i) o.colors[i] = uint8_t(std::clamp(cols.v[i], 0.0, 255.0));
    }
    if (radii.present) o.radii.assign(radii.v.begin(), radii.v.end());
    // The step of each item after a window: the steps of the x slot, else of the positions.
    for (const char* name : {"x", "positions"})
        if (const SlotData* s = find_slot(d, name); s && s->steps.numel() == uint64_t(n)) {
            for (uint64_t i = 0; i < s->steps.numel(); ++i) o.steps.push_back(int64_t(s->steps.element(i)));
            break;
        }
    return true;
}

// Reads the slots of a surface entity. False with the entity's error.
bool stage_surface(Staged& st, const EntityData& d) {
    EntityDraw& o = st.out;
    const SlotData* hs = find_slot(d, "height");
    if (!hs) {
        o.error = shape_error(o.entity, "a surface needs height (R, C)");
        return false;
    }
    const Numbers h = numbers_of(*hs);
    if (h.shape.size() != 2) {
        o.error = shape_error(o.entity, "height is (R, C)");
        return false;
    }
    const int64_t R = h.shape[0], C = h.shape[1];
    if (R < 1 || C < 1 || R > spec::limits::kSurfaceSide || C > spec::limits::kSurfaceSide ||
        R * C > spec::limits::kSurfaceCells) {
        o.error = Error("entity.limit", "The surface " + o.entity + " has " + std::to_string(R) + " x " +
                                            std::to_string(C) + " cells, over the limit of 262,144 and 4,096 a side.");
        return false;
    }
    o.rows = R;
    o.cols = C;
    st.height = h.v;
    Numbers xs, ys, vals;
    if (const SlotData* s = find_slot(d, "x")) xs = numbers_of(*s);
    if (const SlotData* s = find_slot(d, "y")) ys = numbers_of(*s);
    if (const SlotData* s = find_slot(d, "values")) vals = numbers_of(*s);
    if ((xs.present && int64_t(xs.v.size()) != R) || (ys.present && int64_t(ys.v.size()) != C) ||
        (vals.present && int64_t(vals.v.size()) != R * C)) {
        o.error = Error("entity.length", "The slots of the surface " + o.entity +
                                             " do not fit its height: x has R items, y has C and values R x C.");
        return false;
    }
    o.row_coord.resize(size_t(R));
    o.col_coord.resize(size_t(C));
    for (int64_t r = 0; r < R; ++r) o.row_coord[size_t(r)] = xs.present ? xs.v[size_t(r)] : double(r);
    for (int64_t c = 0; c < C; ++c) o.col_coord[size_t(c)] = ys.present ? ys.v[size_t(c)] : double(c);
    o.row_from_x = xs.present;
    o.column_from_y = ys.present;
    o.values = vals.present ? vals.v : h.v;
    // The step of each row after a window: the steps of the height's rows.
    if (hs->steps.numel() == uint64_t(R))
        for (uint64_t i = 0; i < hs->steps.numel(); ++i) o.steps.push_back(int64_t(hs->steps.element(i)));
    st.axis[0].clear();
    st.axis[1].clear();
    st.axis[2].clear();
    return true;
}

void normal_at(const std::vector<float>& p, int64_t R, int64_t C, int64_t r, int64_t c, float* n) {
    auto at = [&](int64_t rr, int64_t cc, int k) { return double(p[size_t((rr * C + cc) * 3 + k)]); };
    auto ok = [&](int64_t rr, int64_t cc) { return std::isfinite(at(rr, cc, 2)); };
    const int64_t r0 = (r > 0 && ok(r - 1, c)) ? r - 1 : r, r1 = (r + 1 < R && ok(r + 1, c)) ? r + 1 : r;
    const int64_t c0 = (c > 0 && ok(r, c - 1)) ? c - 1 : c, c1 = (c + 1 < C && ok(r, c + 1)) ? c + 1 : c;
    double tr[3] = {0, 0, 0}, tc[3] = {0, 0, 0};
    for (int k = 0; k < 3; ++k) {
        tr[k] = at(r1, c, k) - at(r0, c, k);
        tc[k] = at(r, c1, k) - at(r, c0, k);
    }
    if (r0 == r1) tr[0] = 1;   // a single row or a hole beside it: flat along the rows
    if (c0 == c1) tc[1] = 1;
    double x = tr[1] * tc[2] - tr[2] * tc[1], y = tr[2] * tc[0] - tr[0] * tc[2], z = tr[0] * tc[1] - tr[1] * tc[0];
    if (z < 0) x = -x, y = -y, z = -z;
    const double l = std::sqrt(x * x + y * y + z * z);
    if (!(l > 0) || !std::isfinite(l)) x = 0, y = 0, z = 1;
    else x /= l, y /= l, z /= l;
    n[0] = float(x);
    n[1] = float(y);
    n[2] = float(z);
}

}  // namespace

// ---- Building --------------------------------------------------------------------------------------

EntityScene build_entity_scene(const spec::Panel& panel, const PanelData& data) {
    EntityScene scene;
    scene.up = panel.options.scene.up.value_or("+z");
    scene.walls = panel.options.scene.axes.value_or(true);
    std::vector<Staged> staged;
    for (const spec::Entity& e : panel.entities) {
        Staged st;
        st.out.entity = e.path;
        st.out.path = std::string(spec::kSpecEntityRoot) + e.path;
        st.out.draw = e.draw;
        st.style = spec::entity_style(e);
        const spec::EntityStyle& sy = st.style;
        st.out.visible = literal_bool(sy.visible, true);
        st.out.depth_offset = int(sy.depth_offset.value_or(e.draw == spec::DrawKind::surface ? 0 : 1));
        st.out.size_px = float(sy.size_px.value_or(0.0));
        st.out.width_px = float(sy.width_px.value_or(2.0));
        st.out.square = sy.shape.value_or("disc") == "square";
        st.out.strip = sy.mode.value_or("strip") == "strip";
        if (auto name = literal_text(sy.colormap); name && spec::colormap_id(*name))
            st.out.colormap = *spec::colormap_id(*name);
        st.place[0] = sy.x;
        st.place[1] = sy.y;
        st.place[2] = sy.z;
        for (const std::string& a : sy.own) st.own[axis_index(a)] = true;
        const EntityData* d = nullptr;
        for (const EntityData& x : data.entities)
            if (x.path == e.path) d = &x;
        if (!d) {
            st.out.error = Error("entity.no_data", "The entity " + e.path + " has no data yet.");
        } else if (d->error) {
            st.out.error = d->error;
        } else {
            st.out.generation = d->generation;
            if (e.draw == spec::DrawKind::surface)
                stage_surface(st, *d);
            else
                stage_items(st, *d);
        }
        staged.push_back(std::move(st));
    }

    // The shared range of each axis, over the entities that do not map it by their own range.
    bool any[3] = {false, false, false};
    double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
    auto widen = [&](int a, const std::vector<double>& v) {
        double l, h;
        if (!range_of(v, &l, &h)) return;
        if (!any[a]) lo[a] = l, hi[a] = h;
        lo[a] = std::min(lo[a], l);
        hi[a] = std::max(hi[a], h);
        any[a] = true;
    };
    for (Staged& st : staged) {
        if (st.out.error) continue;
        if (st.out.draw == spec::DrawKind::surface) {
            widen(0, st.out.row_coord);
            widen(1, st.out.col_coord);
            widen(2, st.height);
            continue;
        }
        for (int a = 0; a < 3; ++a)
            if (!st.own[a]) widen(a, st.axis[a]);
    }
    for (int a = 0; a < 3; ++a) {
        if (!any[a]) lo[a] = 0, hi[a] = 1;
        if (hi[a] == lo[a]) lo[a] -= 0.5, hi[a] += 0.5;
    }
    // The box: the option's proportions, or the data's with no side below a quarter of the longest.
    double side[3];
    const nlohmann::json& bo = panel.options.box;
    if (bo.is_array() && bo.size() == 3) {
        for (int a = 0; a < 3; ++a) side[a] = bo[size_t(a)].is_number() ? bo[size_t(a)].get<double>() : 1.0;
    } else {
        for (int a = 0; a < 3; ++a) side[a] = hi[a] - lo[a];
    }
    double longest = std::max({side[0], side[1], side[2]});
    if (!(longest > 0) || !std::isfinite(longest)) longest = 1, side[0] = side[1] = side[2] = 1;
    for (int a = 0; a < 3; ++a) {
        double s = side[a] / longest;
        if (!bo.is_array()) s = std::max(s, 0.25);
        if (!(s > 0) || !std::isfinite(s)) s = 0.25;
        scene.box.half[a] = s;   // the longest side is 2 units
        scene.box.lo[a] = lo[a];
        scene.box.hi[a] = hi[a];
    }
    const EntityBox& box = scene.box;

    for (Staged& st : staged) {
        EntityDraw& o = st.out;
        if (!o.error && o.draw == spec::DrawKind::surface) {
            const int64_t R = o.rows, C = o.cols;
            o.positions.resize(size_t(R * C * 3));
            for (int a = 0; a < 3; ++a) o.data[a].resize(size_t(R * C));
            for (int64_t r = 0; r < R; ++r)
                for (int64_t c = 0; c < C; ++c) {
                    const size_t i = size_t(r * C + c);
                    const double h = st.height[i];
                    o.data[0][i] = o.row_coord[size_t(r)];
                    o.data[1][i] = o.col_coord[size_t(c)];
                    o.data[2][i] = h;
                    o.positions[i * 3 + 0] = float(box.to_box(0, o.row_coord[size_t(r)]));
                    o.positions[i * 3 + 1] = float(box.to_box(1, o.col_coord[size_t(c)]));
                    o.positions[i * 3 + 2] = std::isfinite(h) ? float(box.to_box(2, h)) : NAN;
                }
            // Two triangles a cell, for each cell whose four heights are numbers: NaN is a hole.
            for (int64_t r = 0; r + 1 < R; ++r)
                for (int64_t c = 0; c + 1 < C; ++c) {
                    const int32_t a = int32_t(r * C + c), b = a + 1, d = int32_t(a + C), e = d + 1;
                    if (!std::isfinite(o.positions[size_t(a) * 3 + 2]) || !std::isfinite(o.positions[size_t(b) * 3 + 2]) ||
                        !std::isfinite(o.positions[size_t(d) * 3 + 2]) || !std::isfinite(o.positions[size_t(e) * 3 + 2]))
                        continue;
                    o.indices.insert(o.indices.end(), {a, d, b, b, d, e});
                }
            o.normals.resize(o.positions.size());
            for (int64_t r = 0; r < R; ++r)
                for (int64_t c = 0; c < C; ++c)
                    if (std::isfinite(o.positions[size_t((r * C + c) * 3 + 2)]))
                        normal_at(o.positions, R, C, r, c, &o.normals[size_t((r * C + c) * 3)]);
                    else
                        o.normals[size_t((r * C + c) * 3 + 2)] = 1.0f;
            // A hole's vertices sit on the floor, out of every drawn triangle, so the bounds hold.
            for (size_t i = 2; i < o.positions.size(); i += 3)
                if (!std::isfinite(o.positions[i])) o.positions[i] = float(-box.half[2]);
        } else if (!o.error) {
            const size_t n = std::max({st.axis[0].size(), st.axis[1].size(), st.axis[2].size()});
            o.positions.resize(n * 3);
            for (int a = 0; a < 3; ++a) {
                o.data[a].resize(n);
                double own_lo = 0, own_hi = 1;
                const bool own = st.own[a] && !st.axis[a].empty() && range_of(st.axis[a], &own_lo, &own_hi);
                if (own && own_hi == own_lo) own_lo -= 0.5, own_hi += 0.5;
                for (size_t i = 0; i < n; ++i) {
                    double dv, bv;
                    if (!st.axis[a].empty()) {
                        dv = st.axis[a][i];
                        bv = own ? ((dv - own_lo) / (own_hi - own_lo) * 2.0 - 1.0) * box.half[a] : box.to_box(a, dv);
                    } else {
                        const spec::AxisPlace p = st.place[a].value_or(spec::AxisPlace{spec::AxisPlace::Kind::min, 0});
                        bv = p.kind == spec::AxisPlace::Kind::min   ? -box.half[a]
                             : p.kind == spec::AxisPlace::Kind::max ? box.half[a]
                                                                    : box.to_box(a, p.value);
                        dv = box.to_data(a, bv);
                    }
                    o.data[a][i] = dv;
                    o.positions[i * 3 + size_t(a)] = float(bv);
                }
            }
        }
        if (!o.error && !o.values.empty()) {
            double l = 0, h = 1;
            if (!range_of(o.values, &l, &h)) l = 0, h = 1;
            if (l == h) l -= 0.5, h += 0.5;
            o.vmin = literal_number(st.style.vmin).value_or(l);
            o.vmax = literal_number(st.style.vmax).value_or(h);
        }
        scene.draws.push_back(std::move(o));
    }
    return scene;
}

// ---- Picks ----------------------------------------------------------------------------------------

std::optional<PickFields> entity_pick_fields(const EntityScene& scene, const std::string& path, uint64_t item) {
    for (const EntityDraw& d : scene.draws) {
        if (d.path != path) continue;
        PickFields f;
        f.entity = d.entity;
        const uint64_t n = d.data[0].size();
        if (d.error || item >= n) return f;
        f.position = std::array<double, 3>{d.data[0][item], d.data[1][item], d.data[2][item]};
        if (item < d.values.size() && std::isfinite(d.values[item])) f.value = d.values[item];
        if (d.draw == spec::DrawKind::surface) {
            const int64_t r = int64_t(item) / d.cols, c = int64_t(item) % d.cols;
            f.row = d.row_from_x && d.steps.empty() ? int64_t(std::llround(d.row_coord[size_t(r)])) : r;
            f.column = d.column_from_y ? int64_t(std::llround(d.col_coord[size_t(c)])) : c;
            if (size_t(r) < d.steps.size()) f.step = d.steps[size_t(r)];
        } else {
            f.index = int64_t(item);
            if (item < d.steps.size()) f.step = d.steps[item];
        }
        return f;
    }
    return std::nullopt;
}

// ---- The starting camera --------------------------------------------------------------------------

caliper_scene::StartingCamera spec_starting_camera(const spec::Panel& panel) {
    caliper_scene::StartingCamera s;
    const nlohmann::json& v = panel.options.scene.view_camera;
    auto orbit = [&](const nlohmann::json& o) {
        s.form = caliper_scene::StartingCamera::Form::Orbit;
        if (auto t = o.find("target"); t != o.end() && t->is_array() && t->size() == 3)
            s.target = caliper_scene::Vec3{(*t)[0].get<double>(), (*t)[1].get<double>(), (*t)[2].get<double>()};
        if (auto d = o.find("distance"); d != o.end() && d->is_number() && d->get<double>() > 0)
            s.distance = d->get<double>();
        if (auto a = o.find("azimuth"); a != o.end() && a->is_number()) s.azimuth_deg = a->get<double>();
        if (auto e = o.find("elevation"); e != o.end() && e->is_number()) s.elevation_deg = e->get<double>();
    };
    if (v.is_object() && !v.empty()) {
        orbit(v);
    } else if (v.is_string()) {
        const std::string name = v.get<std::string>();
        for (const spec::SavedView& sv : panel.options.views)
            if (sv.name == name) {
                s.form = caliper_scene::StartingCamera::Form::Orbit;
                s.target = caliper_scene::Vec3{sv.target[0], sv.target[1], sv.target[2]};
                s.distance = sv.distance;
                s.azimuth_deg = sv.azimuth;
                s.elevation_deg = sv.elevation;
                return s;
            }
        s.form = caliper_scene::StartingCamera::Form::Entity;
        s.entity = name;
    }
    return s;
}

// ---- Into a scene view ------------------------------------------------------------------------------

namespace {

namespace wire = caliper_net::wire;

template <class T>
wire::Section section(const char* name, int64_t dtype, const std::vector<T>& v, std::vector<int64_t> shape) {
    wire::Section s;
    s.name = name;
    s.dtype = dtype;
    s.shape = std::move(shape);
    s.data = reinterpret_cast<const uint8_t*>(v.data());
    s.size = v.size() * sizeof(T);
    return s;
}

// Round numbers in [lo, hi]: steps of 1, 2 or 5 times a power of ten, about `n` of them.
std::vector<double> round_ticks(double lo, double hi, int n) {
    std::vector<double> out;
    const double span = hi - lo;
    if (!(span > 0) || !std::isfinite(span)) return out;
    const double raw = span / n, p = std::pow(10.0, std::floor(std::log10(raw)));
    const double step = raw / p < 1.5 ? p : raw / p < 3.5 ? 2 * p : raw / p < 7.5 ? 5 * p : 10 * p;
    for (double t = std::ceil(lo / step) * step; t <= hi + step * 1e-9 && out.size() < 32; t += step) out.push_back(t);
    return out;
}

// The walls of the box away from the eye of the starting camera, each with its outline and grid
// lines at round numbers of the data, as segments: (V, 3) with V even.
std::vector<float> wall_segments(const EntityBox& box, caliper_scene::Axis up_axis, const caliper_scene::StartingCamera& start) {
    caliper_scene::Orbit o;
    o.up = up_axis;
    if (start.azimuth_deg) o.azimuth = *start.azimuth_deg * caliper_scene::kPi / 180.0;
    if (start.elevation_deg) o.elevation = *start.elevation_deg * caliper_scene::kPi / 180.0;
    const caliper_scene::Vec3 e = caliper_scene::orbit_direction(o);
    const double ev[3] = {e.x, e.y, e.z};
    std::vector<float> v;
    auto seg = [&](const double a[3], const double b[3]) {
        for (int k = 0; k < 3; ++k) v.push_back(float(a[k]));
        for (int k = 0; k < 3; ++k) v.push_back(float(b[k]));
    };
    for (int w = 0; w < 3; ++w) {
        // The wall across axis w on the side away from the eye; it spans the other two axes.
        const double at = ev[w] > 0 ? -box.half[w] : box.half[w];
        const int p = (w + 1) % 3, q = (w + 2) % 3;
        for (int side = 0; side < 2; ++side)
            for (int dir = 0; dir < 2; ++dir) {
                const int along = dir == 0 ? p : q, across = dir == 0 ? q : p;
                double a[3], b[3];
                a[w] = b[w] = at;
                a[across] = b[across] = side == 0 ? -box.half[across] : box.half[across];
                a[along] = -box.half[along];
                b[along] = box.half[along];
                seg(a, b);
            }
        for (int dir = 0; dir < 2; ++dir) {
            const int along = dir == 0 ? p : q, across = dir == 0 ? q : p;
            for (double t : round_ticks(box.lo[across], box.hi[across], 5)) {
                const double c = box.to_box(across, t);
                if (std::fabs(std::fabs(c) - box.half[across]) < 1e-9) continue;   // on the outline
                double a[3], b[3];
                a[w] = b[w] = at;
                a[across] = b[across] = c;
                a[along] = -box.half[along];
                b[along] = box.half[along];
                seg(a, b);
            }
        }
    }
    return v;
}

}  // namespace

void apply_entity_scene(caliper_live_views::SceneView& view, const EntityScene& scene,
                        std::vector<std::string>* drawn) {
    if (view.tree().space() != caliper_scene::Space::k3D || caliper_scene::axis_name(view.tree().up()) != scene.up) {
        wire::SceneView info;
        info.space = CALIPER_NET_SPACE_3D;
        info.up = scene.up;
        const caliper_scene::StartingCamera keep = view.starting();
        view.describe(info);
        view.set_starting(keep);   // the panel's starting camera, not the VIEWS entry's
    }
    view.begin_round();
    std::vector<std::string> now;
    const EntityBox& box = scene.box;
    for (const EntityDraw& d : scene.draws) {
        if (d.error || !d.visible || d.positions.empty()) continue;
        const int64_t n = int64_t(d.positions.size() / 3);
        std::vector<float> values(d.values.begin(), d.values.end());
        wire::SceneUpdate u;
        u.path = d.path;
        u.params = caliper_net::Json::object();
        u.arrays.push_back(section("positions", CALIPER_NET_DTYPE_F32, d.positions, {n, 3}));
        if (!d.colors.empty())
            u.arrays.push_back(section("colors", CALIPER_NET_DTYPE_U8, d.colors, {n, 4}));
        else if (!values.empty())
            u.arrays.push_back(section("values", CALIPER_NET_DTYPE_F32, values, {n}));
        if (!values.empty() && d.colors.empty()) {
            u.params.set("colormap", int64_t(d.colormap));
            u.params.set("vmin", d.vmin);
            u.params.set("vmax", d.vmax);
        }
        if (d.flat_rgba != 0) {
            caliper_net::Json c = caliper_net::Json::array();
            for (int k = 0; k < 4; ++k) c.push(int64_t((d.flat_rgba >> (8 * k)) & 0xffu));
            u.params.set("color", c);
        }
        // The bounds of every part are the data box, so the fit frames the whole box.
        caliper_net::Json stats = caliper_net::Json::array();
        stats.push(d.vmin);
        stats.push(d.vmax);
        for (int a = 0; a < 3; ++a) {
            stats.push(-box.half[a]);
            stats.push(box.half[a]);
        }
        u.params.set("stats", stats);
        std::vector<int32_t> strips;
        if (d.draw == spec::DrawKind::points) {
            u.part = CALIPER_NET_PART_POINTS;
            if (!d.radii.empty()) u.arrays.push_back(section("radii", CALIPER_NET_DTYPE_F32, d.radii, {n}));
        } else if (d.draw == spec::DrawKind::lines) {
            u.part = CALIPER_NET_PART_LINES;
            u.params.set("width", double(d.width_px));
            if (d.strip) {
                strips.push_back(int32_t(n));
                u.params.set("mode", CALIPER_NET_LINES_STRIPS);
                u.arrays.push_back(section("strip_lengths", CALIPER_NET_DTYPE_I32, strips, {1}));
            } else {
                u.params.set("mode", CALIPER_NET_LINES_SEGMENTS);
            }
        } else {
            if (d.indices.empty()) continue;   // every cell a hole
            u.part = CALIPER_NET_PART_MESH;
            u.arrays.insert(u.arrays.begin() + 1,
                            section("indices", CALIPER_NET_DTYPE_I32, d.indices, {int64_t(d.indices.size() / 3), 3}));
            u.arrays.push_back(section("normals", CALIPER_NET_DTYPE_F32, d.normals, {n, 3}));
        }
        std::string why;
        if (!view.apply(u, &why)) {
            view.note(d.entity + ": " + why);
            continue;
        }
        caliper_live_views::SceneView::BatchStyle style;
        style.depth_offset = d.depth_offset;
        style.size_px = d.size_px;
        if (d.draw == spec::DrawKind::points)
            style.point_shape = d.square ? caliper_scene::PointShape::Square : caliper_scene::PointShape::Disc;
        view.set_style(d.path, style);
        now.push_back(d.path);
    }
    if (scene.walls) {
        const std::vector<float> walls = wall_segments(box, view.tree().up(), view.starting());
        wire::SceneUpdate u;
        u.path = "spec-box/walls";
        u.part = CALIPER_NET_PART_LINES;
        u.params = caliper_net::Json::object();
        u.params.set("mode", CALIPER_NET_LINES_SEGMENTS);
        u.params.set("width", 1.0);
        caliper_net::Json c = caliper_net::Json::array();
        for (int k : {96, 96, 96, 255}) c.push(int64_t(k));
        u.params.set("color", c);
        u.arrays.push_back(section("positions", CALIPER_NET_DTYPE_F32, walls, {int64_t(walls.size() / 3), 3}));
        std::string why;
        if (view.apply(u, &why)) now.push_back(u.path);
    }
    for (const std::string& old : *drawn)
        if (std::find(now.begin(), now.end(), old) == now.end()) {
            wire::SceneClear c;
            c.path = old;
            c.recursive = true;
            std::string why;
            view.apply(c, &why);
        }
    *drawn = std::move(now);
}

void apply_annotation_marks(caliper_live_views::SceneView& view, const EntityScene& scene,
                            const std::vector<SceneAnnotationMark>& marks, std::vector<std::string>* drawn) {
    const EntityBox& box = scene.box;
    const SpecLook look = house_spec_look();
    std::vector<std::string> now;
    auto color_of = [&](bool muted, bool hole = false) {
        const ImU32 c = hole ? look.ground : muted ? look.secondary : look.accent;
        caliper_net::Json out = caliper_net::Json::array();
        for (int k = 0; k < 4; ++k) out.push(int64_t((c >> (8 * k)) & 0xffu));
        return out;
    };
    auto stats_of = [&] {
        caliper_net::Json stats = caliper_net::Json::array();
        stats.push(0.0);
        stats.push(1.0);
        for (int a = 0; a < 3; ++a) {
            stats.push(-box.half[a]);
            stats.push(box.half[a]);
        }
        return stats;
    };
    // One point with its label; `size_px` tells a marker from a label's dot, and a hole is the
    // dark centre that makes a marker's disc a ring.
    auto point = [&](const std::string& path, const float p[3], const std::string& text, bool muted, float size_px,
                     bool hole = false) {
        const std::vector<float> positions(p, p + 3);
        wire::SceneUpdate u;
        u.path = path;
        u.part = CALIPER_NET_PART_POINTS;
        u.params = caliper_net::Json::object();
        u.params.set("color", color_of(muted, hole));
        u.params.set("stats", stats_of());
        if (!text.empty()) {
            caliper_net::Json labels = caliper_net::Json::array();
            labels.push(text);
            u.params.set("labels", labels);
        }
        u.arrays.push_back(section("positions", CALIPER_NET_DTYPE_F32, positions, {1, 3}));
        std::string why;
        if (!view.apply(u, &why)) return;
        caliper_live_views::SceneView::BatchStyle style;
        style.depth_offset = hole ? 5 : 4;   // in front of the surface it marks
        style.size_px = size_px;
        style.point_shape = caliper_scene::PointShape::Disc;
        view.set_style(path, style);
        now.push_back(path);
    };
    for (const SceneAnnotationMark& m : marks) {
        const std::string path = "spec-overlay/" + m.id;
        if (m.kind == spec::AnnotationKind::label || m.kind == spec::AnnotationKind::marker) {
            float p[3] = {0, 0, 0};
            if (!m.entity.empty()) {
                const EntityDraw* d = nullptr;
                for (const EntityDraw& e : scene.draws)
                    if (e.entity == m.entity) d = &e;
                if (!d || m.index < 0 || uint64_t(m.index) * 3 + 2 >= d->positions.size()) continue;
                for (int a = 0; a < 3; ++a) p[a] = d->positions[size_t(m.index) * 3 + size_t(a)];
            } else {
                for (int a = 0; a < 3; ++a) p[a] = float(box.to_box(a, m.at[a]));
            }
            if (m.kind == spec::AnnotationKind::marker) {
                point(path, p, m.text, m.muted, 16.0f);
                point(path + "/hole", p, std::string(), m.muted, 8.0f, true);
            } else {
                point(path, p, m.text, m.muted, 6.0f);
            }
            continue;
        }
        // A threshold's plane, or a band's two planes, as outlines across the box.
        const int a = std::clamp(m.axis, 0, 2), b = (a + 1) % 3, c = (a + 2) % 3;
        std::vector<float> segments;
        for (double v : m.kind == spec::AnnotationKind::band ? std::vector<double>{m.lo, m.hi} : std::vector<double>{m.lo}) {
            const float at = float(box.to_box(a, v));
            const float corners[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
            for (int k = 0; k < 4; ++k)
                for (int e : {k, (k + 1) % 4}) {
                    float q[3];
                    q[a] = at;
                    q[b] = float(corners[e][0] * box.half[b]);
                    q[c] = float(corners[e][1] * box.half[c]);
                    segments.insert(segments.end(), q, q + 3);
                }
        }
        wire::SceneUpdate u;
        u.path = path;
        u.part = CALIPER_NET_PART_LINES;
        u.params = caliper_net::Json::object();
        u.params.set("mode", CALIPER_NET_LINES_SEGMENTS);
        u.params.set("width", 2.0);
        u.params.set("color", color_of(m.muted));
        u.params.set("stats", stats_of());
        u.arrays.push_back(section("positions", CALIPER_NET_DTYPE_F32, segments, {int64_t(segments.size() / 3), 3}));
        std::string why;
        if (view.apply(u, &why)) now.push_back(path);
        if (!m.text.empty()) {
            float corner[3];
            corner[a] = float(box.to_box(a, m.hi));
            corner[b] = float(box.half[b]);
            corner[c] = float(box.half[c]);
            point(path + "/label", corner, m.text, m.muted, 4.0f);
        }
    }
    for (const std::string& old : *drawn)
        if (std::find(now.begin(), now.end(), old) == now.end()) {
            wire::SceneClear c;
            c.path = old;
            c.recursive = true;
            std::string why;
            view.apply(c, &why);
        }
    *drawn = std::move(now);
}

// ---- SpecScene -------------------------------------------------------------------------------------

namespace {

// The content of each SpecScene, by its address: the frozen class keeps its Impl private, so
// the runtime reads what it built through spec_scene_content(). Frame thread only.
std::unordered_map<const SpecScene*, const EntityScene*>& registry() {
    static std::unordered_map<const SpecScene*, const EntityScene*> r;
    return r;
}

}  // namespace

struct SpecScene::Impl {
    EntityScene scene;
    std::map<std::string, caliper_scene::Batch> overlays;   // "spec-overlay/<path>"
};

SpecScene::SpecScene() : impl_(std::make_unique<Impl>()) { registry()[this] = &impl_->scene; }
SpecScene::~SpecScene() { registry().erase(this); }

void SpecScene::set_entities(const spec::Panel& panel, const PanelData& data) {
    impl_->scene = build_entity_scene(panel, data);
}

void SpecScene::set_overlay_batch(const std::string& path, const caliper_scene::Batch& batch) {
    impl_->overlays[path] = batch;
}

void SpecScene::clear(const std::string& path) {
    impl_->overlays.erase(path);
    auto& d = impl_->scene.draws;
    d.erase(std::remove_if(d.begin(), d.end(), [&](const EntityDraw& e) { return e.path == path || e.entity == path; }),
            d.end());
}

std::optional<PickFields> SpecScene::pick_fields(const std::string& path, std::uint64_t item) const {
    return entity_pick_fields(impl_->scene, path, item);
}

const EntityScene& spec_scene_content(const SpecScene& scene) {
    static const EntityScene none;
    auto it = registry().find(&scene);
    return it == registry().end() ? none : *it->second;
}

}  // namespace caliper_host
