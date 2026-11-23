// The item map of a DrawRuntime (draw_runtime.h, design/programmable-graphics.md,
// Picking): the fields of a picked element from the CPU copy of the version
// that a frame showed, the last one or the one whose ID target a pick read.
//
// The item comes from a fragment of the piece and the bytes from the stream,
// thus each is checked where it is used: the item against the version's valid
// count and against the elements that the copy holds, and each field's offset
// and size against the element's stride. A field that the layout lacks, or a
// value that is not finite, is left out; a vector field gives one number for
// each component, named <field>.x, .y, .z and .w.
#include "draw/draw_runtime.h"
#include "draw/draw_runtime_impl.h"

#include <cmath>
#include <cstring>
#include <memory>

namespace caliper_draw {

namespace {

namespace cbp = caliper_builder::programs;

// The slot of a buffer resource or buffer input, or null.
const DrawRuntime::Impl::Slot* buffer_slot(const DrawRuntime::Impl& m, const std::string& id) {
    for (const DrawRuntime::Impl::Slot& s : m.slots)
        if (s.id == id && !s.texture) return &s;
    return nullptr;
}

const cbp::FieldLayout* field_of(const cbp::ElementLayout& l, const std::string& name) {
    for (const cbp::FieldLayout& f : l.fields)
        if (f.name == name) return &f;
    return nullptr;
}

// The scalar kind of a field type and its components: f32, i32 or u32 and 1, 2 or 4; false for a
// type that no layout holds.
bool scalar_of(const std::string& type, char* kind, int* n) {
    static const struct {
        const char* type;
        char kind;
        int n;
    } kTypes[] = {{"f32", 'f', 1},       {"i32", 'i', 1},       {"u32", 'u', 1},       {"vec2<f32>", 'f', 2},
                  {"vec4<f32>", 'f', 4}, {"vec2<u32>", 'u', 2}, {"vec4<u32>", 'u', 4}, {"vec2<i32>", 'i', 2},
                  {"vec4<i32>", 'i', 4}};
    for (const auto& t : kTypes)
        if (type == t.type) {
            *kind = t.kind;
            *n = t.n;
            return true;
        }
    return false;
}

// Component `i` of a field at `p` as a double.
double component(const std::uint8_t* p, char kind, int i) {
    if (kind == 'f') {
        float v;
        std::memcpy(&v, p + 4 * i, 4);
        return double(v);
    }
    if (kind == 'i') {
        std::int32_t v;
        std::memcpy(&v, p + 4 * i, 4);
        return double(v);
    }
    std::uint32_t v;
    std::memcpy(&v, p + 4 * i, 4);
    return double(v);
}

// What ShownItems holds: the program, its map and the element layout of the install that drew
// the frame, and the copy of the version that the frame showed.
struct ShownPick {
    std::string program;
    caliper_builder::spec::PickMap map;
    cbp::ElementLayout layout;
    std::shared_ptr<const DrawRuntime::Impl::Shown> shown;
};

}  // namespace

ShownItems DrawRuntime::shown_items() const {
    const Impl& m = *impl_;
    if (!m.installed || !m.piece.draw || !m.piece.draw->pick) return {};
    const Impl::Slot* s = buffer_slot(m, m.piece.draw->pick->resource);
    if (!s || !s->keep_cpu || !s->shown) return {};
    auto p = std::make_shared<ShownPick>();
    p->program = m.instance.program;
    p->map = *m.piece.draw->pick;
    p->layout = s->layout;
    p->shown = s->shown;
    return ShownItems{std::move(p)};
}

std::optional<PickedItem> DrawRuntime::pick(std::uint32_t item) const { return pick_shown(item, shown_items()); }

std::optional<PickedItem> DrawRuntime::pick_shown(std::uint32_t item, const ShownItems& shown) {
    if (!shown) return std::nullopt;
    const ShownPick& sp = *static_cast<const ShownPick*>(shown.state.get());
    const caliper_builder::spec::PickMap& map = sp.map;
    const Impl::Shown& v = *sp.shown;
    const cbp::ElementLayout& layout = sp.layout;
    const std::uint64_t stride = layout.stride;
    // The item against the valid count and the bytes the copy holds, without a product that wraps.
    if (stride == 0 || item >= v.count || std::uint64_t(item) >= v.bytes.size() / stride) return std::nullopt;
    const std::uint8_t* e = v.bytes.data() + std::uint64_t(item) * stride;
    PickedItem out;
    out.program = sp.program;
    out.item = item;
    out.step = v.step;
    std::size_t integers = 0;
    for (std::size_t k = 0; k < map.fields.size() && k < caliper_builder::spec::piece_limits::kPickFields; ++k) {
        const cbp::FieldLayout* f = field_of(layout, map.fields[k]);
        char kind = 0;
        int n = 0;
        if (!f || !scalar_of(f->type, &kind, &n)) continue;
        if (std::uint64_t(f->offset) + std::uint64_t(4) * std::uint64_t(n) > stride) continue;
        static const char* const kAxis[] = {".x", ".y", ".z", ".w"};
        for (int i = 0; i < n; ++i) {
            const double d = component(e + f->offset, kind, i);
            if (!std::isfinite(d)) continue;
            out.fields.emplace_back(n == 1 ? f->name : f->name + kAxis[i], d);
        }
        // The first four scalar integer fields are the item's tensor index.
        if (n == 1 && kind != 'f' && integers < out.index.size()) {
            const double d = component(e + f->offset, kind, 0);
            out.index[integers++] = d <= 2147483647.0 ? std::int32_t(d) : -1;
        }
    }
    return out;
}

}  // namespace caliper_draw
