// The labels of a DrawRuntime (draw_runtime.h, design/programmable-graphics.md,
// Text): one label at each valid anchor of the version that the last frame
// showed, which the scene view projects and places with place_labels().
//
// The anchors come from the stream, thus each is checked where it is used:
// the anchors that the copy holds bound the valid count, a string index must
// name one of the table's strings, and a position or a priority that is not
// finite leaves the anchor out or its priority at 0. Of more than kLabels
// anchors, those of the highest priority stay, the earlier first between
// equals.
#include "draw/draw_runtime.h"
#include "draw/draw_runtime_impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace caliper_draw {

namespace {

namespace cbp = caliper_builder::programs;

const cbp::FieldLayout* field_of(const cbp::ElementLayout& l, const std::string& name) {
    for (const cbp::FieldLayout& f : l.fields)
        if (f.name == name) return &f;
    return nullptr;
}

// Whether field `f` has type `type` and lies inside an element of `stride` bytes.
bool fits(const cbp::FieldLayout* f, const char* type, std::uint32_t bytes, std::uint64_t stride) {
    return f && f->type == type && std::uint64_t(f->offset) + bytes <= stride;
}

float f32_at(const std::uint8_t* p) {
    float v;
    std::memcpy(&v, p, 4);
    return v;
}

}  // namespace

std::vector<Label> DrawRuntime::labels() const {
    std::vector<Label> out;
    const Impl& m = *impl_;
    if (!m.installed || !m.piece.draw || !m.piece.draw->text) return out;
    const caliper_builder::spec::TextTable& t = *m.piece.draw->text;
    const Impl::Slot* s = nullptr;
    for (const Impl::Slot& slot : m.slots)
        if (slot.id == t.anchors && !slot.texture) s = &slot;
    if (!s || !s->keep_cpu || !s->shown) return out;
    const std::uint64_t stride = s->layout.stride;
    const cbp::FieldLayout* pos = field_of(s->layout, t.position);
    const cbp::FieldLayout* str = field_of(s->layout, t.string);
    const cbp::FieldLayout* pri = t.priority ? field_of(s->layout, *t.priority) : nullptr;
    if (stride == 0 || !fits(pos, "vec4<f32>", 16, stride) || !fits(str, "u32", 4, stride)) return out;
    if (pri && !fits(pri, "f32", 4, stride)) pri = nullptr;
    const Impl::Shown& v = *s->shown;
    const std::uint64_t n = std::min<std::uint64_t>(v.count, v.bytes.size() / stride);
    const std::size_t strings = std::min(t.strings.size(), caliper_builder::spec::piece_limits::kStrings);
    for (std::uint64_t i = 0; i < n; ++i) {
        const std::uint8_t* e = v.bytes.data() + i * stride;
        std::uint32_t index;
        std::memcpy(&index, e + str->offset, 4);
        if (index >= strings) continue;
        Label l;
        bool finite = true;
        for (int k = 0; k < 3; ++k) {
            l.position[std::size_t(k)] = f32_at(e + pos->offset + 4 * k);
            finite = finite && std::isfinite(l.position[std::size_t(k)]);
        }
        if (!finite) continue;
        const std::string& text = t.strings[index];
        // The manifest's check holds a string to kStringBytes; a longer one is cut on a whole character.
        std::size_t cut = std::min(text.size(), caliper_builder::spec::piece_limits::kStringBytes);
        while (cut > 0 && cut < text.size() && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) --cut;
        l.text = text.substr(0, cut);
        if (pri) {
            const float p = f32_at(e + pri->offset);
            l.priority = std::isfinite(p) ? p : 0.0f;
        }
        out.push_back(std::move(l));
    }
    if (out.size() > caliper_builder::spec::piece_limits::kLabels) {
        std::stable_sort(out.begin(), out.end(), [](const Label& a, const Label& b) { return a.priority > b.priority; });
        out.resize(caliper_builder::spec::piece_limits::kLabels);
    }
    return out;
}

}  // namespace caliper_draw
