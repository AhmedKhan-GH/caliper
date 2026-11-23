// The labels of a DrawRuntime (draw_runtime.h, design/programmable-graphics.md,
// Text): one label at each valid anchor of the version that the last frame
// showed, which the scene view projects and places with place_labels().
//
// The anchors come from the stream, thus each is checked where it is used:
// the anchors that the copy holds bound the valid count, a string index must
// name one of the table's strings, and a position or a priority that is not
// finite leaves the anchor out or its priority at 0. Of more than kLabels
// anchors, those of the highest priority stay, the earlier first between
// equals. With strings_from, the strings are the elements of the version of
// that resource that the same frame showed: at most kStrings, each its
// element's bytes up to the first zero byte, and a string that is not valid
// UTF-8 leaves its labels out.
#include "draw/draw_runtime.h"
#include "draw/draw_runtime_impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string_view>
#include <vector>

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

// Whether `s` is valid UTF-8: no stray continuation byte, no overlong form, no surrogate and
// nothing above U+10FFFF.
bool valid_utf8(std::string_view s) {
    std::size_t i = 0;
    while (i < s.size()) {
        const unsigned c = static_cast<unsigned char>(s[i]);
        std::size_t n = 0;
        unsigned lo = 0x80, hi = 0xBF;   // the range of the second byte
        if (c < 0x80) n = 0;
        else if (c >= 0xC2 && c <= 0xDF) n = 1;
        else if (c == 0xE0) n = 2, lo = 0xA0;
        else if ((c >= 0xE1 && c <= 0xEC) || c == 0xEE || c == 0xEF) n = 2;
        else if (c == 0xED) n = 2, hi = 0x9F;
        else if (c == 0xF0) n = 3, lo = 0x90;
        else if (c >= 0xF1 && c <= 0xF3) n = 3;
        else if (c == 0xF4) n = 3, hi = 0x8F;
        else return false;
        if (n > s.size() - i - 1) return false;
        for (std::size_t k = 1; k <= n; ++k) {
            const unsigned b = static_cast<unsigned char>(s[i + k]);
            if (k == 1 ? (b < lo || b > hi) : (b < 0x80 || b > 0xBF)) return false;
        }
        i += n + 1;
    }
    return true;
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
    namespace pl = caliper_builder::spec::piece_limits;
    // The strings of the table: the manifest's, or those of the version that this frame showed.
    std::vector<std::string_view> table;
    std::vector<bool> usable;
    if (t.strings_from) {
        const Impl::Slot* from = nullptr;
        for (const Impl::Slot& slot : m.slots)
            if (slot.id == *t.strings_from && !slot.texture) from = &slot;
        if (!from || !from->keep_cpu || !from->shown) return out;
        const std::uint64_t width = from->layout.stride;
        if (width == 0 || width > pl::kStringBytes) return out;
        const Impl::Shown& sv = *from->shown;
        const std::uint64_t count = std::min<std::uint64_t>({sv.count, sv.bytes.size() / width, pl::kStrings});
        for (std::uint64_t i = 0; i < count; ++i) {
            const char* at = reinterpret_cast<const char*>(sv.bytes.data() + i * width);
            const std::string_view text(at, std::size_t(std::find(at, at + width, '\0') - at));
            table.push_back(text);
            usable.push_back(valid_utf8(text));
        }
    } else {
        for (std::size_t i = 0; i < std::min(t.strings.size(), pl::kStrings); ++i) {
            table.push_back(t.strings[i]);
            usable.push_back(true);
        }
    }
    const std::size_t strings = table.size();
    for (std::uint64_t i = 0; i < n; ++i) {
        const std::uint8_t* e = v.bytes.data() + i * stride;
        std::uint32_t index;
        std::memcpy(&index, e + str->offset, 4);
        if (index >= strings || !usable[index] || table[index].empty()) continue;   // no text, no label, as the built-in
        Label l;
        l.align = t.align;
        bool finite = true;
        for (int k = 0; k < 3; ++k) {
            l.position[std::size_t(k)] = f32_at(e + pos->offset + 4 * k);
            finite = finite && std::isfinite(l.position[std::size_t(k)]);
        }
        if (!finite) continue;
        const std::string_view text = table[index];
        // The manifest's check holds a string to kStringBytes; a longer one is cut on a whole character.
        std::size_t cut = std::min(text.size(), caliper_builder::spec::piece_limits::kStringBytes);
        while (cut > 0 && cut < text.size() && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) --cut;
        l.text = std::string(text.substr(0, cut));
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
