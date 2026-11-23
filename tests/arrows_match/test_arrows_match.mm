// test_arrows_match.mm: the match of the library piece arrows@1 with the built-in arrows primitive
// (design/programmable-graphics.md, Library Pieces).
//
// Each case draws one set of arrows twice on Metal, offscreen: by the scene core as an arrows
// batch, and by the piece end to end, its derive half on the producer runtime over the same arrays
// as tensors and its draw half in a draw runtime with the camera matrix of the built-in draw. The
// cases cover each color mode (one color, values through each colormap, packed RGBA), one value
// for all, fixed and automatic ranges (with a NaN, with equal values), translucent colors, NaN
// coordinates and values, an arrow of length 0, two-column rows, a width at a pixel ratio of 2,
// the depth offset, one arrow, and the most arrows that the piece publishes (262,143, the 8 MiB
// limit, below the built-in's 1,048,576). ../lines_match/line_match.h holds the tolerances and
// the picks.
#include "../lines_match/line_match.h"

#include "scene/scene.h"

namespace {

using namespace line_match;

struct Lcg {
    std::uint32_t state;
    explicit Lcg(std::uint32_t seed) : state(seed) {}
    float next() {
        state = state * 1664525u + 1013904223u;
        return float(double(state >> 8) / double(1u << 24)) * 2.0f - 1.0f;
    }
};

std::uint32_t bits_of(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return u;
}

float float_of(std::uint32_t u) {
    float f = 0;
    std::memcpy(&f, &u, 4);
    return f;
}

std::uint32_t rgba(std::uint32_t r, std::uint32_t g, std::uint32_t b, std::uint32_t a) {
    return r | g << 8 | b << 16 | a << 24;
}

std::uint32_t level_value(std::uint32_t i) {
    return bits_of(float((i * 37u) % 256u) / 255.0f);
}

// The data of one case of arrows.
struct Arrows {
    int dims = 3;
    std::vector<float> origins, vectors;   // N rows of dims
    std::vector<std::uint32_t> sources;    // values as bits, or packed RGBA; empty: one color
    bool colors = false;
    float width = 2.0f;
    std::uint32_t flat = sc::kDefaultRgba;
    std::uint32_t colormap = 0;
    bool auto_range = false;
    float vmin = 0.0f, vmax = 1.0f;
    int depth_offset = 0;

    std::uint32_t count() const { return std::uint32_t(origins.size() / std::size_t(dims)); }
};

void range_of(const Arrows& l, float* vmin, float* vmax) {
    *vmin = l.vmin;
    *vmax = l.vmax;
    if (!l.auto_range) return;
    double lo = NAN, hi = NAN;
    bool nan = l.sources.empty();
    for (std::uint32_t s : l.sources) {
        const float v = float_of(s);
        if (std::isnan(v)) nan = true;
        if (std::isnan(lo) || v < lo) lo = v;
        if (std::isnan(hi) || v > hi) hi = v;
    }
    if (nan) lo = hi = NAN;
    if (!std::isfinite(lo) || !std::isfinite(hi)) {
        lo = 0.0;
        hi = 1.0;
    } else if (lo == hi) {
        lo -= 0.5;
        hi += 0.5;
    }
    *vmin = float(lo);
    *vmax = float(hi);
}

std::vector<float> rows3(const std::vector<float>& rows, int dims, std::uint32_t n) {
    std::vector<float> out(std::size_t(n) * 3, 0.0f);
    for (std::uint32_t i = 0; i < n; ++i)
        for (int c = 0; c < dims; ++c) out[std::size_t(i) * 3 + std::size_t(c)] = rows[std::size_t(i) * std::size_t(dims) + std::size_t(c)];
    return out;
}

// The built-in batch of the case, as the live scene makes it from SCENE_UPDATE.
void add_builtin(Builtin& b, const std::string& path, const Arrows& l) {
    const std::uint32_t n = l.count();
    const std::vector<float> o = rows3(l.origins, l.dims, n), v = rows3(l.vectors, l.dims, n);
    sc::Batch batch;
    batch.primitive = sc::Primitive::Arrows;
    batch.width_px = l.width;
    batch.flat_rgba = l.flat;
    batch.colormap = l.colormap;
    range_of(l, &batch.vmin, &batch.vmax);
    batch.depth_offset = std::int8_t(l.depth_offset);
    for (std::uint32_t i = 0; i < n; ++i) {
        const float* p = o.data() + std::size_t(i) * 3;
        const float* d = v.data() + std::size_t(i) * 3;
        if (std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2])) batch.bounds.add({p[0], p[1], p[2]});
        const double t[3] = {double(p[0]) + d[0], double(p[1]) + d[1], double(p[2]) + d[2]};
        if (std::isfinite(t[0]) && std::isfinite(t[1]) && std::isfinite(t[2])) batch.bounds.add({t[0], t[1], t[2]});
    }
    std::vector<Array> arrays;
    arrays.push_back(array(sc::Role::Origins, o, n, 12, gfx::VertexFormat::F32x3));
    arrays.push_back(array(sc::Role::Vectors, v, n, 12, gfx::VertexFormat::F32x3));
    if (!l.sources.empty()) {
        if (l.colors) {
            for (std::uint32_t c : l.sources) batch.translucent = batch.translucent || (c >> 24) < 255;
            arrays.push_back(array(sc::Role::Colors, l.sources, l.sources.size(), 4, gfx::VertexFormat::Unorm8x4));
        } else {
            arrays.push_back(array(sc::Role::Values, l.sources, l.sources.size(), 4, gfx::VertexFormat::F32));
        }
    }
    b.add(path, batch, arrays);
}

std::vector<Tensor> tensors_of(const Arrows& l) {
    std::vector<Tensor> t;
    t.push_back(tensor("origins", l.origins, {std::int64_t(l.count()), std::int64_t(l.dims)}));
    t.push_back(tensor("vectors", l.vectors, {std::int64_t(l.vectors.size() / std::size_t(l.dims)), std::int64_t(l.dims)}));
    std::vector<float> values;
    for (std::uint32_t s : l.sources) values.push_back(float_of(s));
    if (values.empty()) values.push_back(0.0f);   // the input is always bound; one color reads none
    t.push_back(tensor("values", values, {std::int64_t(values.size())}));
    return t;
}

Json params_of(const Arrows& l) {
    Json v = Json::object();
    v.set("width", double(l.width));
    v.set("color_mode", std::uint64_t(l.sources.empty() ? 0 : l.colors ? 2 : 1));
    v.set("color", std::uint64_t(l.flat));
    v.set("colormap", std::uint64_t(l.colormap));
    v.set("range", std::uint64_t(l.auto_range ? 3 : 0));
    v.set("vmin", double(l.vmin));
    v.set("vmax", double(l.vmax));
    v.set("depth_offset", std::int64_t(l.depth_offset));
    return v;
}

Fields index_of(std::uint32_t k) { return {{"index", double(k)}}; }

Comparison run_case(Metal& m, const std::string& name, const Arrows& l, const std::vector<std::uint32_t>& items,
                    std::size_t min_lit, float pixel_ratio = 1.0f, const std::string& capture_name = "") {
    CAPTURE(name);
    Builtin b(*m.backend, pixel_ratio);
    add_builtin(b, "arrows", l);
    const std::array<float, 16> mvp = b.fit();
    const std::vector<std::uint8_t> builtin = b.draw();
    Program p;
    make_program(m, "arrows", tensors_of(l), params_of(l), kPick, &p);
    const Drawn drawn = draw_programs(*m.backend, {&p}, mvp, pixel_ratio);
    return check_case(name, b, builtin, drawn, p, items, index_of, min_lit, capture_name);
}

// n arrows from a ring outward and up, each a different length.
Arrows ring(std::uint32_t n, float width) {
    Arrows l;
    for (std::uint32_t i = 0; i < n; ++i) {
        const float a = 6.2831853f * float(i) / float(n);
        const float r = 0.4f + 0.05f * float(i % 3);
        l.origins.insert(l.origins.end(), {r * std::cos(a), r * std::sin(a), 0.1f * float(i % 4)});
        const float len = 0.25f + 0.2f * float(i % 5) / 4.0f;
        l.vectors.insert(l.vectors.end(), {len * std::cos(a), len * std::sin(a), 0.15f});
    }
    l.width = width;
    return l;
}

}  // namespace

TEST_CASE("arrows_match: arrows@1 passes piece.check, and the pixel comparison carries it as the library has it") {
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("arrows", &piece, &why), why);
    check_piece(piece);
    check_pinned(piece, dp::kArrowsManifest, dp::kArrowsHash);
}

TEST_CASE("arrows_match: arrows in each color mode draw as the built-in arrows") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    {   // Values through viridis at fixed ends, width 4.
        Arrows l = ring(24, 4.0f);
        for (std::uint32_t i = 0; i < 24; ++i) l.sources.push_back(level_value(i + 1));
        run_case(m, "values through viridis", l, {0, 12, 23}, 400, 1.0f, "arrows-match");
        // Values from -2 to 6 through magma, a range from the values.
        Arrows a = l;
        a.sources.clear();
        for (std::uint32_t i = 0; i < 24; ++i) a.sources.push_back(bits_of(-2.0f + 8.0f * float((i * 37u) % 256u) / 255.0f));
        a.sources[3] = bits_of(-2.0f);
        a.sources[7] = bits_of(6.0f);
        a.colormap = 1;
        a.auto_range = true;
        run_case(m, "values through magma, automatic range", a, {0, 23}, 400);
        // A packed RGBA for each arrow, then one of them translucent.
        Arrows c = l;
        c.sources.clear();
        Lcg r(5);
        for (int i = 0; i < 24; ++i)
            c.sources.push_back(rgba(std::uint32_t(128 + 120 * r.next()), std::uint32_t(128 + 120 * r.next()),
                                     std::uint32_t(128 + 120 * r.next()), 255));
        c.colors = true;
        run_case(m, "RGBA of each arrow", c, {0, 12, 23}, 400);
        c.sources[5] = rgba(255, 255, 255, 90);
        run_case(m, "RGBA of each arrow, one translucent", c, {0, 23}, 400);
        // One translucent color for all, wide arrows that overlap.
        Arrows t = ring(24, 9.0f);
        t.flat = rgba(90, 200, 255, 150);
        run_case(m, "one translucent color", t, {0, 23}, 400);
        // One value for all through rdbu, an automatic range of equal values.
        Arrows o = l;
        o.sources = {bits_of(0.3f)};
        o.colormap = 2;
        o.auto_range = true;
        run_case(m, "one value for all through rdbu, equal range", o, {0, 23}, 400);
    }
    {   // Two-column rows on the plane z = 0, greys, and a pixel ratio of 2.
        Arrows l = ring(20, 3.0f);
        Arrows f;
        f.dims = 2;
        for (std::size_t i = 0; i < l.origins.size(); i += 3) {
            f.origins.insert(f.origins.end(), {l.origins[i], l.origins[i + 1]});
            f.vectors.insert(f.vectors.end(), {l.vectors[i], l.vectors[i + 1]});
        }
        for (std::uint32_t i = 0; i < 20; ++i) f.sources.push_back(level_value(3 * i));
        f.colormap = 3;
        f.width = 3.0f;
        run_case(m, "(N, 2) rows through greys", f, {0, 19}, 300);
        f.width = 1.5f;
        run_case(m, "(N, 2) rows, width 1.5 at a pixel ratio of 2", f, {0, 19}, 100, 2.0f);
    }
}

TEST_CASE("arrows_match: NaN, length 0, the depth offset, one arrow and the edge counts") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    {   // A NaN origin, a NaN vector component, a NaN value, and an arrow of length 0, which draws
        // a square of the width at its origin. A NaN with an automatic range makes it 0 to 1.
        Arrows l = ring(16, 5.0f);
        for (std::uint32_t i = 0; i < 16; ++i) l.sources.push_back(level_value(i + 2));
        l.origins[3 * 3 + 0] = NAN;
        l.vectors[3 * 6 + 2] = NAN;
        l.sources[9] = 0x7fc00000u;
        l.vectors[3 * 12 + 0] = l.vectors[3 * 12 + 1] = l.vectors[3 * 12 + 2] = 0.0f;
        l.auto_range = true;
        run_case(m, "NaN origin, NaN vector, NaN value, length 0", l, {0, 9, 12, 15}, 300);
        // One arrow; and one arrow with a NaN origin, which draws nothing.
        Arrows one;
        one.origins = {-0.6f, -0.3f, 0.0f};
        one.vectors = {1.1f, 0.7f, 0.2f};
        one.width = 6.0f;
        run_case(m, "one arrow", one, {0}, 300);
        Arrows none = one;
        none.origins[0] = NAN;
        none.origins.insert(none.origins.end(), {0.0f, 0.0f, 0.0f});   // a second arrow, for the camera's bounds
        none.vectors.insert(none.vectors.end(), {0.0f, 0.0f, 0.0f});
        none.flat = rgba(0x14, 0x18, 0x20, 0);
        Builtin b(*m.backend);
        add_builtin(b, "arrows", none);
        const std::array<float, 16> mvp = b.fit();
        const std::vector<std::uint8_t> builtin = b.draw();
        Program p;
        make_program(m, "arrows", tensors_of(none), params_of(none), kPick, &p);
        const Drawn drawn = draw_programs(*m.backend, {&p}, mvp);
        const Comparison c = compare(builtin, drawn.color, drawn.ids);
        CHECK(c.differ == 0);
        MESSAGE("a NaN arrow and a transparent one: ", c.differ, " pixels differ");
    }
    {   // The depth offset: arrows of 2 toward the camera drawn first, coplanar arrows across them.
        Arrows a = ring(12, 8.0f);
        for (std::size_t i = 2; i < a.origins.size(); i += 3) a.origins[i] = 0.0f, a.vectors[i] = 0.0f;
        a.flat = rgba(240, 60, 60, 255);
        a.depth_offset = 2;
        Arrows c;
        for (int k = 0; k < 8; ++k) {
            c.origins.insert(c.origins.end(), {-0.9f + 0.25f * float(k), -0.9f, 0.0f});
            c.vectors.insert(c.vectors.end(), {0.0f, 1.8f, 0.0f});
        }
        c.flat = rgba(60, 200, 240, 255);
        c.width = 8.0f;
        Builtin b(*m.backend);
        add_builtin(b, "a", a);
        add_builtin(b, "b", c);
        const std::array<float, 16> mvp = b.fit();
        const std::vector<std::uint8_t> builtin = b.draw();
        Program pa, pb;
        make_program(m, "arrows", tensors_of(a), params_of(a), kPick, &pa);
        make_program(m, "arrows", tensors_of(c), params_of(c), kPick + 1, &pb);
        const Drawn drawn = draw_programs(*m.backend, {&pa, &pb}, mvp);
        check_case("depth offset of 2 over coplanar arrows", b, builtin, drawn, pa, {0, 11}, index_of, 500);
    }
}

TEST_CASE("arrows_match: the most arrows that the piece publishes, and the count above them") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    constexpr std::uint32_t kMost = 262143;
    Arrows l;
    Lcg r(9);
    l.origins.reserve(std::size_t(kMost) * 3);
    l.vectors.reserve(std::size_t(kMost) * 3);
    for (std::uint32_t k = 0; k < kMost; ++k) {
        l.origins.insert(l.origins.end(), {r.next(), r.next(), -0.5f + 0.3f * r.next()});
        l.vectors.insert(l.vectors.end(), {0.03f * r.next(), 0.03f * r.next(), 0.0f});
        l.sources.push_back(level_value(k));
    }
    // The first and the last arrow long and in front.
    for (std::uint32_t k : {0u, kMost - 1}) {
        const float y = k == 0 ? 0.6f : -0.6f;
        std::copy_n(std::array<float, 3>{-0.6f, y, 0.9f}.data(), 3, l.origins.data() + std::size_t(k) * 3);
        std::copy_n(std::array<float, 3>{1.2f, 0.0f, 0.0f}.data(), 3, l.vectors.data() + std::size_t(k) * 3);
    }
    l.width = 3.0f;
    {
        Builtin b(*m.backend);
        add_builtin(b, "arrows", l);
        const std::array<float, 16> mvp = b.fit();
        const std::vector<std::uint8_t> builtin = b.draw();
        Program p;
        make_program(m, "arrows", tensors_of(l), params_of(l), kPick, &p);
        CHECK(find_published(p.published, "arrows")->count == kMost);
        const Drawn drawn = draw_programs(*m.backend, {&p}, mvp);
        check_case("262,143 arrows", b, builtin, drawn, p, {0, kMost - 1}, index_of, 1);
    }
    {   // Above the published limit the piece draws its first 262,143 arrows.
        Arrows more = l;
        for (int k = 0; k < 10; ++k) {
            more.origins.insert(more.origins.end(), {0.0f, 0.0f, 0.0f});
            more.vectors.insert(more.vectors.end(), {0.1f, 0.0f, 0.0f});
            more.sources.push_back(0u);
        }
        Program p;
        make_program(m, "arrows", tensors_of(more), params_of(more), kPick, &p);
        CHECK(find_published(p.published, "arrows")->count == kMost);
        MESSAGE("262,153 arrows publish ", find_published(p.published, "arrows")->count);
    }
}

CALIPER_GPU_TEST_MAIN()
