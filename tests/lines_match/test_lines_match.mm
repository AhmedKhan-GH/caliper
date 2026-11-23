// test_lines_match.mm: the match of the library piece lines@1 with the built-in lines primitive
// (design/programmable-graphics.md, Library Pieces).
//
// Each case draws one set of lines twice on Metal, offscreen: by the scene core as a lines batch,
// and by the piece end to end, its derive half on the producer runtime over the same arrays as
// tensors and its draw half in a draw runtime with the camera matrix of the built-in draw. The
// cases cover each mode (segments, strips with lengths of 0 and 1 and a last strip that runs past
// the vertices, edges with clamped indices), each color mode (one color, values through each
// colormap, packed RGBA), values of the vertices, the strips or the edges and one value for all,
// fixed and automatic ranges (with a NaN, with equal values), translucent colors, NaN coordinates,
// two-column positions, widths at a pixel ratio of 2, the depth offset, no lines, one segment, and
// the most segments that the piece publishes (209,714, the 8 MiB limit, below the built-in's
// 2,097,152 segments of 4,194,304 vertices). line_match.h holds the tolerances and the picks.
#include "line_match.h"

#include "scene/scene.h"

namespace {

using namespace line_match;

// A deterministic stream of numbers in [-1, 1).
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

// The data of one case of lines, in the built-in's terms and the piece's.
struct Lines {
    int dims = 3;                          // 2: the piece gets (V, 2), the built-in z = 0
    std::vector<float> positions;          // V rows of dims
    std::vector<std::uint32_t> sources;    // values as bits, or packed RGBA; empty: one color
    bool colors = false;                   // sources are packed RGBA
    sc::LineMode mode = sc::LineMode::Segments;
    std::vector<std::int32_t> topology;    // strip lengths, or index pairs
    bool per_item = false;                 // sources belong to the strips or the edges
    float width = 2.0f;
    std::uint32_t flat = sc::kDefaultRgba;
    std::uint32_t colormap = 0;
    bool auto_range = false;
    float vmin = 0.0f, vmax = 1.0f;
    int depth_offset = 0;

    std::uint32_t vertices() const { return std::uint32_t(positions.size() / std::size_t(dims)); }
};

// The range that the built-in draws with: the fixed ends, or the producer's stats for automatic
// ones (min and max with NaN propagating, as torch.aminmax), resolved as the live scene resolves
// them: not finite gives 0 to 1, equal ends widen by 0.5.
void range_of(const Lines& l, float* vmin, float* vmax) {
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

// The built-in batch of the case, as the live scene makes it from SCENE_UPDATE: the break mask of
// the strip lengths, per-strip sources expanded to the vertices, and translucent when a packed
// color has an alpha below 255.
void add_builtin(Builtin& b, const std::string& path, const Lines& l) {
    const std::uint32_t n = l.vertices();
    std::vector<float> rows(std::size_t(n) * 3, 0.0f);
    for (std::uint32_t i = 0; i < n; ++i)
        for (int c = 0; c < l.dims; ++c) rows[std::size_t(i) * 3 + std::size_t(c)] = l.positions[std::size_t(i) * std::size_t(l.dims) + std::size_t(c)];
    sc::Batch batch;
    batch.primitive = sc::Primitive::Lines;
    batch.line_mode = l.mode;
    batch.width_px = l.width;
    batch.flat_rgba = l.flat;
    batch.colormap = l.colormap;
    range_of(l, &batch.vmin, &batch.vmax);
    batch.depth_offset = std::int8_t(l.depth_offset);
    for (std::uint32_t i = 0; i < n; ++i) {
        const float* p = rows.data() + std::size_t(i) * 3;
        if (std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2])) batch.bounds.add({p[0], p[1], p[2]});
    }
    std::vector<Array> arrays;
    arrays.push_back(array(sc::Role::Positions, rows, n, 12, gfx::VertexFormat::F32x3));
    std::vector<std::uint32_t> sources = l.sources;
    if (l.mode == sc::LineMode::Strips) {
        std::vector<std::uint32_t> mask(n, 0), strip_of(n, 0);
        std::uint32_t v = 0;
        for (std::size_t s = 0; s < l.topology.size() && v < n; ++s) {
            const std::uint32_t len = std::uint32_t(std::max(l.topology[s], 0));
            for (std::uint32_t k = 0; k < len && v < n; ++k, ++v) {
                mask[v] = k + 1 < len ? 1u : 0u;
                strip_of[v] = std::uint32_t(s);
            }
        }
        arrays.push_back(array(sc::Role::BreakMask, mask, n, 4, gfx::VertexFormat::U8));
        if (l.per_item && sources.size() > 1) {
            std::vector<std::uint32_t> expanded(n);
            for (std::uint32_t i = 0; i < n; ++i) expanded[i] = l.sources[strip_of[i]];
            sources = expanded;
        }
    } else if (l.mode == sc::LineMode::Edges) {
        batch.per_edge = l.per_item;
        arrays.push_back(array(sc::Role::Indices, l.topology, l.topology.size() / 2, 8, gfx::VertexFormat::S32));
    }
    if (!sources.empty()) {
        if (l.colors) {
            for (std::uint32_t c : l.sources) batch.translucent = batch.translucent || (c >> 24) < 255;
            arrays.push_back(array(sc::Role::Colors, sources, sources.size(), 4, gfx::VertexFormat::Unorm8x4));
        } else {
            arrays.push_back(array(sc::Role::Values, sources, sources.size(), 4, gfx::VertexFormat::F32));
        }
    }
    b.add(path, batch, arrays);
}

// The tensors and params of the piece for the same case.
std::vector<Tensor> tensors_of(const Lines& l) {
    std::vector<Tensor> t;
    t.push_back(tensor("positions", l.positions, {std::int64_t(l.vertices()), std::int64_t(l.dims)}));
    std::vector<float> values;
    for (std::uint32_t s : l.sources) values.push_back(float_of(s));
    if (values.empty()) values.push_back(0.0f);   // the input is always bound; one color reads none
    t.push_back(tensor("values", values, {std::int64_t(values.size())}));
    std::vector<std::int32_t> topo = l.topology;
    if (topo.empty()) topo.push_back(0);
    if (l.mode == sc::LineMode::Edges) t.push_back(tensor("topology", topo, {std::int64_t(topo.size() / 2), 2}));
    else t.push_back(tensor("topology", topo, {std::int64_t(topo.size())}));
    return t;
}

Json params_of(const Lines& l) {
    Json v = Json::object();
    v.set("width", double(l.width));
    v.set("mode", std::uint64_t(l.mode));
    v.set("color_mode", std::uint64_t(l.sources.empty() ? 0 : l.colors ? 2 : 1));
    v.set("color", std::uint64_t(l.flat));
    v.set("color_per", std::uint64_t(l.per_item ? 1 : 0));
    v.set("colormap", std::uint64_t(l.colormap));
    v.set("range", std::uint64_t(l.auto_range ? 3 : 0));
    v.set("vmin", double(l.vmin));
    v.set("vmax", double(l.vmax));
    v.set("depth_offset", std::int64_t(l.depth_offset));
    return v;
}

// The ends of element k of the built-in draw: the item map's fields index_a and index_b.
Fields ends_of(const Lines& l, std::uint32_t k) {
    const std::uint32_t n = l.vertices();
    double a = 2.0 * k, b = 2.0 * k + 1;
    if (l.mode == sc::LineMode::Strips) {
        a = k;
        b = k + 1;
    } else if (l.mode == sc::LineMode::Edges) {
        a = std::min<std::uint32_t>(std::uint32_t(l.topology[2 * k]), n - 1);
        b = std::min<std::uint32_t>(std::uint32_t(l.topology[2 * k + 1]), n - 1);
    }
    return {{"index_a", a}, {"index_b", b}};
}

// Draws one case both ways and checks it.
Comparison run_case(Metal& m, const std::string& name, const Lines& l, const std::vector<std::uint32_t>& items,
                    std::size_t min_lit, float pixel_ratio = 1.0f, const std::string& capture_name = "") {
    CAPTURE(name);
    Builtin b(*m.backend, pixel_ratio);
    if (l.vertices() > 0) add_builtin(b, "lines", l);
    else b.tree.ensure("lines");
    const std::array<float, 16> mvp = b.fit();
    const std::vector<std::uint8_t> builtin = b.draw();
    Program p;
    make_program(m, "lines", tensors_of(l), params_of(l), kPick, &p);
    if (const Published* info = find_published(p.published, "info"); info && info->bytes.size() >= 12) {
        float lo = 0, hi = 0;
        std::uint32_t flags = 0;
        std::memcpy(&lo, info->bytes.data(), 4);
        std::memcpy(&hi, info->bytes.data() + 4, 4);
        std::memcpy(&flags, info->bytes.data() + 8, 4);
        MESSAGE(name, ": info lo ", lo, " hi ", hi, " flags ", flags, ", ", find_published(p.published, "segments")->count,
                " segments");
    }
    const Drawn drawn = draw_programs(*m.backend, {&p}, mvp, pixel_ratio);
    return check_case(name, b, builtin, drawn, p, items, [&](std::uint32_t k) { return ends_of(l, k); }, min_lit,
                      capture_name);
}

// A curve of n vertices through the box: two turns of a spiral that rises.
void helix(Lines* l, std::uint32_t n, float radius, float phase, float z0) {
    for (std::uint32_t i = 0; i < n; ++i) {
        const float t = float(i) / float(std::max(n, 2u) - 1);
        const float r = radius * (0.3f + 0.7f * t);
        l->positions.push_back(r * std::cos(phase + 12.5f * t));
        l->positions.push_back(r * std::sin(phase + 12.5f * t));
        l->positions.push_back(z0 + 0.4f * t);
    }
}

// A value at an exact colormap level, so that no value sits on a boundary between two levels.
std::uint32_t level_value(std::uint32_t i) {
    return bits_of(float((i * 37u) % 256u) / 255.0f);
}

}  // namespace

TEST_CASE("lines_match: lines@1 passes piece.check, and the pixel comparison carries it as the library has it") {
    cs::Piece piece;
    std::string why;
    REQUIRE_MESSAGE(load_piece("lines", &piece, &why), why);
    check_piece(piece);
    check_pinned(piece, dp::kLinesManifest, dp::kLinesHash);
}

TEST_CASE("lines_match: strips, segments and edges in each color mode draw as the built-in lines") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);

    {   // Strips of 30, 1, 0 and 50 vertices and a last one of 40 that runs past the 101 vertices,
        // per-vertex values through viridis, width 4.
        Lines l;
        l.mode = sc::LineMode::Strips;
        helix(&l, 101, 0.9f, 0.0f, -0.4f);
        l.topology = {30, 1, 0, 50, 40};
        for (std::uint32_t i = 0; i < 101; ++i) l.sources.push_back(level_value(i));
        l.width = 4.0f;
        run_case(m, "strips, values through viridis", l, {0, 10, 50, 99}, 400, 1.0f, "lines-match");
        // The same strips with one value of each strip through magma, a range from the values.
        Lines s = l;
        // The values span -2 to 6, each at about a whole level of that range.
        auto at = [](float k) { return bits_of(-2.0f + 8.0f * k / 255.0f); };
        s.sources = {bits_of(-2.0f), at(9), bits_of(6.0f), at(81), at(200)};
        s.per_item = true;
        s.colormap = 1;
        s.auto_range = true;
        run_case(m, "strips, one value a strip through magma, automatic range", s, {0, 50, 99}, 400);
        // One packed RGBA a strip, one of them translucent: the batch draws translucent.
        Lines t = l;
        t.colors = true;
        t.per_item = true;
        t.sources = {rgba(250, 80, 40, 255), rgba(40, 200, 90, 255), rgba(0, 0, 0, 255), rgba(60, 120, 240, 120),
                     rgba(240, 220, 40, 255)};
        run_case(m, "strips, one translucent RGBA a strip", t, {0, 50}, 400);
    }
    {   // 40 segments with a packed RGBA at each vertex, width 5.
        Lines l;
        Lcg r(7);
        for (int k = 0; k < 40; ++k) {
            const float x = r.next() * 0.8f, y = r.next() * 0.8f, z = r.next() * 0.8f;
            l.positions.insert(l.positions.end(), {x, y, z, x + 0.25f * r.next(), y + 0.25f * r.next(), z + 0.1f});
            for (int e = 0; e < 2; ++e)
                l.sources.push_back(rgba(std::uint32_t(128 + 120 * r.next()), std::uint32_t(128 + 120 * r.next()),
                                         std::uint32_t(128 + 120 * r.next()), 255));
        }
        l.colors = true;
        l.width = 5.0f;
        run_case(m, "segments, RGBA at each vertex", l, {0, 20, 39}, 300);
        // The same segments in one translucent color.
        Lines t = l;
        t.sources.clear();
        t.colors = false;
        t.flat = rgba(255, 140, 60, 128);
        t.width = 6.0f;
        run_case(m, "segments, one translucent color", t, {0, 39}, 300);
        // And in the default color, two-column positions on the plane z = 0.
        Lines f;
        f.dims = 2;
        for (std::size_t i = 0; i < l.positions.size(); i += 3) f.positions.insert(f.positions.end(), {l.positions[i], l.positions[i + 1]});
        f.width = 4.0f;
        run_case(m, "segments, default color, (V, 2) positions", f, {0, 39}, 300);
    }
    {   // Edges between 30 vertices: 45 index pairs, one index past the vertices and one below 0,
        // both clamped to the last vertex, with one value an edge through rdbu.
        Lines l;
        l.mode = sc::LineMode::Edges;
        Lcg r(11);
        for (int i = 0; i < 30; ++i) {
            const float a = 6.2831853f * float(i) / 30.0f;
            l.positions.insert(l.positions.end(), {0.9f * std::cos(a), 0.9f * std::sin(a), 0.3f * r.next()});
        }
        for (int e = 0; e < 45; ++e) {
            l.topology.push_back((e * 7) % 30);
            l.topology.push_back((e * 13 + 5) % 30);
        }
        l.topology[2 * 10 + 1] = 99;
        l.topology[2 * 20] = -3;
        // The first and the last edge on the rim, where no chord covers them.
        l.topology[0] = 0;
        l.topology[1] = 1;
        l.topology[88] = 28;
        l.topology[89] = 29;
        for (std::uint32_t e = 0; e < 45; ++e) l.sources.push_back(level_value(e + 5));
        l.per_item = true;
        l.colormap = 2;
        l.width = 4.0f;
        run_case(m, "edges, one value an edge through rdbu, clamped indices", l, {0, 10, 20, 44}, 300);
        // Values of the vertices through greys, a range from the values.
        Lines v = l;
        v.per_item = false;
        v.sources.clear();
        for (std::uint32_t i = 0; i < 30; ++i) v.sources.push_back(level_value(i));
        v.sources[1] = bits_of(1.0f);   // with level_value(0) = 0, the range from the values is 0 to 1
        v.colormap = 3;
        v.auto_range = true;
        run_case(m, "edges, values of the vertices through greys, automatic range", v, {0, 44}, 300);
    }
}

TEST_CASE("lines_match: one value, NaN, a pixel ratio of 2, the depth offset and the edge counts") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);

    {   // One value for all, an automatic range of equal values widened by 0.5.
        Lines l;
        l.mode = sc::LineMode::Strips;
        helix(&l, 60, 0.8f, 1.0f, -0.3f);
        l.topology = {60};
        l.sources = {bits_of(0.25f)};
        l.auto_range = true;
        l.width = 4.0f;
        run_case(m, "strip, one value for all, equal range", l, {0, 58}, 300);
        // A NaN coordinate hides the two joins of its vertex; a NaN value takes level 0, and with an
        // automatic range a NaN makes it 0 to 1, as the producer's stats do.
        Lines n = l;
        n.positions[3 * 20 + 1] = NAN;
        n.sources.clear();
        for (std::uint32_t i = 0; i < 60; ++i) n.sources.push_back(level_value(i));
        n.sources[40] = 0x7fc00000u;
        run_case(m, "strip, a NaN coordinate and a NaN value, automatic range", n, {0, 58}, 300);
        // Width 1.5 at a pixel ratio of 2.
        Lines w = l;
        w.width = 1.5f;
        w.sources = {level_value(100)};
        w.auto_range = false;
        run_case(m, "strip, width 1.5 at a pixel ratio of 2", w, {}, 1, 2.0f);
        // The default width of 2 pixels, which has no interior pixel: the edge tolerance holds alone.
        Lines d = l;
        d.width = 2.0f;
        run_case(m, "strip, the default width", d, {}, 0);
    }
    {   // The depth offset: strips of 2 toward the camera, drawn first, then coplanar segments
        // without one that cross them. The strips stay in front where they cross.
        Lines a;
        a.mode = sc::LineMode::Strips;
        for (int i = 0; i < 40; ++i) a.positions.insert(a.positions.end(), {-0.9f + 0.045f * float(i), 0.3f * std::sin(0.4f * float(i)), 0.0f});
        a.topology = {40};
        a.flat = rgba(240, 60, 60, 255);
        a.width = 8.0f;
        a.depth_offset = 2;
        Lines c;
        for (int k = 0; k < 12; ++k) {
            const float x = -0.8f + 0.14f * float(k);
            c.positions.insert(c.positions.end(), {x, -0.8f, 0.0f, x + 0.1f, 0.8f, 0.0f});
        }
        c.flat = rgba(60, 200, 240, 255);
        c.width = 8.0f;
        Builtin b(*m.backend);
        add_builtin(b, "a", a);
        add_builtin(b, "b", c);
        const std::array<float, 16> mvp = b.fit();
        const std::vector<std::uint8_t> builtin = b.draw();
        Program pa, pb;
        make_program(m, "lines", tensors_of(a), params_of(a), kPick, &pa);
        make_program(m, "lines", tensors_of(c), params_of(c), kPick + 1, &pb);
        const Drawn drawn = draw_programs(*m.backend, {&pa, &pb}, mvp);
        check_case("depth offset of 2 over coplanar segments", b, builtin, drawn, pa, {0, 38},
                   [&](std::uint32_t k) { return ends_of(a, k); }, 500);
    }
    {   // No segment: one vertex of segments, and strips whose lengths are all 0. A tensor without
        // elements cannot be bound at all, because the script refuses to import it.
        Lines one;
        one.positions = {0.1f, 0.2f, 0.3f};
        Lines zero;
        zero.mode = sc::LineMode::Strips;
        helix(&zero, 12, 0.8f, 0.0f, 0.0f);
        zero.topology = {0, 0, 0};
        for (const Lines* l : {&one, &zero}) {
            Builtin b(*m.backend);
            add_builtin(b, "lines", *l);
            const std::array<float, 16> mvp = b.fit();
            const std::vector<std::uint8_t> builtin = b.draw();
            Program p;
            make_program(m, "lines", tensors_of(*l), params_of(*l), kPick, &p);
            const Published* seg = find_published(p.published, "segments");
            REQUIRE(seg != nullptr);
            const Drawn drawn = draw_programs(*m.backend, {&p}, mvp);
            const Comparison c = compare(builtin, drawn.color, drawn.ids);
            CHECK(c.differ == 0);
            std::size_t lit = 0;
            for (std::size_t i = 0; i < drawn.color.size(); i += 4) lit += pixel_of(drawn.color, int(i / 4 % kSize), int(i / 4 / kSize)) != kClear;
            CHECK(lit == 0);
            MESSAGE(std::string(l == &one ? "one vertex" : "strips of length 0"), ": ", seg->count, " segments published, ", lit,
                    " pixels lit, ", c.differ, " pixels differ");
        }
    }
    {   // One segment, and an odd vertex count whose last vertex joins nothing.
        Lines l;
        l.positions = {-0.7f, -0.2f, 0.1f, 0.6f, 0.4f, -0.1f, 0.9f, 0.9f, 0.9f};
        l.width = 10.0f;
        l.flat = rgba(200, 200, 90, 255);
        run_case(m, "one segment of three vertices", l, {0}, 300);
    }
}

TEST_CASE("lines_match: the most segments that the piece publishes, and the count above them") {
    Metal m;
    if (!m.backend) CALIPER_SKIP_NO_GPU(m.why);
    constexpr std::uint32_t kMost = 2097152;
    // Short segments behind, and the first and the last segment long in front.
    Lines l;
    Lcg r(3);
    l.positions.reserve(std::size_t(kMost) * 6);
    for (std::uint32_t k = 0; k < kMost; ++k) {
        const float x = r.next(), y = r.next(), z = -0.5f + 0.3f * r.next();
        l.positions.insert(l.positions.end(), {x, y, z, x + 0.02f, y + 0.01f * r.next(), z});
        l.sources.push_back(level_value(2 * k));
        l.sources.push_back(level_value(2 * k + 1));
    }
    for (std::uint32_t k : {0u, kMost - 1}) {
        const float y = k == 0 ? 0.6f : -0.6f;
        float* p = l.positions.data() + std::size_t(k) * 6;
        const float seg[6] = {-0.5f, y, 0.9f, 0.5f, y, 0.9f};
        std::copy(seg, seg + 6, p);
    }
    l.width = 3.0f;
    {
        Builtin b(*m.backend);
        add_builtin(b, "lines", l);
        const std::array<float, 16> mvp = b.fit();
        const std::vector<std::uint8_t> builtin = b.draw();
        Program p;
        make_program(m, "lines", tensors_of(l), params_of(l), kPick, &p);
        REQUIRE(find_published(p.published, "segments") != nullptr);
        CHECK(find_published(p.published, "segments")->count == kMost);
        const Drawn drawn = draw_programs(*m.backend, {&p}, mvp);
        check_case("2,097,152 segments", b, builtin, drawn, p, {}, [&](std::uint32_t k) { return ends_of(l, k); }, 1);
    }
    {   // Above the published limit the piece draws its first 2,097,152 segments.
        Lines more = l;
        for (int k = 0; k < 10; ++k) {
            more.positions.insert(more.positions.end(), {0, 0, 0, 0.1f, 0, 0});
            more.sources.insert(more.sources.end(), {0u, 0u});
        }
        Program p;
        make_program(m, "lines", tensors_of(more), params_of(more), kPick, &p);
        CHECK(find_published(p.published, "segments")->count == kMost);
        MESSAGE("2,097,162 segments publish ", find_published(p.published, "segments")->count);
    }
}

CALIPER_GPU_TEST_MAIN()
